from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional, TypeVar

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.utils.signing import (
    get_timestamp_ms,
    order_request_to_order_wire,
    order_wires_to_order_action,
    sign_l1_action,
)
from hyperliquid.utils.types import Cloid

from app.config import Settings
from app.enums import Side
from app.exchange.exchange_retry import exchange_call_with_retry
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.exchange.hyperliquid_responses import (
    interpret_hl_cancel_response,
    interpret_hl_order_status_response,
    interpret_hl_place_order_response,
    make_deterministic_cloid_hex,
)
from app.exchange.hyperliquid_precision import (
    HL_PERP_LIMIT_PRICE_PIPELINE_ID,
    HL_PERP_MAX_SIG_FIGS,
    validate_hyperliquid_perp_limit_for_submit,
    wire_format_preview_limit_px,
)
from app.exchange.hyperliquid_action_ws import HyperliquidExchangeActionWs
from app.exchange.symbol_spec import (
    FALLBACK_SYMBOL_SPEC,
    SymbolSpec,
    symbol_spec_from_hyperliquid_meta,
)
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _perps_equity_and_withdrawable_from_user_state(
    st: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Parse ``clearinghouseState`` (perps): max of margin/cross accountValue + withdrawable."""
    wd_raw = st.get("withdrawable")
    try:
        withdrawable = float(wd_raw) if wd_raw is not None else None
    except (TypeError, ValueError):
        withdrawable = None

    values: list[float] = []
    for key in ("marginSummary", "crossMarginSummary"):
        box = st.get(key)
        if not isinstance(box, dict):
            continue
        raw = box.get("accountValue")
        try:
            if raw is not None:
                values.append(float(raw))
        except (TypeError, ValueError):
            continue
    equity = max(values) if values else None
    return equity, withdrawable


def _usdc_from_spot_clearinghouse_payload(raw: Any) -> tuple[Optional[float], Optional[float]]:
    """Parse ``spotClearinghouseState`` JSON: USDC total and (total - hold)."""
    if not isinstance(raw, dict):
        return None, None
    for b in raw.get("balances") or []:
        if not isinstance(b, dict):
            continue
        if b.get("coin") != "USDC":
            continue
        try:
            total = float(b["total"]) if b.get("total") is not None else None
            hold_raw = b.get("hold")
            hold = float(hold_raw) if hold_raw is not None else 0.0
        except (TypeError, ValueError, KeyError):
            return None, None
        if total is None:
            return None, None
        free = max(0.0, total - hold)
        return total, free
    return None, None


def _merge_account_snapshot_balances(
    perps_eq: Optional[float],
    perps_wd: Optional[float],
    spot_total: Optional[float],
    spot_free: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Combine perps ``clearinghouseState`` with ``spotClearinghouseState`` USDC.

    Hyperliquid **unified** (and portfolio margin) accounts document spot clearinghouse as the
    source of truth for collateral; perps ``marginSummary`` can under-report. We take the
    **max** of each side so standard (perps-only) accounts stay correct while unified accounts
    pick up full USDC. Withdrawable uses the same rule (same pool; avoids a zero from one API).
    """
    eq_candidates = [x for x in (perps_eq, spot_total) if x is not None]
    equity = max(eq_candidates) if eq_candidates else None

    wd_candidates = [x for x in (perps_wd, spot_free) if x is not None]
    withdrawable = max(wd_candidates) if wd_candidates else None
    return equity, withdrawable


def _hl_side_to_side(s: str) -> Side:
    # TODO(Hyperliquid): verify `side` / dir strings on userFills & openOrders if API format changes.
    s = (s or "").upper()
    if s in ("B", "BUY", "LONG"):
        return Side.BUY
    return Side.SELL


class HyperliquidClient:
    """
    Narrow wrapper around hyperliquid-python-sdk Info + Exchange.

    **Unified margin accounts:** account equity and withdrawable for monitoring / risk use both
    perps ``clearinghouseState`` and ``spotClearinghouseState`` (USDC), merged so collateral is
    not understated when perps summaries read low. Positions and fills still come from the perps
    user APIs for the configured ``SYMBOL`` (main dex by default).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._info = Info(settings.hl_base_url, skip_ws=True)
        self._rest_call_counts: dict[str, int] = defaultdict(int)
        self._rest_retry_counts: dict[str, int] = defaultdict(int)
        self._rest_rate_limited_retry_counts: dict[str, int] = defaultdict(int)
        self._symbol_spec: SymbolSpec
        self.symbol_spec_fetched_ok: bool
        self._symbol_spec, self.symbol_spec_fetched_ok = self._load_symbol_spec()
        self._exchange: Exchange | None = None
        if settings.trading_enabled and settings.hl_secret_key:
            wallet = eth_account.Account.from_key(settings.hl_secret_key)
            self._exchange = Exchange(
                wallet,
                settings.hl_base_url,
                account_address=settings.hl_account_address,
            )
        else:
            self._exchange = None  # type: ignore[assignment]
        # Lazy: signed-action WebSocket transport (POST /exchange equivalent).
        self._action_ws: Any = None
        self.last_exchange_transport_mode: str = "http"
        self.last_exchange_signing_ms: float = 0.0
        self.last_exchange_transport_write_ms: float = 0.0

    def _note_retry(self, operation: str, rate_limited: bool) -> None:
        self._rest_retry_counts[operation] += 1
        if rate_limited:
            self._rest_rate_limited_retry_counts[operation] += 1

    def _retry(self, operation: str, fn: Callable[[], T]) -> T:
        self._rest_call_counts[operation] += 1
        return exchange_call_with_retry(
            operation,
            fn,
            self._settings,
            on_retry=self._note_retry,
        )

    def rest_runtime_counters(self) -> dict[str, int]:
        """Readonly snapshot for heartbeat/diagnostics of REST pressure by operation."""
        out: dict[str, int] = {}
        for op, n in self._rest_call_counts.items():
            out[f"{op}_calls"] = int(n)
        for op, n in self._rest_retry_counts.items():
            out[f"{op}_retries"] = int(n)
        for op, n in self._rest_rate_limited_retry_counts.items():
            out[f"{op}_rate_limited_retries"] = int(n)
        return out

    # --- PerpExchangeAdapter: wire-format interpretation ------------------
    #
    # Thin delegators that keep Hyperliquid wire-format knowledge inside
    # this adapter. The bot core calls these via the generic
    # ``PerpExchangeAdapter`` Protocol — see ``app.exchange.base``.

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_hl_place_order_response(resp)

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        return interpret_hl_cancel_response(resp)

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_hl_order_status_response(resp)

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        return make_deterministic_cloid_hex(symbol, side, quote_cycle_id, price, size)

    def _load_symbol_spec(self) -> tuple[SymbolSpec, bool]:
        sym = (self._settings.symbol or "").strip()
        try:
            meta = self._retry("meta", lambda: self._info.meta())
            spec = symbol_spec_from_hyperliquid_meta(meta, sym)
            logger.info(
                "symbol_spec loaded symbol=%s meta_decimal_grid_price_tick=%s meta_decimal_size_step=%s "
                "min_size=%s min_notional_usd=%s sz_decimals=%s hl_max_sig_figs_nonint=%s "
                "hl_price_normalize_pipeline=%s source=%s",
                sym,
                spec.price_tick,
                spec.size_step,
                spec.min_size,
                spec.min_notional_usd,
                spec.sz_decimals,
                HL_PERP_MAX_SIG_FIGS,
                HL_PERP_LIMIT_PRICE_PIPELINE_ID,
                spec.source,
            )
            return spec, True
        except Exception:
            logger.exception(
                "symbol_spec Hyperliquid meta fetch/parse failed symbol=%s — using "
                "fallback constants (unsafe for live trading if mis-sized)",
                sym,
            )
            return FALLBACK_SYMBOL_SPEC, False

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    @property
    def info(self) -> Info:
        return self._info

    @property
    def exchange(self) -> Exchange | None:
        return self._exchange

    def has_write_access(self) -> bool:
        return self._exchange is not None

    def _lazy_action_ws(self) -> HyperliquidExchangeActionWs:
        if self._action_ws is None:
            self._action_ws = HyperliquidExchangeActionWs(
                self._settings.hl_ws_url,
                timeout_s=float(self._settings.action_ws_timeout_seconds),
            )
        return self._action_ws

    def _exchange_http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        return self._exchange.post("/exchange", payload)

    def send_exchange_action(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """
        Deliver a signed POST /exchange JSON body via WebSocket (preferred) or HTTP fallback.

        Returns (response_json, transport_mode).
        """
        if self._settings.action_ws_enabled:
            try:
                t0 = time.perf_counter()
                resp = self._lazy_action_ws().send_signed_exchange_payload(payload)
                self.last_exchange_transport_write_ms = (time.perf_counter() - t0) * 1000.0
                self.last_exchange_transport_mode = "ws"
                return resp, "ws"
            except Exception:
                logger.exception("exchange_action_ws_failed")
                if not self._settings.action_http_fallback_enabled:
                    raise
        resp = self._exchange_http_post(payload)
        self.last_exchange_transport_mode = "http"
        return resp, "http"

    def _signed_post_only_limit_payload(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        ex = self._exchange
        order: dict[str, Any] = {
            "coin": symbol,
            "is_buy": is_buy,
            "sz": sz,
            "limit_px": limit_px,
            "order_type": {"limit": {"tif": "Alo"}},
            "reduce_only": False,
        }
        if client_order_id:
            order["cloid"] = Cloid.from_str(client_order_id)
        ow = order_request_to_order_wire(order, ex.info.name_to_asset(order["coin"]))
        order_action = order_wires_to_order_action([ow], None, "na")
        ts = get_timestamp_ms()
        sig = sign_l1_action(
            ex.wallet,
            order_action,
            ex.vault_address,
            ts,
            ex.expires_after,
            ex.base_url == MAINNET_API_URL,
        )
        return {
            "action": order_action,
            "nonce": ts,
            "signature": sig,
            "vaultAddress": ex.vault_address
            if order_action["type"] not in ("usdClassTransfer", "sendAsset")
            else None,
            "expiresAfter": ex.expires_after,
        }

    def _signed_cancel_oid_payload(self, symbol: str, oid: int) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        ex = self._exchange
        cancel_action: dict[str, Any] = {
            "type": "cancel",
            "cancels": [{"a": ex.info.name_to_asset(symbol), "o": oid}],
        }
        ts = get_timestamp_ms()
        sig = sign_l1_action(
            ex.wallet,
            cancel_action,
            ex.vault_address,
            ts,
            ex.expires_after,
            ex.base_url == MAINNET_API_URL,
        )
        return {
            "action": cancel_action,
            "nonce": ts,
            "signature": sig,
            "vaultAddress": ex.vault_address
            if cancel_action["type"] not in ("usdClassTransfer", "sendAsset")
            else None,
            "expiresAfter": ex.expires_after,
        }

    def _signed_cancel_cloid_payload(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        ex = self._exchange
        cl = Cloid.from_str(client_order_id)
        cancel_action: dict[str, Any] = {
            "type": "cancelByCloid",
            "cancels": [{"asset": ex.info.name_to_asset(symbol), "cloid": cl.to_raw()}],
        }
        ts = get_timestamp_ms()
        sig = sign_l1_action(
            ex.wallet,
            cancel_action,
            ex.vault_address,
            ts,
            ex.expires_after,
            ex.base_url == MAINNET_API_URL,
        )
        return {
            "action": cancel_action,
            "nonce": ts,
            "signature": sig,
            "vaultAddress": ex.vault_address
            if cancel_action["type"] not in ("usdClassTransfer", "sendAsset")
            else None,
            "expiresAfter": ex.expires_after,
        }

    def reset_market_data_transport(self) -> None:
        """
        Recreate the REST Info client (shallow reset). Used when market-data instrumentation
        detects a silent snapshot stall and ``MARKET_DATA_RESET_TRANSPORT_ON_STALL`` is enabled.
        """
        self._info = Info(self._settings.hl_base_url, skip_ws=True)

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        def _go() -> BestBidAsk:
            snap = self._info.l2_snapshot(symbol)
            levels = snap.get("levels") or [[], []]
            bids = levels[0] if len(levels) > 0 else []
            asks = levels[1] if len(levels) > 1 else []
            best_bid = float(bids[0]["px"]) if bids else None
            best_ask = float(asks[0]["px"]) if asks else None
            mid: float | None = None
            spread_bps: float | None = None
            if best_bid and best_ask and best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2.0
                spread_bps = (best_ask - best_bid) / mid * 10_000.0
            ts_ms = snap.get("time")
            return BestBidAsk(
                symbol=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid,
                spread_bps=spread_bps,
                ts_exchange_ms=int(ts_ms) if ts_ms is not None else None,
            )

        return self._retry("l2_snapshot", _go)

    def fetch_user_state(self, address: str) -> dict[str, Any]:
        return self._retry(
            "clearinghouseState",
            lambda: self._info.user_state(address),
        )

    def _spot_usdc_balances(self, address: str) -> tuple[Optional[float], Optional[float]]:
        """
        USDC (total, total - hold) from ``spotClearinghouseState``.

        Under **unified account** (and portfolio margin), Hyperliquid documents this as the
        source of truth for trading collateral; perps ``marginSummary.accountValue`` may read 0.
        """
        def _go() -> Any:
            return self._info.spot_user_state(address)

        try:
            raw = self._retry("spotClearinghouseState", _go)
        except Exception:
            logger.debug(
                "spotClearinghouseState failed user=%s… (no unified-account fallback)",
                address[:10],
                exc_info=True,
            )
            return None, None
        return _usdc_from_spot_clearinghouse_payload(raw)

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        st = self.fetch_user_state(address)
        perps_eq, perps_wd = _perps_equity_and_withdrawable_from_user_state(st)
        spot_total, spot_free = self._spot_usdc_balances(address)
        equity, withdrawable = _merge_account_snapshot_balances(
            perps_eq, perps_wd, spot_total, spot_free
        )
        return AccountSnapshot(
            equity_usd=equity,
            cash_usd=withdrawable,
            withdrawable_usd=withdrawable,
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        """
        Perp position for ``symbol`` from ``clearinghouseState`` (default / main dex).

        Unified accounts still expose cross/perp positions here for the primary perp DEX; collateral
        for PnL / risk display is merged with spot in :meth:`fetch_account_snapshot`.
        """
        st = self.fetch_user_state(address)
        positions = st.get("assetPositions") or []
        for ap in positions:
            pos = ap.get("position") or {}
            if pos.get("coin") != symbol:
                continue
            szi = pos.get("szi", "0")
            try:
                qty = float(szi)
            except (TypeError, ValueError):
                qty = 0.0
            ep = pos.get("entryPx")
            try:
                entry = float(ep) if ep is not None else None
            except (TypeError, ValueError):
                entry = None
            pv = pos.get("positionValue", "0")
            try:
                notional = abs(float(pv))
            except (TypeError, ValueError):
                notional = 0.0
            upnl = pos.get("unrealizedPnl", "0")
            try:
                unreal = float(upnl)
            except (TypeError, ValueError):
                unreal = 0.0
            mark: float | None = None
            if abs(qty) > 1e-12 and notional > 0:
                mark = notional / abs(qty)
            return PositionSnapshot(
                symbol=symbol,
                position_qty=qty,
                avg_entry_price=entry,
                mark_price=mark,
                position_notional=notional,
                unrealized_pnl_usd=unreal,
            )
        return PositionSnapshot(
            symbol=symbol,
            position_qty=0.0,
            avg_entry_price=None,
            mark_price=None,
            position_notional=0.0,
            unrealized_pnl_usd=0.0,
        )

    def fetch_open_orders_raw(self, address: str) -> list[HLOpenOrderRaw]:
        def _go() -> list[HLOpenOrderRaw]:
            raw = self._info.open_orders(address)
            out: list[HLOpenOrderRaw] = []
            for o in raw:
                if o.get("coin") != self._settings.symbol:
                    continue
                raw_cloid = o.get("cloid")
                cloid_s: str | None
                if isinstance(raw_cloid, str) and raw_cloid.strip():
                    cloid_s = raw_cloid.strip()
                else:
                    cloid_s = None
                out.append(
                    HLOpenOrderRaw(
                        oid=int(o["oid"]),
                        coin=str(o["coin"]),
                        side=_hl_side_to_side(str(o.get("side", "A"))),
                        limit_px=float(o["limitPx"]),
                        sz=float(o["sz"]),
                        timestamp=int(o.get("timestamp", 0)),
                        cloid=cloid_s,
                    )
                )
            return out

        return self._retry("openOrders", _go)

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[HLFillRaw]:
        def _go() -> list[HLFillRaw]:
            raw = self._info.user_fills(address)
            out: list[HLFillRaw] = []
            for f in raw:
                if f.get("coin") != symbol:
                    continue
                oid = f.get("oid")
                try:
                    oid_i = int(oid) if oid is not None else None
                except (TypeError, ValueError):
                    oid_i = None
                tid = str(f.get("hash", "")) + "_" + str(f.get("time", ""))
                fee = f.get("fee")
                try:
                    fee_f = float(fee) if fee is not None else 0.0
                except (TypeError, ValueError):
                    fee_f = 0.0
                cp = f.get("closedPnl", "0")
                try:
                    closed = float(cp)
                except (TypeError, ValueError):
                    closed = 0.0
                out.append(
                    HLFillRaw(
                        fill_id=tid,
                        oid=oid_i,
                        coin=str(f["coin"]),
                        side=_hl_side_to_side(str(f.get("side", "A"))),
                        px=float(f["px"]),
                        sz=float(f["sz"]),
                        fee=fee_f,
                        time_ms=int(f.get("time", 0)),
                        closed_pnl=closed,
                        raw=dict(f),
                    )
                )
            return out

        return self._retry("userFills", _go)

    def place_post_only_limit(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        *,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized (TRADING_ENABLED + keys required)")
        if reduce_only:
            raise NotImplementedError(
                "Hyperliquid adapter: reduce_only=True is not yet "
                "propagated through this client; refusing to silently "
                "send a non-reduce-only order. Wire reduce_only "
                "through the order submission before using soft-flatten "
                "on Hyperliquid."
            )
        tick = self._symbol_spec.price_tick
        wire_p = wire_format_preview_limit_px(limit_px)
        logger.info(
            "hyperliquid_order_submit kind=post_only_limit coin=%s is_buy=%s sz=%s limit_px_float=%s "
            "szDecimals=%s hl_meta_decimal_grid_tick=%s hl_max_sig_figs_nonint=%s "
            "hl_wire_limit_p=%s hl_price_normalize_pipeline=%s",
            symbol,
            is_buy,
            sz,
            limit_px,
            self._symbol_spec.sz_decimals,
            tick,
            HL_PERP_MAX_SIG_FIGS,
            wire_p,
            HL_PERP_LIMIT_PRICE_PIPELINE_ID,
        )
        validate_hyperliquid_perp_limit_for_submit(
            limit_px,
            self._symbol_spec.sz_decimals,
            source_tick=tick,
        )

        def _go() -> dict[str, Any]:
            t_sign0 = time.perf_counter()
            payload = self._signed_post_only_limit_payload(
                symbol, is_buy, sz, limit_px, client_order_id=client_order_id
            )
            self.last_exchange_signing_ms = (time.perf_counter() - t_sign0) * 1000.0
            resp, _mode = self.send_exchange_action(payload)
            return resp

        return self._retry("order_post_only", _go)

    def place_ioc_reduce_only(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
    ) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        order_type: dict[str, Any] = {"limit": {"tif": "Ioc"}}

        def _go() -> dict[str, Any]:
            tick = self._symbol_spec.price_tick
            logger.info(
                "hyperliquid_order_submit kind=ioc_reduce coin=%s is_buy=%s sz=%s limit_px_float=%s "
                "hl_meta_decimal_grid_tick=%s hl_wire_limit_p=%s hl_price_normalize_pipeline=%s",
                symbol,
                is_buy,
                sz,
                limit_px,
                tick,
                wire_format_preview_limit_px(limit_px),
                HL_PERP_LIMIT_PRICE_PIPELINE_ID,
            )
            validate_hyperliquid_perp_limit_for_submit(
                limit_px,
                self._symbol_spec.sz_decimals,
                source_tick=tick,
            )
            return self._exchange.order(
                symbol, is_buy, sz, limit_px, order_type, reduce_only=True
            )

        return self._retry("order_ioc_reduce", _go)

    def market_close(self, symbol: str, sz: Optional[float] = None) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")

        def _go() -> dict[str, Any]:
            return self._exchange.market_close(symbol, sz)

        return self._retry("market_close", _go)

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")

        def _go() -> dict[str, Any]:
            payload = self._signed_cancel_oid_payload(symbol, oid)
            resp, _mode = self.send_exchange_action(payload)
            return resp

        return self._retry("cancel", _go)

    def cancel_order_by_cloid(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")

        def _go() -> dict[str, Any]:
            payload = self._signed_cancel_cloid_payload(symbol, client_order_id)
            resp, _mode = self.send_exchange_action(payload)
            return resp

        return self._retry("cancel_by_cloid", _go)

    def query_order_status_by_cloid(self, address: str, client_order_id: str) -> dict[str, Any]:
        def _go() -> dict[str, Any]:
            return self._info.query_order_by_cloid(address, Cloid.from_str(client_order_id))

        return self._retry("orderStatus_cloid", _go)
