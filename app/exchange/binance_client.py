"""Binance Futures USDM adapter implementing :class:`PerpExchangeAdapter`.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 2.

Differences from the existing Bluefin / Hyperliquid / GRVT adapters:

* **Auth is HMAC-SHA256**, not Sui session-key + intent-bytes. Every
  authenticated request adds ``timestamp`` + ``recvWindow`` query
  params and a final ``signature`` over the urlencoded query string.
  Spot, USDM-Futures, and COIN-Futures all share this scheme; only
  the base URL changes.
* **Cancels are synchronous.** ``DELETE /fapi/v1/order`` returns the
  cancelled order's status immediately. We do NOT implement the
  ``has_pending_cancel`` capability (Bluefin needs it because its
  cancel is async; Binance doesn't).
* **Post-only TIF is ``GTX``** — Binance's name for "Good-Till-Crossing":
  if the order would cross at place time, the venue rejects with code
  ``-5022`` (post-only would cross). Adapter maps this to
  ``outcome="exchange_rejected"`` reason ``post_only_would_cross``,
  same shape as Hyperliquid's ``cloid_already_used`` etc.
* **Market close uses plain ``type=MARKET`` + ``reduceOnly=true`` +
  ``timeInForce=IOC``.** Unlike Bluefin (BUG-014/015), Binance's
  MARKET endpoint does not zero out a `priceE9`-equivalent — there's
  no signed price for MARKET orders at all, the venue uses live mid.

Operational model:

* ``binance_account_address`` doesn't exist — Binance uses an account
  ID indirectly via the API key. The bot's ``venue_account_address``
  helper returns a hash of the API key prefix as a stable identifier
  (purely for log correlation).
* ``listenKey`` lifecycle is in :mod:`app.exchange.binance_ws` (not
  this module). The REST client provides ``spawn_listen_key`` and
  ``keepalive_listen_key`` helpers used by the WS thread.
* Signed-request clock skew: Binance rejects requests where
  ``timestamp + recvWindow < server_now``. We use ``time.time() *
  1000`` and a configurable ``recvWindow`` (default 5000 ms). On
  Tokyo EC2 with NTP enabled this is comfortably safe.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from app.config import Settings
from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw
from app.exchange.binance_responses import (
    interpret_binance_cancel_response,
    interpret_binance_order_status_response,
    interpret_binance_place_response,
    make_deterministic_binance_client_order_id,
)
from app.exchange.exchange_retry import exchange_call_with_retry
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.exchange.symbol_spec import SymbolSpec, FALLBACK_SYMBOL_SPEC
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)


# Binance USDM Futures order types we use.
_ORDER_TYPE_LIMIT = "LIMIT"
_ORDER_TYPE_MARKET = "MARKET"
_TIF_GTX = "GTX"   # post-only — cancel if would cross
_TIF_IOC = "IOC"   # immediate-or-cancel — for market_close


def _normalize_binance_symbol(symbol: str) -> str:
    """Binance Futures symbols are uppercase with no separator
    (``DOGEUSDT``, ``BTCUSDT``). Keep callers tolerant about case.
    """
    return (symbol or "").strip().upper()


def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class BinanceClient:
    """Binance Futures USDM adapter.

    Lifetime: constructed once at bot startup; bootstraps the symbol
    spec via ``GET /fapi/v1/exchangeInfo`` synchronously and caches.
    A failed bootstrap leaves ``symbol_spec_fetched_ok=False`` and the
    bot uses ``FALLBACK_SYMBOL_SPEC`` (matching Bluefin's pattern).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._symbol = _normalize_binance_symbol(settings.symbol)
        self._api_key = (settings.binance_api_key or "").strip()
        self._api_secret = (settings.binance_api_secret or "").strip()
        self._rest_url = (settings.binance_rest_url or "").strip().rstrip("/")
        self._recv_window_ms = int(settings.binance_recv_window_ms)

        # Persistent HTTP client — connection pooling across the bot's
        # lifetime is meaningful at 2 Hz quote cadence.
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=4.0, pool=8.0),
        )

        # Telemetry counters surfaced via rest_runtime_counters().
        self._rest_call_counts: dict[str, int] = {}
        self._rest_retry_counts: dict[str, int] = {}
        self._rest_429_retries_total: int = 0
        self._used_weight_1m: int = 0
        self._order_count_1m: int = 0
        self._lock = threading.Lock()

        # Symbol spec — bootstrapped synchronously below.
        self.symbol_spec_fetched_ok: bool = False
        self._symbol_spec: SymbolSpec = FALLBACK_SYMBOL_SPEC
        try:
            self._symbol_spec = self._bootstrap_symbol_spec()
            self.symbol_spec_fetched_ok = True
            logger.info(
                "binance_symbol_spec_bootstrap_success symbol=%s "
                "price_tick=%s size_step=%s min_size=%s min_notional=%s",
                self._symbol,
                self._symbol_spec.price_tick,
                self._symbol_spec.size_step,
                self._symbol_spec.min_size,
                self._symbol_spec.min_notional_usd,
            )
        except Exception:
            logger.exception(
                "binance_symbol_spec_bootstrap_failed symbol=%s — "
                "falling back to FALLBACK_SYMBOL_SPEC",
                self._symbol,
            )

    # ------------------------------------------------------------------
    # Adapter surface
    # ------------------------------------------------------------------

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    def has_write_access(self) -> bool:
        return bool(self._api_key and self._api_secret)

    # ------------------------------------------------------------------
    # HTTP / signing
    # ------------------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        """Append ``timestamp`` + ``recvWindow`` + ``signature`` to
        ``params``. Returns a NEW dict (params is not mutated).

        Binance accepts the signature as either a query parameter or a
        form-encoded body field; we use the query-parameter form for
        all methods (POST/PUT/DELETE) which is how the official
        examples are shaped.
        """
        out = dict(params)
        out["timestamp"] = int(time.time() * 1000)
        out["recvWindow"] = self._recv_window_ms
        # urlencode preserves insertion order; Binance does not care
        # about parameter ordering as long as the signature was
        # computed over the exact bytes that get sent.
        qs = urlencode(out, doseq=True)
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        out["signature"] = sig
        return out

    def _request(
        self,
        op: str,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        signed: bool = False,
        retry: bool = True,
    ) -> dict[str, Any]:
        """Send one REST request. Returns the parsed JSON response.

        On non-2xx the response dict still comes back with a top-level
        ``code`` (Binance's error code, negative integer) and ``msg``.
        Callers / interpreters branch on ``code``.

        Binance also surfaces ``X-MBX-USED-WEIGHT-1M`` and
        ``X-MBX-ORDER-COUNT-1M`` response headers — we cache the
        latest values so ``rest_runtime_counters`` can show them.
        """
        url = f"{self._rest_url}{path}"
        params_full = dict(params or {})
        if signed:
            params_full = self._sign(params_full)
        headers = {"X-MBX-APIKEY": self._api_key} if signed else {}

        with self._lock:
            self._rest_call_counts[op] = self._rest_call_counts.get(op, 0) + 1

        def _do() -> dict[str, Any]:
            try:
                resp = self._http.request(
                    method,
                    url,
                    params=params_full,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                # Transport-level failure (DNS, timeout, RST). Treat
                # as a Binance-shaped 5xx so callers can branch.
                logger.warning(
                    "binance_request_transport_failed op=%s method=%s "
                    "path=%s err=%s",
                    op,
                    method,
                    path,
                    str(exc)[:200],
                )
                return {"code": -1000, "msg": f"transport_error:{exc}"}

            # Capture rate-limit headers regardless of status code.
            try:
                with self._lock:
                    if "X-MBX-USED-WEIGHT-1M" in resp.headers:
                        self._used_weight_1m = int(
                            resp.headers["X-MBX-USED-WEIGHT-1M"]
                        )
                    if "X-MBX-ORDER-COUNT-1M" in resp.headers:
                        self._order_count_1m = int(
                            resp.headers["X-MBX-ORDER-COUNT-1M"]
                        )
            except (TypeError, ValueError):
                pass

            if resp.status_code == 429 or resp.status_code == 418:
                # 429 = rate limit hit; 418 = ban (rare). Both should be
                # retried with backoff via the wrapper.
                with self._lock:
                    self._rest_429_retries_total += 1
                logger.warning(
                    "binance_rate_limit op=%s status=%s "
                    "retry_after=%s used_weight_1m=%s",
                    op,
                    resp.status_code,
                    resp.headers.get("Retry-After"),
                    self._used_weight_1m,
                )
                # Raise so the retry wrapper backs off. After max
                # retries the wrapper returns the last error dict.
                raise httpx.HTTPStatusError(
                    f"http_{resp.status_code}",
                    request=resp.request,
                    response=resp,
                )

            try:
                body = resp.json()
            except ValueError:
                body = {"code": -1, "msg": resp.text[:500]}

            if not isinstance(body, dict):
                # Binance returns arrays for some endpoints (open
                # orders, user trades, depth). Wrap into a dict so
                # downstream callers can inspect ``data``.
                body = {"data": body}

            # Stamp the HTTP status code so interpreters can branch on
            # it without re-running an HTTPX response object.
            body.setdefault("_http_status", resp.status_code)
            return body

        if retry:
            try:
                return exchange_call_with_retry(op, _do, self._settings)
            except Exception as exc:
                with self._lock:
                    self._rest_retry_counts[op] = (
                        self._rest_retry_counts.get(op, 0) + 1
                    )
                logger.exception(
                    "binance_request_retry_exhausted op=%s err=%s",
                    op,
                    str(exc)[:200],
                )
                return {"code": -1000, "msg": f"retry_exhausted:{exc}"[:500]}
        return _do()

    # ------------------------------------------------------------------
    # Symbol-spec bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_symbol_spec(self) -> SymbolSpec:
        """Fetch ``GET /fapi/v1/exchangeInfo`` and build a SymbolSpec
        from the matched symbol's filters. Raises on failure.
        """
        resp = self._request(
            "exchange_info",
            "GET",
            "/fapi/v1/exchangeInfo",
            signed=False,
            retry=True,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            raise RuntimeError(
                f"exchangeInfo failed: code={resp['code']} msg={resp.get('msg')}"
            )
        symbols = resp.get("symbols")
        if not isinstance(symbols, list):
            raise RuntimeError("exchangeInfo missing 'symbols' array")
        row = next(
            (s for s in symbols if s.get("symbol") == self._symbol),
            None,
        )
        if row is None:
            raise RuntimeError(
                f"symbol {self._symbol!r} not in Binance Futures USDM universe"
            )
        # Filters: PRICE_FILTER (tickSize), LOT_SIZE (stepSize, minQty),
        # MIN_NOTIONAL (notional).
        filters = {f.get("filterType"): f for f in row.get("filters", [])}
        price_tick = _coerce_float(
            (filters.get("PRICE_FILTER") or {}).get("tickSize"),
            default=0.0,
        )
        lot = filters.get("LOT_SIZE") or {}
        size_step = _coerce_float(lot.get("stepSize"), default=0.0)
        min_size = _coerce_float(lot.get("minQty"), default=0.0)
        min_notional = _coerce_float(
            (filters.get("MIN_NOTIONAL") or {}).get("notional"),
            default=5.0,
        )
        if price_tick <= 0 or size_step <= 0:
            raise RuntimeError(
                f"invalid Binance filters for {self._symbol}: "
                f"tick={price_tick} step={size_step}"
            )
        # Binance reports quantityPrecision as a separate top-level
        # field; convert that to sz_decimals for compatibility with
        # the existing make_client_order_id and SymbolSpec consumers.
        sz_decimals = int(row.get("quantityPrecision", 0) or 0)
        return SymbolSpec(
            price_tick=price_tick,
            size_step=size_step,
            min_size=min_size if min_size > 0 else size_step,
            min_notional_usd=max(min_notional, 5.0),  # exchange floor
            sz_decimals=sz_decimals,
            source="binance_meta",
        )

    # ------------------------------------------------------------------
    # Market / account reads
    # ------------------------------------------------------------------

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        sym = _normalize_binance_symbol(symbol)
        resp = self._request(
            "book_ticker",
            "GET",
            "/fapi/v1/ticker/bookTicker",
            params={"symbol": sym},
            signed=False,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            return BestBidAsk(
                symbol=sym,
                best_bid=None,
                best_ask=None,
                mid_price=None,
                spread_bps=None,
            )
        bid = _coerce_float(resp.get("bidPrice")) or None
        ask = _coerce_float(resp.get("askPrice")) or None
        bid_sz = _coerce_float(resp.get("bidQty")) or None
        ask_sz = _coerce_float(resp.get("askQty")) or None
        mid: Optional[float] = None
        spread_bps: Optional[float] = None
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_bps = (ask - bid) / mid * 10_000.0
        return BestBidAsk(
            symbol=sym,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            spread_bps=spread_bps,
            bid_size=bid_sz,
            ask_size=ask_sz,
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        """``GET /fapi/v2/positionRisk?symbol=…`` returns a list of
        position rows (one per symbol-direction in hedge mode; one in
        one-way mode). The bot expects one-way mode (see Phase 0
        operator setup).
        """
        del address  # Binance uses API key auth, not address
        sym = _normalize_binance_symbol(symbol)
        resp = self._request(
            "position_risk",
            "GET",
            "/fapi/v2/positionRisk",
            params={"symbol": sym},
            signed=True,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            raise RuntimeError(
                f"fetch_position failed: code={resp['code']} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data", resp)
        if not isinstance(rows, list):
            raise RuntimeError(
                f"fetch_position unexpected response: {str(resp)[:200]}"
            )
        pos_qty = 0.0
        entry = 0.0
        mark = 0.0
        unreal = 0.0
        notional = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("symbol") != sym:
                continue
            # one-way mode reports positionAmt with sign
            pos_qty = _coerce_float(row.get("positionAmt"), default=0.0)
            entry = _coerce_float(row.get("entryPrice"), default=0.0)
            mark = _coerce_float(row.get("markPrice"), default=0.0)
            unreal = _coerce_float(row.get("unRealizedProfit"), default=0.0)
            notional = abs(pos_qty) * (mark if mark > 0 else entry)
            break
        return PositionSnapshot(
            symbol=sym,
            position_qty=pos_qty,
            avg_entry_price=entry if entry > 0 else None,
            mark_price=mark if mark > 0 else None,
            position_notional=notional,
            unrealized_pnl_usd=unreal,
        )

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        """``GET /fapi/v2/account`` returns aggregate balances. We
        surface ``totalWalletBalance`` (margin USD) as ``equity_usd``
        and ``availableBalance`` as ``withdrawable_usd``.
        """
        del address  # Binance uses API key auth
        resp = self._request(
            "account",
            "GET",
            "/fapi/v2/account",
            signed=True,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            raise RuntimeError(
                f"fetch_account_snapshot failed: code={resp['code']} "
                f"msg={resp.get('msg')}"
            )
        # Binance Futures account fields:
        #   totalWalletBalance     — collateral deposited (margin basis)
        #   availableBalance       — free margin (= equity − used margin)
        #   totalUnrealizedProfit  — sum of open-position MTM
        # AccountSnapshot only carries equity / cash / withdrawable, not
        # a separate unrealized-PnL field (that lives on PositionSnapshot).
        # We surface ``totalWalletBalance`` as both ``equity_usd`` and
        # ``cash_usd`` since on Binance Futures with no open positions
        # they're identical; with open positions, ``cash_usd`` is the
        # realised-PnL slice and ``equity_usd`` includes unrealised.
        equity = _coerce_float(resp.get("totalWalletBalance"), default=0.0)
        avail = _coerce_float(resp.get("availableBalance"), default=0.0)
        unreal = _coerce_float(resp.get("totalUnrealizedProfit"), default=0.0)
        return AccountSnapshot(
            equity_usd=(equity + unreal) if equity > 0 else None,
            cash_usd=equity if equity > 0 else None,
            withdrawable_usd=avail if avail >= 0 else None,
        )

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]:
        del address
        resp = self._request(
            "open_orders",
            "GET",
            "/fapi/v1/openOrders",
            params={"symbol": self._symbol},
            signed=True,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            return []
        rows = resp.get("data", resp)
        if not isinstance(rows, list):
            return []
        out: list[HLOpenOrderRaw] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("symbol") != self._symbol:
                continue
            try:
                oid = int(row.get("orderId") or 0)
            except (TypeError, ValueError):
                continue
            if oid == 0:
                continue
            side_str = str(row.get("side") or "").upper()
            side = Side.BUY if side_str == "BUY" else Side.SELL
            ts_raw = row.get("updateTime") or row.get("time") or 0
            try:
                ts_ms = int(ts_raw)
            except (TypeError, ValueError):
                ts_ms = 0
            out.append(
                HLOpenOrderRaw(
                    oid=oid,
                    coin=self._symbol,
                    side=side,
                    limit_px=_coerce_float(row.get("price"), default=0.0),
                    sz=_coerce_float(row.get("origQty"), default=0.0),
                    timestamp=ts_ms,
                    cloid=str(row.get("clientOrderId") or "") or None,
                )
            )
        return out

    def fetch_recent_fills_raw(
        self, address: str, symbol: str
    ) -> list[FillRaw]:
        del address
        sym = _normalize_binance_symbol(symbol)
        resp = self._request(
            "user_trades",
            "GET",
            "/fapi/v1/userTrades",
            params={"symbol": sym, "limit": 200},
            signed=True,
        )
        if isinstance(resp.get("code"), int) and resp["code"] < 0:
            return []
        rows = resp.get("data", resp)
        if not isinstance(rows, list):
            return []
        out: list[HLFillRaw] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("symbol") != sym:
                continue
            try:
                trade_id = int(row.get("id") or 0)
                order_id = int(row.get("orderId") or 0)
            except (TypeError, ValueError):
                continue
            side_str = str(row.get("side") or "").upper()
            side = Side.BUY if side_str == "BUY" else Side.SELL
            try:
                time_ms = int(row.get("time") or 0)
            except (TypeError, ValueError):
                time_ms = 0
            out.append(
                HLFillRaw(
                    fill_id=str(trade_id),
                    oid=order_id,
                    coin=sym,
                    side=side,
                    px=_coerce_float(row.get("price")),
                    sz=_coerce_float(row.get("qty"), default=0.0),
                    fee=abs(_coerce_float(row.get("commission"), default=0.0)),
                    time_ms=time_ms,
                    closed_pnl=_coerce_float(row.get("realizedPnl"), default=0.0),
                    raw=dict(row),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def _place_order(
        self,
        *,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        post_only: bool,
        reduce_only: bool,
        ioc: bool,
        order_type: str,
        client_order_id: Optional[str],
    ) -> dict[str, Any]:
        sym = _normalize_binance_symbol(symbol)
        cloid = (
            (client_order_id or "").strip()
            or make_deterministic_binance_client_order_id(
                sym,
                Side.BUY if is_buy else Side.SELL,
                "manual",
                limit_px,
                sz,
            )
        )
        params: dict[str, Any] = {
            "symbol": sym,
            "side": "BUY" if is_buy else "SELL",
            "type": order_type,
            "quantity": sz,
            "newClientOrderId": cloid,
            "newOrderRespType": "RESULT",
        }
        # MARKET orders: no price, but use IOC TIF when reduce_only is
        # set (consistent flatten semantics).
        if order_type != _ORDER_TYPE_MARKET:
            params["price"] = limit_px
            if post_only:
                params["timeInForce"] = _TIF_GTX
            elif ioc:
                params["timeInForce"] = _TIF_IOC
            else:
                params["timeInForce"] = "GTC"
        else:
            # Binance MARKET orders don't take a price; reduceOnly +
            # timeInForce are still valid.
            if ioc:
                params["timeInForce"] = _TIF_IOC
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request(
            "create_order",
            "POST",
            "/fapi/v1/order",
            params=params,
            signed=True,
        )

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
        if not self.has_write_access():
            raise RuntimeError("Binance trading credentials missing")
        if reduce_only:
            # Binance USDM DOES support reduceOnly, but the bot's
            # current ``_place_order`` here doesn't propagate it.
            # Refuse loudly so we never silently down-grade a
            # caller's safety guarantee. (When this path is needed
            # for production, propagate ``reduce_only`` through
            # ``_place_order`` and the API request body.)
            raise NotImplementedError(
                "Binance adapter: reduce_only=True is not yet "
                "propagated through this client; refusing to silently "
                "send a non-reduce-only order. Wire reduce_only "
                "through _place_order before using soft-flatten on "
                "Binance."
            )
        return self._place_order(
            symbol=symbol,
            is_buy=is_buy,
            sz=sz,
            limit_px=limit_px,
            post_only=True,
            reduce_only=False,
            ioc=False,
            order_type=_ORDER_TYPE_LIMIT,
            client_order_id=client_order_id,
        )

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        sym = _normalize_binance_symbol(symbol)
        return self._request(
            "cancel_order",
            "DELETE",
            "/fapi/v1/order",
            params={"symbol": sym, "orderId": int(oid)},
            signed=True,
        )

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        sym = _normalize_binance_symbol(symbol)
        cloid = str(client_order_id or "").strip()
        if not cloid:
            return {"code": 400, "msg": "missing_client_order_id"}
        return self._request(
            "cancel_order_by_cloid",
            "DELETE",
            "/fapi/v1/order",
            params={"symbol": sym, "origClientOrderId": cloid},
            signed=True,
        )

    def cancel_all_open_orders(self, symbol: str) -> dict[str, Any]:
        sym = _normalize_binance_symbol(symbol)
        return self._request(
            "cancel_all_open_orders",
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": sym},
            signed=True,
        )

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        del address
        sym = self._symbol
        cloid = str(client_order_id or "").strip()
        if not cloid:
            return {"code": 400, "msg": "missing_client_order_id"}
        return self._request(
            "query_order",
            "GET",
            "/fapi/v1/order",
            params={"symbol": sym, "origClientOrderId": cloid},
            signed=True,
        )

    def market_close(
        self, symbol: str, sz: Optional[float] = None
    ) -> dict[str, Any]:
        """Force-close ``symbol`` via plain MARKET + reduceOnly + IOC.

        Unlike Bluefin (BUG-014/015 retro), Binance's MARKET endpoint
        works as advertised — no priceE9-zeroing quirk, no IOC LIMIT
        workaround needed. The order sweeps the book at any price.
        Reduce-only ensures the order can't grow inventory.
        """
        pos = self.fetch_position("", symbol)
        abs_qty = abs(pos.position_qty)
        if abs_qty <= 0:
            return {"data": {"status": "noop_already_flat"}}
        qty = float(sz) if sz is not None and sz > 0 else abs_qty
        buy = pos.position_qty < 0  # closing long → SELL; closing short → BUY
        return self._place_order(
            symbol=symbol,
            is_buy=buy,
            sz=qty,
            limit_px=0.0,  # ignored for MARKET orders on Binance
            post_only=False,
            reduce_only=True,
            ioc=True,
            order_type=_ORDER_TYPE_MARKET,
            client_order_id=None,
        )

    # ------------------------------------------------------------------
    # listenKey lifecycle (used by binance_ws.py)
    # ------------------------------------------------------------------

    def spawn_listen_key(self) -> str:
        """``POST /fapi/v1/listenKey`` — returns the listenKey string.

        Auth requires only the API key header (no signature). Binance
        silently expires keys after 60 min of no PUT keepalive; the WS
        thread should call ``keepalive_listen_key`` every 30 min.
        """
        url = f"{self._rest_url}/fapi/v1/listenKey"
        resp = self._http.post(url, headers={"X-MBX-APIKEY": self._api_key})
        resp.raise_for_status()
        data = resp.json()
        key = (data.get("listenKey") if isinstance(data, dict) else None) or ""
        if not key:
            raise RuntimeError(f"spawn_listen_key empty: {str(data)[:200]}")
        return str(key)

    def keepalive_listen_key(self) -> bool:
        """``PUT /fapi/v1/listenKey`` — extends the active listenKey by
        another 60 min. Returns True on 2xx; False on non-2xx (the WS
        thread responds by re-spawning a fresh key).
        """
        url = f"{self._rest_url}/fapi/v1/listenKey"
        try:
            resp = self._http.put(
                url, headers={"X-MBX-APIKEY": self._api_key}
            )
            return 200 <= resp.status_code < 300
        except httpx.HTTPError:
            return False

    def close_listen_key(self) -> None:
        """``DELETE /fapi/v1/listenKey`` — explicit close on shutdown."""
        url = f"{self._rest_url}/fapi/v1/listenKey"
        try:
            self._http.delete(url, headers={"X-MBX-APIKEY": self._api_key})
        except httpx.HTTPError:
            pass

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def rest_runtime_counters(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for op, n in self._rest_call_counts.items():
                out[f"{op}_calls"] = int(n)
            for op, n in self._rest_retry_counts.items():
                out[f"{op}_retries"] = int(n)
            out["binance_rest_429_retries_total"] = int(self._rest_429_retries_total)
            out["binance_used_weight_1m"] = int(self._used_weight_1m)
            out["binance_order_count_1m"] = int(self._order_count_1m)
        return out

    # ------------------------------------------------------------------
    # Wire-format interpreters (delegated to binance_responses.py)
    # ------------------------------------------------------------------

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_binance_place_response(resp)

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        return interpret_binance_cancel_response(resp)

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_binance_order_status_response(resp)

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        return make_deterministic_binance_client_order_id(
            _normalize_binance_symbol(symbol), side, quote_cycle_id, price, size
        )


__all__ = ["BinanceClient"]
