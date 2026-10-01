"""GRVT adapter with strict instrument metadata bootstrap."""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from http.cookies import SimpleCookie
from typing import Any, Callable, Optional, TypeVar

import eth_account
import httpx
from eth_account.messages import encode_typed_data

from app.config import Settings
from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw
from app.exchange.exchange_retry import exchange_call_with_retry
from app.exchange.grvt_responses import (
    interpret_grvt_cancel_response,
    interpret_grvt_order_status_response,
    interpret_grvt_place_order_response,
    make_deterministic_grvt_client_order_id,
)
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.exchange.symbol_spec import SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)
T = TypeVar("T")

_CHAIN_IDS: dict[str, int] = {
    "prod": 325,
    "testnet": 326,
    "staging": 327,
    "dev": 327,
}

_ORDER_EIP712_TYPES: dict[str, Any] = {
    "Order": [
        {"name": "subAccountID", "type": "uint64"},
        {"name": "isMarket", "type": "bool"},
        {"name": "timeInForce", "type": "uint8"},
        {"name": "postOnly", "type": "bool"},
        {"name": "reduceOnly", "type": "bool"},
        {"name": "legs", "type": "OrderLeg[]"},
        {"name": "nonce", "type": "uint32"},
        {"name": "expiration", "type": "int64"},
    ],
    "OrderLeg": [
        {"name": "assetID", "type": "uint256"},
        {"name": "contractSize", "type": "uint64"},
        {"name": "limitPrice", "type": "uint64"},
        {"name": "isBuyingContract", "type": "bool"},
    ],
}


# Module-level constants so the per-order signing path doesn't re-materialise
# a fresh mapping / Decimal every call. ``Decimal(1_000_000_000)`` in
# particular is not free — at ~20 Hz quoting it shows up as real CPU.
_TIF_MAP: dict[str, int] = {
    "GOOD_TILL_TIME": 1,
    "ALL_OR_NONE": 2,
    "IMMEDIATE_OR_CANCEL": 3,
    "FILL_OR_KILL": 4,
}
_PRICE_MULTIPLIER_DEC: Decimal = Decimal(1_000_000_000)
_FIVE_MIN_NS: int = 5 * 60 * 1_000_000_000
_ORDER_EXPIRATION_NS: int = _FIVE_MIN_NS


class _InstrumentSignCache:
    """Cached per-instrument signing parameters.

    Populated on first signing attempt for a given instrument. Holds the
    parsed ``instrument_hash`` (expensive: hex/decimal string detection) and
    the ``Decimal(10) ** base_decimals`` size multiplier, which is otherwise
    recomputed for every place / cancel-replace cycle.
    """

    __slots__ = ("instrument_hash", "size_decimals", "size_multiplier_dec")

    def __init__(
        self,
        *,
        instrument_hash: int,
        size_decimals: int,
        size_multiplier_dec: Decimal,
    ) -> None:
        self.instrument_hash = instrument_hash
        self.size_decimals = size_decimals
        self.size_multiplier_dec = size_multiplier_dec


def _default_grvt_urls(env: str) -> tuple[str, str, str, str, str]:
    env_lc = (env or "prod").strip().lower()
    if env_lc == "prod":
        edge = "https://edge.grvt.io"
        trade = "https://trades.grvt.io"
        md = "https://market-data.grvt.io"
        pub_ws = "wss://market-data.grvt.io/ws/full"
        prv_ws = "wss://trades.grvt.io/ws/full"
        return edge, trade, md, pub_ws, prv_ws
    if env_lc == "testnet":
        edge = "https://edge.testnet.grvt.io"
        trade = "https://trades.testnet.grvt.io"
        md = "https://market-data.testnet.grvt.io"
        pub_ws = "wss://market-data.testnet.grvt.io/ws/full"
        prv_ws = "wss://trades.testnet.grvt.io/ws/full"
        return edge, trade, md, pub_ws, prv_ws
    suffix = f"{env_lc}.gravitymarkets.io"
    edge = f"https://edge.{suffix}"
    trade = f"https://trades.{suffix}"
    md = f"https://market-data.{suffix}"
    pub_ws = f"wss://market-data.{suffix}/ws/full"
    prv_ws = f"wss://trades.{suffix}/ws/full"
    return edge, trade, md, pub_ws, prv_ws


def _ns_str_to_ms(v: Any) -> int:
    try:
        ns = int(str(v))
        return int(ns // 1_000_000)
    except (TypeError, ValueError):
        return 0


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_instrument(symbol: str) -> str:
    s = (symbol or "").strip()
    if "_" in s:
        return s
    return f"{s}_USDT_Perp"


def _split_instrument(inst: str) -> tuple[list[str], list[str], list[str]]:
    parts = inst.split("_")
    if len(parts) >= 3:
        kind = "PERPETUAL" if parts[2].lower() == "perp" else parts[2].upper()
        return [kind], [parts[0]], [parts[1]]
    return ["PERPETUAL"], [inst], ["USDT"]


def _grvt_oid_to_int(order_id: Any) -> Optional[int]:
    """Parse a GRVT order id into a non-zero int, or None if absent / invalid.

    Real GRVT oids are never zero (see :func:`_coerce_grvt_order_id`). Any
    ``0`` / ``"0"`` / ``"0x0"`` is a placeholder, not an identifier.
    """
    if order_id is None:
        return None
    if isinstance(order_id, int):
        return order_id if order_id != 0 else None
    s = str(order_id).strip()
    if not s:
        return None
    try:
        if s.lower().startswith("0x"):
            parsed = int(s, 16)
        else:
            parsed = int(s)
    except (TypeError, ValueError):
        return None
    return parsed if parsed != 0 else None


def _int_oid_to_grvt_order_id(oid: int) -> str:
    if oid < 0:
        return str(oid)
    if oid > (1 << 63):
        return "0x" + format(oid, "x")
    return str(oid)


def _decimal_from_scaled_int(scaled: int, decimals: int) -> str:
    """Format a scaled integer (size*10^d or price*10^9) back to a decimal string.

    GRVT recomputes the EIP-712 message from the REST leg ``size`` / ``limit_price``
    strings (multiplied by base_decimals and 10^9 respectively). The signature only
    verifies if those strings round to the exact integers we signed, so we format
    them from the signed integers rather than from the raw float — ``str(0.007142…)``
    drifts at the 15th digit and fails GRVT's signature check.
    """
    if decimals <= 0:
        return str(int(scaled))
    neg = scaled < 0
    n = -scaled if neg else scaled
    s = str(n).rjust(decimals + 1, "0")
    whole = s[:-decimals]
    frac = s[-decimals:].rstrip("0")
    out = whole if not frac else f"{whole}.{frac}"
    return f"-{out}" if neg else out


class GrvtClient:
    """Structural implementation of :class:`PerpExchangeAdapter` for GRVT."""

    symbol_spec_fetched_ok: bool = False

    def __init__(self, *, config: Optional[dict[str, Any]] = None) -> None:
        self._config = dict(config or {})
        self._symbol = _normalize_instrument(str(self._config.get("symbol") or "ETH"))
        self._api_key = str(self._config.get("api_key") or "").strip()
        self._api_secret = str(self._config.get("api_secret") or "").strip()
        self._sub_account_id = str(
            self._config.get("sub_account_id") or self._config.get("account_address") or ""
        ).strip()
        self._env = str(self._config.get("env") or "prod").strip().lower()
        edge, trade, md, _pub_ws, _prv_ws = _default_grvt_urls(self._env)
        self._edge_url = str(self._config.get("edge_url") or edge).rstrip("/")
        self._trade_url = str(self._config.get("trade_url") or trade).rstrip("/")
        self._market_data_url = str(self._config.get("market_data_url") or md).rstrip("/")
        self._timeout = 8.0
        self._http_timeout_s = self._timeout
        self._cookie_gravity = ""
        self._cookie_expiry_epoch = 0.0
        self._grvt_account_id_header = ""
        self._rest_call_counts: dict[str, int] = defaultdict(int)
        self._rest_retry_counts: dict[str, int] = defaultdict(int)
        self._rest_rate_limited_retry_counts: dict[str, int] = defaultdict(int)
        self._http = httpx.Client(timeout=self._timeout)
        self._wallet = (
            eth_account.Account.from_key(self._api_secret) if self._api_secret else None
        )
        self._instruments_by_symbol: dict[str, dict[str, Any]] = {}
        # Per-instrument signing cache. Populated lazily on first signing
        # attempt; invalidated only when :func:`_load_symbol_spec_strict`
        # rebuilds ``_instruments_by_symbol``. Each entry holds the parsed
        # ``instrument_hash``, ``base_decimals`` and the
        # ``Decimal(10) ** base_decimals`` size multiplier — otherwise every
        # place/cancel_replace would repeat those conversions.
        self._instrument_sign_cache: dict[str, _InstrumentSignCache] = {}
        # Pre-build the EIP-712 domain dict and stringified sub-account id.
        # These never change during the lifetime of the client.
        self._chain_id: int = _CHAIN_IDS.get(self._env, 325)
        self._eip712_domain: dict[str, Any] = {
            "name": "GRVT Exchange",
            "version": "0",
            "chainId": self._chain_id,
        }
        self._sub_account_id_str: str = str(self._sub_account_id)
        try:
            self._sub_account_id_int: int = int(self._sub_account_id) if self._sub_account_id else 0
        except (TypeError, ValueError):
            self._sub_account_id_int = 0
        self._wallet_address_str: str = (
            str(self._wallet.address) if self._wallet is not None else ""
        )
        self._symbol_spec = self._load_symbol_spec_strict(self._symbol)
        self.symbol_spec_fetched_ok = True

    def _metadata_endpoint(self) -> str:
        return f"{self._market_data_url}/full/v1/all_instruments"

    @staticmethod
    def _parse_required_positive_float(
        row: dict[str, Any], field: str, *, symbol: str, endpoint: str
    ) -> float:
        raw = row.get(field)
        try:
            v = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"GRVT metadata invalid field={field!r} symbol={symbol!r} "
                f"endpoint={endpoint!r} raw={raw!r}"
            ) from e
        if v <= 0:
            raise ValueError(
                f"GRVT metadata non-positive field={field!r} symbol={symbol!r} "
                f"endpoint={endpoint!r} value={v!r}"
            )
        return v

    @staticmethod
    def _parse_required_int(
        row: dict[str, Any], field: str, *, symbol: str, endpoint: str
    ) -> int:
        raw = row.get(field)
        try:
            if isinstance(raw, str):
                s = raw.strip()
                v = int(s, 16) if s.lower().startswith("0x") else int(s)
            else:
                v = int(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"GRVT metadata invalid int field={field!r} symbol={symbol!r} "
                f"endpoint={endpoint!r} raw={raw!r}"
            ) from e
        return v

    @staticmethod
    def _grvt_symbol_variants(symbol: str) -> list[str]:
        s = (symbol or "").strip()
        out = [s]
        if "_" not in s and s:
            out.append(f"{s}_USDT_Perp")
        return out

    def _load_symbol_spec_strict(self, symbol: str) -> SymbolSpec:
        endpoint = self._metadata_endpoint()
        logger.info(
            "grvt_symbol_spec_bootstrap_start symbol=%s endpoint=%s",
            symbol,
            endpoint,
        )
        try:
            with httpx.Client(timeout=self._http_timeout_s) as cli:
                resp = cli.post(endpoint, json={"is_active": True})
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:
            logger.exception(
                "grvt_symbol_spec_bootstrap_request_failed symbol=%s endpoint=%s",
                symbol,
                endpoint,
            )
            raise RuntimeError(
                f"GRVT symbol metadata bootstrap failed for symbol={symbol!r} "
                f"endpoint={endpoint!r}: request failed"
            ) from e

        rows = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            logger.error(
                "grvt_symbol_spec_bootstrap_bad_payload symbol=%s endpoint=%s payload_type=%s",
                symbol,
                endpoint,
                type(payload).__name__,
            )
            raise RuntimeError(
                f"GRVT symbol metadata bootstrap failed for symbol={symbol!r} "
                f"endpoint={endpoint!r}: missing/invalid result list"
            )

        variants = self._grvt_symbol_variants(symbol)
        row: Optional[dict[str, Any]] = None
        by_symbol: dict[str, dict[str, Any]] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("instrument") or "")
            if name:
                by_symbol[name] = r
            if name in variants:
                row = r
                break
        self._instruments_by_symbol = by_symbol
        if row is None:
            logger.error(
                "grvt_symbol_spec_bootstrap_symbol_not_found symbol=%s endpoint=%s candidates=%s",
                symbol,
                endpoint,
                variants,
            )
            raise RuntimeError(
                f"GRVT symbol metadata bootstrap failed: unknown symbol={symbol!r} "
                f"(candidates={variants!r}) endpoint={endpoint!r}"
            )

        # Trading-critical fields (strictly required; no fallback assumptions).
        price_tick = self._parse_required_positive_float(
            row, "tick_size", symbol=symbol, endpoint=endpoint
        )
        size_step = self._parse_required_positive_float(
            row, "min_size", symbol=symbol, endpoint=endpoint
        )
        min_size = self._parse_required_positive_float(
            row, "min_size", symbol=symbol, endpoint=endpoint
        )
        min_notional = self._parse_required_positive_float(
            row, "min_notional", symbol=symbol, endpoint=endpoint
        )
        base_decimals = self._parse_required_int(
            row, "base_decimals", symbol=symbol, endpoint=endpoint
        )
        instrument_hash = self._parse_required_int(
            row, "instrument_hash", symbol=symbol, endpoint=endpoint
        )
        if base_decimals < 0 or base_decimals > 18:
            raise RuntimeError(
                f"GRVT symbol metadata bootstrap failed: invalid base_decimals={base_decimals!r} "
                f"symbol={symbol!r} endpoint={endpoint!r}"
            )
        if instrument_hash <= 0:
            raise RuntimeError(
                f"GRVT symbol metadata bootstrap failed: invalid instrument_hash={instrument_hash!r} "
                f"symbol={symbol!r} endpoint={endpoint!r}"
            )

        spec = SymbolSpec(
            price_tick=price_tick,
            size_step=size_step,
            min_size=min_size,
            min_notional_usd=min_notional,
            sz_decimals=base_decimals,
            source="grvt_meta",
        )
        logger.info(
            "grvt_symbol_spec_bootstrap_success symbol=%s endpoint=%s "
            "price_tick=%s size_step=%s min_size=%s min_notional=%s base_decimals=%s instrument_hash=%s",
            symbol,
            endpoint,
            spec.price_tick,
            spec.size_step,
            spec.min_size,
            spec.min_notional_usd,
            spec.sz_decimals,
            instrument_hash,
        )
        return spec

    # --- venue metadata ----------------------------------------------------
    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    def has_write_access(self) -> bool:
        return bool(self._api_key and self._wallet is not None and self._sub_account_id)

    def _note_retry(self, operation: str, rate_limited: bool) -> None:
        self._rest_retry_counts[operation] += 1
        if rate_limited:
            self._rest_rate_limited_retry_counts[operation] += 1

    def _retry(self, operation: str, fn: Callable[[], T]) -> T:
        self._rest_call_counts[operation] += 1
        fake = Settings.model_construct(
            exchange_retry_max_attempts=4,
            exchange_retry_base_seconds=0.25,
            exchange_retry_max_backoff_seconds=4.0,
            exchange_rate_limit_extra_delay_seconds=1.5,
        )
        return exchange_call_with_retry(
            operation,
            fn,
            fake,
            on_retry=self._note_retry,
        )

    def _is_cookie_fresh(self) -> bool:
        return bool(self._cookie_gravity) and (self._cookie_expiry_epoch - time.time() > 5.0)

    def _refresh_cookie(self) -> None:
        if self._is_cookie_fresh():
            return
        if not self._api_key:
            raise RuntimeError("GRVT_API_KEY missing")
        resp = self._http.post(
            f"{self._edge_url}/auth/api_key/login",
            headers={"Content-Type": "application/json", "Cookie": "rm=true;"},
            json={"api_key": self._api_key},
        )
        resp.raise_for_status()
        set_cookie = resp.headers.get("set-cookie", "")
        jar = SimpleCookie()
        jar.load(set_cookie)
        gravity = ""
        expiry = 0.0
        if "gravity" in jar:
            gravity = jar["gravity"].value
            expires_s = jar["gravity"].get("expires", "")
            if expires_s:
                try:
                    expiry = time.mktime(time.strptime(expires_s, "%a, %d %b %Y %H:%M:%S %Z"))
                except ValueError:
                    expiry = time.time() + 300.0
        if not gravity:
            gravity = resp.cookies.get("gravity", "")
            expiry = time.time() + 300.0
        self._cookie_gravity = gravity
        self._cookie_expiry_epoch = expiry or (time.time() + 300.0)
        self._grvt_account_id_header = str(resp.headers.get("X-Grvt-Account-Id") or "").strip()
        try:
            data = resp.json()
        except Exception:
            data = {}
        if not self._sub_account_id and isinstance(data, dict):
            sid = data.get("sub_account_id")
            if sid is not None:
                self._sub_account_id = str(sid).strip()

    def _post(self, op: str, url: str, payload: dict[str, Any], *, auth: bool) -> dict[str, Any]:
        def _go() -> dict[str, Any]:
            headers = {"Content-Type": "application/json"}
            if auth:
                self._refresh_cookie()
                if not self._cookie_gravity:
                    raise RuntimeError("GRVT auth cookie missing")
                headers["Cookie"] = f"gravity={self._cookie_gravity}"
                if self._grvt_account_id_header:
                    headers["X-Grvt-Account-Id"] = self._grvt_account_id_header
            resp = self._http.post(url, headers=headers, json=payload)
            body: dict[str, Any]
            try:
                body = dict(resp.json())
            except Exception:
                body = {"status": resp.status_code, "message": resp.text[:500]}
            if resp.status_code >= 400:
                if "code" not in body:
                    body["code"] = resp.status_code
                    body["message"] = body.get("message") or resp.text[:500]
                # Always log non-2xx GRVT responses so order failures are diagnosable
                # from stdout alone. Do not log the request body: it contains the
                # EIP-712 signature (not secret, but noisy) and never the api_key,
                # which is only ever sent during cookie refresh.
                logger.warning(
                    "grvt_http_non_2xx op=%s status=%s url=%s code=%s message=%s",
                    op,
                    resp.status_code,
                    url,
                    body.get("code"),
                    str(body.get("message") or "")[:500],
                )
            return body

        return self._retry(op, _go)

        

    # --- market / account reads -------------------------------------------
    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        inst = _normalize_instrument(symbol)
        resp = self._post(
            "book",
            f"{self._market_data_url}/full/v1/book",
            {"instrument": inst, "depth": 10},
            auth=False,
        )
        result = resp.get("result") if isinstance(resp, dict) else None
        bids = result.get("bids") if isinstance(result, dict) else None
        asks = result.get("asks") if isinstance(result, dict) else None
        best_bid = _to_float((bids or [{}])[0].get("price")) if bids else None
        best_ask = _to_float((asks or [{}])[0].get("price")) if asks else None
        if best_bid is not None and best_bid <= 0:
            best_bid = None
        if best_ask is not None and best_ask <= 0:
            best_ask = None
        mid: Optional[float] = None
        spread: Optional[float] = None
        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            mid = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else None
        ts = _ns_str_to_ms(result.get("event_time")) if isinstance(result, dict) else None
        return BestBidAsk(
            symbol=inst,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread_bps=spread,
            ts_exchange_ms=ts if ts and ts > 0 else None,
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        _ = address
        inst = _normalize_instrument(symbol)
        kind, base, quote = _split_instrument(inst)
        resp = self._post(
            "positions",
            f"{self._trade_url}/full/v1/positions",
            {"sub_account_id": self._sub_account_id, "kind": kind, "base": base, "quote": quote},
            auth=True,
        )
        rows = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("instrument")) != inst:
                continue
            qty = _to_float(row.get("size"))
            entry = _to_float(row.get("entry_price"), 0.0)
            mark = _to_float(row.get("mark_price"), 0.0)
            notional = abs(_to_float(row.get("notional")))
            unreal = _to_float(row.get("unrealized_pnl"))
            return PositionSnapshot(
                symbol=inst,
                position_qty=qty,
                avg_entry_price=entry if entry > 0 else None,
                mark_price=mark if mark > 0 else None,
                position_notional=notional,
                unrealized_pnl_usd=unreal,
            )
        return PositionSnapshot(
            symbol=inst,
            position_qty=0.0,
            avg_entry_price=None,
            mark_price=None,
            position_notional=0.0,
            unrealized_pnl_usd=0.0,
        )

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        _ = address
        resp = self._post(
            "account_summary",
            f"{self._trade_url}/full/v1/account_summary",
            {"sub_account_id": self._sub_account_id},
            auth=True,
        )
        row = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(row, dict):
            return AccountSnapshot(equity_usd=None, cash_usd=None, withdrawable_usd=None)
        equity = _to_float(row.get("total_equity"), 0.0)
        withdrawable = _to_float(row.get("available_balance"), 0.0)
        return AccountSnapshot(
            equity_usd=equity if equity > 0 else None,
            cash_usd=withdrawable if withdrawable > 0 else 0.0,
            withdrawable_usd=withdrawable if withdrawable > 0 else 0.0,
        )

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]:
        _ = address
        kind, base, quote = _split_instrument(self._symbol)
        resp = self._post(
            "open_orders",
            f"{self._trade_url}/full/v1/open_orders",
            {"sub_account_id": self._sub_account_id, "kind": kind, "base": base, "quote": quote},
            auth=True,
        )
        out: list[HLOpenOrderRaw] = []
        rows = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            legs = row.get("legs")
            if not isinstance(legs, list) or not legs:
                continue
            leg0 = legs[0] if isinstance(legs[0], dict) else None
            if leg0 is None:
                continue
            inst = str(leg0.get("instrument") or "")
            if inst != self._symbol:
                continue
            oid = _grvt_oid_to_int(row.get("order_id"))
            if oid is None:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            create_time = metadata.get("create_time")
            side = Side.BUY if bool(leg0.get("is_buying_asset")) else Side.SELL
            out.append(
                HLOpenOrderRaw(
                    oid=oid,
                    coin=inst,
                    side=side,
                    limit_px=_to_float(leg0.get("limit_price")),
                    sz=_to_float(leg0.get("size")),
                    timestamp=_ns_str_to_ms(create_time),
                    cloid=str(metadata.get("client_order_id") or "") or None,
                )
            )
        return out

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[FillRaw]:
        _ = address
        inst = _normalize_instrument(symbol)
        kind, base, quote = _split_instrument(inst)
        resp = self._post(
            "fill_history",
            f"{self._trade_url}/full/v1/fill_history",
            {
                "sub_account_id": self._sub_account_id,
                "kind": kind,
                "base": base,
                "quote": quote,
                "limit": 200,
            },
            auth=True,
        )
        rows = resp.get("result") if isinstance(resp, dict) else None
        out: list[HLFillRaw] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("instrument")) != inst:
                continue
            trade_id = str(row.get("trade_id") or "")
            event_ms = _ns_str_to_ms(row.get("event_time"))
            fill_id = f"{trade_id}_{event_ms}"
            side = Side.BUY if bool(row.get("is_buyer")) else Side.SELL
            out.append(
                HLFillRaw(
                    fill_id=fill_id,
                    oid=_grvt_oid_to_int(row.get("order_id")),
                    coin=inst,
                    side=side,
                    px=_to_float(row.get("price")),
                    sz=_to_float(row.get("size")),
                    fee=_to_float(row.get("fee")),
                    time_ms=event_ms,
                    closed_pnl=_to_float(row.get("realized_pnl")),
                    raw=dict(row),
                )
            )
        return out

    def _instrument_row(self, instrument: str) -> dict[str, Any]:
        row = self._instruments_by_symbol.get(instrument)
        if row is None:
            self._symbol_spec = self._load_symbol_spec_strict(self._symbol)
            self.symbol_spec_fetched_ok = True
            # Fresh metadata — drop the signing cache so the next sign
            # rebuilds against the refetched instrument_hash / base_decimals.
            self._instrument_sign_cache.clear()
            row = self._instruments_by_symbol.get(instrument)
        if row is None:
            raise RuntimeError(f"GRVT instrument metadata not loaded for {instrument}")
        return row

    def _instrument_sign_params(self, instrument: str) -> _InstrumentSignCache:
        cache = self._instrument_sign_cache.get(instrument)
        if cache is not None:
            return cache
        meta = self._instrument_row(instrument)
        size_decimals = int(meta.get("base_decimals") or 9)
        try:
            instrument_hash = self._parse_required_int(
                meta,
                "instrument_hash",
                symbol=instrument,
                endpoint="cached_metadata",
            )
        except ValueError as e:
            raise RuntimeError(
                f"GRVT signing failed: invalid instrument_hash for {instrument!r}"
            ) from e
        cache = _InstrumentSignCache(
            instrument_hash=instrument_hash,
            size_decimals=size_decimals,
            size_multiplier_dec=Decimal(10) ** size_decimals,
        )
        self._instrument_sign_cache[instrument] = cache
        return cache

    def _sign_order_payload(
        self,
        *,
        instrument: str,
        is_market: bool,
        is_buy: bool,
        sz: float,
        limit_px: float,
        post_only: bool,
        reduce_only: bool,
        client_order_id: str,
        time_in_force: str,
    ) -> dict[str, Any]:
        if self._wallet is None:
            raise RuntimeError("GRVT wallet key missing")
        sp = self._instrument_sign_params(instrument)
        contract_size = int(
            (Decimal(str(sz)) * sp.size_multiplier_dec).to_integral_value(rounding=ROUND_DOWN)
        )
        limit_price = 0 if is_market else int(
            (Decimal(str(limit_px)) * _PRICE_MULTIPLIER_DEC).to_integral_value(rounding=ROUND_DOWN)
        )
        nonce = random.randint(0, (1 << 32) - 1)
        expiration_ns = time.time_ns() + _ORDER_EXPIRATION_NS
        expiration_str = str(expiration_ns)
        message = {
            "subAccountID": self._sub_account_id_int,
            "isMarket": is_market,
            "timeInForce": _TIF_MAP.get(time_in_force, 1),
            "postOnly": post_only,
            "reduceOnly": reduce_only,
            "legs": [
                {
                    "assetID": sp.instrument_hash,
                    "contractSize": contract_size,
                    "limitPrice": limit_price,
                    "isBuyingContract": is_buy,
                }
            ],
            "nonce": nonce,
            "expiration": expiration_ns,
        }
        signable = encode_typed_data(self._eip712_domain, _ORDER_EIP712_TYPES, message)
        signed = self._wallet.sign_message(signable)
        # Wire size / price strings must match the integer multiples the EIP-712
        # signature was computed over, otherwise GRVT recomputes and the order
        # is rejected for signature mismatch. Format from the signed integers.
        wire_size = _decimal_from_scaled_int(contract_size, sp.size_decimals)
        wire_limit_price = (
            "0" if is_market else _decimal_from_scaled_int(limit_price, 9)
        )
        return {
            "order": {
                "sub_account_id": self._sub_account_id_str,
                "is_market": is_market,
                "time_in_force": time_in_force,
                "post_only": post_only,
                "reduce_only": reduce_only,
                "legs": [
                    {
                        "instrument": instrument,
                        "size": wire_size,
                        "limit_price": wire_limit_price,
                        "is_buying_asset": is_buy,
                    }
                ],
                "signature": {
                    "signer": self._wallet_address_str,
                    "r": "0x" + signed.r.to_bytes(32, byteorder="big").hex(),
                    "s": "0x" + signed.s.to_bytes(32, byteorder="big").hex(),
                    "v": int(signed.v),
                    "expiration": expiration_str,
                    "nonce": nonce,
                },
                "metadata": {"client_order_id": str(client_order_id)},
            }
        }

    # --- order lifecycle ---------------------------------------------------
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
            raise RuntimeError("GRVT trading credentials missing")
        if reduce_only:
            raise NotImplementedError(
                "GRVT adapter: reduce_only=True is not yet "
                "propagated through this client; refusing to silently "
                "send a non-reduce-only order. Wire reduce_only "
                "through _place_order / payload before using "
                "soft-flatten on GRVT."
            )
        inst = _normalize_instrument(symbol)
        cloid = (client_order_id or "").strip() or self.make_client_order_id(
            inst, Side.BUY if is_buy else Side.SELL, "manual", limit_px, sz
        )
        tif = "GOOD_TILL_TIME"
        post_only = True
        reduce_only = False
        payload = self._sign_order_payload(
            instrument=inst,
            is_market=False,
            is_buy=is_buy,
            sz=sz,
            limit_px=limit_px,
            post_only=post_only,
            reduce_only=reduce_only,
            client_order_id=cloid,
            time_in_force=tif,
        )
        resp = self._post(
            "create_order",
            f"{self._trade_url}/full/v1/create_order",
            payload,
            auth=True,
        )
        if isinstance(resp, dict) and "code" in resp:
            # Structured rejection line with full business context (no secrets).
            # Lets operators diff payload shape vs exchange response without
            # needing to instrument deeper.
            self._log_place_rejection(
                instrument=inst,
                is_buy=is_buy,
                sz=sz,
                limit_px=limit_px,
                post_only=post_only,
                reduce_only=reduce_only,
                tif=tif,
                client_order_id=cloid,
                payload=payload,
                response=resp,
            )
        return resp

    def _log_place_rejection(
        self,
        *,
        instrument: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        post_only: bool,
        reduce_only: bool,
        tif: str,
        client_order_id: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        order = payload.get("order") if isinstance(payload, dict) else {}
        if not isinstance(order, dict):
            order = {}
        leg0 = {}
        legs = order.get("legs")
        if isinstance(legs, list) and legs and isinstance(legs[0], dict):
            leg0 = legs[0]
        sig = order.get("signature") if isinstance(order.get("signature"), dict) else {}
        signer_raw = str(sig.get("signer") or "")
        # Mask the signer to confirm routing without exposing the full wallet.
        signer_masked = (
            f"{signer_raw[:6]}…{signer_raw[-4:]}" if len(signer_raw) >= 10 else signer_raw
        )
        cloid_str = str(client_order_id or "")
        cloid_preview = cloid_str[:24] + ("…" if len(cloid_str) > 24 else "")
        logger.warning(
            "grvt_create_order_rejected instrument=%s side=%s "
            "size=%s limit_price=%s post_only=%s reduce_only=%s time_in_force=%s "
            "wire_size=%s wire_limit_price=%s is_buying_asset=%s "
            "sub_account_id=%s signer=%s cloid=%s cloid_len=%s "
            "resp_code=%s resp_message=%s",
            instrument,
            "BUY" if is_buy else "SELL",
            f"{sz:.12g}",
            f"{limit_px:.12g}",
            post_only,
            reduce_only,
            tif,
            leg0.get("size"),
            leg0.get("limit_price"),
            leg0.get("is_buying_asset"),
            order.get("sub_account_id"),
            signer_masked,
            cloid_preview,
            len(cloid_str),
            response.get("code"),
            str(response.get("message") or "")[:500],
        )

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        _ = symbol
        return self._post(
            "cancel_order",
            f"{self._trade_url}/full/v1/cancel_order",
            {
                "sub_account_id": str(self._sub_account_id),
                "order_id": _int_oid_to_grvt_order_id(oid),
            },
            auth=True,
        )

    # GRVT's ``/v1/cancel_order`` accepts a ``time_to_live_ms`` field whose
    # documented purpose is: "During this period, any order creation with a
    # matching client_order_id will be cancelled rather than added to the
    # matching engine — helps mitigate time-of-flight issues where
    # cancellations might arrive before the corresponding order." Capped at
    # 5000 ms by GRVT. We use it unconditionally on cloid-based cancels
    # because our race is exactly that: quoting can decide to reprice before
    # the ``create_order`` response has been fully indexed on the GRVT side,
    # in which case a cloid-cancel would otherwise return "not found" and
    # let the order materialise into a phantom a few ms later.
    _CLOID_CANCEL_TTL_MS_STR = "5000"

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        _ = symbol
        return self._post(
            "cancel_order_by_cloid",
            f"{self._trade_url}/full/v1/cancel_order",
            {
                "sub_account_id": str(self._sub_account_id),
                "client_order_id": str(client_order_id),
                "time_to_live_ms": self._CLOID_CANCEL_TTL_MS_STR,
            },
            auth=True,
        )

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        _ = address
        return self._post(
            "order_by_cloid",
            f"{self._trade_url}/full/v1/order",
            {
                "sub_account_id": str(self._sub_account_id),
                "client_order_id": str(client_order_id),
            },
            auth=True,
        )

    def cancel_all_orders_bulk_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Single-request cancel-all scoped to one instrument on GRVT.

        Posts to ``/full/v1/cancel_all_orders`` with
        ``{sub_account_id, kind, base, quote}`` filter arrays derived
        from the instrument (e.g. ``ETH_USDT_Perp`` → ``kind=["PERPETUAL"],
        base=["ETH"], quote=["USDT"]``). The matching engine evaluates
        the filter server-side and cancels every matching open order in
        one atomic operation — no client-side fetch-then-loop, so:

          * N+1 round trips collapse to 1 (~300 ms → ~100 ms).
          * Shutdown under any supervisor's SIGTERM → SIGKILL grace
            window is far more likely to complete.
          * An order indexed on GRVT *during* our request is still caught
            (the engine's filter sees it); our prior ``fetch_open_orders``
            snapshot would have missed it.

        Only touches the instrument passed in — other symbols on the
        same sub-account (e.g. a BTC bot running in parallel, or manual
        orders placed via the GRVT UI) are preserved by the base/quote
        filter. GRVT docs caveat: "may not match new orders in flight"
        — we accept that, the shutdown path retries the orchestrating
        caller will run twice.
        """
        inst = _normalize_instrument(symbol)
        kinds, bases, quotes = _split_instrument(inst)
        return self._post(
            "cancel_all_orders",
            f"{self._trade_url}/full/v1/cancel_all_orders",
            {
                "sub_account_id": str(self._sub_account_id),
                "kind": kinds,
                "base": bases,
                "quote": quotes,
            },
            auth=True,
        )

    def market_close(
        self, symbol: str, sz: Optional[float] = None
    ) -> dict[str, Any]:
        pos = self.fetch_position(self._sub_account_id, symbol)
        abs_qty = abs(pos.position_qty)
        if abs_qty <= 0:
            return {"result": {"status": "noop_already_flat"}}
        qty = float(sz) if sz is not None and sz > 0 else abs_qty
        buy = pos.position_qty < 0
        cloid = self.make_client_order_id(
            _normalize_instrument(symbol),
            Side.BUY if buy else Side.SELL,
            "market_close",
            pos.mark_price or 0.0,
            qty,
        )
        payload = self._sign_order_payload(
            instrument=_normalize_instrument(symbol),
            is_market=True,
            is_buy=buy,
            sz=qty,
            limit_px=0.0,
            post_only=False,
            reduce_only=True,
            client_order_id=cloid,
            time_in_force="IMMEDIATE_OR_CANCEL",
        )
        return self._post(
            "market_close",
            f"{self._trade_url}/full/v1/create_order",
            payload,
            auth=True,
        )

    def rest_runtime_counters(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for op, n in self._rest_call_counts.items():
            out[f"{op}_calls"] = int(n)
        for op, n in self._rest_retry_counts.items():
            out[f"{op}_retries"] = int(n)
        for op, n in self._rest_rate_limited_retry_counts.items():
            out[f"{op}_rate_limited_retries"] = int(n)
        return out

    # --- wire-format interpretation ---------------------------------------
    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_grvt_place_order_response(resp)

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        return interpret_grvt_cancel_response(resp)

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_grvt_order_status_response(resp)

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        return make_deterministic_grvt_client_order_id(
            symbol, side, quote_cycle_id, price, size
        )


__all__ = ["GrvtClient"]
