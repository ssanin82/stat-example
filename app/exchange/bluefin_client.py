"""Bluefin (Sui) adapter implementing :class:`PerpExchangeAdapter`.

Migrated to Bluefin Pro (``fireflyprotocol/pro-sdk``) API conventions:

  * Three subdomains: ``api.{env}.bluefin.io`` (general / exchange info /
    account), ``auth.api.{env}.bluefin.io`` (JWT minting),
    ``trade.api.{env}.bluefin.io`` (order placement / cancellation / open
    orders / leverage).
  * REST version bump: exchange info at ``/v1/exchange/info``; trade
    endpoints under ``/api/v1/trade/...``; account endpoints under
    ``/api/v1/account[...]``.
  * Numeric representation: all on-wire quantities use base 1e9
    (``*E9`` fields), not 1e18 as in the deprecated v2 API.
  * Order signing: Sui UserSignature (base64) over a pretty-printed JSON
    payload with a leading ``type`` tag (``"Bluefin Pro Order"``).
    See :mod:`app.exchange.bluefin_auth`.
  * Order cancellation is unsigned — just a PUT with bearer JWT.
  * The bot's internal ``oid: int`` stays an index into a session-local
    ``orderHash`` (hex) map; hashes come back from the server verbatim.

Operational model (unchanged from the first ship):

  * Path B (``BLUEFIN_ONE_CT_ENABLED=false``): configured private key is
    the parent wallet key; we sign orders with it directly.
  * Path A/C (session keys via ``upsertSubAccount``) is still deferred.

Fee telemetry (preserved): each fill carries a ``tradingFeeE9`` field
(matches the pro-sdk ``Trade`` schema). We decode that back to USDC
floats into :attr:`FillRaw.fee` so the post-session telemetry can verify
whether the SUI-PERP zero-fee promo is still in effect.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Optional, TypeVar

import httpx

from app.config import Settings
from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw
from app.exchange.bluefin_auth import (
    BluefinOrderSignPayload,
    BluefinSession,
    BluefinSessionError,
    default_order_expiration_ms,
    fresh_salt,
    load_or_init_session,
    session_expiry_healthy,
    sign_login_request,
    sign_order_payload,
    signable_login,
)
from app.exchange.bluefin_responses import (
    hash_to_oid,
    interpret_bluefin_cancel_response,
    interpret_bluefin_order_status_response,
    interpret_bluefin_place_response,
    make_deterministic_bluefin_client_order_id,
    oid_to_hash,
)
from app.exchange.exchange_retry import exchange_call_with_retry
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.exchange.symbol_spec import SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)
T = TypeVar("T")


# Pro-sdk default leverage when operator hasn't set one via a separate
# leverage-update call (we use cross-margin and rely on server defaults,
# but the signed payload still has to carry an explicit leverage value).
_DEFAULT_LEVERAGE = 3.0

_BASE_E9 = 9  # every numeric field on the pro-sdk wire is in 1e9 base

# Bluefin "type" strings in the OpenAPI common schema.
_ORDER_TYPE_LIMIT = "LIMIT"
_ORDER_TYPE_MARKET = "MARKET"
_TIF_GTT = "GTT"
_TIF_IOC = "IOC"


def _to_e9_str(v: float) -> str:
    """Scale a float to a 1e9-base integer decimal string (pro-sdk wire)."""
    scaled = (Decimal(str(v)) * (Decimal(10) ** _BASE_E9)).to_integral_value(
        rounding=ROUND_DOWN
    )
    return str(int(scaled))


def _from_e9(raw: Any) -> float:
    """Parse a pro-sdk 1e9-scaled integer string back to float."""
    if raw is None:
        return 0.0
    try:
        d = Decimal(str(raw)) / (Decimal(10) ** _BASE_E9)
        return float(d)
    except Exception:
        return 0.0


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_bluefin_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if "-PERP" in s:
        return s
    if "_" in s:
        return s.split("_")[0] + "-PERP"
    return s + "-PERP"


def _default_bluefin_urls(
    network: str,
) -> tuple[str, str, str, str, str]:
    """Return (exchange_url, auth_url, trade_url, public_ws_url, private_ws_url).

    Per pro-sdk ``env.rs``. The two WS URLs share the
    ``wss://stream.api.{env}.bluefin.io`` host and only differ in path
    (``/ws/market`` vs ``/ws/account``). We store the base (host) form here
    so the WS modules can append the path themselves.
    """
    n = (network or "").strip().lower()
    if n in ("sui_prod", "production", "prod", "mainnet"):
        env = "sui-prod"
    elif n in ("sui_staging", "staging", "testnet"):
        env = "sui-staging"
    else:
        env = "sui-dev"
    return (
        f"https://api.{env}.bluefin.io",
        f"https://auth.api.{env}.bluefin.io",
        f"https://trade.api.{env}.bluefin.io",
        f"wss://stream.api.{env}.bluefin.io",
        f"wss://stream.api.{env}.bluefin.io",
    )


def _audience_for_network(network: str) -> str:
    """Return the ``audience`` value to put in the signed LoginRequest.

    Per pro-sdk ``env.rs``::auth::{dev,staging,production}::AUDIENCE,
    all three environments use ``"api"`` as the audience.
    """
    _ = network
    return "api"


class BluefinClient:
    """Structural implementation of :class:`PerpExchangeAdapter` for Bluefin Pro."""

    symbol_spec_fetched_ok: bool = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._symbol = _normalize_bluefin_symbol(settings.symbol)
        self._network = (settings.bluefin_network or "SUI_PROD").strip()
        (
            exchange_default,
            auth_default,
            trade_default,
            _pub_default,
            _prv_default,
        ) = _default_bluefin_urls(self._network)
        self._exchange_url = (
            settings.bluefin_rest_url or exchange_default
        ).rstrip("/")
        self._auth_url = (settings.bluefin_auth_url or auth_default).rstrip("/")
        self._trade_url = (settings.bluefin_trade_url or trade_default).rstrip("/")

        self._http_timeout_s = 8.0
        self._http = httpx.Client(timeout=self._http_timeout_s)
        self._rest_call_counts: dict[str, int] = defaultdict(int)
        self._rest_retry_counts: dict[str, int] = defaultdict(int)
        self._rest_rate_limited_retry_counts: dict[str, int] = defaultdict(int)
        # Rate-limit defense (see config.py BLUEFIN_MIN_PLACE_INTERVAL_SECONDS
        # and BLUEFIN_REST_429_* for tuning rationale). ``_last_place_order_mono``
        # tracks monotonic timestamp of the last POST /orders attempt; the
        # _place_order path sleeps if the next attempt would breach the
        # configured min interval.
        self._last_place_order_mono: float = 0.0
        self._min_place_interval_s: float = float(
            getattr(settings, "bluefin_min_place_interval_seconds", 0.2)
        )
        self._rest_429_max_retries: int = int(
            getattr(settings, "bluefin_rest_429_max_retries", 3)
        )
        self._rest_429_base_backoff_s: float = float(
            getattr(settings, "bluefin_rest_429_base_backoff_seconds", 0.5)
        )
        # Clock-skew safety backoff applied to signedAtMillis on every place.
        # See BLUEFIN_SIGNED_AT_BACKOFF_MS in app/config.py for the full
        # rationale (Bluefin enforces signedAt <= server_now; local clocks
        # drifted forward silently 400 with a misleading error).
        self._signed_at_backoff_ms: int = int(
            getattr(settings, "bluefin_signed_at_backoff_ms", 2000)
        )
        # Whether to bypass cancel-by-hash (currently broken server-side)
        # and route every cancel through the documented cancel-all-for-
        # symbol behaviour instead. See BLUEFIN_CANCEL_BY_HASH_WORKAROUND
        # in app/config.py.
        self._cancel_by_hash_workaround_enabled: bool = bool(
            getattr(settings, "bluefin_cancel_by_hash_workaround", True)
        )
        # Counter surfaced via rest_runtime_counters for operator visibility.
        self._rest_429_retries_total: int = 0
        self._place_throttle_waits_total: int = 0
        self._place_throttle_wait_seconds_total: float = 0.0
        # Counter: how many cancel calls got fanned out into cancel-all
        # because the workaround is enabled. Surfaced via
        # rest_runtime_counters so the operator can see how often the
        # workaround is exercised and estimate the wasted
        # cancel+re-place fee cost.
        self._cancel_workaround_fanout_total: int = 0

        self._session: Optional[BluefinSession] = None
        self._session_error: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._auth_token_expiry_epoch: float = 0.0
        self._init_session_if_possible()

        # Market / contracts metadata populated at bootstrap.
        self._market_id: Optional[str] = None  # Market.marketAddress (Sui)
        self._ids_id: Optional[str] = None  # ContractsConfig.idsId — required on every order
        self._symbol_spec: SymbolSpec = self._load_symbol_spec_strict(self._symbol)
        self.symbol_spec_fetched_ok = True

        # ---- Cancel-confirmation gating (see app/exchange/base.py docstring
        # and app/config.py BLUEFIN_CANCEL_CONFIRM_*). Bluefin Pro cancels
        # return 202 Accepted asynchronously; the matching-engine effect is
        # signalled later by a private WS ``AccountOrderUpdate`` with
        # ``cancellationReason`` set. Track the order hashes we've asked to
        # cancel so the execution layer can skip replacement placement
        # while a cancel is in flight. ``_pending_cancels`` is keyed by the
        # lowercase 0x-prefixed order hash (same form we emit on the wire)
        # and carries enough context (symbol, side, submit time, timeout)
        # to answer ``has_pending_cancel`` and to log a confirmation latency
        # when the WS event lands.
        self._pending_cancel_lock = threading.Lock()
        self._pending_cancels: dict[str, dict[str, Any]] = {}
        # Side cache populated at place-time so a cancel-by-oid path can
        # resolve the side without a round-trip to openOrders. Keyed by
        # the lowercase non-prefixed hash string (matches ``_OID_TO_HASH``).
        self._hash_side_cache: dict[str, Side] = {}
        self._cancel_confirm_timeout_s = float(
            getattr(settings, "bluefin_cancel_confirm_timeout_seconds", 3.0) or 3.0
        )
        self._cancel_confirm_gate_enabled = bool(
            getattr(settings, "bluefin_cancel_confirm_gate_enabled", True)
        )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _init_session_if_possible(self) -> None:
        pk = (self._settings.bluefin_private_key or "").strip()
        if not pk:
            self._session_error = "BLUEFIN_PRIVATE_KEY missing"
            logger.info(
                "bluefin_session_skipped reason=no_private_key "
                "(read-only mode; place/cancel will fail)"
            )
            return
        try:
            self._session = load_or_init_session(
                private_key_raw=pk,
                account_address=self._settings.bluefin_account_address,
                one_ct_enabled=self._settings.bluefin_one_ct_enabled,
                one_ct_duration_hours=self._settings.bluefin_one_ct_duration_hours,
                symbol=self._symbol,
                network=self._network,
            )
        except BluefinSessionError as e:
            self._session_error = str(e)
            logger.error("bluefin_session_init_failed err=%s", e)

    def has_write_access(self) -> bool:
        return self._session is not None

    def _require_session(self) -> BluefinSession:
        if self._session is None:
            raise RuntimeError(
                f"Bluefin signing not available: {self._session_error or 'session not initialised'}"
            )
        ok, remaining = session_expiry_healthy(self._session)
        if not ok:
            logger.warning(
                "bluefin_session_expired_local_hint remaining_s=%s — "
                "adapter will still attempt signing; server will reject "
                "if the on-chain authorization is stale.",
                remaining,
            )
        return self._session

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

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

    def _request(
        self,
        op: str,
        method: str,
        url_base: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        def _single_attempt() -> dict[str, Any]:
            url = f"{url_base}{path}"
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if extra_headers:
                headers.update(extra_headers)
            if auth:
                # Always go through _ensure_auth_token so it can check the
                # cached token's expiry and re-mint if needed. Previously
                # this was gated by `if not token`, which only ran the
                # check when no token was cached at all — meaning a cached
                # but EXPIRED token string was happily reused for every
                # subsequent request and the server rejected all of them
                # with 401 ExpiredSignature. Observed in production
                # 2026-04-24: 828 create_order rejections over ~5 min
                # after the startup token went stale, with zero refresh
                # attempts in the log. _ensure_auth_token is cheap when
                # the token is still fresh (one timestamp compare) and
                # mints only when the 30-s safety margin to expiry is
                # crossed, so calling it every request is a negligible
                # cost compared to the HTTP round-trip.
                token = self._ensure_auth_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            try:
                resp = self._http.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )
            except httpx.TimeoutException as e:
                return {"code": 599, "message": f"timeout:{e}"}
            except httpx.RequestError as e:
                return {"code": 598, "message": f"transport:{e}"}
            body: dict[str, Any]
            try:
                parsed = resp.json() if resp.content else None
            except Exception:
                parsed = {"raw_text": resp.text[:500]}
            if isinstance(parsed, dict):
                body = dict(parsed)
            elif isinstance(parsed, list):
                body = {"data": parsed}
            elif parsed is None:
                body = {}
            else:
                body = {"data": parsed}
            if resp.status_code >= 400:
                body.setdefault("code", resp.status_code)
                body.setdefault("message", resp.text[:500])
                # Surface Retry-After for the 429 retry path below. Only
                # populated here; the retry logic parses it and decides.
                retry_after_raw = resp.headers.get("Retry-After")
                if retry_after_raw is not None:
                    body["_retry_after_raw"] = retry_after_raw
                logger.warning(
                    "bluefin_http_non_2xx op=%s status=%s url=%s code=%s message=%s",
                    op,
                    resp.status_code,
                    url,
                    body.get("code"),
                    str(body.get("message") or "")[:500],
                )
            else:
                # Preserve HTTP status on 2xx bodies for callers that want
                # to distinguish 200 (synchronous response) from 202
                # (accepted but still processing).
                body.setdefault("_http_status", resp.status_code)
            return body

        def _go() -> dict[str, Any]:
            # 429-aware retry loop. Non-429 errors pass through unchanged
            # (existing callers already handle ``code>=400`` in the dict).
            # Bluefin's CloudFront rate-limiter on POST /orders was the
            # motivator (see BLUEFIN_REST_429_* config for context).
            #
            # Also handles 401 "ExpiredSignature" — when the server
            # declares our cached token stale, invalidate locally and
            # retry once with a fresh mint. This catches the case where
            # our expiry tracking drifted from the server's (e.g. server
            # revoked early, or our 30-s safety margin wasn't enough).
            # Second 401 on the retry means something structural is
            # broken (bad key, revoked account, etc.) and we fall through
            # — don't loop forever on a genuinely unauthorised client.
            max_retries = max(0, int(self._rest_429_max_retries))
            base_backoff = max(0.05, float(self._rest_429_base_backoff_s))
            auth_retry_done = False
            for attempt in range(max_retries + 1):
                body = _single_attempt()
                code = body.get("code")
                if code == 401 and auth and not auth_retry_done:
                    msg = str(body.get("message") or "")
                    if "ExpiredSignature" in msg or "Invalid token" in msg:
                        logger.warning(
                            "bluefin_auth_401_retry op=%s message=%s "
                            "(invalidating cached token and re-minting)",
                            op,
                            msg[:200],
                        )
                        # Force a fresh mint on the next _single_attempt.
                        self._auth_token = None
                        self._auth_token_expiry_epoch = 0.0
                        auth_retry_done = True
                        self._note_retry(op, rate_limited=False)
                        continue
                if code != 429:
                    return body
                if attempt >= max_retries:
                    logger.warning(
                        "bluefin_429_retries_exhausted op=%s attempts=%d",
                        op,
                        attempt + 1,
                    )
                    return body
                # Retry-After per RFC 7231: either delta-seconds integer
                # or HTTP-date. CloudFront usually sends delta-seconds.
                retry_after_raw = body.get("_retry_after_raw")
                server_wait: Optional[float] = None
                if retry_after_raw is not None:
                    try:
                        server_wait = max(0.0, float(retry_after_raw))
                    except (TypeError, ValueError):
                        server_wait = None
                backoff_wait = base_backoff * (2 ** attempt)
                # Honour whichever is larger — server knows its own
                # cooldown best — but cap at 10s so the quote loop isn't
                # starved if the server sends an unreasonable value.
                wait_s = min(10.0, max(backoff_wait, server_wait or 0.0))
                self._rest_429_retries_total += 1
                self._note_retry(op, rate_limited=True)
                logger.warning(
                    "bluefin_429_retry op=%s attempt=%d/%d wait_s=%.3f "
                    "server_retry_after=%s",
                    op,
                    attempt + 1,
                    max_retries,
                    wait_s,
                    retry_after_raw,
                )
                time.sleep(wait_s)
            return body  # unreachable, but satisfies type-checker

        return self._retry(op, _go)

    # ------------------------------------------------------------------
    # Auth token minting (pro-sdk /auth/v2/token)
    # ------------------------------------------------------------------

    def _ensure_auth_token(self) -> Optional[str]:
        """POST a signed LoginRequest to /auth/v2/token, cache the JWT."""
        now = time.time()
        if self._auth_token and self._auth_token_expiry_epoch - now > 30.0:
            return self._auth_token
        if self._session is None:
            return None
        payload = signable_login(
            account_address=self._session.parent_address,
            audience=_audience_for_network(self._network),
            signed_at_millis=int(now * 1000),
        )
        try:
            signature = sign_login_request(
                payload, signing_key=self._session.signing_key
            )
        except Exception as e:
            logger.exception("bluefin_auth_sign_failed err=%s", e)
            return None
        resp = self._request(
            "auth_v2_token",
            "POST",
            self._auth_url,
            "/auth/v2/token",
            json_body=payload,
            extra_headers={"payloadSignature": signature},
            auth=False,
        )
        if resp.get("code"):
            logger.warning(
                "bluefin_auth_token_failed code=%s message=%s",
                resp.get("code"),
                str(resp.get("message") or "")[:300],
            )
            return None
        token = str(resp.get("accessToken") or "").strip()
        if not token:
            logger.warning(
                "bluefin_auth_no_token resp_keys=%s", sorted(resp.keys())
            )
            return None
        # Respect server-declared validity (fall back to 12h defensive cap).
        try:
            valid_for = int(resp.get("accessTokenValidForSeconds") or 0)
        except (TypeError, ValueError):
            valid_for = 0
        if valid_for <= 0:
            valid_for = 12 * 3600
        self._auth_token = token
        self._auth_token_expiry_epoch = now + float(valid_for)
        logger.info(
            "bluefin_auth_token_minted valid_for_s=%s parent=%s",
            valid_for,
            self._session.parent_address,
        )
        return token

    def current_auth_token(self) -> Optional[str]:
        """Return (and refresh if needed) the cached auth token.

        Exposed to the WS modules so they don't re-mint their own.
        """
        return self._auth_token or self._ensure_auth_token()

    def exchange_url(self) -> str:
        return self._exchange_url

    def auth_url(self) -> str:
        return self._auth_url

    def trade_url(self) -> str:
        return self._trade_url

    def ids_id(self) -> Optional[str]:
        return self._ids_id

    # ------------------------------------------------------------------
    # Symbol spec bootstrap (/v1/exchange/info)
    # ------------------------------------------------------------------

    def _load_symbol_spec_strict(self, symbol: str) -> SymbolSpec:
        resp = self._request(
            "exchange_info",
            "GET",
            self._exchange_url,
            "/v1/exchange/info",
            auth=False,
        )
        if resp.get("code"):
            raise RuntimeError(
                f"Bluefin /v1/exchange/info bootstrap failed: "
                f"code={resp.get('code')} message={str(resp.get('message') or '')[:200]}"
            )

        # Contracts config is the same for every market — pull idsId once.
        contracts = resp.get("contractsConfig")
        if isinstance(contracts, dict):
            ids_id = str(contracts.get("idsId") or "")
            if ids_id and not ids_id.startswith("0x"):
                ids_id = "0x" + ids_id
            self._ids_id = ids_id or None

        markets = resp.get("markets")
        if not isinstance(markets, list):
            raise RuntimeError(
                f"Bluefin /v1/exchange/info: unexpected body shape "
                f"(keys={sorted(resp.keys())[:12]})"
            )

        row: Optional[dict[str, Any]] = None
        for m in markets:
            if isinstance(m, dict) and str(m.get("symbol") or "").upper() == symbol:
                row = m
                break
        if row is None:
            raise RuntimeError(
                f"Bluefin /v1/exchange/info: symbol={symbol!r} not listed "
                f"(first markets: {[str((m or {}).get('symbol')) for m in markets[:5]]})"
            )

        tick_size = _from_e9(row.get("tickSizeE9"))
        step_size = _from_e9(row.get("stepSizeE9"))
        min_size = _from_e9(
            row.get("minOrderQuantityE9") or row.get("minTradeQuantityE9") or "0"
        )
        min_trade_px = _from_e9(row.get("minTradePriceE9") or row.get("minOrderPriceE9") or "0")
        market_addr = str(row.get("marketAddress") or "").strip()
        if market_addr and not market_addr.startswith("0x"):
            market_addr = "0x" + market_addr
        self._market_id = market_addr or None

        # STRICT: never guess reference data. Bluefin's /v1/exchange/info is
        # the single source of truth for tick/step/min_size/min_trade_price.
        # A missing or zero value indicates either a schema change at the
        # exchange or a wrong symbol — both require operator attention, not
        # a silent hardcoded default (which would mis-quote for anything
        # other than SUI-PERP and hide real bugs). Fail loudly at bootstrap.
        missing: list[str] = []
        if tick_size <= 0:
            missing.append(f"tickSizeE9={row.get('tickSizeE9')!r}")
        if step_size <= 0:
            missing.append(f"stepSizeE9={row.get('stepSizeE9')!r}")
        if min_size <= 0:
            missing.append(
                f"minOrderQuantityE9={row.get('minOrderQuantityE9')!r} / "
                f"minTradeQuantityE9={row.get('minTradeQuantityE9')!r}"
            )
        if min_trade_px <= 0:
            missing.append(
                f"minTradePriceE9={row.get('minTradePriceE9')!r} / "
                f"minOrderPriceE9={row.get('minOrderPriceE9')!r}"
            )
        if missing:
            raise RuntimeError(
                f"Bluefin /v1/exchange/info returned missing/invalid reference "
                f"data for symbol={symbol!r}: {'; '.join(missing)}. Refusing "
                f"to bootstrap — reference data is never guessed. Check the "
                f"symbol name and the /v1/exchange/info response shape."
            )

        # min_notional is fully derived from published fields — no fallback.
        min_notional = min_trade_px * min_size

        sz_decimals = max(0, min(12, int(round(-Decimal(str(step_size)).log10()))))

        spec = SymbolSpec(
            price_tick=tick_size,
            size_step=step_size,
            min_size=min_size,
            min_notional_usd=min_notional,
            sz_decimals=sz_decimals,
            source="grvt_meta",
        )
        logger.info(
            "bluefin_symbol_spec_bootstrap_success symbol=%s "
            "price_tick=%s size_step=%s min_size=%s min_notional=%s "
            "sz_decimals=%s market_id=%s ids_id=%s",
            symbol,
            spec.price_tick,
            spec.size_step,
            spec.min_size,
            spec.min_notional_usd,
            spec.sz_decimals,
            self._market_id,
            self._ids_id,
        )
        return spec

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    # ------------------------------------------------------------------
    # Market reads
    # ------------------------------------------------------------------

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        sym = _normalize_bluefin_symbol(symbol)
        # Pro-sdk exposes depth via /v1/exchange/depth but we prefer the
        # websocket Ticker stream for BBO; this REST fallback is used only
        # on WS outage. Tolerate an occasional empty body without failing
        # the entire quote loop — the caller upstream handles None fields.
        resp = self._request(
            "exchange_depth",
            "GET",
            self._exchange_url,
            "/v1/exchange/depth",
            params={"symbol": sym, "limit": 5},
            auth=False,
        )
        bids = resp.get("bidsE9") or resp.get("bids")
        asks = resp.get("asksE9") or resp.get("asks")
        best_bid: Optional[float] = None
        best_ask: Optional[float] = None
        try:
            if isinstance(bids, list) and bids and isinstance(bids[0], (list, tuple)):
                best_bid = _from_e9(bids[0][0])
            if isinstance(asks, list) and asks and isinstance(asks[0], (list, tuple)):
                best_ask = _from_e9(asks[0][0])
        except (IndexError, TypeError, ValueError):
            pass
        if best_bid is not None and best_bid <= 0:
            best_bid = None
        if best_ask is not None and best_ask <= 0:
            best_ask = None
        mid: Optional[float] = None
        spread_bps: Optional[float] = None
        if best_bid and best_ask and best_ask > best_bid:
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else None
        ts_ms_raw = resp.get("updatedAtMillis") or resp.get("lastUpdateTime")
        try:
            ts_ms = int(ts_ms_raw) if ts_ms_raw else None
        except (TypeError, ValueError):
            ts_ms = None
        return BestBidAsk(
            symbol=sym,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=ts_ms,
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        _ = address
        sym = _normalize_bluefin_symbol(symbol)
        # /api/v1/account returns positions[] as part of the Account object.
        resp = self._request(
            "account",
            "GET",
            self._exchange_url,
            "/api/v1/account",
            params={"accountAddress": self._settings.bluefin_account_address},
            auth=True,
        )
        # BUG-006: do NOT silently zero on error responses. A
        # synthetic flat snapshot would let the caller's safety
        # checks (max position cap, drawdown) read 0 when the venue
        # holds real inventory we just couldn't read. Raise so
        # ``refresh_account_only`` can preserve last-known state and
        # mark the refresh as failed.
        if resp.get("code"):
            raise RuntimeError(
                f"bluefin /api/v1/account returned error code "
                f"{resp.get('code')!r}; refusing to fabricate flat "
                f"PositionSnapshot for symbol={sym}"
            )
        positions = resp.get("positions")
        row: Optional[dict[str, Any]] = None
        if isinstance(positions, list):
            for r in positions:
                if isinstance(r, dict) and str(r.get("symbol") or "").upper() == sym:
                    row = r
                    break
        if row is None:
            return PositionSnapshot(
                symbol=sym,
                position_qty=0.0,
                avg_entry_price=None,
                mark_price=None,
                position_notional=0.0,
                unrealized_pnl_usd=0.0,
            )
        qty = _from_e9(row.get("sizeE9") or "0")
        side = str(row.get("side") or "").upper()
        if side == "SHORT" and qty > 0:
            qty = -qty
        entry = _from_e9(row.get("avgEntryPriceE9") or "0")
        mark = _from_e9(row.get("markPriceE9") or "0")
        unreal = _from_e9(row.get("unrealizedPnlE9") or "0")
        notional = abs(qty) * (mark or entry or 0.0)
        return PositionSnapshot(
            symbol=sym,
            position_qty=qty,
            avg_entry_price=entry if entry > 0 else None,
            mark_price=mark if mark > 0 else None,
            position_notional=notional,
            unrealized_pnl_usd=unreal,
        )

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        _ = address
        resp = self._request(
            "account",
            "GET",
            self._exchange_url,
            "/api/v1/account",
            params={"accountAddress": self._settings.bluefin_account_address},
            auth=True,
        )
        # BUG-006: raise on error rather than returning an
        # all-None snapshot that the caller's health check sees as
        # "successful". Lets the caller preserve last-known-good
        # state and trip the freshness gate.
        if not isinstance(resp, dict):
            raise RuntimeError(
                "bluefin /api/v1/account returned non-dict response; "
                "refusing to fabricate empty AccountSnapshot"
            )
        if resp.get("code"):
            raise RuntimeError(
                f"bluefin /api/v1/account returned error code "
                f"{resp.get('code')!r}; refusing to fabricate empty "
                "AccountSnapshot"
            )
        equity = _from_e9(
            resp.get("totalAccountValueE9")
            or resp.get("crossAccountValueE9")
            or resp.get("crossEffectiveBalanceE9")
            or "0"
        )
        free = _from_e9(resp.get("marginAvailableE9") or "0")
        return AccountSnapshot(
            equity_usd=equity if equity > 0 else None,
            cash_usd=free if free >= 0 else None,
            withdrawable_usd=free if free >= 0 else None,
        )

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]:
        _ = address
        resp = self._request(
            "open_orders",
            "GET",
            self._trade_url,
            "/api/v1/trade/openOrders",
            params={"symbol": self._symbol},
            auth=True,
        )
        rows = resp.get("data") if resp.get("data") is not None else resp
        out: list[HLOpenOrderRaw] = []
        if isinstance(rows, dict):
            # Some pro-sdk proxies wrap the array under "data"; our _request
            # already auto-wraps top-level arrays into {"data":[...]}.
            rows = rows.get("data")
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or "").upper()
            if sym != self._symbol:
                continue
            order_hash = str(row.get("orderHash") or "")
            if not order_hash:
                continue
            oid = hash_to_oid(order_hash)
            side_str = str(row.get("side") or "").upper()
            # OrderSide values in pro-sdk: LONG / SHORT. Map to BUY / SELL.
            side = Side.BUY if side_str == "LONG" else Side.SELL
            ts_raw = (
                row.get("updatedAtMillis")
                or row.get("orderTimeAtMillis")
                or row.get("createdAtMillis")
                or 0
            )
            try:
                ts = int(ts_raw)
            except (TypeError, ValueError):
                ts = 0
            cloid_raw = row.get("clientOrderId")
            cloid = str(cloid_raw) if cloid_raw not in (None, "") else None
            out.append(
                HLOpenOrderRaw(
                    oid=oid,
                    coin=sym,
                    side=side,
                    limit_px=_from_e9(row.get("priceE9")),
                    sz=_from_e9(row.get("quantityE9") or "0"),
                    timestamp=ts,
                    cloid=cloid,
                )
            )
        return out

    def fetch_account_volume_usd(
        self,
        symbol: Optional[str],
        since_ms: int,
        until_ms: int,
        *,
        page_size: int = 1000,
        max_pages: int = 50,
    ) -> dict[str, Any]:
        """Sum trade notional (USD) and fees over a time window.

        Mirrors ``scripts/bluefin_volume.py``'s aggregator but reuses the
        adapter's existing auth / httpx infra so we don't re-mint a token
        on every call. Returns a dict with::

            {
                "volume_usd": float,    # sum of priceE9 * quantityE9 / 1e18
                "fees_usd": float,      # sum of |tradingFeeE9| / 1e9
                "trade_count": int,
                "since_ms": since_ms,
                "until_ms": until_ms,
                "symbol": symbol,
                "pages_fetched": int,
                "pagination_truncated": bool,  # True iff we hit max_pages
            }

        On a busy 30-day SUI-PERP window this is ~1-2 pages and < 1 s.
        Heavier windows pay more; the operator-facing caller wraps this
        in a TTL cache + background refresher so the synchronous
        ``/status`` path never blocks on the REST round trip.
        """
        if until_ms <= since_ms:
            return {
                "volume_usd": 0.0,
                "fees_usd": 0.0,
                "trade_count": 0,
                "since_ms": since_ms,
                "until_ms": until_ms,
                "symbol": symbol,
                "pages_fetched": 0,
                "pagination_truncated": False,
            }
        sym = _normalize_bluefin_symbol(symbol) if symbol else None
        volume = 0.0
        fees = 0.0
        n = 0
        pages = 0
        truncated = False
        for page in range(1, max_pages + 1):
            params: dict[str, Any] = {
                "startTimeAtMillis": str(since_ms),
                "endTimeAtMillis": str(until_ms),
                "limit": int(page_size),
                "page": int(page),
            }
            if sym:
                params["symbol"] = sym
            resp = self._request(
                "account_trades_volume",
                "GET",
                self._exchange_url,
                "/api/v1/account/trades",
                params=params,
                auth=True,
            )
            rows = resp.get("data") if resp.get("data") is not None else resp
            if isinstance(rows, dict):
                rows = rows.get("data") or rows.get("trades")
            if not isinstance(rows, list):
                break
            pages += 1
            if not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # Optional symbol filter at the row level — be defensive in
                # case the server ignores the param for some reason.
                if sym and str(row.get("symbol") or "").upper() != sym:
                    continue
                px = _from_e9(row.get("priceE9") or "0")
                qty = _from_e9(row.get("quantityE9") or "0")
                fee = abs(_from_e9(row.get("tradingFeeE9") or "0"))
                volume += float(px) * float(qty)
                fees += float(fee)
                n += 1
            if len(rows) < int(page_size):
                break
        else:
            # for-else: hit max_pages without breaking → pagination
            # potentially truncated. Surfaced so callers can warn the
            # operator if the window is bigger than the cap allows.
            truncated = True
        return {
            "volume_usd": float(volume),
            "fees_usd": float(fees),
            "trade_count": int(n),
            "since_ms": int(since_ms),
            "until_ms": int(until_ms),
            "symbol": symbol,
            "pages_fetched": pages,
            "pagination_truncated": bool(truncated),
        }

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[FillRaw]:
        _ = address
        sym = _normalize_bluefin_symbol(symbol)
        resp = self._request(
            "account_trades",
            "GET",
            self._exchange_url,
            "/api/v1/account/trades",
            params={"symbol": sym, "limit": 200},
            auth=True,
        )
        rows = resp.get("data") if resp.get("data") is not None else resp
        if isinstance(rows, dict):
            rows = rows.get("data")
        out: list[HLFillRaw] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").upper() != sym:
                continue
            trade_id = str(row.get("id") or "")
            ts_raw = row.get("executedAtMillis") or 0
            try:
                time_ms = int(ts_raw)
            except (TypeError, ValueError):
                time_ms = 0
            fill_id = f"{trade_id}_{time_ms}"
            side_str = str(row.get("side") or "").upper()
            # TradeSide values in pro-sdk: LONG / SHORT.
            side = Side.BUY if side_str == "LONG" else Side.SELL
            order_hash = str(row.get("orderHash") or "")
            oid_val = hash_to_oid(order_hash) if order_hash else None
            # Fee: tradingFeeE9 carries the amount in USDC (1e9 base).
            # Server returns it as a *negative* e9 string when the account
            # paid a fee and positive when a rebate — we fold to abs()
            # because the bot's FillRaw.fee field is unsigned-positive.
            fee_raw = row.get("tradingFeeE9") or "0"
            fee_abs = abs(_from_e9(fee_raw))
            out.append(
                HLFillRaw(
                    fill_id=fill_id,
                    oid=oid_val,
                    coin=sym,
                    side=side,
                    px=_from_e9(row.get("priceE9")),
                    sz=_from_e9(row.get("quantityE9") or "0"),
                    fee=fee_abs,
                    time_ms=time_ms,
                    closed_pnl=_from_e9(row.get("realizedPnlE9") or "0"),
                    raw=dict(row),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def _throttle_place_order(self) -> None:
        """Enforce a minimum interval between POST /orders calls.

        Preemptive defense against the CloudFront rate-limiter in front of
        ``POST /api/v1/trade/orders`` (see ``BLUEFIN_MIN_PLACE_INTERVAL_SECONDS``
        in ``app.config`` for rationale). Simple leaky bucket with
        capacity 1: if the previous place call happened less than
        ``_min_place_interval_s`` ago, sleep the remainder. Threadsafe
        only insofar as the bot's hot path is single-threaded for order
        placement; we're not guarding concurrent callers here.
        """
        if self._min_place_interval_s <= 0:
            return
        now_m = time.monotonic()
        elapsed = now_m - self._last_place_order_mono
        wait = self._min_place_interval_s - elapsed
        if wait > 0:
            self._place_throttle_waits_total += 1
            self._place_throttle_wait_seconds_total += wait
            # INFO-level log: small, bounded, not every tick once steady
            # state is reached. If you're seeing many of these per second
            # the quote loop is placing too aggressively — consider
            # raising REPRICE_THRESHOLD_BPS or lowering QUOTE_LOOP_SECONDS.
            logger.info(
                "bluefin_place_throttle_wait wait_s=%.3f min_interval_s=%.3f",
                wait,
                self._min_place_interval_s,
            )
            time.sleep(wait)
        # Stamp *after* the sleep so the next call measures against when
        # we actually released, not when we started waiting.
        self._last_place_order_mono = time.monotonic()

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
        self._throttle_place_order()
        session = self._require_session()
        if not self._ids_id:
            raise RuntimeError(
                "Bluefin idsId not loaded — /v1/exchange/info bootstrap must populate it first"
            )
        sym = _normalize_bluefin_symbol(symbol)
        cloid = (
            (client_order_id or "").strip()
            or make_deterministic_bluefin_client_order_id(
                sym, Side.BUY if is_buy else Side.SELL, "manual", limit_px, sz
            )
        )

        salt = fresh_salt()
        # Backdate signedAtMillis by a configurable safety offset so it's
        # safely in Bluefin's server past even under forward clock drift.
        # The server enforces signedAtMillis <= server_now; local clocks
        # running ~1 s ahead (common on un-NTP'd Windows hosts) silently
        # fail with a misleading "must be no earlier than 1 minute in the
        # past" error. The backoff only shifts signedAt; expiresAtMillis
        # is computed independently and stays in the far future.
        now_ms = int(time.time() * 1000) - self._signed_at_backoff_ms
        expiration_ms = default_order_expiration_ms(
            is_market=(order_type == _ORDER_TYPE_MARKET)
        )

        # Build the signed-fields struct. Price/quantity/leverage are in
        # 1e9 base as decimal strings.
        price_e9 = (
            _to_e9_str(limit_px) if order_type != _ORDER_TYPE_MARKET else "0"
        )
        qty_e9 = _to_e9_str(sz)
        lev_e9 = _to_e9_str(_DEFAULT_LEVERAGE)

        sign_payload = BluefinOrderSignPayload(
            symbol=sym,
            account_address=session.parent_address,
            ids_id=self._ids_id,
            price_e9=price_e9,
            quantity_e9=qty_e9,
            leverage_e9=lev_e9,
            side="LONG" if is_buy else "SHORT",
            is_isolated=False,  # cross-margin
            expires_at_millis=expiration_ms,
            salt=salt,
            signed_at_millis=now_ms,
        )
        signature = sign_order_payload(
            sign_payload, signing_key=session.signing_key
        )

        # Assemble the full CreateOrderRequest body per trade-api.yaml.
        body: dict[str, Any] = {
            "signedFields": {
                "symbol": sym,
                "accountAddress": session.parent_address,
                "priceE9": price_e9,
                "quantityE9": qty_e9,
                "side": "LONG" if is_buy else "SHORT",
                "leverageE9": lev_e9,
                "isIsolated": False,
                "salt": salt,
                "idsId": self._ids_id,
                "expiresAtMillis": expiration_ms,
                "signedAtMillis": now_ms,
            },
            "signature": signature,
            "clientOrderId": cloid,
            "type": order_type,
            "reduceOnly": bool(reduce_only),
            "postOnly": bool(post_only),
            "timeInForce": _TIF_IOC if ioc else _TIF_GTT,
            "selfTradePreventionType": "MAKER",
        }

        resp = self._request(
            "create_order" if order_type != _ORDER_TYPE_MARKET else "market_close",
            "POST",
            self._trade_url,
            "/api/v1/trade/orders",
            json_body=body,
            auth=True,
        )
        # Cache hash→side so the cancel-by-oid path can register a pending
        # cancel without needing a separate openOrders fetch. Skipped on
        # MARKET orders (they don't rest) and on responses without a hash.
        if isinstance(resp, dict) and order_type != _ORDER_TYPE_MARKET:
            order_hash = str(resp.get("orderHash") or "").lower().strip()
            if order_hash.startswith("0x"):
                order_hash = order_hash[2:]
            if order_hash:
                self._hash_side_cache[order_hash] = Side.BUY if is_buy else Side.SELL
        return resp

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
            raise RuntimeError("Bluefin trading credentials missing")
        if reduce_only:
            raise NotImplementedError(
                "Bluefin adapter: reduce_only=True is not yet "
                "propagated through this client; refusing to silently "
                "send a non-reduce-only order. Wire reduce_only "
                "through _place_order before using soft-flatten on this venue."
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
        """Cancel by internal int oid (resolved to the hash we cached on place)."""
        order_hash = oid_to_hash(int(oid))
        if not order_hash:
            return {
                "code": 404,
                "message": f"no_known_hash_for_oid={oid}",
            }
        side_hint = self._hash_side_cache.get(order_hash)
        return self._cancel_by_hashes(symbol, [order_hash], side_hint=side_hint)

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        """Cancel by client order id via open-orders lookup.

        Pro-sdk's /api/v1/trade/orders/cancel takes order hashes (not
        clientOrderIds), so we resolve by listing open orders and
        matching on clientOrderId.
        """
        sym = _normalize_bluefin_symbol(symbol)
        open_orders = self.fetch_open_orders_raw("")
        cloid_str = str(client_order_id or "").strip()
        matched_hashes: list[str] = []
        matched_side: Optional[Side] = None
        for oo in open_orders:
            if oo.cloid is None:
                continue
            if oo.cloid == cloid_str:
                h = oid_to_hash(oo.oid)
                if h:
                    matched_hashes.append(h)
                    # Cache the side now so subsequent cancel-by-oid calls
                    # on the same order (e.g. retries) don't need openOrders.
                    self._hash_side_cache[h] = oo.side
                    if matched_side is None:
                        matched_side = oo.side
        if not matched_hashes:
            return {"code": 404, "message": f"no_open_order_for_cloid={cloid_str}"}
        return self._cancel_by_hashes(sym, matched_hashes, side_hint=matched_side)

    def _cancel_by_hashes(
        self,
        symbol: str,
        hashes: list[str],
        *,
        side_hint: Optional[Side] = None,
    ) -> dict[str, Any]:
        # pro-sdk cancel is unsigned: just PUT {symbol[, orderHashes]} with
        # a bearer JWT. Normalise hashes to the full 0x-prefixed form the
        # server echoes back in the response.
        sym = _normalize_bluefin_symbol(symbol)
        hashes_norm: list[str] = []
        for h in hashes:
            h2 = (h or "").lower().strip()
            if h2.startswith("0x"):
                h2 = h2[2:]
            hashes_norm.append("0x" + h2)

        # Bluefin bug workaround (see BLUEFIN_CANCEL_BY_HASH_WORKAROUND in
        # app/config.py): when the caller asked to cancel specific hashes,
        # the server's selective-cancel path returns HTTP 202 but silently
        # drops the request. The "cancel all for symbol" path (same
        # endpoint, no orderHashes key) works correctly and synchronously.
        # Until Bluefin fixes the selective path, we fan every call out
        # to cancel-all.
        #
        # BUG-000 (FIXED): the side effect — every other open hash for
        # the symbol gets cancelled too — used to leak ``OrderCancellationUpdate``
        # events that arrived for hashes the bot never registered as
        # pending-cancel. ``on_cancel_confirmed`` was a no-op for those
        # hashes; the OrderManager's WorkingOrder lifecycle never got the
        # cancellation signal; the side stayed "pending acknowledgement";
        # the quote loop refused to place a replacement; eventually the
        # deadlock watchdog fired (exit 42 → supervisor restart). Repro
        # cadence pre-fix: ~once every few hours.
        #
        # Fix: pre-register EVERY currently-open hash for the symbol when
        # we fan out to cancel-all. The extra GET /openOrders adds a few
        # tens of ms per fanout but eliminates the orphan-event class
        # entirely. ``_register_pending_cancels`` is idempotent
        # (registering the same hash twice is a no-op past the first
        # call) so the caller's request still wins for downstream
        # confirmation matching.
        will_fanout = bool(self._cancel_by_hash_workaround_enabled) and bool(hashes_norm)
        register_hashes = list(hashes_norm)
        register_sides: dict[str, Optional[Side]] = {h: side_hint for h in hashes_norm}
        if will_fanout:
            try:
                open_orders = self.fetch_open_orders_raw("")
            except Exception:  # noqa: BLE001
                # Defensive: a transient REST failure here shouldn't break
                # the cancel itself. The fanout still happens; we just
                # have less info to register.
                logger.exception(
                    "bluefin_cancel_workaround_open_orders_fetch_failed; "
                    "fanout proceeds without pre-registering side-effect hashes"
                )
                open_orders = []
            for oo in open_orders:
                h = oid_to_hash(oo.oid)
                if not h:
                    continue
                # Normalise to the same 0x-prefixed lowercase form as
                # ``hashes_norm`` so the union dedup works cleanly.
                h_norm = "0x" + h.lower().lstrip("0x")
                if h_norm in register_sides:
                    continue
                register_hashes.append(h_norm)
                register_sides[h_norm] = oo.side
                # Cache the side so the cancel-confirm path can match it
                # without another openOrders fetch.
                self._hash_side_cache[h_norm[2:]] = oo.side

            body: dict[str, Any] = {"symbol": sym}  # no orderHashes -> cancel-all
            self._cancel_workaround_fanout_total += 1
            logger.info(
                "bluefin_cancel_workaround_fanout requested_hashes=%d "
                "side_effect_hashes=%d symbol=%s",
                len(hashes_norm),
                len(register_hashes) - len(hashes_norm),
                sym,
            )
        else:
            body = {"symbol": sym, "orderHashes": hashes_norm}
        resp = self._request(
            "cancel_order",
            "PUT",
            self._trade_url,
            "/api/v1/trade/orders/cancel",
            json_body=body,
            auth=True,
        )
        # Stamp the hash list so the interpreter can tell which orders the
        # call was about even if the server returns a 202 with empty body.
        # Note: the stamp is the CALLER's hashes (preserves downstream
        # confirmation matching for the original request); the side-effect
        # hashes registered above are tracked in _pending_cancels but are
        # not re-stamped here.
        if isinstance(resp, dict) and not resp.get("code"):
            resp.setdefault("orderHashes", hashes_norm)
            # Register pending-cancel state for the replacement-gate. Only
            # on 2xx (no ``code`` key): if the server rejected the cancel
            # we don't want to block the same side forever waiting for a
            # confirmation that will never arrive.
            if self._cancel_confirm_gate_enabled and register_hashes:
                # BUG-000: ``register_hashes`` includes the side-effect
                # hashes when the workaround fanout fired; one call per
                # hash so each carries its own side hint.
                for h in register_hashes:
                    self._register_pending_cancels(
                        sym, [h], side_hint=register_sides.get(h)
                    )
        return resp

    # ------------------------------------------------------------------
    # Cancel-confirmation gating (Bluefin-only; see base.py docstring)
    # ------------------------------------------------------------------

    def _register_pending_cancels(
        self,
        symbol: str,
        order_hashes: list[str],
        *,
        side_hint: Optional[Side],
    ) -> None:
        """Record in-flight cancel(s) so ``has_pending_cancel`` can report them.

        Uses ``self._hash_side_cache`` when the caller didn't supply a
        ``side_hint``. If we still can't determine the side (cache miss
        and no hint), we register with ``side=None`` so the timeout still
        cleans it up but ``has_pending_cancel`` won't match either side
        — effectively a no-op for the gate. Logging callout makes this
        observable if it happens in production.
        """
        now = time.time()
        timeout = now + self._cancel_confirm_timeout_s
        with self._pending_cancel_lock:
            for h in order_hashes:
                key = (h or "").lower().strip()
                if not key:
                    continue
                if not key.startswith("0x"):
                    key = "0x" + key
                bare = key[2:]
                side = side_hint or self._hash_side_cache.get(bare)
                self._pending_cancels[key] = {
                    "symbol": symbol,
                    "side": side,
                    "submitted_at_epoch_s": now,
                    "timeout_epoch_s": timeout,
                }
                logger.info(
                    "bluefin_cancel_pending_registered hash=%s side=%s symbol=%s timeout_in_s=%s",
                    key,
                    side.value if side is not None else None,
                    symbol,
                    f"{self._cancel_confirm_timeout_s:.3f}",
                )

    def on_cancel_confirmed(self, order_hash: str) -> None:
        """Clear the pending entry for ``order_hash`` (called by the WS layer).

        Idempotent: extra confirmations or confirmations for orders we
        never tracked are silently ignored. Logs confirmation latency in
        ms for operator visibility.

        BUG-000: when a cancellation event arrives for a hash that was
        NOT in ``_pending_cancels``, we log it as
        ``bluefin_cancel_orphan_event`` at WARNING. With the BUG-000 fix
        in place (pre-register every open hash on workaround fanout),
        this should be 0 in healthy operation. A non-zero count means
        either (a) a third-party process cancelled an order we placed
        — possible but rare on Bluefin — or (b) a regression in the
        pre-register path that needs investigation. The defensive log
        was previously silent (events dropped without trace), which is
        what made the BUG-000 deadlock so hard to diagnose for weeks.
        """
        key = (order_hash or "").lower().strip()
        if not key:
            return
        if not key.startswith("0x"):
            key = "0x" + key
        entry: Optional[dict[str, Any]] = None
        with self._pending_cancel_lock:
            entry = self._pending_cancels.pop(key, None)
        if entry is None:
            logger.warning(
                "bluefin_cancel_orphan_event hash=%s symbol=%s "
                "(no matching _pending_cancels entry; should be 0 in "
                "healthy operation post-BUG-000 fix)",
                key,
                self._symbol,
            )
            return
        latency_ms = max(
            0.0, (time.time() - float(entry.get("submitted_at_epoch_s") or 0.0)) * 1000.0
        )
        side = entry.get("side")
        logger.info(
            "bluefin_cancel_confirmed hash=%s side=%s symbol=%s latency_ms=%s",
            key,
            side.value if isinstance(side, Side) else None,
            entry.get("symbol"),
            f"{latency_ms:.1f}",
        )

    def expire_stale_pending_cancels(self, now: Optional[float] = None) -> int:
        """Drop pending entries whose timeout has elapsed; return count dropped.

        Safety fallback for lost or delayed WS confirmations: without this
        a dropped ``OrderCancellationUpdate`` would block quoting on that
        side indefinitely. Called from the execution layer each quote
        tick (see ``app/execution.py::_orchestrate``) and directly from
        tests to simulate timeout.
        """
        now_s = float(now if now is not None else time.time())
        dropped: list[tuple[str, dict[str, Any]]] = []
        with self._pending_cancel_lock:
            expired_keys = [
                k for k, v in self._pending_cancels.items()
                if now_s > float(v.get("timeout_epoch_s") or 0.0)
            ]
            for k in expired_keys:
                dropped.append((k, self._pending_cancels.pop(k)))
        for key, entry in dropped:
            elapsed = now_s - float(entry.get("submitted_at_epoch_s") or now_s)
            side = entry.get("side")
            logger.warning(
                "bluefin_cancel_timeout_elapsed hash=%s symbol=%s side=%s elapsed_s=%s",
                key,
                entry.get("symbol"),
                side.value if isinstance(side, Side) else None,
                f"{elapsed:.3f}",
            )
        return len(dropped)

    def has_pending_cancel(self, symbol: str, side: Side) -> bool:
        """Return True iff a cancel on ``(symbol, side)`` is in flight.

        Consulted by ``app/execution.py::_orchestrate`` before enqueuing a
        same-side replacement placement. When
        ``BLUEFIN_CANCEL_CONFIRM_GATE_ENABLED=false`` the gate is
        disabled and this always returns False (operator escape hatch).
        Symbol is normalised to the Bluefin canonical form so callers
        that pass ``"SUI_PERP"`` or ``"SUI-PERP"`` both work.
        """
        if not self._cancel_confirm_gate_enabled:
            return False
        # First sweep out expired entries so the answer reflects reality
        # (rather than leaving "stuck" state that a previous thread
        # hasn't cleaned up yet).
        self.expire_stale_pending_cancels()
        sym_norm = _normalize_bluefin_symbol(symbol)
        with self._pending_cancel_lock:
            for entry in self._pending_cancels.values():
                if entry.get("side") != side:
                    continue
                entry_sym = entry.get("symbol") or ""
                if _normalize_bluefin_symbol(str(entry_sym)) == sym_norm:
                    return True
        return False

    def pending_cancel_snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of current pending-cancel entries (for tests/debug)."""
        with self._pending_cancel_lock:
            out: list[dict[str, Any]] = []
            for hash_key, entry in self._pending_cancels.items():
                row = dict(entry)
                row["order_hash"] = hash_key
                side = row.get("side")
                row["side_value"] = side.value if isinstance(side, Side) else None
                out.append(row)
            return out

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        _ = address
        cloid_str = str(client_order_id or "").strip()
        # Pro-sdk /api/v1/trade/openOrders only filters by symbol; we
        # fetch + filter client-side on clientOrderId.
        resp = self._request(
            "order_by_cloid",
            "GET",
            self._trade_url,
            "/api/v1/trade/openOrders",
            params={"symbol": self._symbol},
            auth=True,
        )
        rows = resp.get("data") if resp.get("data") is not None else resp
        if isinstance(rows, dict):
            rows = rows.get("data")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("clientOrderId") or "") == cloid_str:
                    return {"data": row}
        return {"code": 404, "message": f"order_not_found_cloid={cloid_str}"}

    def market_close(
        self, symbol: str, sz: Optional[float] = None
    ) -> dict[str, Any]:
        """Force-close ``symbol`` at any price up to a slippage cap.

        BUG-015 (the v1.0.20 fix that ACTUALLY works) — uses an IOC LIMIT
        order rather than a true MARKET order. Why:

        * Bluefin's MARKET endpoint signs ``priceE9 = "0"`` regardless of
          what the caller passes (see ``_place_order:1129-1131``). The
          v1.0.19 BUG-014 Layer-1 fix tried to set a permissive limit
          on a MARKET order, but the price was silently zeroed at
          signature time. The MARKET order then failed to fill in
          production for an undocumented venue-side reason (most likely
          ``selfTradePreventionType="MAKER"`` cancelling the taker
          when it would cross the bot's own resting orders mid-flatten).
          snap_20260427_095951: 3 attempts × 125 s with zero fills.
        * IOC LIMIT orders use the LIMIT path in ``_place_order`` where
          ``priceE9`` is the actual signed limit. An IOC LIMIT priced
          ``MARKET_CLOSE_SLIPPAGE_BPS`` (default 100 = 1 %) AGAINST the
          close direction crosses any realistic depth while preserving
          a hard slippage cap. Reduce-only + IOC keep the order from
          growing inventory or resting at a bad price.

        Operator escape hatch:
        ``BLUEFIN_MARKET_CLOSE_USE_LIMIT_IOC=False`` reverts to the
        legacy MARKET path (broken on SUI-PERP, but kept available in
        case a future Bluefin update fixes it or the LIMIT path
        develops its own quirk).

        * Closing long  → SELL → limit = mark × (1 − slip).
        * Closing short → BUY  → limit = mark × (1 + slip).
        """
        pos = self.fetch_position(self._settings.bluefin_account_address, symbol)
        abs_qty = abs(pos.position_qty)
        if abs_qty <= 0:
            return {"data": {"status": "noop_already_flat"}}
        qty = float(sz) if sz is not None and sz > 0 else abs_qty
        buy = pos.position_qty < 0
        slip_bps = float(
            getattr(self._settings, "market_close_slippage_bps", 100.0)
        )
        slip_factor = max(0.0, slip_bps) / 10_000.0
        mark = float(pos.mark_price or 0.0)
        if mark <= 0.0:
            # Defensive: with no mark we can't construct a sane limit.
            # Fall back to MARKET so the venue uses its own clamps. With
            # the BUG-015 LIMIT-IOC default this branch is unreachable
            # in normal operation (mark is always populated post-seed).
            limit_px = 0.0
            order_type = _ORDER_TYPE_MARKET
        else:
            limit_px = mark * (1.0 + slip_factor) if buy else mark * (1.0 - slip_factor)
            use_limit_ioc = bool(
                getattr(self._settings, "bluefin_market_close_use_limit_ioc", True)
            )
            order_type = _ORDER_TYPE_LIMIT if use_limit_ioc else _ORDER_TYPE_MARKET
        return self._place_order(
            symbol=symbol,
            is_buy=buy,
            sz=qty,
            limit_px=limit_px,
            post_only=False,
            reduce_only=True,
            ioc=True,
            order_type=order_type,
            client_order_id=None,
        )

    def rest_runtime_counters(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for op, n in self._rest_call_counts.items():
            out[f"{op}_calls"] = int(n)
        for op, n in self._rest_retry_counts.items():
            out[f"{op}_retries"] = int(n)
        for op, n in self._rest_rate_limited_retry_counts.items():
            out[f"{op}_rate_limited_retries"] = int(n)
        # Rate-limit defense counters (see BLUEFIN_MIN_PLACE_INTERVAL_SECONDS
        # and BLUEFIN_REST_429_* in app.config). Surface in /state/current
        # so operators can spot a rate-limit episode in telemetry even if
        # the bot self-recovers via retry.
        out["bluefin_rest_429_retries_total"] = int(self._rest_429_retries_total)
        out["bluefin_place_throttle_waits_total"] = int(
            self._place_throttle_waits_total
        )
        out["bluefin_place_throttle_wait_ms_total"] = int(
            self._place_throttle_wait_seconds_total * 1000.0
        )
        # How many cancels have been fanned out to cancel-all because the
        # server-side selective-cancel is broken. Each fanout wastes one
        # cancel+replace round-trip per collateral side. See
        # BLUEFIN_CANCEL_BY_HASH_WORKAROUND for the plan to turn this off.
        out["bluefin_cancel_workaround_fanouts_total"] = int(
            self._cancel_workaround_fanout_total
        )
        return out

    # ------------------------------------------------------------------
    # Wire-format interpreters
    # ------------------------------------------------------------------

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_bluefin_place_response(resp)

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        return interpret_bluefin_cancel_response(resp)

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_bluefin_order_status_response(resp)

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        return make_deterministic_bluefin_client_order_id(
            _normalize_bluefin_symbol(symbol), side, quote_cycle_id, price, size
        )


__all__ = ["BluefinClient"]
