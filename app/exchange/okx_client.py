"""OKX V5 USDT-perpetual SWAP adapter implementing :class:`PerpExchangeAdapter`.

Plan reference: ``plans/20260504-okx-setup/plan.md`` Phase 2.

Differences from the Binance adapter (the closest analog):

* **Auth uses HMAC-SHA256 over a different payload shape.** OKX signs
  ``<timestamp><method><requestPath><body>`` (not a urlencoded query
  string) and base64-encodes the result. Three headers carry the
  signature: ``OK-ACCESS-KEY``, ``OK-ACCESS-SIGN``, ``OK-ACCESS-TIMESTAMP``,
  plus ``OK-ACCESS-PASSPHRASE`` carrying the third secret set at
  API-key creation time.
* **Response envelope is uniformly wrapped** in ``{code, msg, data: [...]}``,
  with per-row ``sCode``/``sMsg``. See ``okx_responses.py`` for the
  interpreter shape.
* **Symbol notation** is dash-separated, e.g. ``DOGE-USDT-SWAP``.
  The bot internally uses the same string -- no normalization
  collapse-and-uppercase the way Binance needs.
* **Contract size != base size.** OKX perpetuals are quoted in
  CONTRACTS, not base-asset units. For ``DOGE-USDT-SWAP``, 1 contract
  = 1000 DOGE (varies by listing; the ``ctVal`` field on the
  instrument metadata gives the conversion). The adapter maintains a
  ``contract_value`` field on its ``SymbolSpec`` extension so the
  bot can quote in base-asset units and the adapter translates at
  the wire boundary. **Important**: ``size_step`` and ``min_size``
  on the SymbolSpec are reported in BASE-ASSET units (post-conversion)
  so the bot's quoting math stays unit-agnostic.
* **Position mode**: OKX defaults accounts to "long_short" mode where
  positions are tracked separately by side. Our bot expects "net" mode
  (one position per symbol with signed quantity). The operator must
  switch their account to net-mode via ``POST /api/v5/account/set-position-mode``
  before live trading. The adapter does NOT switch this automatically
  -- it's an operator concern, and a wrong mode at startup raises
  loudly so the operator notices.
* **Cancels are synchronous** (HTTP 200 with the cancelled order's
  state). No need for the ``has_pending_cancel`` capability.
* **No listenKey concept**: WS auth happens via a signed login frame
  on the WebSocket itself (see ``okx_ws.py``). The REST client
  doesn't expose ``spawn_listen_key`` / ``keepalive_listen_key``.

Operational model:

* OKX expects the ``x-simulated-trading: 1`` header on every request
  in demo / paper-trading mode. Set ``OKX_DEMO_TRADING=true`` to
  enable it. The same flag also flips the WS endpoint path.
* OKX rejects requests where the signed timestamp is more than 30s
  off from server time. NTP-synced hosts are fine; the adapter does
  not currently negotiate a server-time offset (Binance equivalent of
  ``serverTime`` query). If clock skew becomes a problem in practice,
  add a periodic ``GET /api/v5/public/time`` and adjust.
* Rate-limit handling: HTTP 429 + body code ``50011`` both signal
  rate limiting. We honour both via the shared retry layer in
  :mod:`app.exchange.exchange_retry`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx

from app.config import Settings
from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw
from app.exchange.exchange_retry import RetryPolicy, exchange_call_with_retry
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.exchange.okx_action_ws import (
    OkxActionWs,
    OkxActionWsError,
)
from app.exchange.okx_responses import (
    interpret_okx_cancel_response,
    interpret_okx_order_status_response,
    interpret_okx_place_response,
    make_deterministic_okx_client_order_id,
)
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC, SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)


# OKX V5 order types we use.
_OKX_ORDER_TYPE_LIMIT = "limit"
_OKX_ORDER_TYPE_POST_ONLY = "post_only"
_OKX_ORDER_TYPE_MARKET = "market"
_OKX_ORDER_TYPE_IOC = "ioc"


# 1.4.0 cancel-prio Phase 0c: ops in this set get the cancel-specific
# HTTP timeout + retry policy. Keep in sync with the cancel methods
# defined below (``cancel_order``, ``cancel_order_by_cloid``,
# ``cancel_batch``); a method calling ``self._request(op="cancel_X",
# ...)`` is the source of truth for ``op`` names.
_CANCEL_REST_OPS: frozenset[str] = frozenset(
    {"cancel_order", "cancel_order_by_cloid", "cancel_batch"}
)


def _normalize_okx_symbol(symbol: str) -> str:
    """OKX SWAP symbols are dash-separated and case-sensitive,
    e.g. ``DOGE-USDT-SWAP``. Be tolerant of lowercase / no dash
    inputs and normalise.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s
    # Common short forms a caller might pass: "DOGEUSDT" or "DOGE-USDT".
    # Map to canonical SWAP form when possible.
    if "-SWAP" not in s and "USDT" in s:
        # "DOGEUSDT" -> "DOGE-USDT-SWAP"; "DOGE-USDT" -> "DOGE-USDT-SWAP"
        if "-USDT" in s:
            return s + "-SWAP"
        if s.endswith("USDT"):
            base = s[:-4]
            return f"{base}-USDT-SWAP"
    return s


def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _okx_iso_timestamp() -> str:
    """OKX requires ISO-8601 with millisecond precision and a literal
    'Z' suffix, e.g. ``2026-05-04T08:30:00.000Z``. ``datetime.isoformat()``
    emits ``+00:00`` instead, so we replace.
    """
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class OkxApiError(Exception):
    """Raised when OKX returns a non-zero ``code`` on a REST call we
    cannot safely interpret as "no data".

    Carries:
    - ``code`` — the OKX top-level ``code`` field (e.g. ``"50011"`` for
      rate limit, ``"50001"`` for missing key, ``"50113"`` for invalid
      signature).
    - ``msg`` — the OKX top-level ``msg`` field (human-readable).
    - ``status_code`` — populated to 429 when the code is OKX's
      rate-limit code so callers using duck-typed rate-limit detection
      (e.g. ``OrderManager._is_rate_limited_exception``) treat it the
      same as an HTTP 429.

    Why this exists (2026-05-13 Codex bug review, CRITICAL #1):
    ``fetch_open_orders_raw`` and ``fetch_recent_fills_raw`` used to
    return ``[]`` on non-zero ``code``, which the reconcile path could
    not distinguish from a genuinely empty venue. On auth /
    passphrase / rate-limit failures the bot would treat live orders
    as gone (cancel-all became a silent no-op; reconcile would
    re-place duplicates). Raising explicitly forces callers into
    their existing error/rate-limit branches.
    """

    # OKX V5 rate-limit code. Body-level signal that matches HTTP 429.
    OKX_RATE_LIMIT_CODE = "50011"

    def __init__(self, code: str, msg: str, endpoint: str) -> None:
        self.code = str(code)
        self.msg = str(msg)
        self.endpoint = endpoint
        # Make duck-typed rate-limit detection work — see
        # ``OrderManager._is_rate_limited_exception`` which inspects
        # ``status_code`` first.
        self.status_code: Optional[int] = (
            429 if self.code == self.OKX_RATE_LIMIT_CODE else None
        )
        super().__init__(
            f"okx_api_error endpoint={endpoint} code={self.code} msg={self.msg[:200]}"
        )


class RateLimitPool(str, Enum):
    """OKX V5 per-endpoint rate-limit pool identifiers.

    rate-limit-observability Phase 1 (v1.4.20). The OKX rate-limit
    ledger is per-UID and per-endpoint-group; the trading hot path
    distributes across these pools depending on which call type
    fires:

    * ``PLACE_SINGLE`` (60/2s) — ``/trade/order``, ``order`` WS op
    * ``PLACE_BATCH``  (300/2s) — ``/trade/batch-orders``, ``batch-orders`` WS op
    * ``CANCEL_SINGLE`` (60/2s) — ``/trade/cancel-order``, ``cancel-order`` WS op
    * ``CANCEL_BATCH`` (300/2s) — ``/trade/cancel-batch-orders``, ``batch-cancel-orders`` WS op
    * ``AMEND_BATCH``  (300/2s) — ``/trade/amend-batch-orders``, ``batch-amend-orders`` WS op
    * ``AMEND_SINGLE`` (60/2s) — ``/trade/amend-order`` (not currently used)
    * ``READS``        (20/2s) — ``/trade/orders-pending``, ``/trade/fills``,
                                  ``/account/*`` (read-side endpoints)
    * ``OTHER``        — unsigned or unclassified; not tracked against a cap

    Caps are MM-tier values per the operator's note in
    ``memory/project_hbdc_okx.md`` (1200 req/2s sub-account limit;
    300/2s on batch endpoints; 60/2s on single-order endpoints).
    """

    PLACE_SINGLE = "place_single"
    PLACE_BATCH = "place_batch"
    CANCEL_SINGLE = "cancel_single"
    CANCEL_BATCH = "cancel_batch"
    AMEND_SINGLE = "amend_single"
    AMEND_BATCH = "amend_batch"
    READS = "reads"
    OTHER = "other"


# 2-second cap per pool. Used by ``RestRateWindow.snapshot()`` to
# compute ``pct_of_cap`` for the dashboard's Connectivity panel.
# ``OTHER`` has no cap (unsigned bootstraps, etc.); excluded from
# the dict so callers see ``cap is None`` for it.
POOL_CAPS_2S: dict[RateLimitPool, int] = {
    RateLimitPool.PLACE_SINGLE: 60,
    RateLimitPool.PLACE_BATCH: 300,
    RateLimitPool.CANCEL_SINGLE: 60,
    RateLimitPool.CANCEL_BATCH: 300,
    RateLimitPool.AMEND_SINGLE: 60,
    RateLimitPool.AMEND_BATCH: 300,
    RateLimitPool.READS: 20,
}


def pool_for_endpoint(
    path: str = "", op: str = ""
) -> RateLimitPool:
    """Map a REST ``request_path`` and / or WS ``op`` to the OKX
    rate-limit pool. ``path`` is the REST URL path (e.g.
    ``/api/v5/trade/order``); ``op`` is the WS-action op id (e.g.
    ``batch-amend-orders``). Either may be empty depending on the
    call site; the function checks both.

    Specificity matters — ``/trade/order`` is the single-place
    endpoint but ``/trade/batch-orders`` would substring-match
    ``/trade/order`` if checked naively. We check more-specific
    paths first.
    """
    # Check WS op (more specific) before REST path (some paths are
    # substrings of others).
    if op == "batch-amend-orders":
        return RateLimitPool.AMEND_BATCH
    if op == "batch-cancel-orders":
        return RateLimitPool.CANCEL_BATCH
    if op == "batch-orders":
        return RateLimitPool.PLACE_BATCH
    if op == "cancel-order":
        return RateLimitPool.CANCEL_SINGLE
    if op == "amend-order":
        return RateLimitPool.AMEND_SINGLE
    if op == "order":
        return RateLimitPool.PLACE_SINGLE

    # REST paths — use endswith-with-trailing-slash anchoring so
    # ``/trade/order`` doesn't substring-match ``/trade/orders-pending``
    # (the trap that broke v1.4.20 unit tests on first cut).
    # The path may carry a query string (``?ordId=...``) so we strip
    # everything after ``?`` before comparing.
    bare_path = path.split("?", 1)[0].rstrip("/")
    if bare_path.endswith("/trade/amend-batch-orders"):
        return RateLimitPool.AMEND_BATCH
    if bare_path.endswith("/trade/amend-order"):
        return RateLimitPool.AMEND_SINGLE
    if bare_path.endswith("/trade/cancel-batch-orders"):
        return RateLimitPool.CANCEL_BATCH
    if bare_path.endswith("/trade/cancel-order"):
        return RateLimitPool.CANCEL_SINGLE
    if bare_path.endswith("/trade/batch-orders"):
        return RateLimitPool.PLACE_BATCH
    if bare_path.endswith("/trade/order"):
        return RateLimitPool.PLACE_SINGLE
    if (
        bare_path.endswith("/trade/orders-pending")
        or bare_path.endswith("/trade/fills")
        or bare_path.endswith("/trade/fills-history")
        or "/account/" in bare_path
    ):
        return RateLimitPool.READS
    return RateLimitPool.OTHER


class RestRateWindow:
    """Sliding-window REST call-rate tracker for the OKX rate-limit
    dashboard gauge (v1.4.4) + per-endpoint-pool splits
    (v1.4.20 rate-limit-observability Phase 1).

    Design — 20 buckets × 100 ms = 2-second window:
      * On every REST/WS-action call, increment the bucket aligned
        to the current 100 ms. O(1) — single dict get/set behind a
        threading.Lock; cost is microseconds vs. the ms-scale RTT
        of the call it's tracking.
      * On bucket roll, snapshot the rolling-2s sum into a peak
        history deque tagged by timestamp. Decay entries older than
        60 s on each roll.
      * ``snapshot()`` returns the aggregate gauge AND per-pool
        gauges for the dashboard's per-pool panel.

    The 2-second window matches OKX's published cap units
    (``1200 req/2s`` for the MM tier on the per-UID trade
    endpoint; ``300 req/2s`` for the batch endpoints; ``60/2s``
    for the single-order endpoints). The peak metric gives the
    operator "how close did we get in the last minute" without
    retaining minute-scale state.

    Per-pool split (Phase 1): each ``record(pool=...)`` call
    increments BOTH the aggregate window AND the pool-specific
    window. The aggregate stays for back-compat with the legacy
    `okx_rate_window_*` dashboard keys; the per-pool windows feed
    the new Connectivity panel that shows pressure on each cap
    independently. See ``plans/rate-limit-observability.md``.
    """

    _BUCKET_MS = 100
    _NUM_BUCKETS = 20  # 20 × 100 ms = 2 s
    _PEAK_HISTORY_MS = 60_000  # rolling peak over last 60 s
    # rate-limit-observability extension (v1.4.22): 1Hz sample
    # history per pool for min/max/median/p95 stats over a 60-second
    # rolling window. Distinct from the 100ms ``_buckets`` (used for
    # current_2s rolling sum) and the ``_peaks`` deque (used for the
    # peak gauge). The 1Hz samples represent finalized 1-second
    # counts — non-overlapping, so quantiles over them describe the
    # real per-second rate distribution rather than a smoothed
    # rolling-window average.
    _SECOND_MS = 1_000
    _SECOND_HISTORY_S = 60

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Aggregate window (legacy — back-compat with `okx_rate_window_*`).
        self._buckets: list[int] = [0] * self._NUM_BUCKETS
        self._head_idx: int = 0
        self._head_bucket_ms: int = 0
        # (bucket_start_ms, rolling_2s_sum_at_that_moment)
        self._peaks: deque[tuple[int, int]] = deque()
        # Cumulative counters surfaced alongside the gauge — useful
        # for cross-checking that the bucket arithmetic isn't dropping
        # increments under thread contention.
        self._total_recorded: int = 0
        # rate-limit-observability Phase 1 (v1.4.20): per-pool windows.
        # Each pool has its own bucket ring + peak deque + total. The
        # aggregate window above still receives every increment too.
        # Pools are populated lazily on first record() to avoid
        # constructing windows for pools the venue never sees (e.g.
        # AMEND_SINGLE on bots that only use batch endpoints).
        self._pool_buckets: dict[RateLimitPool, list[int]] = {}
        self._pool_head_idx: dict[RateLimitPool, int] = {}
        self._pool_head_bucket_ms: dict[RateLimitPool, int] = {}
        self._pool_peaks: dict[RateLimitPool, deque[tuple[int, int]]] = {}
        self._pool_total: dict[RateLimitPool, int] = {}
        # v1.4.22 — 1Hz sample history per pool. ``_pool_1s_history``
        # holds the last 60 FINALIZED per-second counts (a count is
        # finalized when its 1-second window closes; the
        # currently-accumulating second lives in ``_pool_cur_sec_*``
        # until then). Used by ``snapshot_per_pool()`` to produce
        # min/max/median/p95 rate-per-second stats for the dashboard.
        # Storage: 60 ints/pool × ~8 pools = 480 ints ≈ 4 KB. Trivial.
        self._pool_1s_history: dict[RateLimitPool, deque[int]] = {}
        self._pool_cur_sec_ms: dict[RateLimitPool, int] = {}
        self._pool_cur_sec_count: dict[RateLimitPool, int] = {}

    def record(self, pool: RateLimitPool = RateLimitPool.OTHER) -> None:
        """Record one venue call. ``pool`` should be the
        ``RateLimitPool`` matching the endpoint (use
        ``pool_for_endpoint()`` to derive it from a REST path / WS
        op). Defaults to ``OTHER`` for unsigned bootstrap calls and
        unclassified traffic; OTHER is tracked but has no cap so the
        dashboard surface treats it as informational only.
        """
        now_ms = int(time.monotonic() * 1000.0)
        bucket_ms = (now_ms // self._BUCKET_MS) * self._BUCKET_MS
        with self._lock:
            # Aggregate (back-compat).
            self._advance_locked(bucket_ms)
            self._buckets[self._head_idx] += 1
            self._total_recorded += 1
            # Per-pool 100ms-bucket window.
            self._advance_pool_locked(pool, bucket_ms)
            self._pool_buckets[pool][self._pool_head_idx[pool]] += 1
            self._pool_total[pool] = self._pool_total.get(pool, 0) + 1
            # v1.4.22 — 1Hz sample history per pool. Each second's
            # count finalizes when the next-second boundary crosses;
            # the bounded deque (maxlen=60) auto-evicts samples older
            # than 60 seconds.
            self._advance_1s_locked(pool, now_ms)
            self._pool_cur_sec_count[pool] += 1

    def _ensure_pool_locked(self, pool: RateLimitPool) -> None:
        """Lazy-init the per-pool state on first record(). Caller
        must hold ``self._lock``."""
        if pool in self._pool_buckets:
            return
        self._pool_buckets[pool] = [0] * self._NUM_BUCKETS
        self._pool_head_idx[pool] = 0
        self._pool_head_bucket_ms[pool] = 0
        self._pool_peaks[pool] = deque()
        self._pool_total[pool] = 0
        # v1.4.22 — 1Hz sample history.
        self._pool_1s_history[pool] = deque(maxlen=self._SECOND_HISTORY_S)
        self._pool_cur_sec_ms[pool] = 0
        self._pool_cur_sec_count[pool] = 0

    def _advance_pool_locked(
        self, pool: RateLimitPool, bucket_ms: int
    ) -> None:
        """Per-pool counterpart of ``_advance_locked``. Mirrors the
        aggregate-window bucket-roll logic but on the pool-specific
        state. Caller must hold ``self._lock``."""
        self._ensure_pool_locked(pool)
        head_bucket_ms = self._pool_head_bucket_ms[pool]
        if head_bucket_ms == 0:
            self._pool_head_bucket_ms[pool] = bucket_ms
            return
        diff_buckets = (bucket_ms - head_bucket_ms) // self._BUCKET_MS
        if diff_buckets <= 0:
            return
        if diff_buckets >= self._NUM_BUCKETS:
            pre_clear_sum = sum(self._pool_buckets[pool])
            if pre_clear_sum > 0:
                self._pool_peaks[pool].append(
                    (head_bucket_ms, pre_clear_sum)
                )
            self._pool_buckets[pool] = [0] * self._NUM_BUCKETS
            self._pool_head_idx[pool] = 0
            self._pool_head_bucket_ms[pool] = bucket_ms
            self._pool_peaks[pool].append((bucket_ms, 0))
            self._decay_pool_peaks_locked(pool, bucket_ms)
            return
        for _ in range(diff_buckets):
            window_sum = sum(self._pool_buckets[pool])
            self._pool_peaks[pool].append(
                (self._pool_head_bucket_ms[pool], window_sum)
            )
            self._pool_head_idx[pool] = (
                self._pool_head_idx[pool] + 1
            ) % self._NUM_BUCKETS
            self._pool_buckets[pool][self._pool_head_idx[pool]] = 0
            self._pool_head_bucket_ms[pool] += self._BUCKET_MS
        self._decay_pool_peaks_locked(
            pool, self._pool_head_bucket_ms[pool]
        )

    def _decay_pool_peaks_locked(
        self, pool: RateLimitPool, now_ms: int
    ) -> None:
        cutoff = now_ms - self._PEAK_HISTORY_MS
        q = self._pool_peaks[pool]
        while q and q[0][0] < cutoff:
            q.popleft()

    def _advance_1s_locked(
        self, pool: RateLimitPool, now_ms: int
    ) -> None:
        """v1.4.22 — roll the per-pool 1Hz sample history forward.
        Called from ``record()`` while ``self._lock`` is held; the
        ``_ensure_pool_locked`` invariant guarantees the per-pool
        state already exists.

        When the current second boundary changes, finalize the prior
        second's count by pushing it to ``_pool_1s_history`` (the
        bounded deque auto-evicts samples older than 60 entries).
        If multiple seconds elapsed (idle gap), pad zero samples for
        each missed second so the deque stays time-aligned — the
        absence of pads would understate ``rate_per_sec_min`` after
        a gap (since the resumed activity's high samples wouldn't
        be balanced by the zero samples that should sit between).

        Reading current_sec_ms == 0 means first record() for this
        pool; just stamp the current second and start accumulating.
        """
        sec_ms = (now_ms // self._SECOND_MS) * self._SECOND_MS
        cur_sec = self._pool_cur_sec_ms[pool]
        if cur_sec == 0:
            # First record for this pool — start the current second.
            self._pool_cur_sec_ms[pool] = sec_ms
            return
        if sec_ms == cur_sec:
            # Same second still accumulating — caller will increment.
            return
        # New second boundary crossed. Finalize the prior second's
        # count, then pad zeros for any fully-elapsed seconds in
        # between (idle gap).
        hist = self._pool_1s_history[pool]
        hist.append(self._pool_cur_sec_count[pool])
        seconds_missed = (sec_ms - cur_sec) // self._SECOND_MS - 1
        # Cap pad count at the deque's capacity to avoid quadratic
        # work on multi-minute idle gaps — the deque would drop the
        # leading values anyway.
        seconds_missed = min(
            seconds_missed, self._SECOND_HISTORY_S
        )
        for _ in range(seconds_missed):
            hist.append(0)
        self._pool_cur_sec_ms[pool] = sec_ms
        self._pool_cur_sec_count[pool] = 0

    def _advance_locked(self, bucket_ms: int) -> None:
        """Move the head pointer forward to ``bucket_ms``, snapshotting
        the rolling-2s sum at each bucket boundary. Caller must hold
        ``self._lock``."""
        if self._head_bucket_ms == 0:
            self._head_bucket_ms = bucket_ms
            return
        diff_buckets = (bucket_ms - self._head_bucket_ms) // self._BUCKET_MS
        if diff_buckets <= 0:
            return
        if diff_buckets >= self._NUM_BUCKETS:
            # Window completely stale (no calls in >2 s) — clear.
            # BEFORE clearing, snapshot the current rolling-2s sum
            # into the peak history at the OLD head's timestamp.
            # Without this, a burst of activity followed by a long
            # idle would lose its peak signal entirely (test
            # ``test_old_records_decay_after_two_seconds`` locks
            # this behaviour in).
            pre_clear_sum = sum(self._buckets)
            if pre_clear_sum > 0:
                self._peaks.append((self._head_bucket_ms, pre_clear_sum))
            self._buckets = [0] * self._NUM_BUCKETS
            self._head_idx = 0
            self._head_bucket_ms = bucket_ms
            # Also mark a zero peak at the new boundary so the
            # post-gap state is visible — but the pre-clear snapshot
            # above ensures the burst is remembered.
            self._peaks.append((bucket_ms, 0))
            self._decay_peaks_locked(bucket_ms)
            return
        for _ in range(diff_buckets):
            window_sum = sum(self._buckets)
            self._peaks.append((self._head_bucket_ms, window_sum))
            self._head_idx = (self._head_idx + 1) % self._NUM_BUCKETS
            self._buckets[self._head_idx] = 0
            self._head_bucket_ms += self._BUCKET_MS
        self._decay_peaks_locked(self._head_bucket_ms)

    def _decay_peaks_locked(self, now_ms: int) -> None:
        cutoff = now_ms - self._PEAK_HISTORY_MS
        while self._peaks and self._peaks[0][0] < cutoff:
            self._peaks.popleft()

    def snapshot(self) -> dict[str, int]:
        """Return the AGGREGATE gauge values for ``rest_runtime_counters``.

        ``current_2s_rate`` is the sum of all 20 buckets right now;
        ``peak_2s_in_last_60s`` is the maximum rolling-2s seen in the
        last 60 s (or the current rate, whichever is larger — handles
        the case where the current rate just hit a new high but we
        haven't crossed a bucket boundary to snapshot it yet).

        Back-compat: this method's flat shape feeds the legacy
        ``okx_rate_window_*`` dashboard keys. Use ``snapshot_per_pool()``
        for the new per-endpoint-pool surface (Phase 1).
        """
        now_ms = int(time.monotonic() * 1000.0)
        bucket_ms = (now_ms // self._BUCKET_MS) * self._BUCKET_MS
        with self._lock:
            self._advance_locked(bucket_ms)
            current = sum(self._buckets)
            peak_hist = max((p[1] for p in self._peaks), default=0)
            peak = max(peak_hist, current)
            return {
                "current_2s_rate": int(current),
                "peak_2s_in_last_60s": int(peak),
                "total_recorded": int(self._total_recorded),
            }

    def snapshot_per_pool(self) -> dict[str, dict[str, Any]]:
        """rate-limit-observability Phase 1 (v1.4.20) + per-second
        stats extension (v1.4.22): per-pool gauge for the Connectivity
        dashboard panel.

        Returns one entry per pool that has ever recorded a call
        (lazy population — pools with no traffic are absent rather
        than zero, to keep the surface lean). Each entry contains:

          * ``current_2s``  — rolling 2s sum (live, not finalized)
          * ``peak_2s_60s`` — max rolling-2s in last 60s (or current,
                              whichever is larger)
          * ``total``       — cumulative counter since process start
          * ``cap``         — OKX 2s cap (None for OTHER)
          * ``pct_of_cap``  — peak_2s_60s / cap (or None when no cap)
          * ``rate_per_sec_min`` / ``_max`` / ``_median`` / ``_p95``
                            — distribution of FINALIZED 1-second
                              counts over the last 60s. None when
                              fewer than 2 seconds of data exist
                              (median over a single sample is the
                              sample itself; we want the distribution
                              shape). p95 is the 95th percentile of
                              the same series (interpolated). Lets
                              the operator distinguish "steady high
                              rate" (median ≈ max) from "bursty"
                              (median << max).
          * ``rate_per_sec_samples`` — count of finalized samples
                              in the deque (0-60).

        The aggregate window is also included under the key
        ``"aggregate"`` for cross-checks against the legacy gauge.
        Aggregate per-second stats are NOT computed (the per-pool
        stats are what's actionable; aggregate is a sanity check).
        """
        now_ms = int(time.monotonic() * 1000.0)
        bucket_ms = (now_ms // self._BUCKET_MS) * self._BUCKET_MS
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            # Aggregate.
            self._advance_locked(bucket_ms)
            agg_current = sum(self._buckets)
            agg_peak_hist = max((p[1] for p in self._peaks), default=0)
            agg_peak = max(agg_peak_hist, agg_current)
            out["aggregate"] = {
                "current_2s": int(agg_current),
                "peak_2s_60s": int(agg_peak),
                "total": int(self._total_recorded),
                "cap": None,
                "pct_of_cap": None,
            }
            # Per-pool.
            for pool in list(self._pool_buckets.keys()):
                self._advance_pool_locked(pool, bucket_ms)
                # v1.4.22 — also advance the 1Hz sample state so
                # idle pools (no record() in the current second)
                # still get their prior seconds finalized with the
                # correct zero-padding. Otherwise stats for a pool
                # that hasn't received traffic in the last second
                # would lag behind the active pools.
                self._advance_1s_locked(pool, now_ms)
                current = sum(self._pool_buckets[pool])
                peak_hist = max(
                    (p[1] for p in self._pool_peaks[pool]),
                    default=0,
                )
                peak = max(peak_hist, current)
                cap = POOL_CAPS_2S.get(pool)
                pct = (
                    (float(peak) / float(cap))
                    if cap is not None and cap > 0
                    else None
                )
                # Per-second stats from finalized 1-second samples.
                # We do NOT include the currently-accumulating
                # second (incomplete; would bias the min toward 0).
                rate_stats = self._compute_rate_per_sec_stats_locked(pool)
                out[pool.value] = {
                    "current_2s": int(current),
                    "peak_2s_60s": int(peak),
                    "total": int(self._pool_total.get(pool, 0)),
                    "cap": cap,
                    "pct_of_cap": pct,
                    **rate_stats,
                }
        return out

    def _compute_rate_per_sec_stats_locked(
        self, pool: RateLimitPool
    ) -> dict[str, Optional[float]]:
        """v1.4.22 — derive min/max/median/p95 from the finalized
        1-second samples deque. Caller must hold ``self._lock``.

        Returns None for all stats when fewer than 2 samples exist
        (a single sample's median is just itself; the distribution
        shape is meaningless until ≥2 samples). The 0-sample case
        applies to a pool that just started recording in the
        current second — by the next snapshot 1+ samples will have
        finalized.
        """
        hist = self._pool_1s_history.get(pool)
        if hist is None or len(hist) < 2:
            return {
                "rate_per_sec_min": None,
                "rate_per_sec_max": None,
                "rate_per_sec_median": None,
                "rate_per_sec_p95": None,
                "rate_per_sec_samples": 0 if hist is None else len(hist),
            }
        samples = sorted(hist)
        n = len(samples)
        # p95 by linear interpolation; matches numpy's default. For
        # the typical 60-sample case the index 0.95 * 59 = 56.05,
        # which interpolates between samples[56] and samples[57].
        rank = 0.95 * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        p95 = samples[lo] + (samples[hi] - samples[lo]) * frac
        return {
            "rate_per_sec_min": int(samples[0]),
            "rate_per_sec_max": int(samples[-1]),
            "rate_per_sec_median": float(statistics.median(samples)),
            "rate_per_sec_p95": float(p95),
            "rate_per_sec_samples": n,
        }


class OkxClient:
    """OKX V5 USDT-perpetual SWAP adapter.

    Lifetime: constructed once at bot startup; bootstraps the symbol
    spec via ``GET /api/v5/public/instruments`` synchronously and
    caches. A failed bootstrap leaves ``symbol_spec_fetched_ok=False``
    and the bot uses ``FALLBACK_SYMBOL_SPEC`` (matching the Binance
    pattern).

    Quote-unit conversion: OKX trades in CONTRACTS but the bot quotes
    in base-asset units. The adapter maintains the contract value
    (``_contract_value``, e.g. 1000.0 for DOGE-USDT-SWAP) and converts
    on each call. SymbolSpec.size_step / min_size are reported in
    base-asset units so the bot's rounding stays clean.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._symbol = _normalize_okx_symbol(settings.symbol)
        self._api_key = (settings.okx_api_key or "").strip()
        self._api_secret = (settings.okx_api_secret or "").strip()
        self._passphrase = (settings.okx_api_passphrase or "").strip()
        self._rest_url = (settings.okx_rest_url or "").strip().rstrip("/")
        self._demo = bool(settings.okx_demo_trading)

        # 1.3.108 cancel-prio Phase 2: per-op HTTP client split — separate
        # connection pools for cancel ops vs. everything else. Rationale:
        # under load the place pool can stall (OKX takes 1-3s to ack a
        # crowded place), and with a SHARED pool an outbound cancel
        # would have to wait for a place slot to free up — visible in
        # the Phase 0.5 cancel-latency decomposition as a
        # ts_cancel_requested → ts_cancel_sent gap. Phase 3 (parallel
        # workers) will saturate pools concurrently, so this split lands
        # ahead of that as defensive prep with no behavior change today.
        #
        # _http_place keeps the current generous timeouts (no behavior
        # change for the dominant happy path). _http_cancel uses tight
        # cancel-specific budgets baked into the client baseline; the
        # per-request override in ``_request`` for cancel ops still
        # applies on top (it reads ``cancel_http_read_timeout_seconds``
        # dynamically so the operator can re-tune at runtime via env).
        # pool=0.2 on the cancel client is the meaningful change: if the
        # cancel pool is saturated, fail-fast in 200ms rather than block
        # for 8s on the shared default.
        #
        # http2 enabled by default — OKX's REST CDN supports it, and
        # HTTP/2 stream multiplexing removes TCP head-of-line blocking
        # on the shared TLS connection. One-line revert via
        # OKX_HTTP2_ENABLED=false if any compatibility issue surfaces.
        http2_enabled = bool(getattr(settings, "okx_http2_enabled", True))
        self._http_place = httpx.Client(
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=4.0, pool=8.0),
            http2=http2_enabled,
        )
        self._http_cancel = httpx.Client(
            timeout=httpx.Timeout(connect=2.0, read=1.5, write=1.0, pool=0.2),
            http2=http2_enabled,
        )

        # Telemetry counters surfaced via rest_runtime_counters().
        self._rest_call_counts: dict[str, int] = {}
        self._rest_retry_counts: dict[str, int] = {}
        self._rest_429_retries_total: int = 0
        # 1.4.4: row-level "Rate limit reached" hits on the batch / single
        # place endpoints (sCode-or-sMsg matched but top envelope was
        # already past auth/rate-limit checks). Pre-1.4.4 these were
        # mis-classified as exchange_rejected and dropped permanently;
        # the new interpreter surfaces them as transport_rejected so
        # callers can retry. Counter exists to make "how often did we
        # actually hit the row-level wall" visible on the dashboard.
        self._okx_row_rate_limit_total: int = 0
        # 1.4.4: REST call-rate sliding window — sliding-2s rate + 60s
        # peak surfaced via rest_runtime_counters() for the dashboard
        # gauge. Tracks ALL signed REST calls + WS-action sends (place
        # / cancel) — both consume the same OKX rate-limit budgets at
        # the venue. See RestRateWindow above for design rationale.
        self._rate_window = RestRateWindow()
        self._lock = threading.Lock()

        # 1.3.110 cancel-prio Phase 4a: OKX trade-via-WS for cancels.
        # Construct the action-WS coordinator lazily on first use to
        # avoid spawning the daemon thread / opening a socket for
        # processes that don't trade (e.g. read-only dashboards
        # spinning up an OkxClient just for symbol-spec lookup).
        # ``last_exchange_transport_mode`` is read by execution.py /
        # the postmortem tool — matches the existing HL convention.
        self._action_ws: Optional[OkxActionWs] = None
        self._action_ws_lock = threading.Lock()
        self.last_exchange_transport_mode: str = "http"
        self._cancel_ws_send_count: int = 0
        self._cancel_ws_fallback_count: int = 0
        # 1.3.111 Phase 4b: WS place telemetry. Parallel surface to the
        # cancel counters above — operator watches the send vs. fallback
        # ratio to verify the WS place path is delivering after Stage 4
        # activation (≥99% send is the acceptance bar).
        self._place_ws_send_count: int = 0
        self._place_ws_fallback_count: int = 0

        # Symbol spec + contract-value conversion.
        self.symbol_spec_fetched_ok: bool = False
        self._symbol_spec: SymbolSpec = FALLBACK_SYMBOL_SPEC
        self._contract_value: float = 1.0  # base units per 1 contract
        # 1.3.123 Phase 4a v2: cache the numeric ``instIdCode`` returned
        # by /api/v5/public/instruments. The OKX colo trade-WS endpoint
        # requires this field on every cancel frame (sCode 50014
        # "Parameter instIdCode can not be empty" if missing).
        # Standard endpoints accept frames without it. None when the
        # bootstrap failed or the field is absent from the response.
        self._inst_id_code: Optional[int] = None
        try:
            spec, ctval, inst_id_code = self._bootstrap_symbol_spec()
            self._symbol_spec = spec
            self._contract_value = ctval
            self._inst_id_code = inst_id_code
            self.symbol_spec_fetched_ok = True
            logger.info(
                "okx_symbol_spec_bootstrap_success symbol=%s "
                "price_tick=%s size_step=%s min_size=%s "
                "contract_value=%s inst_id_code=%s",
                self._symbol,
                self._symbol_spec.price_tick,
                self._symbol_spec.size_step,
                self._symbol_spec.min_size,
                self._contract_value,
                self._inst_id_code,
            )
        except Exception:
            logger.exception(
                "okx_symbol_spec_bootstrap_failed symbol=%s -- "
                "falling back to FALLBACK_SYMBOL_SPEC",
                self._symbol,
            )

    # ------------------------------------------------------------------
    # Adapter surface
    # ------------------------------------------------------------------

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    @property
    def contract_value(self) -> float:
        """Base-asset units per 1 OKX contract for this adapter's symbol.
        E.g. 1000.0 for DOGE-USDT-SWAP. Exposed for tests + diagnostics.
        """
        return self._contract_value

    def has_write_access(self) -> bool:
        return bool(self._api_key and self._api_secret and self._passphrase)

    # ------------------------------------------------------------------
    # HTTP / signing
    # ------------------------------------------------------------------

    def _sign(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str,
    ) -> str:
        """OKX signing: base64(HMAC-SHA256(secret,
        timestamp + method + requestPath + body)).

        ``request_path`` includes the query string (e.g.
        ``/api/v5/trade/order?ordId=123``). ``body`` is the literal
        JSON string sent on POSTs (or empty string on GETs).
        """
        prehash = f"{timestamp}{method.upper()}{request_path}{body or ''}"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def _request(
        self,
        op: str,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        signed: bool = False,
        retry: bool = True,
    ) -> dict[str, Any]:
        """Send one REST request. Returns the parsed JSON response.

        OKX always returns ``{code, msg, data: [...]}`` on 2xx (with
        ``code != "0"`` carrying error info) AND on 4xx (rare; usually
        for malformed-JSON cases). We don't raise on body-level errors
        -- we let interpreters branch.

        On HTTP 429 we raise so the retry layer backs off; everything
        else returns the parsed body.
        """
        # Build request path including query string for signing.
        request_path = path
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            request_path = f"{path}?{qs}"
        url = f"{self._rest_url}{request_path}"

        body_str = json.dumps(body, separators=(",", ":")) if body else ""

        headers: dict[str, str] = {}
        if signed:
            ts = _okx_iso_timestamp()
            headers["OK-ACCESS-KEY"] = self._api_key
            headers["OK-ACCESS-SIGN"] = self._sign(
                ts, method, request_path, body_str
            )
            headers["OK-ACCESS-TIMESTAMP"] = ts
            headers["OK-ACCESS-PASSPHRASE"] = self._passphrase
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._demo:
            headers["x-simulated-trading"] = "1"

        with self._lock:
            self._rest_call_counts[op] = self._rest_call_counts.get(op, 0) + 1
        # 1.4.4: bump the sliding-window rate tracker on every signed
        # REST call. Unsigned calls (symbol-spec bootstrap, public
        # ticker) don't count against the trade-endpoint budget, so
        # we only record signed traffic — that's what the dashboard
        # gauge measures distance to.
        # v1.4.20 Phase 1 (rate-limit-observability): tag with the
        # pool inferred from the request path so per-pool windows
        # can split apart cancel / place / amend / reads pressure.
        if signed:
            self._rate_window.record(
                pool=pool_for_endpoint(path=path, op=op)
            )

        # 1.4.0 cancel-prio Phase 0c: cancel ops get a shorter read
        # timeout (1.5 s vs the default 8 s baked into the place client).
        # We pass it per-request via httpx's ``timeout=`` kwarg so the
        # value can be re-tuned at runtime via the
        # ``cancel_http_read_timeout_seconds`` setting without rebuilding
        # the client. ``None`` falls back to the client's baseline.
        #
        # 1.3.108 Phase 2: also pick the right pool — cancel ops route
        # through ``_http_cancel`` (small, fail-fast pool); everything
        # else uses ``_http_place`` (current generous defaults). This is
        # the pool-isolation guarantee that lets Phase 3's parallel
        # workers run cancels and places concurrently without
        # cross-blocking on a single shared pool.
        is_cancel_op = op in _CANCEL_REST_OPS
        http_client = self._http_cancel if is_cancel_op else self._http_place
        request_timeout: Optional[httpx.Timeout] = None
        if is_cancel_op:
            request_timeout = httpx.Timeout(
                connect=2.0,
                read=float(self._settings.cancel_http_read_timeout_seconds),
                write=1.0,
                pool=2.0,
            )

        def _do() -> dict[str, Any]:
            try:
                resp = http_client.request(
                    method,
                    url,
                    content=body_str.encode("utf-8") if body_str else None,
                    headers=headers,
                    timeout=request_timeout if request_timeout is not None else http_client.timeout,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "okx_request_transport_failed op=%s method=%s "
                    "path=%s err=%s",
                    op,
                    method,
                    path,
                    str(exc)[:200],
                )
                # Transport failure -- synthesize an OKX-shaped envelope
                # so interpreters can branch uniformly.
                return {
                    "code": "50000",
                    "msg": f"transport_error:{exc}"[:500],
                    "data": [],
                }

            if resp.status_code == 429:
                with self._lock:
                    self._rest_429_retries_total += 1
                logger.warning(
                    "okx_rate_limit op=%s status=%s retry_after=%s",
                    op,
                    resp.status_code,
                    resp.headers.get("Retry-After"),
                )
                raise httpx.HTTPStatusError(
                    f"http_{resp.status_code}",
                    request=resp.request,
                    response=resp,
                )

            try:
                parsed = resp.json()
            except ValueError:
                parsed = {
                    "code": "50000",
                    "msg": (resp.text or "")[:500],
                    "data": [],
                }

            if not isinstance(parsed, dict):
                # OKX should always return a dict; if not, wrap so
                # downstream code can branch.
                parsed = {"code": "50000", "msg": "non_dict_body", "data": []}

            parsed.setdefault("_http_status", resp.status_code)
            return parsed

        if retry:
            # 1.4.0 cancel-prio Phase 0c: cancel ops use a custom retry
            # policy (more aggressive — 50/100/200/400ms vs default
            # 350/700/1400/2800ms) so a transient failure on a
            # time-critical cancel re-attempts sooner.
            cancel_policy: Optional[RetryPolicy] = None
            if is_cancel_op:
                cancel_policy = RetryPolicy(
                    max_attempts=int(self._settings.cancel_retry_max_attempts),
                    base_seconds=float(self._settings.cancel_retry_base_seconds),
                    cap_seconds=float(self._settings.cancel_retry_cap_seconds),
                )
            try:
                return exchange_call_with_retry(
                    op, _do, self._settings, policy=cancel_policy
                )
            except Exception as exc:
                with self._lock:
                    self._rest_retry_counts[op] = (
                        self._rest_retry_counts.get(op, 0) + 1
                    )
                logger.exception(
                    "okx_request_retry_exhausted op=%s err=%s",
                    op,
                    str(exc)[:200],
                )
                return {
                    "code": "50000",
                    "msg": f"retry_exhausted:{exc}"[:500],
                    "data": [],
                }
        return _do()

    # ------------------------------------------------------------------
    # Symbol-spec bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_symbol_spec(self) -> tuple[SymbolSpec, float, Optional[int]]:
        """Fetch ``GET /api/v5/public/instruments?instType=SWAP&instId=...``
        and build a SymbolSpec from the matched row.

        Returns ``(SymbolSpec, contract_value_in_base_units, inst_id_code)``.

        OKX instrument fields used:
          * ``tickSz``      -- minimum price increment
          * ``lotSz``       -- minimum size increment in CONTRACTS
          * ``minSz``       -- minimum order size in CONTRACTS
          * ``ctVal``       -- contract value, i.e. base units per contract
                                (e.g. "1000" for DOGE-USDT-SWAP)
          * ``ctValCcy``    -- currency the contract value is quoted in
                                (should be the base currency for our usage)
          * ``instIdCode``  -- 1.3.123 Phase 4a v2: numeric instrument
                                ID required on every OKX colo trade-WS
                                cancel/order frame. Returned as an
                                integer in the JSON response. None when
                                absent (older instrument versions /
                                non-colo endpoints may omit it).

        We convert ``lotSz`` and ``minSz`` to base-asset units so the
        SymbolSpec the bot consumes stays unit-consistent across venues.
        """
        resp = self._request(
            "instruments",
            "GET",
            "/api/v5/public/instruments",
            params={"instType": "SWAP", "instId": self._symbol},
            signed=False,
            retry=True,
        )
        if str(resp.get("code")) != "0":
            raise RuntimeError(
                f"OKX instruments lookup failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                f"OKX instruments returned no rows for {self._symbol!r}"
            )
        row = next(
            (r for r in rows if isinstance(r, dict) and r.get("instId") == self._symbol),
            None,
        )
        if row is None:
            raise RuntimeError(
                f"symbol {self._symbol!r} not in OKX SWAP universe"
            )

        price_tick = _coerce_float(row.get("tickSz"))
        contract_value = _coerce_float(row.get("ctVal"))
        lot_contracts = _coerce_float(row.get("lotSz"))
        min_contracts = _coerce_float(row.get("minSz"))

        if price_tick <= 0 or contract_value <= 0 or lot_contracts <= 0:
            raise RuntimeError(
                f"invalid OKX instrument fields for {self._symbol}: "
                f"tickSz={price_tick} ctVal={contract_value} "
                f"lotSz={lot_contracts}"
            )

        # Convert contracts -> base-asset units.
        size_step = lot_contracts * contract_value
        min_size = (min_contracts or lot_contracts) * contract_value

        # OKX publishes ``minSz`` in contracts. The bot also wants a
        # USD min-notional. OKX doesn't publish one explicitly per
        # instrument (it uses contract-count gates), so we synthesise a
        # conservative floor. v1.5.265 — make this configurable per
        # profile via ``MIN_VENUE_NOTIONAL_USD`` (venue-agnostic name
        # — the same hook applies to any venue that lacks a published
        # USD min-notional). The legacy default of $5 "to match
        # Binance" was preventing rung-1 (behind-touch) from firing
        # on low-priced perps like TON-USDT-SWAP ($1.73) where 3
        # contracts × $1.73 = $5.19 borderline and any downstream
        # sizing shrinkage takes it below — 68,605 rejections in
        # 64 min on snapshot v1.5.257-260529-170937. Profiles can
        # lower (e.g. 3.5) to unblock rung-1, or raise to preserve
        # tighter rebate-economics.
        min_notional_usd = float(
            getattr(self._settings, "min_venue_notional_usd", 5.0) or 5.0
        )

        # OKX's lot size in contracts is a clean power-of-10 in most
        # listings (e.g. 1, 0.1, 0.01). Derive sz_decimals from the
        # base-units step.
        sz_decimals = 0
        s = f"{size_step:.10f}".rstrip("0")
        if "." in s:
            sz_decimals = len(s.split(".")[1])

        spec = SymbolSpec(
            price_tick=price_tick,
            size_step=size_step,
            min_size=min_size if min_size > 0 else size_step,
            min_notional_usd=min_notional_usd,
            sz_decimals=sz_decimals,
            source="okx_meta",
        )
        # 1.3.123 Phase 4a v2: extract numeric instIdCode. OKX returns
        # this as a JSON integer (e.g. 120850 for SUI-USDT-SWAP), but
        # coerce defensively in case the API ever emits it as a string.
        inst_id_code: Optional[int] = None
        raw_code = row.get("instIdCode")
        if raw_code is not None:
            try:
                inst_id_code = int(raw_code)
            except (TypeError, ValueError):
                logger.warning(
                    "okx_inst_id_code_parse_failed symbol=%s raw=%r",
                    self._symbol,
                    raw_code,
                )
        return spec, contract_value, inst_id_code

    # ------------------------------------------------------------------
    # Quote-unit conversion helpers
    # ------------------------------------------------------------------

    def _base_to_contracts(self, base_qty: float) -> float:
        """Convert a base-asset quantity to OKX-contract count."""
        if self._contract_value <= 0:
            return base_qty
        return base_qty / self._contract_value

    def _contracts_to_base(self, contracts: float) -> float:
        """Convert OKX-contract count to a base-asset quantity."""
        return contracts * self._contract_value

    # ------------------------------------------------------------------
    # Market / account reads
    # ------------------------------------------------------------------

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        sym = _normalize_okx_symbol(symbol)
        resp = self._request(
            "books",
            "GET",
            "/api/v5/market/books",
            params={"instId": sym, "sz": 1},
            signed=False,
        )
        if str(resp.get("code")) != "0":
            return BestBidAsk(
                symbol=sym,
                best_bid=None,
                best_ask=None,
                mid_price=None,
                spread_bps=None,
            )
        rows = resp.get("data") or []
        row = rows[0] if isinstance(rows, list) and rows else {}
        # OKX books rows are {"bids": [["px","sz","liq","numOrders"], ...], "asks": [...]}.
        bids = row.get("bids") or []
        asks = row.get("asks") or []
        bid = _coerce_float(bids[0][0]) if bids else None
        ask = _coerce_float(asks[0][0]) if asks else None
        bid_sz_contracts = _coerce_float(bids[0][1]) if bids else None
        ask_sz_contracts = _coerce_float(asks[0][1]) if asks else None
        bid_sz = (
            self._contracts_to_base(bid_sz_contracts)
            if bid_sz_contracts is not None
            else None
        )
        ask_sz = (
            self._contracts_to_base(ask_sz_contracts)
            if ask_sz_contracts is not None
            else None
        )
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
        """``GET /api/v5/account/positions?instId=...`` returns a list of
        position rows. In net-mode (which we require), a symbol has at
        most one row.
        """
        del address  # OKX uses API key auth, not address
        sym = _normalize_okx_symbol(symbol)
        resp = self._request(
            "positions",
            "GET",
            "/api/v5/account/positions",
            params={"instId": sym},
            signed=True,
        )
        if str(resp.get("code")) != "0":
            raise RuntimeError(
                f"fetch_position failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        pos_qty = 0.0
        entry = 0.0
        mark = 0.0
        unreal = 0.0
        notional = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("instId") != sym:
                continue
            # OKX ``pos`` is signed contract count in net-mode.
            pos_contracts = _coerce_float(row.get("pos"))
            pos_qty = self._contracts_to_base(pos_contracts)
            entry = _coerce_float(row.get("avgPx"))
            mark = _coerce_float(row.get("markPx"))
            unreal = _coerce_float(row.get("upl"))
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
        """``GET /api/v5/account/balance`` returns aggregate balances.
        OKX's unified-account model returns multiple currency rows; we
        aggregate USDT-equivalent for now.
        """
        del address
        resp = self._request(
            "account",
            "GET",
            "/api/v5/account/balance",
            signed=True,
        )
        if str(resp.get("code")) != "0":
            raise RuntimeError(
                f"fetch_account_snapshot failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        # OKX returns one row per account with totals at the top:
        #   row["totalEq"]   -- total equity in USD (incl. unrealised)
        #   row["adjEq"]     -- equity available for new margin
        #   row["details"]   -- per-currency breakdown
        if not rows or not isinstance(rows[0], dict):
            return AccountSnapshot()
        row = rows[0]
        total_eq = _coerce_float(row.get("totalEq"))
        avail = _coerce_float(row.get("adjEq"))
        # ``cash_usd`` analog: sum of cashBal across USDT-class details.
        cash = 0.0
        details = row.get("details") or []
        for d in details:
            if not isinstance(d, dict):
                continue
            ccy = str(d.get("ccy") or "").upper()
            if ccy in ("USDT", "USDC", "USD"):
                cash += _coerce_float(d.get("cashBal"))
        return AccountSnapshot(
            equity_usd=total_eq if total_eq > 0 else None,
            cash_usd=cash if cash > 0 else None,
            withdrawable_usd=avail if avail > 0 else None,
        )

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]:
        del address
        resp = self._request(
            "open_orders",
            "GET",
            "/api/v5/trade/orders-pending",
            params={"instType": "SWAP", "instId": self._symbol},
            signed=True,
        )
        if str(resp.get("code")) != "0":
            # 2026-05-13 Codex bug review CRITICAL #1 — RAISE rather
            # than return []. The original comment below was correct
            # about the danger; the implementation just didn't act on
            # it. Returning [] would mean "the bot has no open
            # orders" to the reconcile path, which is the same shape
            # as a healthy account with nothing resting. Operationally
            # that turned cancel-all into a silent no-op on auth/
            # rate-limit/exchange-side errors and could let reconcile
            # arm desync handling against live venue orders. Now
            # callers (``_sync_open_orders_impl`` /
            # ``cancel_all_orders_for_symbol``) take their existing
            # exception branches (rate-limited / error) instead.
            code = str(resp.get("code"))
            msg = str(resp.get("msg") or "")
            logger.warning(
                "okx_open_orders_failed code=%s msg=%s -- raising OkxApiError",
                code,
                msg[:200],
            )
            raise OkxApiError(code, msg, endpoint="/trade/orders-pending")
        rows = resp.get("data") or []
        out: list[HLOpenOrderRaw] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("instId") != self._symbol:
                continue
            try:
                oid = int(row.get("ordId") or 0)
            except (TypeError, ValueError):
                continue
            if oid == 0:
                continue
            side_str = str(row.get("side") or "").lower()
            side = Side.BUY if side_str == "buy" else Side.SELL
            ts_raw = row.get("uTime") or row.get("cTime") or 0
            try:
                ts_ms = int(ts_raw)
            except (TypeError, ValueError):
                ts_ms = 0
            sz_contracts = _coerce_float(row.get("sz"))
            sz_base = self._contracts_to_base(sz_contracts)
            out.append(
                HLOpenOrderRaw(
                    oid=oid,
                    coin=self._symbol,
                    side=side,
                    limit_px=_coerce_float(row.get("px")),
                    sz=sz_base,
                    timestamp=ts_ms,
                    cloid=str(row.get("clOrdId") or "") or None,
                )
            )
        return out

    def fetch_recent_fills_raw(
        self, address: str, symbol: str
    ) -> list[FillRaw]:
        """Fetch recent fills via OKX REST, paginated.

        2026-05-16 Codex review #2 fix: previously this method fetched
        a single 100-row page. After a private-WS gap or reconnect, any
        burst larger than 100 fills permanently dropped the older fills
        from local ingestion. Position truth was rebuilt via a separate
        REST endpoint, but trade-rate counters, fill-burst detection,
        recent-fill cooldowns, session fill counters, and markout
        attribution all silently undercounted from that point on.

        Now walks the OKX ``/trade/fills`` endpoint via its ``after``
        cursor (older-than-billId semantics) until either:

          * a partial page (<100 rows) is returned — end of history;
          * the configured ``okx_fills_rest_max_pages`` cap is hit;
          * a page is malformed (no usable billId on the oldest row).

        Storage dedupes by ``fill_id`` so overlapping rows from
        consecutive REST polls are idempotent — re-ingestion is safe.
        """
        del address
        sym = _normalize_okx_symbol(symbol)
        max_pages = int(
            getattr(self._settings, "okx_fills_rest_max_pages", 10) or 10
        )
        out: list[HLFillRaw] = []
        cursor_billid: Optional[str] = None
        for page in range(max_pages):
            params: dict[str, Any] = {
                "instType": "SWAP",
                "instId": sym,
                "limit": 100,
            }
            if cursor_billid is not None:
                # OKX ``after`` returns fills with billId STRICTLY LESS
                # than the supplied value (i.e. older). The cursor we
                # pass is the OLDEST billId from the previous page.
                params["after"] = cursor_billid
            resp = self._request(
                "fills",
                "GET",
                "/api/v5/trade/fills",
                params=params,
                signed=True,
            )
            if str(resp.get("code")) != "0":
                # 2026-05-13 Codex bug review MED #3 — RAISE rather than
                # return []. Previously the silent-[] return let the
                # account-refresh path log "fills_returned=0" as if the
                # bot had cleanly observed an empty fills page; on
                # actual venue errors that meant lost realized-PnL /
                # markout / toxicity attribution while the bot appeared
                # healthy. ``refresh_account_only`` catches this in its
                # own (now two-stage) try block (Bug 2 fix) so the
                # position+account refresh still succeeds.
                code = str(resp.get("code"))
                msg = str(resp.get("msg") or "")
                logger.warning(
                    "okx_fills_failed page=%d code=%s msg=%s -- raising OkxApiError",
                    page,
                    code,
                    msg[:200],
                )
                raise OkxApiError(code, msg, endpoint="/trade/fills")
            rows = resp.get("data") or []
            if not isinstance(rows, list):
                break
            oldest_billid_this_page: Optional[str] = None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if row.get("instId") != sym:
                    continue
                try:
                    trade_id_raw = row.get("tradeId") or row.get("billId") or "0"
                    order_id = int(row.get("ordId") or 0)
                except (TypeError, ValueError):
                    continue
                side_str = str(row.get("side") or "").lower()
                side = Side.BUY if side_str == "buy" else Side.SELL
                try:
                    time_ms = int(row.get("ts") or 0)
                except (TypeError, ValueError):
                    time_ms = 0
                sz_contracts = _coerce_float(row.get("fillSz"))
                sz_base = self._contracts_to_base(sz_contracts)
                # OKX V5 ``fee`` field on /trade/fills: POSITIVE = rebate
                # received by user; NEGATIVE = fee paid to platform. The bot's
                # canonical convention is the OPPOSITE (positive=cost,
                # negative=income), so we negate here. See HLFillRaw docstring
                # for the full convention spec. BUG-018 fix 2026-05-05 --
                # previously this used abs(), silently dropping the sign and
                # making rebates indistinguishable from fees in PnL math.
                okx_fee_raw = _coerce_float(row.get("fee"))
                fee_bot_convention = -okx_fee_raw
                out.append(
                    HLFillRaw(
                        fill_id=str(trade_id_raw),
                        oid=order_id,
                        coin=sym,
                        side=side,
                        px=_coerce_float(row.get("fillPx")),
                        sz=sz_base,
                        fee=fee_bot_convention,
                        time_ms=time_ms,
                        closed_pnl=_coerce_float(row.get("fillPnl")),
                        raw=dict(row),
                    )
                )
                # Track the oldest billId we've seen on this page; it
                # becomes the cursor for the next page.
                bid = row.get("billId")
                if bid:
                    oldest_billid_this_page = str(bid)
            # Stop conditions:
            #   * Page short (<100 rows) — venue has no more older
            #     fills than this page. End of history.
            #   * No usable billId — cursor would be lost; bail
            #     rather than risk an infinite loop.
            if len(rows) < 100:
                break
            if oldest_billid_this_page is None:
                logger.warning(
                    "okx_fills_pagination_lost_cursor page=%d rows=%d "
                    "-- stopping early; older fills may be missed",
                    page,
                    len(rows),
                )
                break
            cursor_billid = oldest_billid_this_page
        else:
            # Loop exited via ``range`` exhaustion rather than a
            # ``break``. The venue had MORE fills than our cap —
            # operator-tunable via OKX_FILLS_REST_MAX_PAGES.
            logger.warning(
                "okx_fills_pagination_cap_hit max_pages=%d collected=%d "
                "-- older fills may be missed; raise OKX_FILLS_REST_MAX_PAGES",
                max_pages,
                len(out),
            )
        return out

    # ------------------------------------------------------------------
    # Account diagnostics (used by scripts/okx_preflight.py)
    # ------------------------------------------------------------------

    def fetch_account_config(self) -> dict[str, Any]:
        """``GET /api/v5/account/config`` returns the operator-level
        account configuration. Most relevant fields:

        * ``posMode``     -- ``net_mode`` | ``long_short_mode``
                             (the bot REQUIRES net_mode)
        * ``acctLv``      -- account-tier code (1=spot, 2=futures+margin,
                             3=multi-currency margin, 4=portfolio margin)
        * ``ip``          -- IP whitelist on the API key (CSV string)
        * ``mainUid``     -- master account UID (sub-accounts only)

        Returned dict is the full row from OKX. Not part of the
        adapter Protocol -- preflight scripts use it.
        """
        resp = self._request(
            "account_config",
            "GET",
            "/api/v5/account/config",
            signed=True,
        )
        if str(resp.get("code")) != "0":
            raise RuntimeError(
                f"fetch_account_config failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    def fetch_leverage_info(
        self, symbol: str, mgn_mode: str = "cross"
    ) -> dict[str, Any]:
        """``GET /api/v5/account/leverage-info`` returns the
        configured leverage for ``(symbol, mgnMode)``. Both isolated
        and cross modes have separate rows; we query whichever is
        relevant. Returns the first row of the response, or empty
        dict on failure / missing data.

        Used by the bot at startup to surface leverage + margin
        mode in the dashboard's position panel; never blocks
        trading on failure (caller should swallow exceptions).

        Portfolio-margin accounts return code ``59111`` ("Leverage
        query isn't supported in portfolio margin account mode") —
        treat that as a clean "not applicable" and return an empty
        dict so the caller can leave the field as None without a
        traceback in startup logs.
        """
        sym = _normalize_okx_symbol(symbol)
        resp = self._request(
            "leverage_info",
            "GET",
            "/api/v5/account/leverage-info",
            params={"instId": sym, "mgnMode": mgn_mode},
            signed=True,
        )
        code = str(resp.get("code"))
        if code == "59111":
            return {}
        if code != "0":
            raise RuntimeError(
                f"fetch_leverage_info failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    def fetch_trade_fee(self, symbol: str) -> dict[str, Any]:
        """``GET /api/v5/account/trade-fee`` returns the EFFECTIVE
        fee schedule for THIS account. Differs from the public fee
        tier when the operator has an MM-tier upgrade, VIP discount,
        or a sub-account override applied.

        Per OKX docs:

            * ``instType`` (required): SPOT / MARGIN / SWAP / FUTURES / OPTION
            * ``instId``   (only valid for SPOT)
            * ``uly`` / ``instFamily`` (optional, derivatives only)

        For SWAP we pass ``instType=SWAP`` plus ``instFamily`` derived
        from the symbol (e.g. ``SUI-USDT-SWAP`` -> ``SUI-USDT``). This
        returns per-family fees, which is the closest analog to the
        per-symbol view we want.

        Returns the first data row. Fields:
          * ``maker`` / ``maker_U``   -- maker rate (USDT-margin variant)
          * ``taker`` / ``taker_U``   -- taker rate
          * ``level``                 -- "Lv1" / "VIP1" / "MM1" / ...

        Rates are strings like ``"-0.00002"`` for -0.002%.
        """
        sym = _normalize_okx_symbol(symbol)
        # Derive instFamily by stripping the "-SWAP" suffix
        # (e.g. "SUI-USDT-SWAP" -> "SUI-USDT").
        family = sym.rsplit("-SWAP", 1)[0] if sym.endswith("-SWAP") else sym
        resp = self._request(
            "trade_fee",
            "GET",
            "/api/v5/account/trade-fee",
            params={"instType": "SWAP", "instFamily": family},
            signed=True,
        )
        if str(resp.get("code")) != "0":
            raise RuntimeError(
                f"fetch_trade_fee failed: code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            )
        rows = resp.get("data") or []
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    def set_position_mode(self, mode: str) -> dict[str, Any]:
        """``POST /api/v5/account/set-position-mode`` -- WRITE operation.
        The bot ITSELF never calls this; only the preflight script does,
        gated behind explicit operator opt-in.

        ``mode`` is OKX's enum string: ``net_mode`` (the bot expects
        this) or ``long_short_mode`` (default for new accounts; bot
        cannot work with it).
        """
        if mode not in ("net_mode", "long_short_mode"):
            raise ValueError(
                f"set_position_mode: invalid mode {mode!r} "
                f"(expected 'net_mode' | 'long_short_mode')"
            )
        return self._request(
            "set_position_mode",
            "POST",
            "/api/v5/account/set-position-mode",
            body={"posMode": mode},
            signed=True,
        )

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def _place_order(
        self,
        *,
        symbol: str,
        is_buy: bool,
        sz_base: float,
        limit_px: float,
        post_only: bool,
        reduce_only: bool,
        ioc: bool,
        order_type: str,
        client_order_id: Optional[str],
    ) -> dict[str, Any]:
        # ------------------------------------------------------------
        # HARD PRE-FLIGHT NOTIONAL CAP
        # ------------------------------------------------------------
        # Last-resort safety net: regardless of WHATEVER pathological
        # size came in from upstream sizing logic, refuse to send a
        # notional > MAX_ORDER_NOTIONAL_USD * MAX_ORDER_NOTIONAL_HARD_MULTIPLIER
        # (default 2x). Catches calculation bugs / unit confusion /
        # corrupted state that would otherwise let a 2000-SUI order
        # leave the process. Refusal raises -- caller's exception
        # handler treats it as a place failure and the bot will
        # retry with fresh inputs.
        #
        # Incident this guards against: 2026-05-06 SUI-USDT-SWAP, fills
        # in the 2000-2700 SUI range vs configured $20 cap; root cause
        # was upstream sizing path producing huge sz_base values that
        # the adapter passed through verbatim.
        try:
            notional_usd = abs(float(sz_base) * float(limit_px))
        except (TypeError, ValueError):
            notional_usd = 0.0
        max_order = float(getattr(self._settings, "max_order_notional_usd", 0.0) or 0.0)
        hard_mult = float(
            getattr(self._settings, "max_order_notional_hard_multiplier", 2.0)
            or 2.0
        )
        hard_cap = max_order * hard_mult
        if max_order > 0 and notional_usd > hard_cap:
            msg = (
                f"OKX_HARD_NOTIONAL_CAP refused side={'BUY' if is_buy else 'SELL'} "
                f"sz_base={sz_base:.6f} px={limit_px:.8f} "
                f"notional_usd={notional_usd:.2f} > hard_cap={hard_cap:.2f} "
                f"(MAX_ORDER_NOTIONAL_USD={max_order:.2f} × {hard_mult:.2f}x)"
            )
            logger.critical(msg)
            raise RuntimeError(msg)
        sym = _normalize_okx_symbol(symbol)
        cloid = (
            (client_order_id or "").strip()
            or make_deterministic_okx_client_order_id(
                sym,
                Side.BUY if is_buy else Side.SELL,
                "manual",
                limit_px,
                sz_base,
            )
        )
        # OKX expects size in CONTRACTS, not base units.
        sz_contracts = self._base_to_contracts(sz_base)
        body: dict[str, Any] = {
            "instId": sym,
            "tdMode": "cross",  # margin mode; could be made configurable
            "side": "buy" if is_buy else "sell",
            "ordType": order_type,
            "sz": f"{sz_contracts:.10g}",
            "clOrdId": cloid,
        }
        if order_type != _OKX_ORDER_TYPE_MARKET:
            body["px"] = f"{limit_px:.10g}"
        if reduce_only:
            body["reduceOnly"] = True
        # Note: OKX doesn't use a separate timeInForce field; the
        # ordType captures it (post_only / ioc / limit / market).
        #
        # 1.3.111 Phase 4b: WS-first dispatch via op=``order`` when
        # ``OKX_ACTION_WS_PLACE_ENABLED`` is True (Stage 4 activation
        # lever). HTTP fallback engages automatically on any
        # ``OkxActionWsError`` when ``action_http_fallback_enabled``
        # is True (default). Hard pre-flight notional cap above runs
        # before either transport, so the safety guarantee is
        # transport-agnostic. Response envelope is identical between
        # transports — Phase 1a's sync ordId binding in
        # ``OrderManager`` (execution.py:4548) reads ``data[0].ordId``
        # via ``interpret_okx_place_response`` and works for both.
        def _http() -> dict[str, Any]:
            return self._request(
                "create_order",
                "POST",
                "/api/v5/trade/order",
                body=body,
                signed=True,
            )

        # 1.3.123 Phase 4a v2: same instIdCode injection as the cancel
        # path. HTTP /api/v5/trade/order ignores the extra field; the
        # colo trade-WS endpoint requires it.
        return self._try_ws_place(
            args=[self._decorate_with_inst_id_code(body)],
            http_fallback=_http,
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
            raise RuntimeError("OKX trading credentials missing")
        return self._place_order(
            symbol=symbol,
            is_buy=is_buy,
            sz_base=sz,
            limit_px=limit_px,
            post_only=True,
            reduce_only=reduce_only,
            ioc=False,
            order_type=_OKX_ORDER_TYPE_POST_ONLY,
            client_order_id=client_order_id,
        )

    def place_ioc_reduce_only(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
    ) -> dict[str, Any]:
        """v1.4.172 (Phase 4D wiring): IOC limit reduce-only.

        Used by the soft-flatten phase-ladder dispatcher in phases 2-3
        (cross-tick IOC ladders inserted between the legacy post-only
        chase and the terminal ``market_close``). Delegates to
        ``_place_order`` with OKX's native ``ordType="ioc"`` — fills
        what it can at-or-better-than ``limit_px`` and cancels the
        remainder immediately.

        Reduce-only is always set: SF can never grow exposure. No
        client-order-id — IOCs are fire-and-forget; no separate WO row
        is created in local state, and the resulting fills are stamped
        with the active ``soft_flatten_event_id`` at ingestion time
        (see ``app/fill_ingestion.py``).

        Errors are propagated; the caller (bot.py) logs + escalates
        the phase ladder if the placement fails.
        """
        if not self.has_write_access():
            raise RuntimeError("OKX trading credentials missing")
        return self._place_order(
            symbol=symbol,
            is_buy=is_buy,
            sz_base=sz,
            limit_px=limit_px,
            post_only=False,
            reduce_only=True,
            ioc=True,
            order_type=_OKX_ORDER_TYPE_IOC,
            client_order_id=None,
        )

    # ------------------------------------------------------------------
    # Batch place (v1.4.4)
    # ------------------------------------------------------------------

    def batch_place_post_only_limit(
        self,
        symbol: str,
        orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """``POST /api/v5/trade/batch-orders`` — place multiple post-only
        orders in one HTTP/WS round-trip.

        ``orders`` is a list of dicts. Each dict supports:
          * ``is_buy``           (bool, required)
          * ``sz``               (float, base-asset units, required)
          * ``limit_px``         (float, required)
          * ``client_order_id``  (str, optional — generated when absent)
          * ``reduce_only``      (bool, default False)

        OKX accepts up to 20 orders per request and the batch endpoint
        has its OWN rate-limit budget (300/2s) separate from the
        single-order /trade/order endpoint (60/2s). For a multi-rung
        MM ladder this means the bot taps a much larger pool when
        repricing — which is the entire reason this method exists.

        Returns the OKX response envelope. The caller MUST parse
        ``data[]`` via :func:`interpret_okx_place_batch_response` to
        attribute success / failure per submitted order, including the
        critical row-level rate-limit reclassification (v1.4.2 had a
        bug here where row-level rate-limit hits were treated as
        permanent exchange_rejected rather than transport-retryable).

        Per-order hard pre-flight notional cap runs INSIDE the loop —
        catches the same calculation-bug-or-corrupted-state class of
        failure that the single-place version guards against. If any
        single order exceeds the cap, the whole batch is refused
        (RuntimeError); we do not partial-submit.

        1.4.4 always-batch routing: caller code routes every place
        through this method, including 1-element payloads (batch
        endpoint accepts size-1 lists). The shape of the response is
        identical to single-order — top envelope + data[] of size 1 —
        so reconciliation logic doesn't need a separate path.
        """
        if not self.has_write_access():
            raise RuntimeError("OKX trading credentials missing")
        if not orders:
            return {"code": "0", "msg": "", "data": []}
        if len(orders) > 20:
            raise ValueError(
                f"batch_place_post_only_limit: OKX caps batch at 20 "
                f"orders/request, got {len(orders)}"
            )

        sym = _normalize_okx_symbol(symbol)
        max_order = float(
            getattr(self._settings, "max_order_notional_usd", 0.0) or 0.0
        )
        hard_mult = float(
            getattr(
                self._settings, "max_order_notional_hard_multiplier", 2.0
            )
            or 2.0
        )
        hard_cap = max_order * hard_mult

        body_rows: list[dict[str, Any]] = []
        for spec in orders:
            is_buy = bool(spec.get("is_buy"))
            try:
                sz_base = float(spec.get("sz") or 0.0)
                limit_px = float(spec.get("limit_px") or 0.0)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"batch_place: bad sz/limit_px spec={spec!r}"
                ) from exc
            client_order_id = spec.get("client_order_id")
            reduce_only = bool(spec.get("reduce_only", False))

            # Same hard pre-flight notional cap as the single-order
            # path. Refuses the WHOLE batch on any over-cap order
            # rather than partial-submitting — surfacing the bug
            # loudly is safer than silently dropping bad rungs.
            try:
                notional_usd = abs(sz_base * limit_px)
            except (TypeError, ValueError):
                notional_usd = 0.0
            if max_order > 0 and notional_usd > hard_cap:
                msg = (
                    f"OKX_HARD_NOTIONAL_CAP_BATCH refused "
                    f"side={'BUY' if is_buy else 'SELL'} "
                    f"sz_base={sz_base:.6f} px={limit_px:.8f} "
                    f"notional_usd={notional_usd:.2f} > hard_cap={hard_cap:.2f} "
                    f"(MAX_ORDER_NOTIONAL_USD={max_order:.2f} × "
                    f"{hard_mult:.2f}x) batch_size={len(orders)}"
                )
                logger.critical(msg)
                raise RuntimeError(msg)

            cloid = (
                (client_order_id or "").strip()
                or make_deterministic_okx_client_order_id(
                    sym,
                    Side.BUY if is_buy else Side.SELL,
                    "batch",
                    limit_px,
                    sz_base,
                )
            )
            sz_contracts = self._base_to_contracts(sz_base)
            row: dict[str, Any] = {
                "instId": sym,
                "tdMode": "cross",
                "side": "buy" if is_buy else "sell",
                "ordType": _OKX_ORDER_TYPE_POST_ONLY,
                "sz": f"{sz_contracts:.10g}",
                "px": f"{limit_px:.10g}",
                "clOrdId": cloid,
            }
            if reduce_only:
                row["reduceOnly"] = True
            body_rows.append(row)

        def _http() -> dict[str, Any]:
            return self._request(
                "batch_place",
                "POST",
                "/api/v5/trade/batch-orders",
                body=body_rows,
                signed=True,
            )

        # WS op for multi-place is ``batch-orders`` per OKX V5 trade-WS
        # docs. Same instIdCode decoration as the cancel-batch path —
        # required on colo, ignored on standard endpoints.
        ws_rows = [self._decorate_with_inst_id_code(r) for r in body_rows]
        return self._try_ws_place_batch(
            args=ws_rows,
            http_fallback=_http,
        )

    # ------------------------------------------------------------------
    # Amend (v1.4.4 Pass 2)
    # ------------------------------------------------------------------

    def amend_batch_orders(
        self,
        symbol: str,
        amends: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """``POST /api/v5/trade/amend-batch-orders`` — modify price/size
        of up to 20 existing orders in one HTTP/WS round-trip.

        ``amends`` is a list of dicts. Each dict supports:
          * ``ord_id`` (str) OR ``client_order_id`` (str) — one is
            required to identify the target order
          * ``new_px`` (float, optional) — new limit price
          * ``new_sz`` (float, base-asset units, optional) — new size
          * ``req_id`` (str, optional) — caller's request id, echoed
            back per row for correlation

        OKX semantics worth knowing:
          * AMEND PRESERVES THE VENUE ORDID — queue position is NOT
            refreshed (the order keeps its place in the queue). This
            is the win vs cancel-then-place: no queue-loss penalty
            for repricing.
          * If the order has partially filled, ``new_sz`` updates the
            REMAINING size (not the original). If ``new_sz`` is below
            the already-filled amount OKX rejects with sCode 51016.
          * post_only orders that would cross after the amend get
            rejected with sCode 51604 — same as place.
          * The amend rate-limit pool is /trade/amend-batch-orders
            specific (300/2s on standard tier), distinct from both
            /trade/order (60/2s) and /trade/batch-orders (300/2s).

        Returns the OKX response envelope. Per-row parsing happens in
        ``interpret_okx_amend_batch_response`` — same shape as
        ``interpret_okx_place_batch_response`` so callers can reuse
        their per-row reconciliation logic.
        """
        if not self.has_write_access():
            raise RuntimeError("OKX trading credentials missing")
        if not amends:
            return {"code": "0", "msg": "", "data": []}
        if len(amends) > 20:
            raise ValueError(
                f"amend_batch_orders: OKX caps batch at 20 amends/req, "
                f"got {len(amends)}"
            )

        sym = _normalize_okx_symbol(symbol)
        body_rows: list[dict[str, Any]] = []
        for spec in amends:
            ord_id = spec.get("ord_id")
            client_order_id = spec.get("client_order_id")
            if not ord_id and not client_order_id:
                raise RuntimeError(
                    f"amend_batch: missing ord_id and client_order_id in {spec!r}"
                )
            row: dict[str, Any] = {"instId": sym}
            if ord_id:
                row["ordId"] = str(ord_id)
            if client_order_id:
                row["clOrdId"] = str(client_order_id)
            new_px = spec.get("new_px")
            new_sz = spec.get("new_sz")
            if new_px is not None:
                row["newPx"] = f"{float(new_px):.10g}"
            if new_sz is not None:
                # OKX expects contract count.
                new_sz_contracts = self._base_to_contracts(float(new_sz))
                row["newSz"] = f"{new_sz_contracts:.10g}"
            req_id = spec.get("req_id")
            if req_id:
                row["reqId"] = str(req_id)
            body_rows.append(row)

        def _http() -> dict[str, Any]:
            return self._request(
                "amend_batch",
                "POST",
                "/api/v5/trade/amend-batch-orders",
                body=body_rows,
                signed=True,
            )

        # WS op for batch-amend per OKX V5 docs is ``batch-amend-orders``.
        ws_rows = [self._decorate_with_inst_id_code(r) for r in body_rows]
        return self._try_ws_amend_batch(
            args=ws_rows,
            http_fallback=_http,
        )

    def _try_ws_amend_batch(
        self,
        args: list[dict[str, Any]],
        http_fallback: Any,
    ) -> dict[str, Any]:
        """Send a batch-amend via the action WS using
        op=``batch-amend-orders``. Mirror of ``_try_ws_place_batch``.

        Reuses the place-side WS feature flag
        (``okx_action_ws_place_enabled``) because amend is the
        place-side companion operation and the WS frame routing /
        latency profile are identical. No separate amend-WS knob.
        """
        ws = self._get_action_ws()
        if (
            ws is not None
            and bool(
                getattr(self._settings, "okx_action_ws_place_enabled", False)
            )
            and ws.is_connected()
        ):
            try:
                timeout_ms = int(
                    float(
                        getattr(
                            self._settings,
                            "okx_action_ws_request_timeout_seconds",
                            1.5,
                        )
                    )
                    * 1000
                )
                resp = ws.send_and_await(
                    "batch-amend-orders", args, timeout_ms=timeout_ms
                )
                with self._lock:
                    self._place_ws_send_count += 1
                # v1.4.20 Phase 1: tag with AMEND_BATCH pool.
                self._rate_window.record(pool=RateLimitPool.AMEND_BATCH)
                self.last_exchange_transport_mode = "ws"
                return resp
            except OkxActionWsError as exc:
                with self._lock:
                    self._place_ws_fallback_count += 1
                fallback_enabled = bool(
                    getattr(
                        self._settings,
                        "action_http_fallback_enabled",
                        True,
                    )
                )
                if not fallback_enabled:
                    raise
                logger.warning(
                    "okx_action_ws_amend_batch_fallback_to_http err=%s",
                    str(exc)[:200],
                )
        self.last_exchange_transport_mode = "http"
        return http_fallback()

    # ------------------------------------------------------------------
    # 1.3.110 Phase 4a: WS-first cancel dispatch helpers
    # ------------------------------------------------------------------

    def _get_action_ws(self) -> Optional[OkxActionWs]:
        """Lazily allocate the action-WS coordinator and start its
        daemon thread on first use. Returns None when the WS path is
        disabled or write credentials are missing — callers must
        treat None as "use HTTP."

        Lazy construction avoids spawning the WS thread for processes
        that hold an OkxClient purely for read-only purposes (symbol-
        spec bootstrap, dashboards). ``okx_action_ws_cancel_enabled``
        is the per-path opt-in; ``action_ws_enabled`` is the umbrella
        flag (matches the legacy HL convention)."""
        if not bool(getattr(self._settings, "action_ws_enabled", True)):
            return None
        if not bool(
            getattr(self._settings, "okx_action_ws_cancel_enabled", False)
        ) and not bool(
            getattr(self._settings, "okx_action_ws_place_enabled", False)
        ):
            return None
        if not self.has_write_access():
            return None
        with self._action_ws_lock:
            if self._action_ws is None:
                self._action_ws = OkxActionWs(self._settings)
                self._action_ws.start()
            return self._action_ws

    def _try_ws_cancel(
        self,
        op: str,
        args: list[dict[str, Any]],
        http_fallback: Any,
    ) -> dict[str, Any]:
        """Send a cancel via the action WS; on any failure fall back to
        HTTP if ``action_http_fallback_enabled`` is True. Stamps
        ``last_exchange_transport_mode`` per call so execution.py /
        postmortem tooling can attribute the transport for every
        cancel intent.

        Argument shapes match OKX V5 trade-WS docs:
          * op="cancel-order", args=[{instId, ordId|clOrdId}]
          * op="batch-cancel-orders", args=[{instId, ordId|clOrdId}, ...]
        """
        ws = self._get_action_ws()
        if (
            ws is not None
            and bool(
                getattr(self._settings, "okx_action_ws_cancel_enabled", False)
            )
            and ws.is_connected()
        ):
            try:
                timeout_ms = int(
                    float(
                        getattr(
                            self._settings,
                            "okx_action_ws_request_timeout_seconds",
                            1.5,
                        )
                    )
                    * 1000
                )
                resp = ws.send_and_await(op, args, timeout_ms=timeout_ms)
                with self._lock:
                    self._cancel_ws_send_count += 1
                # 1.4.4: WS-action sends consume the same OKX rate-limit
                # budget as REST. Record on success so the dashboard
                # gauge reflects total venue pressure (REST + WS).
                # v1.4.20 Phase 1: tag with the pool inferred from
                # the WS op (``cancel-order`` → CANCEL_SINGLE;
                # ``batch-cancel-orders`` → CANCEL_BATCH).
                self._rate_window.record(pool=pool_for_endpoint(op=op))
                self.last_exchange_transport_mode = "ws"
                return resp
            except OkxActionWsError as exc:
                with self._lock:
                    self._cancel_ws_fallback_count += 1
                fallback_enabled = bool(
                    getattr(self._settings, "action_http_fallback_enabled", True)
                )
                if not fallback_enabled:
                    raise
                logger.warning(
                    "okx_action_ws_cancel_fallback_to_http op=%s err=%s",
                    op,
                    str(exc)[:200],
                )
                # fall through to HTTP
        # HTTP path
        self.last_exchange_transport_mode = "http"
        return http_fallback()

    def _try_ws_place_batch(
        self,
        args: list[dict[str, Any]],
        http_fallback: Any,
    ) -> dict[str, Any]:
        """Send a batch-place via the action WS using op=``batch-orders``;
        on failure fall back to HTTP if ``action_http_fallback_enabled``
        is True. Mirror of :meth:`_try_ws_place` for the multi-order op.

        Reuses the same place-side feature flag
        (``okx_action_ws_place_enabled``) — when WS-place is opted in,
        BOTH single-place and batch-place use the WS transport. There
        is no separate ``..._batch_enabled`` knob because the WS frame
        shape is identical apart from ``args[]`` length, and OKX
        treats them as the same op category for routing.

        Response envelope shape matches the REST batch endpoint
        (top-level ``code/msg/data`` with per-row
        ``ordId/sCode/sMsg``), so :func:`interpret_okx_place_batch_response`
        works transport-agnostically. Counters reuse the existing
        single-place WS counters — operator visibility is "WS place
        traffic in total", batch is a multiplier on row count.
        """
        ws = self._get_action_ws()
        if (
            ws is not None
            and bool(
                getattr(self._settings, "okx_action_ws_place_enabled", False)
            )
            and ws.is_connected()
        ):
            try:
                timeout_ms = int(
                    float(
                        getattr(
                            self._settings,
                            "okx_action_ws_request_timeout_seconds",
                            1.5,
                        )
                    )
                    * 1000
                )
                resp = ws.send_and_await(
                    "batch-orders", args, timeout_ms=timeout_ms
                )
                with self._lock:
                    self._place_ws_send_count += 1
                # v1.4.20 Phase 1: tag with PLACE_BATCH pool.
                self._rate_window.record(pool=RateLimitPool.PLACE_BATCH)
                self.last_exchange_transport_mode = "ws"
                return resp
            except OkxActionWsError as exc:
                with self._lock:
                    self._place_ws_fallback_count += 1
                fallback_enabled = bool(
                    getattr(
                        self._settings,
                        "action_http_fallback_enabled",
                        True,
                    )
                )
                if not fallback_enabled:
                    raise
                logger.warning(
                    "okx_action_ws_place_batch_fallback_to_http err=%s",
                    str(exc)[:200],
                )
                # fall through to HTTP
        self.last_exchange_transport_mode = "http"
        return http_fallback()

    def _try_ws_place(
        self,
        args: list[dict[str, Any]],
        http_fallback: Any,
    ) -> dict[str, Any]:
        """Send a place via the action WS using op=``order``; on any
        failure fall back to HTTP if ``action_http_fallback_enabled``
        is True. Mirror of :meth:`_try_ws_cancel` with the
        place-specific flag + counters.

        The WS ``order`` op response carries the same envelope shape
        as ``POST /api/v5/trade/order`` (top-level ``code/msg/data``
        with per-row ``ordId/sCode/sMsg``), so the caller — and
        critically the Phase 1a sync ordId binding in
        :class:`OrderManager` — operates identically on both
        transports. No double-bind risk: there is exactly one
        bind site (``execution.py``), which reads the unified
        envelope via ``interpret_okx_place_response``.
        """
        ws = self._get_action_ws()
        if (
            ws is not None
            and bool(
                getattr(self._settings, "okx_action_ws_place_enabled", False)
            )
            and ws.is_connected()
        ):
            try:
                timeout_ms = int(
                    float(
                        getattr(
                            self._settings,
                            "okx_action_ws_request_timeout_seconds",
                            1.5,
                        )
                    )
                    * 1000
                )
                resp = ws.send_and_await("order", args, timeout_ms=timeout_ms)
                with self._lock:
                    self._place_ws_send_count += 1
                # v1.4.20 Phase 1: tag with PLACE_SINGLE pool.
                self._rate_window.record(pool=RateLimitPool.PLACE_SINGLE)
                self.last_exchange_transport_mode = "ws"
                return resp
            except OkxActionWsError as exc:
                with self._lock:
                    self._place_ws_fallback_count += 1
                fallback_enabled = bool(
                    getattr(self._settings, "action_http_fallback_enabled", True)
                )
                if not fallback_enabled:
                    raise
                logger.warning(
                    "okx_action_ws_place_fallback_to_http err=%s",
                    str(exc)[:200],
                )
                # fall through to HTTP
        self.last_exchange_transport_mode = "http"
        return http_fallback()

    # ------------------------------------------------------------------
    # Cancel ops (WS-first when enabled; HTTP fallback)
    # ------------------------------------------------------------------

    def _decorate_with_inst_id_code(
        self, ref: dict[str, Any]
    ) -> dict[str, Any]:
        """1.3.123 Phase 4a v2: inject the numeric ``instIdCode`` into a
        cancel/order WS-args ref dict.

        The OKX colo trade-WS endpoint requires ``instIdCode`` on every
        cancel frame; without it, the venue returns sCode 50014
        "Parameter instIdCode can not be empty". Standard
        (non-colo) endpoints ignore the extra field, so always
        injecting when known is safe across deployments.

        When the bootstrap didn't capture an inst_id_code (offline /
        failed fetch / older instrument with no field), the ref is
        returned unchanged — the standard endpoint still accepts it,
        and on colo the cancel will fail with the same 50014 it would
        have failed with before this helper existed. The
        ``_inst_id_code`` field is set at construction time so this
        check is a single attribute read (no lock).
        """
        code = self._inst_id_code
        if code is None:
            return ref
        return {**ref, "instIdCode": code}

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        sym = _normalize_okx_symbol(symbol)

        def _http() -> dict[str, Any]:
            return self._request(
                "cancel_order",
                "POST",
                "/api/v5/trade/cancel-order",
                body={"instId": sym, "ordId": str(oid)},
                signed=True,
            )

        return self._try_ws_cancel(
            op="cancel-order",
            args=[
                self._decorate_with_inst_id_code(
                    {"instId": sym, "ordId": str(oid)}
                )
            ],
            http_fallback=_http,
        )

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        sym = _normalize_okx_symbol(symbol)
        cloid = str(client_order_id or "").strip()
        if not cloid:
            return {"code": "50000", "msg": "missing_client_order_id", "data": []}

        def _http() -> dict[str, Any]:
            return self._request(
                "cancel_order_by_cloid",
                "POST",
                "/api/v5/trade/cancel-order",
                body={"instId": sym, "clOrdId": cloid},
                signed=True,
            )

        return self._try_ws_cancel(
            op="cancel-order",
            args=[
                self._decorate_with_inst_id_code(
                    {"instId": sym, "clOrdId": cloid}
                )
            ],
            http_fallback=_http,
        )

    def cancel_batch_orders(
        self, symbol: str, refs: list[dict[str, str]]
    ) -> dict[str, Any]:
        """``POST /api/v5/trade/cancel-batch-orders`` — cancel multiple
        orders in one HTTP call.

        ``refs`` is a list of `{instId, ordId}` or `{instId, clOrdId}`
        dicts. OKX accepts up to 20 entries per call. Caller is
        responsible for splitting larger batches.

        Returns the OKX response envelope. The ``data`` array carries
        one row per submitted ref with its own ``sCode`` /
        ``sMsg`` — callers must iterate to attribute success /
        failure per-intent (use ``interpret_okx_cancel_response_row``).

        1.4.0 cancel-prio Phase 1b: surfaced as a public method so the
        outbound dispatcher can opportunistically batch 2+ same-flush
        cancels into one HTTP call (saves ~5 ms of one RTT on full-
        reprice cycles where BUY + SELL cancel simultaneously).
        Previously this endpoint was only invoked internally by
        ``cancel_all_open_orders``.
        """
        sym = _normalize_okx_symbol(symbol)
        if not refs:
            return {"code": "0", "msg": "", "data": []}
        # Defensive: normalise instId on every ref.
        body_rows = [{**r, "instId": sym} for r in refs]

        def _http() -> dict[str, Any]:
            return self._request(
                "cancel_batch",
                "POST",
                "/api/v5/trade/cancel-batch-orders",
                body=body_rows,
                signed=True,
            )

        # OKX V5 WS op for multi-cancel is ``batch-cancel-orders``;
        # frame shape mirrors the HTTP body (list of refs in ``args``).
        # 1.3.123 Phase 4a v2: inject instIdCode on every row for colo.
        ws_rows = [self._decorate_with_inst_id_code(r) for r in body_rows]
        return self._try_ws_cancel(
            op="batch-cancel-orders",
            args=ws_rows,
            http_fallback=_http,
        )

    def cancel_all_open_orders(self, symbol: str) -> dict[str, Any]:
        """OKX does not expose a single ``cancel-all`` endpoint analogous
        to Binance's ``DELETE /fapi/v1/allOpenOrders``. We fetch open
        orders for the symbol and issue per-order cancels via the
        batch endpoint ``/api/v5/trade/cancel-batch-orders`` (max 20
        per call).
        """
        sym = _normalize_okx_symbol(symbol)
        opens = self.fetch_open_orders_raw("")
        if not opens:
            return {"code": "0", "msg": "", "data": []}
        # Batch cancels in chunks of 20.
        results: list[dict[str, Any]] = []
        chunk_size = 20
        for i in range(0, len(opens), chunk_size):
            chunk = opens[i : i + chunk_size]
            body_rows = [{"instId": sym, "ordId": str(o.oid)} for o in chunk]
            r = self._request(
                "cancel_batch",
                "POST",
                "/api/v5/trade/cancel-batch-orders",
                body=body_rows,  # OKX accepts a top-level list for batch endpoints
                signed=True,
            )
            results.append(r)
        # Surface the first response so callers can see the shape; each
        # attempt is also visible via ``rest_runtime_counters``.
        return results[0] if results else {"code": "0", "msg": "", "data": []}

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        del address
        cloid = str(client_order_id or "").strip()
        if not cloid:
            return {"code": "50000", "msg": "missing_client_order_id", "data": []}
        return self._request(
            "query_order",
            "GET",
            "/api/v5/trade/order",
            params={"instId": self._symbol, "clOrdId": cloid},
            signed=True,
        )

    def market_close(
        self, symbol: str, sz: Optional[float] = None
    ) -> dict[str, Any]:
        """Force-close ``symbol`` via OKX's dedicated close-position endpoint.

        OKX has ``POST /api/v5/trade/close-position`` that handles
        position-mode + reduce-only + market-execution semantics in one
        call. Cleaner than building a reduce-only IOC market order
        manually.
        """
        sym = _normalize_okx_symbol(symbol)
        pos = self.fetch_position("", sym)
        if abs(pos.position_qty) <= 0:
            return {"code": "0", "msg": "noop_already_flat", "data": []}
        body: dict[str, Any] = {
            "instId": sym,
            "mgnMode": "cross",
            # In net-mode, posSide is "net". (long_short would be "long"/"short".)
            "posSide": "net",
        }
        # ``sz`` argument: if given, partial close. OKX expects this
        # in CONTRACTS, so convert from base if provided.
        if sz is not None and sz > 0:
            body["sz"] = f"{self._base_to_contracts(sz):.10g}"
        return self._request(
            "close_position",
            "POST",
            "/api/v5/trade/close-position",
            body=body,
            signed=True,
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def bump_row_rate_limit_counter(self, n: int = 1) -> None:
        """Increment the row-level rate-limit hit counter (v1.4.4).

        Called by execution / dispatcher code when
        :func:`interpret_okx_place_batch_response` reclassifies a row
        as ``transport_rejected`` due to a row-level "Rate limit
        reached" sMsg. Lock-protected because the dispatcher runs
        from multiple worker threads.
        """
        if n <= 0:
            return
        with self._lock:
            self._okx_row_rate_limit_total += int(n)

    def rest_runtime_counters(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for op, n in self._rest_call_counts.items():
                out[f"{op}_calls"] = int(n)
            for op, n in self._rest_retry_counts.items():
                out[f"{op}_retries"] = int(n)
            out["okx_rest_429_retries_total"] = int(self._rest_429_retries_total)
            # 1.3.110 Phase 4a: WS cancel telemetry. The send count
            # vs. fallback count ratio is the operator's "is the WS
            # path actually delivering?" indicator — ≥99% WS send
            # post-activation is the stage-3 acceptance bar.
            out["okx_ws_cancel_send_count"] = int(self._cancel_ws_send_count)
            out["okx_ws_cancel_fallback_count"] = int(
                self._cancel_ws_fallback_count
            )
            # 1.3.111 Phase 4b: WS place telemetry. Stage-4 acceptance
            # bar is the same ≥99% WS-send ratio; spikes in fallback
            # are the operator's signal that the WS layer has
            # regressed (and the bot is silently still working via
            # HTTP, hiding the issue without the counter).
            out["okx_ws_place_send_count"] = int(self._place_ws_send_count)
            out["okx_ws_place_fallback_count"] = int(
                self._place_ws_fallback_count
            )
            # 1.4.4: row-level rate-limit hits on place responses
            # (single + batch). Pre-1.4.4 these were silently
            # mis-classified as exchange_rejected. The counter lets
            # the operator see the row-level wall pressure even when
            # the retry layer absorbs each hit transparently.
            out["okx_row_rate_limit_total"] = int(
                self._okx_row_rate_limit_total
            )
        # 1.4.4: REST sliding-window gauge. Exposed under the
        # ``okx_rate_window_*`` namespace so it sits next to the
        # cumulative counters in the dashboard. ``current_2s_rate``
        # is the live gauge (compare to the venue's 1200/2s the partner
        # cap); ``peak_2s_in_last_60s`` is the recent-pressure tick
        # mark; ``total_recorded`` is a sanity check against
        # ``okx_rest_call_count_*`` to catch lost increments.
        try:
            gauge = self._rate_window.snapshot()
            out["okx_rate_window_current_2s"] = int(gauge["current_2s_rate"])
            out["okx_rate_window_peak_2s_60s"] = int(
                gauge["peak_2s_in_last_60s"]
            )
            out["okx_rate_window_total_recorded"] = int(
                gauge["total_recorded"]
            )
        except Exception:
            logger.exception("okx_rate_window_snapshot_failed")
        # v1.4.20 rate-limit-observability Phase 2: surface the
        # per-pool snapshot for the Connectivity dashboard panel.
        # Nested dict keyed by pool name; each entry has
        # ``current_2s``, ``peak_2s_60s``, ``total``, ``cap``,
        # ``pct_of_cap``. The aggregate window is also included
        # under key ``"aggregate"`` so the operator can cross-check
        # against the flat legacy gauge above.
        #
        # When a pool has never seen traffic it's ABSENT from the
        # dict (lazy population in ``RestRateWindow``) — keeps the
        # surface lean for bots that don't exercise every endpoint.
        try:
            out["okx_rate_window_per_pool"] = (
                self._rate_window.snapshot_per_pool()
            )
        except Exception:
            logger.exception("okx_rate_window_per_pool_snapshot_failed")
        # Merge the action-WS coordinator's own snapshot (RTT, login
        # state, rate-limit) so the operator sees one unified view.
        # Read without the OkxClient lock — the WS coordinator has
        # its own internal locking.
        with self._action_ws_lock:
            ws = self._action_ws
        if ws is not None:
            try:
                ws_stats = ws.snapshot_stats()
                for k, v in ws_stats.items():
                    # ``snapshot_stats`` returns ints AND floats; the
                    # rest_runtime_counters contract is ``dict[str, int]``
                    # so we cast — floats become ints, which loses
                    # sub-ms RTT precision but matches the existing
                    # dashboard surface (dashboards multiply by 1000
                    # downstream when they need more precision).
                    out[k] = int(v)
            except Exception:
                logger.exception("okx_action_ws_snapshot_failed")
        return out

    def close(self) -> None:
        """Release both HTTP connection pools AND stop the action WS
        daemon (if started). Idempotent and exception-safe so it is
        callable from process-shutdown paths (atexit, ``finally``,
        ``__del__``). One subsystem's close failure does not prevent
        the others from closing.

        Phase 2 (1.3.108): two HTTP pools to release — cancel + place.
        Phase 4a (1.3.110): also stops the action-WS daemon thread,
        which fails any in-flight pendings with
        ``OkxActionWsDisconnected`` so blocked callers wake.
        """
        # Stop the WS first so in-flight pending requests can fall
        # back to HTTP via the still-open pools (mostly relevant
        # during the parallel-shutdown of the bot's worker threads).
        with self._action_ws_lock:
            ws = self._action_ws
            self._action_ws = None
        if ws is not None:
            try:
                ws.stop()
            except Exception:
                logger.exception("okx_action_ws_close_failed")
        for attr in ("_http_place", "_http_cancel"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            try:
                client.close()
            except Exception:
                logger.exception("okx_client_http_close_failed attr=%s", attr)

    # ------------------------------------------------------------------
    # Wire-format interpreters (delegated to okx_responses.py)
    # ------------------------------------------------------------------

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_okx_place_response(resp)

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        return interpret_okx_cancel_response(resp)

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        return interpret_okx_order_status_response(resp)

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        return make_deterministic_okx_client_order_id(
            symbol, side, quote_cycle_id, price, size
        )
