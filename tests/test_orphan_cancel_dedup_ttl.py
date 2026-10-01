"""
Invariant tests for the orphan-cancel dedup cache
(`OrderManager._recent_orphan_cancels` +
`_ORPHAN_CANCEL_DEDUP_TTL_SECONDS` / `_ORPHAN_CANCEL_DEDUP_MAX_ENTRIES`).

The cache protects against cancel storms during the exchange-settlement
window: once an orphan cancel has been dispatched with a confirmed outcome
(`success` / `benign_missing`), repeat calls for the same `(symbol, oid)`
within the TTL skip the HTTP transport and return `True` (treated as
already-dispatched). Failures MUST NOT be cached — retries must still reach
the exchange. The cache is bounded in size: on overflow the oldest entry is
evicted before insert.

These tests pin each of those four behaviors.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.execution import (
    _ORPHAN_CANCEL_DEDUP_MAX_ENTRIES,
    _ORPHAN_CANCEL_DEDUP_TTL_SECONDS,
    OrderManager,
)
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _ok_cancel() -> dict:
    return {
        "status": "ok",
        "response": {"type": "cancel", "data": {"statuses": ["success"]}},
    }


def _err_cancel() -> dict:
    # Non-benign error — `_cancel_error_is_benign_missing` returns False for this text,
    # so `interpret_hl_cancel_response` classifies it as `error` → method returns False.
    return {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": [{"error": "internal server error"}]},
        },
    }


def _setup() -> tuple[UnitTestSettings, Path, OrderManager]:
    path = Path(tempfile.gettempdir()) / f"mm_dedup_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_order.return_value = _ok_cancel()
    client.cancel_order_by_cloid.return_value = _ok_cancel()
    om = OrderManager(s, client, storage, state)
    return s, path, om


def test_dedup_within_ttl_skips_second_call() -> None:
    """
    A. Two calls with the same (symbol, oid) within TTL:
       - first call hits transport and returns True
       - second call is skipped (no second transport call), still returns True
       - `_orphan_cancel_dedup_skips` counter increments by 1
    """
    s, path, om = _setup()
    sym = s.symbol

    ok1 = om._cancel_orphan_remote_order(
        symbol=sym, oid=7777, cloid=None, reason="dedup_ttl_A"
    )
    assert ok1 is True
    assert om._client.cancel_order.call_count == 1, "first call must hit transport"
    assert (sym, 7777) in om._recent_orphan_cancels, "success must populate cache"
    skips_before = om._orphan_cancel_dedup_skips

    ok2 = om._cancel_orphan_remote_order(
        symbol=sym, oid=7777, cloid=None, reason="dedup_ttl_A_repeat"
    )
    assert ok2 is True, "dedup skip must return True (already-dispatched semantics)"
    assert om._client.cancel_order.call_count == 1, (
        "second call within TTL must NOT hit transport; "
        f"got call_count={om._client.cancel_order.call_count}"
    )
    assert om._orphan_cancel_dedup_skips == skips_before + 1, (
        f"dedup skip counter must increment; "
        f"before={skips_before} after={om._orphan_cancel_dedup_skips}"
    )
    path.unlink(missing_ok=True)


def test_retry_after_ttl_hits_transport_again() -> None:
    """
    B. Time passes beyond TTL → the cached entry is stale and the next call
       hits the transport again. We simulate elapsed time by rewriting the
       cache entry's mono timestamp into the past.
    """
    s, path, om = _setup()
    sym = s.symbol

    ok1 = om._cancel_orphan_remote_order(
        symbol=sym, oid=8888, cloid=None, reason="dedup_ttl_B"
    )
    assert ok1 is True
    assert om._client.cancel_order.call_count == 1
    key = (sym, 8888)
    assert key in om._recent_orphan_cancels

    # Age the cache entry past TTL (no monkeypatch of `time.monotonic` needed —
    # the check is a subtraction against `time.monotonic()` at call time).
    om._recent_orphan_cancels[key] = (
        time.monotonic() - _ORPHAN_CANCEL_DEDUP_TTL_SECONDS - 1.0
    )
    skips_before = om._orphan_cancel_dedup_skips

    ok2 = om._cancel_orphan_remote_order(
        symbol=sym, oid=8888, cloid=None, reason="dedup_ttl_B_retry"
    )
    assert ok2 is True
    assert om._client.cancel_order.call_count == 2, (
        "after TTL expiry, the transport must be invoked again; "
        f"got call_count={om._client.cancel_order.call_count}"
    )
    assert om._orphan_cancel_dedup_skips == skips_before, (
        "stale entry must not count as a dedup skip"
    )
    # Cache entry refreshed to a recent timestamp.
    assert (
        time.monotonic() - om._recent_orphan_cancels[key]
    ) < _ORPHAN_CANCEL_DEDUP_TTL_SECONDS, "cache entry must be refreshed on retry"
    path.unlink(missing_ok=True)


def test_failure_does_not_populate_dedup_cache() -> None:
    """
    C. A non-benign exchange failure MUST NOT populate the cache. Operators
       retain retry semantics: the next invocation still hits the transport.
    """
    s, path, om = _setup()
    sym = s.symbol
    om._client.cancel_order.return_value = _err_cancel()

    ok1 = om._cancel_orphan_remote_order(
        symbol=sym, oid=9999, cloid=None, reason="dedup_ttl_C"
    )
    assert ok1 is False, "non-benign error must return False"
    assert om._client.cancel_order.call_count == 1
    assert (sym, 9999) not in om._recent_orphan_cancels, (
        "failure must NOT populate cache — retries must still reach the exchange"
    )

    # Second call — cache is empty for this key, so retry hits the exchange.
    ok2 = om._cancel_orphan_remote_order(
        symbol=sym, oid=9999, cloid=None, reason="dedup_ttl_C_retry"
    )
    assert ok2 is False
    assert om._client.cancel_order.call_count == 2, (
        "second call must hit transport (not deduped) after prior failure"
    )
    assert om._orphan_cancel_dedup_skips == 0, (
        "no dedup skip should count across two failures"
    )
    path.unlink(missing_ok=True)


def test_cache_eviction_drops_oldest_when_full() -> None:
    """
    D. When the cache is already at `_ORPHAN_CANCEL_DEDUP_MAX_ENTRIES`, a new
       successful dispatch triggers oldest-entry eviction before insert.
       Final size stays at the cap; the oldest key is gone and the new key
       is present. No exceptions raised.
    """
    s, path, om = _setup()
    sym = s.symbol

    # Pre-populate the cache to the cap with synthetic entries whose monotonic
    # timestamps are strictly increasing with oid — so oid=0 is unambiguously
    # the oldest entry and must be the one evicted.
    now = time.monotonic()
    for i in range(_ORPHAN_CANCEL_DEDUP_MAX_ENTRIES):
        # Older entries get earlier (more negative) offsets.
        om._recent_orphan_cancels[(sym, i)] = now - (
            _ORPHAN_CANCEL_DEDUP_MAX_ENTRIES - i
        ) * 1e-6
    assert len(om._recent_orphan_cancels) == _ORPHAN_CANCEL_DEDUP_MAX_ENTRIES

    # One real dispatch with a brand-new oid must trigger eviction.
    new_oid = 10_000_000
    ok = om._cancel_orphan_remote_order(
        symbol=sym, oid=new_oid, cloid=None, reason="dedup_ttl_D_evict"
    )
    assert ok is True
    assert len(om._recent_orphan_cancels) == _ORPHAN_CANCEL_DEDUP_MAX_ENTRIES, (
        "cache size must stay at cap after eviction+insert; "
        f"got {len(om._recent_orphan_cancels)}"
    )
    assert (sym, 0) not in om._recent_orphan_cancels, (
        "oldest entry (oid=0) must have been evicted"
    )
    assert (sym, new_oid) in om._recent_orphan_cancels, (
        "new entry must be inserted after eviction"
    )
    path.unlink(missing_ok=True)


def test_dedup_skip_does_not_mark_cancel_failure_for_dup_sync() -> None:
    """
    E. (defense-in-depth) A dedup skip returns True, which is critical for the
       `_sync_open_orders_impl` dup branch: it tracks `all_extra_cancels_ok`
       off the return value, and a False would incorrectly latch
       `requires_confirm=True`. This test pins that contract directly at the
       helper level — if a future refactor flips the skip's return value the
       dup-branch safety latch would become over-eager.
    """
    s, path, om = _setup()
    sym = s.symbol

    ok1 = om._cancel_orphan_remote_order(
        symbol=sym, oid=55555, cloid=None, reason="dedup_ttl_E"
    )
    assert ok1 is True

    # Second call within TTL — must be a dedup skip returning True.
    ok2 = om._cancel_orphan_remote_order(
        symbol=sym, oid=55555, cloid=None, reason="dedup_ttl_E_skip"
    )
    assert ok2 is True, (
        "dedup skip must return True to preserve the dup-sync `all_extra_cancels_ok` contract"
    )
    assert om._orphan_cancel_dedup_skips == 1
    path.unlink(missing_ok=True)
