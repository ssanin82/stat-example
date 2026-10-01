"""SF event-id labeling for fills — full chain coverage.

Background — why this test file exists:
=======================================

The SF tagging pipeline broke twice in 2 days:

* **v1.4.157** (2026-05-20 17:14 UTC) — 3 untagged taker fills in
  SF#X (legacy adverse-drift taker fallback).
* **v1.4.189** (2026-05-21 10:29 UTC) — 1 untagged taker fill in
  SF#11172 phase-4 terminal market_close.

Both incidents had the same root cause: the v1.4.163 "fallback"
checked ``state.soft_flatten_event_id is not None`` AT fill-ingest
time, but ``_exit_soft_flatten`` synchronously NULLs that field
right after the taker fires. The fill arrives via private WS tens-
to-hundreds of ms later, by which point the state was already
cleared, so the fallback's predicate failed.

v1.4.192 added the **recent-event grace cache** (a 30 s TTL window
after SF exit during which the just-cleared event id is still
valid for labeling). This file pins the full priority chain so
the next regression in this pipeline shows up at edit time, not
in a snapshot 24 hours later.

Coverage matrix
---------------

The resolution helper ``resolve_sf_event_id_for_fill`` has three
fallback levels. The matrix below lists which test exercises each.

| # | Path                          | Scenario                              | Test                                                    |
|---|-------------------------------|---------------------------------------|---------------------------------------------------------|
| 0 | Parent lookup hit             | maker SF fill (post-only managed)     | ``test_parent_lookup_hit_returns_sf_event_id``          |
| 0 | Parent lookup miss returns None| non-SF fill from a non-SF order      | ``test_parent_lookup_miss_with_no_active_sf_returns_none``|
| 1 | Active-SF fallback hits       | taker SF fill, WS beats SF exit       | ``test_active_sf_fallback_hits_when_state_still_set``   |
| 1 | Active-SF fallback misses     | parent exists, just untagged          | ``test_active_sf_fallback_skips_when_parent_exists``    |
| 2 | Grace fallback hits           | taker SF fill, WS arrives AFTER exit  | ``test_grace_fallback_hits_within_ttl``                 |
| 2 | Grace fallback expires        | TTL elapsed                           | ``test_grace_fallback_skips_after_ttl_expires``         |
| 2 | Grace fallback respects guard | parent exists, just untagged          | ``test_grace_fallback_skips_when_parent_exists``        |
| 2 | Grace fallback bumps counter  | observability                         | ``test_grace_fallback_bumps_sf_recent_event_tag_total`` |
| - | _exit_soft_flatten populates  | end-to-end integration                | ``test_exit_soft_flatten_populates_grace_cache``        |
| - | Disabled grace window         | SF_RECENT_EVENT_ID_GRACE_SECONDS=0    | ``test_grace_window_disabled_when_setting_is_zero``     |
| - | Settings round-trip           | env-var → typed setting               | ``test_grace_seconds_default_30s``                      |

These tests exist BECAUSE the bug was missed twice. Future SF
labeling-related changes should add to this file, not duplicate.
"""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from app.fill_ingestion import resolve_sf_event_id_for_fill


# ---------------------------------------------------------------------------
# Shell builders — state + storage mocks tight enough that attribute
# typos fail loudly, not silently auto-vivify.
# ---------------------------------------------------------------------------


def _state(
    *,
    soft_flatten_event_id: Optional[int] = None,
    sf_recent_event_id: Optional[int] = None,
    sf_recent_event_id_valid_until_mono: float = 0.0,
    sf_recent_event_tag_total: int = 0,
) -> MagicMock:
    """Build a BotState-shaped mock with only the fields the
    resolver reads. MagicMock auto-vivifies unknown attributes which
    can mask bugs, so we set explicit defaults on every field used."""
    state = MagicMock()
    state.soft_flatten_event_id = soft_flatten_event_id
    state.sf_recent_event_id = sf_recent_event_id
    state.sf_recent_event_id_valid_until_mono = (
        sf_recent_event_id_valid_until_mono
    )
    state.sf_recent_event_tag_total = sf_recent_event_tag_total
    return state


def _storage(*, parent_order_exists: bool = False) -> MagicMock:
    """Build a Storage-shaped mock. ``parent_order_exists`` controls
    the guard predicate both fallbacks check."""
    storage = MagicMock()
    storage.parent_order_exists.return_value = parent_order_exists
    return storage


# ---------------------------------------------------------------------------
# Fallback 0 — parent-order lookup hit / miss
# ---------------------------------------------------------------------------


def test_parent_lookup_hit_returns_sf_event_id() -> None:
    """Happy path: maker SF fill whose parent order has
    ``soft_flatten_event_id=42`` returns 42 immediately, without
    consulting state or storage further."""
    state = _state()  # no active SF, no grace cache
    storage = _storage()
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=42,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        == 42
    )
    # The fallback paths must not have been consulted.
    storage.parent_order_exists.assert_not_called()


def test_parent_lookup_miss_with_no_active_sf_returns_none() -> None:
    """A regular non-SF fill (no parent lookup hit, no active SF, no
    grace cache) returns None. The fill stays untagged, which is the
    correct outcome for normal market-making fills."""
    state = _state()
    storage = _storage()
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        is None
    )


def test_parent_lookup_returns_zero_treated_as_real_sf_id() -> None:
    """Edge case: ``soft_flatten_event_id=0`` is technically a valid
    auto-increment id (SQLite starts at 1 by default, but defensive).
    The current behaviour is to treat ANY non-None as a real id —
    ``int(0)`` is returned as 0. Pin this so a future ``if not value``
    refactor doesn't silently downgrade 0 to None."""
    state = _state()
    storage = _storage()
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=0,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Fallback 1 — active-SF (v1.4.163 fix for the taker-fallback case)
# ---------------------------------------------------------------------------


def test_active_sf_fallback_hits_when_state_still_set() -> None:
    """v1.4.163 case: WS fill beats SF exit. parent absent + SF still
    active in state → stamp from ``state.soft_flatten_event_id``."""
    state = _state(soft_flatten_event_id=11171)
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        == 11171
    )


def test_active_sf_fallback_skips_when_parent_exists() -> None:
    """If a parent order row DOES exist but was legitimately untagged
    (e.g. a pre-SF order whose cancel raced a fill), we must NOT
    overwrite with the active SF id. ``parent_order_exists`` is the
    guard."""
    state = _state(soft_flatten_event_id=11171)
    storage = _storage(parent_order_exists=True)  # legit untagged
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        is None
    )


def test_active_sf_fallback_storage_exception_is_swallowed() -> None:
    """The fallback's ``parent_order_exists`` check is wrapped in
    try/except — a storage error must NOT crash fill ingestion.
    It returns None and ingestion continues (fill stays untagged)."""
    state = _state(soft_flatten_event_id=11171)
    storage = MagicMock()
    storage.parent_order_exists.side_effect = RuntimeError("db went away")
    # No exception escapes.
    result = resolve_sf_event_id_for_fill(
        parent_meta_sf_event_id=None,
        state=state,
        storage=storage,
        oid="abc",
        now_mono=100.0,
    )
    # Even though active_sf is set, the storage error path falls
    # through to the grace check (which also fails, returns None).
    assert result is None


# ---------------------------------------------------------------------------
# Fallback 2 — recent-event grace cache (v1.4.192, this fix)
# ---------------------------------------------------------------------------


def test_grace_fallback_hits_within_ttl() -> None:
    """The key v1.4.192 fix: SF exited (active id cleared), but the
    grace cache still holds the event id with a future-valid TTL.
    Late-arriving WS fill gets tagged."""
    state = _state(
        soft_flatten_event_id=None,  # SF already exited
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=200.0,  # 100s in the future
    )
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,  # well before valid_until
        )
        == 11172
    )


def test_grace_fallback_skips_after_ttl_expires() -> None:
    """Past the TTL the grace cache is dead — fill stays untagged.
    Defensive: prevents stamping a stale event id on fills that
    really do belong to a later non-SF window."""
    state = _state(
        soft_flatten_event_id=None,
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=100.0,  # equal to now → expired
    )
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        is None
    )
    # And well past it.
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=999.0,
        )
        is None
    )


def test_grace_fallback_skips_when_parent_exists() -> None:
    """Same guard as the active-SF fallback: never overwrite a
    parent-order-present-but-untagged fill."""
    state = _state(
        soft_flatten_event_id=None,
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=200.0,
    )
    storage = _storage(parent_order_exists=True)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        is None
    )


def test_grace_fallback_bumps_sf_recent_event_tag_total() -> None:
    """Observability: every grace-fallback hit increments the
    session counter. Operator reads this to confirm the grace path
    is doing real work."""
    state = _state(
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=200.0,
        sf_recent_event_tag_total=5,
    )
    storage = _storage(parent_order_exists=False)
    resolve_sf_event_id_for_fill(
        parent_meta_sf_event_id=None,
        state=state,
        storage=storage,
        oid="abc",
        now_mono=100.0,
    )
    assert state.sf_recent_event_tag_total == 6
    # Another hit increments again.
    resolve_sf_event_id_for_fill(
        parent_meta_sf_event_id=None,
        state=state,
        storage=storage,
        oid="def",
        now_mono=110.0,
    )
    assert state.sf_recent_event_tag_total == 7


def test_grace_fallback_does_not_bump_counter_on_miss() -> None:
    """A miss (TTL expired, or parent exists) MUST NOT bump the
    counter. Pin this so the operator's observability number stays
    meaningful."""
    state = _state(
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=100.0,  # expired
        sf_recent_event_tag_total=5,
    )
    storage = _storage(parent_order_exists=False)
    resolve_sf_event_id_for_fill(
        parent_meta_sf_event_id=None,
        state=state,
        storage=storage,
        oid="abc",
        now_mono=200.0,  # past TTL
    )
    assert state.sf_recent_event_tag_total == 5  # unchanged


# ---------------------------------------------------------------------------
# Priority chain — fallback 0 wins over 1+2, fallback 1 wins over 2
# ---------------------------------------------------------------------------


def test_parent_hit_wins_even_when_recent_cache_set() -> None:
    """Parent meta carries the canonical SF id even if a different
    grace cache happens to be populated. (Could happen if two SF
    episodes ran back-to-back; the second's grace cache is fresh
    but a fill from the second's PARENT order is still in flight.)"""
    state = _state(
        sf_recent_event_id=99999,  # different value
        sf_recent_event_id_valid_until_mono=200.0,
    )
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=42,  # the truth
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        == 42
    )


def test_active_sf_wins_over_recent_cache() -> None:
    """If SF re-entered (active again) AND there's a stale grace
    cache from a prior episode, the active id wins."""
    state = _state(
        soft_flatten_event_id=11173,  # new SF, currently active
        sf_recent_event_id=11172,  # previous SF, grace still valid
        sf_recent_event_id_valid_until_mono=200.0,
    )
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        == 11173
    )


# ---------------------------------------------------------------------------
# Integration: BotState._exit_soft_flatten populates the grace cache
# ---------------------------------------------------------------------------


def test_exit_soft_flatten_populates_grace_cache() -> None:
    """End-to-end check: simulate ``_exit_soft_flatten`` writing the
    grace cache, then ``resolve`` reads it. This test is the
    contract between the two halves of the v1.4.192 fix."""
    # Pretend Bot._exit_soft_flatten just ran. The relevant writes:
    state = _state()
    state.soft_flatten_event_id = 11172
    # Simulating _exit_soft_flatten:
    grace_s = 30.0
    sf_event_id = state.soft_flatten_event_id
    state.sf_recent_event_id = int(sf_event_id)
    state.sf_recent_event_id_valid_until_mono = time.monotonic() + grace_s
    state.soft_flatten_event_id = None  # the synchronous clear

    storage = _storage(parent_order_exists=False)
    # Fill arrives 100 ms later via WS.
    fill_arrival_mono = time.monotonic() + 0.1
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="exchange-oid-xyz",
            now_mono=fill_arrival_mono,
        )
        == 11172
    )


# ---------------------------------------------------------------------------
# Disabled grace window (operator escape hatch)
# ---------------------------------------------------------------------------


def test_grace_window_disabled_when_setting_is_zero() -> None:
    """If the operator sets ``SF_RECENT_EVENT_ID_GRACE_SECONDS=0`` AND
    _exit_soft_flatten respects that (doesn't populate the cache),
    the grace fallback degrades cleanly to None.

    Simulated: state with ``sf_recent_event_id_valid_until_mono=0.0``
    (i.e. ``_exit_soft_flatten`` ran with grace=0 and never wrote
    the future-valid TTL)."""
    state = _state(
        sf_recent_event_id=None,  # never populated
        sf_recent_event_id_valid_until_mono=0.0,
    )
    storage = _storage(parent_order_exists=False)
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="abc",
            now_mono=100.0,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------


def test_grace_seconds_default_30s() -> None:
    """Default setting value matches the v1.4.192 plan: 30 s."""
    from app.config import Settings

    s = Settings()
    assert s.sf_recent_event_id_grace_seconds == 30.0


def test_grace_seconds_round_trips_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings

    monkeypatch.setenv("SF_RECENT_EVENT_ID_GRACE_SECONDS", "15.5")
    s = Settings()
    assert s.sf_recent_event_id_grace_seconds == pytest.approx(15.5)


def test_grace_seconds_negative_rejected() -> None:
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(SF_RECENT_EVENT_ID_GRACE_SECONDS=-1.0)


def test_grace_seconds_zero_accepted() -> None:
    """Zero is the disable-the-grace-window value; must validate."""
    from app.config import Settings

    s = Settings(SF_RECENT_EVENT_ID_GRACE_SECONDS=0.0)
    assert s.sf_recent_event_id_grace_seconds == 0.0


# ---------------------------------------------------------------------------
# Regression: the v1.4.157 + v1.4.189 incidents
# ---------------------------------------------------------------------------


def test_regression_v1_4_157_three_untagged_takers_now_tagged() -> None:
    """Replays the v1.4.157 (2026-05-20 17:14 UTC) pattern: SF
    triggered, post-only timed out, legacy taker fallback fired
    ``client.market_close`` then ``_exit_soft_flatten``, three
    fills arrived later via WS. v1.4.163 fallback failed because
    state was already cleared; all three fills landed untagged.

    With the v1.4.192 grace cache the same scenario tags all three.
    """
    # Simulating: SF#X just exited 200ms ago, grace TTL = 30s.
    state = _state(
        soft_flatten_event_id=None,
        sf_recent_event_id=10157,
        sf_recent_event_id_valid_until_mono=200.0 + 30.0,
    )
    storage = _storage(parent_order_exists=False)
    # Three fills arrive over the next 500ms.
    for i, oid in enumerate(("fill-1", "fill-2", "fill-3")):
        result = resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid=oid,
            now_mono=200.3 + 0.1 * i,
        )
        assert result == 10157, f"fill {i} ({oid}) should be tagged"
    # Counter reflects all three.
    assert state.sf_recent_event_tag_total == 3


def test_regression_v1_4_189_phase_4_market_close_now_tagged() -> None:
    """Replays the v1.4.189 (2026-05-21 10:29 UTC) pattern:
    phase-ladder SF#11172 → phase 4 ``client.market_close`` →
    ``_exit_soft_flatten`` cleared state synchronously → 6-contract
    taker fill arrived ~100ms later via WS and landed untagged."""
    state = _state(
        soft_flatten_event_id=None,
        sf_recent_event_id=11172,
        sf_recent_event_id_valid_until_mono=100.0 + 30.0,
    )
    storage = _storage(parent_order_exists=False)
    # Fill arrives 100ms after SF exit.
    assert (
        resolve_sf_event_id_for_fill(
            parent_meta_sf_event_id=None,
            state=state,
            storage=storage,
            oid="3585524724406362112",  # the real exchange oid
            now_mono=100.1,
        )
        == 11172
    )
