"""v1.4.93 wedge-elimination-cleanup Phase 6A (minimal) —
snapshot replay tests (INTEGRATION TIER — operator-run, NOT in default CI).

**This file is under ``tests/integration/`` which is excluded from
default pytest collection** (see ``pytest.ini``: ``norecursedirs = integration``).

Why integration-tier:
* ``snapshots/`` is in ``.gitignore`` — operator-local artifacts dropped
  by ``scripts/fetch_bot_snapshot.ps1`` and friends. CI worktrees don't
  have them; the operator periodically cleans the directory; the
  snapshot schema can change.
* These tests exercise the replay framework against REAL production
  capture data. That's the right level of testing — it catches schema
  drift, validates the loader against true OKX shapes — but it can't
  run in default CI because the data isn't there.

To run::

    python -m pytest tests/integration/test_snapshot_replay.py -v

The replay framework itself (``tests/integration/snapshot_replay.py``)
is library code — unit-test it via inline-constructed fixtures, not
via filesystem snapshots.

Three test groups exercised against any available snapshots:

1. **Healthy-snapshot regression**: a curated list of recent healthy
   snapshots. ALL invariants must pass when the snapshot is present.

2. **Historical-incident pins**: known incident snapshots that SHOULD
   fail specific invariants — pin the wedge each one caught.

3. **Loader-resilience smoke**: the loader handles missing files,
   missing fields, irregular open-orders payloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.integration.snapshot_replay import (
    ReplayResult,
    load_snapshot,
    run_snapshot_replay,
)


_SNAPSHOTS_ROOT = Path(__file__).resolve().parents[1] / "snapshots"


# ---------------------------------------------------------------------------
# Healthy snapshots — all invariants must pass
# ---------------------------------------------------------------------------

# Curated list of recent healthy snapshots. Each is expected to pass
# all 5 invariants in ``snapshot_replay.run_snapshot_replay``.
HEALTHY_SNAPSHOTS = [
    "v1.4.67-260518-195033-prod.okx.ton.usdt.perp",  # Phase 1A landed
    "v1.4.71-260518-220025-prod.okx.ton.usdt.perp",  # Phase 1D CI fix
    "v1.4.73-260518-222923-prod.okx.ton.usdt.perp",  # Phase 1C
    "v1.4.74-260518-225854-prod.okx.ton.usdt.perp",  # Phase 1D
    "v1.4.75-260518-232201-prod.okx.ton.usdt.perp",  # Phase 2B reaper
    "v1.4.77-260518-234804-prod.okx.ton.usdt.perp",  # Phase 2D
    "v1.4.78-260519-081151-prod.okx.ton.usdt.perp",  # Phase 2½A
    "v1.4.80-260519-084912-prod.okx.ton.usdt.perp",  # Phase 3A
    "v1.4.82-260519-092025-prod.okx.ton.usdt.perp",  # Phase 3 complete
    "v1.4.85-260519-095608-prod.okx.ton.usdt.perp",  # Phase 4C
    "v1.4.90-260519-115155-prod.okx.ton.usdt.perp",  # current
]


@pytest.mark.parametrize("snapshot_name", HEALTHY_SNAPSHOTS)
def test_phase6a_healthy_snapshot_passes_all_invariants(
    snapshot_name: str,
) -> None:
    """Curated recent healthy snapshots: every invariant must pass.

    If this fails on a snapshot that previously passed, EITHER:
    1. The replay framework regressed (broken loader / invariant logic), OR
    2. The bot's recorded state shape changed in a way the framework
       doesn't tolerate.

    Either way the test pins what we expect a "clean session capture"
    to look like.
    """
    snap_path = _SNAPSHOTS_ROOT / snapshot_name
    if not snap_path.is_dir():
        pytest.skip(f"snapshot not available: {snapshot_name}")
    result = run_snapshot_replay(snap_path)
    assert result.ok, (
        f"snapshot {snapshot_name} failed {result.invariants_failed} "
        f"invariant(s):\n"
        + "\n".join(f"  - {nm}: {reason}" for nm, reason in result.failures)
    )
    assert result.invariants_passed == 5


# ---------------------------------------------------------------------------
# Historical-incident pins — these snapshots MUST fail specific invariants
# ---------------------------------------------------------------------------


def test_phase6a_v1_4_85_104553_fu1_drift_event() -> None:
    """v1.4.85 snapshot at 06:26:39 captured the FU-1 cancel-then-place
    drift event documented in the plan: two simultaneously-live BUYs
    at adjacent prices, reconciler caught and recovered with
    ``orphan_remote_cancel_dispatched`` events. The residue:
    ``ws_event_unmatched_to_local_wo_total: 4`` at capture time.

    This test PINS that residue. If a future fix makes this snapshot
    pass the wedge-counters invariant (counter back to 0), it means
    we retroactively cleaned the historical state — fine, but we
    should reclassify the snapshot as healthy and remove this test.
    """
    snap = "v1.4.85-260519-104553-prod.okx.ton.usdt.perp"
    snap_path = _SNAPSHOTS_ROOT / snap
    if not snap_path.is_dir():
        pytest.skip(f"snapshot not available: {snap}")
    result = run_snapshot_replay(snap_path)
    # Expected failure: ws_event_unmatched_to_local_wo_total != 0
    failed_names = {nm for nm, _ in result.failures}
    assert "invariant_no_wedge_counters_nonzero" in failed_names, (
        f"FU-1 pin: expected ws_unmatched counter > 0 on this snapshot. "
        f"If this test now passes, the historical event has been retroactively "
        f"cleaned and the snapshot should be moved to HEALTHY_SNAPSHOTS."
    )


def test_phase6a_v1_4_66_192658_original_wedge_series() -> None:
    """v1.4.66 snapshot at 19:26:58 captured the original wedge:
    ``risk_exec_state=CANCELLING`` and ``desync_phase=RECOVERED`` at
    capture. Phase 1A landed in v1.4.67 to address the WS-matcher
    race that drove this incident. The snapshot residue is FIXED
    historical evidence — these counter values are part of the
    incident's permanent record.

    This test pins the historical capture, not a future contract.
    """
    snap = "v1.4.66-260518-192658-prod.okx.ton.usdt.perp"
    snap_path = _SNAPSHOTS_ROOT / snap
    if not snap_path.is_dir():
        pytest.skip(f"snapshot not available: {snap}")
    result = run_snapshot_replay(snap_path)
    failed_names = {nm for nm, _ in result.failures}
    assert "invariant_risk_state_normal" in failed_names
    assert "invariant_desync_phase_ok" in failed_names


def test_phase6a_v1_4_69_v1_4_70_ws_matcher_race() -> None:
    """v1.4.69 / v1.4.70 captured the WS-event matcher race that
    Phase 1B fixed via the (oid, cloid) buffer. The residue is a
    non-zero ``ws_event_unmatched_to_local_wo_total`` counter.

    v1.4.69: 3 unmatched. v1.4.70: 22 unmatched (worse — the matcher
    race compounded as the session ran). Phase 1B's hydration dedup
    + WS buffer combo brought subsequent sessions to 0.
    """
    for snap, expected_min in [
        ("v1.4.69-260518-210941-prod.okx.ton.usdt.perp", 1),
        ("v1.4.70-260518-213819-prod.okx.ton.usdt.perp", 1),
    ]:
        snap_path = _SNAPSHOTS_ROOT / snap
        if not snap_path.is_dir():
            pytest.skip(f"snapshot not available: {snap}")
        result = run_snapshot_replay(snap_path)
        failed_names = {nm for nm, _ in result.failures}
        assert "invariant_no_wedge_counters_nonzero" in failed_names, (
            f"{snap}: expected ws_unmatched counter > 0 (historical pin); "
            f"failures={result.failures}"
        )


# ---------------------------------------------------------------------------
# Loader-resilience smoke
# ---------------------------------------------------------------------------


def test_phase6a_loader_resolves_relative_paths() -> None:
    """``load_snapshot`` accepts a bare directory name and resolves it
    against ``<repo>/snapshots/`` automatically. Lets tests refer to
    snapshots by short name without hardcoded paths."""
    snap = "v1.4.90-260519-115155-prod.okx.ton.usdt.perp"
    if not (_SNAPSHOTS_ROOT / snap).is_dir():
        pytest.skip(f"snapshot not available: {snap}")
    fixture = load_snapshot(snap)
    assert fixture.bot_version == "1.4.90"
    assert fixture.bot_profile == "prod.okx.ton.usdt.perp"
    assert fixture.captured_at_utc is not None
    assert isinstance(fixture.state_dict, dict)
    assert isinstance(fixture.open_orders_raw, list)


def test_phase6a_loader_raises_on_missing_snapshot() -> None:
    """Bare directory name that doesn't exist → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_snapshot("v9.9.99-nonexistent-prod.fake")


def test_phase6a_replay_returns_clean_result_on_healthy_snapshot() -> None:
    """Smoke check the ReplayResult shape — invariants_passed +
    invariants_failed should sum to the configured invariant count."""
    snap = "v1.4.90-260519-115155-prod.okx.ton.usdt.perp"
    if not (_SNAPSHOTS_ROOT / snap).is_dir():
        pytest.skip(f"snapshot not available: {snap}")
    result = run_snapshot_replay(snap)
    assert result.invariants_passed + result.invariants_failed == 5
    assert result.ok is True
    assert result.bot_version == "1.4.90"


# ---------------------------------------------------------------------------
# Coverage: every recent healthy snapshot has at least one open order
# (sanity that the loader is actually exercising the WorkingOrder
#  reconstruction path, not just running empty)
# ---------------------------------------------------------------------------


def test_phase6a_loader_reconstructs_working_orders_when_present() -> None:
    """If at least one HEALTHY_SNAPSHOTS entry is available locally
    AND has open orders at capture, exercise the WorkingOrder
    reconstruction path and verify the fields.

    CI environment note: snapshots/ is in .gitignore (operator's
    local artifacts). On a fresh worktree no snapshots are present —
    this test SKIPS rather than fails. Dev machines with prod
    snapshots dropped under snapshots/ will exercise the path.
    """
    from tests.integration.snapshot_replay import working_orders_from_snapshot

    available_snapshots = [
        snap for snap in HEALTHY_SNAPSHOTS
        if (_SNAPSHOTS_ROOT / snap).is_dir()
    ]
    if not available_snapshots:
        pytest.skip(
            "no HEALTHY_SNAPSHOTS available locally — "
            "snapshots/ is gitignored, CI worktrees won't have them. "
            "Dev machines with operator snapshots will run this test."
        )

    found_with_orders = False
    for snap in available_snapshots:
        snap_path = _SNAPSHOTS_ROOT / snap
        fixture = load_snapshot(snap_path)
        wos = working_orders_from_snapshot(fixture)
        if wos:
            found_with_orders = True
            # Sanity: each reconstructed WO has the required fields.
            for side, lvl, wo in wos:
                assert wo.symbol
                assert wo.order_id_exchange is not None
                assert wo.price > 0
                assert wo.size > 0
                assert wo.level_idx == lvl
            break

    if not found_with_orders:
        # All available snapshots had empty open_orders.json at capture
        # (bot just started, or in a no-quote period). That's possible
        # but the test isn't really exercising the reconstruction path.
        # Skip rather than fail — the data we have just doesn't cover it.
        pytest.skip(
            f"all {len(available_snapshots)} available snapshots had "
            f"no open orders at capture — reconstruction path not "
            f"exercised, but framework loaded all of them without error"
        )
