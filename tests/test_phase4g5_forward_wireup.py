"""Phase 4G.5 (v1.4.211) — forward-classifier live-wire tests.

4G.5 ships the live wiring from settings → history buffers →
classifier → ``evaluate_mode(forward_regime=...)`` → snapshot_dict.
Until 4G.5, the classifier existed but was never invoked: 4G.2
added the ``forward_regime=`` parameter to ``evaluate_mode``, but
``app/bot.py`` never passed anything to it — meaning CALM / CAUTIOUS
were unreachable in practice.

Test scope:
  * Settings round-trip from env aliases.
  * History-buffer append + truncation on BotState.
  * ``snapshot_dict`` carries the ``forward_signal`` placeholder
    block when ``forward_reading`` is None (always-render contract).
  * ``snapshot_dict`` populates the ``forward_signal`` block when a
    real reading is passed (dashboard / Telegram surfaces).
  * Bot-tick wire-up smoke (deferred to integration; not covered
    here — see ``test_phase4g2_calm_cautious_fsm.py`` for FSM
    behaviour with ``forward_regime`` set directly).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from tests.settings_helpers import UnitTestSettings


# -------------------------------------------------------------------
# Settings round-trip
# -------------------------------------------------------------------


def test_forward_settings_round_trip_from_env_aliases() -> None:
    """All Phase 4G.5 + v1.5.191 settings load from their REGIME_FORWARD_*
    env aliases. Pins the operator's tuning surface. v1.5.191 split the
    single-threshold knobs into enter/exit pairs for Schmitt-trigger
    band hysteresis."""
    path = Path(tempfile.gettempdir()) / f"mm_4g5_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "REGIME_FORWARD_ENABLED": True,
        "REGIME_FORWARD_CAUTIOUS_ENTRY_DWELL_SECONDS": 5.0,
        "REGIME_FORWARD_CAUTIOUS_EXIT_DWELL_SECONDS": 90.0,
        "REGIME_FORWARD_CALM_ENTRY_DWELL_SECONDS": 180.0,
        "REGIME_FORWARD_CALM_EXIT_DWELL_SECONDS": 2.0,
        # v1.5.191 — CAUTIOUS pairs (enter strict, exit loose)
        "REGIME_FORWARD_CAUTIOUS_ENTER_VOL_SLOPE_BPS_PER_MIN": 0.75,
        "REGIME_FORWARD_CAUTIOUS_EXIT_VOL_SLOPE_BPS_PER_MIN": 0.40,
        "REGIME_FORWARD_VOL_SLOPE_LOOKBACK_SECONDS": 45.0,
        "REGIME_FORWARD_CAUTIOUS_ENTER_DRIFT_MAGNITUDE_RISING_RATIO": 1.8,
        "REGIME_FORWARD_CAUTIOUS_EXIT_DRIFT_MAGNITUDE_RISING_RATIO": 1.3,
        "REGIME_FORWARD_DRIFT_MAGNITUDE_LOOKBACK_SECONDS": 20.0,
        "REGIME_FORWARD_DRIFT_MAGNITUDE_FLOOR_BPS": 7.0,
        "REGIME_FORWARD_CAUTIOUS_ENTER_OB_IMBALANCE_WIDENING_DELTA": 0.4,
        "REGIME_FORWARD_CAUTIOUS_EXIT_OB_IMBALANCE_WIDENING_DELTA": 0.2,
        "REGIME_FORWARD_OB_IMBALANCE_LOOKBACK_SECONDS": 75.0,
        "REGIME_FORWARD_CAUTIOUS_ENTER_BASIS_STRETCH_RATIO": 2.5,
        "REGIME_FORWARD_CAUTIOUS_EXIT_BASIS_STRETCH_RATIO": 1.8,
        "REGIME_FORWARD_BASIS_STRETCH_FLOOR_BPS": 1.5,
        # v1.5.191 — CALM pairs (enter strict, exit loose)
        "REGIME_FORWARD_CALM_ENTER_MAX_VOL_BPS": 2.5,
        "REGIME_FORWARD_CALM_EXIT_MAX_VOL_BPS": 5.0,
        "REGIME_FORWARD_CALM_ENTER_MAX_DRIFT_MAGNITUDE_BPS": 4.0,
        "REGIME_FORWARD_CALM_EXIT_MAX_DRIFT_MAGNITUDE_BPS": 8.0,
        "REGIME_FORWARD_CALM_ENTER_MAX_OB_IMBALANCE_MAGNITUDE": 0.15,
        "REGIME_FORWARD_CALM_EXIT_MAX_OB_IMBALANCE_MAGNITUDE": 0.30,
        "REGIME_FORWARD_CALM_MIN_HISTORY_SECONDS": 90.0,
        "REGIME_FORWARD_HISTORY_BUFFER_SECONDS": 240.0,
    })
    assert s.regime_forward_enabled is True
    assert s.regime_forward_cautious_entry_dwell_seconds == pytest.approx(5.0)
    assert s.regime_forward_cautious_exit_dwell_seconds == pytest.approx(90.0)
    assert s.regime_forward_calm_entry_dwell_seconds == pytest.approx(180.0)
    assert s.regime_forward_calm_exit_dwell_seconds == pytest.approx(2.0)
    # v1.5.191 — CAUTIOUS pairs
    assert s.regime_forward_cautious_enter_vol_slope_bps_per_min == pytest.approx(0.75)
    assert s.regime_forward_cautious_exit_vol_slope_bps_per_min == pytest.approx(0.40)
    assert s.regime_forward_vol_slope_lookback_seconds == pytest.approx(45.0)
    assert s.regime_forward_cautious_enter_drift_magnitude_rising_ratio == pytest.approx(1.8)
    assert s.regime_forward_cautious_exit_drift_magnitude_rising_ratio == pytest.approx(1.3)
    assert s.regime_forward_drift_magnitude_lookback_seconds == pytest.approx(20.0)
    assert s.regime_forward_drift_magnitude_floor_bps == pytest.approx(7.0)
    assert s.regime_forward_cautious_enter_ob_imbalance_widening_delta == pytest.approx(0.4)
    assert s.regime_forward_cautious_exit_ob_imbalance_widening_delta == pytest.approx(0.2)
    assert s.regime_forward_ob_imbalance_lookback_seconds == pytest.approx(75.0)
    assert s.regime_forward_cautious_enter_basis_stretch_ratio == pytest.approx(2.5)
    assert s.regime_forward_cautious_exit_basis_stretch_ratio == pytest.approx(1.8)
    assert s.regime_forward_basis_stretch_floor_bps == pytest.approx(1.5)
    # v1.5.191 — CALM pairs
    assert s.regime_forward_calm_enter_max_vol_bps == pytest.approx(2.5)
    assert s.regime_forward_calm_exit_max_vol_bps == pytest.approx(5.0)
    assert s.regime_forward_calm_enter_max_drift_magnitude_bps == pytest.approx(4.0)
    assert s.regime_forward_calm_exit_max_drift_magnitude_bps == pytest.approx(8.0)
    assert s.regime_forward_calm_enter_max_ob_imbalance_magnitude == pytest.approx(0.15)
    assert s.regime_forward_calm_exit_max_ob_imbalance_magnitude == pytest.approx(0.30)
    assert s.regime_forward_calm_min_history_seconds == pytest.approx(90.0)
    assert s.regime_forward_history_buffer_seconds == pytest.approx(240.0)


def test_forward_enabled_default_on_for_storm_protection() -> None:
    """``REGIME_FORWARD_ENABLED`` defaults to True (flipped from
    False in v1.4.213). Reasoning: Phase 4G exists BECAUSE of the
    v1.4.200 SF#11175 incident. Shipping the feature off-by-default
    means the very next storm trades with zero forward protection —
    i.e. the bot eats another SF before the operator gets around to
    flipping the flag. Default ON; operator overrides to OFF only
    if explicitly bisecting a regression."""
    path = Path(tempfile.gettempdir()) / f"mm_4g5_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    })
    assert s.regime_forward_enabled is True


def test_forward_enabled_explicit_false_disables_via_env() -> None:
    """The opt-OUT path: operator can set ``REGIME_FORWARD_ENABLED=false``
    in ``.env`` to disable. Used for bisecting regressions only —
    not the recommended default."""
    path = Path(tempfile.gettempdir()) / f"mm_4g5_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "REGIME_FORWARD_ENABLED": False,
    })
    assert s.regime_forward_enabled is False


# -------------------------------------------------------------------
# History buffer
# -------------------------------------------------------------------


def _make_state():
    """Build a minimal BotState for buffer tests. Avoid the full
    bot-start path (DB / WS init) by using model_validate + direct
    BotState construction.
    """
    from app.state import BotState

    path = Path(tempfile.gettempdir()) / f"mm_4g5_state_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    })
    st = BotState(settings=s)
    return st


def test_buffer_appends_values_with_timestamps() -> None:
    st = _make_state()
    st.append_forward_signal_history(
        now_mono=100.0,
        vol_bps=2.5,
        drift_30s_bps=1.0,
        ob_imbalance=0.05,
        max_age_seconds=180.0,
    )
    assert st.forward_vol_bps_history == [(100.0, 2.5)]
    assert st.forward_drift_30s_history == [(100.0, 1.0)]
    assert st.forward_ob_imbalance_history == [(100.0, 0.05)]


def test_buffer_skips_none_per_channel() -> None:
    """Defensive: each channel is independently None-tolerant.
    Early ticks may be missing any subset (e.g. drift before
    mid_history is populated, ob_imbalance before WS warm-up)."""
    st = _make_state()
    st.append_forward_signal_history(
        now_mono=100.0,
        vol_bps=2.5,
        drift_30s_bps=None,
        ob_imbalance=0.05,
        max_age_seconds=180.0,
    )
    assert st.forward_vol_bps_history == [(100.0, 2.5)]
    assert st.forward_drift_30s_history == []
    assert st.forward_ob_imbalance_history == [(100.0, 0.05)]


def test_buffer_truncates_entries_older_than_cutoff() -> None:
    """After enough ticks the buffer prunes old entries — keeps the
    classifier's lookback windows bounded."""
    st = _make_state()
    # Seed 5 entries spaced 60 s apart, then push a 6th that puts the
    # first 2 outside a 180 s window.
    for t in (0.0, 60.0, 120.0, 180.0, 240.0):
        st.append_forward_signal_history(
            now_mono=t,
            vol_bps=1.0,
            drift_30s_bps=1.0,
            ob_imbalance=0.0,
            max_age_seconds=180.0,
        )
    # At t=240, cutoff = 60. Entries at t=0 should be pruned;
    # t=60 is right on the boundary (>= cutoff).
    assert st.forward_vol_bps_history[0][0] >= 60.0
    assert all(t >= 60.0 for t, _ in st.forward_vol_bps_history)


def test_buffer_skips_truncation_when_unnecessary() -> None:
    """Optimisation pin: when the oldest entry is still inside the
    window we should NOT rebuild the list. Verified indirectly by
    checking the list grows monotonically until the boundary."""
    st = _make_state()
    for t in (0.0, 30.0, 60.0):
        st.append_forward_signal_history(
            now_mono=t,
            vol_bps=1.0,
            drift_30s_bps=None,
            ob_imbalance=None,
            max_age_seconds=180.0,
        )
    # All three should still be present — none aged out.
    assert len(st.forward_vol_bps_history) == 3


# -------------------------------------------------------------------
# snapshot_dict — forward_signal block always renders
# -------------------------------------------------------------------


def test_snapshot_forward_signal_block_placeholder_when_no_reading() -> None:
    """No reading → placeholder block with None/0.0 fields. The
    dashboard's Bot Stats card must always render."""
    from app.regime_controller import RegimeControllerState, snapshot_dict

    rc = RegimeControllerState()
    snap = snapshot_dict(rc, now_mono=100.0)
    fwd = snap["forward_signal"]
    assert isinstance(fwd, dict)
    assert fwd["classification"] is None
    assert fwd["reason"] is None
    assert fwd["history_span_seconds"] == 0.0
    assert fwd["vol_slope_bps_per_min"] is None


def test_snapshot_forward_signal_block_populated_when_reading_passed() -> None:
    """A real ForwardSignalReading flows through to the snapshot
    block. Pins the Telegram / dashboard contract."""
    from app.regime_controller import RegimeControllerState, snapshot_dict
    from app.regime_forward_signals import ForwardRegime, ForwardSignalReading

    rc = RegimeControllerState()
    reading = ForwardSignalReading(
        classification=ForwardRegime.CAUTIOUS,
        reason="vol_slope:1.20bp/min",
        vol_slope_bps_per_min=1.2,
        drift_magnitude_30s_bps=6.5,
        drift_magnitude_rising_ratio_observed=1.3,
        ob_imbalance_widening_delta_observed=0.1,
        basis_stretch_ratio_observed=None,
        history_span_seconds=42.0,
    )
    snap = snapshot_dict(rc, now_mono=100.0, forward_reading=reading)
    fwd = snap["forward_signal"]
    assert fwd["classification"] == "CAUTIOUS"
    assert fwd["reason"] == "vol_slope:1.20bp/min"
    assert fwd["vol_slope_bps_per_min"] == pytest.approx(1.2)
    assert fwd["drift_magnitude_30s_bps"] == pytest.approx(6.5)
    assert fwd["drift_magnitude_rising_ratio_observed"] == pytest.approx(1.3)
    assert fwd["ob_imbalance_widening_delta_observed"] == pytest.approx(0.1)
    assert fwd["basis_stretch_ratio_observed"] is None
    assert fwd["history_span_seconds"] == pytest.approx(42.0)


def test_snapshot_forward_signal_block_calm_classification() -> None:
    """CALM classification surfaces through. The dashboard chip flips
    to a 'green' / 'aggressive' indicator under CALM."""
    from app.regime_controller import RegimeControllerState, snapshot_dict
    from app.regime_forward_signals import ForwardRegime, ForwardSignalReading

    rc = RegimeControllerState()
    reading = ForwardSignalReading(
        classification=ForwardRegime.CALM,
        reason="all_quiet",
        vol_slope_bps_per_min=0.0,
        drift_magnitude_30s_bps=1.0,
        drift_magnitude_rising_ratio_observed=None,
        ob_imbalance_widening_delta_observed=None,
        basis_stretch_ratio_observed=None,
        history_span_seconds=120.0,
    )
    snap = snapshot_dict(rc, now_mono=100.0, forward_reading=reading)
    assert snap["forward_signal"]["classification"] == "CALM"
    assert snap["forward_signal"]["reason"] == "all_quiet"


# -------------------------------------------------------------------
# State-publish smoke: BotState.snapshot_dict carries the block
# -------------------------------------------------------------------


def test_state_snapshot_includes_regime_mode_with_forward_signal_block() -> None:
    """Pin the top-level wire-up — ``BotState.snapshot_dict()`` exposes
    ``behavioural_gates.regime_mode.forward_signal`` so the frontend
    + postmortem can read the current forward classification."""
    st = _make_state()
    full = st.snapshot_dict()
    rm = full.get("behavioural_gates", {}).get("regime_mode")
    assert isinstance(rm, dict)
    assert "forward_signal" in rm
    # No reading on state yet → placeholder.
    assert rm["forward_signal"]["classification"] is None


def test_state_snapshot_surfaces_real_forward_reading() -> None:
    """Same smoke test, with a real reading stamped on state. Frontend
    sees the live classification via behavioural_gates."""
    from app.regime_forward_signals import ForwardRegime, ForwardSignalReading

    st = _make_state()
    st.last_forward_signal_reading = ForwardSignalReading(
        classification=ForwardRegime.CAUTIOUS,
        reason="ob_widening:+0.45",
        vol_slope_bps_per_min=0.1,
        drift_magnitude_30s_bps=2.0,
        drift_magnitude_rising_ratio_observed=None,
        ob_imbalance_widening_delta_observed=0.45,
        basis_stretch_ratio_observed=None,
        history_span_seconds=75.0,
    )
    full = st.snapshot_dict()
    fwd = full["behavioural_gates"]["regime_mode"]["forward_signal"]
    assert fwd["classification"] == "CAUTIOUS"
    assert fwd["reason"] == "ob_widening:+0.45"
    assert fwd["ob_imbalance_widening_delta_observed"] == pytest.approx(0.45)
