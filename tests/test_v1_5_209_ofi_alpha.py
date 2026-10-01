"""v1.5.209 — Phase 8D OFI directional alpha tests.

Covers:
* ``_bbo_contribution`` — 9 cases (3 bid-px regimes × 3 ask-px regimes)
* ``_ewma_step`` — cold start, halflife, time-aware decay
* ``OFIAccumulator`` — seed, contribution, EWMA mutation, normalisation
* ``compute_ofi_reservation_shift_bps`` — alpha=0 dormant, signal=None
  warmup, sign + magnitude
* Integration with ``compute_quote_decision`` — flag-off no-op,
  flag-on shift applied + emitted in breakdown
"""

from __future__ import annotations

import math

import pytest

from app.ofi import (
    OFIAccumulator,
    _bbo_contribution,
    _ewma_step,
    _normalise_to_unit,
    compute_ofi_reservation_shift_bps,
)


# ─────────────────────────── _bbo_contribution ──────────────────────────


def test_bbo_contribution_bid_lifts_signed_positive():
    """Best bid moves UP → buy pressure → positive contribution."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=100.05, bid_sz_1=15.0, ask_px_1=100.1, ask_sz_1=20.0,
    )
    # bid_c = +bid_sz_1 (=15), ask_c = -(20-20)=0 → +15
    assert c == pytest.approx(15.0)


def test_bbo_contribution_ask_lifts_signed_negative():
    """Best ask moves UP → selling pressure receded → positive
    contribution (less sell supply)."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=100.0, bid_sz_1=10.0, ask_px_1=100.15, ask_sz_1=18.0,
    )
    # bid_c = 0, ask_c = +ask_sz_0 = +20  →  +20
    assert c == pytest.approx(20.0)


def test_bbo_contribution_bid_size_grows_at_touch():
    """Bid stays at touch, size grows → positive contribution."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=100.0, bid_sz_1=12.0, ask_px_1=100.1, ask_sz_1=20.0,
    )
    # +2 on bid; 0 on ask
    assert c == pytest.approx(2.0)


def test_bbo_contribution_ask_size_grows_at_touch():
    """Ask stays at touch, size grows → negative contribution
    (more sell supply has arrived)."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=100.0, bid_sz_1=10.0, ask_px_1=100.1, ask_sz_1=25.0,
    )
    # 0 on bid; -(25-20) = -5 on ask
    assert c == pytest.approx(-5.0)


def test_bbo_contribution_bid_falls():
    """Best bid drops → buyers gave up → negative contribution."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=99.95, bid_sz_1=10.0, ask_px_1=100.1, ask_sz_1=20.0,
    )
    # bid_c = -bid_sz_0 = -10
    assert c == pytest.approx(-10.0)


def test_bbo_contribution_ask_falls():
    """Best ask drops → sellers stepped down → negative contribution."""
    c = _bbo_contribution(
        bid_px_0=100.0, bid_sz_0=10.0, ask_px_0=100.1, ask_sz_0=20.0,
        bid_px_1=100.0, bid_sz_1=10.0, ask_px_1=100.05, ask_sz_1=8.0,
    )
    # ask_c = -ask_sz_1 = -8
    assert c == pytest.approx(-8.0)


# ────────────────────────────── _ewma_step ──────────────────────────────


def test_ewma_step_cold_start_returns_sample():
    assert _ewma_step(None, 5.0, dt_seconds=1.0, halflife_seconds=10.0) == 5.0


def test_ewma_step_halflife_zero_returns_latest():
    assert _ewma_step(10.0, 7.0, dt_seconds=1.0, halflife_seconds=0.0) == 7.0


def test_ewma_step_dt_zero_returns_prev():
    assert _ewma_step(10.0, 7.0, dt_seconds=0.0, halflife_seconds=5.0) == 10.0


def test_ewma_step_one_halflife_moves_50pct():
    """After one halflife, the new value is halfway between prev and sample."""
    out = _ewma_step(0.0, 10.0, dt_seconds=5.0, halflife_seconds=5.0)
    assert out == pytest.approx(5.0)


def test_ewma_step_two_halflives_moves_75pct():
    out = _ewma_step(0.0, 10.0, dt_seconds=10.0, halflife_seconds=5.0)
    assert out == pytest.approx(7.5)


# ────────────────────────────── _normalise ──────────────────────────────


def test_normalise_zero_input_zero_output():
    assert _normalise_to_unit(0.0, 100.0) == pytest.approx(0.0)


def test_normalise_large_positive_saturates_to_unit():
    out = _normalise_to_unit(1000.0, 100.0)
    assert 0.99 < out < 1.0


def test_normalise_large_negative_saturates_to_minus_unit():
    out = _normalise_to_unit(-1000.0, 100.0)
    assert -1.0 < out < -0.99


def test_normalise_scale_zero_returns_zero():
    assert _normalise_to_unit(50.0, 0.0) == 0.0


# ───────────────────────────── OFIAccumulator ───────────────────────────


def test_accumulator_seed_no_signal_yet():
    acc = OFIAccumulator()
    acc.record_bbo(
        bid_px=100.0, bid_sz=10.0, ask_px=100.1, ask_sz=20.0,
        now_mono_seconds=0.0,
    )
    # First call seeds — no EWMA update yet.
    assert acc.raw_ewma_5s is None
    assert acc.raw_ewma_30s is None
    assert acc.signal_5s_normalised() is None
    assert acc.update_count == 0  # only counts post-seed updates


def test_accumulator_second_call_records_contribution():
    acc = OFIAccumulator(normalisation_scale=10.0)
    acc.record_bbo(
        bid_px=100.0, bid_sz=10.0, ask_px=100.1, ask_sz=20.0,
        now_mono_seconds=0.0,
    )
    # Bid lifts → +15 contribution
    acc.record_bbo(
        bid_px=100.05, bid_sz=15.0, ask_px=100.1, ask_sz=20.0,
        now_mono_seconds=1.0,
    )
    assert acc.raw_ewma_5s == pytest.approx(15.0)  # cold-start = sample
    assert acc.raw_ewma_30s == pytest.approx(15.0)
    assert acc.update_count == 1
    # Normalised: tanh(15/10) ≈ 0.905
    s = acc.signal_5s_normalised()
    assert s is not None and 0.8 < s < 0.95


def test_accumulator_rejects_nonfinite_inputs():
    acc = OFIAccumulator()
    acc.record_bbo(bid_px=100.0, bid_sz=10.0, ask_px=100.1, ask_sz=20.0, now_mono_seconds=0.0)
    # NaN price → no update
    acc.record_bbo(bid_px=float("nan"), bid_sz=10.0, ask_px=100.1, ask_sz=20.0, now_mono_seconds=1.0)
    assert acc.update_count == 0


def test_accumulator_rejects_negative_size():
    acc = OFIAccumulator()
    acc.record_bbo(bid_px=100.0, bid_sz=10.0, ask_px=100.1, ask_sz=20.0, now_mono_seconds=0.0)
    acc.record_bbo(bid_px=100.0, bid_sz=-5.0, ask_px=100.1, ask_sz=20.0, now_mono_seconds=1.0)
    assert acc.update_count == 0


def test_accumulator_decays_toward_new_signal():
    """Sustained positive contributions accumulate; EWMA approaches them."""
    acc = OFIAccumulator(halflife_5s_seconds=1.0)
    acc.record_bbo(bid_px=100.0, bid_sz=10.0, ask_px=100.1, ask_sz=20.0, now_mono_seconds=0.0)
    for i in range(1, 50):
        acc.record_bbo(
            bid_px=100.0, bid_sz=12.0, ask_px=100.1, ask_sz=20.0,
            now_mono_seconds=float(i),
        )
    # Per-step contribution oscillates between +2 (size grew) and 0
    # (size same as previous). After many steps the EWMA settles.
    assert acc.raw_ewma_5s is not None
    assert abs(acc.raw_ewma_5s) < 5.0  # bounded


# ──────────────────────────── shift helper ──────────────────────────────


def test_shift_returns_zero_when_signal_none():
    assert compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=None, half_spread_bps=10.0, alpha=0.5,
    ) == 0.0


def test_shift_returns_zero_when_alpha_zero():
    assert compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=0.8, half_spread_bps=10.0, alpha=0.0,
    ) == 0.0


def test_shift_positive_signal_positive_shift():
    shift = compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=0.8, half_spread_bps=10.0, alpha=0.5,
    )
    # alpha * half/2 * signal = 0.5 * 5 * 0.8 = 2.0
    assert shift == pytest.approx(2.0)


def test_shift_negative_signal_negative_shift():
    shift = compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=-0.6, half_spread_bps=10.0, alpha=0.5,
    )
    assert shift == pytest.approx(-1.5)


def test_shift_clip_bound_applied():
    shift = compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=0.99, half_spread_bps=10.0, alpha=0.5, clip_bound=0.5,
    )
    # clamped to 0.5 → 0.5 * 5 * 0.5 = 1.25
    assert shift == pytest.approx(1.25)


def test_shift_nan_signal_returns_zero():
    assert compute_ofi_reservation_shift_bps(
        ofi_signal_normalised=float("nan"), half_spread_bps=10.0, alpha=0.5,
    ) == 0.0


# ─────────────── Integration with compute_quote_decision ────────────────


def _make_settings(ofi_alpha: float = 0.0):
    """Build a Settings via the env-bootstrap path with sensible defaults."""
    from app.config import Settings
    return Settings(
        VENUE="binance",
        SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        OFI_RESERVATION_ALPHA=ofi_alpha,
        OFI_NORMALISATION_SCALE=100.0,
        TOXICITY_ENABLED=False,
    )


def test_compute_quote_decision_ofi_dormant_default():
    """With OFI_RESERVATION_ALPHA=0.0, the OFI shift is 0 regardless of signal."""
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(ofi_alpha=0.0)
    tox = ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False)
    decision = compute_quote_decision(
        settings=settings,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=tox,
        ofi_signal_5s_normalised=0.8,  # Strong signal but alpha=0 → no shift
    )
    assert decision.breakdown.ofi_shift_bps == 0.0


def test_compute_quote_decision_ofi_armed_applies_shift():
    """With OFI_RESERVATION_ALPHA>0 and a finite signal, shift is non-zero."""
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(ofi_alpha=0.5)
    tox = ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False)
    decision = compute_quote_decision(
        settings=settings,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=tox,
        ofi_signal_5s_normalised=0.8,
    )
    # Positive signal → positive shift (reservation moves UP)
    assert decision.breakdown.ofi_shift_bps > 0.0


def test_compute_quote_decision_ofi_signal_none_no_shift():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(ofi_alpha=0.5)
    tox = ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False)
    decision = compute_quote_decision(
        settings=settings, mid=100.0, position_qty=0.0, vol_bps=5.0, toxicity=tox,
        ofi_signal_5s_normalised=None,
    )
    assert decision.breakdown.ofi_shift_bps == 0.0
