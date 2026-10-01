"""Tests for the two Tier-2 calibration items shipped together:

* ``#1`` — ``apply_inventory_high_adding_side_buffer``: pushes the
  *adding-side* quote behind the touch when ``|position|/cap`` is at
  or above ``QUOTE_INVENTORY_PRESSURE_PCT``.
* ``#5`` — ``markout_adverse_bps_for_side`` + ``MarkoutAdverseTracker``:
  pre-cancel signal for resting quotes whose price has drifted
  adverse to current mid for at least the configured duration.

Both default off; SUI's profile keeps them off until post-colo data
is available. Tests cover both the helpers in isolation AND the
default-off semantics so the SUI profile stays a no-op until
explicitly opted in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.quote_aging import (
    MarkoutAdverseTracker,
    apply_inventory_high_adding_side_buffer,
    markout_adverse_bps_for_side,
)
from tests.settings_helpers import UnitTestSettings


def _settings(
    *,
    buffer_ticks: float = 0.0,
    pressure_pct: float = 0.5,
    max_abs_position: float = 10.0,
) -> UnitTestSettings:
    return UnitTestSettings(
        max_abs_position=max_abs_position,
        quote_inventory_pressure_pct=pressure_pct,
        inventory_high_adding_side_buffer_ticks=buffer_ticks,
        # Disable the touch-relax floor so the threshold under test is
        # the only gate firing — the buffer logic shares the same
        # ``inventory_pressure_active`` predicate.
        inventory_fair_touch_relax_min_util_pct=0.0,
    )


def _wo(
    *,
    side: Side,
    price: float,
    status: OrderStatus = OrderStatus.ACKED,
    order_id_local: str = "ord-1",
) -> WorkingOrder:
    return WorkingOrder(
        order_id_local=order_id_local,
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=side,
        price=price,
        size=0.1,
        post_only=True,
        status=status,
        ts_ack=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# =============================================================================
# #1 — apply_inventory_high_adding_side_buffer
# =============================================================================


def test_inv_buffer_disabled_by_default_no_op() -> None:
    s = _settings(buffer_ticks=0.0)
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=8.0,  # 80% util — well above 50% pressure threshold
        bid_px=99.99,
        ask_px=100.01,
        best_bid=100.0,
        best_ask=100.0,
        tick=0.01,
    )
    assert bid == 99.99
    assert ask == 100.01
    assert diag == {}


def test_inv_buffer_long_buffers_bid_only() -> None:
    s = _settings(buffer_ticks=1.0, pressure_pct=0.5)
    # position +8 / cap 10 = 80% util (>= 50%) — bid is adding side.
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=8.0,
        bid_px=100.0,
        ask_px=100.05,
        best_bid=100.0,
        best_ask=100.05,
        tick=0.01,
    )
    # Bid should be pushed to best_bid - 1 tick = 99.99.
    assert bid == 99.99
    # Ask is the reducing side — left untouched.
    assert ask == 100.05
    assert diag.get("inventory_high_adding_side_bid_buffered") is True
    assert "inventory_high_adding_side_ask_buffered" not in diag


def test_inv_buffer_short_buffers_ask_only() -> None:
    s = _settings(buffer_ticks=2.0, pressure_pct=0.5)
    # position -7 / cap 10 = 70% util — ask is adding side.
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=-7.0,
        bid_px=99.95,
        ask_px=100.0,
        best_bid=99.95,
        best_ask=100.0,
        tick=0.01,
    )
    # Bid is reducing side — untouched.
    assert bid == 99.95
    # Ask should be pushed to best_ask + 2 ticks = 100.02.
    assert ask == 100.02
    assert diag.get("inventory_high_adding_side_ask_buffered") is True
    assert "inventory_high_adding_side_bid_buffered" not in diag


def test_inv_buffer_below_pressure_threshold_no_op() -> None:
    s = _settings(buffer_ticks=1.0, pressure_pct=0.5)
    # 30% util — below 50% pressure threshold.
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=3.0,
        bid_px=100.0,
        ask_px=100.05,
        best_bid=100.0,
        best_ask=100.05,
        tick=0.01,
    )
    assert bid == 100.0
    assert ask == 100.05
    assert diag == {}


def test_inv_buffer_flat_position_no_op() -> None:
    """Flat position has no adding side — nothing to buffer."""
    s = _settings(buffer_ticks=1.0, pressure_pct=0.01)  # threshold so low any nonzero pos triggers
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=0.0,
        bid_px=100.0,
        ask_px=100.01,
        best_bid=100.0,
        best_ask=100.01,
        tick=0.01,
    )
    assert bid == 100.0
    assert ask == 100.01
    assert diag == {}


def test_inv_buffer_does_not_loosen_already_behind_quote() -> None:
    """If the model price is already behind the buffer target, leave it.
    The buffer is a *minimum* defensive distance, not a setpoint."""
    s = _settings(buffer_ticks=1.0, pressure_pct=0.5)
    # Long position, bid already 3 ticks behind touch (97 vs 100).
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=8.0,
        bid_px=99.97,  # already 3 ticks behind
        ask_px=100.05,
        best_bid=100.0,
        best_ask=100.05,
        tick=0.01,
    )
    assert bid == 99.97  # untouched — already behind further than buffer wants
    assert ask == 100.05
    # No diag entry because buffer didn't actually move anything.
    assert "inventory_high_adding_side_bid_buffered" not in diag


def test_inv_buffer_no_best_bid_skips_bid_side() -> None:
    s = _settings(buffer_ticks=1.0, pressure_pct=0.5)
    bid, ask, diag = apply_inventory_high_adding_side_buffer(
        settings=s,
        position_qty=8.0,  # long → bid is adding side
        bid_px=100.0,
        ask_px=100.05,
        best_bid=None,
        best_ask=100.05,
        tick=0.01,
    )
    # Without best_bid we can't compute a target; stay put.
    assert bid == 100.0


# =============================================================================
# #5 — markout_adverse_bps_for_side helper
# =============================================================================


def test_markout_bid_above_mid_is_adverse() -> None:
    wo = _wo(side=Side.BUY, price=100.05)
    # mid 100.0, bid 100.05 → bid is 5 bps above mid → adverse +5.0.
    bps = markout_adverse_bps_for_side(side=Side.BUY, working=wo, mid=100.0)
    assert bps is not None
    assert abs(bps - 5.0) < 1e-6


def test_markout_bid_below_mid_is_favorable() -> None:
    wo = _wo(side=Side.BUY, price=99.95)
    # mid 100.0, bid 99.95 → bid 5 bps below mid → favorable (negative bps).
    bps = markout_adverse_bps_for_side(side=Side.BUY, working=wo, mid=100.0)
    assert bps is not None
    assert bps < 0
    assert abs(bps - (-5.0)) < 1e-6


def test_markout_ask_below_mid_is_adverse() -> None:
    wo = _wo(side=Side.SELL, price=99.95)
    # mid 100.0, ask 99.95 → ask 5 bps below mid → adverse +5.0.
    bps = markout_adverse_bps_for_side(side=Side.SELL, working=wo, mid=100.0)
    assert bps is not None
    assert abs(bps - 5.0) < 1e-6


def test_markout_ask_above_mid_is_favorable() -> None:
    wo = _wo(side=Side.SELL, price=100.05)
    bps = markout_adverse_bps_for_side(side=Side.SELL, working=wo, mid=100.0)
    assert bps is not None
    assert bps < 0


def test_markout_no_working_order_returns_none() -> None:
    assert markout_adverse_bps_for_side(side=Side.BUY, working=None, mid=100.0) is None


def test_markout_pre_acked_order_returns_none() -> None:
    """SENT (pre-ack) orders aren't truly resting yet — exclude them."""
    wo = _wo(side=Side.BUY, price=100.05, status=OrderStatus.SENT)
    assert markout_adverse_bps_for_side(side=Side.BUY, working=wo, mid=100.0) is None


def test_markout_invalid_mid_returns_none() -> None:
    wo = _wo(side=Side.BUY, price=100.0)
    assert markout_adverse_bps_for_side(side=Side.BUY, working=wo, mid=0.0) is None
    assert markout_adverse_bps_for_side(side=Side.BUY, working=wo, mid=None) is None


# =============================================================================
# #5 — MarkoutAdverseTracker stateful timer
# =============================================================================


def test_tracker_disabled_threshold_zero_never_fires() -> None:
    tr = MarkoutAdverseTracker()
    wo = _wo(side=Side.BUY, price=100.05)
    fire, elapsed = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=10.0,  # very adverse
        threshold_bps=0.0,  # but feature off
        duration_seconds=2.0,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert fire is False
    assert elapsed is None


def test_tracker_disabled_duration_zero_never_fires() -> None:
    tr = MarkoutAdverseTracker()
    wo = _wo(side=Side.BUY, price=100.05)
    fire, _ = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=10.0,
        threshold_bps=1.5,
        duration_seconds=0.0,  # off
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert fire is False


def test_tracker_arms_on_first_breach_fires_after_duration() -> None:
    tr = MarkoutAdverseTracker()
    wo = _wo(side=Side.BUY, price=100.05)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fire, elapsed = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0,
    )
    assert fire is False  # first observation arms the timer
    assert elapsed == 0.0

    # 1 second later — still above threshold but under duration.
    fire2, elapsed2 = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=1.0),
    )
    assert fire2 is False
    assert elapsed2 == 1.0

    # 2.5 seconds — past duration, should fire.
    fire3, elapsed3 = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=2.5),
    )
    assert fire3 is True
    assert elapsed3 == 2.5


def test_tracker_resets_when_drift_drops_below_threshold() -> None:
    tr = MarkoutAdverseTracker()
    wo = _wo(side=Side.BUY, price=100.05)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Arm at t0.
    tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0,
    )
    # Drift drops at t0+1 → reset.
    fire, elapsed = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=0.5,  # below threshold
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=1.0),
    )
    assert fire is False
    assert elapsed is None
    # Drift comes back at t0+1.5 — timer must restart fresh.
    fire2, elapsed2 = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=1.5),
    )
    assert fire2 is False
    assert elapsed2 == 0.0
    # Need full duration from this NEW arming, not from t0.
    fire3, _ = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=3.0),  # 1.5 s elapsed since rearm — not enough
    )
    assert fire3 is False
    fire4, _ = tr.evaluate(
        side=Side.BUY,
        working=wo,
        adverse_bps=3.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=3.6),  # 2.1 s elapsed since rearm
    )
    assert fire4 is True


def test_tracker_resets_when_resting_order_changes() -> None:
    """A new resting order id means a fresh quote — reset the timer."""
    tr = MarkoutAdverseTracker()
    wo1 = _wo(side=Side.BUY, price=100.05, order_id_local="A")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Arm + saturate timer on order A.
    tr.evaluate(
        side=Side.BUY,
        working=wo1,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0,
    )
    # 5 s elapsed — would fire on order A.
    fire_a, _ = tr.evaluate(
        side=Side.BUY,
        working=wo1,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=5.0),
    )
    assert fire_a is True

    # Now order changes to B — new id, fresh timer, no fire on first obs.
    wo2 = _wo(side=Side.BUY, price=100.05, order_id_local="B")
    fire_b, elapsed_b = tr.evaluate(
        side=Side.BUY,
        working=wo2,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=5.1),
    )
    assert fire_b is False
    assert elapsed_b == 0.0


def test_tracker_clears_when_no_resting_order() -> None:
    tr = MarkoutAdverseTracker()
    fire, elapsed = tr.evaluate(
        side=Side.BUY,
        working=None,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert fire is False
    assert elapsed is None


# =============================================================================
# Integration — verify wiring into QuoteEngine.build_quotes
# =============================================================================


def _engine_with(**overrides):
    """Mirrors tests/test_quote_engine_inventory_bias.py setup."""
    import os
    import tempfile
    import uuid
    from pathlib import Path

    from app.quote_engine import QuoteEngine
    from tests.exchange_client_mocks import mock_mm_client

    path = Path(tempfile.gettempdir()) / f"mm_calib_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 10.0,
        "MAX_POSITION_NOTIONAL_USD": 100_000.0,
        # Suppress the inventory execution bias so #1 buffer is the
        # only adding-side defense exercised in the wiring test.
        "INVENTORY_EXEC_BIAS_RATIO": 0.0,
        "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.0,
        # Disable the "bid moves toward touch on long" relax so the
        # buffer isn't fighting another aggressor.
        "INVENTORY_FAIR_TOUCH_RELAX_MIN_UTIL_PCT": 0.0,
        "QUOTE_INVENTORY_PRESSURE_PCT": 0.5,
        # We want the engine to use the model bid/ask without the
        # market-anchored spread re-anchor (which overwrites bid_px /
        # ask_px to mid ± half_spread); that path is exercised by
        # other tests. Disabling the anchor keeps the input prices
        # explicit so the buffer's effect on bid_px is observable.
        "NORMAL_MM_USE_MARKET_SPREAD_ANCHOR": False,
        # Disable the economic spread floor so the buffer is the only
        # thing modifying bid/ask away from the model touch values.
        # With the default 8 bps floor, the engine would push bid to
        # mid - 2.4 = 2998.1, which is already far behind the buffer
        # target (best_bid - 2 ticks = 2999.98) → buffer no-op.
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.0,
        "ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        # MIN_HALF_SPREAD_BPS feeds into ``compute_effective_min_half_spread_bps``
        # via ``base = max(min_half_spread_bps, econ)`` — keep it 0 so
        # the profitability floor doesn't widen the model bid/ask.
        "MIN_HALF_SPREAD_BPS": 0.0,
        # Disable aging tightening so the buffer is the sole price-mover.
        "QUOTE_AGING_ENABLED": False,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    return eng, s, path


def _ctx_with(
    s,
    *,
    position_qty: float,
    resting_bid=None,
    resting_ask=None,
    quoted_bid: float = 3000.0,  # at best_bid by default — buffer can move it
    quoted_ask: float = 3001.0,  # at best_ask by default
):
    from app.enums import ActiveSides, RiskAction
    from app.models import QuoteDecision
    from app.quote_engine import QuoteBuildContext
    from app.utils.time import utc_now as _utc_now
    from tests.test_quote_reprice_maintenance import _fresh_market

    decision = QuoteDecision(
        ts=_utc_now(),
        symbol="ETH",
        mid_price=3000.5,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=3000.5,
        target_spread_bps=2.0,  # tight — keeps bid/ask near the touch
        target_bid=quoted_bid,
        target_ask=quoted_ask,
        quoted_bid=quoted_bid,
        quoted_ask=quoted_ask,
        quoted_bid_sz=0.01,
        quoted_ask_sz=0.01,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="ok",
        quote_cycle_id="qc-test",
    )
    return QuoteBuildContext(
        decision=decision,
        market=_fresh_market(s),
        risk_action=RiskAction.ALLOW,
        bid_mult=1.0,
        ask_mult=1.0,
        spread_add_bps=0.0,
        position_qty=position_qty,
        position_notional=abs(position_qty) * 3000.5,
        resting_bid=resting_bid,
        resting_ask=resting_ask,
    )


def test_engine_inv_buffer_off_by_default_bid_at_touch() -> None:
    eng, s, path = _engine_with(INVENTORY_HIGH_ADDING_SIDE_BUFFER_TICKS=0.0)
    out = eng.build_quotes(_ctx_with(s, position_qty=8.0))
    # Default off — no buffer telemetry, bid not yanked behind touch.
    assert "inventory_high_adding_side_bid_buffered" not in out.telemetry
    path.unlink(missing_ok=True)


def test_engine_inv_buffer_pushes_bid_behind_touch_when_long() -> None:
    """Wiring check: with the feature on, telemetry shows the bid was
    buffered on a long position. Price arithmetic is covered by the
    unit tests of ``apply_inventory_high_adding_side_buffer`` —
    here we only care that the helper is *called* by build_quotes
    and that its diag flows into telemetry."""
    eng, s, path = _engine_with(INVENTORY_HIGH_ADDING_SIDE_BUFFER_TICKS=2.0)
    out = eng.build_quotes(_ctx_with(s, position_qty=8.0))  # 80% util long
    assert out.telemetry.get("inventory_high_adding_side_bid_buffered") is True
    assert out.telemetry.get("inventory_high_adding_side_bid_buffer_ticks") == 2.0
    # Adding side (bid) buffered; reducing side (ask) untouched.
    assert "inventory_high_adding_side_ask_buffered" not in out.telemetry
    path.unlink(missing_ok=True)


def test_engine_inv_buffer_pushes_ask_behind_touch_when_short() -> None:
    eng, s, path = _engine_with(INVENTORY_HIGH_ADDING_SIDE_BUFFER_TICKS=1.0)
    out = eng.build_quotes(_ctx_with(s, position_qty=-8.0))
    assert out.telemetry.get("inventory_high_adding_side_ask_buffered") is True
    assert out.telemetry.get("inventory_high_adding_side_ask_buffer_ticks") == 1.0
    assert "inventory_high_adding_side_bid_buffered" not in out.telemetry
    path.unlink(missing_ok=True)


def test_engine_markout_aging_off_by_default_no_cancel() -> None:
    """Default (threshold=0, duration=0) must never cancel — even with a
    resting bid sitting deeply above mid."""
    eng, s, path = _engine_with()
    # Resting bid at 3010 vs market mid ~3000.5 → 31 bps adverse, way
    # above any reasonable threshold. With the feature off, the engine
    # should still produce a bid order (or a reason that ISN'T markout).
    rb = _wo(side=Side.BUY, price=3010.0, order_id_local="bid-1")
    out = eng.build_quotes(_ctx_with(s, position_qty=0.0, resting_bid=rb))
    assert out.telemetry.get("markout_adverse_cancel_bid") is False
    assert out.telemetry.get("quote_engine_bid_reason") != "markout_adverse_cancel"
    path.unlink(missing_ok=True)


def test_engine_markout_aging_fires_after_sustained_drift(monkeypatch) -> None:
    """With threshold=1.5 / duration=2.0, two cycles ≥2 s apart with the
    same resting bid in adverse territory should result in the second
    cycle returning bid_order=None and reason ``markout_adverse_cancel``."""
    import app.quote_engine as qe_mod

    eng, s, path = _engine_with(
        QUOTE_AGING_MARKOUT_ADVERSE_BPS_THRESHOLD=1.5,
        QUOTE_AGING_MARKOUT_ADVERSE_DURATION_SECONDS=2.0,
    )
    rb = _wo(side=Side.BUY, price=3010.0, order_id_local="bid-1")

    # Patch the imported ``utc_now`` symbol inside quote_engine — the
    # engine uses ``from app.utils.time import utc_now`` so the bound
    # reference lives on the engine module, not on app.utils.time.
    base_t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fake_now = [base_t]
    monkeypatch.setattr(qe_mod, "utc_now", lambda: fake_now[0])
    try:
        # Cycle 1 — arms the timer.
        out1 = eng.build_quotes(_ctx_with(s, position_qty=0.0, resting_bid=rb))
        assert out1.telemetry.get("markout_adverse_cancel_bid") is False
        # 2.5 s later — past the 2 s duration → should fire.
        fake_now[0] = base_t + timedelta(seconds=2.5)
        out2 = eng.build_quotes(_ctx_with(s, position_qty=0.0, resting_bid=rb))
        assert out2.telemetry.get("markout_adverse_cancel_bid") is True
        assert out2.bid_order is None
        assert out2.telemetry.get("quote_engine_bid_reason") == "markout_adverse_cancel"
    finally:
        path.unlink(missing_ok=True)


def test_tracker_per_side_isolation() -> None:
    """Bid-side timer must not interfere with ask-side timer."""
    tr = MarkoutAdverseTracker()
    wb = _wo(side=Side.BUY, price=100.05, order_id_local="bid-1")
    wa = _wo(side=Side.SELL, price=99.95, order_id_local="ask-1")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Arm bid timer.
    tr.evaluate(
        side=Side.BUY,
        working=wb,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0,
    )
    # Ask side observes adverse — separate timer.
    fire, elapsed = tr.evaluate(
        side=Side.SELL,
        working=wa,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=1.0),
    )
    # Ask just armed at t0+1; not fired yet.
    assert fire is False
    assert elapsed == 0.0
    # Bid side has 1 s elapsed — not fired yet either.
    fire_bid, elapsed_bid = tr.evaluate(
        side=Side.BUY,
        working=wb,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=1.0),
    )
    assert fire_bid is False
    assert elapsed_bid == 1.0
    # Bid eventually fires at 2.5 s.
    fire_bid_late, _ = tr.evaluate(
        side=Side.BUY,
        working=wb,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=2.5),
    )
    assert fire_bid_late is True
    # Ask side at 2.5 s only has 1.5 s elapsed since its arming — still off.
    fire_ask, elapsed_ask = tr.evaluate(
        side=Side.SELL,
        working=wa,
        adverse_bps=5.0,
        threshold_bps=1.5,
        duration_seconds=2.0,
        now=t0 + timedelta(seconds=2.5),
    )
    assert fire_ask is False
    assert elapsed_ask == 1.5
