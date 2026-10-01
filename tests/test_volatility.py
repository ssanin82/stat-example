from app.volatility import VolatilityEstimator
from tests.settings_helpers import UnitTestSettings


def _vol_settings(samples: int = 8) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "VOL_WINDOW_SAMPLES": samples,
        }
    )


def test_cold_start_no_sigma() -> None:
    est = VolatilityEstimator(_vol_settings())
    assert est.sigma_and_bps() == (None, 0.0)


def test_flat_mids_dedup_prevents_warmup() -> None:
    """1.2.24 fix: pushing the same mid repeatedly must NOT warm up
    the estimator. Pre-fix, the cycle-frequency caller flooded the
    deque with identical values and the variance read as 0 even in
    active markets — three vol-conditioned defenses sat silently
    inert. Dedup at push time keeps the deque at size 1 when the
    venue mid hasn't ticked, so warmup correctly fails and
    sigma_and_bps returns (None, 0)."""
    est = VolatilityEstimator(_vol_settings(6))
    for _ in range(10):
        est.push_mid(100.0)
    sig, bps = est.sigma_and_bps()
    assert sig is None
    assert bps == 0.0


def test_dedup_skips_consecutive_duplicates_only() -> None:
    """A->A->B->B->C should yield deque [A, B, C] — three distinct
    samples, not five. Confirms the dedup doesn't drop genuine
    re-occurrences (e.g. mid bounces back to a previous value)."""
    est = VolatilityEstimator(_vol_settings(4))
    for v in (100.0, 100.0, 101.0, 101.0, 100.0):
        est.push_mid(v)
    # Internal deque: 3 distinct consecutive values
    assert list(est._mids) == [100.0, 101.0, 100.0]


def test_distinct_mids_yield_nonzero_vol() -> None:
    """When the venue mid actually ticks, vol_bps reflects real
    movement. Push 8 mids each 1bp apart from 100 → 100.08; the
    log-return per step is ~1bp, so vol_bps should be ~1 (not 0,
    not undefined)."""
    est = VolatilityEstimator(_vol_settings(6))
    for i in range(8):
        est.push_mid(100.0 + i * 0.01)  # +1 bp per step
    sig, bps = est.sigma_and_bps()
    assert sig is not None
    # Returns are very close to identical (constant 1bp/step) so
    # variance is near zero — the LEVEL of vol is small but it is
    # at least defined, not stuck at 0 from the dedup bug.
    assert bps >= 0.0
    # Now alternate up/down to create non-trivial variance.
    est2 = VolatilityEstimator(_vol_settings(6))
    seq = [100.0, 100.05, 100.0, 100.05, 100.0, 100.05, 100.0, 100.05]
    for v in seq:
        est2.push_mid(v)
    sig2, bps2 = est2.sigma_and_bps()
    assert sig2 is not None
    # ~5bp swings → vol_bps should be in the 4-6 range, definitely > 1.
    assert bps2 > 1.0


def test_phase2_state_mid_change_listener_pushes_to_estimator() -> None:
    """1.2.26 (todo-010 Phase 2): the bot wires
    ``state.add_mid_change_listener(self._vol.push_mid)`` so every
    real mid change from the public-WS BBO handler reaches the
    estimator. This test verifies the wiring end-to-end without
    spinning up the bot.
    """
    from app.state import BotState
    from app.models import BestBidAsk

    settings = _vol_settings(6)
    state = BotState(settings)
    est = VolatilityEstimator(settings)
    state.add_mid_change_listener(est.push_mid)

    # Simulate 8 BBO updates with progressively-higher mids.
    for i in range(8):
        mid = 100.0 + i * 0.05
        market = BestBidAsk(
            symbol=settings.symbol,
            best_bid=mid - 0.005,
            best_ask=mid + 0.005,
            mid_price=mid,
            spread_bps=10.0,
        )
        state.apply_market_book_only(market)
    sig, bps = est.sigma_and_bps()
    assert sig is not None
    assert bps > 0.0
    # Counter sanity: 8 BBO events, 8 distinct mids → 8 mid changes
    # (first BBO counts as a mid change because prev was None).
    assert state.bbo_event_count_session == 8
    assert state.mid_change_count_session == 8


def test_phase2_listener_only_fires_on_real_mid_changes() -> None:
    """If the BBO updates but mid stays the same (e.g. pure size
    change at the touch), the listener must NOT fire — otherwise
    we re-introduce the cycle-rate-flooding bug."""
    from app.state import BotState
    from app.models import BestBidAsk

    settings = _vol_settings(6)
    state = BotState(settings)

    fire_count = 0
    def listener(_mid: float) -> None:
        nonlocal fire_count
        fire_count += 1
    state.add_mid_change_listener(listener)

    # Three BBO events, all the same mid (size-only changes
    # in the real venue would trigger this pattern).
    for _ in range(3):
        market = BestBidAsk(
            symbol=settings.symbol,
            best_bid=99.995,
            best_ask=100.005,
            mid_price=100.0,
            spread_bps=10.0,
        )
        state.apply_market_book_only(market)

    # First BBO: prev=None != new=100.0 → fires once. Subsequent
    # two BBOs: prev=100.0 == new=100.0 → don't fire.
    assert fire_count == 1
    assert state.bbo_event_count_session == 3
    assert state.mid_change_count_session == 1


def test_phase2_listener_exception_is_swallowed() -> None:
    """A misbehaving listener must not break the WS handler. The
    fire path catches Exception, logs, and continues so other
    listeners still get notified."""
    from app.state import BotState
    from app.models import BestBidAsk

    settings = _vol_settings(6)
    state = BotState(settings)

    fired_after_bad = False
    def bad_listener(_mid: float) -> None:
        raise RuntimeError("simulated listener failure")

    def good_listener(_mid: float) -> None:
        nonlocal fired_after_bad
        fired_after_bad = True

    state.add_mid_change_listener(bad_listener)
    state.add_mid_change_listener(good_listener)

    market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=99.995,
        best_ask=100.005,
        mid_price=100.0,
        spread_bps=10.0,
    )
    state.apply_market_book_only(market)
    # Good listener should still fire despite the bad one raising.
    assert fired_after_bad is True
