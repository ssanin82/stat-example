"""Phase 2K.5 — favorable-exit predicate for the
``adaptive_spread_widen`` overlay.

Unit-tests the per-reason ``adaptive_spread_widen_signal_cleared``
helper in ``app/quoting.py``. The integration with the bot's tick
loop (dwell timer + ceiling attribution + edge detection) lives in
``app/bot.py``; this test file covers the pure-function predicate
that drives the decision.
"""

from __future__ import annotations

from app.models import ToxicitySnapshot
from app.quoting import adaptive_spread_widen_signal_cleared
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
        "TOXICITY_MARKOUT_SOFT_BPS": 2.0,
        "TOXICITY_ONE_SIDED_FILL_RATIO": 0.75,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _tox(**overrides) -> ToxicitySnapshot:
    base = dict(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
        delayed_markout_sample_count=10,
        adverse_uses_delayed_markouts=True,
        buy_side_avg_markout_bps=None,
        sell_side_avg_markout_bps=None,
        buy_side_fill_count=0,
        sell_side_fill_count=0,
    )
    base.update(overrides)
    return ToxicitySnapshot(**base)


# ---------------------------------------------------------------------------
# toxicity_hard / toxicity_soft
# ---------------------------------------------------------------------------


def test_toxicity_hard_cleared_when_trigger_false() -> None:
    s = _settings()
    tox = _tox(hard_trigger=False)
    assert adaptive_spread_widen_signal_cleared(
        s, "toxicity_hard", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_toxicity_hard_not_cleared_while_trigger_true() -> None:
    s = _settings()
    tox = _tox(hard_trigger=True)
    assert not adaptive_spread_widen_signal_cleared(
        s, "toxicity_hard", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_toxicity_soft_cleared_when_trigger_false() -> None:
    s = _settings()
    tox = _tox(soft_trigger=False)
    assert adaptive_spread_widen_signal_cleared(
        s, "toxicity_soft", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_toxicity_soft_not_cleared_while_trigger_true() -> None:
    s = _settings()
    tox = _tox(soft_trigger=True)
    assert not adaptive_spread_widen_signal_cleared(
        s, "toxicity_soft", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


# ---------------------------------------------------------------------------
# markout_adverse
# ---------------------------------------------------------------------------


def test_markout_adverse_cleared_when_no_delayed_samples() -> None:
    """If the bot is using the legacy (non-delayed) markout path, the
    arming condition (which requires delayed markouts) can't be true,
    so the predicate considers the signal cleared by definition."""
    s = _settings()
    tox = _tox(
        adverse_uses_delayed_markouts=False,
        avg_adverse_markout_bps=-5.0,
    )
    assert adaptive_spread_widen_signal_cleared(
        s, "markout_adverse", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_markout_adverse_cleared_when_markout_recovered() -> None:
    """Threshold is -2.0 bps; signal cleared when avg > -2.0."""
    s = _settings()
    tox = _tox(
        adverse_uses_delayed_markouts=True,
        avg_adverse_markout_bps=-1.5,
    )
    assert adaptive_spread_widen_signal_cleared(
        s, "markout_adverse", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_markout_adverse_not_cleared_while_still_below_threshold() -> None:
    s = _settings()
    tox = _tox(
        adverse_uses_delayed_markouts=True,
        avg_adverse_markout_bps=-3.0,
    )
    assert not adaptive_spread_widen_signal_cleared(
        s, "markout_adverse", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


# ---------------------------------------------------------------------------
# one_sided_ratio
# ---------------------------------------------------------------------------


def test_one_sided_ratio_cleared_when_below_threshold() -> None:
    """Threshold is 0.75; signal cleared when ratio < 0.75."""
    s = _settings()
    tox = _tox(one_sided_fill_ratio=0.5)
    assert adaptive_spread_widen_signal_cleared(
        s, "one_sided_ratio", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_one_sided_ratio_not_cleared_at_or_above_threshold() -> None:
    s = _settings()
    tox = _tox(one_sided_fill_ratio=0.8)
    assert not adaptive_spread_widen_signal_cleared(
        s, "one_sided_ratio", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


# ---------------------------------------------------------------------------
# quote_quality
# ---------------------------------------------------------------------------


def test_quote_quality_cleared_when_signal_false() -> None:
    s = _settings()
    tox = _tox()
    assert adaptive_spread_widen_signal_cleared(
        s, "quote_quality", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_quote_quality_not_cleared_while_signal_true() -> None:
    s = _settings()
    tox = _tox()
    assert not adaptive_spread_widen_signal_cleared(
        s, "quote_quality", tox,
        quote_quality_signal=True, slow_trend_signal=False,
    )


# ---------------------------------------------------------------------------
# slow_trend
# ---------------------------------------------------------------------------


def test_slow_trend_cleared_when_signal_false() -> None:
    s = _settings()
    tox = _tox()
    assert adaptive_spread_widen_signal_cleared(
        s, "slow_trend", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


def test_slow_trend_not_cleared_while_signal_true() -> None:
    s = _settings()
    tox = _tox()
    assert not adaptive_spread_widen_signal_cleared(
        s, "slow_trend", tox,
        quote_quality_signal=False, slow_trend_signal=True,
    )


# ---------------------------------------------------------------------------
# Unknown / safety fallthrough
# ---------------------------------------------------------------------------


def test_unknown_reason_falls_to_ceiling_only_clearing() -> None:
    """An unrecognised arm-time reason (e.g. legacy snapshot, future
    reason not yet covered) returns False — favorable-exit doesn't
    fire, the cooldown clears via the MAX ceiling only. Conservative
    default."""
    s = _settings()
    tox = _tox()
    assert not adaptive_spread_widen_signal_cleared(
        s, "residual_decay", tox,  # not yet a known reason
        quote_quality_signal=False, slow_trend_signal=False,
    )
    assert not adaptive_spread_widen_signal_cleared(
        s, "", tox,
        quote_quality_signal=False, slow_trend_signal=False,
    )


# ---------------------------------------------------------------------------
# State surface: new BotState fields exist with correct defaults
# ---------------------------------------------------------------------------


def test_botstate_phase2k5_fields_initialised() -> None:
    """The four new BotState fields must initialise to safe values
    so the bot's first-tick favorable-exit / ceiling-attribution
    logic operates on known initial state."""
    from app.state import BotState
    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": False,
        "EXCHANGE": "okx",
        "SYMBOL": "ETH-USDT-SWAP",
    })
    bs = BotState(s)
    assert bs.adaptive_spread_widen_favorable_dwell_started_mono is None
    assert bs.adaptive_spread_widen_cleared_via_favorable_total == 0
    assert bs.adaptive_spread_widen_cleared_via_ceiling_total == 0
    assert bs.adaptive_spread_widen_was_active_last_tick is False
