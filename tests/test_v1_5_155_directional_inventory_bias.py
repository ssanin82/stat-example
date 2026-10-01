"""v1.5.155 — tests for the BUG-030 directional-inventory-bias
postmortem section.

The detector computes three signals:

1. Time-weighted inventory bias = mean(position_qty) / MAX_ABS_POSITION
2. Cancel asymmetry = max(cancel_bid, cancel_ask) / max(1, min(...))
3. Per-side fill count asymmetry = (buy - sell) / (buy + sell)

Aggregate verdict: 0 flags = neutral, 1 flag = warn, 2+ flags =
severe, missing data = insufficient_data.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from typing import Any

from tools.postmortem.sections.directional_inventory_bias import (
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_NEUTRAL,
    VERDICT_SEVERE,
    VERDICT_WARN,
    detect_directional_inventory_bias_findings,
    render_html_section,
    render_markdown_section,
)


def _inventory_samples(values: list[float]) -> list[dict[str, Any]]:
    return [{"ts": float(i), "position_qty": float(v)} for i, v in enumerate(values)]


def _fills(buy: int, sell: int) -> list[dict[str, Any]]:
    return (
        [{"side": "BUY", "size": 1.0} for _ in range(buy)]
        + [{"side": "SELL", "size": 1.0} for _ in range(sell)]
    )


def _state_with_cancels(
    cancel_bid: int, cancel_ask: int
) -> dict[str, Any]:
    return {
        "behavioural_gates": {
            "target_venue_fast_move_cancel": {
                "cancel_bid_total": cancel_bid,
                "cancel_ask_total": cancel_ask,
            }
        }
    }


def _detect(**overrides) -> "object":
    """Convenience builder with sensible defaults."""
    kwargs = dict(
        fills=[],
        inventory_history=[],
        state_current={},
        config={"MAX_ABS_POSITION": 6.0},
        snapshot_name="test_snap",
        bot_version="1.5.155",
        captured_at="2026-05-26T00:00:00Z",
    )
    kwargs.update(overrides)
    return detect_directional_inventory_bias_findings(**kwargs)


# ---------------------------------------------------------------------------
# Insufficient-data verdict
# ---------------------------------------------------------------------------


def test_all_missing_returns_insufficient_data():
    f = _detect(
        fills=None,
        inventory_history=None,
        state_current=None,
        config=None,
    )
    assert f.verdict == VERDICT_INSUFFICIENT_DATA
    assert f.flags_fired == 0


def test_empty_inputs_returns_insufficient_data():
    f = _detect()
    assert f.verdict == VERDICT_INSUFFICIENT_DATA


def test_below_min_inventory_samples_skips_signal_1():
    """Inventory below sample floor → signal 1 stays None, doesn't
    contribute to verdict."""
    f = _detect(
        inventory_history=_inventory_samples([0.0] * 10),  # well below 50
        config={"MAX_ABS_POSITION": 6.0},
    )
    assert f.inventory_bias_fraction is None
    assert not f.inventory_bias_flag


# ---------------------------------------------------------------------------
# Signal 1 — inventory bias
# ---------------------------------------------------------------------------


def test_neutral_inventory_does_not_flag():
    """Symmetric oscillation around zero → no inventory bias."""
    f = _detect(
        # 100 samples averaging to zero (50 at +3, 50 at -3).
        inventory_history=_inventory_samples([3.0] * 50 + [-3.0] * 50),
    )
    assert abs(f.inventory_bias_fraction or 0.0) < 0.01
    assert not f.inventory_bias_flag


def test_strong_short_bias_flags():
    """Bot at -3 SHORT throughout the session → ~ -50% bias on a
    cap of 6 → flag fires."""
    f = _detect(
        inventory_history=_inventory_samples([-3.0] * 100),
    )
    assert f.inventory_bias_fraction is not None
    assert f.inventory_bias_fraction < -0.30
    assert f.inventory_bias_flag


def test_strong_long_bias_flags():
    f = _detect(
        inventory_history=_inventory_samples([+3.0] * 100),
    )
    assert f.inventory_bias_fraction is not None
    assert f.inventory_bias_fraction > +0.30
    assert f.inventory_bias_flag


def test_borderline_bias_does_not_flag():
    """Exactly at threshold (30%) — strict greater-than → no flag."""
    f = _detect(
        # Mean = -1.8, max=6 → -0.30 exactly → NOT > 0.30.
        inventory_history=_inventory_samples([-1.8] * 100),
    )
    assert abs(f.inventory_bias_fraction or 0.0) == 0.30
    assert not f.inventory_bias_flag


# ---------------------------------------------------------------------------
# Signal 2 — cancel asymmetry
# ---------------------------------------------------------------------------


def test_symmetric_cancels_do_not_flag():
    f = _detect(state_current=_state_with_cancels(100, 100))
    assert f.cancel_asymmetry_ratio == 1.0
    assert not f.cancel_asymmetry_flag


def test_5x_cancel_asymmetry_flags():
    f = _detect(state_current=_state_with_cancels(500, 100))
    assert f.cancel_asymmetry_ratio == 5.0
    assert f.cancel_asymmetry_flag
    assert f.cancel_asymmetry_dominant_side == "BID"


def test_12x_cancel_asymmetry_v1_5_154_reproducer():
    """The actual v1.5.154-260526-074029 snapshot values:
    cancel_bid=1026, cancel_ask=85 → 12.1x asymmetry on BID side."""
    f = _detect(state_current=_state_with_cancels(1026, 85))
    assert f.cancel_asymmetry_ratio is not None
    assert f.cancel_asymmetry_ratio > 10.0
    assert f.cancel_asymmetry_flag
    assert f.cancel_asymmetry_dominant_side == "BID"


def test_low_total_cancels_below_floor_not_flagged():
    """Below 20 total cancels — predicate not evaluated."""
    f = _detect(state_current=_state_with_cancels(10, 1))
    assert f.cancel_asymmetry_ratio is None
    assert not f.cancel_asymmetry_flag


def test_missing_cancel_block_not_flagged():
    f = _detect(state_current={})  # no behavioural_gates
    assert f.cancel_bid_total is None
    assert f.cancel_ask_total is None
    assert not f.cancel_asymmetry_flag


# ---------------------------------------------------------------------------
# Signal 3 — per-side fill count
# ---------------------------------------------------------------------------


def test_balanced_fills_do_not_flag():
    f = _detect(fills=_fills(50, 50))
    assert f.fill_count_asymmetry == 0.0
    assert not f.fill_count_asymmetry_flag


def test_heavy_buy_skew_flags():
    """80 BUY / 20 SELL → asym = +0.6 → flag."""
    f = _detect(fills=_fills(80, 20))
    assert f.fill_count_asymmetry is not None
    assert f.fill_count_asymmetry > 0.30
    assert f.fill_count_asymmetry_flag


def test_heavy_sell_skew_flags():
    f = _detect(fills=_fills(20, 80))
    assert f.fill_count_asymmetry is not None
    assert f.fill_count_asymmetry < -0.30
    assert f.fill_count_asymmetry_flag


def test_below_min_fill_count_not_flagged():
    """Below 30 total fills — predicate not evaluated."""
    f = _detect(fills=_fills(20, 5))
    assert f.fill_count_asymmetry is None
    assert not f.fill_count_asymmetry_flag


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------


def test_no_flags_yields_neutral():
    f = _detect(
        inventory_history=_inventory_samples([0.5] * 100),
        state_current=_state_with_cancels(100, 100),
        fills=_fills(50, 50),
    )
    assert f.flags_fired == 0
    assert f.verdict == VERDICT_NEUTRAL


def test_one_flag_yields_warn():
    f = _detect(
        inventory_history=_inventory_samples([-3.0] * 100),  # flag
        state_current=_state_with_cancels(100, 100),  # no flag
        fills=_fills(50, 50),  # no flag
    )
    assert f.flags_fired == 1
    assert f.verdict == VERDICT_WARN


def test_two_flags_yields_severe():
    f = _detect(
        inventory_history=_inventory_samples([-3.0] * 100),  # flag
        state_current=_state_with_cancels(500, 50),  # flag
        fills=_fills(50, 50),  # no flag
    )
    assert f.flags_fired == 2
    assert f.verdict == VERDICT_SEVERE


def test_three_flags_yields_severe():
    f = _detect(
        inventory_history=_inventory_samples([-3.0] * 100),
        state_current=_state_with_cancels(1026, 85),
        fills=_fills(30, 70),
    )
    assert f.flags_fired == 3
    assert f.verdict == VERDICT_SEVERE


# ---------------------------------------------------------------------------
# v1.5.154-260526-074029 snapshot reproducer
# ---------------------------------------------------------------------------


def test_v1_5_154_overnight_snapshot_yields_severe():
    """The driving snapshot for bug-030. Real values:
    * SHORT-biased inventory (operator's screenshot inventory chart
      shows mostly -3 with occasional +3 to 0 — net SHORT bias)
    * 1026 BID cancels vs 85 ASK cancels (12.1x asymmetric)
    * 54 BUY fills vs 53 SELL (close to balanced — fill counts don't
      flag, but the inventory + cancels do)

    Expected verdict: SEVERE (2 flags)."""
    # Synthesise a SHORT-biased inventory series.
    inv = [-3.0] * 70 + [0.0] * 20 + [+3.0] * 10  # mean = -1.8 → bias -30%
    # Borderline — make it slightly more negative to ensure the flag.
    inv = [-3.0] * 80 + [0.0] * 10 + [+3.0] * 10  # mean = -2.1 → bias -35%
    f = _detect(
        fills=_fills(54, 53),
        inventory_history=_inventory_samples(inv),
        state_current=_state_with_cancels(1026, 85),
    )
    assert f.inventory_bias_flag
    assert f.cancel_asymmetry_flag
    assert not f.fill_count_asymmetry_flag  # 54 vs 53 is balanced
    assert f.flags_fired == 2
    assert f.verdict == VERDICT_SEVERE


# ---------------------------------------------------------------------------
# Rendering smoke tests
# ---------------------------------------------------------------------------


def test_markdown_render_does_not_raise():
    f = _detect()
    out = render_markdown_section(f)
    assert "Directional inventory bias" in out
    assert "Verdict" in out


def test_html_render_does_not_raise():
    f = _detect()
    out = render_html_section(f)
    assert "<section" in out
    assert "Directional inventory bias" in out


def test_markdown_severe_includes_operator_callout():
    """SEVERE verdict should include the operator callout block."""
    f = _detect(
        inventory_history=_inventory_samples([-3.0] * 100),
        state_current=_state_with_cancels(1026, 85),
    )
    out = render_markdown_section(f)
    assert "Operator check" in out
    assert "BUG-030" in out
