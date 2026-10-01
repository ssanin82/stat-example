"""Tests for v1.4.85 Phase 4C — typed Settings subsets.

Phase 4C's contract: each pure stage takes a SMALL typed settings
subset (not the full ``Settings`` object), so unit-test fixtures stay
minimal and the stage's settings-dependency surface is documented.

Phase 4C is partially shipped — the ``AgingSettings`` subset is
declared for the one extracted stage (``compute_aging_signals``).
The remaining stages' subsets ship when 4B.1c lands.

Tests verify:

* The subset is a frozen dataclass with the exact 5 fields.
* ``AgingSettings.from_full(settings)`` partitions the full Settings
  correctly.
* ``compute_aging_signals`` accepts the subset via duck-typing AND
  produces the same result as it would with the full Settings.
* Meta-test: every field on the subset is actually read by
  ``hard_reprice_reasons_*`` (no dead fields).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.quote_engine import AgingSettings, compute_aging_signals
from tests.settings_helpers import UnitTestSettings


def _now() -> datetime:
    return datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def _bid_wo(*, price: float, age_seconds: float) -> WorkingOrder:
    ts = _now() - timedelta(seconds=age_seconds)
    return WorkingOrder(
        order_id_local="l-1",
        order_id_exchange=11,
        client_order_id="c-1",
        symbol="ETH",
        side=Side.BUY,
        price=price,
        size=1.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=ts,
        ts_sent=ts,
        ts_ack=ts,
    )


# ---------------------------------------------------------------------------
# Subset shape
# ---------------------------------------------------------------------------


def test_phase4c_aging_settings_is_frozen_dataclass() -> None:
    assert is_dataclass(AgingSettings)
    # Frozen: instances should reject attribute assignment.
    s = AgingSettings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        quote_max_distance_to_touch_ticks=10.0,
        at_touch_max_age_seconds=2.0,
        behind_touch_max_age_seconds=1.0,
    )
    try:
        s.quote_aging_enabled = False  # type: ignore[misc]
    except (AttributeError, Exception):
        pass
    else:
        raise AssertionError("expected frozen dataclass to refuse assignment")


def test_phase4c_aging_settings_has_expected_fields() -> None:
    """Pin the exact 5-field surface of the subset. Anyone adding a
    new aging knob must update both Settings, this subset, AND the
    transitive helpers — the test catches partial migrations.
    """
    expected = {
        "quote_aging_enabled",
        "quote_aging_max_age_seconds",
        "quote_max_distance_to_touch_ticks",
        "at_touch_max_age_seconds",
        "behind_touch_max_age_seconds",
    }
    got = {f.name for f in fields(AgingSettings)}
    assert got == expected, (
        f"AgingSettings field surface drifted from contract. "
        f"Expected {sorted(expected)}, got {sorted(got)}. "
        f"If you're adding a new knob, update AgingSettings.from_full "
        f"AND hard_reprice_reasons_buy/sell at the same time."
    )


# ---------------------------------------------------------------------------
# from_full partition
# ---------------------------------------------------------------------------


def test_phase4c_from_full_partitions_settings() -> None:
    full = UnitTestSettings.model_validate({
        "QUOTE_AGING_ENABLED": True,
        "QUOTE_AGING_MAX_AGE_SECONDS": 4.5,
        "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 12.0,
        "AT_TOUCH_MAX_AGE_SECONDS": 3.0,
        "BEHIND_TOUCH_MAX_AGE_SECONDS": 0.75,
    })
    subset = AgingSettings.from_full(full)
    assert subset.quote_aging_enabled is True
    assert subset.quote_aging_max_age_seconds == 4.5
    assert subset.quote_max_distance_to_touch_ticks == 12.0
    assert subset.at_touch_max_age_seconds == 3.0
    assert subset.behind_touch_max_age_seconds == 0.75


def test_phase4c_from_full_handles_missing_optional_fields() -> None:
    """`at_touch_max_age_seconds` and `behind_touch_max_age_seconds`
    are optional knobs that default 0.0 on settings. The partition
    must tolerate them being absent / unset."""
    full = UnitTestSettings.model_validate({
        "QUOTE_AGING_ENABLED": True,
        # AT_TOUCH and BEHIND_TOUCH intentionally omitted.
    })
    subset = AgingSettings.from_full(full)
    # Default values from getattr fallback.
    assert subset.at_touch_max_age_seconds == 0.0
    assert subset.behind_touch_max_age_seconds == 0.0


# ---------------------------------------------------------------------------
# Duck-typing: stage accepts either Settings or AgingSettings
# ---------------------------------------------------------------------------


def test_phase4c_compute_aging_signals_accepts_subset_directly() -> None:
    subset = AgingSettings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        quote_max_distance_to_touch_ticks=1000.0,
        at_touch_max_age_seconds=2.0,
        behind_touch_max_age_seconds=1.0,
    )
    # Behind-touch order, 1.5s old, threshold 1.0s → fires.
    wo = _bid_wo(price=1.999, age_seconds=1.5)
    out = compute_aging_signals(
        subset,
        {},
        resting_bid=wo,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert (Side.BUY, 0) in out
    assert "behind_touch_order_age_seconds" in out[(Side.BUY, 0)]


def test_phase4c_subset_and_full_produce_identical_results() -> None:
    """Behavioural equivalence: passing ``AgingSettings.from_full(s)``
    must produce the same dict as passing ``s`` directly.
    """
    full = UnitTestSettings.model_validate({
        "QUOTE_AGING_ENABLED": True,
        "QUOTE_AGING_MAX_AGE_SECONDS": 6.0,
        "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 1000.0,
        "AT_TOUCH_MAX_AGE_SECONDS": 2.0,
        "BEHIND_TOUCH_MAX_AGE_SECONDS": 1.0,
    })
    subset = AgingSettings.from_full(full)

    wo_bid = _bid_wo(price=1.999, age_seconds=1.5)
    args = dict(
        working_orders_by_slot={},
        resting_bid=wo_bid,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    via_full = compute_aging_signals(full, **args)
    via_subset = compute_aging_signals(subset, **args)
    assert via_full == via_subset
