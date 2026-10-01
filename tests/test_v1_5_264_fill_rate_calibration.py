"""v1.5.264 — fill-rate calibration batch.

Three env-knob changes shipped together to lift fill rate out of the
~0.7 fills/min floor observed on TON-USDT-SWAP in quiet markets:

* A) AT_TOUCH_MAX_AGE_SECONDS 2.5 → 5.0
     BEHIND_TOUCH_MAX_AGE_SECONDS 1.5 → 3.0
     Lengthen passive order lifetime so orders rest through more
     mid changes before age-driven reprice fires.

* B) LADDER_SIZE_DECAY 0.9 → 1.0  (no decay between rungs)
     MIN_VENUE_NOTIONAL_USD 5.0 → 3.5  (new knob; default 5.0)
     Resurrect rung-1 (behind-touch). Snapshot v1.5.257-260529-170937
     showed 68,605 rung-1 normalize rejections in 64 min — rung-1
     never fires under the old config. The new knob makes the
     synthetic OKX USD min-notional configurable (OKX V5 doesn't
     publish one — the bot synthesises it).

* C) TARGET_VENUE_FAST_MOVE_CANCEL_DRIFT_WINDOW_SECONDS 5.0 → 10.0
     Reduce drift-driven cancel churn (70% cancel-to-place ratio in
     the snapshot).

These tests verify only the new knob (MIN_VENUE_NOTIONAL_USD) since
the other four are pre-existing knobs being retuned. The env-file
retuning is verified by the snapshot acceptance script + the next
session's snapshot.
"""

from __future__ import annotations


def test_min_venue_notional_usd_default_is_5():
    """Default preserves legacy behaviour."""
    from app.config import Settings
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
    )
    assert s.min_venue_notional_usd == 5.0


def test_min_venue_notional_usd_can_be_overridden_via_env():
    """The new knob honours the env alias."""
    from app.config import Settings
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
        MIN_VENUE_NOTIONAL_USD=3.5,
    )
    assert s.min_venue_notional_usd == 3.5


def test_min_venue_notional_usd_rejects_zero_or_negative():
    """The knob is gt=0 — Settings should raise on bad values."""
    import pytest
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(
            VENUE="okx", SYMBOL="TON-USDT-SWAP",
            QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
            MAX_ABS_POSITION=6.0,
            MIN_VENUE_NOTIONAL_USD=0.0,
        )


def test_prod_profile_ships_calibration_knobs():
    """The prod profile must have v1.5.264 calibration applied —
    if someone reverts the .env file by mistake, this test catches it."""
    from pathlib import Path

    env_path = (
        Path(__file__).parent.parent
        / "config"
        / "profiles"
        / "prod.okx.ton.usdt.perp.env"
    )
    text = env_path.read_text(encoding="utf-8")

    def _value(key: str) -> str:
        """Return the LAST occurrence's value (env files allow overrides
        and the deploy process treats the last write as canonical)."""
        last = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                last = line.split("=", 1)[1].strip()
        assert last is not None, f"{key} missing from prod profile"
        return last

    # Calibration A — order lifetime.
    assert _value("AT_TOUCH_MAX_AGE_SECONDS") == "5.0", (
        "v1.5.264 calibration A: AT_TOUCH_MAX_AGE_SECONDS should be 5.0"
    )
    assert _value("BEHIND_TOUCH_MAX_AGE_SECONDS") == "3.0", (
        "v1.5.264 calibration A: BEHIND_TOUCH_MAX_AGE_SECONDS should be 3.0"
    )

    # Calibration B — rung-1 resurrection.
    assert _value("LADDER_SIZE_DECAY") == "1.0", (
        "v1.5.264 calibration B: LADDER_SIZE_DECAY should be 1.0"
    )
    assert _value("MIN_VENUE_NOTIONAL_USD") == "3.0", (
        "v1.5.268 calibration B (refined): MIN_VENUE_NOTIONAL_USD should "
        "be 3.0 — was 3.5 in v1.5.264 but rung-1 was still mostly rejected "
        "because at pos=0 / MAX_ABS_POSITION=6 / rung-0=4 contracts, "
        "rung-1 capacity = 2 contracts × $1.73 = $3.46 (just below $3.5)"
    )

    # Calibration C — drift-cancel window.
    assert _value("TARGET_VENUE_FAST_MOVE_CANCEL_DRIFT_WINDOW_SECONDS") == "10.0", (
        "v1.5.264 calibration C: drift-cancel window should be 10.0"
    )
