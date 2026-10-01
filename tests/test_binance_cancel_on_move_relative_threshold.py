"""v1.4.45 — Binance cross-venue cancel threshold becomes RELATIVE.

Before v1.4.45 the cancel threshold was a static config value
(``BINANCE_CANCEL_ON_MOVE_BPS``). If the bot's organic quote distance
(``target_half_spread_bps``) grew larger than the static threshold,
every placed order was born past the threshold and got instant-
canceled — 30+ minutes of zero trading observed 2026-05-18 in
``snapshots/v1.4.44-260518-140038`` with half_spread 13.7 bps and
threshold 10 bps.

v1.4.45 design: effective_threshold = max(static_floor, half_spread
+ buffer). The dynamic component auto-tracks the engine's current
half-spread, so the invariant "threshold > half_spread" holds
across all vol/toxicity regimes by construction.

This file pins the behaviour at the
``_maybe_cancel_on_binance_move`` boundary, NOT inside the wider
end-to-end machinery (those tests already exist in
``test_binance_cross_venue_cancel.py``). The questions answered
here:

  * Buffer disabled (=0) → static-only (back-compat, pre-v1.4.45
    behaviour preserved).
  * Buffer > 0, no breakdown yet → static-only (graceful fallback
    on pre-first-tick / engine-no-quote cycles).
  * Buffer > 0, breakdown.target_half_spread_bps < static_floor →
    floor wins (calm regime).
  * Buffer > 0, breakdown.target_half_spread_bps + buffer >
    static_floor → dynamic wins (vol / toxicity regime).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _build(
    *,
    cancel_on_move_bps: float,
    cancel_on_move_buffer_bps: float,
    breakdown_target_half_spread_bps: float | None,
    binance_mid: float = 2000.0,
    basis_ewma: float = 0.0,
    bid_px: float = 2000.0,
) -> tuple[OrderManager, BotState]:
    """Construct an OrderManager + state primed for one call to
    ``_maybe_cancel_on_binance_move``. The bid is placed near the
    Binance fair value; the actual delta_bps is irrelevant — the
    test asserts on the THRESHOLD value emitted in the cancel-event
    payload, not on whether the cancel fired.
    """
    path = Path(tempfile.gettempdir()) / (
        f"mm_bxc_rel_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "BINANCE_WS_ENABLED": True,
            "BINANCE_CANCEL_ON_MOVE_BPS": cancel_on_move_bps,
            "BINANCE_CANCEL_ON_MOVE_BUFFER_BPS": cancel_on_move_buffer_bps,
            "BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS": 15.0,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    # Stub the enqueue calls so we just observe whether they were
    # invoked (the delta-vs-threshold decision); the actual transport
    # isn't relevant to this test.
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)
    om._enqueue_amend_quote_path = MagicMock(return_value=False)
    # Populate Binance state (recent feed). v1.5.46: bid/ask required
    # by the side-aware cancel gate; synthesise a symmetric narrow
    # spread (0.50 abs, ≈ 2.5 bps at $2000) around mid so the BUY is
    # comfortably outside ref_bid in the _bid_far_from_fair() scenarios.
    state.binance_mid = binance_mid
    state.binance_best_bid = binance_mid - 0.5
    state.binance_best_ask = binance_mid + 0.5
    state.binance_basis_ewma = basis_ewma
    state.binance_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(
        seconds=1.0
    )
    # Populate one working bid that's reasonably close to fair so we
    # can flip the cancel decision by tuning the threshold.
    state.working_bid = WorkingOrder(
        order_id_local=f"b_{uuid.uuid4().hex[:8]}",
        order_id_exchange=100_001,
        client_order_id="cloid_b",
        symbol="ETH",
        side=Side.BUY,
        price=bid_px,
        size=0.05,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    # Populate breakdown if requested. We only need the field the
    # code reads (``target_half_spread_bps``); a MagicMock with that
    # attr works without dragging in the whole breakdown class.
    if breakdown_target_half_spread_bps is None:
        state.last_quote_breakdown = None
    else:
        bd = MagicMock()
        bd.target_half_spread_bps = float(
            breakdown_target_half_spread_bps
        )
        state.last_quote_breakdown = bd
    return om, state


def _captured_threshold_from_event(state: BotState) -> float | None:
    """The cancel/amend event payload carries the threshold_bps the
    code used for the decision. Pull it out of bot_events so we can
    assert on the exact value, independent of whether the cancel
    actually fired."""
    rows = state.storage_provider().recent_bot_events(limit=10) if hasattr(state, "storage_provider") else []  # type: ignore[attr-defined]
    # Fall back to nothing if storage_provider helper isn't wired in
    # this minimal setup — most tests below assert on the cancel-
    # path MagicMock call args instead.
    if not rows:
        return None
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            import json as _json

            p = _json.loads(r["payload_json"])
            return float(p["threshold_bps"])
    return None


def _bid_far_from_fair() -> dict[str, float]:
    """Bid placed 50 bps below Binance fair so any reasonable
    threshold ≤ 50 bps fires a cancel. Lets us observe the
    threshold via the cancel-event payload."""
    return {"binance_mid": 2000.0, "basis_ewma": 0.0, "bid_px": 1990.0}


# ---------------------------------------------------------------------------
# Static-only mode (back-compat — buffer disabled)
# ---------------------------------------------------------------------------


def test_buffer_zero_uses_static_floor_only() -> None:
    """v1.4.45 back-compat: BUFFER=0 → pre-v1.4.45 behaviour.
    Effective threshold = the static floor verbatim, ignoring
    half_spread."""
    om, state = _build(
        cancel_on_move_bps=10.0,
        cancel_on_move_buffer_bps=0.0,
        breakdown_target_half_spread_bps=30.0,  # would be huge dynamic, but disabled
        **_bid_far_from_fair(),
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_called_once()
    # Verify the threshold the code used. Pull the payload from the
    # cancel event in storage (the function writes a bot_event with
    # ``threshold_bps`` in the payload).
    rows = state._storage.recent_bot_events(limit=10) if hasattr(state, "_storage") else []
    # storage is held by the OrderManager, not state
    rows = om._storage.recent_bot_events(limit=10)
    threshold = None
    import json as _json
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            threshold = _json.loads(r["payload_json"])["threshold_bps"]
            break
    assert threshold == 10.0, (
        f"BUFFER=0 must yield threshold == static floor (10), got "
        f"{threshold}"
    )


# ---------------------------------------------------------------------------
# Dynamic-mode floor wins (calm regime)
# ---------------------------------------------------------------------------


def test_dynamic_mode_floor_wins_when_half_spread_small() -> None:
    """v1.4.45: when half_spread + buffer < static_floor, the floor
    wins. Calm-regime quoting still respects the absolute minimum
    distance the operator configured."""
    om, _state = _build(
        cancel_on_move_bps=25.0,        # high floor
        cancel_on_move_buffer_bps=5.0,
        breakdown_target_half_spread_bps=8.0,  # 8 + 5 = 13 < 25
        **_bid_far_from_fair(),
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_called_once()
    import json as _json
    rows = om._storage.recent_bot_events(limit=10)
    threshold = None
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            threshold = _json.loads(r["payload_json"])["threshold_bps"]
            break
    assert threshold == 25.0, (
        f"floor (25) > dynamic (13) → threshold should be 25, got "
        f"{threshold}"
    )


# ---------------------------------------------------------------------------
# Dynamic-mode dynamic wins (vol/toxicity regime — the main v1.4.45 win)
# ---------------------------------------------------------------------------


def test_dynamic_mode_dynamic_wins_when_half_spread_large() -> None:
    """v1.4.45 THE FIX: when half_spread + buffer > static_floor, the
    threshold scales up automatically. This is the case that broke
    v1.4.44 in production: half_spread 13.7 bps, threshold 10 bps,
    every order canceled at placement. Post-fix: same half_spread
    + buffer 5 = 18.7 effective threshold (vs. e.g. floor=10) → bid
    survives."""
    om, _state = _build(
        cancel_on_move_bps=10.0,         # low floor (the v1.4.44 setting)
        cancel_on_move_buffer_bps=5.0,
        breakdown_target_half_spread_bps=13.7,  # the snapshot value
        **_bid_far_from_fair(),
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_called_once()
    import json as _json
    rows = om._storage.recent_bot_events(limit=10)
    threshold = None
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            threshold = _json.loads(r["payload_json"])["threshold_bps"]
            break
    expected = 13.7 + 5.0
    assert threshold is not None
    assert abs(threshold - expected) < 1e-9, (
        f"v1.4.45 THE FIX: dynamic (half_spread {13.7} + buffer 5 = "
        f"{expected}) > floor (10) → threshold should be {expected}, "
        f"got {threshold}. Without this, every order is born past "
        f"the threshold (13.7 > 10) and gets instant-canceled. "
        f"Production wedge regression scenario."
    )


# ---------------------------------------------------------------------------
# Graceful fallbacks
# ---------------------------------------------------------------------------


def test_dynamic_mode_no_breakdown_falls_back_to_floor() -> None:
    """v1.4.45 fallback: when ``last_quote_breakdown`` is None
    (pre-first-tick, or engine returned NO_QUOTE this cycle), the
    code can't compute the dynamic ceiling — fall back to the static
    floor. Otherwise the bot would briefly use a threshold of 0
    during warmup and cancel every order."""
    om, _state = _build(
        cancel_on_move_bps=10.0,
        cancel_on_move_buffer_bps=5.0,
        breakdown_target_half_spread_bps=None,  # no breakdown yet
        **_bid_far_from_fair(),
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_called_once()
    import json as _json
    rows = om._storage.recent_bot_events(limit=10)
    threshold = None
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            threshold = _json.loads(r["payload_json"])["threshold_bps"]
            break
    assert threshold == 10.0, (
        f"warmup fallback: with no breakdown, threshold must be the "
        f"static floor (10), got {threshold}"
    )


def test_dynamic_mode_zero_half_spread_in_breakdown_falls_back() -> None:
    """v1.4.45 defensive: if the breakdown is present but
    ``target_half_spread_bps`` is 0 / None / negative (engine bug or
    transient state), don't use it — fall back to the static floor.
    Otherwise a bad breakdown would let the bot quote/cancel at
    arbitrarily tight thresholds."""
    om, _state = _build(
        cancel_on_move_bps=10.0,
        cancel_on_move_buffer_bps=5.0,
        breakdown_target_half_spread_bps=0.0,  # bad value
        **_bid_far_from_fair(),
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_called_once()
    import json as _json
    rows = om._storage.recent_bot_events(limit=10)
    threshold = None
    for r in rows:
        if r.get("event_type") in (
            "binance_cross_venue_cancel",
            "binance_cross_venue_amend",
        ):
            threshold = _json.loads(r["payload_json"])["threshold_bps"]
            break
    assert threshold == 10.0, (
        f"defensive fallback on half_spread=0: threshold must be "
        f"static floor (10), got {threshold}"
    )
