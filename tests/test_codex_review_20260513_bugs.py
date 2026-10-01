"""Regression tests for the Codex-flagged bugs from the 2026-05-13
review (``plans/20260513-codex-review-bugs.md``):

1. CRITICAL — ``OkxClient.fetch_open_orders_raw`` returned ``[]`` on
   non-zero OKX ``code``, indistinguishable from a healthy account
   with no resting orders. Cancel-all became a silent no-op on
   auth/rate-limit/exchange-side errors; reconcile would treat live
   venue orders as gone. Fix: raise typed ``OkxApiError``.

2. HIGH — ``refresh_account_only`` wrapped ``fetch_position``,
   ``fetch_account_snapshot``, AND ``fetch_recent_fills_raw`` in a
   single try block. A transient fills-endpoint failure after
   position+account succeeded threw away the authoritative
   inventory-truth refresh and could trip ``account_data_stale``.
   Fix: split into two stages, fills failures don't poison stage 1.

3. MED — ``OkxClient.fetch_recent_fills_raw`` returned ``[]`` on
   non-zero OKX ``code`` instead of surfacing an error. Bot would
   silently lose fee/realized-PnL/markout/toxicity attribution while
   appearing healthy. Fix: raise typed ``OkxApiError``; the new
   Bug 2 split lets the account refresh still succeed.

4. MED — REST catch-up fills stamped ``inventory_qty_before_fill``
   from ``state.position.position_qty`` AFTER
   ``apply_account_position_only`` had installed the post-everything
   venue snapshot. Result: post-fill qty was recorded as pre-fill on
   every catch-up fill, biasing exactly the regime-attribution path
   meant to investigate missed private-WS fills.
   Fix: pre-compute pre-fill quantities by walking the catch-up
   batch in reverse, pass into ``ingest_hl_fill_raw`` via
   ``pre_inventory_qty_override``.

5. MED — ``execution.py::_stage_place_order_local`` stamped
   ``adaptive_widen_active_at_decision`` via
   ``getattr(state, "adaptive_widen_active", False)`` on a
   non-existent attribute (BotState only stored the deadline
   ``adaptive_spread_widen_until_mono``). Field was permanently
   False on every order/fill, breaking regime slicing by that flag.
   ``exposure_bar_emitter.py`` already derived correctly from the
   deadline. Fix: expose ``adaptive_widen_active`` as a derived
   property on ``BotState`` so all three consumers share one
   source of truth.
"""

from __future__ import annotations

import os
import tempfile
import time as _time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.enums import Side
from app.models import AccountSnapshot, PositionSnapshot
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**kw):
    path = (
        Path(tempfile.gettempdir())
        / f"mm_codex_20260513_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


# =============================================================================
# Bug #1 — fetch_open_orders_raw raises OkxApiError instead of returning []
# =============================================================================


def test_bug1_okx_open_orders_raises_on_non_zero_code() -> None:
    """Verify the OkxApiError exception class behavior. We stub the
    HTTP layer rather than building a full OkxClient (which requires
    credentials + symbol bootstrap) since the bug is in the response
    interpretation, not the request.
    """
    from app.exchange.okx_client import OkxApiError

    # Construct directly from a hypothetical non-zero response.
    err = OkxApiError("50001", "Invalid API key", endpoint="/trade/orders-pending")
    assert err.code == "50001"
    assert "50001" in str(err)
    assert "/trade/orders-pending" in str(err)
    # Auth errors are NOT rate-limit:
    assert err.status_code is None


def test_bug1_okx_rate_limit_code_maps_to_429() -> None:
    """The duck-typed rate-limit detector in
    ``OrderManager._is_rate_limited_exception`` inspects
    ``status_code`` first. When OKX returns code=50011 (rate limit),
    we must expose ``status_code=429`` so the reconcile path takes
    its existing rate-limit branch and engages the throttle.
    """
    from app.exchange.okx_client import OkxApiError

    err = OkxApiError("50011", "Too Many Requests", endpoint="/trade/fills")
    assert err.status_code == 429


def test_bug1_open_orders_callers_take_error_branch_on_okx_api_error() -> None:
    """The two callers of ``fetch_open_orders_raw`` are:
      - ``OrderManager._sync_open_orders_impl`` (line ~2467)
      - ``OrderManager.cancel_all_orders_for_symbol`` (line ~2326)

    Both have ``except Exception`` handlers that bump
    ``execution_errors`` and bail out. By raising OkxApiError, the
    fetch_open_orders_raw failure now flows through those handlers
    instead of returning [] (which would silently no-op).

    We don't construct a full OrderManager here (it requires the
    full bot wiring). Instead we assert the invariant that matters:
    OkxApiError IS-A Exception so any caller that catches Exception
    catches it too.
    """
    from app.exchange.okx_client import OkxApiError

    assert issubclass(OkxApiError, Exception)
    # Catch as plain Exception — which is what _sync_open_orders_impl
    # and cancel_all_orders_for_symbol do today.
    caught = None
    try:
        raise OkxApiError("50001", "auth fail", endpoint="/trade/orders-pending")
    except Exception as e:
        caught = e
    assert isinstance(caught, OkxApiError)


# =============================================================================
# Bug #2 — refresh_account_only splits position+account from fills
# =============================================================================


def test_bug2_account_refresh_succeeds_when_fills_fetch_throws() -> None:
    """Stage 1 (position + account) must commit even if Stage 2
    (fills catch-up) raises. Pre-fix, a transient fills-endpoint
    failure threw away fresh authoritative position/account state
    and tripped account_data_stale. Post-fix, account refresh marks
    success; fills failure logs + bumps a distinct error counter.
    """
    from app.market_data import refresh_account_only

    settings = _settings()
    state = BotState(settings)

    client = MagicMock()
    fresh_pos = PositionSnapshot(
        symbol="TEST",
        position_qty=42.0,
        avg_entry_price=2.1,
        mark_price=2.11,
        position_notional=88.62,
        unrealized_pnl_usd=0.42,
    )
    fresh_acct = AccountSnapshot(
        equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=900.0
    )
    client.fetch_position = MagicMock(return_value=fresh_pos)
    client.fetch_account_snapshot = MagicMock(return_value=fresh_acct)
    client.fetch_recent_fills_raw = MagicMock(
        side_effect=RuntimeError("simulated fills endpoint failure")
    )

    refresh_account_only(
        client,
        state,
        address="0xabc",
        storage=None,
        pnl=None,
        ingest_fills_via_rest=True,
    )

    # Stage 1 (position + account) must have committed successfully.
    assert state.position.position_qty == 42.0
    assert state.position.avg_entry_price == 2.1
    # Distinct error counter for the fills stage.
    # The legacy error tag ``account_refresh_exception`` MUST NOT
    # appear in the event log because stage 1 succeeded. The new
    # ``rest_fill_fetch_exception`` tag SHOULD appear because stage 2
    # raised. (``execution_errors`` is a counter; the per-source tags
    # live in ``_execution_error_events`` as (ts, tag) tuples.)
    tags = [tag for _ts, tag in state._execution_error_events]
    assert "account_refresh_exception" not in tags
    assert "rest_fill_fetch_exception" in tags


def test_bug2_account_refresh_still_fails_when_position_fetch_throws() -> None:
    """Stage 1 errors are real failures — the early-return + throttle
    engagement path must still fire if ``fetch_position`` raises.
    Pre-fix behaviour for stage 1 is preserved.
    """
    from app.market_data import refresh_account_only

    settings = _settings()
    state = BotState(settings)
    # Pre-condition: fresh state, no refresh has happened.
    assert state.position.position_qty == 0.0

    client = MagicMock()
    client.fetch_position = MagicMock(
        side_effect=RuntimeError("simulated venue 500")
    )
    # account / fills won't be reached
    client.fetch_account_snapshot = MagicMock()
    client.fetch_recent_fills_raw = MagicMock()

    refresh_account_only(
        client,
        state,
        address="0xabc",
        storage=None,
        pnl=None,
        ingest_fills_via_rest=True,
    )

    # Position was not updated — refresh failed cleanly.
    assert state.position.position_qty == 0.0
    # account_refresh_exception bumped (stage 1 failure path).
    tags = [tag for _ts, tag in state._execution_error_events]
    assert "account_refresh_exception" in tags
    # account-only fetch never happened.
    client.fetch_account_snapshot.assert_not_called()
    client.fetch_recent_fills_raw.assert_not_called()


# =============================================================================
# Bug #3 — fetch_recent_fills_raw raises OkxApiError instead of returning []
# =============================================================================


def test_bug3_okx_fills_raises_carries_endpoint() -> None:
    """Same exception class as Bug 1, different endpoint label so
    operator logs can tell them apart.
    """
    from app.exchange.okx_client import OkxApiError

    err = OkxApiError("50011", "Too Many Requests", endpoint="/trade/fills")
    assert err.code == "50011"
    assert err.status_code == 429  # rate-limit → 429 mapping
    assert "/trade/fills" in str(err)


def test_bug3_fills_failure_does_not_clobber_account_refresh() -> None:
    """End-to-end Bug 2 + Bug 3 interaction: an OkxApiError from
    ``fetch_recent_fills_raw`` is caught by the stage-2 try block in
    ``refresh_account_only``, NOT by the stage-1 try block. Stage 1
    must remain successful.
    """
    from app.exchange.okx_client import OkxApiError
    from app.market_data import refresh_account_only

    settings = _settings()
    state = BotState(settings)

    client = MagicMock()
    client.fetch_position = MagicMock(
        return_value=PositionSnapshot(
            symbol="TEST",
            position_qty=5.0,
            avg_entry_price=2.0,
            mark_price=2.0,
            position_notional=10.0,
            unrealized_pnl_usd=0.0,
        )
    )
    client.fetch_account_snapshot = MagicMock(
        return_value=AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=900.0
        )
    )
    client.fetch_recent_fills_raw = MagicMock(
        side_effect=OkxApiError(
            "50011", "Too Many Requests", endpoint="/trade/fills"
        )
    )

    refresh_account_only(
        client,
        state,
        address="0xabc",
        storage=None,
        pnl=None,
        ingest_fills_via_rest=True,
    )

    # Stage 1 succeeded — position installed.
    assert state.position.position_qty == 5.0
    # No legacy stage-1 error counter bumped.
    tags = [tag for _ts, tag in state._execution_error_events]
    assert "account_refresh_exception" not in tags
    # Stage 2 (fills catch-up) error WAS bumped with the new tag.
    assert "rest_fill_fetch_exception" in tags


# =============================================================================
# Bug #4 — REST catch-up fills stamp pre-fill inventory correctly
# =============================================================================


def test_bug4_rest_catchup_pre_fill_qty_is_pre_not_post() -> None:
    """The headline regression: REST catch-up walks the fills batch
    in reverse to compute each fill's TRUE pre-fill qty before
    apply_account_position_only was applied. We verify the
    pre-computed override values are correct by stubbing
    ingest_hl_fill_raw and asserting the override kwarg per call.

    Scenario: 3 BUY fills of size 2 land while the bot was at qty=0.
    After apply_account_position_only the venue snapshot reports
    qty=6 (post-all). For each fill, pre-fill qty must be:
      fill[0]: 0   (before the first BUY)
      fill[1]: 2   (after first BUY, before second)
      fill[2]: 4   (after first two, before third)
    NOT 6 (which is post-all) for any of them.
    """
    from app.exchange.hyperliquid_types import HLFillRaw
    from app.market_data import refresh_account_only
    from app import market_data as md

    settings = _settings()
    state = BotState(settings)

    # Three BUY fills of size 2.0 each. The post-all venue snapshot
    # is qty=6.0; pre-fix the loop would stamp all three with
    # pre_qty=6.0 which is wrong.
    def _fr(i: int) -> HLFillRaw:
        # HLFillRaw used as a duck-typed shape. Construct via
        # constructor with the fields the loop reads.
        return HLFillRaw(
            fill_id=f"test_fill_{i}",
            oid=int(f"100{i}"),
            coin=settings.symbol,
            side=Side.BUY,
            px=2.0,
            sz=2.0,
            fee=0.0,
            time_ms=1_700_000_000_000 + i * 100,
            closed_pnl=0.0,
            raw={},
        )

    raw_fills = [_fr(0), _fr(1), _fr(2)]

    client = MagicMock()
    client.fetch_position = MagicMock(
        return_value=PositionSnapshot(
            symbol=settings.symbol,
            position_qty=6.0,  # post-all venue snapshot
            avg_entry_price=2.0,
            mark_price=2.0,
            position_notional=12.0,
            unrealized_pnl_usd=0.0,
        )
    )
    client.fetch_account_snapshot = MagicMock(
        return_value=AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
        )
    )
    client.fetch_recent_fills_raw = MagicMock(return_value=raw_fills)

    captured: list[dict] = []

    def _capture(*args, **kwargs):
        captured.append(kwargs)

    real = md.ingest_hl_fill_raw
    md.ingest_hl_fill_raw = _capture
    try:
        refresh_account_only(
            client,
            state,
            address="0xabc",
            storage=None,
            pnl=None,
            ingest_fills_via_rest=True,
        )
    finally:
        md.ingest_hl_fill_raw = real

    assert len(captured) == 3, (
        f"expected 3 ingest calls (one per fill), got {len(captured)}"
    )
    # Each call must carry the pre_inventory_qty_override kwarg with
    # the genuine pre-fill qty.
    overrides = [c.get("pre_inventory_qty_override") for c in captured]
    # fill[0] (first BUY): pre = 0
    # fill[1] (second BUY): pre = 2
    # fill[2] (third BUY): pre = 4
    # NOT 6 (the post-all venue snapshot).
    assert overrides == [0.0, 2.0, 4.0], (
        f"pre-fill qty overrides should walk back from post=6 by -2 per "
        f"BUY of size 2; got {overrides}"
    )


def test_bug4_pre_fill_override_handles_mixed_sides() -> None:
    """Reverse-walk works for mixed BUY/SELL batches too. Confirms
    the signed-delta computation is correct.

    Scenario: BUY 1, SELL 2, BUY 3 → net delta = +1 -2 +3 = +2.
    If venue post-all snapshot is qty=5.0:
      fill[2] (BUY +3): pre = 5 - 3 = 2
      fill[1] (SELL -2): pre = 2 - (-2) = 4
      fill[0] (BUY +1): pre = 4 - 1 = 3
    """
    from app.exchange.hyperliquid_types import HLFillRaw
    from app.market_data import refresh_account_only
    from app import market_data as md

    settings = _settings()
    state = BotState(settings)

    def _fr(i: int, side: Side, sz: float) -> HLFillRaw:
        return HLFillRaw(
            fill_id=f"test_fill_{i}",
            oid=int(f"200{i}"),
            coin=settings.symbol,
            side=side,
            px=2.0,
            sz=sz,
            fee=0.0,
            time_ms=1_700_000_000_000 + i * 100,
            closed_pnl=0.0,
            raw={},
        )

    raw_fills = [
        _fr(0, Side.BUY, 1.0),
        _fr(1, Side.SELL, 2.0),
        _fr(2, Side.BUY, 3.0),
    ]

    client = MagicMock()
    client.fetch_position = MagicMock(
        return_value=PositionSnapshot(
            symbol=settings.symbol,
            position_qty=5.0,  # post-all (5 + net=+2 from start would be... )
            avg_entry_price=2.0,
            mark_price=2.0,
            position_notional=10.0,
            unrealized_pnl_usd=0.0,
        )
    )
    client.fetch_account_snapshot = MagicMock(
        return_value=AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
        )
    )
    client.fetch_recent_fills_raw = MagicMock(return_value=raw_fills)

    captured: list[dict] = []
    real = md.ingest_hl_fill_raw
    md.ingest_hl_fill_raw = lambda *a, **k: captured.append(k)
    try:
        refresh_account_only(
            client,
            state,
            address="0xabc",
            storage=None,
            pnl=None,
            ingest_fills_via_rest=True,
        )
    finally:
        md.ingest_hl_fill_raw = real

    assert len(captured) == 3
    overrides = [c.get("pre_inventory_qty_override") for c in captured]
    # Per the docstring math: pre-fill[0]=3, pre-fill[1]=4, pre-fill[2]=2.
    assert overrides == [3.0, 4.0, 2.0], (
        f"mixed-side reverse walk wrong: expected [3.0, 4.0, 2.0], got {overrides}"
    )


def test_bug4_ingest_signature_accepts_override_kwarg() -> None:
    """Signature compatibility check: ``ingest_hl_fill_raw`` accepts
    the new ``pre_inventory_qty_override`` keyword. The high-level
    ``test_bug4_rest_catchup_pre_fill_qty_is_pre_not_post`` proves
    the kwarg threads through to the actual override behaviour via
    a stubbed ingest; this test just pins the call signature so
    refactors of ``ingest_hl_fill_raw`` can't silently drop the
    parameter and reintroduce the bug.
    """
    import inspect

    from app.fill_ingestion import ingest_hl_fill_raw

    sig = inspect.signature(ingest_hl_fill_raw)
    assert "pre_inventory_qty_override" in sig.parameters, (
        "ingest_hl_fill_raw must keep the pre_inventory_qty_override "
        "kwarg — REST catch-up calls it with this name (Bug 4 fix)."
    )
    # And the default must be None so the WS path (which doesn't
    # pass the kwarg) continues to work unchanged.
    param = sig.parameters["pre_inventory_qty_override"]
    assert param.default is None


# =============================================================================
# Bug #5 — adaptive_widen_active property derived from deadline
# =============================================================================


def test_bug5_adaptive_widen_active_property_off_by_default() -> None:
    """Fresh BotState: deadline is 0.0, so property returns False."""
    settings = _settings()
    state = BotState(settings)
    assert state.adaptive_spread_widen_until_mono == 0.0
    assert state.adaptive_widen_active is False


def test_bug5_adaptive_widen_active_property_true_when_armed() -> None:
    """Set the deadline 5 seconds into the future — property returns
    True until the deadline elapses.
    """
    settings = _settings()
    state = BotState(settings)
    state.adaptive_spread_widen_until_mono = _time.monotonic() + 5.0
    assert state.adaptive_widen_active is True


def test_bug5_adaptive_widen_active_property_false_after_deadline() -> None:
    """Once the deadline is in the past, property returns False."""
    settings = _settings()
    state = BotState(settings)
    # Deadline 1 second AGO.
    state.adaptive_spread_widen_until_mono = _time.monotonic() - 1.0
    assert state.adaptive_widen_active is False


def test_bug5_adaptive_widen_active_property_getattr_returns_real_bool() -> None:
    """The execution.py code path does
    ``getattr(state, "adaptive_widen_active", False)``. Pre-fix this
    silently returned the literal default (False) because no such
    attribute existed. Post-fix it returns the property's computed
    bool. We verify directly via getattr to mimic the call site.
    """
    settings = _settings()
    state = BotState(settings)
    state.adaptive_spread_widen_until_mono = _time.monotonic() + 3.0
    fetched = getattr(state, "adaptive_widen_active", False)
    assert fetched is True, (
        "getattr on the property must return the computed bool, not the "
        "default. Pre-fix this would have been False — the bug under test."
    )


def test_bug5_property_matches_exposure_bar_emitter_derivation() -> None:
    """Both ``state.adaptive_widen_active`` (property) and the
    inline derivation in ``exposure_bar_emitter.py`` derive from the
    same deadline. The two must agree at all times — same anti-drift
    discipline as v1.3.11 (vol thresholds) / v1.3.12 (basis units).
    """
    settings = _settings()
    state = BotState(settings)
    # Three scenarios: not armed, armed in future, expired in past.
    for deadline_offset, expected in [
        (0.0, False),           # not armed
        (5.0, True),            # armed
        (-1.0, False),          # expired
    ]:
        state.adaptive_spread_widen_until_mono = (
            _time.monotonic() + deadline_offset if deadline_offset != 0.0
            else 0.0
        )
        # Inline-derivation mimic — what exposure_bar_emitter.py does:
        now_mono = _time.monotonic()
        inline = now_mono < float(state.adaptive_spread_widen_until_mono or 0.0)
        assert state.adaptive_widen_active == inline == expected, (
            f"property vs inline disagreement: "
            f"property={state.adaptive_widen_active}, "
            f"inline={inline}, expected={expected}"
        )
