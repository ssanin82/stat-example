"""1.3.130 multi-rung Phase 2 — integration tests for the
per-rung orchestration path in ``OrderManager.maybe_refresh_quotes``.

These tests verify the dispatch behaviour:

* At ``LADDER_NUM_LEVELS_PER_SIDE=1`` the orchestrator is bit-identical
  to the pre-Phase-2 single-rung path. The ``state.last_ladder_decision``
  field can be absent or present — the dispatch checks the config flag
  AND the field together.
* At N=2 with no working orders, both sides place TWO orders each
  (inside + outer rung).
* At N=2 with the inside rung already ACKED, the outer rung is
  placed independently.
* Cancellation of stale outer rungs: when a previous cycle produced
  a working outer rung at level_idx=1 but the new cycle's
  ``ladder.asks`` has only the inside rung, the outer is cancelled.

Each test follows the same structure as
``tests/test_execution_inventory_bias.py`` —
real ``OrderManager`` against a mock OkxClient, then assert place /
cancel call counts.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.ladder import LadderDecision, LadderRung
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _settings_ladder(num_levels: int) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_ladder_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
            "STALE_DATA_WARN_SECONDS": 30.0,
            "STALE_DATA_KILL_SECONDS": 120.0,
            "REPRICE_THRESHOLD_BPS": 20.0,
            "LADDER_NUM_LEVELS_PER_SIDE": num_levels,
            # 1.4.6: disable batch-place for these orchestration tests
            # so the legacy per-intent path is exercised (the assertions
            # below count individual place_post_only_limit calls on the
            # mock). The batch-place adapter is independently covered
            # by tests/test_okx_responses_place_batch.py.
            "BATCH_PLACES_ENABLED": False,
            "BATCH_PLACES_ALWAYS": False,
        }
    )
    return s, path


def _make_ladder(
    *,
    bid_rungs: list[tuple[int, float, float]],
    ask_rungs: list[tuple[int, float, float]],
    requested_levels: int,
) -> LadderDecision:
    """Build a LadderDecision from explicit (level_idx, px, sz) tuples."""
    return LadderDecision(
        bids=[
            LadderRung(level_idx=i, side=Side.BUY, px=p, sz=z)
            for (i, p, z) in bid_rungs
        ],
        asks=[
            LadderRung(level_idx=i, side=Side.SELL, px=p, sz=z)
            for (i, p, z) in ask_rungs
        ],
        requested_levels=requested_levels,
        effective_levels_buy=len(bid_rungs),
        effective_levels_sell=len(ask_rungs),
        gate_caps={},
    )


def test_n1_path_ignores_ladder_decision_field() -> None:
    """At LADDER_NUM_LEVELS_PER_SIDE=1, even if a multi-rung
    ``last_ladder_decision`` is stashed, the orchestrator falls
    through to the scalar single-rung path. The number of place
    calls is bounded by the scalar QuoteDecision (2 — one BUY, one
    SELL inside) regardless of how many rungs the ladder field has."""
    s, path = _settings_ladder(num_levels=1)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # Stash a phantom 2-rung ladder; the orchestrator should ignore it
    # because the settings knob says N=1.
    state.last_ladder_decision = _make_ladder(
        bid_rungs=[(0, 2999.0, 0.01), (1, 2998.0, 0.007)],
        ask_rungs=[(0, 3001.0, 0.01), (1, 3002.0, 0.007)],
        requested_levels=2,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(oid=1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    # Scalar path: exactly 2 places (one per side).
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_n2_places_two_rungs_per_side_when_empty() -> None:
    """At N=2 with no working orders, both sides place inside +
    outer rung concurrently. Total: 4 places."""
    s, path = _settings_ladder(num_levels=2)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.last_ladder_decision = _make_ladder(
        bid_rungs=[(0, 2999.0, 0.01), (1, 2998.0, 0.007)],
        ask_rungs=[(0, 3001.0, 0.01), (1, 3002.0, 0.007)],
        requested_levels=2,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True

    # Each place returns a unique oid so the mock's side-effect counter
    # produces distinct WO bindings (the bot doesn't care about
    # ordering, but the place_post_only_limit return type contract
    # requires a successful envelope).
    place_oid_counter = {"n": 100}

    def _place_side_effect(*args, **kwargs):
        place_oid_counter["n"] += 1
        return _ok_place(oid=place_oid_counter["n"])

    client.place_post_only_limit.side_effect = _place_side_effect
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    # 2 sides × 2 rungs = 4 places.
    assert client.place_post_only_limit.call_count == 4
    # And the per-rung dict holds 4 WOs.
    bids = state.iter_working_orders(Side.BUY)
    asks = state.iter_working_orders(Side.SELL)
    assert {idx for idx, _ in bids} == {0, 1}
    assert {idx for idx, _ in asks} == {0, 1}
    path.unlink(missing_ok=True)


def test_n2_places_only_outer_when_inside_already_acked() -> None:
    """At N=2 when the inside rung is already ACKED at the right
    price, only the outer rung gets a fresh place. The inside slot
    is left alone (no spurious cancel)."""
    s, path = _settings_ladder(num_levels=2)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # Seed an ACKED inside bid at exactly the ladder's target price.
    state.set_working_order(
        Side.BUY,
        0,
        WorkingOrder(
            order_id_local="L1",
            order_id_exchange=900001,
            client_order_id="cloid-inside",
            symbol=s.symbol,
            side=Side.BUY,
            price=2999.0,
            size=0.01,
            post_only=True,
            status=OrderStatus.ACKED,
            level_idx=0,
        ),
    )
    state.last_ladder_decision = _make_ladder(
        bid_rungs=[(0, 2999.0, 0.01), (1, 2998.0, 0.007)],
        ask_rungs=[(0, 3001.0, 0.01), (1, 3002.0, 0.007)],
        requested_levels=2,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True

    place_oid_counter = {"n": 200}

    def _place_side_effect(*args, **kwargs):
        place_oid_counter["n"] += 1
        return _ok_place(oid=place_oid_counter["n"])

    client.place_post_only_limit.side_effect = _place_side_effect
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    # Expect 3 places: BUY outer + SELL inside + SELL outer.
    # The inside BUY is already ACKED at the target price → no place.
    assert client.place_post_only_limit.call_count == 3
    # And no cancel on the inside BUY (it was at the right price).
    client.cancel_order.assert_not_called()
    path.unlink(missing_ok=True)


def test_n2_normalizes_fractional_rung_sizes_to_lot_grid() -> None:
    """1.3.132 regression: ``LadderRung.sz`` comes from
    ``decision.quoted_bid_sz`` which is PRE-normalization (raw float
    from quoting.py:860). Without re-normalization, fractional rung
    sizes reach the venue and get rejected with "Order quantity must
    be a multiple of the lot size" — exactly the 2036/2039 reject
    pattern from snapshot 260517-094255-colo. The fix re-normalizes
    each rung's (price, size) through ``normalize_order_pair`` before
    staging the place.

    Test: build a ladder with fractional sizes (5.18, 3.626), assert
    the orders that actually reach ``place_post_only_limit`` carry
    integer-rounded sizes (the SUI lot grid has step=1.0)."""
    # Raise the per-order notional cap so fractional sizes at the
    # ETH fixture price ($3000) don't trip the central pre-send cap;
    # the test is about lot-grid normalization, not notional caps.
    s, path = _settings_ladder(num_levels=2)
    s = s.model_copy(update={"max_order_notional_usd": 50_000.0})
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # Fractional rung sizes — what the QuoteEngine would produce after
    # inventory / toxicity multipliers shrink the base size, BEFORE
    # build_quotes normalization. These are the pre-normalized values
    # that flow into the LadderDecision.
    state.last_ladder_decision = _make_ladder(
        bid_rungs=[
            (0, 2999.0, 5.18017),   # inside — fractional
            (1, 2998.0, 3.62612),   # outer — fractional (5.18 × 0.7)
        ],
        ask_rungs=[
            (0, 3001.0, 5.18017),
            (1, 3002.0, 3.62612),
        ],
        requested_levels=2,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    place_oid_counter = {"n": 700}
    captured_sizes: list[float] = []

    def _place_side_effect(*args, **kwargs):
        # args: (symbol, is_buy, sz, limit_px, ...)
        captured_sizes.append(float(args[2]))
        place_oid_counter["n"] += 1
        return _ok_place(oid=place_oid_counter["n"])

    client.place_post_only_limit.side_effect = _place_side_effect
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    # Every size delivered to the venue must be aligned to the
    # symbol's size_step grid. The mock client's
    # FALLBACK_SYMBOL_SPEC.size_step is 0.0001 (4 decimals); for SUI
    # on OKX it would be 1.0 (integer contracts). The fix's
    # normalize_order_pair call clamps to whatever grid the symbol
    # spec declares.
    from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
    step = float(FALLBACK_SYMBOL_SPEC.size_step)
    for sz in captured_sizes:
        # sz must be an integer multiple of step. Use the same
        # tolerance as the venue (which would reject otherwise).
        quotient = sz / step
        assert abs(quotient - round(quotient)) < 1e-6, (
            f"unaligned size {sz} reached the venue — not a multiple "
            f"of step {step}; lot-grid normalization is broken"
        )
    # Sanity: at least one rung per side must have placed.
    assert len(captured_sizes) >= 2
    # Crucial: the captured sizes should NOT be the raw fractional
    # ladder values (5.18017, 3.62612). Each should differ from the
    # raw value by some rounding adjustment (or match by coincidence
    # if the raw was already grid-aligned).
    raw_inputs = {5.18017, 3.62612}
    for sz in captured_sizes:
        # If the size matches a raw input within floating-point noise,
        # normalization didn't happen at all — that's the bug.
        for raw in raw_inputs:
            assert abs(sz - raw) > 1e-9 or (
                abs(sz / step - round(sz / step)) < 1e-6
            ), f"unnormalized raw size {raw} flowed through"
    path.unlink(missing_ok=True)


def test_n2_cancels_stale_outer_rung_when_ladder_shrinks() -> None:
    """At N=2 when the previous cycle's outer rung remains in
    ``working_orders[BUY][1]`` but the new ladder.bids only carries
    the inside rung, the orchestrator cancels the stale outer.
    Models a gate cap shrinking N from 2 → 1 between cycles."""
    s, path = _settings_ladder(num_levels=2)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # Seed both rungs as ACKED at the prior cycle's prices.
    state.set_working_order(
        Side.BUY,
        0,
        WorkingOrder(
            order_id_local="L1",
            order_id_exchange=900001,
            client_order_id="cloid-inside",
            symbol=s.symbol,
            side=Side.BUY,
            price=2999.0,
            size=0.01,
            post_only=True,
            status=OrderStatus.ACKED,
            level_idx=0,
        ),
    )
    state.set_working_order(
        Side.BUY,
        1,
        WorkingOrder(
            order_id_local="L2",
            order_id_exchange=900002,
            client_order_id="cloid-outer",
            symbol=s.symbol,
            side=Side.BUY,
            price=2998.0,
            size=0.007,
            post_only=True,
            status=OrderStatus.ACKED,
            level_idx=1,
        ),
    )
    # New ladder: only inside bid, no outer. Gate cap N=1 scenario.
    state.last_ladder_decision = _make_ladder(
        bid_rungs=[(0, 2999.0, 0.01)],
        ask_rungs=[(0, 3001.0, 0.01), (1, 3002.0, 0.007)],
        requested_levels=2,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True

    place_oid_counter = {"n": 300}

    def _place_side_effect(*args, **kwargs):
        place_oid_counter["n"] += 1
        return _ok_place(oid=place_oid_counter["n"])

    client.place_post_only_limit.side_effect = _place_side_effect
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    # v1.4.33+ routes every cancel through ``cancel_batch_orders``
    # (CANCEL_BATCH pool, 300/2 s). The mock client's default
    # ``cancel_batch_orders.side_effect`` returns a per-ref success
    # echo (see ``tests/exchange_client_mocks.py``), so the stale-
    # outer-rung cancel goes via the batch endpoint with a 1-row
    # body. The executor prefers ``clOrdId`` over ``ordId`` when the
    # WO has a cloid bound (more reliable when ordId might lag); the
    # outer rung's WO has ``client_order_id="cloid-outer"`` so we
    # look for that in the refs.
    cancel_calls = client.cancel_batch_orders.call_args_list
    cancelled_keys: list[str] = []
    for call in cancel_calls:
        # cancel_batch_orders(symbol, refs)
        refs = call.args[1] if len(call.args) > 1 else call.kwargs.get("refs", [])
        for r in refs:
            cancelled_keys.append(
                str(r.get("clOrdId") or r.get("ordId") or "")
            )
    assert "cloid-outer" in cancelled_keys, (
        f"expected outer BUY (cloid 'cloid-outer' / oid 900002) "
        f"cancelled, got {cancelled_keys}"
    )
    path.unlink(missing_ok=True)
