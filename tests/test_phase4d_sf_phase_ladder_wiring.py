"""Phase 4D wiring (v1.4.172) — ``Bot._run_sf_phase_ladder_dispatch``.

Pure-helper logic is covered in ``test_phase4d_sf_phase_ladder.py``.
This file pins the WIRING that consumes the helper's
``PhaseLadderDecision`` and dispatches to the matching transport:

* ``post_only`` → ``OrderManager.place_passive_order_manual_only``
* ``ioc``       → ``client.place_ioc_reduce_only``  (NEW transport)
* ``market``    → ``client.market_close`` + ``_exit_soft_flatten``

Approach: drive the dispatcher as an unbound method against a stub
``self`` with the SF-related state + collaborators mocked. Same
pattern the existing ``test_soft_flatten_worker.py`` uses for
``_maybe_handle_cancel_pending_timeout``-style tests.

Targets the v1.4.157 failure: with the legacy 2-phase post-only the
bot timed out 14 s before crossing; the ladder should escalate to
IOC inside the first tick once the touch has drifted ≥ 3 ticks past
the SF entry mid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

from app.bot import Bot
from app.enums import OrderStatus, Side
from app.models import BestBidAsk
from app.soft_flatten import (
    PHASE_0_POST_ONLY_NEAR,
    PHASE_1_POST_ONLY_FAR_PLUS_TICK,
    PHASE_2_IOC_CROSS_1,
    PHASE_3_IOC_CROSS_2,
    PHASE_4_MARKET,
)


# ---------------------------------------------------------------------------
# Stub harness
# ---------------------------------------------------------------------------


@dataclass
class _ExecRec:
    """Records every call to the OrderManager surface."""

    placed: list[tuple[Side, float, float, str, bool]]
    cancel_all_count: int


@dataclass
class _ClientRec:
    """Records every call to the venue adapter surface."""

    ioc_calls: list[tuple[str, bool, float, float]]
    market_close_calls: list[str]


def _make_settings(**kw: Any) -> Any:
    """Test settings. Phase ladder ENABLED by default for this file."""
    s = MagicMock()
    s.symbol = "TEST-USDT-SWAP"
    s.sf_phase_ladder_enabled = kw.get("sf_phase_ladder_enabled", True)
    s.sf_phase_0_duration_seconds = kw.get(
        "sf_phase_0_duration_seconds", 3.0
    )
    s.sf_phase_1_duration_seconds = kw.get(
        "sf_phase_1_duration_seconds", 4.0
    )
    s.sf_phase_2_duration_seconds = kw.get(
        "sf_phase_2_duration_seconds", 4.0
    )
    s.sf_phase_3_duration_seconds = kw.get(
        "sf_phase_3_duration_seconds", 2.0
    )
    s.sf_fast_escalate_ticks = kw.get("sf_fast_escalate_ticks", 3.0)
    s.sf_consecutive_rejects_to_escalate = kw.get(
        "sf_consecutive_rejects_to_escalate", 10
    )
    # v1.5.198 — episode hard-timeout + re-entry cooldown. Set
    # to disabled defaults so existing phase-ladder tests aren't
    # affected by the new rules. Tests that exercise these
    # mechanisms set them explicitly via ``kw``.
    s.soft_flatten_episode_max_duration_seconds = kw.get(
        "soft_flatten_episode_max_duration_seconds", 0.0
    )
    s.soft_flatten_reentry_cooldown_seconds = kw.get(
        "soft_flatten_reentry_cooldown_seconds", 0.0
    )
    s.max_order_notional_usd = kw.get("max_order_notional_usd", 1000.0)
    s.max_abs_position = kw.get("max_abs_position", 1000.0)
    s.max_position_notional_usd = kw.get(
        "max_position_notional_usd", 1000.0
    )
    s.soft_flatten_reprice_ticks = kw.get(
        "soft_flatten_reprice_ticks", 1
    )
    return s


def _make_state(
    *,
    current_phase: int = PHASE_0_POST_ONLY_NEAR,
    phase_started_mono: Optional[float] = None,
    entry_mid_for_phase_ladder: Optional[float] = 2.040,
    consecutive_rejects_in_phase: int = 0,
) -> Any:
    if phase_started_mono is None:
        # Default: "started just now" → no time-based escalation.
        phase_started_mono = time.monotonic()
    state = MagicMock()
    state._lock = MagicMock()
    state._lock.__enter__ = MagicMock(return_value=None)
    state._lock.__exit__ = MagicMock(return_value=None)
    state.sf_phase_ladder_phase = current_phase
    state.sf_phase_started_mono = phase_started_mono
    state.sf_consecutive_rejects_in_phase = consecutive_rejects_in_phase
    state.sf_entry_mid_for_phase_ladder = entry_mid_for_phase_ladder
    state.soft_flatten_entry_mid = entry_mid_for_phase_ladder
    state.soft_flatten_active = True
    # v1.4.191 Phase 4D.5 throttle state. Default None so the wiring
    # tests don't accidentally hit the throttle (these tests pin
    # dispatch routing, not rate-limiting — Phase 4D.5 is exercised
    # separately in test_phase4d5_sf_action_throttle.py).
    state.sf_last_action_mono = None
    state.sf_throttle_suppressed_total = 0
    state.sf_throttle_first_arm_logged = False
    return state


def _make_bot_stub(
    *,
    settings: Any,
    state: Any,
    exec_rec: _ExecRec,
    client_rec: _ClientRec,
) -> Any:
    """Build a minimal stub that satisfies the dispatcher's reads.

    Plain object (not MagicMock) so attribute typos surface as
    AttributeError instead of being auto-vivified into stale mocks.
    """

    class _Stub:
        pass

    stub = _Stub()
    stub._settings = settings
    stub._state = state
    # Phase 1c (v1.4.231) — the SF dispatch path now reads
    # ``self._clock.monotonic()`` for phase-elapsed timing checks. Give
    # the stub a SystemClock so existing tests don't need to mock it.
    from app.clock import SystemClock
    stub._clock = SystemClock()

    # Exec collaborator — record placements + cancels.
    class _Exec:
        def __init__(self, rec: _ExecRec) -> None:
            self._rec = rec

        def place_passive_order_manual_only(
            self,
            side: Side,
            price: float,
            size: float,
            cycle: str,
            *,
            reduce_only: bool = False,
        ) -> None:
            self._rec.placed.append(
                (side, price, size, cycle, reduce_only)
            )

        def cancel_all_orders_for_symbol(self) -> None:
            self._rec.cancel_all_count += 1

    stub._exec = _Exec(exec_rec)

    # Client (venue adapter) — record IOC + market_close calls.
    class _Client:
        def __init__(self, rec: _ClientRec) -> None:
            self._rec = rec

        def place_ioc_reduce_only(
            self,
            symbol: str,
            is_buy: bool,
            sz: float,
            limit_px: float,
        ) -> dict[str, Any]:
            self._rec.ioc_calls.append((symbol, is_buy, sz, limit_px))
            return {"code": "0", "data": []}

        def market_close(
            self, symbol: str, sz: Optional[float] = None
        ) -> dict[str, Any]:
            self._rec.market_close_calls.append(symbol)
            return {"code": "0", "data": []}

    stub._client = _Client(client_rec)

    # ``_exit_soft_flatten`` is a method on the real Bot; replace with
    # a recorder so phase-4 dispatch can be observed without spinning
    # up a full Bot.
    stub._exit_soft_flatten = MagicMock()
    # v1.4.173: the dispatcher writes a synthetic ORDERS row before
    # firing the IOC / market_close. Stub this out — the row-write
    # behaviour is exercised by `test_phase4d4_sf_phase_observability`;
    # here we only want to verify the placement dispatch path.
    stub._persist_synthetic_sf_order_row = MagicMock()
    # v1.4.191 Phase 4D.5 — the dispatcher consults the action-rate
    # throttle before non-terminal placements. Bind the real methods
    # so the dispatcher's call site works; with
    # ``state.sf_last_action_mono = None`` the throttle is dormant
    # and never suppresses (verified in
    # test_phase4d5_sf_action_throttle.py).
    stub._sf_action_throttled = Bot._sf_action_throttled.__get__(
        stub, _Stub
    )
    stub._note_sf_throttle_suppression = (
        Bot._note_sf_throttle_suppression.__get__(stub, _Stub)
    )
    return stub


def _make_market(best_bid: float, best_ask: float) -> BestBidAsk:
    return BestBidAsk(
        symbol="TEST-USDT-SWAP",
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=(best_bid + best_ask) / 2.0,
        spread_bps=(best_ask - best_bid) / best_bid * 10_000.0,
        bid_size=100.0,
        ask_size=100.0,
    )


# ---------------------------------------------------------------------------
# Phase 0: post-only at near touch → places via OrderManager
# ---------------------------------------------------------------------------


def test_phase_0_dispatches_post_only_via_exec() -> None:
    """Phase 0 → ``place_passive_order_manual_only`` with the close
    side, the phase-0 target price, and ``reduce_only=True``.

    Bot is short -9; close side = BUY; phase 0 target = best_bid.
    """
    settings = _make_settings()
    state = _make_state(current_phase=PHASE_0_POST_ONLY_NEAR)
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )

    market = _make_market(best_bid=2.040, best_ask=2.041)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert len(exec_rec.placed) == 1
    side, price, size, _cycle, reduce_only = exec_rec.placed[0]
    assert side == Side.BUY  # close short → BUY
    assert price == 2.040  # best_bid
    assert reduce_only is True
    assert size > 0
    assert client_rec.ioc_calls == []
    assert client_rec.market_close_calls == []
    assert stub._exit_soft_flatten.call_count == 0


# ---------------------------------------------------------------------------
# Phase 2 IOC: cancel resting + fire IOC reduce-only
# ---------------------------------------------------------------------------


def test_phase_2_ioc_cancels_resting_then_fires_ioc_next_tick() -> None:
    """Tick 1 (phase transition to 2 with resting post-only):
    cancel-all; no IOC placed. Tick 2 (no resting): IOC placed."""
    settings = _make_settings()
    state = _make_state(
        current_phase=PHASE_2_IOC_CROSS_1,
        # Just-started phase 2: no time-based escalation within the
        # 4 s phase-2 budget.
        phase_started_mono=time.monotonic(),
        entry_mid_for_phase_ladder=2.040,
    )
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )

    # Tick 1: still a resting post-only on the close side (BUY).
    resting_bid = MagicMock()
    resting_bid.status = OrderStatus.ACKED
    resting_bid.price = 2.040
    resting_bid.size = 9.0

    market = _make_market(best_bid=2.058, best_ask=2.059)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=resting_bid,
        wo_ask=None,
        tick_size=0.001,
    )

    # Cancel issued; no IOC yet (waiting for cancel to clear).
    assert exec_rec.cancel_all_count == 1
    assert client_rec.ioc_calls == []

    # Tick 2: cancel propagated; no resting orders.
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert len(client_rec.ioc_calls) == 1
    symbol, is_buy, sz, limit_px = client_rec.ioc_calls[0]
    assert symbol == "TEST-USDT-SWAP"
    assert is_buy is True  # close short → BUY
    assert sz > 0
    assert limit_px == 2.059  # phase 2 BUY = at the ask (crosses)
    # No post-only placement on the IOC path.
    assert exec_rec.placed == []


def test_phase_2_ioc_falls_back_to_market_when_adapter_missing() -> None:
    """When the venue adapter has no ``place_ioc_reduce_only``, the
    dispatcher logs + falls back to ``market_close`` + exits SF (safer
    than stalling in phase 2 forever)."""
    settings = _make_settings()
    state = _make_state(
        current_phase=PHASE_2_IOC_CROSS_1,
        phase_started_mono=time.monotonic(),
        entry_mid_for_phase_ladder=2.040,
    )
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )
    # Shadow the IOC method with ``None`` so the dispatcher's
    # ``getattr(...) → callable(...)`` check fails and the fallback
    # path engages. (Deleting the class method off the instance is
    # an AttributeError; instance-level None overrides it.)
    stub._client.place_ioc_reduce_only = None

    market = _make_market(best_bid=2.058, best_ask=2.059)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert client_rec.market_close_calls == ["TEST-USDT-SWAP"]
    assert stub._exit_soft_flatten.call_count == 1


# ---------------------------------------------------------------------------
# Phase 4 market: cancel + market_close + exit
# ---------------------------------------------------------------------------


def test_phase_4_dispatches_market_close_and_exits_sf() -> None:
    """Terminal phase 4: cancel resting + market_close + exit SF."""
    settings = _make_settings()
    # Force phase 4 by starting at phase 3 with a long-elapsed dwell
    # (well past the 2 s phase-3 budget).
    state = _make_state(
        current_phase=PHASE_3_IOC_CROSS_2,
        phase_started_mono=time.monotonic() - 10.0,
        entry_mid_for_phase_ladder=2.040,
    )
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )

    market = _make_market(best_bid=2.058, best_ask=2.059)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert client_rec.market_close_calls == ["TEST-USDT-SWAP"]
    assert exec_rec.cancel_all_count >= 1
    assert stub._exit_soft_flatten.call_count == 1


# ---------------------------------------------------------------------------
# v1.4.157 replay — drift escalation triggers IOC inside first tick
# ---------------------------------------------------------------------------


def test_v1_4_157_replay_escalates_to_ioc_after_cancel() -> None:
    """Tick at t=0.5s with ask = $2.052 (12 ticks above entry mid
    $2.040) → drift rule fires → escalate to phase 2 IOC. Cancel-all
    issued first to clear any resting post-only; the IOC fires on the
    NEXT tick.

    With the legacy 2-phase post-only the bot would have stayed in
    phase 0/1 for 14 s and then market_closed at $2.059. The ladder
    closes at $2.052 instead — 7 ticks better.
    """
    settings = _make_settings()
    state = _make_state(
        current_phase=PHASE_0_POST_ONLY_NEAR,
        phase_started_mono=time.monotonic() - 0.5,  # 0.5 s into phase 0
        entry_mid_for_phase_ladder=2.040,
    )
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )

    # First tick: still has a resting post-only at the touch from the
    # SF entry.
    resting_bid = MagicMock()
    resting_bid.status = OrderStatus.ACKED
    resting_bid.price = 2.040
    resting_bid.size = 9.0

    market = _make_market(best_bid=2.051, best_ask=2.052)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=resting_bid,
        wo_ask=None,
        tick_size=0.001,
    )

    # Phase advanced to 2; cancel issued for the resting post-only.
    assert state.sf_phase_ladder_phase == PHASE_2_IOC_CROSS_1
    assert exec_rec.cancel_all_count == 1
    assert client_rec.ioc_calls == []

    # Next tick: cancel propagated, no resting → IOC at the ask.
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert len(client_rec.ioc_calls) == 1
    _symbol, _is_buy, _sz, limit_px = client_rec.ioc_calls[0]
    # IOC limit price = current best_ask = $2.052 (7 ticks below the
    # eventual $2.059 the legacy worker market_closed at).
    assert limit_px == 2.052


# ---------------------------------------------------------------------------
# State write-back on transition
# ---------------------------------------------------------------------------


def test_phase_transition_writes_back_new_phase_and_resets_rejects() -> None:
    """Transition phase 0 → 1: state.sf_phase_ladder_phase updated;
    rejects counter reset to 0."""
    settings = _make_settings()
    state = _make_state(
        current_phase=PHASE_0_POST_ONLY_NEAR,
        # 5 s into phase 0 (past the 3 s budget) → time-based advance.
        phase_started_mono=time.monotonic() - 5.0,
        entry_mid_for_phase_ladder=2.041,  # no drift escalation
        consecutive_rejects_in_phase=5,
    )
    exec_rec = _ExecRec(placed=[], cancel_all_count=0)
    client_rec = _ClientRec(ioc_calls=[], market_close_calls=[])
    stub = _make_bot_stub(
        settings=settings,
        state=state,
        exec_rec=exec_rec,
        client_rec=client_rec,
    )

    market = _make_market(best_bid=2.040, best_ask=2.041)
    Bot._run_sf_phase_ladder_dispatch(
        stub,
        pos_qty=-9.0,
        market=market,
        wo_bid=None,
        wo_ask=None,
        tick_size=0.001,
    )

    assert state.sf_phase_ladder_phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK
    assert state.sf_consecutive_rejects_in_phase == 0


# ---------------------------------------------------------------------------
# Sentinel: dispatcher exists, branches on settings flag
# ---------------------------------------------------------------------------


def test_dispatcher_method_exists_on_bot() -> None:
    """Sentinel for the wiring entry point — the method must exist on
    Bot for ``_run_soft_flatten_tick`` to branch into. If a refactor
    deletes it, the SF tick stalls when the operator enables the
    ladder via the env var."""
    assert hasattr(Bot, "_run_sf_phase_ladder_dispatch")


def test_soft_flatten_tick_branches_on_settings_flag() -> None:
    """Sentinel: ``_run_soft_flatten_tick`` checks the
    ``sf_phase_ladder_enabled`` setting and calls the dispatcher.
    Source-text contract; behaviour is exercised by the integration
    tests above."""
    import inspect

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "sf_phase_ladder_enabled" in src
    assert "_run_sf_phase_ladder_dispatch" in src


def test_okx_client_has_place_ioc_reduce_only() -> None:
    """The OKX adapter is the one the operator's bot is running.
    Phase-ladder phases 2/3 require the IOC transport — without it the
    dispatcher falls back to market_close immediately."""
    import inspect

    from app.exchange.okx_client import OkxClient

    sig = inspect.signature(OkxClient.place_ioc_reduce_only)
    assert "symbol" in sig.parameters
    assert "is_buy" in sig.parameters
    assert "sz" in sig.parameters
    assert "limit_px" in sig.parameters
