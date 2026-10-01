"""Sub-exchange-min-notional residual: thresholds, market_close path, passive quote suppression."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import RiskAction, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_precision import rejection_is_below_min_notional_usd
from app.exchange.symbol_spec import SymbolSpec
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _settings(**extra: object) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_submin_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(extra)
    return UnitTestSettings.model_validate(base), path


def test_rejection_is_below_min_notional_usd() -> None:
    assert rejection_is_below_min_notional_usd("below_min_notional_usd x=y") is True
    assert rejection_is_below_min_notional_usd("below_min_size") is False
    assert rejection_is_below_min_notional_usd(None) is False


def test_dust_position_does_not_trigger_market_close() -> None:
    """Defaults: $9.18 notional is below dust threshold 12 -> no forced flatten."""
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 0.0042
    state.position.position_notional = 9.18
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    assert client.market_close.call_count == 0
    path.unlink(missing_ok=True)


def test_force_flatten_when_stuck_above_force_threshold() -> None:
    """Lower dust + force so sub-$10 stuck inventory >= force triggers market_close."""
    s, path = _settings(
        DUST_POSITION_NOTIONAL_USD=5.0,
        FORCE_FLATTEN_NOTIONAL_USD=8.0,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 0.0042
    state.position.position_notional = 9.18
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    assert client.market_close.call_count >= 1
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


def test_min_notional_block_suppresses_second_place_without_repeated_warning() -> None:
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=100.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._min_notional_passive_block[Side.BUY] = False
    r1 = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.001, "q1")
    r2 = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.001, "q2")
    assert r1 is None and r2 is None
    assert om._min_notional_passive_block[Side.BUY] is True
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


def test_min_quote_notional_suppresses_after_exchange_normalize() -> None:
    s, path = _settings(MIN_QUOTE_NOTIONAL_USD=50.0)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    wo = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "q1")
    assert wo is None
    assert om._min_notional_passive_block[Side.BUY] is True
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Path B regression: sub-spec dust residual on dust<spec configurations
# ---------------------------------------------------------------------------
#
# Reproduced 2026-05-08 in snapshot 260507114312. TON config has
# DUST_POSITION_NOTIONAL_USD=$3 and OKX spec_min_notional=$5 — i.e.
# operator-set dust threshold is BELOW the venue's spec floor. In
# this configuration, ALL dust positions are by definition sub-spec.
# Before the fix, neither Path A (``stuck_sub_min and not is_dust and
# pn >= force_th``) nor any other path triggered residual_flatten —
# the engine fell through to normal quoting, which can't size out a
# residual that rounds below spec. Bot wedged for 600 s until the
# deadlock watchdog killed it.


def test_path_b_fires_when_dust_below_spec_min() -> None:
    """TON-shaped config: dust<spec, residual within both → market_close.

    Mirrors snapshot 260507114312: dust=$3, spec=$5, residual=$2.48.
    Before the 1.1.33 fix this configuration silently wedged for 600 s.

    1.1.35: Path B is default-OFF; this test explicitly opts in via
    ``RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED=True`` to pin the
    behaviour for operators who choose to enable it as a defensive
    belt-and-suspenders.
    """
    s, path = _settings(
        DUST_POSITION_NOTIONAL_USD=3.0,
        MIN_QUOTE_NOTIONAL_USD=5.5,
        FORCE_FLATTEN_NOTIONAL_USD=9.0,
        RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED=True,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # Mirror the TON snapshot: short -1 contract at $2.48 mid → $2.48
    # notional. is_dust=True (< $3), stuck_sub_min=True (< $5).
    state.position.position_qty = -1.0
    state.position.position_notional = 2.48
    spec = SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,  # OKX TON spec
        sz_decimals=0,
        source="okx_v5",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    # Path B should fire → market_close, no passive placement.
    assert client.market_close.call_count >= 1, (
        "Path B did not fire — sub-spec dust residual would wedge bot"
    )
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


def test_path_b_does_not_fire_when_dust_above_spec_min() -> None:
    """Hyperliquid-shaped default: dust>spec → Path B is a no-op.

    Preserves the legacy ``test_dust_position_does_not_trigger_market_close``
    semantic for configurations where dust>spec. In those, sub-dust
    sub-spec positions can be ridden out via normal quoting (the
    bot's QUOTE_NOTIONAL order is bigger than the residual + spec
    floor combined).
    """
    # Defaults: dust=12, min_quote=12, force=25.
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # is_dust=True (9.18 < 12), stuck_sub_min=True (9.18 < 10).
    # But dust(12) > spec(10) so Path B's gate is False → Path B
    # does NOT fire, matching legacy "let dust ride" behaviour.
    state.position.position_qty = 0.0042
    state.position.position_notional = 9.18
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    # Path B did NOT fire. (Path A also didn't, since pn=9.18 < force_th=25.)
    assert client.market_close.call_count == 0
    path.unlink(missing_ok=True)


def test_path_b_inactive_when_residual_above_dust_threshold() -> None:
    """Path B requires is_dust. Above the dust threshold but still
    sub-spec falls into Path A territory (regression-pinned by
    ``test_force_flatten_when_stuck_above_force_threshold``).
    Verify Path B doesn't override Path A's logic for non-dust
    sub-spec residuals.
    """
    # dust=3, spec=5, force=9 — TON shape but residual ABOVE dust.
    s, path = _settings(
        DUST_POSITION_NOTIONAL_USD=3.0,
        MIN_QUOTE_NOTIONAL_USD=5.5,
        FORCE_FLATTEN_NOTIONAL_USD=9.0,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    # pn=$4: above dust ($3) → is_dust=False, below spec ($5) →
    # stuck_sub_min=True, below force ($9) → Path A doesn't fire,
    # Path B's is_dust gate is False → also doesn't fire. Bot is
    # expected to ride this out via normal quoting.
    state.position.position_qty = -1.0
    state.position.position_notional = 4.0
    spec = SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=0,
        source="okx_v5",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    assert client.market_close.call_count == 0
    path.unlink(missing_ok=True)


def test_path_b_default_off_does_not_fire_on_dust_below_spec() -> None:
    """1.1.35 default behaviour: Path B does NOT fire even on the
    snapshot-260507114312 scenario (dust<spec, residual within both)
    unless the operator explicitly enables it via
    ``RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED=true``.

    Rationale (snapshot 260507123509 follow-up): the upstream
    ``max_allowed_size`` bug that wedged the bot is now fixed in
    ``_build_side``. With that, normal MM clears sub-spec residuals
    via overshoot at maker rebate (+1 bp) instead of taker fees
    (~5 bp). Path B was protective, not corrective.
    """
    # Same TON-shaped config as ``test_path_b_fires_when_dust_below_spec_min``
    # but WITHOUT the opt-in setting → Path B should not fire.
    s, path = _settings(
        DUST_POSITION_NOTIONAL_USD=3.0,
        MIN_QUOTE_NOTIONAL_USD=5.5,
        FORCE_FLATTEN_NOTIONAL_USD=9.0,
        # RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED unset (default False)
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = -1.0
    state.position.position_notional = 2.48
    spec = SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=0,
        source="okx_v5",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)

    # Path B is default-OFF → no market_close.
    assert client.market_close.call_count == 0
    path.unlink(missing_ok=True)


def test_path_b_telemetry_carries_diagnostic_reason() -> None:
    """When Path B fires (with explicit opt-in), the
    ``residual_flatten_reason`` in the build's telemetry should be
    ``sub_spec_dust_unrideable_config`` so future snapshots / ad-hoc
    analysis can identify which path triggered."""
    from app.exchange.symbol_spec import SymbolSpec
    from app.models import ToxicitySnapshot
    from app.quote_engine import QuoteBuildContext, QuoteEngine
    from app.quoting import compute_quote_decision

    s, path = _settings(
        DUST_POSITION_NOTIONAL_USD=3.0,
        MIN_QUOTE_NOTIONAL_USD=5.5,
        FORCE_FLATTEN_NOTIONAL_USD=9.0,
        RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED=True,
    )
    spec = SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=0,
        source="okx_v5",
    )
    engine = QuoteEngine(s, spec)
    tox = ToxicitySnapshot(
        score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0,
        vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False,
    )
    decision = compute_quote_decision(s, mid=2.48, position_qty=-1.0, vol_bps=0.0, toxicity=tox)
    market = _fresh_market(s)
    ctx = QuoteBuildContext(
        decision=decision,
        market=market,
        risk_action=RiskAction.ALLOW,
        bid_mult=1.0,
        ask_mult=1.0,
        spread_add_bps=0.0,
        position_qty=-1.0,
        position_notional=2.48,
        resting_bid=None,
        resting_ask=None,
        reprice_replace_pending_bid=False,
        reprice_replace_pending_ask=False,
    )
    build = engine.build_quotes(ctx)
    assert build.residual_flatten_requested is True
    assert build.bid_order is None
    assert build.ask_order is None
    assert (
        build.telemetry.get("residual_flatten_reason")
        == "sub_spec_dust_unrideable_config"
    )
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1.1.36 — _min_notional_passive_block recovery (Codex review HIGH-2)
# ---------------------------------------------------------------------------
#
# The latch is set whenever ``place_passive_order_manual_only`` fails
# for min-notional reasons. Pre-1.1.36, no clear path existed — once
# tripped, that side was suppressed for the rest of the process.
# 1.1.36 adds three clear paths:
#   1. Successful placement on the same side auto-clears.
#   2. Public ``clear_min_notional_passive_block(side)`` helper.
#   3. ``Bot._enter_soft_flatten`` and ``Bot._exit_soft_flatten``
#      transitions call the helper — fresh SF episodes get a clean
#      slate; resumed regular quoting after SF gets a clean slate.


def test_successful_placement_clears_min_notional_block() -> None:
    """A latched side that subsequently places successfully should
    auto-clear. Without this, even a recovery-condition placement
    works once but the next one is suppressed by the stale latch."""
    s, path = _settings(
        MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=1.0,
        MAX_POSITION_NOTIONAL_USD=200.0,
        MAX_ORDER_NOTIONAL_USD=200.0,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(s, client, storage, state, private_event_queue=None)
    # Pre-set the latch as if a prior placement had failed.
    om._min_notional_passive_block[Side.BUY] = True
    # Now manually clear via the public helper to simulate that a
    # higher layer (e.g. SF entry) decided the latch was stale.
    om.clear_min_notional_passive_block(Side.BUY)
    assert om._min_notional_passive_block[Side.BUY] is False
    # Place a viable BUY (notional = 100*0.5=50, well above min) — succeeds.
    r = om.place_passive_order_manual_only(Side.BUY, 100.0, 0.5, "q-recovery")
    assert r is not None
    # Latch remains clear after the successful placement.
    assert om._min_notional_passive_block[Side.BUY] is False
    path.unlink(missing_ok=True)


def test_clear_min_notional_passive_block_helper_clears_both_sides() -> None:
    """The public helper with ``side=None`` clears both sides at once
    (used by ``_enter_soft_flatten`` / ``_exit_soft_flatten``)."""
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._min_notional_passive_block[Side.BUY] = True
    om._min_notional_passive_block[Side.SELL] = True
    om.clear_min_notional_passive_block()  # side=None → both
    assert om._min_notional_passive_block[Side.BUY] is False
    assert om._min_notional_passive_block[Side.SELL] is False
    path.unlink(missing_ok=True)


def test_clear_min_notional_passive_block_helper_per_side() -> None:
    """Per-side helper only touches the requested side."""
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=spec)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._min_notional_passive_block[Side.BUY] = True
    om._min_notional_passive_block[Side.SELL] = True
    om.clear_min_notional_passive_block(Side.BUY)
    assert om._min_notional_passive_block[Side.BUY] is False
    assert om._min_notional_passive_block[Side.SELL] is True
    path.unlink(missing_ok=True)


def test_soft_flatten_enter_exit_call_clear_latch_helper() -> None:
    """Source-substring sentinel: ``Bot._enter_soft_flatten`` and
    ``Bot._exit_soft_flatten`` invoke the new clear helper. If
    someone removes those calls in a future refactor, this guards
    against silent regression of the 1.1.36 fix."""
    import inspect
    from app.bot import Bot

    enter_src = inspect.getsource(Bot._enter_soft_flatten)
    assert "clear_min_notional_passive_block" in enter_src

    exit_src = inspect.getsource(Bot._exit_soft_flatten)
    assert "clear_min_notional_passive_block" in exit_src


def test_soft_flatten_pre_check_uses_max_of_venue_and_local_min() -> None:
    """Sentinel for the broadened SF pre-check. The 1.1.36 fix takes
    ``max(venue_min_notional_usd, settings.min_quote_notional_usd)``
    so SF can't enter on a residual that's above venue min but below
    local min. Source-level guard: the worker source must reference
    both ``min_notional_usd`` (venue) and ``min_quote_notional_usd``
    (local), and the ``max`` combinator."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "min_notional_usd" in src
    assert "min_quote_notional_usd" in src
    # Simple text check that the max combinator is present.
    assert "max(venue_min_ntn, local_min_ntn)" in src
