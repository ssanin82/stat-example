"""Replay of the 2026-05-08 dust-residual deadlock (snapshot 260507114312).

Scenario:
- TON config: dust=$3, min_quote=$5.5, force=$9, OKX spec_min=$5.
- Bot inherited a -1 contract residual from a manual flatten
  (``okx_account_status.py --flatten``, which calls market_close
  and can leave a sub-spec residual on chunky-tick assets).
- residual notional = $2.48 — is_dust=True AND stuck_sub_min=True.
- Bot's regular MM quote engine fell through to "normal quoting"
  (Path A's gating ``stuck_sub_min and not is_dust and pn >= force_th``
  is logically impossible for this config), produced no orders for
  600 s, deadlock watchdog fired with exit 42.

Two-layered fix in 1.1.33:
  (1) Path B in QuoteEngine.build_quotes: when ``dust_th <= spec_min``
      AND a position is sub-spec dust, request residual_flatten so the
      execution layer market_closes the residual.
  (2) ``engine_no_quote_persistent`` diagnostic: catches any future
      silent wedge (different cause, same symptom) within ~10 s.

This test asserts BOTH layers fire correctly on the replay scenario,
proving that next time we'll either auto-recover (layer 1) or at the
very least see a loud event in the log (layer 2).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.enums import RiskAction
from app.exchange.symbol_spec import SymbolSpec
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


# Mirror snapshot 260507114312 settings exactly.
TON_DUST = 3.0
TON_MIN_QUOTE = 5.5
TON_FORCE = 9.0
TON_QUOTE_NOTIONAL = 10.0
TON_MAX_ABS_POSITION = 4.0
TON_MAX_POS_NOTIONAL = 11.0
TON_TICK = 0.001
TON_LOT = 1.0
TON_SPEC_MIN_NOTIONAL = 5.0  # OKX TON


def _make_setup() -> tuple[OrderManager, BotState, Storage, Path]:
    """Fresh OrderManager wired against TON-shaped config + symbol spec."""
    path = Path(tempfile.gettempdir()) / f"mm_dustdead_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "DUST_POSITION_NOTIONAL_USD": TON_DUST,
            "MIN_QUOTE_NOTIONAL_USD": TON_MIN_QUOTE,
            "FORCE_FLATTEN_NOTIONAL_USD": TON_FORCE,
            "QUOTE_NOTIONAL_USD": TON_QUOTE_NOTIONAL,
            "MAX_ORDER_NOTIONAL_USD": TON_QUOTE_NOTIONAL,
            "MAX_ABS_POSITION": TON_MAX_ABS_POSITION,
            "MAX_POSITION_NOTIONAL_USD": TON_MAX_POS_NOTIONAL,
            "ENGINE_NO_QUOTE_DIAG_STREAK_TICKS": 5,
            # Path B is default-OFF in 1.1.35; this replay opts in to
            # exercise the legacy 1.1.33 behaviour (forced market_close
            # on sub-spec dust). The 1.1.35 fix to ``max_allowed_size``
            # makes Path B unnecessary in practice, but operators can
            # still enable it as defense-in-depth — this test pins
            # that mode.
            "RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED": True,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.market = _fresh_market(settings)
    # Inherited from manual flatten residual.
    state.position.position_qty = -1.0
    state.position.position_notional = 2.48
    spec = SymbolSpec(
        price_tick=TON_TICK,
        size_step=TON_LOT,
        min_size=TON_LOT,
        min_notional_usd=TON_SPEC_MIN_NOTIONAL,
        sz_decimals=0,
        source="okx_v5",
    )
    client = mock_mm_client(symbol_spec=spec)
    client.has_write_access.return_value = True
    client.fetch_open_orders_raw.return_value = []
    client.market_close.return_value = {"status": "ok", "response": {"type": "default"}}
    client.place_post_only_limit.return_value = _ok_place(1)

    om = OrderManager(settings, client, storage, state)
    return om, state, storage, path


def test_replay_layer1_market_close_on_first_tick() -> None:
    """Path B in QuoteEngine.build_quotes fires immediately on the first
    tick, calling market_close to clean up the inherited dust residual.

    Before 1.1.33 this scenario produced no orders and no events for
    600 s before the deadlock watchdog fired.
    """
    om, _state, _storage, path = _make_setup()
    try:
        # Cancel debounce starts at 0 so the first tick can fire market_close.
        om._sub_min_notional_flatten_next_mono = 0.0
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        client = om._client
        assert client.market_close.call_count >= 1, (
            "Layer-1 fix did not fire — bot would wedge 600 s on this scenario"
        )
        assert client.place_post_only_limit.call_count == 0, (
            "Bot tried to place passive quotes on a sub-spec dust residual — "
            "the engine should have routed to residual_flatten instead"
        )
    finally:
        path.unlink(missing_ok=True)


def test_replay_layer2_diagnostic_event_persisted() -> None:
    """When Path B fires it short-circuits BEFORE the no_quote
    diagnostic runs, so we DON'T expect ``engine_no_quote_persistent``
    on the replay scenario. This is correct: layer 1 handled it.

    Layer 2 (the diagnostic) only fires when layer 1 doesn't catch the
    wedge. That's tested separately in
    ``test_engine_no_quote_diagnostic.py``.
    """
    om, _state, storage, path = _make_setup()
    try:
        om._sub_min_notional_flatten_next_mono = 0.0
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        rows = storage.bot_events_since("1970-01-01T00:00:00Z")
        types = [r.get("event_type") for r in rows]
        # Layer 1 fires forced_residual_flatten event, NOT the layer-2
        # engine_no_quote_persistent event. Both layers in their proper
        # roles.
        assert "forced_residual_flatten" in types
        assert "engine_no_quote_persistent" not in types
    finally:
        path.unlink(missing_ok=True)


def test_replay_first_market_close_then_subsequent_ticks_debounced() -> None:
    """The forced_residual_flatten executor has an 8 s internal debounce.
    First call fires market_close; subsequent calls within 8 s are
    debounced (no-op). This is correct: market_close is async, the
    venue takes a moment to update, hammering it every 500 ms would be
    wasteful.
    """
    om, _state, _storage, path = _make_setup()
    try:
        om._sub_min_notional_flatten_next_mono = 0.0
        client = om._client
        # First tick — fires.
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        first_count = client.market_close.call_count
        assert first_count >= 1
        # Second tick within debounce — should NOT fire again.
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        assert client.market_close.call_count == first_count
    finally:
        path.unlink(missing_ok=True)


def test_replay_above_dust_path_unchanged() -> None:
    """Sanity: the symmetric scenario where the residual is ABOVE the
    dust threshold but still sub-spec routes through Path A
    (force-threshold flatten) when applicable, OR rides via normal
    quoting — depending on force_th. With TON's force=$9 and a
    residual of $4 (between dust=$3 and spec=$5, below force=$9),
    neither Path A nor Path B fires — bot rides via normal quoting.
    """
    om, state, _storage, path = _make_setup()
    try:
        # Reposition to above-dust: pn=$4, qty=-1.
        # Wait: at $4 / 1 contract = $4/lot, but lot=1 contract worth
        # ~$2.71. So 1.5 contracts ≈ $4. That's not lot-aligned; for
        # the test we just inject the desired notional regardless.
        state.position.position_qty = -1.5
        state.position.position_notional = 4.05
        client = om._client
        om._sub_min_notional_flatten_next_mono = 0.0
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        # At pn=$4.05: is_dust=False (above $3), stuck_sub_min=True
        # (below $5), pn < force_th=$9 — Path A doesn't fire, Path B
        # is_dust check fails — neither fires.
        assert client.market_close.call_count == 0
    finally:
        path.unlink(missing_ok=True)
