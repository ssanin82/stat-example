"""Tests for ``app.backtest.bot_runner`` — Phase 4b (v1.4.236).

Real-Bot replay integration. Each test instantiates a fresh Bot
against a synthetic fixture and verifies:

* ``BotRunner`` constructs without raising
* The bot's ``one_tick()`` completes without raising on each tick
* State mutations propagate (``state.market`` updated from BBO,
  Binance basis EWMA fills in over time)
* ``extract_gates_fired()`` returns counter dicts (zero counts are OK)
* The replay driver's ``replay(bot_runner=runner)`` mode end-to-ends

Each test uses ``--new__``-bypass-free construction — the real Bot
constructor runs in full. SQLite uses unique temp paths so concurrent
tests don't collide. ``TRADING_ENABLED=True`` lets the strategy
actually run; the paper executor handles the order routing.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from queue import Queue
from typing import Optional

import pytest

from app.backtest import (
    DEFAULT_REPLAY_WARMUP_SECONDS,
    PaperExecutorConfig,
    ReplayConfig,
    SortedEventStream,
    build_bot_runner,
    build_harness,
    make_replay_settings,
    replay,
    seed_state_from_snapshot,
)
from app.clock import ReplayClock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _temp_db_path() -> Path:
    return (
        Path(tempfile.gettempdir())
        / f"replay_bot_runner_{os.getpid()}_{uuid.uuid4().hex}.db"
    )


def _build_fixture(tmp_path: Path) -> Path:
    """8 seconds of OKX BBO + Binance bookTicker, deterministic."""
    fixture = tmp_path / "synth"
    fixture.mkdir(parents=True, exist_ok=True)
    base_t = 1_700_000_000_000_000_000
    base_ts_ms = base_t // 1_000_000  # ns → ms for the OKX "ts" field
    okx_events = [
        {"t_recv_ns": base_t + i * 200_000_000,
         "source": "okx_public",
         "msg": {"arg": {"channel": "bbo-tbt"},
                 "data": [{"asks": [[f"{2.05 + i*0.001:.3f}", "100", "0", "1"]],
                           "bids": [[f"{2.04 + i*0.001:.3f}", "100", "0", "1"]],
                           "ts": str(base_ts_ms + i * 200),  # ms granularity
                           "seqId": i}]}}
        for i in range(40)
    ]
    binance_events = [
        {"t_recv_ns": base_t + i * 250_000_000 + 50_000_000,
         "source": "binance_public",
         "msg": {"e": "bookTicker", "s": "TONUSDT",
                 "b": f"{2.040 + i*0.001:.3f}", "B": "200",
                 "a": f"{2.050 + i*0.001:.3f}", "A": "200",
                 "T": 1779353641000 + i * 250}}
        for i in range(32)
    ]
    with gzip.open(fixture / "okx_public.jsonl.gz", "wt") as f:
        for d in okx_events:
            f.write(json.dumps(d) + "\n")
    with gzip.open(fixture / "binance_public.jsonl.gz", "wt") as f:
        for d in binance_events:
            f.write(json.dumps(d) + "\n")
    (fixture / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "session_id": "bot-runner-synth",
        "recorder_version": "0.1.0",
        "is_finalized": True,
        "files": [
            {"path": "okx_public.jsonl.gz", "source": "okx_public",
             "lines": len(okx_events),
             "first_t_recv_ns": okx_events[0]["t_recv_ns"],
             "last_t_recv_ns": okx_events[-1]["t_recv_ns"], "gaps": 0},
            {"path": "binance_public.jsonl.gz", "source": "binance_public",
             "lines": len(binance_events),
             "first_t_recv_ns": binance_events[0]["t_recv_ns"],
             "last_t_recv_ns": binance_events[-1]["t_recv_ns"], "gaps": 0},
        ],
    }))
    return fixture


def _make_runner(start_t_ns: int, *, overrides: Optional[dict] = None):
    """Settings + harness + runner factory."""
    settings = make_replay_settings(
        symbol="TON-USDT-SWAP",
        overrides=overrides,
        db_path=_temp_db_path(),
    )
    harness = build_harness(
        start_t_ns=start_t_ns,
        config=ReplayConfig(),
    )
    runner = build_bot_runner(settings=settings, harness=harness)
    return settings, harness, runner


# ---------------------------------------------------------------------------
# make_replay_settings
# ---------------------------------------------------------------------------


def test_make_replay_settings_returns_valid_settings() -> None:
    s = make_replay_settings()
    assert s.symbol == "TON-USDT-SWAP"
    assert s.trading_enabled is True
    assert s.binance_ws_enabled is False


def test_make_replay_settings_accepts_overrides() -> None:
    s = make_replay_settings(overrides={"MAX_ABS_POSITION": 25.0})
    assert s.max_abs_position == 25.0


def test_make_replay_settings_unique_db_per_call() -> None:
    s1 = make_replay_settings()
    s2 = make_replay_settings()
    # Same env keys but distinct DB paths so concurrent replays don't
    # share schema.
    assert s1.database_url != s2.database_url


# ---------------------------------------------------------------------------
# BotRunner construction
# ---------------------------------------------------------------------------


def test_bot_runner_constructs_cleanly() -> None:
    """The most basic smoke check — does the wiring complete without
    raising?"""
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    assert runner.bot is not None
    assert runner.state is not None
    assert runner.storage is not None
    # Bot's deps point at the harness's components.
    assert runner.bot._client is runner.harness.paper
    assert runner.bot._public_stream is runner.harness.public_stream
    assert runner.bot._private_stream is runner.harness.private_stream
    assert runner.bot._clock is runner.harness.clock


def test_bot_runner_disables_private_stream_forwarding() -> None:
    """--with-bot contract: build_bot_runner MUST flip the recorded
    private stream's forward_to_sink to False so the original live
    session's order-updates/fills don't get pushed into the queue the
    bot drains. Forwarding them bumps the bot's shadow position vs
    paper-exec's true position → shadow_position_divergence →
    private_ws_overflow_recovery storm (v1.5.294 audit). Paper-exec is
    the authoritative private-event source in with-bot mode."""
    _, harness, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    assert harness.private_stream.forward_to_sink is False
    assert runner.bot._private_stream.forward_to_sink is False


def test_bot_runner_one_tick_runs_without_raising() -> None:
    """Five consecutive ticks with no market data should not raise."""
    _, harness, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    for _ in range(5):
        harness.clock.advance_to(harness.clock._t_ns + 500_000_000)
        runner.one_tick()
    # No assertion needed — absence of exception is the test.


# ---------------------------------------------------------------------------
# State mutation via BBO callbacks
# ---------------------------------------------------------------------------


def test_okx_bbo_updates_state_market(tmp_path: Path) -> None:
    """An OKX BBO event delivered via the harness's public stream
    should land in BotState.market (mirroring production wiring)."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, harness, runner = _make_runner(start_t_ns=events.first_t_ns)

    # Pull the first OKX event and deliver it.
    okx_event = next(e for e in events if e.source == "okx_public")
    harness.clock.advance_to(okx_event.t_recv_ns)
    harness.public_stream.deliver(okx_event)

    assert runner.state.market is not None
    assert runner.state.market.best_bid is not None
    assert runner.state.market.best_ask is not None
    assert runner.state.market.best_bid < runner.state.market.best_ask


def test_binance_bbo_updates_state_basis(tmp_path: Path) -> None:
    """A Binance BBO event after an OKX BBO should fold into the
    basis EWMA on BotState."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, harness, runner = _make_runner(start_t_ns=events.first_t_ns)

    # Need OKX BBO first so state.market has a mid for the basis calc.
    delivered_okx = False
    delivered_binance = False
    for e in events:
        if e.source == "okx_public" and not delivered_okx:
            harness.clock.advance_to(e.t_recv_ns)
            harness.public_stream.deliver(e)
            delivered_okx = True
        elif e.source == "binance_public" and delivered_okx and not delivered_binance:
            harness.clock.advance_to(e.t_recv_ns)
            harness.public_stream.deliver(e)
            delivered_binance = True
        if delivered_okx and delivered_binance:
            break

    assert runner.state.binance_best_bid is not None
    assert runner.state.binance_mid is not None
    # Basis EWMA initialised after first folding.
    assert runner.state.binance_basis_ewma is not None


# ---------------------------------------------------------------------------
# extract_gates_fired
# ---------------------------------------------------------------------------


def test_extract_gates_fired_returns_dict() -> None:
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    gates = runner.extract_gates_fired()
    assert isinstance(gates, dict)
    # Zero-counts for a freshly-constructed runner.
    for k, v in gates.items():
        assert isinstance(k, str)
        assert isinstance(v, int)


def test_extract_gates_fired_after_ticks() -> None:
    """After running ticks the counter dict should still be non-
    decreasing (counters never go down)."""
    _, harness, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    before = runner.extract_gates_fired()
    for _ in range(3):
        harness.clock.advance_to(harness.clock._t_ns + 500_000_000)
        runner.one_tick()
    after = runner.extract_gates_fired()
    for k in before.keys() & after.keys():
        assert after[k] >= before[k], f"counter {k} decreased: {before[k]} → {after[k]}"


def test_extract_gates_fired_auto_discovers_new_gate() -> None:
    """BUG-A fix (v1.5.302): a gate the extractor was NEVER told about
    still surfaces in the report, keyed by its bare name.

    Before the fix, ``extract_gates_fired`` hard-coded ~11 attribute
    names, so any gate added after that list was written (microprice_
    gate, at_touch_adverse_pause, sf_fatigue, inventory_skew, …) was
    silently dropped from replay reports. The fix folds in every gate
    the canonical ``gate_attribution_snapshot`` registry knows about.
    """
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    state = runner.state

    # None of these names are in the old hard-coded list — they only
    # surface via the auto-discovery path. Record a rising edge for each
    # (was-inactive → firing-now bumps fire_count to 1).
    state.record_gate_firing("microprice_gate", firing_now=True, now_mono=100.0)
    state.record_gate_firing(
        "at_touch_adverse_pause", firing_now=True, now_mono=100.0
    )

    gates = runner.extract_gates_fired()

    # Bare gate name (no prefix) — matches session_gate_fire_stats and
    # the dashboard's rendering.
    assert gates.get("microprice_gate") == 1
    assert gates.get("at_touch_adverse_pause") == 1
    # Contract preserved: all values are plain ints.
    for k, v in gates.items():
        assert isinstance(k, str)
        assert isinstance(v, int)

    # A second rising edge (after a falling edge) increments the count,
    # and the extractor reflects it — proving it reads live, not a
    # one-shot snapshot.
    state.record_gate_firing("microprice_gate", firing_now=False, now_mono=101.0)
    state.record_gate_firing("microprice_gate", firing_now=True, now_mono=102.0)
    assert runner.extract_gates_fired().get("microprice_gate") == 2


def test_extract_gates_fired_includes_aqc_counters() -> None:
    """BUG-A fix (v1.5.302): the Active Quoting Controller's session
    counters surface under an ``aqc_`` namespace.

    ``update_count`` alone is too generic to stand at top level, so it
    is namespaced. The controller is constructed unconditionally on
    BotState, so the keys are always present (zero on a fresh runner).
    """
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    gates = runner.extract_gates_fired()
    assert "aqc_update_count" in gates
    assert "aqc_safety_floor_engagement_count" in gates
    assert isinstance(gates["aqc_update_count"], int)
    assert isinstance(gates["aqc_safety_floor_engagement_count"], int)


def test_extract_gates_fired_ops_counters_read_live_session_attrs() -> None:
    """BUG-C fix (v1.5.303): the operational counters read the REAL,
    live ``session_*`` BotState attributes.

    The original list mapped ``place_attempts``/``place_rejects``/
    ``cancel_attempts`` to phantom ``*_total`` names that never existed
    on BotState, so ``getattr(s, attr, 0)`` reported a constant 0 in
    EVERY replay — making real place/cancel/reject activity invisible.
    This test drives the genuine counters and asserts they surface.
    """
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    state = runner.state
    # Drive the real session counters the way the live bot does.
    state.session_place_attempt_count = 7
    state.session_place_reject_count_total = 2
    state.session_cancel_attempt_count = 5
    state.session_cancel_reject_count_total = 1
    gates = runner.extract_gates_fired()
    assert gates["place_attempts"] == 7
    assert gates["place_rejects"] == 2
    assert gates["cancel_attempts"] == 5
    assert gates["cancel_rejects"] == 1


def test_extract_gates_fired_warns_once_on_missing_ops_counter(caplog) -> None:
    """BUG-C fix (v1.5.303): a missing ops counter (renamed/removed
    upstream) is (a) OMITTED from the report — never silently 0 — and
    (b) named in a one-shot WARN so the wiring drift is visible instead
    of masquerading as a real zero."""
    import app.backtest.bot_runner as br
    # Fresh dedup so this test observes the WARN regardless of order.
    br._WARNED_MISSING_OPS_COUNTERS.discard(
        "place_attempts<-session_place_attempt_count"
    )

    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    state = runner.state
    assert hasattr(state, "session_place_attempt_count")
    assert "place_attempts" in runner.extract_gates_fired()

    # Simulate an upstream rename/removal of the backing attr.
    delattr(state, "session_place_attempt_count")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.backtest.bot_runner"):
        gates = runner.extract_gates_fired()
    # (a) absent, not a silent 0.
    assert "place_attempts" not in gates
    # (b) one-shot WARN names the report_key<-attr pair.
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "place_attempts<-session_place_attempt_count" in msgs

    # One-shot: a second extract does NOT re-warn (dedup set holds it).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.backtest.bot_runner"):
        runner.extract_gates_fired()
    msgs2 = "\n".join(r.getMessage() for r in caplog.records)
    assert "place_attempts<-session_place_attempt_count" not in msgs2


# ---------------------------------------------------------------------------
# is_quoting
# ---------------------------------------------------------------------------


def test_is_quoting_false_with_no_orders() -> None:
    _, _, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    assert runner.is_quoting() is False


def test_is_quoting_true_after_paper_executor_place() -> None:
    """Direct paper executor placement — not via the bot — proves
    the is_quoting check reads the paper book correctly."""
    _, harness, runner = _make_runner(start_t_ns=1_700_000_000_000_000_000)
    harness.paper.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    harness.paper.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000,
    )
    assert runner.is_quoting() is True


# ---------------------------------------------------------------------------
# End-to-end driver integration
# ---------------------------------------------------------------------------


def test_replay_driver_drives_real_bot(tmp_path: Path) -> None:
    """Full end-to-end: replay() with bot_runner=... drives the real
    bot through every tick on a synthetic fixture."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, _, runner = _make_runner(start_t_ns=events.first_t_ns)

    cfg = ReplayConfig(tick_interval_s=0.5)
    report = replay(fixture, config=cfg, bot_runner=runner)

    # Bot ran — state mutated.
    assert runner.state.market is not None
    assert runner.state.binance_best_bid is not None
    # Replay completed.
    assert report.total_events > 0
    assert report.ticks_executed > 0
    # gates_fired is populated (zero counts OK) when bot_runner is wired.
    assert isinstance(report.gates_fired, dict)


def test_replay_driver_rejects_runner_with_wrong_clock(tmp_path: Path) -> None:
    """Building a runner at start_t_ns ≠ events.first_t_ns must be
    rejected — the clock can't replay forward from the wrong anchor."""
    fixture = _build_fixture(tmp_path)
    # Construct runner with a clock anchor that doesn't match fixture.
    _, _, runner = _make_runner(start_t_ns=1)  # way before fixture
    with pytest.raises(ValueError, match="clock starts at"):
        replay(fixture, bot_runner=runner)


def test_replay_with_runner_byte_identical_across_two_runs(tmp_path: Path) -> None:
    """Phase 4b shouldn't break Phase 3's determinism contract."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, _, r1 = _make_runner(start_t_ns=events.first_t_ns)
    _, _, r2 = _make_runner(start_t_ns=events.first_t_ns)
    j1 = replay(fixture, bot_runner=r1).to_json(exclude_wall_time=True)
    j2 = replay(fixture, bot_runner=r2).to_json(exclude_wall_time=True)
    # Note: gates_fired may legitimately differ across runs if any
    # gate depends on memory addresses / hash randomization (e.g.
    # OrderedDict iteration). All counters we extract are pure-count
    # though, so determinism should hold.
    assert j1 == j2


# ---------------------------------------------------------------------------
# Warmup window pre-rolls state without quoting (audit P2 #14)
# ---------------------------------------------------------------------------


def test_warmup_window_prerolls_state_without_quoting(tmp_path: Path) -> None:
    """audit P2 #14 — the corrected warmup contract.

    The earlier bot_runner docstring claimed warmup "gates tick-callback
    execution but does NOT pre-roll any history." That was wrong: the
    driver dispatches EVERY event during the warmup window (only the
    bot's ``one_tick`` is gated — see ``driver.py`` ``_maybe_fire_ticks``).
    So the market-data estimators warm purely from event callbacks while
    the bot stays silent.

    Set ``warmup_seconds`` past the whole fixture span so EVERY scheduled
    tick is gated, then assert (a) the bot never quoted, yet (b) the basis
    EWMA folded and the vol estimator warmed up from dispatched events.
    """
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, _, runner = _make_runner(start_t_ns=events.first_t_ns)

    # Fixture spans ~8 s; a 60 s warmup gates every scheduled tick.
    span_s = (events.last_t_ns - events.first_t_ns) / 1e9
    assert span_s < 60.0  # guard: the warmup below must exceed the span
    cfg = ReplayConfig(tick_interval_s=0.5, warmup_seconds=60.0)
    report = replay(fixture, config=cfg, bot_runner=runner)

    # (a) The bot never quoted — every scheduled tick fell inside warmup.
    assert report.ticks_executed == 0
    assert report.ticks_skipped_warmup > 0

    # (b) …yet events still dispatched, so market-data state pre-rolled:
    assert runner.state.market is not None
    # basis EWMA folded on Binance BBOs (needs a prior OKX mid — the
    # interleaved fixture supplies one before the first Binance event).
    assert runner.state.binance_basis_ewma is not None
    # vol estimator accrued >= _window distinct mids via the mid-change
    # listener fired by apply_market_book_only on each OKX BBO.
    assert runner.bot._vol.warmed_up is True


def test_default_replay_warmup_seconds_is_positive() -> None:
    """The CLI default warmup is a positive float (the whole point of
    audit P2 #14 — replays used to cold-start at 0.0). The driver's
    ``ReplayConfig`` default stays 0.0; only the *CLI* default moved."""
    assert isinstance(DEFAULT_REPLAY_WARMUP_SECONDS, float)
    assert DEFAULT_REPLAY_WARMUP_SECONDS > 0.0
    assert ReplayConfig().warmup_seconds == 0.0  # library default unchanged


# ---------------------------------------------------------------------------
# seed_state_from_snapshot — initial_state.json warm-start (audit P2 #14)
# ---------------------------------------------------------------------------


def _fresh_state_and_vol():
    """A bare BotState + its own VolatilityEstimator for unit-testing the
    seed function in isolation (no harness / clock needed)."""
    from app.state import BotState
    from app.volatility import VolatilityEstimator

    settings = make_replay_settings(db_path=_temp_db_path())
    return BotState(settings), VolatilityEstimator(settings)


def test_seed_state_from_snapshot_applies_scalars() -> None:
    state, vol = _fresh_state_and_vol()
    applied = seed_state_from_snapshot(
        state,
        {"binance_basis_ewma": 0.0123, "vol_bps": 7.5, "vol_sigma": 0.00075},
        vol_estimator=vol,
    )
    assert state.binance_basis_ewma == 0.0123
    assert state.vol_bps == 7.5
    assert state.vol_sigma == 0.00075
    # vol_bps seeded the estimator's cold-start sigma — served until the
    # live estimator warms (a warm-START, not an override).
    sigma, bps = vol.sigma_and_bps()
    assert sigma is not None
    assert bps == pytest.approx(7.5, rel=1e-9)
    # Report dict names each applied field (drives the caller's log line).
    assert applied["binance_basis_ewma"] == 1
    assert applied["vol_bps"] == 1
    assert applied["vol_sigma"] == 1
    assert applied["vol_estimator_seed"] == 1


def test_seed_state_from_snapshot_noop_on_empty_or_non_dict() -> None:
    state, vol = _fresh_state_and_vol()
    for snap in (None, {}, [], "x", 3):
        assert seed_state_from_snapshot(state, snap, vol_estimator=vol) == {}
    # Nothing was set.
    assert state.binance_basis_ewma is None
    assert state.vol_bps is None
    assert state.vol_sigma is None


def test_seed_state_from_snapshot_skips_invalid_values() -> None:
    """Non-finite / non-positive vol values degrade to a no-op for that
    field (never raise, never write a bad value)."""
    state, vol = _fresh_state_and_vol()
    applied = seed_state_from_snapshot(
        state,
        {
            "binance_basis_ewma": float("nan"),
            "vol_bps": -1.0,  # non-positive → skipped
            "vol_sigma": float("inf"),
        },
        vol_estimator=vol,
    )
    assert applied == {}
    assert state.binance_basis_ewma is None
    assert state.vol_bps is None
    assert state.vol_sigma is None
    # estimator NOT seeded → still cold (serves no sigma).
    sigma, bps = vol.sigma_and_bps()
    assert sigma is None
    assert bps == 0.0


def test_seed_state_from_snapshot_partial_snapshot() -> None:
    """Only the keys present are applied; absent keys stay at defaults."""
    state, vol = _fresh_state_and_vol()
    applied = seed_state_from_snapshot(
        state, {"binance_basis_ewma": -0.05}, vol_estimator=vol
    )
    assert applied == {"binance_basis_ewma": 1}
    assert state.binance_basis_ewma == -0.05
    assert state.vol_bps is None


def test_seed_state_from_snapshot_without_vol_estimator() -> None:
    """``vol_estimator=None`` still publishes vol_bps onto state (for the
    first-tick reads) — it just can't seed the estimator's cold-start."""
    state, _ = _fresh_state_and_vol()
    applied = seed_state_from_snapshot(state, {"vol_bps": 4.0})
    assert state.vol_bps == 4.0
    assert applied.get("vol_bps") == 1
    assert "vol_estimator_seed" not in applied


def test_build_bot_runner_warm_starts_from_initial_state() -> None:
    """build_bot_runner(initial_state=...) seeds the fresh state before
    the first tick (audit P2 #14)."""
    settings = make_replay_settings(db_path=_temp_db_path())
    harness = build_harness(
        start_t_ns=1_700_000_000_000_000_000, config=ReplayConfig()
    )
    runner = build_bot_runner(
        settings=settings,
        harness=harness,
        initial_state={"binance_basis_ewma": 0.02, "vol_bps": 5.0},
    )
    assert runner.state.binance_basis_ewma == 0.02
    assert runner.state.vol_bps == 5.0
    # The bot's live estimator carries the cold-start seed.
    sigma, bps = runner.bot._vol.sigma_and_bps()
    assert sigma is not None
    assert bps == pytest.approx(5.0, rel=1e-9)


def test_build_bot_runner_none_initial_state_is_cold() -> None:
    """Default (no initial_state) leaves the state cold — the warmup
    window is what warms it instead."""
    settings = make_replay_settings(db_path=_temp_db_path())
    harness = build_harness(
        start_t_ns=1_700_000_000_000_000_000, config=ReplayConfig()
    )
    runner = build_bot_runner(settings=settings, harness=harness)
    assert runner.state.binance_basis_ewma is None
    assert runner.state.vol_bps is None


# ---------------------------------------------------------------------------
# Real fixture acceptance (laptop-smoke-1)
# ---------------------------------------------------------------------------


_REAL_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "backtesting" / "data" / "sessions" / "laptop-smoke-1"
)


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_replay_with_real_bot() -> None:
    """The real 4-minute laptop-smoke-1 fixture should replay through
    a real Bot end-to-end without raising. ``time_in_quote_pct`` /
    ``gates_fired`` will reflect actual strategy behaviour."""
    events = SortedEventStream(_REAL_FIXTURE)
    settings = make_replay_settings(db_path=_temp_db_path())
    harness = build_harness(
        start_t_ns=events.first_t_ns,
        config=ReplayConfig(),
    )
    runner = build_bot_runner(settings=settings, harness=harness)
    report = replay(_REAL_FIXTURE, bot_runner=runner)
    # Report fields populated.
    assert report.total_events > 0
    assert report.ticks_executed > 0
    assert isinstance(report.gates_fired, dict)
    # State mutated.
    assert runner.state.market is not None
    # Wall time acceptable (real-Bot tick is heavier than paper-only;
    # acceptance §3.4's 30s cap covers 5-min fixture — we're 4 min).
    assert report.wall_time_s < 60.0, f"too slow: {report.wall_time_s}s"


# ---------------------------------------------------------------------------
# Phase 4c: end-to-end Bot→Outbound→Paper→Fill cycle
# ---------------------------------------------------------------------------


def test_starting_equity_unblocks_reconcile_stall(tmp_path: Path) -> None:
    """Phase 4c §4c.1: with starting_equity_usd > 0 the bot must NOT
    pause on reconcile_stall after 5 ticks.

    Pre-fix: PaperExecutor.fetch_account_snapshot returned equity=0
    → ``_exchange_snapshot_healthy`` returned False → pause after 5.
    """
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, _, runner = _make_runner(start_t_ns=events.first_t_ns)
    cfg = ReplayConfig(tick_interval_s=0.25)
    report = replay(fixture, config=cfg, bot_runner=runner)
    # After 8s of replay at 0.25s ticks = ~32 ticks; well past the 5-
    # tick reconcile_stall threshold.
    assert report.ticks_executed > 10
    # Status should NOT be PAUSED(reconcile_stall).
    from app.enums import BotStatus
    assert runner.state.bot_status in (BotStatus.RUNNING, BotStatus.STARTING)


def test_paper_executor_starting_equity_visible_to_bot(tmp_path: Path) -> None:
    """The bot's state.account should reflect the configured starting
    equity after the first REST refresh."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    settings = make_replay_settings(db_path=_temp_db_path())
    from app.backtest import PaperExecutorConfig
    cfg = ReplayConfig(
        paper=PaperExecutorConfig(starting_equity_usd=5000.0),
        tick_interval_s=0.25,
    )
    harness = build_harness(start_t_ns=events.first_t_ns, config=cfg)
    runner = build_bot_runner(settings=settings, harness=harness)
    replay(fixture, config=cfg, bot_runner=runner)
    assert runner.state.account is not None
    assert runner.state.account.equity_usd == pytest.approx(5000.0)


def test_state_seeding_via_warmup_window(tmp_path: Path) -> None:
    """Phase 4c §4c.3: BBO callbacks fire during the warmup window
    (when tick callbacks are gated off). After warmup, state.market
    and binance_basis_ewma must already be populated — no further
    seeding step required."""
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    _, harness, runner = _make_runner(start_t_ns=events.first_t_ns)
    # 2-second warmup with 0.25s tick cadence. Events fire throughout;
    # tick callbacks skipped for the first 8 ticks.
    cfg = ReplayConfig(tick_interval_s=0.25, warmup_seconds=2.0)
    report = replay(fixture, config=cfg, bot_runner=runner)
    # By the end of warmup, both state.market and binance_basis_ewma
    # should be populated.
    assert runner.state.market is not None
    assert runner.state.market.best_bid is not None
    assert runner.state.binance_basis_ewma is not None
    assert report.ticks_skipped_warmup > 0
    assert report.ticks_executed > 0


def test_bot_to_paper_to_fill_pipeline_via_forced_placement(tmp_path: Path) -> None:
    """End-to-end Bot integration test that doesn't depend on the
    strategy choosing to quote. We:
    1. Build a runner against a synthetic fixture
    2. Run the replay with an on_tick that forces a place via the
       paper executor + a trade event right after to fill it
    3. Verify the synthetic fill emerges through the bot's
       drain_private_events on the next tick (state.position updates)

    This proves the outbound drain + paper-executor fill emission
    + private-event-queue flow all work end-to-end under the real
    Bot's tick path.
    """
    from app.backtest import PaperExecutorConfig
    fixture = _build_fixture(tmp_path)
    events = SortedEventStream(fixture)
    # Zero place-latency so the on_tick's trade fills the just-placed
    # order on the same monotonic instant (the driver advances the
    # clock between events but not within one tick callback).
    settings = make_replay_settings(db_path=_temp_db_path())
    cfg = ReplayConfig(
        tick_interval_s=0.5,
        paper=PaperExecutorConfig(sim_place_latency_s=0.0),
    )
    harness = build_harness(start_t_ns=events.first_t_ns, config=cfg)
    runner = build_bot_runner(settings=settings, harness=harness)

    placed = [False]
    fills_seen_via_queue: list = []

    def on_tick(t_ns, h):
        # Skip the first tick to give the bot time to populate state.
        # On the second qualifying tick, place an order via the paper
        # executor (bypassing strategy gates) and drive a trade large
        # enough to consume the queue + fill it. The next bot.one_tick
        # will drain the synthetic fill from the queue.
        if not placed[0] and h.paper._best_bid is not None and h.paper._best_ask is not None:
            # Place at the top of book.
            resp = h.paper.place_post_only_limit(
                symbol="TON-USDT-SWAP",
                is_buy=True,
                sz=5.0,
                limit_px=h.paper._best_bid,
            )
            placed[0] = True
            # SELL trade large enough to eat the visible queue (100)
            # + fill our 5-lot. Clock has already advanced past the
            # 20ms place-latency by the time the next event arrives;
            # we don't advance here (the driver owns the clock).
            h.paper.process_trade_event(
                price=h.paper._best_bid, size=200.0, side="SELL",
            )

    report = replay(fixture, config=cfg, bot_runner=runner, on_tick=on_tick)

    # The paper executor must have processed the fill.
    assert report.paper_executor["fills_emitted"] >= 1, \
        f"no fills observed: {report.paper_executor}"
    assert report.paper_executor["acks_emitted"] >= 1
    # The bot's position should reflect the fill (it drained the
    # PrivateFillEvent from the queue and applied it).
    assert runner.state.position is not None
    # Note: state.position_qty reflects the bot's view of position
    # after drain_private_events processes synthetic fills.
    assert runner.state.position.position_qty == pytest.approx(5.0, abs=0.01), (
        f"bot position mismatch: state={runner.state.position.position_qty}, "
        f"paper={runner.harness.paper._position_qty}"
    )
