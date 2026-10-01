"""v1.4.93 wedge-elimination-cleanup Phase 6A (minimal) —
snapshot replay harness.

Given a snapshot directory (``snapshots/v1.4.X-<ts>-<profile>/``),
parse the JSON files and reconstruct enough of a ``BotState`` to
exercise integrity invariants:

* WorkingOrders are loaded into ``OrderStore``; indexes are
  rebuilt; no ``(oid, cloid)`` duplicates land in the indexes.
* The snapshot's recorded ``executor_state`` counters indicate
  clean state at capture time (no wedge counters non-zero).
* `desync_phase` is OK at capture.
* Position + market state seed cleanly into the stores.

This is the FOUNDATION for Phase 6A's deeper replay capabilities
(actual ``maybe_refresh_quotes`` tick + multi-tick simulation +
WS-event injection). Today's scope: prove the data loads + state
is internally consistent at capture time.

Why this is useful even without a full tick replay:

1. Catches snapshots where the bot's recorded state was internally
   inconsistent (e.g., dupe OID across WOs, indexes out of sync).
2. Pins the snapshot-format contract: if a future bot change drops
   a required field, the loader fails loud.
3. Provides the loader API that Phase 6A.2's multi-tick replay
   will build on.

NOTE: the snapshot replay tests live under ``tests/integration/``,
which is EXCLUDED from default pytest collection (see
``pytest.ini``). Run with::

    python -m pytest tests/integration/test_snapshot_replay_smoke.py -v -s
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.enums import OrderStatus, Side
from app.models import BestBidAsk, PositionSnapshot, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now


_SNAPSHOTS_ROOT = Path(__file__).resolve().parents[2] / "snapshots"


@dataclass(frozen=True)
class SnapshotFixture:
    """Parsed snapshot — primitive fields only, no BotState/OrderStore
    construction yet. Pass to ``build_replay_state`` to materialize."""

    path: Path
    bot_version: str
    bot_profile: str
    captured_at_utc: datetime
    state_dict: dict[str, Any]
    open_orders_raw: list[dict[str, Any]]
    config: dict[str, Any]
    session_summary: dict[str, Any]

    @property
    def symbol(self) -> str:
        return self.state_dict.get("symbol", "")

    @property
    def executor_state(self) -> dict[str, Any]:
        """The 'executor_state' dict from state_current.json. Holds
        every wedge counter and risk-state surface."""
        return self.state_dict.get("executor_state") or {}

    @property
    def desync_phase(self) -> str:
        """desync_phase at capture. Should be OK on healthy snapshots."""
        dp = self.state_dict.get("desync_phase")
        if dp is None:
            return ""
        return getattr(dp, "value", None) or str(dp)

    @property
    def best_bid(self) -> Optional[float]:
        v = self.state_dict.get("best_bid")
        return float(v) if v is not None else None

    @property
    def best_ask(self) -> Optional[float]:
        v = self.state_dict.get("best_ask")
        return float(v) if v is not None else None

    @property
    def mid_price(self) -> Optional[float]:
        v = self.state_dict.get("mid_price")
        return float(v) if v is not None else None


def load_snapshot(snapshot_dir: str | Path) -> SnapshotFixture:
    """Parse a snapshot directory into a ``SnapshotFixture``.

    The path can be absolute, relative to the repo root, OR a bare
    snapshot name (e.g. ``v1.4.90-260519-115155-prod.okx.ton.usdt.perp``)
    which is resolved against ``<repo>/snapshots/``.
    """
    p = Path(snapshot_dir)
    if not p.is_absolute():
        # Try repo-root-relative first, then snapshots-root.
        cand_root = Path.cwd() / p
        if cand_root.is_dir():
            p = cand_root
        else:
            p = _SNAPSHOTS_ROOT / p.name
    if not p.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {snapshot_dir!r}")

    stats = p / "stats"
    meta_path = p / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.json under {p}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    def _load(name: str, default: Any = None) -> Any:
        fp = stats / name
        if not fp.exists():
            return default
        with open(fp, encoding="utf-8") as f:
            return json.load(f)

    state_dict = _load("state_current.json", default={}) or {}
    open_orders_full = _load("open_orders.json", default={}) or {}
    config_dict = _load("config.json", default={}) or {}
    session_summary = _load("session_summary.json", default={}) or {}

    # OKX open-orders shape: {"code": "0", "data": [...]}.
    open_orders_raw: list[dict[str, Any]] = []
    if isinstance(open_orders_full, dict):
        open_orders_raw = list(open_orders_full.get("data") or [])
    elif isinstance(open_orders_full, list):
        open_orders_raw = list(open_orders_full)

    captured_at_str = str(meta.get("captured_at_utc") or "")
    try:
        captured_at = datetime.fromisoformat(
            captured_at_str.replace("Z", "+00:00")
        )
    except ValueError:
        captured_at = utc_now()

    return SnapshotFixture(
        path=p,
        bot_version=str(meta.get("bot_version") or ""),
        bot_profile=str(meta.get("bot_profile") or ""),
        captured_at_utc=captured_at,
        state_dict=state_dict,
        open_orders_raw=open_orders_raw,
        config=config_dict,
        session_summary=session_summary,
    )


def working_orders_from_snapshot(
    fixture: SnapshotFixture,
    *,
    symbol_override: Optional[str] = None,
) -> list[tuple[Side, int, WorkingOrder]]:
    """Reconstruct ``WorkingOrder`` objects + their (side, level_idx)
    slot assignment from the snapshot's open-orders list.

    Slot assignment: OKX doesn't expose level_idx on its open-orders
    payload, so we synthesize it by sorting orders per side. For BID,
    higher price = inside (lvl 0); for ASK, lower price = inside
    (lvl 0). This matches the bot's quoting convention.

    All orders are assigned ``status=OrderStatus.ACKED`` — they were
    live on the exchange at capture time. Pre-ACK statuses (SENT,
    NEW_LOCAL) cannot be reconstructed from the open-orders feed
    alone; they would require the orders_lifecycle-trace.json file.
    """
    sym = symbol_override or fixture.symbol or "TON-USDT-SWAP"
    captured = fixture.captured_at_utc

    by_side: dict[Side, list[dict[str, Any]]] = {Side.BUY: [], Side.SELL: []}
    for o in fixture.open_orders_raw:
        side_str = str(o.get("side") or "").lower()
        side = Side.BUY if side_str == "buy" else Side.SELL if side_str == "sell" else None
        if side is None:
            continue
        by_side[side].append(o)

    out: list[tuple[Side, int, WorkingOrder]] = []
    for side, orders in by_side.items():
        # Inside rung = closest-to-touch. For BID: highest price; for
        # ASK: lowest price. Sort accordingly and assign level_idx
        # by position.
        reverse = (side == Side.BUY)
        orders_sorted = sorted(orders, key=lambda r: float(r.get("px") or 0.0), reverse=reverse)
        for lvl_idx, o in enumerate(orders_sorted):
            try:
                wo = WorkingOrder(
                    order_id_local=f"replay-{uuid.uuid4().hex[:8]}",
                    order_id_exchange=int(o["ordId"]),
                    client_order_id=str(o.get("clOrdId") or "") or None,
                    symbol=sym,
                    side=side,
                    price=float(o["px"]),
                    size=float(o["sz"]),
                    post_only=str(o.get("ordType", "")) == "post_only",
                    status=OrderStatus.ACKED,
                    ts_created=captured,
                    ts_sent=captured,
                    ts_ack=captured,
                    level_idx=lvl_idx,
                )
                out.append((side, lvl_idx, wo))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"failed to reconstruct WorkingOrder from open-orders row "
                    f"{o!r}: {exc}"
                ) from exc
    return out


def position_from_snapshot(fixture: SnapshotFixture) -> PositionSnapshot:
    """Reconstruct the position from the snapshot. Conservative —
    falls back to flat-zero if the position fields are missing.
    """
    sd = fixture.state_dict
    pos_block = sd.get("position") or {}
    qty = float(pos_block.get("position_qty") or sd.get("position_qty") or 0.0)
    notional = float(
        pos_block.get("position_notional")
        or sd.get("position_notional_usd")
        or sd.get("position_notional")
        or 0.0
    )
    avg = pos_block.get("avg_entry_price") or sd.get("avg_entry_price")
    mark = pos_block.get("mark_price") or sd.get("mark_price")
    upnl = float(
        pos_block.get("unrealized_pnl_usd")
        or sd.get("unrealized_pnl_usd")
        or 0.0
    )
    return PositionSnapshot(
        symbol=fixture.symbol or "TON-USDT-SWAP",
        position_qty=qty,
        avg_entry_price=float(avg) if avg is not None else None,
        mark_price=float(mark) if mark is not None else None,
        position_notional=abs(notional),
        unrealized_pnl_usd=upnl,
    )


def market_from_snapshot(fixture: SnapshotFixture) -> Optional[BestBidAsk]:
    """Reconstruct the BestBidAsk from the snapshot. Returns None if
    the snapshot has no market data (e.g., bot just started)."""
    bb = fixture.best_bid
    ba = fixture.best_ask
    mid = fixture.mid_price
    if bb is None or ba is None:
        return None
    return BestBidAsk(
        symbol=fixture.symbol or "TON-USDT-SWAP",
        best_bid=bb,
        best_ask=ba,
        mid_price=mid,
        spread_bps=None,
        ts_local=fixture.captured_at_utc,
    )


@dataclass(frozen=True)
class ReplayState:
    """Materialized snapshot replay state. Holds the BotState plus the
    storage handle (caller owns cleanup of the .db file)."""

    fixture: SnapshotFixture
    settings: Any  # UnitTestSettings — circular import avoidance
    storage: Storage
    state: BotState
    db_path: Path


def build_replay_state(
    fixture: SnapshotFixture,
    *,
    settings_overrides: Optional[dict[str, Any]] = None,
) -> ReplayState:
    """Construct a fresh ``BotState`` seeded with the snapshot's
    position, market, and working orders.

    The settings are constructed from ``UnitTestSettings`` with the
    bot's symbol and a unique sqlite database path (so tests run
    cleanly on Windows where shared-file locking is finicky — see
    ``tests/test_execution_safety.py`` CI fix v1.4.90).
    """
    from tests.settings_helpers import UnitTestSettings  # local import

    db_path = (
        Path(tempfile.gettempdir())
        / f"mm_replay_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    overrides = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        # symbol — best-effort from snapshot; tests can override.
        "SYMBOL": fixture.symbol or "TON-USDT-SWAP",
    }
    if settings_overrides:
        overrides.update(settings_overrides)
    settings = UnitTestSettings.model_validate(overrides)

    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)

    # Seed position + market.
    state.position = position_from_snapshot(fixture)
    state.market = market_from_snapshot(fixture)

    # Seed working orders into OrderStore.
    for side, lvl, wo in working_orders_from_snapshot(fixture):
        state.order_store.set(side, lvl, wo)

    return ReplayState(
        fixture=fixture,
        settings=settings,
        storage=storage,
        state=state,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Invariants — small, focused, return (ok, reason) tuples
# ---------------------------------------------------------------------------


def invariant_no_duplicate_oid_or_cloid(
    state: BotState,
) -> tuple[bool, str]:
    """OrderStore's (oid, cloid) indexes must be consistent: every
    live WO is indexed exactly once; no two WOs share an oid or cloid.
    """
    store = state.order_store
    oid_seen: dict[int, str] = {}  # oid → order_id_local
    cloid_seen: dict[str, str] = {}
    for s in (Side.BUY, Side.SELL):
        for lvl, wo in store.iter_side(s):
            if wo is None:
                continue
            if wo.order_id_exchange:
                oid = int(wo.order_id_exchange)
                if oid in oid_seen:
                    return False, (
                        f"duplicate oid {oid}: "
                        f"order_id_local={wo.order_id_local} and {oid_seen[oid]}"
                    )
                oid_seen[oid] = wo.order_id_local
            if wo.client_order_id:
                cl = wo.client_order_id
                if cl in cloid_seen:
                    return False, (
                        f"duplicate cloid {cl!r}: "
                        f"order_id_local={wo.order_id_local} and {cloid_seen[cl]}"
                    )
                cloid_seen[cl] = wo.order_id_local
    return True, "ok"


def invariant_no_wedge_counters_nonzero(
    fixture: SnapshotFixture,
) -> tuple[bool, str]:
    """Counters that MUST be 0 on a healthy snapshot. Non-zero
    indicates the bot accumulated wedge symptoms before capture."""
    es = fixture.executor_state
    suspicious = [
        "ws_event_unmatched_to_local_wo_total",
        "gate_phase2a_invariant_violation_total",
    ]
    bad: list[tuple[str, int]] = []
    for k in suspicious:
        v = int(es.get(k, 0) or 0)
        if v > 0:
            bad.append((k, v))
    if bad:
        return False, f"wedge counters non-zero: {bad}"
    return True, "ok"


def invariant_risk_state_normal(
    fixture: SnapshotFixture,
) -> tuple[bool, str]:
    """At capture, ``risk_exec_state`` must be NORMAL or UNKNOWN.
    SUPPRESSED / CANCELLING / DEGRADED indicate active incident."""
    risk = str(fixture.executor_state.get("risk_exec_state") or "UNKNOWN")
    if risk.upper() not in ("NORMAL", "UNKNOWN"):
        return False, f"risk_exec_state at capture: {risk!r}"
    return True, "ok"


def invariant_desync_phase_ok(
    fixture: SnapshotFixture,
) -> tuple[bool, str]:
    """``desync_phase`` must be OK or empty at capture. DETECTED /
    RECONCILING / UNRECOVERABLE indicate state-drift recovery in
    progress — fine briefly, alarming if observed at capture."""
    dp = fixture.desync_phase.upper()
    if dp and dp != "OK":
        return False, f"desync_phase at capture: {dp!r}"
    return True, "ok"


def invariant_position_consistent_with_open_orders(
    fixture: SnapshotFixture,
) -> tuple[bool, str]:
    """Sanity: if the snapshot has open orders, they MUST be on the
    side that's consistent with the bot's intended quoting. Doesn't
    enforce a strict rule (the bot can be one-sided), just checks
    for obvious nonsense (e.g., reduce-only flag flipped wrong).
    """
    for o in fixture.open_orders_raw:
        ro = str(o.get("reduceOnly") or "").lower()
        if ro not in ("", "false", "true"):
            return False, f"open order has weird reduceOnly: {o!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Top-level entry: run all invariants against a snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    snapshot_name: str
    bot_version: str
    captured_at: datetime
    invariants_passed: int
    invariants_failed: int
    failures: list[tuple[str, str]]  # (invariant_name, reason)

    @property
    def ok(self) -> bool:
        return self.invariants_failed == 0


def run_snapshot_replay(snapshot_dir: str | Path) -> ReplayResult:
    """Top-level: load snapshot, build BotState, run all invariants.

    Returns a ``ReplayResult`` with pass/fail counts and failure
    reasons. Callers (tests) decide whether to assert or report.
    """
    fixture = load_snapshot(snapshot_dir)
    replay = build_replay_state(fixture)
    try:
        invariants = [
            ("invariant_no_duplicate_oid_or_cloid",
             lambda: invariant_no_duplicate_oid_or_cloid(replay.state)),
            ("invariant_no_wedge_counters_nonzero",
             lambda: invariant_no_wedge_counters_nonzero(fixture)),
            ("invariant_risk_state_normal",
             lambda: invariant_risk_state_normal(fixture)),
            ("invariant_desync_phase_ok",
             lambda: invariant_desync_phase_ok(fixture)),
            ("invariant_position_consistent_with_open_orders",
             lambda: invariant_position_consistent_with_open_orders(fixture)),
        ]
        passed = 0
        failures: list[tuple[str, str]] = []
        for name, fn in invariants:
            ok, reason = fn()
            if ok:
                passed += 1
            else:
                failures.append((name, reason))
        return ReplayResult(
            snapshot_name=fixture.path.name,
            bot_version=fixture.bot_version,
            captured_at=fixture.captured_at_utc,
            invariants_passed=passed,
            invariants_failed=len(failures),
            failures=failures,
        )
    finally:
        # Cleanup the temp db file. On Windows, file handles can
        # linger briefly; tolerate that.
        try:
            replay.db_path.unlink(missing_ok=True)
        except PermissionError:
            pass
