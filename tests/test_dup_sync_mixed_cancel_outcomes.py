"""
Integration test: duplicate same-side remote orders where the cancel dispatch
for one extra succeeds and another fails. This exercises the end-to-end chain:

    `_sync_open_orders_impl` dup branch
      → calls `_cancel_orphan_remote_order` once per extra
      → tracks `all_extra_cancels_ok[side]`
      → passes `requires_confirm=(not all_extra_cancels_ok[side])` to
        `_set_side_unresolved`
      → `_side_unresolved_requires_confirm[side]` latches True
      → `_is_side_unresolved(side)` refuses failsafe auto-release

If any link in this chain regresses, the bot could resume quoting over a
still-live phantom duplicate — the exact class of bug the robustness polish
pass was designed to prevent.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.enums import Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _ok_cancel() -> dict:
    return {
        "status": "ok",
        "response": {"type": "cancel", "data": {"statuses": ["success"]}},
    }


def _err_cancel() -> dict:
    # Non-benign error text; `interpret_hl_cancel_response` → kind="error" → helper returns False.
    return {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": [{"error": "internal server error"}]},
        },
    }


def _setup() -> tuple[UnitTestSettings, Path, OrderManager, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_mix_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_order_by_cloid.return_value = _ok_cancel()
    om = OrderManager(s, client, storage, state)
    return s, path, om, storage


def test_mixed_cancel_outcomes_latches_requires_confirm_and_blocks_auto_release() -> None:
    """
    3 BUY remote duplicates; `cancel_order` returns OK for the first extra
    and a non-benign error for the second. Expected:

      - both extras are dispatched (no dedup for fresh oids)
      - keeper (newest ts) survives — NOT canceled
      - `all_extra_cancels_ok[Side.BUY]` becomes False because one failed
      - `_set_side_unresolved(Side.BUY, ..., requires_confirm=True)` fires
      - `_side_unresolved_requires_confirm[Side.BUY]` is True
      - after the failsafe timeout elapses with an empty local slot,
        `_is_side_unresolved(Side.BUY)` still returns True (no auto-release)
      - `_side_unresolved_confirm_block_count` increments on that blocked check
    """
    s, path, om, storage = _setup()
    sym = s.symbol

    # 3 BUY remotes; keeper is newest ts=300 (oid=103); extras are 101 and 102.
    remotes = [
        HLOpenOrderRaw(
            oid=101, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=100
        ),
        HLOpenOrderRaw(
            oid=102, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=200
        ),
        HLOpenOrderRaw(
            oid=103, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=300
        ),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes
    # Two extras → two cancel_order calls. First OK, second non-benign error.
    om._client.cancel_order.side_effect = [_ok_cancel(), _err_cancel()]

    om.sync_open_orders(force=True, emergency=True)

    # Both extras were dispatched. Keeper (103) must NOT have been canceled.
    canceled_oids = {int(c.args[1]) for c in om._client.cancel_order.call_args_list}
    assert canceled_oids == {101, 102}, (
        f"both extras (not keeper 103) must be dispatched; got {canceled_oids}"
    )
    assert 103 not in canceled_oids, "keeper must not be canceled"

    # Any-failure on an extra → latch True for that side.
    assert om._side_unresolved_active[Side.BUY] is True, (
        "dup detection must enter side-unresolved"
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True, (
        "one extra's cancel failed → requires_confirm latch must be True"
    )
    # The opposite side was clean — no latch or unresolved state there.
    assert om._side_unresolved_active[Side.SELL] is False
    assert om._side_unresolved_requires_confirm[Side.SELL] is False

    # Failsafe must NOT auto-release the latched side past timeout.
    om._state.working_bid = None
    timeout_s = float(s.side_unresolved_suppression_timeout_seconds)
    om._side_unresolved_since_mono[Side.BUY] = time.monotonic() - timeout_s - 1.0
    blocks_before = om._side_unresolved_confirm_block_count

    still_unresolved = om._is_side_unresolved(Side.BUY)

    assert still_unresolved is True, (
        "latch must block failsafe auto-release over still-live phantom"
    )
    assert om._side_unresolved_active[Side.BUY] is True, (
        "state must NOT be cleared by the failsafe while latch is set"
    )
    assert om._side_unresolved_confirm_block_count == blocks_before + 1, (
        "block counter must increment on the blocked check"
    )
    path.unlink(missing_ok=True)


def test_all_cancels_succeed_does_not_latch_requires_confirm() -> None:
    """
    Contrast case: if every extra's cancel succeeds, the side enters
    unresolved but `requires_confirm` stays False — the failsafe timeout
    is allowed to auto-release normally (preserving liveness). This pins
    the asymmetry: only failures trigger the latch.
    """
    s, path, om, _ = _setup()
    sym = s.symbol
    remotes = [
        HLOpenOrderRaw(
            oid=201, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=100
        ),
        HLOpenOrderRaw(
            oid=202, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=200
        ),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes
    om._client.cancel_order.return_value = _ok_cancel()

    om.sync_open_orders(force=True, emergency=True)

    # Loser 201 canceled, keeper 202 kept.
    canceled_oids = {int(c.args[1]) for c in om._client.cancel_order.call_args_list}
    assert canceled_oids == {201}, f"only non-keeper extras canceled; got {canceled_oids}"

    # Side is unresolved (dup was detected) but `requires_confirm` is False.
    assert om._side_unresolved_active[Side.BUY] is True
    assert om._side_unresolved_requires_confirm[Side.BUY] is False, (
        "all-extras-succeed path must NOT latch requires_confirm"
    )

    # Empty slot + elapsed timeout → failsafe allowed to release.
    om._state.working_bid = None
    timeout_s = float(s.side_unresolved_suppression_timeout_seconds)
    om._side_unresolved_since_mono[Side.BUY] = time.monotonic() - timeout_s - 1.0
    assert om._is_side_unresolved(Side.BUY) is False, (
        "without the latch, failsafe must auto-release (liveness preserved)"
    )
    path.unlink(missing_ok=True)
