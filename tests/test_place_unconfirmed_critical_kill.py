"""BUG-024 regression tests — hard-kill on unconfirmed place response.

Per the live production data (2026-05-14): 294, 367, 805 "phantom"
order rows were created per day across 2026-05-12/13/14 on the SUI
session — every one with ``ts_ack=NULL`` and ``cancel_reason=
gone_on_exchange``. The shape proves these orders went through the
place flow with ``ts_sent`` stamped but no ack was ever recorded.
The bot had NO way of knowing whether the order landed on the
venue (and could silently fill) or was rejected entirely.

Operator decree (verbatim 2026-05-14): "Missing order ack/reject —
is NOT BENIGN!!! IT IS A CRITICAL CONNECTIVITY FAILURE! Once a
month is critical!!! If this happens, the bot must stop, flatten,
and the dashboard must display a huge error! Telegram must also
receive an error!"

The fix forces ``Bot.kill("place_response_unconfirmed")`` on the
FIRST occurrence, via the new ``request_kill_fn`` callback wired
from Bot through OrderManager.

Tests below assert:

1. ``request_kill_fn`` is called with reason="place_response_unconfirmed"
   and a payload carrying enough forensic context to diagnose.
2. The CRITICAL bot_event row is persisted.
3. The counter ``state.place_unconfirmed_critical_total`` is bumped.
4. The order is transitioned to REJECTED (not left in SENT — it
   would orphan if we did).
5. Clean rejections (``exchange_rejected`` for post-only crossing)
   do NOT trigger the kill — the existing benign path still runs.
6. The fallback "no callback wired" branch sets ``state.killed=True``
   directly (so test contexts still see the halt signal).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from app.enums import OrderStatus, Side
from app.exchange.okx_responses import interpret_okx_place_response
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup(*, with_kill_callback: bool = True) -> tuple[
    UnitTestSettings,
    Path,
    BotState,
    OrderManager,
    Storage,
    list[tuple[str, dict | None]],
]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_unconfirmed_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
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
    # Use the OKX response parser explicitly. The default MagicMock
    # would auto-generate ``interpret_place_response`` as a callable
    # MagicMock that returns a MagicMock (not a tuple) — execution.py
    # then falls back to the HL parser, which misclassifies our OKX-
    # shaped responses as ``transport_rejected``. Wiring the real OKX
    # parser keeps the test exercising the actual production
    # classification path.
    client.interpret_place_response.side_effect = interpret_okx_place_response

    # Capture every (reason, payload) pair the production
    # request_kill_fn callback would receive.
    kill_calls: list[tuple[str, dict | None]] = []

    def fake_kill(reason: str, payload: dict | None = None) -> None:
        kill_calls.append((reason, payload))

    om = OrderManager(
        s,
        client,
        storage,
        state,
        request_kill_fn=fake_kill if with_kill_callback else None,
    )
    return s, path, state, om, storage, kill_calls


def _stage_sent_ask(om: OrderManager) -> object:
    """Stage a local SELL in SENT status (typical state when the
    place response is about to be processed)."""
    wo = om._stage_place_order_local(
        Side.SELL, price=3100.0, size=0.01, quote_cycle_id="kill-test"
    )
    assert wo is not None
    return wo


def _payload_dict(ev: dict) -> dict:
    raw = ev.get("payload_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


# -------------------------------------------------- unconfirmed → kill
def test_unconfirmed_place_response_triggers_kill() -> None:
    """``outcome == unconfirmed`` MUST call request_kill_fn with the
    fixed reason 'place_response_unconfirmed' and a forensic payload."""
    _s, path, state, om, storage, kill_calls = _setup()
    try:
        wo = _stage_sent_ask(om)

        # Simulate an OKX-style response shape that the parser will
        # classify as ``unconfirmed`` — top code "0" but ``data: []``
        # (no row → "missing_row_in_data" branch).
        client = om._client
        client.place_post_only_limit.return_value = {  # type: ignore[attr-defined]
            "code": "0",
            "msg": "",
            "data": [],
        }

        intent_seq = wo.transport_intent_seq  # type: ignore[attr-defined]
        om._complete_place_http(
            wo,
            quote_cycle_id="kill-test",
            intent_seq=intent_seq,
        )

        # 1. Kill was called exactly once with the correct reason.
        assert len(kill_calls) == 1
        reason, payload = kill_calls[0]
        assert reason == "place_response_unconfirmed"
        assert payload is not None
        # 2. Payload carries forensic context (side, symbol, price,
        # size, cloid, raw response).
        assert payload["side"] == "SELL"
        assert payload["symbol"] == _s.symbol
        assert payload["outcome"] == "unconfirmed"
        assert "raw_response_truncated" in payload
        assert "missing_row_in_data" in payload["reason"]

        # 3. Counter incremented.
        assert state.place_unconfirmed_critical_total == 1

        # 4. Order transitioned to REJECTED (no longer in SENT).
        assert wo.status == OrderStatus.REJECTED  # type: ignore[attr-defined]

        # 5. CRITICAL bot_event row was persisted.
        evs = storage.recent_bot_events(50)
        unconf = [
            e for e in evs if e.get("event_type") == "place_response_unconfirmed"
        ]
        assert len(unconf) == 1
        assert unconf[0]["severity"] == "CRITICAL"
        ev_payload = _payload_dict(unconf[0])
        assert ev_payload["outcome"] == "unconfirmed"
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------- benign rejection: NO kill
def test_clean_post_only_rejection_does_not_trigger_kill() -> None:
    """Existing benign path: a clean ``exchange_rejected`` for a
    post-only crossing MUST NOT trigger the BUG-024 kill. Only
    ``unconfirmed`` does."""
    _s, path, state, om, _storage, kill_calls = _setup()
    try:
        wo = _stage_sent_ask(om)

        client = om._client
        # OKX 51604 = post-only would immediately match.
        client.place_post_only_limit.return_value = {  # type: ignore[attr-defined]
            "code": "0",
            "msg": "",
            "data": [
                {
                    "ordId": "",
                    "clOrdId": wo.client_order_id,  # type: ignore[attr-defined]
                    "sCode": "51604",
                    "sMsg": "post only would match immediately",
                    "tag": "",
                }
            ],
        }

        intent_seq = wo.transport_intent_seq  # type: ignore[attr-defined]
        om._complete_place_http(
            wo,
            quote_cycle_id="kill-test",
            intent_seq=intent_seq,
        )

        # Kill NOT called — this is the benign path.
        assert kill_calls == []
        # Counter NOT incremented.
        assert state.place_unconfirmed_critical_total == 0
        # Order DID transition to REJECTED (existing benign behaviour).
        assert wo.status == OrderStatus.REJECTED  # type: ignore[attr-defined]
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------- transport_rejected: NO kill
def test_auth_failure_does_not_trigger_unconfirmed_kill() -> None:
    """Auth failure → ``transport_rejected``. The bot still rejects
    the order locally but does NOT fire the BUG-024 kill (a
    different kill path may handle auth failures separately)."""
    _s, path, state, om, _storage, kill_calls = _setup()
    try:
        wo = _stage_sent_ask(om)

        client = om._client
        client.place_post_only_limit.return_value = {  # type: ignore[attr-defined]
            "code": "50113",  # auth-failure top code
            "msg": "Invalid sign",
            "data": [],
        }

        intent_seq = wo.transport_intent_seq  # type: ignore[attr-defined]
        om._complete_place_http(
            wo,
            quote_cycle_id="kill-test",
            intent_seq=intent_seq,
        )

        # BUG-024 kill not fired.
        assert kill_calls == []
        assert state.place_unconfirmed_critical_total == 0
        # Order transitioned to REJECTED via the existing path.
        assert wo.status == OrderStatus.REJECTED  # type: ignore[attr-defined]
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------- no callback wired: set killed flag
def test_unconfirmed_without_callback_sets_state_killed() -> None:
    """If the OrderManager isn't wired to a Bot (test / legacy
    context), the unconfirmed path MUST still mark
    ``state.killed=True`` so the next tick of any quote loop halts.
    This is the last-line-of-defence path."""
    _s, path, state, om, _storage, kill_calls = _setup(
        with_kill_callback=False
    )
    try:
        wo = _stage_sent_ask(om)

        client = om._client
        client.place_post_only_limit.return_value = {  # type: ignore[attr-defined]
            "code": "0",
            "msg": "",
            "data": [],
        }

        intent_seq = wo.transport_intent_seq  # type: ignore[attr-defined]
        om._complete_place_http(
            wo,
            quote_cycle_id="kill-test",
            intent_seq=intent_seq,
        )

        # Callback wasn't wired — but state.killed was still set.
        assert kill_calls == []
        assert state.killed is True
        assert state.kill_reason == "place_response_unconfirmed"
        # Counter still bumped.
        assert state.place_unconfirmed_critical_total == 1
        # Order still transitioned to REJECTED.
        assert wo.status == OrderStatus.REJECTED  # type: ignore[attr-defined]
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------- counter default
def test_place_unconfirmed_critical_total_initialises_to_zero() -> None:
    """BotState must expose the counter so /state/current can render
    it on the dashboard. Default 0 on a fresh session."""
    _s, path, state, _om, _storage, _kills = _setup()
    try:
        assert hasattr(state, "place_unconfirmed_critical_total")
        assert state.place_unconfirmed_critical_total == 0
    finally:
        path.unlink(missing_ok=True)
