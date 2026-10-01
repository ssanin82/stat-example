"""Tests for the inbound Telegram command poller.

Coverage:
* disabled when token or allowed_user_ids is empty
* parses chat/user IDs and dispatches to handlers
* unauthorized user_id -> silent rejection
* unauthorized chat_id (when allowlist is set) -> silent rejection
* per-user rate limit
* read-only commands (/status, /stats, /help, /version, /position, /equity, /fills, /health)
* /pause and /resume mutate state.set_manual_pause and write events
* destructive commands (/kill, /flatten, /restart) require two-step confirmation
* expired confirmation slot
* /restart calls exit_fn with code 43
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.state import BotState
from app.storage import Storage
from app.telegram_commands import (
    RESTART_EXIT_CODE,
    TelegramCommandPoller,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "EXCHANGE": "bluefin",
        "SYMBOL": "SUI-PERP",
        "BLUEFIN_PRIVATE_KEY": "00" * 32,
        "BLUEFIN_ACCOUNT_ADDRESS": "0x" + "ab" * 32,
        "TELEGRAM_BOT_TOKEN": "TESTTOKEN",
        "TELEGRAM_OPS_CHAT_ID": "-100111",
        "TELEGRAM_TRADES_CHAT_ID": "-100222",
        "TELEGRAM_ALLOWED_USER_IDS": "1001",
        "TELEGRAM_LONG_POLL_TIMEOUT_SECONDS": 1,
        "TELEGRAM_COMMAND_MIN_INTERVAL_SECONDS": 0.0,
        "TELEGRAM_CONFIRM_TIMEOUT_SECONDS": 5.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _storage(tmp_path) -> Storage:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}")
    st = Storage(s)
    st.init_schema()
    return st


def _make_poller(
    settings, *, bot=None, state=None, storage=None, client=None
) -> tuple[TelegramCommandPoller, list[tuple[str, dict]], list[int]]:
    """Returns (poller, recorded_posts, recorded_exit_codes)."""
    bot = bot or MagicMock()
    state = state or BotState(settings)
    storage = storage or MagicMock(recent_fills=lambda limit=100: [])

    posts: list[tuple[str, dict]] = []

    def http_post(url: str, body: dict[str, Any]) -> tuple[int, str]:
        posts.append((url, dict(body)))
        return 200, '{"ok":true}'

    def http_get(url: str, params: dict, timeout: float) -> tuple[int, str]:
        return 200, '{"ok":true,"result":[]}'

    exits: list[int] = []
    p = TelegramCommandPoller(
        settings,
        bot=bot,
        state=state,
        storage=storage,
        notifier=None,
        client=client,
        http_post=http_post,
        http_get=http_get,
        exit_fn=lambda code: exits.append(code),
    )
    return p, posts, exits


def _msg(*, user_id: int = 1001, chat_id: int | None = None, text: str = "/help") -> dict:
    if chat_id is None:
        chat_id = user_id  # DM
    return {
        "update_id": 42,
        "message": {
            "from": {"id": user_id},
            "chat": {"id": chat_id},
            "text": text,
        },
    }


# --- enablement ---


def test_disabled_when_no_token() -> None:
    s = _settings(TELEGRAM_BOT_TOKEN="")
    p, posts, _ = _make_poller(s)
    assert p.enabled is False
    p.start()
    p.stop()


def test_disabled_when_no_allowed_users() -> None:
    s = _settings(TELEGRAM_ALLOWED_USER_IDS="")
    p, posts, _ = _make_poller(s)
    assert p.enabled is False


# --- auth ---


def test_unauthorized_user_silently_rejected() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    # User 99 is NOT in the allowlist.
    p._handle_update(_msg(user_id=99, text="/status"))
    # Silently rejected: no reply.
    assert posts == []


def test_unauthorized_chat_silently_rejected_when_allowlist_set() -> None:
    s = _settings(TELEGRAM_ALLOWED_CHAT_IDS="-100333")
    p, posts, _ = _make_poller(s)
    # Allowed user, but chat is the user's DM (1001), not -100333.
    p._handle_update(_msg(user_id=1001, chat_id=1001, text="/help"))
    assert posts == []


def test_authorized_user_in_dm() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(user_id=1001, text="/help"))
    assert len(posts) == 1
    assert "Commands" in posts[0][1]["text"]


# --- rate limit ---


def test_rate_limit_replies_when_too_fast() -> None:
    s = _settings(TELEGRAM_COMMAND_MIN_INTERVAL_SECONDS=10.0)
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/help"))
    p._handle_update(_msg(text="/help"))
    # First reply: full /help. Second: rate-limited message.
    assert len(posts) == 2
    assert "rate-limited" in posts[1][1]["text"].lower()


# --- read-only command surface ---


def test_help_lists_commands() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/help"))
    text = posts[-1][1]["text"]
    for cmd in ["/status", "/stats", "/fills", "/pause", "/kill", "/flatten", "/restart"]:
        assert cmd in text


def test_status_replies_with_state_snapshot() -> None:
    s = _settings()
    state = BotState(s)
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "status" in text.lower()
    assert "session" in text.lower()


def test_status_includes_regime_mode_line_v1_5_26() -> None:
    """v1.5.26 Phase 2A.3: /status renders a ``regime: <mode> · <dwell>
    [· <reason>]`` line so the operator can see at a glance which mode
    the FSM is in without a separate dashboard tab. Driven by
    ``state.regime_controller`` (mode, mode_since_mono, last_transition_reason).

    Default-constructed BotState's regime controller is NORMAL, dwell
    0s, no transition reason yet -- so the line is "regime: NORMAL · 0s".
    """
    s = _settings()
    state = BotState(s)
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "regime" in text.lower(), \
        "expected a regime: line in /status output"
    # The default mode at construction is NORMAL.
    assert "NORMAL" in text, \
        "expected default NORMAL mode in /status regime line"


def test_status_regime_line_includes_transition_reason_when_set() -> None:
    """When the controller has a ``last_transition_reason``, /status
    includes it in the regime line for at-a-glance context (e.g.
    ``regime: CAUTIOUS · 2m 14s · forward_classifier:vol_slope``)."""
    from app.regime_controller import Mode
    s = _settings()
    state = BotState(s)
    # Force the controller into CAUTIOUS with a synthetic transition
    # reason. Tests only the rendering path; the FSM's actual
    # transition machinery is covered by test_regime_controller.py.
    state.regime_controller.mode = Mode.CAUTIOUS
    state.regime_controller.last_transition_reason = "forward_classifier:vol_slope"
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "CAUTIOUS" in text
    assert "forward_classifier:vol_slope" in text


def test_status_includes_runtime_new_orders_and_actions() -> None:
    """The new /status shape includes a hh:mm:ss runtime line, a
    new_orders count (placements only — bumped per
    place_post_only_limit), an actions total (placements + cancels),
    and is wrapped in a Markdown code block for monospaced rendering
    in Telegram."""
    s = _settings()
    state = BotState(s)
    # Place attempts: 137. Combined-action counters: 12 + 200 = 212
    # (which exceeds new_orders by design — actions includes cancels).
    state.session_place_attempt_count = 137
    state.outbound_ws_action_send_count = 12
    state.outbound_http_action_send_count = 200
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    body = posts[-1][1]
    text = body["text"]
    assert body.get("parse_mode") == "Markdown"
    assert text.startswith("```") and text.rstrip().endswith("```")
    assert "runtime" in text.lower()
    import re
    assert re.search(r"\d{2}:\d{2}:\d{2}", text), \
        "expected hh:mm:ss runtime in /status output"
    # Both counters are present and labelled distinctly.
    assert "new_orders" in text
    assert "137" in text  # placements only
    assert "actions" in text
    assert "212" in text  # combined ws + http
    assert "place+cancel" in text


def test_status_amend_line_hidden_when_no_traffic() -> None:
    """amend-prio Phase 4 (v1.4.17): when the knob is off / no amends
    have fired, the ``amends:`` row must NOT appear in /status. We
    don't want operators to see a misleading ``0 ok / 0 below-filled``
    row in profiles that don't use amend."""
    s = _settings()
    state = BotState(s)
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "amends" not in text.lower()


def test_status_amend_line_visible_when_traffic_exists() -> None:
    """amend-prio Phase 4 (v1.4.17): once amends have fired, /status
    shows a compact summary line breaking down outcomes."""
    s = _settings()
    state = BotState(s)
    state.amend_intents_emitted_total = 124
    state.amend_success_total = 120
    state.amend_below_filled_total = 3
    state.amend_post_only_cross_total = 1
    state.amend_pending_high_watermark = 2
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "amends" in text
    assert "120 ok" in text
    assert "3 below-filled" in text
    assert "1 cross-rejected" in text
    assert "emitted 124" in text
    assert "hwm 2" in text


def test_stats_includes_runtime() -> None:
    s = _settings()
    state = BotState(s)
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/stats"))
    text = posts[-1][1]["text"]
    assert "runtime" in text.lower()
    import re
    assert re.search(r"\d{2}:\d{2}:\d{2}", text)


def test_version_replies() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/version"))
    text = posts[-1][1]["text"]
    assert "version" in text.lower()


def test_unknown_command() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/wat"))
    text = posts[-1][1]["text"]
    assert "unknown" in text.lower()


# --- /pause /resume ---


def test_pause_sets_manual_pause_and_writes_event() -> None:
    s = _settings()
    state = BotState(s)
    storage = MagicMock(recent_fills=lambda limit=100: [])
    p, posts, _ = _make_poller(s, state=state, storage=storage)
    p._handle_update(_msg(text="/pause"))
    assert state.status_flags_dict()["manual_pause"] is True
    storage.insert_bot_event.assert_called_once()
    assert "paused" in posts[-1][1]["text"].lower()


def test_resume_clears_manual_pause() -> None:
    s = _settings()
    state = BotState(s)
    state.set_manual_pause(True)
    storage = MagicMock(recent_fills=lambda limit=100: [])
    p, posts, _ = _make_poller(s, state=state, storage=storage)
    p._handle_update(_msg(text="/resume"))
    assert state.status_flags_dict()["manual_pause"] is False
    assert "resumed" in posts[-1][1]["text"].lower()


def test_resume_blocked_when_killed() -> None:
    s = _settings()
    state = BotState(s)
    state.killed = True
    storage = MagicMock(recent_fills=lambda limit=100: [])
    p, posts, _ = _make_poller(s, state=state, storage=storage)
    p._handle_update(_msg(text="/resume"))
    text = posts[-1][1]["text"].lower()
    assert "killed" in text


# --- two-step confirmation ---


def test_kill_requires_confirmation_first_request_arms_slot() -> None:
    s = _settings()
    bot = MagicMock()
    p, posts, _ = _make_poller(s, bot=bot)
    p._handle_update(_msg(text="/kill"))
    # First request → arm slot, no kill yet.
    bot.kill.assert_not_called()
    text = posts[-1][1]["text"]
    assert "confirm" in text.lower()


def test_kill_confirm_executes() -> None:
    s = _settings()
    bot = MagicMock()
    p, posts, _ = _make_poller(s, bot=bot)
    p._handle_update(_msg(text="/kill"))
    p._handle_update(_msg(text="/kill confirm"))
    bot.kill.assert_called_once_with(reason="manual_kill_via_telegram")
    final_text = posts[-1][1]["text"].lower()
    assert "killed" in final_text


def test_kill_confirm_without_prior_request_is_rejected() -> None:
    s = _settings()
    bot = MagicMock()
    p, posts, _ = _make_poller(s, bot=bot)
    p._handle_update(_msg(text="/kill confirm"))
    bot.kill.assert_not_called()
    assert "no pending" in posts[-1][1]["text"].lower()


def test_kill_confirm_after_timeout_is_rejected() -> None:
    s = _settings(TELEGRAM_CONFIRM_TIMEOUT_SECONDS=0.001)
    bot = MagicMock()
    p, posts, _ = _make_poller(s, bot=bot)
    p._handle_update(_msg(text="/kill"))
    import time as _time
    _time.sleep(0.05)
    p._handle_update(_msg(text="/kill confirm"))
    bot.kill.assert_not_called()
    assert "expired" in posts[-1][1]["text"].lower() or "no pending" in posts[-1][1]["text"].lower()


def test_flatten_requires_confirmation_then_calls_bot_flatten() -> None:
    s = _settings()
    bot = MagicMock()
    fake_result = MagicMock()
    fake_result.value = "FLATTENED_OK"
    bot.flatten.return_value = fake_result
    p, posts, _ = _make_poller(s, bot=bot)
    p._handle_update(_msg(text="/flatten"))
    bot.flatten.assert_not_called()
    p._handle_update(_msg(text="/flatten confirm"))
    bot.flatten.assert_called_once_with(blocking=True)
    assert "flatten complete" in posts[-1][1]["text"].lower()


def test_restart_requires_confirmation_then_calls_exit_fn_43() -> None:
    s = _settings()
    p, posts, exits = _make_poller(s)
    p._handle_update(_msg(text="/restart"))
    assert exits == []
    p._handle_update(_msg(text="/restart confirm"))
    assert exits == [RESTART_EXIT_CODE]


# --- regression: advancing the offset on rejected updates ---


def test_offset_advances_even_on_unauthorized() -> None:
    s = _settings()
    p, posts, _ = _make_poller(s)
    # update_id 50 from unauthorized user
    p._handle_update({
        "update_id": 50,
        "message": {
            "from": {"id": 99},
            "chat": {"id": 99},
            "text": "/status",
        },
    })
    assert p._offset == 51, "offset must advance for rejected msgs to prevent replay"


# --- /fills ---


def test_fills_returns_recent_rows() -> None:
    s = _settings()
    storage = MagicMock()
    storage.recent_fills.return_value = [
        {"ts_fill": "2026-04-25T01:00:00Z", "side": "BUY", "size": 10, "price": 2.10, "fee": 0.0011, "markout_5s_bps": -1.4},
        {"ts_fill": "2026-04-25T01:00:01Z", "side": "SELL", "size": 10, "price": 2.11, "fee": 0.0011, "markout_5s_bps": 0.2},
    ]
    p, posts, _ = _make_poller(s, storage=storage)
    p._handle_update(_msg(text="/fills 2"))
    text = posts[-1][1]["text"]
    assert "BUY" in text and "SELL" in text
    storage.recent_fills.assert_called_with(limit=2)


def test_fills_default_limit_is_10() -> None:
    s = _settings()
    storage = MagicMock()
    storage.recent_fills.return_value = []
    p, posts, _ = _make_poller(s, storage=storage)
    p._handle_update(_msg(text="/fills"))
    storage.recent_fills.assert_called_with(limit=10)


# --- /status market-regime block --------------------------------------------


def test_status_includes_market_regime_block() -> None:
    """`/status` should surface enough state for an operator to reason about
    the current market regime without pulling klines: short-window vol,
    venue spread, basis to Binance, last quote sides, book staleness, and
    position-cap headroom.
    """
    s = _settings()
    state = BotState(s)
    # Make the market-regime fields populated.
    from app.models import BestBidAsk, PositionSnapshot
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=100.0,
        best_ask=100.05,
        mid_price=100.025,
        spread_bps=5.0,
    )
    state.vol_bps = 6.5
    state.binance_basis_ewma = -0.0001  # -1.0 bps
    state.last_active_sides = "ASK_ONLY"
    state.book_age_seconds = 0.250
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=10.0,
        avg_entry_price=100.0,
        mark_price=100.0,
        position_notional=1000.0,
        unrealized_pnl_usd=0.0,
    )
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    # Market section header
    assert "--- market ---" in text
    # Each metric is present
    assert "vol" in text and "6.50 bps" in text
    assert "spread" in text and "5.00 bps" in text
    assert "basis" in text and "-1.00 bps" in text  # binance_basis * 10000
    assert "active_sides" in text and "ASK_ONLY" in text
    assert "book_age" in text and "250" in text
    assert "pos_cap" in text


def test_status_market_regime_handles_missing_market_data() -> None:
    """Pre-warmup state (no market, no vol) renders ``"?"`` placeholders
    instead of crashing.
    """
    s = _settings()
    state = BotState(s)
    # No market set (state.market is None).
    state.vol_bps = None
    state.binance_basis_ewma = None
    state.last_active_sides = None
    state.book_age_seconds = None
    p, posts, _ = _make_poller(s, state=state)
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "--- market ---" in text
    # Each metric falls back to "?" when its source is None.
    assert "vol" in text


def test_fmt_vol_with_threshold_regime_labels() -> None:
    """The regime hint inside the vol label exercises four bands."""
    from app.telegram_commands import _fmt_vol_with_threshold

    assert "quiet" in _fmt_vol_with_threshold(1.0, threshold_bps=4.0)
    assert "normal" in _fmt_vol_with_threshold(4.0, threshold_bps=4.0)
    assert "active" in _fmt_vol_with_threshold(8.0, threshold_bps=4.0)
    assert "volatile" in _fmt_vol_with_threshold(20.0, threshold_bps=4.0)
    assert _fmt_vol_with_threshold(None, threshold_bps=4.0) == "?"


# --- /config command --------------------------------------------------------


def test_config_lists_settings_and_redacts_secrets() -> None:
    """`/config` returns a code-block dump with public addresses visible
    and private keys / API secrets redacted with character counts.
    """
    s = _settings(
        BLUEFIN_PRIVATE_KEY="0x" + "a" * 64,
        BLUEFIN_ACCOUNT_ADDRESS="0xabc123",
    )
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config"))
    assert posts, "expected at least one reply"
    full = "\n".join(post[1]["text"] for post in posts)
    # Public address visible.
    assert "BLUEFIN_ACCOUNT_ADDRESS" in full
    assert "0xabc123" in full
    # Private key NEVER visible — only the redaction marker.
    assert "BLUEFIN_PRIVATE_KEY" in full
    assert "a" * 64 not in full
    # HTML-escaped in the wire payload: ``<redacted N chars>`` →
    # ``&lt;redacted N chars&gt;``. Telegram's HTML parser un-escapes it
    # for display; the operator sees the original. Asserting on the bare
    # word "redacted" sidesteps escape-form coupling.
    assert "redacted" in full


def test_config_filter_substring_match() -> None:
    """Optional positional filter narrows the dump to matching field names."""
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config max_position"))
    full = "\n".join(post[1]["text"] for post in posts)
    assert "MAX_POSITION_NOTIONAL_USD" in full
    # Other unrelated fields not present in this filtered output.
    assert "REPRICE_THRESHOLD_BPS" not in full


def test_config_redacts_telegram_token() -> None:
    """Telegram bot token is also a secret — must be redacted."""
    s = _settings(TELEGRAM_BOT_TOKEN="1234567890:abcdefghij")
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config telegram_bot_token"))
    full = "\n".join(post[1]["text"] for post in posts)
    assert "TELEGRAM_BOT_TOKEN" in full
    assert "1234567890:abcdefghij" not in full
    # HTML-escaped in the wire payload: ``<redacted N chars>`` →
    # ``&lt;redacted N chars&gt;``. Telegram's HTML parser un-escapes it
    # for display; the operator sees the original. Asserting on the bare
    # word "redacted" sidesteps escape-form coupling.
    assert "redacted" in full


def test_config_filter_no_match_replies_with_message() -> None:
    """A filter that matches nothing should reply with a clear message,
    not an empty code block."""
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config zzznotafield"))
    text = posts[-1][1]["text"]
    assert "no settings match" in text


def test_config_redaction_helper_unset_and_empty() -> None:
    """Direct unit test for the redaction helper's edge cases."""
    from app.telegram_commands import TelegramCommandPoller

    assert TelegramCommandPoller._redact_config_value(None) == "<unset>"
    assert TelegramCommandPoller._redact_config_value("") == "<empty>"
    assert "<redacted 5 chars>" == TelegramCommandPoller._redact_config_value("hello")


def test_config_secret_field_classifier_substring() -> None:
    """The secret-field classifier matches on substring fragments."""
    from app.telegram_commands import TelegramCommandPoller as P

    assert P._is_config_secret_field("hl_secret_key") is True
    assert P._is_config_secret_field("bluefin_private_key") is True
    assert P._is_config_secret_field("grvt_api_secret") is True
    assert P._is_config_secret_field("grvt_api_key") is True
    assert P._is_config_secret_field("telegram_bot_token") is True
    # Public addresses and URLs are NOT secrets.
    assert P._is_config_secret_field("hl_account_address") is False
    assert P._is_config_secret_field("bluefin_account_address") is False
    assert P._is_config_secret_field("bluefin_rest_url") is False
    assert P._is_config_secret_field("max_position_notional_usd") is False


# --- /status volume cache (7d / 30d) ---------------------------------------


def test_status_omits_volume_block_when_no_client() -> None:
    """Without a client wired in, the volume cache is empty and the
    ``--- volume ---`` block must not appear (don't show a misleading $0).
    """
    s = _settings()
    p, posts, _ = _make_poller(s)  # no client passed
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "--- volume ---" not in text


def test_status_renders_volume_block_when_cache_populated() -> None:
    """When the background refresher has populated the cache, /status
    surfaces 7d and 30d volume rows with a fresh-age indicator.
    """
    s = _settings()
    p, posts, _ = _make_poller(s)
    # Simulate the refresher having run.
    p._volume_7d_usd = 1234.56
    p._volume_30d_usd = 9876.54
    p._volume_refreshed_at_mono = p._clock()
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "--- volume ---" in text
    assert "vol_7d" in text and "$1,234.56" in text
    assert "vol_30d" in text and "$9,876.54" in text
    assert "vol_age" in text


def test_status_volume_block_handles_partial_cache() -> None:
    """If only one window's fetch succeeded, render that one and skip the
    other — never render an undefined value.
    """
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._volume_7d_usd = 500.0
    p._volume_30d_usd = None  # 30d fetch failed
    p._volume_refreshed_at_mono = p._clock()
    p._volume_last_error = "30d: rate limited"
    p._handle_update(_msg(text="/status"))
    text = posts[-1][1]["text"]
    assert "vol_7d" in text and "$500.00" in text
    assert "vol_30d" not in text
    assert "vol_err" in text


def test_volume_refresher_disabled_when_interval_zero() -> None:
    """``TELEGRAM_VOLUME_REFRESH_SECONDS=0`` disables the background thread."""
    s = _settings(TELEGRAM_VOLUME_REFRESH_SECONDS=0.0)
    fake_client = MagicMock()
    fake_client.fetch_account_volume_usd.return_value = {"volume_usd": 1.0}
    p, _, _ = _make_poller(s, client=fake_client)
    p.start()
    try:
        # Refresher thread should NOT have started.
        assert p._volume_thread is None
    finally:
        p.stop()


def test_volume_refresher_skipped_when_client_missing_method() -> None:
    """A client that doesn't expose ``fetch_account_volume_usd`` (non-Bluefin
    venues) doesn't cause the refresher to start or crash."""
    s = _settings()
    fake_client = MagicMock(spec=[])  # no fetch_account_volume_usd attr
    p, _, _ = _make_poller(s, client=fake_client)
    p.start()
    try:
        assert p._volume_thread is None
    finally:
        p.stop()


def test_refresh_volume_cache_once_populates_both_windows() -> None:
    """Direct unit test for the synchronous refresh helper. Both 7d and 30d
    windows go through; cache is populated; ``_volume_last_error`` is None.
    """
    s = _settings()
    fake_client = MagicMock()
    # Different return values per call so we can verify both windows fetched.
    fake_client.fetch_account_volume_usd.side_effect = [
        {"volume_usd": 700.0, "trade_count": 7},
        {"volume_usd": 3000.0, "trade_count": 30},
    ]
    p, _, _ = _make_poller(s, client=fake_client)
    p._refresh_volume_cache_once()
    assert p._volume_7d_usd == 700.0
    assert p._volume_30d_usd == 3000.0
    assert p._volume_last_error is None
    assert p._volume_refreshed_at_mono is not None
    # Two paginated calls — one per window.
    assert fake_client.fetch_account_volume_usd.call_count == 2


def test_refresh_volume_cache_once_keeps_last_good_on_partial_failure() -> None:
    """If only the 30d fetch fails this time, the 7d cache still updates
    AND the previous 30d value is preserved.
    """
    s = _settings()
    fake_client = MagicMock()
    fake_client.fetch_account_volume_usd.side_effect = [
        {"volume_usd": 700.0},
        Exception("rate limited"),
    ]
    p, _, _ = _make_poller(s, client=fake_client)
    # Pre-populate with a "previous good" value.
    p._volume_30d_usd = 2000.0
    p._refresh_volume_cache_once()
    assert p._volume_7d_usd == 700.0
    assert p._volume_30d_usd == 2000.0  # preserved
    assert p._volume_last_error is not None
    assert "rate limited" in p._volume_last_error


def test_config_chunks_long_dumps_into_multiple_messages() -> None:
    """The full Settings dump is bigger than Telegram's 4096-char cap;
    the command must split into multiple messages.
    """
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config"))
    # Full dump is ~150 fields → expect more than one message.
    assert len(posts) >= 2
    # Every message stays under Telegram's 4096-char cap with headroom for
    # the 3900-char ``_reply`` truncation guardrail (need to land fully
    # below it so no chunk gets the ``...[truncated]`` suffix appended).
    for post in posts:
        assert len(post[1]["text"]) < 3900, (
            f"chunk too large ({len(post[1]['text'])} chars) — "
            "_reply will truncate, secrets risk being cut mid-word"
        )


def test_config_uses_html_parse_mode_to_avoid_underscore_collisions() -> None:
    """`/config` must use HTML, not legacy Markdown — Telegram rejects
    Markdown messages with unbalanced ``_`` pairs (HTTP 400) and the
    operator sees no reply. With ~100 underscored field names, the legacy
    parser is essentially guaranteed to misfire on the full dump.
    """
    s = _settings()
    p, posts, _ = _make_poller(s)
    p._handle_update(_msg(text="/config max_position"))
    assert posts, "expected at least one reply"
    for post in posts:
        body = post[1]
        # parse_mode is set to HTML, never Markdown.
        assert body.get("parse_mode") == "HTML"
        # Output uses <pre> for monospace alignment.
        assert "<pre>" in body["text"]
        assert "</pre>" in body["text"]
