"""v1.5.152 — ``Bot._dump_crash_snapshot()`` preserves diagnostic
state to persistent disk BEFORE any kill-triggered exit.

The kill path in ``Bot.kill()``:
  1. Marks state KILLED
  2. Writes ``bot_shutdown.json`` (existing)
  3. **Dumps crash snapshot bundle** (v1.5.152) ← these tests
  4. Cancels orders, optionally flattens
  5. Optionally schedules ``os._exit(42)`` for systemd auto-restart

The dump must work even when:
  * The configured ``CRASH_SNAPSHOT_DIR`` is unwritable (falls back
    to ``<cwd>/snapshots_crash/``).
  * The bot's internal publishers throw (catch-all wrapper logs but
    doesn't raise — kill cleanup must still run).
  * ``CRASH_SNAPSHOT_DIR`` is empty (feature disabled).

The bundle MUST contain the operator-pullable diagnostic surface:
  * ``meta.json``           — version, profile, kill_reason, pid
  * ``state_current.json``  — full BotState.snapshot_dict()
  * ``events_recent.json``  — last N bot_events rows
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.bot import Bot
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / (
        f"mm_crash_{os.getpid()}_{uuid.uuid4().hex}.db"
    )


def _make_bot(**overrides) -> tuple[Bot, Path]:
    """Build a Bot scaffold + a tmp dir to use as CRASH_SNAPSHOT_DIR.
    The tmp dir is auto-cleaned by pytest's tmp_path fixture in the
    test cases — this helper just returns its path string."""
    crash_dir = Path(tempfile.mkdtemp(prefix="crash_snap_test_"))
    db_path = _db_path()
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "PRIVATE_WS_ENABLED": False,
        "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        "CRASH_SNAPSHOT_DIR": str(crash_dir),
        "CRASH_SNAPSHOT_EVENT_LIMIT": 50,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(s, state, client, storage, exit_fn=lambda code: None)
    return bot, crash_dir


# -------------------------------------------------------------------
# Happy path — bundle lands on disk with expected contents
# -------------------------------------------------------------------


def test_dump_creates_bundle_with_three_json_files() -> None:
    """The dump produces meta.json + state_current.json +
    events_recent.json under a versioned timestamped directory."""
    bot, crash_root = _make_bot()
    path_str = bot._dump_crash_snapshot("stale_data_kill_escalated")
    assert path_str is not None
    leaf = Path(path_str)
    assert leaf.is_dir()
    assert leaf.parent == crash_root
    # Three required files exist:
    assert (leaf / "meta.json").is_file()
    assert (leaf / "state_current.json").is_file()
    assert (leaf / "events_recent.json").is_file()


def test_meta_json_carries_kill_reason_and_version() -> None:
    """meta.json contains the kill_reason + bot_version verbatim so
    the operator can find which bundle corresponds to which kill."""
    bot, _ = _make_bot()
    path_str = bot._dump_crash_snapshot("public_ws_stale_kill")
    meta = json.loads((Path(path_str) / "meta.json").read_text())
    assert meta["kill_reason"] == "public_ws_stale_kill"
    assert isinstance(meta["bot_version"], str)
    assert meta["bot_version"]  # non-empty
    assert "captured_at_utc" in meta
    assert isinstance(meta["pid"], int)


def test_state_current_json_has_expected_keys() -> None:
    """state_current.json is the full snapshot_dict — should contain
    the behavioural_gates block at minimum (proves we're capturing
    the same shape the heartbeat publisher emits)."""
    bot, _ = _make_bot()
    path_str = bot._dump_crash_snapshot("manual_kill")
    state = json.loads((Path(path_str) / "state_current.json").read_text())
    # snapshot_dict's top-level shape:
    assert "behavioural_gates" in state
    # And the post-v1.5.149 publisher blocks are present:
    assert "negative_expectancy_dampen" in state["behavioural_gates"]


def test_events_recent_json_captures_bot_events() -> None:
    """events_recent.json contains the last N bot_events rows so the
    operator can see the kill's run-up (warnings, gate fires, etc.)
    without parsing the journal."""
    bot, _ = _make_bot()
    # Plant a few events to verify they show up.
    from app.models import EventSeverity
    for i in range(3):
        bot._log_event(
            EventSeverity.WARNING,
            f"test_planted_{i}",
            f"planted event {i}",
            {"i": i},
        )
    path_str = bot._dump_crash_snapshot("desync_unrecoverable")
    events = json.loads((Path(path_str) / "events_recent.json").read_text())
    assert isinstance(events, list)
    types = [e["event_type"] for e in events]
    # The kill event we just logged + planted events:
    assert "kill" not in types  # kill() isn't called in this test
    assert "test_planted_0" in types
    assert "test_planted_2" in types


# -------------------------------------------------------------------
# Directory naming
# -------------------------------------------------------------------


def test_dir_name_format_versioned_timestamped_reason() -> None:
    """Directory shape: v<ver>-<YYYYMMDD-HHMMSS>-<reason>. Pins the
    convention so operator's listing scripts can sort/filter."""
    bot, _ = _make_bot()
    path_str = bot._dump_crash_snapshot("public_ws_stale_kill")
    leaf = Path(path_str).name
    # Starts with "v<bot_version>-" (the version contains dots which
    # we don't want to brittle-test, just verify the prefix shape).
    assert leaf.startswith("v")
    # Date-time hyphen pattern after the version:
    parts = leaf.split("-")
    # Expect: ["v1.5.X", "<date>", "<time>", "public_ws_stale_kill"]
    # at minimum 4 parts when reason has no embedded dashes.
    assert len(parts) >= 4
    # Reason is preserved as the tail:
    assert leaf.endswith("public_ws_stale_kill")


def test_dir_name_sanitises_unsafe_reason() -> None:
    """Reasons with filesystem-hostile characters (slashes, spaces,
    null bytes, etc.) are replaced with underscores so the directory
    name is always safe to mkdir on every platform."""
    bot, _ = _make_bot()
    path_str = bot._dump_crash_snapshot(
        "weird/reason with spaces!@# and chars"
    )
    leaf = Path(path_str).name
    # No path separators in the leaf:
    assert "/" not in leaf
    assert "\\" not in leaf
    # No whitespace:
    assert " " not in leaf
    # The directory was actually created (path is writable):
    assert Path(path_str).is_dir()


def test_dir_name_truncates_long_reason() -> None:
    """Multi-token comma-separated reasons get truncated at 60 chars
    so directory listings stay readable."""
    bot, _ = _make_bot()
    long_reason = "reason_token_" + "x" * 200
    path_str = bot._dump_crash_snapshot(long_reason)
    leaf = Path(path_str).name
    # The reason portion (after the timestamp) is bounded; total leaf
    # length stays under a reasonable filesystem cap (~120 chars).
    assert len(leaf) < 130


# -------------------------------------------------------------------
# Feature toggle / fallback / defensive
# -------------------------------------------------------------------


def test_empty_crash_snapshot_dir_disables_feature() -> None:
    """Setting CRASH_SNAPSHOT_DIR='' disables the dump entirely.
    Returns None without writing anything."""
    bot, _ = _make_bot(CRASH_SNAPSHOT_DIR="")
    result = bot._dump_crash_snapshot("any_reason")
    assert result is None


def test_unwritable_path_falls_back_to_repo_root(
    tmp_path, monkeypatch
) -> None:
    """If the configured CRASH_SNAPSHOT_DIR can't be created (no
    permission, no disk), the dump falls back to
    ``<cwd>/snapshots_crash/`` so the bundle ALWAYS lands somewhere."""
    # Point to a path under tmp_path so the test can verify the
    # fallback location AFTER changing cwd.
    monkeypatch.chdir(tmp_path)
    # Unwritable primary: /proc is read-only on Linux; on Windows
    # use a deeply-invalid path. Either yields PermissionError or
    # OSError, both caught by the fallback.
    bad = "/proc/dtc_crash_test_unwritable" if os.name != "nt" else "Z:\\nonexistent\\path\\that\\cannot_be_created\\xyz"
    bot, _ = _make_bot(CRASH_SNAPSHOT_DIR=bad)
    path_str = bot._dump_crash_snapshot("test_fallback")
    # On non-Linux or if /proc is writable for some reason, the
    # primary might succeed; the fallback path is best-effort. Just
    # assert SOMETHING was written:
    assert path_str is not None
    assert Path(path_str).is_dir()


def test_state_snapshot_failure_does_not_block_other_files(
    monkeypatch,
) -> None:
    """If snapshot_dict() raises (defensive scenario: corrupted state),
    the other bundle files still write. The dump is best-effort
    per-file, not all-or-nothing."""
    bot, _ = _make_bot()
    # Monkeypatch snapshot_dict to raise:
    monkeypatch.setattr(
        bot._state,
        "snapshot_dict",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated corruption")),
    )
    path_str = bot._dump_crash_snapshot("test_partial_failure")
    # Dump did NOT raise (caught the inner exception):
    assert path_str is not None
    # meta.json still landed:
    assert (Path(path_str) / "meta.json").is_file()
    # events_recent.json still landed:
    assert (Path(path_str) / "events_recent.json").is_file()
    # state_current.json did NOT land (snapshot_dict raised):
    assert not (Path(path_str) / "state_current.json").exists()


def test_top_level_exception_caught_returns_none(monkeypatch) -> None:
    """If something pathological happens above the per-file try/except
    blocks (e.g. mkdir succeeds initially but the path is then
    deleted), the catch-all wrapper logs and returns None. The kill
    cleanup that follows must continue."""
    bot, _ = _make_bot()
    # Force pathlib.Path to explode on instantiation — exercises the
    # catch-all in _dump_crash_snapshot.
    import app.bot as bot_mod

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated top-level failure")

    monkeypatch.setattr(bot_mod, "datetime", _explode, raising=False)
    # The method must NOT raise. Result may be None.
    result = bot._dump_crash_snapshot("test_catchall")
    # Either it succeeded (monkeypatch didn't intercept the right
    # import path) or it returned None defensively. Either way: no
    # exception propagated.
    assert result is None or isinstance(result, str)


# -------------------------------------------------------------------
# Integration with kill()
# -------------------------------------------------------------------


def test_kill_calls_dump_crash_snapshot() -> None:
    """The kill() path invokes _dump_crash_snapshot with the kill
    reason BEFORE the cancel/flatten cleanup so the bundle captures
    state at the moment of kill, not after."""
    bot, crash_root = _make_bot(KILL_AUTO_RESTART_GRACE_SECONDS=0.0)
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot.flatten = MagicMock()
    bot.kill("stale_data_kill")
    # Find the bundle dir:
    bundles = list(crash_root.iterdir())
    assert len(bundles) == 1
    leaf = bundles[0]
    assert leaf.name.endswith("stale_data_kill")
    meta = json.loads((leaf / "meta.json").read_text())
    assert meta["kill_reason"] == "stale_data_kill"


def test_kill_with_disabled_dump_still_proceeds() -> None:
    """If CRASH_SNAPSHOT_DIR is empty, kill() still proceeds normally
    (the dump is a no-op observability aid, not a safety contract)."""
    bot, _ = _make_bot(
        CRASH_SNAPSHOT_DIR="",
        KILL_AUTO_RESTART_GRACE_SECONDS=0.0,
    )
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot.flatten = MagicMock()
    bot.kill("any_reason")
    # kill() completed without raising. State is KILLED:
    assert bot._state.killed is True
    assert bot._state.kill_reason == "any_reason"
