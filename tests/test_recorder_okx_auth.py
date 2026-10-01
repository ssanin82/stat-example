"""Recorder OKX auth — regression pins for WS-login format.

v1.5.51 — pinning the fix for a production bug shipped in v1.5.45 →
v1.5.50: ``backtesting/recorder/okx_auth.ws_login_args`` reused the
REST ``iso_timestamp_ms()`` helper for the WS login payload's
``timestamp`` field. OKX V5 requires Unix-epoch-SECONDS there (NOT
ISO 8601, which is the REST format). OKX silently rejected login with
a non-zero ``code``; the recorder's private-WS subscriber re-tried in
a tight reconnect loop and never wrote a byte to
``okx_private.jsonl.gz`` — the file didn't even get created. The bot
itself was unaffected because ``app/exchange/okx_ws.py:329-330`` uses
the correct epoch-seconds format.

These tests pin two invariants so the bug can't reappear:

1. ``ws_login_timestamp`` returns a parsable Unix-epoch-seconds string
   (not ISO, not milliseconds, not microseconds).

2. The ``timestamp`` field in ``ws_login_args`` payload is the SAME
   value used in the signature pre-hash. (Producing a fresh ``ts``
   for the payload and a different one for the sign would silently
   fail — the signature wouldn't match the timestamp OKX sees.)

3. Given a known (timestamp, secret) pair, ``ws_login_args``
   produces byte-identical output to the bot's
   ``app.exchange.okx_ws._okx_login_sign`` — pinning the recorder to
   the bot's auth so they can never drift apart again.
"""

from __future__ import annotations

import time

import pytest

from app.exchange.okx_ws import _okx_login_sign
from backtesting.recorder.okx_auth import (
    iso_timestamp_ms,
    sign,
    ws_login_args,
    ws_login_timestamp,
)


# ---------------------------------------------------------------------------
# Format pins
# ---------------------------------------------------------------------------


def test_ws_login_timestamp_is_unix_epoch_seconds_as_string() -> None:
    """WS login timestamp MUST be Unix epoch seconds as a string.

    Returning ISO 8601 (the production bug) would be a valid Python
    string and look correct at a glance — but OKX rejects it. Pin
    the format explicitly.
    """
    ts = ws_login_timestamp()
    # It's a string.
    assert isinstance(ts, str)
    # Parses cleanly as int.
    parsed = int(ts)
    # Within ±5 s of the test wall clock (sanity — not testing wall
    # clock semantics, just that it's a CURRENT epoch-seconds value,
    # not e.g. ms or microseconds which would be 1000x / 1_000_000x
    # larger).
    now = int(time.time())
    assert abs(parsed - now) < 5, (
        f"ws_login_timestamp returned {ts} (parsed={parsed}) but the "
        f"wall clock is {now}. Likely the wrong unit — OKX expects "
        f"SECONDS, not ms / μs / ISO."
    )


def test_ws_login_timestamp_is_NOT_iso_format() -> None:
    """Defense-in-depth: the production bug returned an ISO string
    here. Pin that it never reverts."""
    ts = ws_login_timestamp()
    # ISO 8601 has 'T' and 'Z' in well-known positions; epoch seconds
    # is pure digits.
    assert "T" not in ts
    assert "Z" not in ts
    assert "-" not in ts
    assert "." not in ts
    assert ts.isdigit()


def test_iso_timestamp_ms_unchanged_for_REST() -> None:
    """The REST helper still produces ISO 8601 with milliseconds —
    the fix must not regress the working REST path."""
    ts = iso_timestamp_ms()
    assert isinstance(ts, str)
    assert ts.endswith("Z")
    assert "T" in ts
    # Format: YYYY-MM-DDTHH:MM:SS.mmmZ → 24 chars.
    assert len(ts) == 24, f"unexpected ISO length {len(ts)}: {ts!r}"


# ---------------------------------------------------------------------------
# Payload-vs-signature consistency
# ---------------------------------------------------------------------------


def test_ws_login_args_timestamp_field_matches_signature_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``timestamp`` field in the payload MUST equal the timestamp
    used in the signature pre-hash. If they differ (e.g. one helper
    reads the clock twice), OKX would silently reject the login —
    even with the right format.
    """
    # Freeze the clock by pinning ws_login_timestamp.
    frozen_ts = "1779523131"
    monkeypatch.setattr(
        "backtesting.recorder.okx_auth.ws_login_timestamp",
        lambda: frozen_ts,
    )
    args = ws_login_args(
        api_key="test-key",
        api_secret="test-secret-32-chars-min-please-x",
        api_passphrase="test-passphrase",
    )
    assert args["timestamp"] == frozen_ts
    # Compute the expected signature with the same frozen ts and
    # verify the payload's sign matches.
    expected_sig = sign(
        timestamp=frozen_ts,
        method="GET",
        request_path="/users/self/verify",
        body="",
        secret_key="test-secret-32-chars-min-please-x",
    )
    assert args["sign"] == expected_sig
    # Spot-check the other fields too.
    assert args["apiKey"] == "test-key"
    assert args["passphrase"] == "test-passphrase"


# ---------------------------------------------------------------------------
# Cross-implementation parity (recorder vs bot)
# ---------------------------------------------------------------------------


def test_recorder_ws_signature_matches_bot_implementation() -> None:
    """The recorder's ``sign(timestamp, GET, /users/self/verify, ...)``
    MUST produce byte-identical output to the bot's
    ``_okx_login_sign(secret, ts)`` for the same (secret, ts).

    This is the strongest regression pin: if either side ever changes
    its prehash recipe in a non-cross-compatible way, this test fails.
    The bot's implementation is the authoritative one (it's been
    talking to OKX successfully for months); the recorder is the
    follower.
    """
    secret = "abcdefghijklmnopqrstuvwxyz123456"
    ts = "1779523131"  # arbitrary epoch-seconds value

    bot_sig = _okx_login_sign(secret, ts)
    recorder_sig = sign(
        timestamp=ts,
        method="GET",
        request_path="/users/self/verify",
        body="",
        secret_key=secret,
    )
    assert recorder_sig == bot_sig, (
        f"recorder vs bot signature mismatch — recorder will fail "
        f"OKX login. recorder={recorder_sig!r} bot={bot_sig!r}"
    )


def test_signature_changes_when_timestamp_changes() -> None:
    """Trivial — make sure the test fixture itself isn't degenerate."""
    s1 = sign(
        timestamp="1000000000",
        method="GET",
        request_path="/users/self/verify",
        body="",
        secret_key="x" * 32,
    )
    s2 = sign(
        timestamp="2000000000",
        method="GET",
        request_path="/users/self/verify",
        body="",
        secret_key="x" * 32,
    )
    assert s1 != s2
