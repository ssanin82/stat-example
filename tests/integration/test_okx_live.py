"""Integration tests against a REAL OKX account.

**These tests place + cancel real orders.** They are skipped unless
``OKX_API_KEY``, ``OKX_API_SECRET``, and ``OKX_API_PASSPHRASE`` are
all set in the environment. See ``tests/integration/README.md`` for
the operator playbook.

Scope:

* Public REST: instruments lookup (no creds needed -- runs even
  without the env vars, useful as a connectivity smoke).
* Signed REST: ``/account/balance``, ``/account/positions``,
  ``/orders-pending`` (read-only, exercises auth end-to-end).
* Order lifecycle: place ONE tiny post-only order at a price that
  cannot fill (mid * 0.5 for buy), verify it appears in
  ``/orders-pending``, cancel it, verify it's gone.
* Private WS: connect, login, subscribe to ``orders`` channel, place
  an order, verify the order-update event arrives, cancel.

What these tests do NOT do:
* Place market orders or anything that could fill at venue prices.
* Place orders larger than the configured test size (default 1
  contract = 1000 DOGE for DOGE-USDT-SWAP).
* Close positions you didn't open.
* Touch any other symbol than the one configured.

Worst case on test interruption: a single small post-only order is
resting on the book. Run ``python scripts/okx_flatten.py`` to clean.
"""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.exchange.okx_client import OkxClient
from tests.settings_helpers import UnitTestSettings


# ---------------------------------------------------------------------------
# Credential loading: prefer a gitignored tests/integration/.env file so
# the operator (or an automated runner) doesn't have to set env vars in
# a shell that may mangle special characters. Process env still wins if
# both are present -- that lets CI / one-off `$env:VAR=...` overrides
# beat the file without editing.
# ---------------------------------------------------------------------------


def _load_env_file_if_present() -> None:
    """Load ``tests/integration/.env`` into ``os.environ`` if the file
    exists. Manual parser (no python-dotenv dependency) so this stays
    runnable in any venv. Lines like ``KEY=value`` or ``KEY="value"``;
    ``#`` starts a comment; surrounding quotes on values are stripped
    only when both ends match. Mismatched-quote lines (e.g. ``KEY="abc``
    with no closing quote) are kept verbatim AND a warning is printed,
    because that's almost always a typo that would silently produce a
    wrong value (with a leading or trailing literal quote). Process
    env wins -- we don't overwrite an already-set var.
    """
    import sys

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for lineno, line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip()
        # Strip matching surrounding quotes (single or double). Warn
        # loudly on mismatched-quote lines -- those are typos that
        # would otherwise silently bake a literal quote into the value.
        starts_q = len(v) >= 1 and v[0] in ("'", '"')
        ends_q = len(v) >= 1 and v[-1] in ("'", '"')
        both_match = len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"')
        if both_match:
            v = v[1:-1]
        elif starts_q or ends_q:
            sys.stderr.write(
                f"[okx-live] WARNING: mismatched quote on '{k}' "
                f"({env_path.name}:{lineno}). Value kept verbatim, "
                f"which probably is NOT what you want. Use either no "
                f"quotes (recommended) or matching quotes on both ends.\n"
            )
        if k and k not in os.environ:
            os.environ[k] = v


_load_env_file_if_present()


_OKX_KEY = os.environ.get("OKX_API_KEY", "").strip()
_OKX_SECRET = os.environ.get("OKX_API_SECRET", "").strip()
_OKX_PASS = os.environ.get("OKX_API_PASSPHRASE", "").strip()

_HAS_CREDS = bool(_OKX_KEY and _OKX_SECRET and _OKX_PASS)


def _redact_fingerprint(value: str) -> str:
    """Render a non-secret fingerprint of a credential: length + first
    2 + last 2 chars. Lets the operator visually compare to what the partner
    sent without putting the secret on screen.
    """
    if not value:
        return "<empty>"
    n = len(value)
    if n <= 4:
        return f"len={n}, ***"
    return f"len={n}, {value[:2]}***{value[-2:]}"


# One-line diagnostic at module-import time. Visible with `pytest -s`.
# Helps catch the "passphrase got mangled by my shell" failure mode.
if _HAS_CREDS:
    print(
        f"\n[okx-live] OKX_API_KEY        {_redact_fingerprint(_OKX_KEY)}"
        f"\n[okx-live] OKX_API_SECRET     {_redact_fingerprint(_OKX_SECRET)}"
        f"\n[okx-live] OKX_API_PASSPHRASE {_redact_fingerprint(_OKX_PASS)}"
    )

requires_okx_creds = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE not set",
)


# Tunable knobs -- pick conservative defaults so the test is safe to
# run without thinking. Operator can override via env if needed.
_SYMBOL = os.environ.get("OKX_INTEGRATION_SYMBOL", "DOGE-USDT-SWAP").strip()
_TEST_SIZE_BASE = float(os.environ.get("OKX_INTEGRATION_TEST_SIZE_BASE", "1000"))
_DEMO = os.environ.get("OKX_DEMO_TRADING", "").strip().lower() in ("true", "1", "yes")


def _live_settings() -> Settings:
    """Build a Settings object pointing at the operator's real OKX
    account. Other venues blanked so they don't trip side-effects.
    """
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": False,  # we still construct the client without
                                   # the trading-flag gate; place_order
                                   # checks has_write_access independently.
        "EXCHANGE": "okx",
        "REFERENCE_EXCHANGE": "off",
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "BLUEFIN_PRIVATE_KEY": "",
        "BLUEFIN_ACCOUNT_ADDRESS": "",
        "BINANCE_API_KEY": "",
        "BINANCE_API_SECRET": "",
        "OKX_API_KEY": _OKX_KEY,
        "OKX_API_SECRET": _OKX_SECRET,
        "OKX_API_PASSPHRASE": _OKX_PASS,
        "OKX_REST_URL": os.environ.get("OKX_REST_URL", "https://www.okx.com"),
        "OKX_PUBLIC_WS_URL": os.environ.get(
            "OKX_PUBLIC_WS_URL", "wss://ws.okx.com:8443/ws/v5/public"
        ),
        "OKX_PRIVATE_WS_URL": os.environ.get(
            "OKX_PRIVATE_WS_URL", "wss://ws.okx.com:8443/ws/v5/private"
        ),
        "OKX_DEMO_TRADING": _DEMO,
        "SYMBOL": _SYMBOL,
    })


# ---------------------------------------------------------------------------
# Public REST -- no creds needed; runs even without OKX env vars
# ---------------------------------------------------------------------------


def test_public_instruments_endpoint_returns_swap_universe() -> None:
    """Connectivity smoke: hit OKX's public instruments endpoint and
    confirm we get back a SWAP universe row for the test symbol.
    """
    import httpx

    url = os.environ.get("OKX_REST_URL", "https://www.okx.com")
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            f"{url}/api/v5/public/instruments",
            params={"instType": "SWAP", "instId": _SYMBOL},
        )
    assert resp.status_code == 200, f"OKX returned HTTP {resp.status_code}"
    body = resp.json()
    assert body.get("code") == "0", f"OKX error: {body.get('msg')}"
    rows = body.get("data") or []
    assert rows, f"OKX returned no rows for {_SYMBOL}"
    row = rows[0]
    assert row.get("instId") == _SYMBOL
    assert "tickSz" in row
    assert "ctVal" in row
    print(f"\n  [public] {_SYMBOL} tickSz={row['tickSz']} ctVal={row['ctVal']}")


# ---------------------------------------------------------------------------
# Signed REST -- read-only
# ---------------------------------------------------------------------------


@requires_okx_creds
def test_signed_account_balance_returns_a_dict() -> None:
    """End-to-end auth check: account balance returns code "0" and a
    data row. If this fails, the most likely causes are:
      * Wrong passphrase (OKX-specific gotcha)
      * Wrong API secret (signature mismatch)
      * IP not whitelisted on the API key
    """
    s = _live_settings()
    client = OkxClient(s)
    snap = client.fetch_account_snapshot("")
    print(
        f"\n  [signed] account: equity={snap.equity_usd} "
        f"cash={snap.cash_usd} withdrawable={snap.withdrawable_usd}"
    )
    # Equity may be zero for a brand-new account -- accept None or >= 0.
    assert snap.equity_usd is None or snap.equity_usd >= 0


@requires_okx_creds
def test_signed_position_returns_position_snapshot() -> None:
    """Should return a PositionSnapshot for the test symbol. Quantity
    may be zero (no open position) -- that's fine.
    """
    s = _live_settings()
    client = OkxClient(s)
    pos = client.fetch_position("", _SYMBOL)
    print(
        f"\n  [signed] position {_SYMBOL}: qty={pos.position_qty} "
        f"mark={pos.mark_price} entry={pos.avg_entry_price}"
    )
    assert pos.symbol == _SYMBOL


@requires_okx_creds
def test_signed_open_orders_returns_a_list() -> None:
    s = _live_settings()
    client = OkxClient(s)
    opens = client.fetch_open_orders_raw("")
    print(f"\n  [signed] open orders for {_SYMBOL}: count={len(opens)}")
    assert isinstance(opens, list)


# ---------------------------------------------------------------------------
# Order lifecycle -- place tiny post-only far from market, verify, cancel
# ---------------------------------------------------------------------------


@requires_okx_creds
def test_place_then_cancel_post_only_order_far_from_market() -> None:
    """Critical lifecycle test. Steps:
      1. Read mid via /market/books.
      2. Place a tiny post-only BUY at mid * 0.5 (cannot possibly fill).
      3. Verify the order appears in /orders-pending.
      4. Cancel it.
      5. Verify it's gone from /orders-pending.

    On any failure between steps 2 and 4, the cleanup `finally` block
    will attempt to cancel the order. As a final safety net, run
    ``python scripts/okx_flatten.py`` if anything looks off.
    """
    s = _live_settings()
    client = OkxClient(s)

    bbo = client.fetch_best_bid_ask(_SYMBOL)
    assert bbo.mid_price and bbo.mid_price > 0, "no mid -- venue down?"
    far_buy_px = round(bbo.mid_price * 0.5, 5)
    print(
        f"\n  [lifecycle] mid={bbo.mid_price} far_buy_px={far_buy_px} "
        f"sz_base={_TEST_SIZE_BASE}"
    )

    placed_oid: int | None = None
    placed_cloid: str | None = None
    try:
        resp = client.place_post_only_limit(
            _SYMBOL, is_buy=True, sz=_TEST_SIZE_BASE, limit_px=far_buy_px
        )
        oid, outcome, reason = client.interpret_place_response(resp)
        print(f"  [lifecycle] place outcome={outcome} oid={oid} reason={reason!r}")
        assert outcome == "accepted", f"place rejected: {reason}"
        assert oid is not None and oid > 0
        placed_oid = oid
        # Pull the cloid from the raw response for cancel-by-cloid path.
        data = resp.get("data") or []
        if data and isinstance(data[0], dict):
            placed_cloid = str(data[0].get("clOrdId") or "") or None

        # Verify it appears in open orders.
        time.sleep(0.5)  # OKX needs a beat to surface the order
        opens = client.fetch_open_orders_raw("")
        oids_seen = {o.oid for o in opens}
        assert placed_oid in oids_seen, (
            f"placed oid {placed_oid} not in /orders-pending: {oids_seen}"
        )
        print(f"  [lifecycle] confirmed in /orders-pending: oid={placed_oid}")

        # Cancel it.
        cancel_resp = client.cancel_order(_SYMBOL, placed_oid)
        kind, detail = client.interpret_cancel_response(cancel_resp)
        print(f"  [lifecycle] cancel kind={kind} detail={detail!r}")
        assert kind in ("success", "benign_missing"), (
            f"cancel failed: {kind} / {detail}"
        )

        # Verify it's gone.
        time.sleep(0.5)
        opens_after = client.fetch_open_orders_raw("")
        oids_after = {o.oid for o in opens_after}
        assert placed_oid not in oids_after, (
            f"oid {placed_oid} still present after cancel: {oids_after}"
        )
        print(f"  [lifecycle] confirmed gone from /orders-pending")
        placed_oid = None  # avoid double-cancel in finally
    finally:
        # Defensive cleanup: if we placed but didn't cancel, try once
        # more from the finally block. Helps when an assert fires
        # mid-test.
        if placed_oid is not None:
            try:
                client.cancel_order(_SYMBOL, placed_oid)
                print(f"  [lifecycle] cleanup-cancelled oid={placed_oid}")
            except Exception as e:
                print(
                    f"  [lifecycle] CLEANUP CANCEL FAILED for oid={placed_oid}: "
                    f"{e!r}\n  RUN scripts/okx_flatten.py NOW"
                )


# ---------------------------------------------------------------------------
# Private WS -- connect, login, subscribe, see an order update
# ---------------------------------------------------------------------------


@requires_okx_creds
def test_private_ws_login_subscribe_and_receive_order_update() -> None:
    """Open the private WS, login, subscribe to orders. Place a tiny
    post-only order via REST, then assert that an order-update event
    arrives on the WS queue within a reasonable timeout.
    """
    from app.exchange.okx_ws import OkxPrivateStream
    from app.exchange.private_events import PrivateOrderUpdateEvent

    s = _live_settings()
    rest = OkxClient(s)
    q: "queue.Queue[Any]" = queue.Queue(maxsize=200)

    stream = OkxPrivateStream(s, q, on_queue_drop=None, state=None)
    stream.set_contract_value(rest.contract_value)
    stream.start()

    placed_oid: int | None = None
    try:
        # Wait up to 10s for login + subscribe acks (visible as
        # connection events on the queue).
        login_deadline = time.time() + 10.0
        while time.time() < login_deadline:
            if stream._logged_in.is_set():
                break
            time.sleep(0.2)
        assert stream._logged_in.is_set(), "WS did not log in within 10s"
        print("\n  [ws] logged in")

        # Wait one more second for the subscribe ack to land before
        # we trigger an order event we want to capture.
        time.sleep(1.0)

        bbo = rest.fetch_best_bid_ask(_SYMBOL)
        far_buy_px = round(bbo.mid_price * 0.5, 5)
        resp = rest.place_post_only_limit(
            _SYMBOL, is_buy=True, sz=_TEST_SIZE_BASE, limit_px=far_buy_px
        )
        oid, outcome, _ = rest.interpret_place_response(resp)
        assert outcome == "accepted"
        placed_oid = oid
        print(f"  [ws] placed oid={oid} via REST")

        # Drain the queue for up to 5s waiting for an order-update
        # event matching our oid.
        seen_event = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                ev = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(ev, PrivateOrderUpdateEvent) and ev.oid == placed_oid:
                seen_event = True
                print(
                    f"  [ws] received order update oid={ev.oid} "
                    f"status={ev.status}"
                )
                break
        assert seen_event, "did not receive order update via private WS"
    finally:
        stream.stop()
        if placed_oid is not None:
            try:
                rest.cancel_order(_SYMBOL, placed_oid)
                print(f"  [ws] cleanup-cancelled oid={placed_oid}")
            except Exception as e:
                print(
                    f"  [ws] CLEANUP CANCEL FAILED for oid={placed_oid}: "
                    f"{e!r}\n  RUN scripts/okx_flatten.py NOW"
                )


# ---------------------------------------------------------------------------
# Public WS -- trades + books5 subscription, flow_score plumbing
# ---------------------------------------------------------------------------
#
# These tests verify the 1.1.29 change: the OKX public WS now subscribes
# to BOTH ``books5`` (BBO -- always was) AND ``trades`` (public prints --
# new) and feeds trade prints into ``BotState.flow_score`` so the
# Priority #3 v2 reservation-shift alpha actually has signal on OKX.
#
# Public WS does NOT require API credentials -- these tests run even
# without OKX_API_KEY set. They use the same ``OKX_INTEGRATION_SYMBOL``
# env var as the rest of the suite (default DOGE-USDT-SWAP). For the
# operator's verification, set ``OKX_INTEGRATION_SYMBOL=TON-USDT-SWAP``
# to mirror the live bot config.


def _public_ws_settings() -> Settings:
    """Minimal Settings for spinning up the public WS only. Skip the
    venue-specific creds we don't need; only the OKX URL + symbol +
    reconnect knobs matter."""
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": False,
        "EXCHANGE": "okx",
        "REFERENCE_EXCHANGE": "off",
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "BLUEFIN_PRIVATE_KEY": "",
        "BLUEFIN_ACCOUNT_ADDRESS": "",
        "BINANCE_API_KEY": "",
        "BINANCE_API_SECRET": "",
        "OKX_API_KEY": "x",       # public WS doesn't auth, but settings model requires presence
        "OKX_API_SECRET": "x",
        "OKX_API_PASSPHRASE": "x",
        "OKX_REST_URL": os.environ.get("OKX_REST_URL", "https://www.okx.com"),
        "OKX_PUBLIC_WS_URL": os.environ.get(
            "OKX_PUBLIC_WS_URL", "wss://ws.okx.com:8443/ws/v5/public"
        ),
        "OKX_PRIVATE_WS_URL": os.environ.get(
            "OKX_PRIVATE_WS_URL", "wss://ws.okx.com:8443/ws/v5/private"
        ),
        "SYMBOL": _SYMBOL,
        # Make flow-score alpha non-zero so the integration test also
        # exercises the end-to-end path through compute_quote_decision.
        "FLOW_SCORE_RESERVATION_ALPHA": 0.4,
        "FLOW_SCORE_RESERVATION_CLIP": 0.85,
        "OB_IMBALANCE_ALPHA": 0.0,         # isolate flow-score effect
        "TREND_DRIFT_RESERVATION_ALPHA": 0.0,
        "BASIS_DEVIATION_ALPHA": 0.0,
        "MICROPRICE_RESERVATION_ENABLED": False,
        "INVENTORY_SKEW_COEFF_BPS": 0.0,
        "BASE_HALF_SPREAD_BPS": 5.0,
        "MIN_HALF_SPREAD_BPS": 0.5,
        "MAX_HALF_SPREAD_BPS": 30.0,
        "VOL_MULTIPLIER": 0.0,
        "TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.5,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.5,
        "MAX_ABS_POSITION": 10.0,
    })


def test_public_ws_subscribes_to_trades_and_feeds_flow_score() -> None:
    """1.1.29 change verification: connect the OKX public WS for the
    integration symbol, wait up to 20s, and confirm:

    1. We received at least one BBO tick (existing books5 path still works).
    2. We received at least one public trade print (new ``trades`` path).
    3. The print landed in ``state.flow_score`` (record_trade dispatch).
    4. ``state.recent_trades`` deque has the print too.
    5. ``flow_score.snapshot()`` reports non-zero trade_history_count.

    Symbol must be ACTIVELY TRADING for this to terminate within 20s.
    Default test symbol DOGE-USDT-SWAP / TON-USDT-SWAP both qualify
    during normal hours. If you're running this when OKX is in
    maintenance, raise the timeout or use a busier pair.
    """
    from app.exchange.okx_public_ws import OkxPublicStream
    from app.state import BotState

    s = _public_ws_settings()
    state = BotState(s)

    bbo_count = 0

    def _on_bbo(_bbo) -> None:
        nonlocal bbo_count
        bbo_count += 1

    stream = OkxPublicStream(s, state, _SYMBOL, _on_bbo)
    stream.set_contract_value(1.0)  # placeholder; real value from REST
    stream.start()

    deadline = time.time() + 20.0
    seen_trade = False
    try:
        # Tight poll: trades typically arrive in <2s for active symbols.
        while time.time() < deadline:
            with state._lock:
                trade_count = len(state.recent_trades)
            if trade_count > 0 and bbo_count > 0:
                seen_trade = True
                break
            time.sleep(0.25)
    finally:
        stream.stop()

    print(
        f"\n  [public-ws] symbol={_SYMBOL} bbo_count={bbo_count} "
        f"trade_count={len(state.recent_trades)}"
    )
    assert bbo_count > 0, (
        f"no BBO received in 20s on {_SYMBOL} -- books5 channel may have "
        f"regressed or the symbol is not trading right now"
    )
    assert seen_trade, (
        f"no trade prints received in 20s on {_SYMBOL} -- the new trades "
        f"subscription didn't deliver. Check OKX channel name + parser. "
        f"For a busier symbol set OKX_INTEGRATION_SYMBOL=BTC-USDT-SWAP."
    )

    # Verify the print actually went through flow_score.record_trade,
    # not just into recent_trades.
    snap = state.flow_score.snapshot()
    print(
        f"  [public-ws] flow_score: history={state.flow_score.snapshot_dict()['trade_history_count']} "
        f"tfi={snap.tfi_signed_normalised:+.3f} "
        f"streak_buy={snap.streak_buy_count} streak_sell={snap.streak_sell_count}"
    )
    assert state.flow_score.snapshot_dict()["trade_history_count"] > 0, (
        "trade prints landed in recent_trades but flow_score got zero --"
        " record_trade dispatch is broken in _handle_trades_message"
    )
    # tfi should be in [-1, +1] (could be 0 in a perfectly balanced
    # window, so don't assert non-zero strictly). Streak sides are
    # mutually exclusive at the tail.
    assert -1.0 <= snap.tfi_signed_normalised <= 1.0
    assert snap.streak_buy_count == 0 or snap.streak_sell_count == 0


def test_public_ws_flow_score_drives_reservation_shift_end_to_end() -> None:
    """End-to-end smoke test of the 1.1.29 reservation-alpha pipeline:
    real WS prints feed flow_score, flow_score.snapshot() feeds
    compute_quote_decision via the same call shape used in bot.py,
    reservation shifts away from raw mid by a bounded amount.

    If the flow over the sample window is perfectly balanced (tfi=0,
    streaks=0) the shift is legitimately zero. We accept that case
    rather than retry forever -- the test still proves the call path
    doesn't crash and stays bounded.
    """
    from app.exchange.okx_public_ws import OkxPublicStream
    from app.models import ToxicitySnapshot
    from app.quoting import compute_quote_decision
    from app.state import BotState

    s = _public_ws_settings()
    state = BotState(s)

    last_bbo: dict = {"mid": None, "bid": None, "ask": None}

    def _on_bbo(bbo) -> None:
        last_bbo["mid"] = bbo.mid_price
        last_bbo["bid"] = bbo.best_bid
        last_bbo["ask"] = bbo.best_ask

    stream = OkxPublicStream(s, state, _SYMBOL, _on_bbo)
    stream.set_contract_value(1.0)
    stream.start()

    try:
        deadline = time.time() + 15.0
        while time.time() < deadline:
            with state._lock:
                got_trade = len(state.recent_trades) > 0
            if got_trade and last_bbo["mid"]:
                break
            time.sleep(0.25)
    finally:
        stream.stop()

    assert last_bbo["mid"] is not None, "no BBO received -- can't run end-to-end test"
    snap = state.flow_score.snapshot()
    decision = compute_quote_decision(
        s,
        last_bbo["mid"],
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=ToxicitySnapshot(
            score=0.0,
            one_sided_fill_ratio=0.5,
            avg_adverse_markout_bps=0.0,
            vol_spike_ratio=0.0,
            hard_trigger=False,
            soft_trigger=False,
        ),
        flow_score_tfi_signed=snap.tfi_signed_normalised,
        flow_score_streak_buy=snap.streak_buy_count,
        flow_score_streak_sell=snap.streak_sell_count,
        flow_score_streak_window_prints=int(s.flow_score_streak_window_prints),
    )
    shift_bps = (decision.reservation_price - last_bbo["mid"]) / last_bbo["mid"] * 10_000
    print(
        f"\n  [end-to-end] mid={last_bbo['mid']} reservation={decision.reservation_price} "
        f"shift={shift_bps:+.3f}bps tfi={snap.tfi_signed_normalised:+.3f} "
        f"streak_buy={snap.streak_buy_count} streak_sell={snap.streak_sell_count}"
    )
    # Bound: at alpha=0.4, half_spread<=5 bps (we capped MAX at 30 here
    # but BASE=5), and clip=0.85 -> max |shift| = 0.4 * 2.5 * 0.85 = 0.85 bps.
    # Allow some slack for exact half_spread variation.
    assert abs(shift_bps) <= 2.0, (
        f"flow-score reservation shift {shift_bps:.3f} bps exceeds expected "
        f"bound -- composition / clip math may be wrong"
    )
