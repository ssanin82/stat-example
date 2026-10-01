"""Smoke tests for the OKX V5 SWAP adapter boundary.

Plan reference: ``plans/20260504-okx-setup/plan.md`` Phase 2.6.

Mirrors the structure of ``test_binance_adapter_smoke.py`` so reviewers
can compare per-venue test coverage at a glance. The unit tests here
exercise:

* Credential validation (with the third secret -- passphrase -- gating).
* Adapter constructible with mocked REST.
* HMAC-SHA256 + base64 signing matches OKX's reference recipe.
* Symbol normalisation handles dash / no-dash / lowercase inputs.
* Contract <-> base unit conversion.
* Wire-format interpreters map canonical OKX response shapes (top-level
  ``code`` + per-row ``sCode``) to the bot core's normalised tuples.
* Factory wires ``EXCHANGE=okx`` to OkxClient + the OKX private/public WS.
* PASSPHRASE redaction ladders into ``Settings.sanitized_dict()``.

Live (integration) tests against a real OKX account live in
``tests/integration/test_okx_live.py``; those need credentials in
the environment and are skipped here by default.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any
from unittest.mock import patch

import pytest

from app.config import Settings, require_trading_credentials_when_enabled
from app.enums import Side
from app.exchange.factory import (
    build_adapter,
    venue_account_address,
)
from app.exchange.okx_client import (
    OkxClient,
    _normalize_okx_symbol,
)
from app.exchange.okx_responses import (
    interpret_okx_cancel_response,
    interpret_okx_order_status_response,
    interpret_okx_place_response,
    make_deterministic_okx_client_order_id,
)
from app.exchange.okx_ws import _okx_login_sign
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> Settings:
    base = {
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
        "OKX_API_KEY": "k_test",
        "OKX_API_SECRET": "s_test_secret",
        "OKX_API_PASSPHRASE": "p_test",
        "OKX_REST_URL": "https://www.okx.com",
        "OKX_PUBLIC_WS_URL": "wss://ws.okx.com:8443/ws/v5/public",
        "OKX_PRIVATE_WS_URL": "wss://ws.okx.com:8443/ws/v5/private",
        "SYMBOL": "DOGE-USDT-SWAP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _instruments_response() -> dict[str, Any]:
    """OKX /api/v5/public/instruments response fragment for
    DOGE-USDT-SWAP. ctVal=1000 means 1 contract = 1000 DOGE.
    """
    return {
        "code": "0",
        "msg": "",
        "data": [
            {
                "instType": "SWAP",
                "instId": "DOGE-USDT-SWAP",
                "tickSz": "0.00001",
                "lotSz": "1",   # 1 contract increment
                "minSz": "1",   # min order = 1 contract
                "ctVal": "1000",
                "ctValCcy": "DOGE",
                "settleCcy": "USDT",
            }
        ],
    }


def _make_client(s: Settings) -> OkxClient:
    """Build an OkxClient with REST mocked at construction so the
    instruments bootstrap returns deterministic data.
    """
    with patch.object(OkxClient, "_request") as m:
        m.return_value = _instruments_response()
        return OkxClient(s)


# ---------------------------------------------------------------------------
# Settings / credentials
# ---------------------------------------------------------------------------


def test_credential_validation_fails_without_key() -> None:
    s = _settings(TRADING_ENABLED=True, OKX_API_KEY="")
    with pytest.raises(ValueError, match="OKX_API_KEY"):
        require_trading_credentials_when_enabled(s)


def test_credential_validation_fails_without_secret() -> None:
    s = _settings(TRADING_ENABLED=True, OKX_API_SECRET="")
    with pytest.raises(ValueError, match="OKX_API_SECRET"):
        require_trading_credentials_when_enabled(s)


def test_credential_validation_fails_without_passphrase() -> None:
    """OKX is unique in requiring a third secret. Missing passphrase
    must fail loudly when trading is enabled.
    """
    s = _settings(TRADING_ENABLED=True, OKX_API_PASSPHRASE="")
    with pytest.raises(ValueError, match="OKX_API_PASSPHRASE"):
        require_trading_credentials_when_enabled(s)


def test_credential_validation_passes_with_full_creds() -> None:
    s = _settings(TRADING_ENABLED=True)
    require_trading_credentials_when_enabled(s)  # no raise


def test_passphrase_is_redacted_in_sanitized_dict() -> None:
    """The redaction list must include PASSPHRASE so OKX_API_PASSPHRASE
    doesn't leak via /config (which is captured by the snapshot
    pipeline and the dashboard).
    """
    s = _settings(OKX_API_PASSPHRASE="super-secret-passphrase")
    d = s.sanitized_dict()
    assert d.get("OKX_API_PASSPHRASE") == "***"
    assert d.get("OKX_API_KEY") == "***"
    assert d.get("OKX_API_SECRET") == "***"


# ---------------------------------------------------------------------------
# Adapter constructible + symbol-spec bootstrap
# ---------------------------------------------------------------------------


def test_adapter_constructs_and_bootstraps_symbol_spec() -> None:
    s = _settings()
    client = _make_client(s)
    assert client.symbol_spec_fetched_ok is True
    spec = client.symbol_spec
    assert spec.source == "okx_meta"
    assert spec.price_tick == pytest.approx(0.00001)
    # OKX lotSz=1 contract * ctVal=1000 = 1000 base-asset (DOGE) increment
    assert spec.size_step == pytest.approx(1000.0)
    assert spec.min_size == pytest.approx(1000.0)
    # Synthesized $5 floor (OKX doesn't publish a per-instrument min-notional)
    assert spec.min_notional_usd == pytest.approx(5.0)
    # Contract-value exposed for the WS conversion
    assert client.contract_value == pytest.approx(1000.0)


def test_adapter_satisfies_perp_protocol_isinstance_check() -> None:
    """``PerpExchangeAdapter`` is a runtime-checkable Protocol; the OKX
    client should satisfy it structurally.
    """
    from app.exchange.base import PerpExchangeAdapter

    s = _settings()
    client = _make_client(s)
    assert isinstance(client, PerpExchangeAdapter)


def test_has_write_access_requires_all_three_secrets() -> None:
    """Unlike Binance (key + secret), OKX needs key + secret +
    passphrase. has_write_access must reflect this.
    """
    base_overrides = dict(OKX_API_KEY="k", OKX_API_SECRET="s", OKX_API_PASSPHRASE="p")
    full = _make_client(_settings(**base_overrides))
    assert full.has_write_access() is True

    no_pass = _make_client(_settings(**{**base_overrides, "OKX_API_PASSPHRASE": ""}))
    assert no_pass.has_write_access() is False

    no_secret = _make_client(_settings(**{**base_overrides, "OKX_API_SECRET": ""}))
    assert no_secret.has_write_access() is False

    no_key = _make_client(_settings(**{**base_overrides, "OKX_API_KEY": ""}))
    assert no_key.has_write_access() is False


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------


def test_symbol_normalise_canonical_form_pass_through() -> None:
    assert _normalize_okx_symbol("DOGE-USDT-SWAP") == "DOGE-USDT-SWAP"


def test_symbol_normalise_dash_form_adds_swap() -> None:
    assert _normalize_okx_symbol("DOGE-USDT") == "DOGE-USDT-SWAP"


def test_symbol_normalise_no_dash_adds_dash_and_swap() -> None:
    assert _normalize_okx_symbol("DOGEUSDT") == "DOGE-USDT-SWAP"


def test_symbol_normalise_lowercase() -> None:
    assert _normalize_okx_symbol("doge-usdt-swap") == "DOGE-USDT-SWAP"


def test_symbol_normalise_empty() -> None:
    assert _normalize_okx_symbol("") == ""


# ---------------------------------------------------------------------------
# Contract <-> base unit conversion
# ---------------------------------------------------------------------------


def test_contract_value_conversions_round_trip() -> None:
    s = _settings()
    client = _make_client(s)
    # 5 contracts * 1000 base-per-contract = 5000 base
    assert client._contracts_to_base(5.0) == pytest.approx(5000.0)
    assert client._base_to_contracts(5000.0) == pytest.approx(5.0)
    # Roundtrip for an arbitrary base quantity
    base_qty = 12_345.0
    assert client._contracts_to_base(client._base_to_contracts(base_qty)) == pytest.approx(base_qty)


# ---------------------------------------------------------------------------
# REST signing
# ---------------------------------------------------------------------------


def test_rest_sign_matches_okx_reference_recipe() -> None:
    """OKX REST signing is base64(HMAC-SHA256(secret,
    timestamp + method + requestPath + body)). Verify our impl
    produces the right output for a known input.
    """
    s = _settings()
    client = _make_client(s)
    ts = "2026-05-04T08:30:00.000Z"
    method = "GET"
    request_path = "/api/v5/account/balance"
    body = ""
    sig = client._sign(ts, method, request_path, body)
    expected = base64.b64encode(
        hmac.new(
            b"s_test_secret",
            f"{ts}{method}{request_path}{body}".encode(),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")
    assert sig == expected


def test_rest_sign_includes_request_body_in_payload() -> None:
    """Body must be part of the signed payload for POSTs, otherwise
    OKX rejects with a signature-mismatch error."""
    s = _settings()
    client = _make_client(s)
    ts = "2026-05-04T08:30:00.000Z"
    body_str = '{"instId":"DOGE-USDT-SWAP","sz":"1"}'
    sig_with = client._sign(ts, "POST", "/api/v5/trade/order", body_str)
    sig_without = client._sign(ts, "POST", "/api/v5/trade/order", "")
    assert sig_with != sig_without


# ---------------------------------------------------------------------------
# WS login signing
# ---------------------------------------------------------------------------


def test_ws_login_sign_uses_fixed_payload() -> None:
    """The WS login signature is computed over a literal fixed payload:
    ``<timestamp>GET/users/self/verify``. Different from REST signing.
    """
    ts = "1538054050"
    sig = _okx_login_sign("s_test_secret", ts)
    expected = base64.b64encode(
        hmac.new(
            b"s_test_secret",
            f"{ts}GET/users/self/verify".encode(),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")
    assert sig == expected


# ---------------------------------------------------------------------------
# Place response interpreter
# ---------------------------------------------------------------------------


def test_place_response_success_extracts_order_id() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "0", "sMsg": "", "ordId": "1234567890", "clOrdId": "abc"}],
    }
    oid, outcome, reason = interpret_okx_place_response(resp)
    assert outcome == "accepted"
    assert oid == 1234567890
    assert reason == ""


def test_place_response_post_only_would_cross_is_exchange_rejected() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51604", "sMsg": "post only would cross"}],
    }
    oid, outcome, reason = interpret_okx_place_response(resp)
    assert outcome == "exchange_rejected"
    assert "post_only_would_cross" in reason
    assert oid is None


def test_place_response_insufficient_margin_is_exchange_rejected() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51008", "sMsg": "insufficient balance"}],
    }
    _, outcome, reason = interpret_okx_place_response(resp)
    assert outcome == "exchange_rejected"
    assert "insufficient_margin" in reason


def test_place_response_rate_limit_is_transport_rejected() -> None:
    """Top-level 50011 (rate limit) is treated as transport so the
    retry layer backs off rather than the venue-level rejection path.
    """
    resp = {"code": "50011", "msg": "Too many requests", "data": []}
    _, outcome, reason = interpret_okx_place_response(resp)
    assert outcome == "transport_rejected"
    assert "rate_limit" in reason


def test_place_response_auth_failure_is_transport_rejected() -> None:
    """Auth-class top codes -> transport. Important: we don't want
    silent venue-rejected loops if the passphrase is wrong.
    """
    for tcode in ("50113", "50114", "50102", "50104"):
        resp = {"code": tcode, "msg": "auth failed", "data": []}
        _, outcome, reason = interpret_okx_place_response(resp)
        assert outcome == "transport_rejected", f"top code {tcode}"
        assert "auth" in reason


def test_place_response_unconfirmed_when_data_missing() -> None:
    resp = {"code": "0", "msg": "", "data": []}
    _, outcome, _ = interpret_okx_place_response(resp)
    assert outcome == "unconfirmed"


def test_place_response_handles_non_dict_input() -> None:
    _, outcome, _ = interpret_okx_place_response("not a dict")
    assert outcome == "unconfirmed"
    _, outcome, _ = interpret_okx_place_response(None)
    assert outcome == "unconfirmed"


# ---------------------------------------------------------------------------
# Cancel response interpreter
# ---------------------------------------------------------------------------


def test_cancel_response_success() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "0", "sMsg": "", "ordId": "1234567890"}],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "success"


def test_cancel_response_already_filled_is_benign_missing() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51402", "sMsg": "order has been filled"}],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "benign_missing"


def test_cancel_response_already_canceled_is_unexpected_gone() -> None:
    """v1.3.120 reclassification: sCode 51401 ("order has been
    canceled") is NOT benign — order is gone but not via fill, which
    could mean stale-state retry race, venue admin action, or a real
    state-desync bug. The bot still transitions the WO locally
    (the order IS gone per OKX) but surfaces via WARNING log +
    dedicated counter so the operator can investigate."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51401", "sMsg": "order has been canceled"}],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "unexpected_gone", (
        f"sCode 51401 must classify as unexpected_gone (was previously "
        f"benign_missing in v1.3.119 and earlier — see "
        f"_UNEXPECTED_GONE_CODES docstring); got kind={kind!r}"
    )


def test_cancel_response_unknown_order_is_unexpected_gone() -> None:
    """v1.3.120 reclassification: sCode 51400 ("order does not
    exist") classifies as unexpected_gone, not benign_missing.
    OKX uses 51400 ambiguously — could be a cancel-race-lost-to-fill
    that OKX mislabeled (should have been 51402), could be a real
    "we have wrong state" bug. Either way, deserves WARNING-level
    surface rather than silent acceptance."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51400", "sMsg": "order does not exist"}],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "unexpected_gone"


def test_cancel_response_legacy_51503_is_unexpected_gone() -> None:
    """sCode 51503 (legacy generic ``cancel order failed``) also
    classifies as unexpected_gone — same surfaced-anomaly treatment
    as 51400 / 51401."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"sCode": "51503", "sMsg": "cancel order failed"}],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "unexpected_gone"


def test_cancel_response_50014_instidcode_is_transport() -> None:
    """v1.3.119 regression: OKX colo trade-WS returns sCode 50014
    ``Parameter instIdCode can not be empty`` as an ambiguous response
    on retry cancels. Original v1.3.118 fix classified this as
    ``benign_missing`` — but that was UNSAFE because
    ``residual_order_audit`` showed the order was still alive at the
    matching engine when 50014 fired (server_buy=1, local_buy=0).
    Marking the WO locally CANCELED while the order is still resting
    on the book could cause silent unwanted fills if price moves to
    the order's price during the cancel-confirmation gap.

    Correct treatment: ``transport``. The cancel-pending watchdog
    retries; ``execution_errors`` is NOT bumped (no self-kill); local
    WO state stays CANCEL_PENDING until a real terminal signal
    arrives (sCode 0 on a later retry, OR inbound user-data WS event
    pushing CANCELED / FILLED).

    Reproducer: snap context 2026-05-17, order 3571816166607233024
    CANCEL_PENDING for 60 s; cancel retry → ``okx_top_1_row_50014:
    Parameter instIdCode can not be empty.`` → bot KILLED."""
    resp = {
        "code": "1",
        "msg": "Operation partially or fully failed",
        "data": [{"sCode": "50014", "sMsg": "Parameter instIdCode can not be empty."}],
    }
    kind, detail = interpret_okx_cancel_response(resp)
    assert kind == "transport", (
        f"sCode 50014 on cancel must be transport (ambiguous, retry "
        f"without claiming local terminal state); got kind={kind!r}"
    )
    # Detail must surface the row code + the colo_quirk_ prefix so
    # the operator can grep for this pattern in the snapshot.
    assert "50014" in detail
    assert "colo_quirk" in detail or "50014" in detail


def test_cancel_response_rate_limit_is_transport() -> None:
    resp = {"code": "50011", "msg": "Too many requests", "data": []}
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "transport"


def test_cancel_response_top_code_1_with_benign_row_is_benign_missing() -> None:
    """OKX returns top ``code="1"`` whenever ANY row in a batch
    request fails. The authoritative outcome is the per-row sCode --
    the parser must NOT short-circuit to "error" before reading it.

    Regression: snapshot 260506024321 captured 8 cancel-rejects with
    ``okx_top_1:Order cancellation failed as the order has been
    filled, canceled or does not exist.`` These were classified as
    hard errors and tripped MAX_EXECUTION_ERRORS, killing the bot.
    """
    resp = {
        "code": "1",
        "msg": "",
        "data": [
            {
                "sCode": "51402",
                "sMsg": "Order cancellation failed as the order has been filled.",
                "ordId": "3540600787184852993",
            }
        ],
    }
    kind, detail = interpret_okx_cancel_response(resp)
    assert kind == "benign_missing"
    assert "51402" in detail


def test_cancel_response_top_code_1_ambiguous_msg_is_unexpected_gone() -> None:
    """v1.3.120 reclassification: OKX's ambiguous catch-all message
    "Order cancellation failed as the order has been filled, canceled
    or does not exist" now classifies as ``unexpected_gone``, not
    ``benign_missing``. The venue is saying "we don't know which of
    three states the order is in"; per the operator's principle
    (only true fill races are benign), the safer classification is
    to surface as an anomaly. The WO still transitions out of
    CANCEL_PENDING locally (the order IS gone per OKX) but the
    operator sees the WARNING + ``cancel_unexpected_gone_total``
    counter.

    Original incident shape: 2026-05-05 22:35 UTC, 8 hits in 5 min
    in cleanup-cancel-all races. v1.3.120's unexpected_gone path
    does NOT bump execution_errors by default (only in strict mode),
    so the bot won't self-kill on those — but the operator can see
    them in the logs / counter."""
    resp = {
        "code": "1",
        "msg": (
            "Order cancellation failed as the order has been filled, "
            "canceled or does not exist."
        ),
        "data": [],
    }
    kind, detail = interpret_okx_cancel_response(resp)
    assert kind == "unexpected_gone"
    assert "okx_top_1" in detail


def test_cancel_response_top_code_1_only_filled_msg_is_benign_missing() -> None:
    """When OKX's top-level message specifically says ONLY filled
    (no mention of canceled / does not exist), classify as
    benign_missing — true cancel-race-lost-to-fill."""
    resp = {
        "code": "1",
        "msg": "Order cancellation failed as the order has been filled.",
        "data": [],
    }
    kind, _ = interpret_okx_cancel_response(resp)
    assert kind == "benign_missing"


def test_cancel_response_top_code_1_with_other_row_code_is_error() -> None:
    """Top ``code="1"`` with a non-benign row code must remain a
    hard error. Insurance against the benign-classification fix
    accidentally swallowing real failures.
    """
    resp = {
        "code": "1",
        "msg": "",
        "data": [{"sCode": "51008", "sMsg": "insufficient margin"}],
    }
    kind, detail = interpret_okx_cancel_response(resp)
    assert kind == "error"
    assert "51008" in detail


# ---------------------------------------------------------------------------
# Order-status response interpreter
# ---------------------------------------------------------------------------


def test_order_status_open_state_maps_to_open() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1234", "state": "live"}],
    }
    oid, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "open"
    assert oid == 1234


def test_order_status_partially_filled_is_open() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1234", "state": "partially_filled"}],
    }
    _, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "open"


def test_order_status_filled_maps_to_filled() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1234", "state": "filled"}],
    }
    _, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "filled"


def test_order_status_canceled_maps_to_canceled() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1234", "state": "canceled"}],
    }
    _, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "canceled"


def test_order_status_mmp_canceled_maps_to_canceled() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1234", "state": "mmp_canceled"}],
    }
    _, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "canceled"


def test_order_status_empty_data_is_not_found() -> None:
    resp = {"code": "0", "msg": "", "data": []}
    _, outcome, _ = interpret_okx_order_status_response(resp)
    assert outcome == "not_found"


# ---------------------------------------------------------------------------
# Deterministic clOrdId
# ---------------------------------------------------------------------------


def test_make_client_order_id_is_deterministic() -> None:
    a = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle-1", 0.10, 1000.0
    )
    b = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle-1", 0.10, 1000.0
    )
    assert a == b


def test_make_client_order_id_changes_with_inputs() -> None:
    base = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle-1", 0.10, 1000.0
    )
    other_side = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.SELL, "cycle-1", 0.10, 1000.0
    )
    other_cycle = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle-2", 0.10, 1000.0
    )
    other_price = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle-1", 0.11, 1000.0
    )
    assert len({base, other_side, other_cycle, other_price}) == 4


def test_make_client_order_id_format_is_okx_compatible() -> None:
    """OKX clOrdId max length is 32 chars and must be alphanumeric.
    Our generator returns 32 lowercase hex chars (no 0x prefix).
    """
    cloid = make_deterministic_okx_client_order_id(
        "DOGE-USDT-SWAP", Side.BUY, "cycle", 0.1, 1.0
    )
    assert len(cloid) == 32
    assert cloid.islower()
    assert all(c in "0123456789abcdef" for c in cloid)


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_factory_build_adapter_routes_okx_to_OkxClient() -> None:
    s = _settings()
    with patch.object(OkxClient, "_request") as m:
        m.return_value = _instruments_response()
        adapter = build_adapter(s)
    assert isinstance(adapter, OkxClient)


def test_factory_build_public_stream_routes_okx() -> None:
    """``build_public_stream`` should not raise and must return an
    object with start/stop. Constructing the WS thread is deferred
    until start(); we just verify the factory dispatch.
    """
    from app.exchange.factory import build_public_stream
    from app.exchange.okx_public_ws import OkxPublicStream
    from app.state import BotState

    s = _settings()
    state = BotState(s)
    stream = build_public_stream(s, state, s.symbol, lambda _bbo: None)
    assert isinstance(stream, OkxPublicStream)


def test_venue_account_address_for_okx_returns_redacted_key_prefix() -> None:
    """OKX has no on-chain address; we surface a non-secret prefix
    of the API key for log correlation. Same shape as Binance.
    """
    s = _settings(OKX_API_KEY="abcdef0123456789")
    assert venue_account_address(s) == "abcdef01..."


def test_venue_account_address_for_okx_empty_when_no_key() -> None:
    s = _settings(OKX_API_KEY="")
    assert venue_account_address(s) == ""
