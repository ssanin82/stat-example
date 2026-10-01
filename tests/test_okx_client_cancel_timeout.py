"""Phase 0c regression — cancel ops get a 1.5 s read timeout (vs the
default 8 s) and a more aggressive retry policy (50/100/200/400 ms
vs 350/700/1400/2800 ms).

Mocks ``httpx.Client.request`` to sleep, then asserts:
  * cancel ops respect ``cancel_http_read_timeout_seconds``
  * non-cancel ops (place) still use the default ``read=8.0``
  * the cancel-specific ``RetryPolicy`` is threaded to
    ``exchange_call_with_retry`` for cancel ops only
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from app.exchange import okx_client as okx_client_module
from app.exchange.exchange_retry import RetryPolicy
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "EXCHANGE": "okx",
            "SYMBOL": "SUI-USDT-SWAP",
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "OKX_API_KEY": "k",
            "OKX_API_SECRET": "s",
            "OKX_API_PASSPHRASE": "p",
        }
    )


def _make_client_with_mocked_http():
    """Build an OkxClient with the bootstrap network call patched out
    and BOTH split HTTP clients (Phase 2, v1.3.108) replaced with
    MagicMocks ready for per-test inspection. We patch both because
    ``_request`` now picks ``_http_cancel`` for cancel ops and
    ``_http_place`` for everything else — leaving one as a real client
    would either leak a real network call or break tests that expect
    a recordable mock."""
    settings = _settings()
    with patch.object(
        okx_client_module.OkxClient,
        "_bootstrap_symbol_spec",
        return_value=(okx_client_module.FALLBACK_SYMBOL_SPEC, 1.0, None),
    ):
        client = okx_client_module.OkxClient(settings)
    # Close the real httpx Clients before replacing so we don't leak
    # connection pools across tests. ``close()`` is idempotent.
    client.close()
    # Place pool — current generous defaults so non-cancel ops still
    # see ``read=8.0`` when the code falls through to client baseline.
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_place.timeout = httpx.Timeout(
        connect=3.0, read=8.0, write=4.0, pool=8.0
    )
    # Cancel pool — tighter Phase 2 defaults. The per-request override
    # in ``_request`` (Phase 0c) supplies the read=1.5 cancel-specific
    # timeout on top, which is what the existing assertions verify.
    client._http_cancel = MagicMock(spec=httpx.Client)
    client._http_cancel.timeout = httpx.Timeout(
        connect=2.0, read=1.5, write=1.0, pool=0.2
    )
    return client


def test_cancel_op_uses_cancel_specific_read_timeout() -> None:
    """When the op is in ``_CANCEL_REST_OPS``, the per-request
    ``timeout`` kwarg on ``httpx.Client.request`` must carry
    ``read=1.5`` (the configured cancel timeout). Non-cancel ops use
    the client's default ``read=8.0``."""
    client = _make_client_with_mocked_http()

    # Make request() return a successful OKX envelope on both pools
    # so the retry path doesn't engage regardless of which client the
    # split routes us to.
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"code": "0", "msg": "", "data": []}
    response_mock.headers = {}
    client._http_cancel.request.return_value = response_mock
    client._http_place.request.return_value = response_mock

    # Issue a cancel — check the per-request timeout AND that the
    # cancel pool was used (Phase 2 routing).
    client.cancel_order("SUI-USDT-SWAP", oid=12345)
    assert client._http_cancel.request.called, (
        "cancel op must route through _http_cancel (Phase 2 split)"
    )
    assert not client._http_place.request.called, (
        "cancel op must NOT touch _http_place (would leak across pools)"
    )
    call_kwargs = client._http_cancel.request.call_args.kwargs
    timeout_arg = call_kwargs.get("timeout")
    assert isinstance(timeout_arg, httpx.Timeout), (
        f"cancel op should pass a per-request httpx.Timeout, got {timeout_arg!r}"
    )
    assert timeout_arg.read == 1.5, (
        f"cancel read timeout should be 1.5s (Phase 0c default), got {timeout_arg.read}"
    )

    # Reset + issue a NON-cancel call. We don't have a benign GET
    # handy that doesn't engage the response parser, so call
    # ``_request`` directly with a known non-cancel op name.
    client._http_cancel.request.reset_mock()
    client._http_place.request.reset_mock()
    client._http_place.request.return_value = response_mock
    client._request("query_order", "GET", "/api/v5/trade/order", signed=True)
    assert client._http_place.request.called, (
        "non-cancel op must route through _http_place (Phase 2 split)"
    )
    assert not client._http_cancel.request.called, (
        "non-cancel op must NOT touch _http_cancel"
    )
    call_kwargs2 = client._http_place.request.call_args.kwargs
    timeout_arg2 = call_kwargs2.get("timeout")
    # Non-cancel ops fall through to the place client's default
    # timeout (read=8.0 per the place-pool baseline).
    assert timeout_arg2 == client._http_place.timeout, (
        f"non-cancel op should pass the place client's default timeout "
        f"(read=8.0), got {timeout_arg2!r}"
    )


def test_cancel_op_uses_cancel_specific_retry_policy() -> None:
    """The retry path inside ``_request`` must call
    ``exchange_call_with_retry`` with a ``policy=RetryPolicy(...)`` whose
    ``max_attempts`` / ``base_seconds`` / ``cap_seconds`` match the
    cancel-specific settings, NOT the shared ``exchange_retry_*``
    defaults."""
    client = _make_client_with_mocked_http()

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"code": "0", "msg": "", "data": []}
    response_mock.headers = {}
    client._http_cancel.request.return_value = response_mock

    with patch.object(
        okx_client_module,
        "exchange_call_with_retry",
        wraps=okx_client_module.exchange_call_with_retry,
    ) as wrapped:
        client.cancel_order("SUI-USDT-SWAP", oid=99)
        assert wrapped.called
        kwargs = wrapped.call_args.kwargs
        policy = kwargs.get("policy")
        assert isinstance(policy, RetryPolicy), (
            f"cancel op must pass a RetryPolicy, got {policy!r}"
        )
        # Defaults set in app/config.py:
        assert policy.max_attempts == 4
        assert policy.base_seconds == 0.05
        assert policy.cap_seconds == 0.5


def test_non_cancel_op_passes_no_retry_policy() -> None:
    """Non-cancel ops must keep using the shared retry settings —
    confirmed by the ``policy`` kwarg being ``None`` (the default)."""
    client = _make_client_with_mocked_http()

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"code": "0", "msg": "", "data": []}
    response_mock.headers = {}
    client._http_place.request.return_value = response_mock

    with patch.object(
        okx_client_module,
        "exchange_call_with_retry",
        wraps=okx_client_module.exchange_call_with_retry,
    ) as wrapped:
        client._request(
            "query_order", "GET", "/api/v5/trade/order", signed=True
        )
        assert wrapped.called
        kwargs = wrapped.call_args.kwargs
        assert kwargs.get("policy") is None, (
            f"non-cancel op should pass policy=None, got "
            f"{kwargs.get('policy')!r}"
        )


def test_retry_policy_overrides_shared_defaults() -> None:
    """Direct unit-test on ``exchange_call_with_retry``: when a
    ``policy`` is provided, the loop uses its values, not the
    settings'. This protects against a future refactor that adds new
    settings-driven overrides masking the policy kwarg."""
    from app.exchange.exchange_retry import exchange_call_with_retry

    settings = _settings()
    # The shared default is max_attempts=4. Try with policy max=2.
    call_count = {"n": 0}

    def fn():
        call_count["n"] += 1
        # Always raise a transient error so the retry loop iterates.
        raise TimeoutError("transient")

    policy = RetryPolicy(max_attempts=2, base_seconds=0.001, cap_seconds=0.002)
    try:
        exchange_call_with_retry("test_op", fn, settings, policy=policy)
    except TimeoutError:
        pass  # expected after retries exhausted
    assert call_count["n"] == 2, (
        f"policy max_attempts=2 should yield exactly 2 calls, got {call_count['n']}"
    )
