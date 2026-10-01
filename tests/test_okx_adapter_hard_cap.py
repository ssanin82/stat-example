"""OKX adapter pre-flight notional hard cap tests.

The 2026-05-06 incident produced fills of 2672 SUI ($2700 notional)
against a $20 configured cap. Root cause was the soft-flatten worker
bypassing every upstream cap. The OKX adapter now refuses any
order whose notional exceeds
``MAX_ORDER_NOTIONAL_USD * MAX_ORDER_NOTIONAL_HARD_MULTIPLIER`` --
last-resort defense regardless of caller bug class.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _build_client_with_settings(
    *,
    max_order_notional_usd: float = 10.0,
    hard_multiplier: float = 2.0,
):
    """Construct an OkxClient with mocked HTTP layer so we can drive
    ``_place_order`` without a real network call. ``_request`` is
    monkey-patched to capture / control."""
    from app.exchange.okx_client import OkxClient

    settings = MagicMock()
    settings.exchange = "okx"
    settings.symbol = "TEST-USDT-SWAP"
    settings.max_order_notional_usd = max_order_notional_usd
    settings.max_order_notional_hard_multiplier = hard_multiplier
    settings.okx_api_key = "k"
    settings.okx_api_secret = "s"
    settings.okx_api_passphrase = "p"
    settings.okx_rest_url = "https://example.invalid"
    # 1.3.111 Phase 4b: keep the WS layer out of these HTTP-only
    # tests. ``_place_order`` now routes through ``_try_ws_place``
    # first; the umbrella ``action_ws_enabled=False`` short-circuits
    # the WS branch in ``_get_action_ws`` before any other state
    # access happens, so the bare-bones ``OkxClient.__new__`` instance
    # below stays valid (no ``_api_key`` access via has_write_access,
    # no ``_action_ws_lock`` needed).
    settings.action_ws_enabled = False
    settings.okx_action_ws_cancel_enabled = False
    settings.okx_action_ws_place_enabled = False

    # Build without symbol-spec lookup (skips network call).
    client = OkxClient.__new__(OkxClient)
    client._settings = settings
    client._symbol = "TEST-USDT-SWAP"
    client._contract_value = 1.0
    # 1.3.123 Phase 4a v2: ``_decorate_with_inst_id_code`` reads
    # ``self._inst_id_code`` even on the HTTP-only path (via place_order
    # WS args decoration). Initialise to None so the decoration is a no-op.
    client._inst_id_code = None
    # Phase 4b: ``_try_ws_place`` writes ``last_exchange_transport_mode``
    # on the HTTP-fallback path. Initialize so the assignment doesn't
    # silently fail under ``__slots__`` if that's ever added.
    client.last_exchange_transport_mode = "http"
    client._request_capture: list[dict] = []

    def _capture(op, method, path, *, params=None, body=None, signed=False, retry=False):
        client._request_capture.append(
            {"op": op, "method": method, "path": path, "body": body}
        )
        return {"code": "0", "msg": "", "data": [{"sCode": "0", "ordId": "12345"}]}

    client._request = _capture
    return client


def test_adapter_refuses_order_above_hard_cap_long() -> None:
    """50 SUI × $1.0 = $50; cap = $10 × 2x = $20. Refused."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=2.0
    )
    with pytest.raises(RuntimeError, match="OKX_HARD_NOTIONAL_CAP"):
        client._place_order(
            symbol="TEST-USDT-SWAP",
            is_buy=True,
            sz_base=50.0,
            limit_px=1.0,
            post_only=True,
            reduce_only=False,
            ioc=False,
            order_type="post_only",
            client_order_id=None,
        )
    # No HTTP call leaked to ``_request``.
    assert client._request_capture == []


def test_adapter_refuses_order_at_3x_cap() -> None:
    """3x cap = $30; refuse."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=2.0
    )
    with pytest.raises(RuntimeError):
        client._place_order(
            symbol="TEST-USDT-SWAP",
            is_buy=False,
            sz_base=30.0,
            limit_px=1.0,
            post_only=True,
            reduce_only=False,
            ioc=False,
            order_type="post_only",
            client_order_id=None,
        )


def test_adapter_passes_order_at_cap() -> None:
    """20 SUI × $1.0 = $20 = exactly hard_cap. Allowed."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=2.0
    )
    result = client._place_order(
        symbol="TEST-USDT-SWAP",
        is_buy=True,
        sz_base=20.0,
        limit_px=1.0,
        post_only=True,
        reduce_only=False,
        ioc=False,
        order_type="post_only",
        client_order_id=None,
    )
    assert result.get("code") == "0"
    assert len(client._request_capture) == 1


def test_adapter_passes_order_below_cap() -> None:
    """8 SUI × $1.0 = $8. Far under cap."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=2.0
    )
    client._place_order(
        symbol="TEST-USDT-SWAP",
        is_buy=True,
        sz_base=8.0,
        limit_px=1.0,
        post_only=True,
        reduce_only=False,
        ioc=False,
        order_type="post_only",
        client_order_id=None,
    )
    assert len(client._request_capture) == 1


def test_adapter_2026_05_06_incident_replay_blocked() -> None:
    """The actual incident size: 2672 SUI × $1.007 = $2691. Must be
    blocked even with generous multiplier."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=2.0
    )
    with pytest.raises(RuntimeError, match="OKX_HARD_NOTIONAL_CAP"):
        client._place_order(
            symbol="TEST-USDT-SWAP",
            is_buy=True,
            sz_base=2672.0,
            limit_px=1.007,
            post_only=True,
            reduce_only=False,
            ioc=False,
            order_type="post_only",
            client_order_id=None,
        )


def test_adapter_strict_cap_when_multiplier_one() -> None:
    """Operator can set hard_multiplier=1.0 to make
    MAX_ORDER_NOTIONAL_USD an absolute cap."""
    client = _build_client_with_settings(
        max_order_notional_usd=10.0, hard_multiplier=1.0
    )
    # 10.5 SUI = $10.5, marginally over $10 cap → refused.
    with pytest.raises(RuntimeError):
        client._place_order(
            symbol="TEST-USDT-SWAP",
            is_buy=True,
            sz_base=10.5,
            limit_px=1.0,
            post_only=True,
            reduce_only=False,
            ioc=False,
            order_type="post_only",
            client_order_id=None,
        )
    # 10 exactly → allowed.
    client._place_order(
        symbol="TEST-USDT-SWAP",
        is_buy=True,
        sz_base=10.0,
        limit_px=1.0,
        post_only=True,
        reduce_only=False,
        ioc=False,
        order_type="post_only",
        client_order_id=None,
    )
    assert len(client._request_capture) == 1


def test_adapter_disabled_when_cap_is_zero() -> None:
    """Defensive: cap=0 disables the check entirely (e.g. operator
    intentionally running unlimited)."""
    client = _build_client_with_settings(max_order_notional_usd=0.0)
    client._place_order(
        symbol="TEST-USDT-SWAP",
        is_buy=True,
        sz_base=10_000.0,
        limit_px=1.0,
        post_only=True,
        reduce_only=False,
        ioc=False,
        order_type="post_only",
        client_order_id=None,
    )
    assert len(client._request_capture) == 1


def test_adapter_propagates_reduce_only_through_place_order() -> None:
    """The new ``reduce_only`` kwarg on ``place_post_only_limit`` must
    reach the OKX request body as ``reduceOnly: true``."""
    client = _build_client_with_settings()
    client.has_write_access = lambda: True
    # Workaround: ``has_write_access`` is normally a method; we
    # monkey-patched the instance with a lambda above.
    client.place_post_only_limit(
        "TEST-USDT-SWAP",
        is_buy=True,
        sz=5.0,
        limit_px=1.0,
        client_order_id="test-cloid",
        reduce_only=True,
    )
    assert len(client._request_capture) == 1
    body = client._request_capture[0]["body"]
    assert body.get("reduceOnly") is True


def test_adapter_does_not_set_reduce_only_when_default_false() -> None:
    client = _build_client_with_settings()
    client.has_write_access = lambda: True
    client.place_post_only_limit(
        "TEST-USDT-SWAP",
        is_buy=True,
        sz=5.0,
        limit_px=1.0,
        client_order_id="test-cloid",
    )
    body = client._request_capture[0]["body"]
    # Body should NOT contain reduceOnly when False (omitted).
    assert "reduceOnly" not in body or body["reduceOnly"] is False
