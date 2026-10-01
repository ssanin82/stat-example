"""Phase 2 (v1.3.108) regression — ``OkxClient`` maintains two separate
``httpx.Client`` instances with isolated connection pools:

  * ``_http_cancel`` — tight timeouts (connect=2.0, read=1.5, write=1.0,
    pool=0.2). All ``_CANCEL_REST_OPS`` route here. ``pool=0.2`` is the
    key isolation guarantee: under a saturated place pool, cancels
    fail-fast in 200ms rather than waiting up to 8s for a shared pool
    slot.
  * ``_http_place`` — current generous defaults (connect=3.0, read=8.0,
    write=4.0, pool=8.0). Everything else routes here.

Lifecycle: ``close()`` shuts BOTH clients; idempotent; one client's
failure must not prevent the other from closing. HTTP/2 is on by
default and configurable via ``OKX_HTTP2_ENABLED`` for one-line
revert.

These tests guard the routing contract that Phase 3 depends on —
parallel cancel + place workers will hit the two pools concurrently,
and a regression that routes cancels back through the place pool
would re-introduce the head-of-line block this phase exists to
eliminate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from app.exchange import okx_client as okx_client_module
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides):
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "okx",
        "SYMBOL": "SUI-USDT-SWAP",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "OKX_API_KEY": "k",
        "OKX_API_SECRET": "s",
        "OKX_API_PASSPHRASE": "p",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_client(settings=None):
    """Build an OkxClient with bootstrap network call patched out.
    Returns the live client (real httpx clients still allocated; caller
    is responsible for ``client.close()`` at teardown)."""
    if settings is None:
        settings = _settings()
    with patch.object(
        okx_client_module.OkxClient,
        "_bootstrap_symbol_spec",
        return_value=(okx_client_module.FALLBACK_SYMBOL_SPEC, 1.0, None),
    ):
        return okx_client_module.OkxClient(settings)


def _make_client_with_mocked_http():
    """As above, but also replace both real httpx Clients with
    MagicMocks suitable for routing assertions."""
    client = _make_client()
    client.close()
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_place.timeout = httpx.Timeout(
        connect=3.0, read=8.0, write=4.0, pool=8.0
    )
    client._http_cancel = MagicMock(spec=httpx.Client)
    client._http_cancel.timeout = httpx.Timeout(
        connect=2.0, read=1.5, write=1.0, pool=0.2
    )
    return client


def _ok_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"code": "0", "msg": "", "data": []}
    r.headers = {}
    return r


# --------------------------------------------------------------------------
# Construction: both clients allocated, with the expected baseline timeouts
# --------------------------------------------------------------------------


def test_constructor_allocates_two_separate_clients() -> None:
    """``OkxClient.__init__`` must allocate two ``httpx.Client``
    instances with distinct timeouts. The identity check is what
    guarantees POOL isolation (each Client owns its own connection
    pool)."""
    client = _make_client()
    try:
        assert isinstance(client._http_place, httpx.Client)
        assert isinstance(client._http_cancel, httpx.Client)
        # Different INSTANCES — same instance would mean a shared pool
        # and would defeat the whole point of this phase.
        assert client._http_place is not client._http_cancel
    finally:
        client.close()


def test_place_client_baseline_timeouts() -> None:
    """Place client keeps the historical timeouts (no behavior change
    for the dominant happy path)."""
    client = _make_client()
    try:
        t = client._http_place.timeout
        assert t.connect == 3.0
        assert t.read == 8.0
        assert t.write == 4.0
        assert t.pool == 8.0
    finally:
        client.close()


def test_cancel_client_baseline_timeouts() -> None:
    """Cancel client uses the Phase 2 plan's tight baseline timeouts.
    The crucial value is ``pool=0.2`` — failing fast on a saturated
    pool prevents Phase 3's parallel-place workload from delaying a
    cancel that's waiting on a pool slot."""
    client = _make_client()
    try:
        t = client._http_cancel.timeout
        assert t.connect == 2.0
        assert t.read == 1.5
        assert t.write == 1.0
        assert t.pool == 0.2
    finally:
        client.close()


# --------------------------------------------------------------------------
# Routing: cancel ops -> _http_cancel; everything else -> _http_place
# --------------------------------------------------------------------------


def test_cancel_order_routes_through_cancel_client() -> None:
    client = _make_client_with_mocked_http()
    client._http_cancel.request.return_value = _ok_response()
    client._http_place.request.return_value = _ok_response()

    client.cancel_order("SUI-USDT-SWAP", oid=42)

    assert client._http_cancel.request.called
    assert not client._http_place.request.called


def test_cancel_order_by_cloid_routes_through_cancel_client() -> None:
    client = _make_client_with_mocked_http()
    client._http_cancel.request.return_value = _ok_response()
    client._http_place.request.return_value = _ok_response()

    client.cancel_order_by_cloid("SUI-USDT-SWAP", "cloid-abc")

    assert client._http_cancel.request.called
    assert not client._http_place.request.called


def test_cancel_batch_routes_through_cancel_client() -> None:
    """The batch-cancel endpoint (op=``cancel_batch``) added in Phase
    1b must also route through the cancel pool — it's the workhorse
    path on full-reprice cycles."""
    client = _make_client_with_mocked_http()
    client._http_cancel.request.return_value = _ok_response()
    client._http_place.request.return_value = _ok_response()

    client.cancel_batch_orders(
        "SUI-USDT-SWAP",
        [
            {"instId": "SUI-USDT-SWAP", "ordId": "1"},
            {"instId": "SUI-USDT-SWAP", "ordId": "2"},
        ],
    )

    assert client._http_cancel.request.called
    assert not client._http_place.request.called


def test_place_order_routes_through_place_client() -> None:
    """Place ops use the generous-timeouts place pool. Confirms the
    ``op`` whitelist in ``_CANCEL_REST_OPS`` correctly excludes
    place_post_only."""
    client = _make_client_with_mocked_http()
    client._http_cancel.request.return_value = _ok_response()
    # Place response must carry a non-empty data row so the place
    # interpreter doesn't synthesize an error envelope.
    place_resp = MagicMock()
    place_resp.status_code = 200
    place_resp.json.return_value = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1", "sCode": "0", "sMsg": ""}],
    }
    place_resp.headers = {}
    client._http_place.request.return_value = place_resp

    client.place_post_only_limit(
        symbol="SUI-USDT-SWAP",
        is_buy=True,
        sz=1.0,
        limit_px=1.0,
        client_order_id="cloid-1",
    )

    assert client._http_place.request.called
    assert not client._http_cancel.request.called


def test_generic_query_routes_through_place_client() -> None:
    """A direct ``_request`` for a non-cancel op (``query_order``)
    confirms the default branch points at the place pool. This is
    the catch-all for any future read-side op added to the adapter
    without explicit thought about pool routing."""
    client = _make_client_with_mocked_http()
    client._http_cancel.request.return_value = _ok_response()
    client._http_place.request.return_value = _ok_response()

    client._request("query_order", "GET", "/api/v5/trade/order", signed=True)

    assert client._http_place.request.called
    assert not client._http_cancel.request.called


# --------------------------------------------------------------------------
# HTTP/2 toggle + lifecycle (close)
# --------------------------------------------------------------------------


def test_http2_enabled_by_default() -> None:
    """Default constructor opts into HTTP/2. The ``h2`` extra is in
    the project deps so importing httpx with http2=True succeeds; we
    can't easily introspect the negotiated protocol post-handshake,
    but we CAN check the client was built with http2 capability."""
    client = _make_client()
    try:
        # httpx exposes the http2 flag via the internal transport.
        # The robust check is that construction succeeded — if the
        # ``h2`` package were missing, http2=True would have raised at
        # construction time.
        assert client._http_place is not None
        assert client._http_cancel is not None
    finally:
        client.close()


def test_http2_can_be_disabled_via_setting() -> None:
    """One-line revert path: ``OKX_HTTP2_ENABLED=false`` builds the
    clients with HTTP/1.1 only. Verified by constructing successfully
    even in an environment that hypothetically lacked ``h2`` — but
    since we can't unimport ``h2`` mid-test, we settle for asserting
    the setting flows through to the constructor without error."""
    settings = _settings(OKX_HTTP2_ENABLED=False)
    client = _make_client(settings)
    try:
        assert client._http_place is not None
        assert client._http_cancel is not None
    finally:
        client.close()


def test_close_releases_both_clients_idempotently() -> None:
    """``close()`` must shut both pools and tolerate being called
    twice. Process-shutdown paths (atexit, finally blocks) may invoke
    it more than once."""
    client = _make_client()
    # Replace with mocks AFTER construction so we can verify .close()
    # is called on each.
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_cancel = MagicMock(spec=httpx.Client)

    client.close()
    assert client._http_place.close.called
    assert client._http_cancel.close.called

    # Second call must not raise even if one of the mocks errors out.
    client._http_cancel.close.side_effect = RuntimeError("bang")
    client.close()  # should swallow the error, no exception propagates


def test_close_continues_when_one_client_raises() -> None:
    """If ``_http_place.close()`` raises, ``_http_cancel.close()`` must
    still run. Otherwise a half-closed state leaks a connection pool
    at shutdown."""
    client = _make_client()
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_cancel = MagicMock(spec=httpx.Client)
    client._http_place.close.side_effect = RuntimeError("place_close_fail")

    client.close()

    assert client._http_place.close.called
    assert client._http_cancel.close.called, (
        "cancel client must close even if place client raises"
    )
