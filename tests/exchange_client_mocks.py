"""Shared MagicMock setup for HyperliquidMMClient-shaped doubles."""

from __future__ import annotations

from typing import Any

from unittest.mock import MagicMock

from app.exchange.hyperliquid_responses import make_deterministic_cloid_hex
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC, SymbolSpec


def _default_cancel_batch_response(
    _symbol: str, refs: list[dict[str, str]]
) -> dict[str, Any]:
    """Default OKX-shaped batch-cancel response for the mock client.

    v1.4.36: every cancel now routes through ``cancel_batch_orders``
    (v1.4.33 dispatcher fix + v1.4.35 executor-threshold completion).
    Without a real response shape, the MagicMock returns a bare
    MagicMock object that ``interpret_okx_cancel_batch_response``
    treats as ``non_dict_response`` → every row becomes
    ``missing_row_for_wo`` → tests asserting on cancel-call success
    fail spuriously.

    Echo each input ref back as a success row in the same order;
    ``interpret_okx_cancel_batch_response`` walks rows positionally
    (and by cloid when present), so this matches the per-row
    accounting executor code expects. ``ordId``/``clOrdId`` echo the
    input so per-row attribution by cloid still works.
    """
    rows: list[dict[str, Any]] = []
    for r in refs:
        rows.append(
            {
                "sCode": "0",
                "sMsg": "",
                "ordId": str(r.get("ordId") or ""),
                "clOrdId": str(r.get("clOrdId") or ""),
            }
        )
    return {"code": "0", "msg": "", "data": rows}


def mock_mm_client(
    *,
    symbol_spec: SymbolSpec | None = None,
    symbol_spec_fetched_ok: bool = True,
    **kwargs: Any,
) -> MagicMock:
    c = MagicMock(**kwargs)
    c.symbol_spec = symbol_spec if symbol_spec is not None else FALLBACK_SYMBOL_SPEC
    c.symbol_spec_fetched_ok = symbol_spec_fetched_ok
    # Adapter-owned cloid generation: execution calls this instead of importing
    # the HL-specific helper directly, so mocks must route back to the same
    # deterministic implementation to keep existing tests asserting on cloids.
    c.make_client_order_id.side_effect = make_deterministic_cloid_hex
    # Cancel-confirmation gate (Bluefin async-cancel race mitigation, see
    # app/exchange/base.py docstring). Tests that mock the client surface
    # are venue-agnostic and should behave like sync-cancel venues (HL/GRVT):
    # no pending cancels, gate is transparent. Without this, MagicMock's
    # auto-attribute generation would return a truthy MagicMock instance
    # from ``has_pending_cancel(...)`` and spuriously suppress placements.
    c.has_pending_cancel.return_value = False
    # v1.4.36: default ``cancel_batch_orders`` to a per-ref success
    # echo. Tests that need a specific response (rejects, partial
    # benign_missing) override via ``c.cancel_batch_orders.side_effect``
    # or ``.return_value``.
    c.cancel_batch_orders.side_effect = _default_cancel_batch_response
    return c
