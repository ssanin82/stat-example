"""Phase 1 (cross-venue basis disable on Binance) regression tests.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 1.

When the trading venue ``EXCHANGE`` itself is Binance, the existing
``binance_basis_ewma`` (Bluefin/HL/GRVT vs Binance) is structurally
meaningless — we'd be measuring Binance against itself. The clean
operational pairing is ``EXCHANGE=binance`` + ``REFERENCE_EXCHANGE=off``.

These tests pin three invariants:

1. ``EXCHANGE`` accepts ``binance`` as a valid trading venue.
2. The ``EXCHANGE=binance`` + ``REFERENCE_EXCHANGE=binance`` misconfig
   auto-overrides ``reference_exchange`` to ``"off"`` and logs WARN.
3. Other ``REFERENCE_EXCHANGE`` values are unaffected (cross-venue
   pairings remain valid for HL / GRVT / Bluefin trading venues).
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from tests.settings_helpers import UnitTestSettings


# --- enum coverage --------------------------------------------------------


def test_exchange_accepts_binance() -> None:
    """The Phase 2 adapter wires ``EXCHANGE=binance``; the field
    validator must accept it.
    """
    s = UnitTestSettings.model_validate({"EXCHANGE": "binance"})
    assert s.exchange == "binance"


def test_exchange_rejects_unknown() -> None:
    """Sanity: random venue string still fails (typo guard)."""
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate({"EXCHANGE": "kraken"})


def test_exchange_legacy_hl_still_aliases_to_hyperliquid() -> None:
    """The existing ``hl`` shorthand keeps working — don't break old
    profiles when adding Binance.
    """
    s = UnitTestSettings.model_validate({"EXCHANGE": "hl"})
    assert s.exchange == "hyperliquid"


# --- self-referential reference auto-override -----------------------------


def test_binance_with_binance_reference_auto_overrides_to_off(caplog) -> None:
    """Headline regression: misconfig is silently corrected and logged."""
    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = UnitTestSettings.model_validate(
            {"EXCHANGE": "binance", "REFERENCE_EXCHANGE": "binance"}
        )
    assert s.exchange == "binance"
    assert s.reference_exchange == "off"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("auto-overriding" in m and "binance" in m.lower() for m in msgs), (
        f"expected auto-override warning; got {msgs!r}"
    )


def test_binance_with_explicit_off_keeps_off_no_warning(caplog) -> None:
    """The recommended pairing — ``EXCHANGE=binance`` + ``REFERENCE_EXCHANGE=off``
    — must NOT emit the misconfig warning.
    """
    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = UnitTestSettings.model_validate(
            {"EXCHANGE": "binance", "REFERENCE_EXCHANGE": "off"}
        )
    assert s.reference_exchange == "off"
    auto_override_msgs = [
        r.getMessage()
        for r in caplog.records
        if "auto-overriding" in r.getMessage()
    ]
    assert not auto_override_msgs


def test_bluefin_with_binance_reference_unchanged(caplog) -> None:
    """Counter-test: the cross-venue pairing
    (``EXCHANGE=bluefin/hyperliquid/grvt`` + ``REFERENCE_EXCHANGE=binance``)
    is a valid configuration and must NOT trip the auto-override.
    """
    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = UnitTestSettings.model_validate(
            {"EXCHANGE": "bluefin", "REFERENCE_EXCHANGE": "binance"}
        )
    assert s.exchange == "bluefin"
    assert s.reference_exchange == "binance"


def test_bybit_reference_paired_with_binance_trading_venue_is_left_alone(
    caplog,
) -> None:
    """Bybit is a different cross-venue reference and is independent
    of the Binance auto-override path.
    """
    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = UnitTestSettings.model_validate(
            {"EXCHANGE": "binance", "REFERENCE_EXCHANGE": "bybit"}
        )
    assert s.exchange == "binance"
    assert s.reference_exchange == "bybit"
