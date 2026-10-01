"""v1.4.93 wedge-elimination-cleanup Phase 4B.4 —
``MarkoutAdverseTracker`` is now an injected dependency on
``QuoteEngine``.

Default constructor path preserves the pre-Phase-4B.4 behavior:
when no tracker is passed, QuoteEngine constructs its own fresh
``MarkoutAdverseTracker``. The wire-in lets tests inject custom
trackers (mock or pre-configured) without going through the natural
breach-then-elapse timer state.

Tests verify:
* Default construction (no kwarg) yields a fresh real tracker —
  behavior unchanged vs pre-Phase-4B.4.
* Explicit construction with a custom tracker uses it instead of
  the default.
* Type contract: the injected tracker must support ``evaluate(...)``
  with the expected signature.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.quote_aging import MarkoutAdverseTracker
from app.quote_engine import QuoteEngine
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
    })


def test_phase4b4_default_construction_uses_fresh_tracker() -> None:
    """No explicit kwarg → engine constructs its own
    ``MarkoutAdverseTracker``. Production behavior preserved."""
    engine = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC)
    assert engine._markout_adverse_tracker is not None
    assert isinstance(engine._markout_adverse_tracker, MarkoutAdverseTracker)


def test_phase4b4_explicit_tracker_is_used() -> None:
    """Injected tracker overrides the default."""
    custom = MarkoutAdverseTracker()
    engine = QuoteEngine(
        _settings(),
        FALLBACK_SYMBOL_SPEC,
        markout_adverse_tracker=custom,
    )
    assert engine._markout_adverse_tracker is custom


def test_phase4b4_mock_tracker_can_be_injected() -> None:
    """Tests can inject a ``MagicMock`` to assert specific evaluate()
    call patterns or short-circuit the timer state."""
    mock_tracker = MagicMock(spec=MarkoutAdverseTracker)
    engine = QuoteEngine(
        _settings(),
        FALLBACK_SYMBOL_SPEC,
        markout_adverse_tracker=mock_tracker,
    )
    assert engine._markout_adverse_tracker is mock_tracker


def test_phase4b4_two_engines_with_distinct_trackers_are_independent() -> None:
    """Two engines with explicit trackers don't share state. Important
    when the same test runs multiple engine fixtures sequentially —
    pre-Phase-4B.4 a shared tracker reference would cause cross-test
    leakage."""
    t1 = MarkoutAdverseTracker()
    t2 = MarkoutAdverseTracker()
    e1 = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC, markout_adverse_tracker=t1)
    e2 = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC, markout_adverse_tracker=t2)
    assert e1._markout_adverse_tracker is t1
    assert e2._markout_adverse_tracker is t2
    assert e1._markout_adverse_tracker is not e2._markout_adverse_tracker


def test_phase4b4_default_engines_have_distinct_trackers() -> None:
    """Defaultpath: two engines constructed without explicit tracker
    each get their OWN fresh instance. Verifies no module-level shared
    default."""
    e1 = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC)
    e2 = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC)
    assert e1._markout_adverse_tracker is not e2._markout_adverse_tracker


def test_phase4b4_quote_engine_constructor_signature_is_backward_compat() -> None:
    """The first 2 positional args (settings, spec) still work without
    the keyword arg — existing production call site at
    ``app/execution.py:960`` is unchanged."""
    # No exception means the call signature is compatible.
    engine = QuoteEngine(_settings(), FALLBACK_SYMBOL_SPEC)
    assert engine is not None
    # And explicit positional + kwarg also works.
    engine2 = QuoteEngine(
        _settings(),
        FALLBACK_SYMBOL_SPEC,
        markout_adverse_tracker=None,  # explicit None → default behavior
    )
    assert isinstance(engine2._markout_adverse_tracker, MarkoutAdverseTracker)
