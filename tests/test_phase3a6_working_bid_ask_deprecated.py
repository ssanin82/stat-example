"""v1.4.91 wedge-elimination-cleanup Phase 3A.6 — DeprecationWarning
on legacy ``working_bid`` / ``working_ask`` accessors.

The accessors stay functional (60+ legacy call sites depend on them)
but emit a ``DeprecationWarning`` ONCE per process so devs / CI see
the migration marker. Test runs with ``-W error::DeprecationWarning``
will flag new uses; prod with the default silent-DW filter is
unaffected.

Tests verify:

* First access to ``working_bid`` emits exactly one DeprecationWarning.
* Subsequent accesses (same process) emit ZERO additional warnings —
  the once-per-process gate prevents test/log spam from the 60+
  legacy call sites.
* Same independent gate for ``working_ask``.
* Warning message mentions the recommended replacement
  (``state.order_store.get`` / ``state.tick_snapshot()``).
"""

from __future__ import annotations

import os
import tempfile
import uuid
import warnings
from pathlib import Path

from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_phase3a6_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })


def _reset_gate() -> None:
    """Reset the once-per-process gate so individual tests can verify
    the FIRST-emit behavior. Pokes the module-level flags directly."""
    import app.state as _state_mod
    _state_mod._WORKING_BID_DEPRECATION_EMITTED = False
    _state_mod._WORKING_ASK_DEPRECATION_EMITTED = False


def test_phase3a6_working_bid_emits_deprecation_warning_on_first_access() -> None:
    _reset_gate()
    state = BotState(_settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = state.working_bid
    matching = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(matching) == 1, (
        f"first access to working_bid should emit ONE DeprecationWarning; "
        f"got {len(matching)}"
    )
    msg = str(matching[0].message)
    assert "working_bid" in msg
    assert "order_store" in msg or "tick_snapshot" in msg, (
        f"warning message must reference the recommended replacement; got: {msg}"
    )


def test_phase3a6_working_bid_only_emits_once_per_process() -> None:
    _reset_gate()
    state = BotState(_settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(100):
            _ = state.working_bid
    matching = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    # The once-per-process gate must produce exactly 1 warning despite
    # 100 accesses. This is the critical property — without it, the
    # 60+ legacy call sites would flood every test run.
    assert len(matching) == 1, (
        f"expected exactly 1 DeprecationWarning for 100 accesses (once-per-process gate); "
        f"got {len(matching)}"
    )


def test_phase3a6_working_ask_emits_deprecation_warning_on_first_access() -> None:
    _reset_gate()
    state = BotState(_settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = state.working_ask
    matching = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(matching) == 1
    msg = str(matching[0].message)
    assert "working_ask" in msg
    assert "order_store" in msg or "tick_snapshot" in msg


def test_phase3a6_working_ask_only_emits_once_per_process() -> None:
    _reset_gate()
    state = BotState(_settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(100):
            _ = state.working_ask
    matching = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(matching) == 1


def test_phase3a6_bid_and_ask_gates_are_independent() -> None:
    """The two properties have separate gates — accessing working_bid
    does not consume the working_ask gate (and vice versa). Otherwise
    devs migrating away from one accessor would silently lose the
    marker for the other."""
    _reset_gate()
    state = BotState(_settings())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = state.working_bid
        _ = state.working_ask
    matching = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(matching) == 2
    msgs = [str(w.message) for w in matching]
    assert any("working_bid" in m for m in msgs)
    assert any("working_ask" in m for m in msgs)


def test_phase3a6_property_still_returns_correct_value() -> None:
    """Behavioural invariant — the deprecation marker doesn't change
    what the property returns. Legacy callers keep working."""
    from app.enums import OrderStatus, Side
    from app.models import WorkingOrder
    from app.utils.time import utc_now
    _reset_gate()
    state = BotState(_settings())
    # Initially None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # don't care about warnings here
        assert state.working_bid is None
        assert state.working_ask is None
    # Set via order_store
    now = utc_now()
    wo_bid = WorkingOrder(
        order_id_local="l-1", order_id_exchange=1, client_order_id="c1",
        symbol="ETH", side=Side.BUY, price=2.0, size=1.0,
        post_only=True, status=OrderStatus.ACKED,
        ts_created=now, ts_sent=now, ts_ack=now,
    )
    state.order_store.set(Side.BUY, 0, wo_bid)
    # Legacy accessor reflects the set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert state.working_bid is wo_bid
        assert state.working_ask is None
