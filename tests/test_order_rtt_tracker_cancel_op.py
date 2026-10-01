"""Phase 0.5 regression — OrderRttTracker op_filter support.

The tracker has always carried an ``op`` discriminator on samples
(``"place"`` or ``"cancel"``) but the ``summary()`` method did not
filter by it — meaning a mixed-op tracker would produce a polluted
distribution. v1.4.0 adds an ``op_filter`` kwarg used by
``OrderManager.cancel_rtt_summary()`` to isolate cancels.
"""

from __future__ import annotations

from app.order_rtt_tracker import OrderRttTracker


def test_op_filter_isolates_cancel_samples() -> None:
    t = OrderRttTracker(max_samples=1024)
    # 3 place samples, 2 cancel samples.
    t.ingest(rtt_ms=10.0, op="place", outcome="accepted")
    t.ingest(rtt_ms=12.0, op="place", outcome="accepted")
    t.ingest(rtt_ms=14.0, op="place", outcome="accepted")
    t.ingest(rtt_ms=4.0, op="cancel", outcome="accepted")
    t.ingest(rtt_ms=5.0, op="cancel", outcome="accepted")

    # Default summary: no op filter → includes all 5.
    all_summary = t.summary()
    assert all_summary["sample_count"] == 5

    # op_filter="place" → 3 samples; median is 12.0.
    place_summary = t.summary(op_filter="place")
    assert place_summary["sample_count"] == 3
    assert place_summary["median_ms"] == 12.0

    # op_filter="cancel" → 2 samples; median is 4.5.
    cancel_summary = t.summary(op_filter="cancel")
    assert cancel_summary["sample_count"] == 2
    assert cancel_summary["median_ms"] == 4.5


def test_op_filter_with_outcome_filter_composes() -> None:
    """When both filters are set they compose (AND). A cancel sample
    that errored out shouldn't appear in either the cancel-accepted
    summary or the place-accepted summary."""
    t = OrderRttTracker(max_samples=1024)
    t.ingest(rtt_ms=10.0, op="place", outcome="accepted")
    t.ingest(rtt_ms=200.0, op="cancel", outcome="transport_error")
    t.ingest(rtt_ms=5.0, op="cancel", outcome="accepted")

    cancel_accepted = t.summary(op_filter="cancel", outcome_filter="accepted")
    assert cancel_accepted["sample_count"] == 1
    assert cancel_accepted["median_ms"] == 5.0

    # No outcome filter, cancel only → 2 samples.
    cancel_all = t.summary(op_filter="cancel", outcome_filter=None)
    assert cancel_all["sample_count"] == 2


def test_op_filter_empty_returns_empty_summary() -> None:
    """No samples for the requested op → all-None summary, not crash."""
    t = OrderRttTracker(max_samples=1024)
    t.ingest(rtt_ms=10.0, op="place", outcome="accepted")
    # No cancels ingested yet.
    cancel_summary = t.summary(op_filter="cancel")
    assert cancel_summary["sample_count"] == 0
    assert cancel_summary["median_ms"] is None
    assert cancel_summary["p95_ms"] is None


def test_op_filter_tuple_unifies_place_amend_cancel() -> None:
    """v1.4.58 todo-037: passing a tuple of op kinds pools their
    samples. The dashboard's unified "Tx send → ack" row uses
    ``op_filter=("place", "amend", "cancel")`` for a single
    distribution across all outbound transport actions.
    """
    t = OrderRttTracker(max_samples=1024)
    # Realistic-ish distribution mix:
    #  - 3 places at ~3.5 ms
    #  - 5 amends at ~3.5 ms (amends dominate volume on this venue)
    #  - 1 cancel at 3.4 ms
    t.ingest(rtt_ms=3.5, op="place", outcome="accepted")
    t.ingest(rtt_ms=3.6, op="place", outcome="accepted")
    t.ingest(rtt_ms=3.7, op="place", outcome="accepted")
    t.ingest(rtt_ms=3.4, op="amend", outcome="accepted")
    t.ingest(rtt_ms=3.5, op="amend", outcome="accepted")
    t.ingest(rtt_ms=3.6, op="amend", outcome="accepted")
    t.ingest(rtt_ms=3.7, op="amend", outcome="accepted")
    t.ingest(rtt_ms=3.8, op="amend", outcome="accepted")
    t.ingest(rtt_ms=3.4, op="cancel", outcome="accepted")

    # Unified pool = all 9 samples.
    unified = t.summary(op_filter=("place", "amend", "cancel"))
    assert unified["sample_count"] == 9, (
        "v1.4.58 todo-037: unified op_filter must pool all three "
        f"op kinds. Got: {unified}"
    )
    # Per-op filters still work for the postmortem path.
    assert t.summary(op_filter="place")["sample_count"] == 3
    assert t.summary(op_filter="amend")["sample_count"] == 5
    assert t.summary(op_filter="cancel")["sample_count"] == 1
    # ``("place", "amend")`` — the pre-v1.4.58 default for
    # ``order_rtt_summary`` — still works and excludes the cancel.
    assert t.summary(op_filter=("place", "amend"))["sample_count"] == 8


def test_tx_rtt_summary_method_unifies_all_three_ops() -> None:
    """v1.4.58 todo-037: ``OrderManager.tx_rtt_summary()`` is the
    public API for the unified dashboard row. It must return the
    same shape as the per-op summaries but with the pooled count.
    """
    import os
    import tempfile
    import uuid
    from pathlib import Path
    from app.execution import OrderManager
    from app.state import BotState
    from app.storage import Storage
    from tests.exchange_client_mocks import mock_mm_client
    from tests.settings_helpers import UnitTestSettings

    s = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_tx_rtt_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    db_path.unlink(missing_ok=True)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Feed the tracker on the executor directly.
        om._order_rtt_tracker.ingest(rtt_ms=3.0, op="place", outcome="accepted")
        om._order_rtt_tracker.ingest(rtt_ms=3.5, op="amend", outcome="accepted")
        om._order_rtt_tracker.ingest(rtt_ms=4.0, op="cancel", outcome="accepted")

        tx = om.tx_rtt_summary()
        assert tx["sample_count"] == 3, (
            "tx_rtt_summary must pool all three op kinds"
        )
        # Per-op summaries still work — they don't see the others.
        assert om.order_rtt_summary()["sample_count"] == 2  # place+amend
        assert om.cancel_rtt_summary()["sample_count"] == 1  # cancel only

        # Provider wired on state.
        assert state.tx_rtt_summary_provider is not None
        # And it returns the same thing.
        assert state.tx_rtt_summary_provider()["sample_count"] == 3
    finally:
        db_path.unlink(missing_ok=True)
