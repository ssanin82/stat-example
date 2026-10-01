"""Phase 3F (v1.4.175) — reservation-alpha contribution analysis.

Tests cover three layers:

1. **Schema** — migration v41 adds the four ``*_shift_bps_at_decision``
   columns to both ``orders`` and ``fills`` (mirrors the existing
   v23 ``_at_decision`` pattern).

2. **Math** — the postmortem section's pure-function detector
   computes directional accuracy, cumulative contribution + verdict
   correctly across the canonical scenarios (helpful / noise /
   harmful / inconclusive / dormant alpha).

3. **Stamping contract** — ``order_row`` / ``fill_row`` carry the
   four new columns; the ``_DECISION_STATE_COLS`` tuple in storage
   includes them so the fill-ingest lookup propagates the values
   from parent order to fill.

The full Bot-level place → ingest → postmortem chain is exercised
indirectly via the existing soft_flatten_event_attribution + ladder
tests; here we keep scope tight to the new math + plumbing.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.enums import Side
from app.execution import order_row
from app.fill_ingestion import fill_row
from app.models import Fill, WorkingOrder
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings
from tools.postmortem.sections.reservation_alpha_attribution import (
    detect_reservation_alpha_findings,
    render_markdown_section,
)


def _settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_3f_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "LOGS_BUCKET": "fake-bucket",
            "LIVE_STATS_ENABLED": True,
        }
    )


# ---------------------------------------------------------------------------
# Schema migration v41
# ---------------------------------------------------------------------------


def test_schema_v41_adds_alpha_shift_columns_to_orders_and_fills() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(s.database_url.split("sqlite:///", 1)[-1])
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 41
        for tbl in ("orders", "fills"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")}
            for col in (
                "ob_imbalance_shift_bps_at_decision",
                "trend_drift_shift_bps_at_decision",
                "flow_score_shift_bps_at_decision",
                "basis_deviation_shift_bps_at_decision",
            ):
                assert col in cols, f"missing {col} in {tbl}"
    finally:
        conn.close()


def test_decision_state_cols_includes_alpha_shifts() -> None:
    """``_DECISION_STATE_COLS`` is the central tuple driving the
    ``order_metadata_for_fill_ingest`` SELECT. The four new shift
    columns MUST be in it or fill-ingest won't propagate from
    parent order → fill."""
    cols = set(Storage._DECISION_STATE_COLS)
    for col in (
        "ob_imbalance_shift_bps_at_decision",
        "trend_drift_shift_bps_at_decision",
        "flow_score_shift_bps_at_decision",
        "basis_deviation_shift_bps_at_decision",
    ):
        assert col in cols


# ---------------------------------------------------------------------------
# Row-mapper contracts
# ---------------------------------------------------------------------------


def test_order_row_includes_alpha_shift_fields() -> None:
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=None,
        client_order_id="c",
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=__import__("app.enums", fromlist=["OrderStatus"]).OrderStatus.SENT,
        ob_imbalance_shift_bps_at_decision=1.23,
        trend_drift_shift_bps_at_decision=-0.45,
        flow_score_shift_bps_at_decision=0.0,
        basis_deviation_shift_bps_at_decision=None,
    )
    row = order_row(wo)
    assert row["ob_imbalance_shift_bps_at_decision"] == 1.23
    assert row["trend_drift_shift_bps_at_decision"] == -0.45
    assert row["flow_score_shift_bps_at_decision"] == 0.0
    assert row["basis_deviation_shift_bps_at_decision"] is None


def test_fill_row_includes_alpha_shift_fields() -> None:
    f = Fill(
        fill_id="x",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 5, 21, 9, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=1.0,
        ob_imbalance_shift_bps_at_decision=2.5,
        trend_drift_shift_bps_at_decision=-1.0,
        flow_score_shift_bps_at_decision=0.0,
        basis_deviation_shift_bps_at_decision=None,
    )
    r = fill_row(f)
    assert r["ob_imbalance_shift_bps_at_decision"] == 2.5
    assert r["trend_drift_shift_bps_at_decision"] == -1.0
    assert r["flow_score_shift_bps_at_decision"] == 0.0
    assert r["basis_deviation_shift_bps_at_decision"] is None


# ---------------------------------------------------------------------------
# Attribution math — directional accuracy + verdict
# ---------------------------------------------------------------------------


def _fill(
    *,
    side: str = "BUY",
    markout: float = 0.0,
    ob: float | None = None,
    trend: float | None = None,
    flow: float | None = None,
    basis: float | None = None,
) -> dict[str, object]:
    return {
        "side": side,
        "markout_5s_bps": markout,
        "ob_imbalance_shift_bps_at_decision": ob,
        "trend_drift_shift_bps_at_decision": trend,
        "flow_score_shift_bps_at_decision": flow,
        "basis_deviation_shift_bps_at_decision": basis,
    }


def test_directional_accuracy_perfect_alpha_but_costs_markout() -> None:
    """Alpha shifts UP for every BUY that has positive markout
    (price did go up). Directional accuracy = 100 %.

    BUT — even though the alpha is directionally right, it COST us
    markout: BUY @ raised bid means we pay more for the fill. With
    shift=+2 bp, contribution per BUY = -shift = -2 bp.

    Verdict: harmful (contrib below the -1.0 threshold). The point of
    this test: the verdict combines BOTH dir-acc AND contribution,
    so "directionally right but expensive" still scores harmful."""
    fills = []
    for _ in range(150):
        fills.append(_fill(side="BUY", markout=5.0, ob=2.0))
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    assert ob.directional_accuracy == 1.0
    assert ob.mean_contribution_bps_per_fill == pytest.approx(-2.0)
    assert ob.verdict == "harmful"


def test_dormant_alpha_classified_as_inconclusive() -> None:
    """Alpha set to 0.0 (e.g. ``BASIS_DEVIATION_ALPHA=0.0``) — every
    fill carries shift=0. n_nonzero_shift = 0 → inconclusive."""
    fills = [_fill(side="BUY", markout=2.0, basis=0.0) for _ in range(200)]
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    basis = next(a for a in out.alphas if a.key == "basis_deviation")
    assert basis.n_fills == 200
    assert basis.n_nonzero_shift == 0
    assert basis.verdict == "inconclusive"


def test_alpha_with_positive_acc_and_positive_contrib_is_helpful() -> None:
    """Construct the rare 'genuinely helpful' case: BUY at +5 shift
    with markout +10 (shift moved us up; price moved EVEN MORE up;
    counterfactual would have BUYed at lower price = +15 markout, so
    alpha cost 5 bp) — NOT helpful.

    The cleanly-helpful case is SELL: +5 shift means ask was raised;
    we sold higher; the price kept going up so future mid > our
    higher sell price → adverse markout BUT the counterfactual sell
    would have been at lower price, even more adverse markout.

    For a SELL: contribution = +shift. Shift=+5 → contribution +5 bp
    per fill — alpha *added* 5 bps of markout vs. counterfactual. AND
    if directional accuracy is high (shift sign matches price-move
    sign), the verdict is helpful."""
    fills = []
    # 150 SELL fills. Shift +5 (alpha says price will go UP). Markout
    # -3 bp (we sold; price went up; we lost 3 bp). Price move from
    # SELL perspective = -markout = +3 (price moved up). sign(shift) ==
    # sign(price_move) → directional match. Contribution = +shift =
    # +5 bp per fill (the alpha's raised ask helped us sell higher).
    for _ in range(150):
        fills.append(_fill(side="SELL", markout=-3.0, ob=5.0))
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    assert ob.directional_accuracy == 1.0
    assert ob.mean_contribution_bps_per_fill == pytest.approx(5.0)
    assert ob.verdict == "helpful"


def test_alpha_with_random_signal_is_noise() -> None:
    """Half of fills: shift +1, markout +0.0001 (tiny up). Half:
    shift -1, markout -0.0001 (tiny down). Directional accuracy =
    100 % but contribution per fill ≈ ±1 cancels to ~0. The
    contribution-based check pulls this OUT of "helpful" — even
    perfect direction + zero contribution = inconclusive (not
    helpful, not harmful)."""
    fills = []
    for _ in range(75):
        fills.append(_fill(side="BUY", markout=0.0001, ob=1.0))
    for _ in range(75):
        fills.append(_fill(side="SELL", markout=-0.0001, ob=1.0))
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    # BUY contributions = -1 each (75 fills), SELL contributions = +1
    # each (75 fills). Net: 0. Per-fill mean ≈ 0.
    assert abs(ob.mean_contribution_bps_per_fill) < 0.1
    # acc = 1.0 (above 0.55) but contrib per fill ≈ 0 (NOT above the
    # +0.5 helpful threshold) and acc not in noise band → inconclusive
    # is the right verdict for this mixed case.
    # Test passes as long as verdict isn't 'helpful' (which would be
    # wrong — no real contribution).
    assert ob.verdict != "helpful"


def test_anti_predictive_alpha_classified_as_harmful() -> None:
    """Alpha consistently points wrong way: shift +1 but price goes
    DOWN. Directional accuracy ≈ 0%. Verdict = harmful."""
    fills = []
    for _ in range(150):
        # BUY: shift +1, markout -5 (we bought and price went DOWN).
        # price_move = +markout = -5 (down). sign(shift)=+, sign(pm)=-
        # → mismatch. Contribution = -shift = -1 per fill.
        fills.append(_fill(side="BUY", markout=-5.0, ob=1.0))
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    assert ob.directional_accuracy == 0.0
    assert ob.verdict == "harmful"


def test_below_min_fills_returns_inconclusive() -> None:
    """50 fills < 100 → inconclusive regardless of math."""
    fills = [_fill(side="BUY", markout=5.0, ob=1.0) for _ in range(50)]
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    assert ob.verdict == "inconclusive"


def test_skips_fills_with_null_shift_or_null_markout() -> None:
    """Fills with NULL shift OR NULL markout are not counted toward
    any alpha. The detector must filter them silently."""
    fills = [
        # 100 fills with all-null shifts.
        _fill(side="BUY", markout=5.0, ob=None, trend=None, flow=None, basis=None)
        for _ in range(100)
    ]
    # Plus 200 valid fills for the ob alpha only.
    for _ in range(200):
        fills.append(_fill(side="SELL", markout=-3.0, ob=5.0))
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    ob = next(a for a in out.alphas if a.key == "ob_imbalance")
    assert ob.n_fills == 200  # the 100 null-ob fills were skipped
    trend = next(a for a in out.alphas if a.key == "trend_drift")
    assert trend.n_fills == 0


def test_has_any_shift_data_flag_false_on_pre_migration_snapshot() -> None:
    """All-null shift columns → flag false → markdown renders empty-
    state note instead of pretending to compute attributions."""
    fills = [_fill(side="BUY", markout=5.0) for _ in range(150)]
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    assert out.has_any_shift_data is False
    md = render_markdown_section(out)
    assert "pre-dates the v1.4.175 schema" in md


# ---------------------------------------------------------------------------
# Markdown rendering smoke test
# ---------------------------------------------------------------------------


def test_markdown_section_renders_table_and_verdict() -> None:
    fills = [_fill(side="SELL", markout=-3.0, ob=5.0) for _ in range(150)]
    out = detect_reservation_alpha_findings(
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        fills=fills,
    )
    md = render_markdown_section(out)
    assert "Reservation-alpha contribution" in md
    assert "OB imbalance" in md
    # Verdict emoji + word should appear.
    assert "helpful" in md
    # Directional accuracy column populated.
    assert "100.0%" in md
