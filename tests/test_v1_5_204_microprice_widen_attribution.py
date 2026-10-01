"""v1.5.204 — Microprice widen per-fill attribution tests.

Covers:

1. Schema migration v44 → v45 adds the four new columns
   (orders.microprice_{bid,ask}_widen_bps_at_decision +
    fills.microprice_{bid,ask}_widen_bps_at_decision).
2. ``_DECISION_STATE_COLS`` includes both microprice fields.
3. WorkingOrder + Fill dataclasses carry the fields.
4. Postmortem section ``microprice_widen_attribution`` correctly
   splits fills by firing/silent and emits the expected verdict.
5. Acceptance check ``check_v1_5_204_microprice_widen_markout_within_noise``
   issues PASS / WARN / FAIL / INSUFFICIENT_DATA / N/A per the
   documented thresholds.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# --------------------------- schema migration ---------------------------- #


def test_schema_v45_adds_microprice_widen_columns(tmp_path: Path) -> None:
    """Fresh storage opens at v45 with both new columns on orders + fills."""
    from app.storage import Storage
    from tests.settings_helpers import UnitTestSettings

    db_path = tmp_path / "mm.db"
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        }
    )
    s = Storage(settings)
    s.init_schema()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            ver = conn.execute("PRAGMA user_version").fetchone()[0]
            assert int(ver) == Storage.SCHEMA_VERSION
            assert Storage.SCHEMA_VERSION >= 45
            order_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(orders)")
            }
            fill_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(fills)")
            }
            assert "microprice_bid_widen_bps_at_decision" in order_cols
            assert "microprice_ask_widen_bps_at_decision" in order_cols
            assert "microprice_bid_widen_bps_at_decision" in fill_cols
            assert "microprice_ask_widen_bps_at_decision" in fill_cols
    finally:
        # Force close any sqlite handles before the temp dir cleanup
        # (Windows test isolation).
        try:
            s.close()
        except Exception:
            pass


def test_decision_state_cols_includes_microprice() -> None:
    from app.storage import Storage

    cols = Storage._DECISION_STATE_COLS
    assert "microprice_bid_widen_bps_at_decision" in cols
    assert "microprice_ask_widen_bps_at_decision" in cols


# --------------------------- dataclass fields ---------------------------- #


def test_working_order_has_microprice_fields() -> None:
    from app.models import WorkingOrder

    names = {f.name for f in dataclasses.fields(WorkingOrder)}
    assert "microprice_bid_widen_bps_at_decision" in names
    assert "microprice_ask_widen_bps_at_decision" in names


def test_fill_has_microprice_fields() -> None:
    from app.models import Fill

    names = {f.name for f in dataclasses.fields(Fill)}
    assert "microprice_bid_widen_bps_at_decision" in names
    assert "microprice_ask_widen_bps_at_decision" in names


# --------------------------- postmortem section -------------------------- #


def _fill(
    *,
    side: str,
    bid_w: float | None,
    ask_w: float | None,
    m5: float | None = None,
    m30: float | None = None,
    m60: float | None = None,
) -> dict[str, Any]:
    return {
        "side": side,
        "microprice_bid_widen_bps_at_decision": bid_w,
        "microprice_ask_widen_bps_at_decision": ask_w,
        "markout_5s_bps": m5,
        "markout_30s_bps": m30,
        "markout_60s_bps": m60,
    }


def test_detect_microprice_findings_empty_returns_empty_state() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
        render_markdown_section,
    )
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=[],
    )
    assert findings.fills_with_data == 0
    # Empty state — markdown render returns empty string.
    assert render_markdown_section(findings) == ""


def test_detect_microprice_findings_splits_firing_vs_silent() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
    )
    fills = [
        # Firing: ask side widened on a SELL.
        _fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-2.0),
        _fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-1.5),
        # Silent.
        _fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+1.0),
        _fill(side="SELL", bid_w=0.0, ask_w=0.0, m30=+1.5),
    ]
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=fills,
    )
    assert findings.firing.n_fills == 2
    assert findings.not_firing.n_fills == 2
    # delta = firing_mean − silent_mean = (-1.75) − (1.25) = -3.0
    assert findings.delta_markout_30s_bps == pytest.approx(-3.0)
    # Not enough samples (<25 per bucket) → inconclusive.
    assert findings.verdict == "inconclusive"


def test_detect_microprice_protective_verdict() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
    )
    # 30 firing fills, mean -2.0 bp; 30 silent fills, mean +0.5 bp
    # → delta = -2.5 bp ≤ -0.5 → protective.
    fills: list[dict] = []
    for _ in range(30):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-2.0))
    for _ in range(30):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.5))
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=fills, configured_widen_bps=10.0,
    )
    assert findings.verdict == "protective"
    assert findings.configured_widen_bps == 10.0


def test_detect_microprice_harmful_verdict() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
    )
    # 30 firing fills, mean +1.5 bp; 30 silent fills, mean 0
    # → delta = +1.5 bp ≥ +1.0 → harmful.
    fills: list[dict] = []
    for _ in range(30):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=+1.5))
    for _ in range(30):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=0.0))
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=fills,
    )
    assert findings.verdict == "harmful"


def test_detect_microprice_neutral_verdict() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
    )
    # Both buckets at the same markout → delta ~0 → neutral.
    fills: list[dict] = []
    for _ in range(30):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=+0.0))
    for _ in range(30):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.0))
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=fills,
    )
    assert findings.verdict == "neutral"


def test_render_markdown_includes_configured_widen_bps() -> None:
    from tools.postmortem.sections.microprice_widen_attribution import (
        detect_microprice_widen_findings,
        render_markdown_section,
    )
    fills = [_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-1.0)]
    findings = detect_microprice_widen_findings(
        snapshot_name="x", bot_version="1.5.204", captured_at="t",
        fills=fills, configured_widen_bps=10.0,
    )
    md = render_markdown_section(findings)
    assert "10.0 bps" in md  # configured widen is surfaced
    assert "Microprice-gate widening attribution" in md


# --------------------------- acceptance check ---------------------------- #


from dataclasses import dataclass, field


@dataclass
class _FakeSnap:
    bot_version: str = "1.5.204"
    fills_since: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    captured_at: str = "t"
    snapshot_dir: Path = field(default_factory=lambda: Path("."))
    snapshot_name: str = "fake"
    state_current: dict | None = None
    session_summary: dict | None = None
    inventory_since: list | None = None
    meta: dict = field(default_factory=dict)


def test_acceptance_check_na_pre_v1_5_204() -> None:
    import snapshot_acceptance as sa

    snap = _FakeSnap(bot_version="1.5.203")
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "N/A"


def test_acceptance_check_insufficient_data_no_fills() -> None:
    import snapshot_acceptance as sa

    snap = _FakeSnap(bot_version="1.5.204", fills_since=[])
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "INSUFFICIENT_DATA"


def test_acceptance_check_na_no_microprice_attribution() -> None:
    """All fills have NULL on both microprice columns → N/A
    (legacy fills, ingest path didn't propagate)."""
    import snapshot_acceptance as sa

    fills = [
        _fill(side="BUY", bid_w=None, ask_w=None, m30=+0.5)
        for _ in range(30)
    ]
    snap = _FakeSnap(bot_version="1.5.204", fills_since=fills)
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "N/A"


def test_acceptance_check_insufficient_data_low_bucket() -> None:
    import snapshot_acceptance as sa

    # 5 firing + 30 silent → INSUFFICIENT_DATA (firing < 15).
    fills: list[dict] = []
    for _ in range(5):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-1.0))
    for _ in range(30):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.5))
    snap = _FakeSnap(bot_version="1.5.204", fills_since=fills)
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "INSUFFICIENT_DATA"


def test_acceptance_check_pass_when_delta_within_noise() -> None:
    import snapshot_acceptance as sa

    fills: list[dict] = []
    # Same mean → delta = 0 → PASS.
    for _ in range(20):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=+0.1))
    for _ in range(20):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.1))
    snap = _FakeSnap(
        bot_version="1.5.204",
        fills_since=fills,
        config={"MICROPRICE_WIDEN_BPS": 10.0},
    )
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "PASS"
    assert r.measured["configured_widen_bps"] == 10.0


def test_acceptance_check_fail_when_adverse_delta_large() -> None:
    import snapshot_acceptance as sa

    fills: list[dict] = []
    # Firing bucket markedly worse than silent → FAIL.
    for _ in range(20):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=+2.0))
    for _ in range(20):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.0))
    snap = _FakeSnap(bot_version="1.5.204", fills_since=fills)
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "FAIL"


def test_acceptance_check_warn_when_borderline() -> None:
    import snapshot_acceptance as sa

    # Borderline: delta = +0.7 bp (between 0.5 and 1.0).
    fills: list[dict] = []
    for _ in range(20):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=+0.7))
    for _ in range(20):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.0))
    snap = _FakeSnap(bot_version="1.5.204", fills_since=fills)
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "WARN"


def test_acceptance_check_warn_when_strongly_protective() -> None:
    """Delta < -1.0 means the gate is doing GOOD work but the
    test reports WARN to surface the magnitude, not PASS."""
    import snapshot_acceptance as sa

    fills: list[dict] = []
    for _ in range(20):
        fills.append(_fill(side="SELL", bid_w=0.0, ask_w=10.0, m30=-2.0))
    for _ in range(20):
        fills.append(_fill(side="BUY", bid_w=0.0, ask_w=0.0, m30=+0.0))
    snap = _FakeSnap(bot_version="1.5.204", fills_since=fills)
    r = sa.check_v1_5_204_microprice_widen_markout_within_noise(snap)
    assert r.status == "WARN"
    assert "PROTECTIVE" in (r.detail or "")
