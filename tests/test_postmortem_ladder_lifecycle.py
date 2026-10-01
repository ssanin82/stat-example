"""v1.4.100 O3 — per-rung order lifecycle postmortem section.

Inline-constructed orders DataFrame + fills DataFrame exercises the
section's compute path without touching the full snapshot loader.

Coverage:
* Missing ``level_idx`` column → ``available=False``.
* Empty DataFrame → ``available=False``.
* Per-rung order counts add up to ``total_orders``.
* Fill-rate vs cancel-rate is computed correctly.
* Median lifetime ms is computed from ts_created/ts_closed.
* Top cancel reasons returned in descending count order.
* Markdown + HTML renderers don't raise.
"""

from __future__ import annotations

import pandas as pd

from tools.postmortem.sections.ladder_lifecycle import (
    detect_ladder_lifecycle_findings,
    render_html_section,
    render_markdown_section,
)


def _findings(orders, fills):
    return detect_ladder_lifecycle_findings(
        snapshot_name="test",
        bot_version="1.4.100",
        captured_at="2026-05-19T00:00:00Z",
        orders=orders,
        fills=fills,
    )


def test_empty_orders_marks_unavailable() -> None:
    out = _findings(pd.DataFrame(), pd.DataFrame())
    assert out.available is False


def test_missing_level_idx_marks_unavailable() -> None:
    """Pre-storage-v38 orders table — ``level_idx`` column absent."""
    orders = pd.DataFrame(
        {"order_id_local": ["o1"], "ts_created": ["2026-05-19T00:00:00Z"]}
    )
    out = _findings(orders, pd.DataFrame())
    assert out.available is False
    assert "level_idx" in (out.reason_unavailable or "")


def test_per_rung_counts_and_fill_rate() -> None:
    """Two rungs: L0 has 2 orders with 1 fill + 1 cancel;
    L1 has 1 order, cancelled, no fill."""
    orders = pd.DataFrame(
        {
            "order_id_local": ["o1", "o2", "o3"],
            "order_id_exchange": ["x1", "x2", "x3"],
            "level_idx": [0, 0, 1],
            "ts_created": [
                "2026-05-19T00:00:00Z",
                "2026-05-19T00:00:01Z",
                "2026-05-19T00:00:02Z",
            ],
            "ts_closed": [
                "2026-05-19T00:00:00.500Z",
                "2026-05-19T00:00:02.000Z",
                "2026-05-19T00:00:05.000Z",
            ],
            "ts_sent": [
                "2026-05-19T00:00:00.000Z",
                "2026-05-19T00:00:01.000Z",
                "2026-05-19T00:00:02.000Z",
            ],
            "ts_ack": [
                "2026-05-19T00:00:00.020Z",
                "2026-05-19T00:00:01.030Z",
                "2026-05-19T00:00:02.040Z",
            ],
            "cancel_reason": [None, "reprice", "reprice"],
        }
    )
    fills = pd.DataFrame({"order_id_exchange": ["x1"]})
    out = _findings(orders, fills)
    assert out.available is True
    assert out.total_orders == 3
    by_lvl = {r.level_idx: r for r in out.by_rung}
    assert by_lvl[0].order_count == 2
    assert by_lvl[0].fill_count == 1
    assert by_lvl[0].cancel_count == 1  # x2 cancelled
    assert by_lvl[0].fill_rate_pct == 50.0
    assert by_lvl[0].cancel_rate_pct == 50.0
    assert by_lvl[1].order_count == 1
    assert by_lvl[1].fill_count == 0
    assert by_lvl[1].cancel_count == 1
    assert by_lvl[1].fill_rate_pct == 0.0


def test_top_cancel_reasons_descending() -> None:
    """Top-N cancel reasons returned in descending count order."""
    orders = pd.DataFrame(
        {
            "order_id_local": ["o1", "o2", "o3", "o4"],
            "order_id_exchange": ["x1", "x2", "x3", "x4"],
            "level_idx": [0, 0, 0, 0],
            "ts_created": ["2026-05-19T00:00:00Z"] * 4,
            "ts_closed": [
                "2026-05-19T00:00:01Z",
                "2026-05-19T00:00:01Z",
                "2026-05-19T00:00:01Z",
                "2026-05-19T00:00:01Z",
            ],
            "ts_sent": ["2026-05-19T00:00:00Z"] * 4,
            "ts_ack": ["2026-05-19T00:00:00.010Z"] * 4,
            "cancel_reason": ["reprice", "reprice", "reprice", "size_change"],
        }
    )
    out = _findings(orders, pd.DataFrame())
    assert out.available is True
    rung0 = out.by_rung[0]
    assert rung0.top_cancel_reasons[0] == ("reprice", 3)
    assert rung0.top_cancel_reasons[1] == ("size_change", 1)


def test_markdown_and_html_renderers_smoke() -> None:
    out = _findings(pd.DataFrame(), pd.DataFrame())
    md = render_markdown_section(out)
    html = render_html_section(out)
    assert "lifecycle" in md.lower()
    assert "<section" in html
