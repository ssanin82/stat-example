"""Tests for the postmortem package (formerly app.reporting).

Exchange-only architecture: the reporting layer reads from the Bluefin
REST API, never from local SQLite. These tests exercise the pure
calculation functions on synthetic DataFrames, plus the REST loader
via a mock ``client_factory``.

Coverage:
* PnL: spread capture, inventory PnL, decomposition residual, time
  series resampling.
* Risk: Sharpe / Sortino / Calmar / drawdown / win-loss / turnover /
  inventory.
* Bootstrap: insufficient-sample skip + happy-path distribution shape.
* Narrative: knowledge-bucket assembly + confidence tagging.
* Data ingestion: REST mock + audit-dump roundtrip + back-derived
  starting equity + missing-credentials guard.
* Render: end-to-end Markdown + HTML rendering.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tools.postmortem.loaders.rest_loader import (
    DataValidationError,
    ReportData,
    TimeWindow,
    load_report_data,
    save_raw_dump,
)
from tools.postmortem.render.html import (
    render_html_report,
    render_md_report,
    write_reports,
)
from tools.postmortem.sections.bootstrap import run_block_bootstrap
from tools.postmortem.sections.narrative import build_risk_profile
from tools.postmortem.sections.pnl import (
    compute_inventory_pnl,
    compute_pnl_summary,
    compute_spread_capture,
    hourly_equity_series,
    hourly_returns_series,
    per_symbol_breakdown,
)
from tools.postmortem.sections.risk import (
    compute_risk_metrics,
    drawdown_series,
    drawdown_stats,
    sharpe_ratio,
    sortino_ratio,
    turnover_stats,
    win_loss_stats,
)


# --- fixtures: synthetic DataFrames ---


def _ts(offset_hours: float) -> datetime:
    base = datetime(2026, 4, 23, 0, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(hours=offset_hours)


def _make_fills(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["ts_fill"] = pd.to_datetime(df["ts_fill"], utc=True)
    return df


def _make_equity(curve_usd: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for off, eq in curve_usd:
        rows.append(
            dict(
                ts=_ts(off),
                equity_usd=eq,
                cash_usd=eq,
                realized_pnl_usd=0.0,
                unrealized_pnl_usd=0.0,
                fees_usd=0.0,
                drawdown_usd=0.0,
            )
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _make_positions(qtys: list[tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for off, qty, mark in qtys:
        rows.append(
            dict(
                ts=_ts(off),
                symbol="SUI-PERP",
                position_qty=qty,
                avg_entry_price=mark,
                mark_price=mark,
                position_notional=qty * mark,
                unrealized_pnl_usd=0.0,
            )
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# --- pnl ---


def test_spread_capture_buy_below_mid_is_positive() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), symbol="X", side="BUY", price=2.10, size=10,
                 notional=21.0, fee=0.001, closed_pnl=0.0, mid_at_fill=2.105),
        ]
    )
    s = compute_spread_capture(fills)
    assert s.iloc[0] == pytest.approx(0.05)


def test_spread_capture_sell_above_mid_is_positive() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), symbol="X", side="SELL", price=2.11, size=10,
                 notional=21.1, fee=0.001, closed_pnl=0.0, mid_at_fill=2.105),
        ]
    )
    s = compute_spread_capture(fills)
    assert s.iloc[0] == pytest.approx(0.05)


def test_spread_capture_handles_missing_mid() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), symbol="X", side="BUY", price=2.10, size=10,
                 notional=21.0, fee=0.001, closed_pnl=0.0, mid_at_fill=None),
        ]
    )
    s = compute_spread_capture(fills)
    assert s.iloc[0] == 0.0


def test_inventory_pnl_basic() -> None:
    pos = _make_positions(
        [
            (0.0, 0.0, 2.00),
            (1.0, 10.0, 2.00),
            (2.0, 10.0, 2.10),
            (3.0, 0.0, 2.10),
        ]
    )
    assert compute_inventory_pnl(pos) == pytest.approx(1.0)


def test_inventory_pnl_empty_or_singleton() -> None:
    assert compute_inventory_pnl(pd.DataFrame()) == 0.0
    assert compute_inventory_pnl(_make_positions([(0.0, 5.0, 2.0)])) == 0.0


def test_pnl_summary_decomposition() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(1), symbol="X", side="BUY", price=2.10, size=10,
                 notional=21.0, fee=0.001, closed_pnl=0.0, mid_at_fill=2.105),
            dict(ts_fill=_ts(2), symbol="X", side="SELL", price=2.20, size=10,
                 notional=22.0, fee=0.001, closed_pnl=1.0, mid_at_fill=2.195),
        ]
    )
    equity = _make_equity([(0.0, 100.0), (3.0, 101.0)])
    positions = _make_positions(
        [(0.0, 0.0, 2.10), (1.5, 10.0, 2.10), (3.0, 0.0, 2.20)]
    )
    summary = compute_pnl_summary(
        fills=fills, equity=equity, positions=positions
    )
    assert summary.starting_equity_usd == 100.0
    assert summary.ending_equity_usd == 101.0
    assert summary.total_pnl_usd == pytest.approx(1.0)
    assert summary.realised_pnl_usd == pytest.approx(1.0)
    assert summary.fees_usd == pytest.approx(0.002)
    assert summary.spread_capture_usd == pytest.approx(0.1)
    assert any("funding" in a.lower() for a in summary.assumptions)


def test_pnl_summary_handles_empty_window() -> None:
    summary = compute_pnl_summary(
        fills=pd.DataFrame(
            columns=[
                "mid_at_fill", "price", "size", "side",
                "fee", "closed_pnl", "notional",
            ]
        ),
        equity=pd.DataFrame(),
        positions=pd.DataFrame(),
    )
    assert summary.total_pnl_usd is None
    assert summary.realised_pnl_usd == 0.0


def test_pnl_summary_assumes_flat_start_when_first_unrealised_nan() -> None:
    """REST-derived equity has the first row's unrealised = NaN. The
    PnL summary should treat that as 'started flat' and surface the
    assumption."""
    fills = _make_fills(
        [
            dict(ts_fill=_ts(1), symbol="X", side="BUY", price=2.10, size=10,
                 notional=21.0, fee=0.001, closed_pnl=0.0, mid_at_fill=None),
            dict(ts_fill=_ts(2), symbol="X", side="SELL", price=2.20, size=10,
                 notional=22.0, fee=0.001, closed_pnl=1.0, mid_at_fill=None),
        ]
    )
    equity = pd.DataFrame(
        [
            dict(
                ts=_ts(1), equity_usd=100.0, cash_usd=np.nan,
                realized_pnl_usd=0.0, unrealized_pnl_usd=np.nan,
                fees_usd=0.001, drawdown_usd=np.nan,
            ),
            dict(
                ts=_ts(2), equity_usd=101.5, cash_usd=np.nan,
                realized_pnl_usd=1.0, unrealized_pnl_usd=0.5,
                fees_usd=0.002, drawdown_usd=np.nan,
            ),
        ]
    )
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True)
    positions = pd.DataFrame()
    summary = compute_pnl_summary(fills=fills, equity=equity, positions=positions)
    assert summary.unrealised_pnl_change_usd == pytest.approx(0.5)
    assert any("flat" in a.lower() for a in summary.assumptions)


def test_hourly_equity_series_resamples() -> None:
    eq = _make_equity([(0.0, 100.0), (0.5, 100.5), (1.5, 101.0), (3.0, 102.0)])
    s = hourly_equity_series(eq)
    assert len(s) >= 4
    assert s.iloc[-1] == 102.0


def test_hourly_returns_drops_first() -> None:
    eq = _make_equity([(0.0, 100.0), (1.0, 101.0), (2.0, 99.0)])
    rets = hourly_returns_series(eq)
    assert len(rets) >= 2
    assert rets.iloc[0] == pytest.approx(0.01, abs=1e-6)


def test_per_symbol_breakdown_groups_correctly() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="SUI-PERP", side="BUY",
                 price=2.0, size=10, notional=20, fee=0.001, closed_pnl=0,
                 mid_at_fill=2.0),
            dict(ts_fill=_ts(0), fill_id="b", symbol="SUI-PERP", side="SELL",
                 price=2.05, size=10, notional=20.5, fee=0.001, closed_pnl=0.5,
                 mid_at_fill=2.0),
            dict(ts_fill=_ts(0), fill_id="c", symbol="ETH-PERP", side="BUY",
                 price=3000, size=0.001, notional=3, fee=0.0001, closed_pnl=0,
                 mid_at_fill=3000),
        ]
    )
    df = per_symbol_breakdown(fills)
    assert len(df) == 2
    sui_row = df[df["symbol"] == "SUI-PERP"].iloc[0]
    assert sui_row["fills"] == 2
    assert sui_row["realised_pnl_usd"] == pytest.approx(0.5)


# --- risk ---


def test_sharpe_zero_when_zero_variance() -> None:
    rets = pd.Series([0.0, 0.0, 0.0, 0.0])
    per, ann = sharpe_ratio(rets)
    assert per is None and ann is None


def test_sharpe_basic() -> None:
    rets = pd.Series([0.001, 0.0005, 0.002, 0.0008, 0.0015, 0.0012])
    per, ann = sharpe_ratio(rets)
    assert per is not None and per > 0
    assert ann is not None and ann > 0


def test_sortino_handles_no_downside() -> None:
    rets = pd.Series([0.001, 0.002, 0.003])
    per, ann = sortino_ratio(rets)
    assert per is None


def test_drawdown_simple() -> None:
    eq = pd.Series(
        [100.0, 105.0, 110.0, 95.0, 100.0, 90.0, 92.0],
        index=pd.date_range("2026-01-01", periods=7, freq="1h", tz="UTC"),
    )
    dd = drawdown_series(eq)
    assert dd.min() == pytest.approx(-20.0 / 110.0, abs=1e-6)
    stats = drawdown_stats(eq)
    assert stats["max_dd_pct"] == pytest.approx(-100 * 20 / 110, abs=1e-3)
    assert stats["time_under_water_pct"] == pytest.approx(100 * 4 / 7, abs=1e-3)


def test_win_loss_stats_excludes_zero_closed_pnl() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="X", side="BUY", price=1,
                 size=1, notional=1, fee=0, closed_pnl=0, mid_at_fill=1),
            dict(ts_fill=_ts(0), fill_id="b", symbol="X", side="SELL", price=1,
                 size=1, notional=1, fee=0, closed_pnl=0.5, mid_at_fill=1),
            dict(ts_fill=_ts(0), fill_id="c", symbol="X", side="SELL", price=1,
                 size=1, notional=1, fee=0, closed_pnl=-0.3, mid_at_fill=1),
        ]
    )
    stats = win_loss_stats(fills)
    assert stats["n_fills"] == 3
    assert stats["n_winning_fills"] == 1
    assert stats["n_losing_fills"] == 1
    assert stats["win_rate_pct"] == pytest.approx(50.0)
    assert stats["avg_win_usd"] == pytest.approx(0.5)
    assert stats["avg_loss_usd"] == pytest.approx(-0.3)
    assert stats["profit_factor"] == pytest.approx(0.5 / 0.3)


def test_turnover_stats() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="X", side="BUY", price=1,
                 size=10, notional=10, fee=0.005, closed_pnl=0, mid_at_fill=1),
            dict(ts_fill=_ts(0), fill_id="b", symbol="X", side="SELL", price=1,
                 size=10, notional=10, fee=0.005, closed_pnl=0, mid_at_fill=1),
        ]
    )
    eq = _make_equity([(0.0, 100.0), (1.0, 100.0)])
    s = turnover_stats(fills, eq)
    assert s["turnover_usd"] == 20.0
    assert s["turnover_ratio"] == pytest.approx(0.2)
    assert s["fee_drag_pct"] == pytest.approx(0.01, abs=1e-6)


def test_compute_risk_metrics_smoke() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(i), fill_id=f"f{i}", symbol="X", side="BUY",
                 price=2.0, size=5, notional=10, fee=0.001, closed_pnl=0.05,
                 mid_at_fill=2.005)
            for i in range(24)
        ]
    )
    equity = _make_equity([(i, 100.0 + 0.05 * i) for i in range(25)])
    positions = _make_positions([(i, 0.0, 2.0) for i in range(25)])
    metrics = compute_risk_metrics(fills=fills, equity=equity, positions=positions)
    assert metrics.n_fills == 24
    assert metrics.win_rate_pct == 100.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.sharpe_ratio_hourly is not None
    assert metrics.sharpe_ratio_hourly > 0
    assert metrics.time_under_water_pct == pytest.approx(0.0)


# --- bootstrap ---


def test_bootstrap_skipped_on_insufficient_sample() -> None:
    rets = pd.Series([0.001] * 10)
    r = run_block_bootstrap(rets, n_simulations=100, block_size=4)
    assert r.skipped is True
    assert "24" in r.skip_reason


def test_bootstrap_runs_and_returns_distribution() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.005, size=48))
    r = run_block_bootstrap(rets, n_simulations=500, block_size=6, seed=42)
    assert r.skipped is False
    assert r.n_simulations == 500
    assert r.terminal_p05_pct is not None
    assert r.terminal_mean_pct is not None
    assert r.terminal_p05_pct < r.terminal_median_pct < r.terminal_p95_pct


# --- narrative ---


def test_build_risk_profile_short_sample_flags_insufficient() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="X", side="BUY", price=1,
                 size=1, notional=1, fee=0, closed_pnl=0, mid_at_fill=1),
        ]
    )
    equity = _make_equity([(0.0, 100.0), (1.0, 100.5)])
    positions = _make_positions([(0.0, 0.0, 2.0)])

    pnl = compute_pnl_summary(fills=fills, equity=equity, positions=positions)
    risk = compute_risk_metrics(fills=fills, equity=equity, positions=positions)
    rets = hourly_returns_series(equity)
    boot = run_block_bootstrap(rets, n_simulations=100, block_size=4)

    profile = build_risk_profile(
        pnl=pnl, risk=risk, bootstrap=boot, duration_hours=1.0
    )
    assert profile.confidence_level == "insufficient"
    assert "Insufficient" in " ".join(profile.layer4_robustness)
    assert "known" in profile.knowledge_buckets
    assert "estimated" in profile.knowledge_buckets
    assert "unreliable_for_short_sample" in profile.knowledge_buckets


# --- render ---


def _build_synthetic_report_data(
    fills: pd.DataFrame,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    starting_equity: Optional[float] = None,
    ending_equity: Optional[float] = None,
) -> ReportData:
    return ReportData(
        window=TimeWindow(since_utc=_ts(0), until_utc=_ts(48)),
        fills=fills,
        equity=equity,
        positions=positions,
        bot_events=pd.DataFrame(
            columns=["ts", "severity", "event_type", "message", "payload"]
        ),
        starting_equity_inferred_usd=starting_equity,
        ending_equity_observed_usd=ending_equity,
    )


def test_render_md_smokes_through() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="SUI-PERP", side="BUY",
                 price=2.10, size=10, notional=21, fee=0.001, closed_pnl=0.0,
                 mid_at_fill=2.105),
            dict(ts_fill=_ts(1), fill_id="b", symbol="SUI-PERP", side="SELL",
                 price=2.20, size=10, notional=22, fee=0.001, closed_pnl=1.0,
                 mid_at_fill=2.195),
        ]
    )
    equity = _make_equity([(i, 100.0 + 0.05 * i) for i in range(48)])
    positions = _make_positions([(i, 0.0, 2.10) for i in range(48)])

    pnl = compute_pnl_summary(fills=fills, equity=equity, positions=positions)
    risk = compute_risk_metrics(fills=fills, equity=equity, positions=positions)
    rets = hourly_returns_series(equity)
    boot = run_block_bootstrap(rets, n_simulations=200, block_size=6, seed=1)
    profile = build_risk_profile(
        pnl=pnl, risk=risk, bootstrap=boot, duration_hours=48.0
    )
    data = _build_synthetic_report_data(fills, equity, positions)

    md = render_md_report(
        data=data, pnl=pnl, risk=risk, bootstrap=boot, profile=profile,
        generated_at=_ts(48),
    )
    assert "# Postmortem metrics report" in md
    assert "Executive summary" in md
    assert "Per-symbol breakdown" in md
    assert "SUI-PERP" in md


def test_render_html_contains_charts() -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="X", side="BUY", price=1,
                 size=1, notional=1, fee=0, closed_pnl=0, mid_at_fill=1),
        ]
    )
    equity = _make_equity([(i, 100.0 + 0.05 * i) for i in range(48)])
    positions = _make_positions([(i, 0.0, 2.0) for i in range(48)])

    pnl = compute_pnl_summary(fills=fills, equity=equity, positions=positions)
    risk = compute_risk_metrics(fills=fills, equity=equity, positions=positions)
    rets = hourly_returns_series(equity)
    boot = run_block_bootstrap(rets, n_simulations=200, block_size=6, seed=1)
    profile = build_risk_profile(
        pnl=pnl, risk=risk, bootstrap=boot, duration_hours=48.0
    )
    data = _build_synthetic_report_data(fills, equity, positions)
    html = render_html_report(
        data=data, pnl=pnl, risk=risk, bootstrap=boot, profile=profile,
        generated_at=_ts(48),
    )
    assert "<html" in html
    assert html.count("data:image/png;base64,") >= 4
    assert "Postmortem metrics report" in html


def test_write_reports_creates_both_files(tmp_path: Path) -> None:
    fills = _make_fills(
        [
            dict(ts_fill=_ts(0), fill_id="a", symbol="X", side="BUY", price=1,
                 size=1, notional=1, fee=0, closed_pnl=0, mid_at_fill=1),
        ]
    )
    equity = _make_equity([(i, 100.0 + 0.05 * i) for i in range(48)])
    positions = _make_positions([(i, 0.0, 2.0) for i in range(48)])

    pnl = compute_pnl_summary(fills=fills, equity=equity, positions=positions)
    risk = compute_risk_metrics(fills=fills, equity=equity, positions=positions)
    rets = hourly_returns_series(equity)
    boot = run_block_bootstrap(rets, n_simulations=200, block_size=6, seed=1)
    profile = build_risk_profile(
        pnl=pnl, risk=risk, bootstrap=boot, duration_hours=48.0
    )
    data = _build_synthetic_report_data(fills, equity, positions)
    out_dir = tmp_path / "reports"
    h, m = write_reports(
        out_dir=out_dir, data=data, pnl=pnl, risk=risk, bootstrap=boot,
        profile=profile, generated_at=_ts(48),
    )
    assert h.is_file() and m.is_file()
    assert h.suffix == ".html" and m.suffix == ".md"


# --- REST loader (mocked Bluefin client) ---


@dataclass
class _RawFill:
    """Mimic ``HLFillRaw`` shape used by ``fetch_recent_fills_raw``."""

    fill_id: str
    oid: Any
    coin: str
    side: Any
    px: float
    sz: float
    fee: float
    time_ms: int
    closed_pnl: float
    raw: dict


class _Side:
    """Side enum stand-in. ``.value`` returns 'BUY' / 'SELL'."""

    def __init__(self, val: str) -> None:
        self.value = val

    def __str__(self) -> str:
        return self.value


def _build_fake_fills(n: int = 12, *, base_ms: int = 1_777_000_000_000) -> list:
    """Synthesise n fills, half BUY, half SELL at slightly different
    prices so closed_pnl is a non-trivial sequence."""
    out = []
    for i in range(n):
        out.append(
            _RawFill(
                fill_id=f"f{i}",
                oid=None,
                coin="SUI-PERP",
                side=_Side("BUY") if i % 2 == 0 else _Side("SELL"),
                px=0.95 + i * 0.0001,
                sz=10.0,
                fee=0.0005,
                time_ms=base_ms + i * 60_000,  # one fill per minute
                closed_pnl=0.0 if i % 2 == 0 else 0.05,
                raw={},
            )
        )
    return out


class _FakeBluefinClient:
    """Test double for the BluefinClient REST surface used by the
    reporting loader."""

    def __init__(
        self,
        *,
        fills: Optional[list] = None,
        account: Optional[Any] = None,
        position: Optional[Any] = None,
        best_bid_ask: Optional[Any] = None,
        open_orders: Optional[list] = None,
    ) -> None:
        self._fills = fills if fills is not None else _build_fake_fills()
        self._account = (
            account if account is not None
            else {"equity_usd": 422.10, "withdrawable_usd": 422.10}
        )
        self._position = (
            position if position is not None
            else {"position_qty": 0.0, "mark_price": 0.95, "unrealized_pnl_usd": 0.0}
        )
        self._best = (
            best_bid_ask if best_bid_ask is not None
            else {"best_bid": 0.949, "best_ask": 0.951}
        )
        self._open_orders = open_orders if open_orders is not None else []

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list:
        return list(self._fills)

    def fetch_account_snapshot(self, address: str):
        return self._account

    def fetch_position(self, address: str, symbol: str):
        return self._position

    def fetch_best_bid_ask(self, symbol: str):
        return self._best

    def fetch_open_orders_raw(self, address: str) -> list:
        return list(self._open_orders)


@dataclass
class _Settings:
    """Minimal Settings stand-in for the REST loader tests."""

    bluefin_account_address: str = "0xdeadbeef" + "00" * 30
    symbol: str = "SUI-PERP"
    bluefin_private_key: str = "00" * 32
    exchange: str = "bluefin"


def test_load_report_data_via_rest_mock_happy_path(tmp_path: Path) -> None:
    fake = _FakeBluefinClient()
    data = load_report_data(
        settings=_Settings(),
        window=TimeWindow(),
        raw_dump_dir=tmp_path / "raw",
        client_factory=lambda s: fake,
    )
    # 12 fills derived from REST mock
    assert len(data.fills) == 12
    # equity curve ends at the snapshot equity
    assert data.equity.iloc[-1]["equity_usd"] == pytest.approx(422.10)
    # ending_equity_observed reflects the snapshot
    assert data.ending_equity_observed_usd == pytest.approx(422.10)
    # raw dump landed
    assert any(p.is_file() for p in (tmp_path / "raw").glob("bluefin_*.json"))
    # Notes mention funding/deposits/spread-capture limitations
    notes_blob = " ".join(data.notes).lower()
    assert "funding" in notes_blob
    assert "deposits" in notes_blob
    assert "spread-capture" in notes_blob or "spread_capture" in notes_blob or "mid_at_fill" in notes_blob


def test_load_report_data_missing_address_raises(tmp_path: Path) -> None:
    s = _Settings(bluefin_account_address="")
    with pytest.raises(DataValidationError):
        load_report_data(
            settings=s, window=TimeWindow(),
            raw_dump_dir=tmp_path / "raw",
            client_factory=lambda s: _FakeBluefinClient(),
        )


def test_load_report_data_missing_symbol_raises(tmp_path: Path) -> None:
    s = _Settings(symbol="")
    with pytest.raises(DataValidationError):
        load_report_data(
            settings=s, window=TimeWindow(),
            raw_dump_dir=tmp_path / "raw",
            client_factory=lambda s: _FakeBluefinClient(),
        )


def test_load_report_data_rest_failure_raises(tmp_path: Path) -> None:
    """If the fills endpoint outright fails we cannot build a report."""
    class _BoomClient(_FakeBluefinClient):
        def fetch_recent_fills_raw(self, address, symbol):
            raise RuntimeError("connection reset")
    with pytest.raises(DataValidationError):
        load_report_data(
            settings=_Settings(), window=TimeWindow(),
            raw_dump_dir=tmp_path / "raw",
            client_factory=lambda s: _BoomClient(),
        )


def test_load_report_data_missing_account_equity_falls_back(tmp_path: Path) -> None:
    """When the account snapshot lacks usable equity fields, the loader
    anchors equity at 1.0 dimensionless, records the assumption in
    ``notes`` and still produces a usable report."""
    fake = _FakeBluefinClient(account={"foo": "bar"})
    data = load_report_data(
        settings=_Settings(), window=TimeWindow(),
        raw_dump_dir=tmp_path / "raw",
        client_factory=lambda s: fake,
    )
    assert data.ending_equity_observed_usd is None
    notes_blob = " ".join(data.notes).lower()
    assert "1.0" in notes_blob


def test_load_report_data_fills_window_filter_applies(tmp_path: Path) -> None:
    fills = _build_fake_fills(n=10, base_ms=1_777_000_000_000)
    fake = _FakeBluefinClient(fills=fills)
    # Drop the first 5 fills via since
    since = datetime.fromtimestamp(
        (1_777_000_000_000 + 5 * 60_000) / 1000.0, tz=timezone.utc
    )
    data = load_report_data(
        settings=_Settings(),
        window=TimeWindow(since_utc=since),
        raw_dump_dir=tmp_path / "raw",
        client_factory=lambda s: fake,
    )
    assert len(data.fills) == 5
    assert data.fills.iloc[0]["fill_id"] == "f5"


def test_load_report_data_full_page_appends_pagination_note(tmp_path: Path) -> None:
    """When REST returns the maximum page (200), the loader notes it."""
    fills = _build_fake_fills(n=200, base_ms=1_777_000_000_000)
    fake = _FakeBluefinClient(fills=fills)
    data = load_report_data(
        settings=_Settings(), window=TimeWindow(),
        raw_dump_dir=tmp_path / "raw",
        client_factory=lambda s: fake,
    )
    notes_blob = " ".join(data.notes).lower()
    assert "200" in notes_blob


def test_save_raw_dump_creates_audit_json(tmp_path: Path) -> None:
    p = save_raw_dump(
        tmp_path / "raw",
        datetime(2026, 4, 25, 10, 30, 0, tzinfo=timezone.utc),
        {"hello": "world", "n": 3},
    )
    assert p.is_file()
    j = json.loads(p.read_text(encoding="utf-8"))
    assert j["hello"] == "world"


# --- end-to-end with mock REST ---


def test_end_to_end_via_rest_mock(tmp_path: Path) -> None:
    """REST mock → load → compute → render → write. Covers the path
    the operator runs in production."""
    # Build a 48-fill series spanning ~48 hours so bootstrap doesn't skip.
    base_ms = int(datetime(2026, 4, 23, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    fills = []
    for i in range(48):
        fills.append(
            _RawFill(
                fill_id=f"f{i}",
                oid=None,
                coin="SUI-PERP",
                side=_Side("BUY") if i % 2 == 0 else _Side("SELL"),
                px=0.95 + i * 0.0001,
                sz=10.0,
                fee=0.0005,
                time_ms=base_ms + i * 3600_000,  # one per hour
                closed_pnl=0.0 if i % 2 == 0 else 0.05,
                raw={},
            )
        )
    fake = _FakeBluefinClient(fills=fills)

    data = load_report_data(
        settings=_Settings(), window=TimeWindow(),
        raw_dump_dir=tmp_path / "raw",
        client_factory=lambda s: fake,
    )

    pnl = compute_pnl_summary(
        fills=data.fills, equity=data.equity, positions=data.positions,
        funding_usd=data.funding_usd,
    )
    risk = compute_risk_metrics(
        fills=data.fills, equity=data.equity, positions=data.positions,
        funding_usd=data.funding_usd,
    )
    rets = hourly_returns_series(data.equity)
    boot = run_block_bootstrap(rets, n_simulations=200, block_size=6, seed=1)
    profile = build_risk_profile(
        pnl=pnl, risk=risk, bootstrap=boot,
        duration_hours=data.window.duration_hours or 0.0,
        extra_assumptions=data.notes,
    )

    out_dir = tmp_path / "reports"
    html_path, md_path = write_reports(
        out_dir=out_dir, data=data, pnl=pnl, risk=risk, bootstrap=boot,
        profile=profile, generated_at=datetime.now(timezone.utc),
    )
    assert html_path.is_file()
    assert md_path.is_file()
    assert "Postmortem metrics report" in html_path.read_text(encoding="utf-8")
    md_blob = md_path.read_text(encoding="utf-8")
    # Per-symbol breakdown should list the synthesised symbol.
    assert "SUI-PERP" in md_blob
    # Risk profile knowledge buckets should be present.
    assert "Known" in md_blob or "known" in md_blob


# ---------------------------------------------------------------------------
# Snapshot autodetect (2026-05-12: hyphenated filename format)
# ---------------------------------------------------------------------------


def test_find_newest_snapshot_handles_hyphenated_names(tmp_path) -> None:
    """``tools.postmortem._find_newest_snapshot`` must accept the
    current ``YYMMDD-HHMMSS`` folder-name format (introduced 2026-05-
    11) in addition to the legacy ``YYMMDDHHMMSS`` compact form, and
    must sort them correctly into chronological order regardless of
    which form they're in."""
    from tools.postmortem.__main__ import _find_newest_snapshot

    # Build 3 snapshot folders in a temp dir; mix old + new formats.
    names = [
        "snapshot_x.y.z_260510033208",                  # legacy compact
        "snapshot_x.y.z_260511-215321-evening",         # new with tag
        "snapshot_x.y.z_260512-084302",                 # new bare
    ]
    for n in names:
        d = tmp_path / n
        d.mkdir()
        (d / "meta.json").write_text("{}", encoding="utf-8")
    # Add a non-snapshot dir; it must not be picked.
    (tmp_path / "SAVED").mkdir()
    (tmp_path / "unrelated").mkdir()

    newest = _find_newest_snapshot(tmp_path)
    assert newest is not None
    assert newest.name == "snapshot_x.y.z_260512-084302"


def test_find_newest_snapshot_returns_none_on_empty_dir(tmp_path) -> None:
    from tools.postmortem.__main__ import _find_newest_snapshot

    assert _find_newest_snapshot(tmp_path) is None


def test_find_newest_snapshot_skips_dirs_without_meta(tmp_path) -> None:
    """A folder that matches the suffix regex but has no
    ``meta.json`` is skipped — it's not a real snapshot."""
    from tools.postmortem.__main__ import _find_newest_snapshot

    legit = tmp_path / "snapshot_x.y.z_260512-084302"
    legit.mkdir()
    (legit / "meta.json").write_text("{}", encoding="utf-8")

    fake = tmp_path / "snapshot_x.y.z_260513-091500"
    fake.mkdir()  # no meta.json

    newest = _find_newest_snapshot(tmp_path)
    assert newest is not None
    assert newest.name == "snapshot_x.y.z_260512-084302"
