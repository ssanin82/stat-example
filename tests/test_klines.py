"""Tests for the klines fetcher + regime summary used by the metrics
report and the standalone ``scripts/klines.py``.

The HTTP layer is mocked so tests stay deterministic and offline. The
regime classifier is tested with synthetic OHLCV that exercises each
label boundary.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tools.postmortem.render.charts import (
    KlinesFetchResult,
    fetch_binance_klines,
    pick_default_interval,
    save_klines_parquet,
    save_klines_table,
    summarise_regime,
)


# ----------------------- pick_default_interval ----------------------------


@pytest.mark.parametrize(
    "duration_s,expected",
    [
        (60 * 30, "1m"),       # 30 min
        (60 * 90, "1m"),       # 1.5 h
        (3600 * 4, "3m"),      # 4 h
        (3600 * 12, "5m"),     # 12 h
        (3600 * 48, "15m"),    # 2 days
        (3600 * 24 * 5, "30m"),  # 5 days
        (3600 * 24 * 30, "1h"),  # 1 month
    ],
)
def test_pick_default_interval(duration_s, expected) -> None:
    assert pick_default_interval(duration_s) == expected


# ----------------------- fetch_binance_klines -----------------------------


def _binance_kline_row(open_ms: int, *, o: float, h: float, l: float, c: float,
                       v: float = 1.0) -> list:
    return [
        open_ms,
        f"{o:.6f}",
        f"{h:.6f}",
        f"{l:.6f}",
        f"{c:.6f}",
        f"{v:.6f}",
        open_ms + 60_000 - 1,  # close_time
        f"{c * v:.6f}",        # quote_volume
        10,                    # trades
        f"{v / 2:.6f}",        # taker_buy_base
        f"{c * v / 2:.6f}",    # taker_buy_quote
        "0",                   # ignore
    ]


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _MockHttpxClient:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, params):
        self.calls.append({"url": url, "params": dict(params)})
        if not self._batches:
            return _MockResponse([])
        return _MockResponse(self._batches.pop(0))


def test_fetch_binance_klines_single_page() -> None:
    """One batch of 3 bars returns a 3-row frame."""
    t0 = int(datetime(2026, 4, 26, tzinfo=timezone.utc).timestamp() * 1000)
    batch = [
        _binance_kline_row(t0, o=1.0, h=1.01, l=0.99, c=1.005),
        _binance_kline_row(t0 + 60_000, o=1.005, h=1.02, l=1.0, c=1.015),
        _binance_kline_row(t0 + 120_000, o=1.015, h=1.025, l=1.01, c=1.02),
    ]
    mock = _MockHttpxClient([batch])
    with patch("tools.postmortem.render.charts.httpx.Client", return_value=mock):
        result = fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 0, 5, tzinfo=timezone.utc),
            interval="1m",
        )
    assert result.bar_count == 3
    assert result.symbol == "SUIUSDT"
    assert result.interval == "1m"
    assert result.df.iloc[0]["open"] == 1.0
    assert result.df.iloc[-1]["close"] == 1.02
    # index is open_time UTC
    assert result.df.index.tz is not None


def test_fetch_binance_klines_paginates_when_limit_hit() -> None:
    """A first batch at exactly the limit forces a follow-up request."""
    from tools.postmortem.render.charts import _BINANCE_KLINES_MAX_LIMIT

    t0 = int(datetime(2026, 4, 26, tzinfo=timezone.utc).timestamp() * 1000)
    full_batch = [
        _binance_kline_row(t0 + i * 60_000, o=1.0, h=1.0, l=1.0, c=1.0)
        for i in range(_BINANCE_KLINES_MAX_LIMIT)
    ]
    second = [
        _binance_kline_row(
            t0 + _BINANCE_KLINES_MAX_LIMIT * 60_000, o=1.0, h=1.0, l=1.0, c=1.0
        ),
    ]
    mock = _MockHttpxClient([full_batch, second])
    with patch("tools.postmortem.render.charts.httpx.Client", return_value=mock):
        result = fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 5, 1, tzinfo=timezone.utc),
            interval="1m",
        )
    assert result.bar_count == _BINANCE_KLINES_MAX_LIMIT + 1
    assert len(mock.calls) == 2


def test_fetch_binance_klines_empty_window_returns_empty_frame() -> None:
    mock = _MockHttpxClient([[]])
    with patch("tools.postmortem.render.charts.httpx.Client", return_value=mock):
        result = fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 0, 1, tzinfo=timezone.utc),
            interval="1m",
        )
    assert result.bar_count == 0
    assert result.df.empty


def test_fetch_binance_klines_rejects_unknown_interval() -> None:
    with pytest.raises(RuntimeError, match="unknown interval"):
        fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 1, tzinfo=timezone.utc),
            interval="7m",  # not supported
        )


def test_fetch_binance_klines_rejects_inverted_window() -> None:
    with pytest.raises(RuntimeError, match="must be after"):
        fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, 1, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 0, tzinfo=timezone.utc),
            interval="1m",
        )


# ----------------------- save_klines_parquet ------------------------------


def _make_synth_result_for_save(
    rows: int = 2,
) -> KlinesFetchResult:
    """Synthesise a KlinesFetchResult independent of HTTP mocking — used
    by the save-path tests that just exercise serialisation."""
    t0 = int(datetime(2026, 4, 26, tzinfo=timezone.utc).timestamp() * 1000)
    batch = [
        _binance_kline_row(t0 + i * 60_000, o=1.0, h=1.01, l=0.99, c=1.005)
        for i in range(rows)
    ]
    mock = _MockHttpxClient([batch])
    with patch("tools.postmortem.render.charts.httpx.Client", return_value=mock):
        return fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 0, rows + 5, tzinfo=timezone.utc),
            interval="1m",
        )


def test_save_klines_parquet_roundtrip(tmp_path: Path) -> None:
    """Parquet write + read returns the same column data."""
    pytest.importorskip("pyarrow")  # skip when no parquet engine in env
    t0 = int(datetime(2026, 4, 26, tzinfo=timezone.utc).timestamp() * 1000)
    batch = [
        _binance_kline_row(t0, o=1.0, h=1.01, l=0.99, c=1.005),
        _binance_kline_row(t0 + 60_000, o=1.005, h=1.02, l=1.0, c=1.015),
    ]
    mock = _MockHttpxClient([batch])
    with patch("tools.postmortem.render.charts.httpx.Client", return_value=mock):
        result = fetch_binance_klines(
            symbol="SUIUSDT",
            since_utc=datetime(2026, 4, 26, tzinfo=timezone.utc),
            until_utc=datetime(2026, 4, 26, 0, 5, tzinfo=timezone.utc),
            interval="1m",
        )
    out = tmp_path / "klines.parquet"
    saved = save_klines_table(result, out)
    assert saved.exists()
    assert saved.suffix == ".parquet"
    loaded = pd.read_parquet(saved)
    assert len(loaded) == 2
    assert pytest.approx(loaded["open"].iloc[0]) == 1.0
    assert pytest.approx(loaded["close"].iloc[-1]) == 1.015


def test_save_klines_falls_back_to_csv_when_parquet_engine_missing(
    tmp_path: Path,
) -> None:
    """Reproduces the operator's traceback (no pyarrow / fastparquet
    installed in the venv): the saver must NOT crash; it should write a
    CSV alongside, return the CSV path, and the metrics caller picks
    that path up to write the regime sidecar in the right place.
    """
    result = _make_synth_result_for_save(rows=3)
    out = tmp_path / "klines.parquet"

    # Patch DataFrame.to_parquet to raise the same ImportError the user hit.
    real_to_parquet = pd.DataFrame.to_parquet

    def boom(self, *args, **kwargs):
        raise ImportError(
            "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'."
        )

    with patch.object(pd.DataFrame, "to_parquet", boom):
        saved = save_klines_table(result, out)
    assert saved.exists()
    # Suffix swapped to .csv on fallback.
    assert saved.suffix == ".csv"
    # The original .parquet path was NOT created.
    assert not out.exists()
    # CSV is round-trip-readable by pandas.
    loaded = pd.read_csv(saved)
    assert len(loaded) == 3
    assert "open" in loaded.columns and "close" in loaded.columns
    # Sanity-check the original to_parquet binding is unaffected after the patch.
    assert pd.DataFrame.to_parquet is real_to_parquet


def test_save_klines_parquet_alias_still_works() -> None:
    """The legacy ``save_klines_parquet`` name is kept as an alias for
    back-compat (tests / external scripts that import it). It should
    point at the same callable as ``save_klines_table``.
    """
    assert save_klines_parquet is save_klines_table


# ----------------------- summarise_regime ---------------------------------


def _make_result_from_closes(closes: list[float], *, interval: str = "1m") -> KlinesFetchResult:
    """Synthesise a KlinesFetchResult with a fixed close path and constant
    open/high/low so vol/return computations exercise the close column."""
    t0 = datetime(2026, 4, 26, tzinfo=timezone.utc)
    rows = []
    for i, c in enumerate(closes):
        open_ts = t0 + timedelta(seconds=i * 60)
        close_ts = open_ts + timedelta(seconds=59)
        rows.append(
            {
                "open_time": open_ts,
                "open": closes[max(0, i - 1)] if i > 0 else c,
                "high": max(c, closes[max(0, i - 1)] if i > 0 else c),
                "low": min(c, closes[max(0, i - 1)] if i > 0 else c),
                "close": c,
                "volume": 1.0,
                "close_time": close_ts,
                "quote_volume": c,
                "trades": 1,
                "taker_buy_base": 0.5,
                "taker_buy_quote": c / 2,
            }
        )
    df = pd.DataFrame(rows).set_index("open_time")
    return KlinesFetchResult(
        df=df,
        symbol="SYNTH",
        interval=interval,
        requested_since_utc=t0,
        requested_until_utc=t0 + timedelta(seconds=60 * len(closes)),
        bar_count=len(df),
        source="synthetic",
    )


def test_regime_quiet_when_low_vol() -> None:
    """Closes barely move (sub-bp swings) → label 'quiet'."""
    closes = [1.0 + i * 1e-6 for i in range(60)]
    summary = summarise_regime(_make_result_from_closes(closes))
    assert summary.regime_label == "quiet"
    assert summary.realized_vol_per_bar_bps is not None
    assert summary.realized_vol_per_bar_bps < 5.0


def test_regime_volatile_on_huge_per_bar_swings() -> None:
    """Big random walk → 'volatile'."""
    rng = np.random.default_rng(42)
    closes = [1.0]
    for _ in range(100):
        closes.append(closes[-1] * (1.0 + rng.normal(0, 0.01)))
    summary = summarise_regime(_make_result_from_closes(closes))
    assert summary.regime_label == "volatile"
    assert summary.realized_vol_per_bar_bps > 30.0


def test_regime_trending_up_on_monotonic_climb() -> None:
    """Steady upward drift dominates the random walk → 'trending_up'."""
    closes = [1.0 + 0.001 * i for i in range(60)]  # +6% over 60 bars, low noise
    summary = summarise_regime(_make_result_from_closes(closes))
    assert summary.regime_label == "trending_up"
    assert summary.total_return_bps > 0


def test_regime_trending_down_on_monotonic_decline() -> None:
    closes = [1.0 - 0.001 * i for i in range(60)]
    summary = summarise_regime(_make_result_from_closes(closes))
    assert summary.regime_label == "trending_down"
    assert summary.total_return_bps < 0


def test_regime_insufficient_data_on_short_window() -> None:
    summary = summarise_regime(_make_result_from_closes([1.0, 1.0001]))
    assert summary.regime_label == "insufficient_data"


def test_regime_to_dict_round_floats() -> None:
    closes = [1.0 + 0.0005 * i for i in range(20)]
    summary = summarise_regime(_make_result_from_closes(closes))
    d = summary.to_dict()
    # Every float ends up rounded to 4 decimals; integers untouched.
    assert d["bar_count"] == 20
    assert isinstance(d["realized_vol_per_bar_bps"], float)
    assert "regime_label" in d


# ----------------------- klines CLI smoke ---------------------------------


def test_klines_script_argparse_last_form_parses() -> None:
    """The standalone script's `--last 30m` parser handles unit suffixes."""
    from scripts.klines import _parse_last_arg

    assert _parse_last_arg("30m") == timedelta(minutes=30)
    assert _parse_last_arg("2h") == timedelta(hours=2)
    assert _parse_last_arg("1d") == timedelta(days=1)
    assert _parse_last_arg("45s") == timedelta(seconds=45)


def test_klines_script_argparse_rejects_garbage() -> None:
    from scripts.klines import _parse_last_arg

    with pytest.raises(SystemExit):
        _parse_last_arg("yesterday")
