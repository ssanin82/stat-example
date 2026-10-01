from datetime import datetime, timedelta, timezone

from app.book_history import TopOfBookRow, match_top_of_book, snapshot_fullness


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_match_exact_or_prior_latest_before_fill() -> None:
    rows = [
        TopOfBookRow(_ts("2024-01-01T12:00:00+00:00"), 99.0, 101.0, 100.0),
        TopOfBookRow(_ts("2024-01-01T12:00:02+00:00"), 98.0, 102.0, 100.0),
    ]
    wide = timedelta(seconds=10)
    row, q = match_top_of_book(rows, _ts("2024-01-01T12:00:03+00:00"), wide)
    assert q == "exact_or_prior"
    assert row is not None and row.mid == 100.0 and row.best_bid == 98.0


def test_match_approximate_after_when_fill_before_buffer() -> None:
    rows = [
        TopOfBookRow(_ts("2024-01-01T12:00:05+00:00"), 97.0, 103.0, 100.0),
    ]
    wide = timedelta(seconds=10)
    row, q = match_top_of_book(rows, _ts("2024-01-01T12:00:01+00:00"), wide)
    assert q == "approximate_after"
    assert row is not None and row.best_bid == 97.0


def test_prior_outside_skew_is_missing_not_far_snapshot() -> None:
    """Do not attach a book row seconds before the fill when skew is tight."""
    rows = [
        TopOfBookRow(_ts("2024-01-01T12:00:00+00:00"), 99.0, 101.0, 100.0),
    ]
    fill = _ts("2024-01-01T12:00:01+00:00")
    row, q = match_top_of_book(rows, fill, timedelta(milliseconds=250))
    assert row is None and q == "missing_reference"


def test_prior_within_skew_used_before_after() -> None:
    base = _ts("2024-01-01T12:00:00+00:00")
    rows = [
        TopOfBookRow(base, 99.0, 101.0, 100.0),
        TopOfBookRow(_ts("2024-01-01T12:00:00.200000+00:00"), 98.0, 102.0, 100.0),
    ]
    fill = _ts("2024-01-01T12:00:00.300000+00:00")
    row, q = match_top_of_book(rows, fill, timedelta(milliseconds=250))
    assert q == "exact_or_prior"
    assert row is not None and row.best_bid == 98.0


def test_after_only_used_within_skew() -> None:
    fill = _ts("2024-01-01T12:00:00+00:00")
    rows = [
        TopOfBookRow(_ts("2024-01-01T12:00:00.100000+00:00"), 97.0, 103.0, 100.0),
    ]
    row, q = match_top_of_book(rows, fill, timedelta(milliseconds=250))
    assert q == "approximate_after"
    assert row is not None


def test_after_beyond_skew_is_missing() -> None:
    fill = _ts("2024-01-01T12:00:00+00:00")
    rows = [
        TopOfBookRow(_ts("2024-01-01T12:00:01+00:00"), 97.0, 103.0, 100.0),
    ]
    row, q = match_top_of_book(rows, fill, timedelta(milliseconds=250))
    assert row is None and q == "missing_reference"


def test_match_missing_when_empty() -> None:
    row, q = match_top_of_book([], _ts("2024-01-01T12:00:00+00:00"), timedelta(seconds=1))
    assert row is None and q == "missing_reference"


def test_snapshot_fullness_mid_only() -> None:
    assert snapshot_fullness(None, None, 100.0) == "mid_only"


def test_snapshot_fullness_full() -> None:
    assert snapshot_fullness(99.0, 101.0, 100.0) == "full"
