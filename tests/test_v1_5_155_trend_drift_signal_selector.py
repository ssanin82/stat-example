"""v1.5.155 — tests for ``Bot._select_trend_drift_signal``.

The constructive trend skew in ``app/quoting.py:compute_quote_decision``
shifts the reservation by ``trend_drift_reservation_alpha * drift_bps``.
Pre-v1.5.155 ``drift_bps`` was hardcoded to ``eff_q.mid_return_250ms_bps``
— a 250-millisecond window so short that in any sustained trend the
alpha-weighted shift was sub-bp. The bot quoted symmetrically around
mid in trends and got adversely selected on both sides. See
``snapshots/v1.5.154-260526-074029`` for the canonical failure (67%
time in CAUTIOUS, 32% win rate, -2.6 bps mean 5 s markout).

v1.5.155 added ``TREND_DRIFT_SIGNAL_WINDOW_SECONDS`` so the
constructive skew can read from longer-horizon drift signals
(``state.mid_drift_windows`` 5/10/30/60 s windows). The selector logic
plus the fallback chain are tested here.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.bot import Bot
from app.config import Settings
from app.mid_drift_windows import MidDriftWindows
from app.quote_eligibility import QuoteEligibilityResult
from app.state import BotState


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@dataclass
class _SelectorHarness:
    """Minimal stub providing the attributes
    ``_select_trend_drift_signal`` touches. Letting tests invoke the
    bound method without spinning up a real Bot."""

    _settings: Settings
    _state: BotState


def _make_harness(
    *,
    window_s: float = 0.25,
    mdw: Optional[MidDriftWindows] = None,
) -> _SelectorHarness:
    settings = Settings(
        TREND_DRIFT_SIGNAL_WINDOW_SECONDS=window_s,
    )
    state = BotState(settings)
    state.mid_drift_windows = mdw
    return _SelectorHarness(_settings=settings, _state=state)


def _eff_q(
    *,
    mid_return_250ms_bps: Optional[float] = None,
    mid_return_500ms_bps: Optional[float] = None,
) -> QuoteEligibilityResult:
    """Fresh eligibility result with the two fast-drift fields filled.
    The selector only reads those two attributes via ``getattr``."""
    return QuoteEligibilityResult(
        eligibility=None,  # selector doesn't touch eligibility
        reason="",
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=0.0,
        market_data_gap_p95_ms=None,
        market_data_gap_median_ms=None,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=mid_return_250ms_bps,
        mid_return_500ms_bps=mid_return_500ms_bps,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
    )


def _mdw(
    *,
    d_500ms: Optional[float] = None,
    d_5s: Optional[float] = None,
    d_10s: Optional[float] = None,
    d_30s: Optional[float] = None,
    d_60s: Optional[float] = None,
    d_5min: Optional[float] = None,
    d_15min: Optional[float] = None,
) -> MidDriftWindows:
    return MidDriftWindows(
        drift_500ms_bps=d_500ms,
        drift_5s_bps=d_5s,
        drift_10s_bps=d_10s,
        drift_30s_bps=d_30s,
        drift_60s_bps=d_60s,
        drift_5min_bps=d_5min,
        drift_15min_bps=d_15min,
    )


# ---------------------------------------------------------------------------
# Legacy path: window <= 0.3 → mid_return_250ms_bps
# ---------------------------------------------------------------------------


def test_default_window_returns_250ms_signal():
    """Default ``TREND_DRIFT_SIGNAL_WINDOW_SECONDS=0.25`` preserves
    pre-v1.5.155 behaviour: returns ``eff_q.mid_return_250ms_bps``
    even when longer windows are populated."""
    h = _make_harness(window_s=0.25, mdw=_mdw(d_10s=-10.0))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -0.5


def test_zero_window_returns_250ms_signal():
    """Window=0 also takes the legacy path (clamped to <= 0.3)."""
    h = _make_harness(window_s=0.0, mdw=_mdw(d_10s=-10.0))
    eff_q = _eff_q(mid_return_250ms_bps=1.7)
    assert Bot._select_trend_drift_signal(h, eff_q) == 1.7


def test_legacy_window_returns_none_when_250ms_missing():
    """No mid_return_250ms_bps and no longer windows → ``None``."""
    h = _make_harness(window_s=0.25, mdw=None)
    eff_q = _eff_q(mid_return_250ms_bps=None)
    assert Bot._select_trend_drift_signal(h, eff_q) is None


# ---------------------------------------------------------------------------
# 500 ms path: 0.3 < window <= 0.6
# ---------------------------------------------------------------------------


def test_500ms_window_returns_500ms_signal():
    """``window=0.5`` returns ``eff_q.mid_return_500ms_bps``."""
    h = _make_harness(window_s=0.5)
    eff_q = _eff_q(mid_return_250ms_bps=-0.5, mid_return_500ms_bps=-1.2)
    assert Bot._select_trend_drift_signal(h, eff_q) == -1.2


def test_500ms_window_falls_back_to_250ms_when_500ms_missing():
    """``window=0.5`` but mid_return_500ms_bps is ``None`` →
    falls back to mid_return_250ms_bps."""
    h = _make_harness(window_s=0.5)
    eff_q = _eff_q(mid_return_250ms_bps=2.0, mid_return_500ms_bps=None)
    assert Bot._select_trend_drift_signal(h, eff_q) == 2.0


# ---------------------------------------------------------------------------
# 5 s path: 0.6 < window <= 7.5
# ---------------------------------------------------------------------------


def test_5s_window_returns_drift_5s():
    """``window=5.0`` returns ``mid_drift_windows.drift_5s_bps``."""
    h = _make_harness(window_s=5.0, mdw=_mdw(d_5s=-8.3, d_10s=-10.5))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -8.3


def test_5s_window_warmup_falls_back_to_500ms():
    """``window=5.0`` but drift_5s is ``None`` → falls back to
    mid_return_500ms_bps."""
    h = _make_harness(window_s=5.0, mdw=_mdw(d_5s=None))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5, mid_return_500ms_bps=-0.9)
    assert Bot._select_trend_drift_signal(h, eff_q) == -0.9


def test_5s_window_warmup_falls_back_to_250ms_when_500ms_also_missing():
    h = _make_harness(window_s=5.0, mdw=_mdw(d_5s=None))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5, mid_return_500ms_bps=None)
    assert Bot._select_trend_drift_signal(h, eff_q) == -0.5


# ---------------------------------------------------------------------------
# 10 s path: 7.5 < window <= 20.0
# ---------------------------------------------------------------------------


def test_10s_window_returns_drift_10s():
    """``window=10.0`` returns ``mid_drift_windows.drift_10s_bps``."""
    h = _make_harness(window_s=10.0, mdw=_mdw(d_5s=-8.3, d_10s=-10.5))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -10.5


def test_10s_window_warmup_falls_back_through_chain():
    """``window=10.0`` but drift_10s is ``None`` → falls through to
    drift_5s → mid_return_500ms_bps → mid_return_250ms_bps."""
    h = _make_harness(
        window_s=10.0,
        mdw=_mdw(d_5s=-7.1, d_10s=None),
    )
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -7.1


def test_10s_window_warmup_no_mid_drift_windows_at_all():
    """``state.mid_drift_windows`` is ``None`` (very early lifetime)
    → falls back directly to fast eligibility-snapshot signals."""
    h = _make_harness(window_s=10.0, mdw=None)
    eff_q = _eff_q(mid_return_250ms_bps=-0.5, mid_return_500ms_bps=-0.9)
    assert Bot._select_trend_drift_signal(h, eff_q) == -0.9


def test_10s_window_returns_none_when_everything_warming_up():
    """All sources None → returns None (caller treats as no skew)."""
    h = _make_harness(window_s=10.0, mdw=_mdw())
    eff_q = _eff_q(mid_return_250ms_bps=None, mid_return_500ms_bps=None)
    assert Bot._select_trend_drift_signal(h, eff_q) is None


# ---------------------------------------------------------------------------
# 30 s and 60 s paths
# ---------------------------------------------------------------------------


def test_30s_window_returns_drift_30s():
    h = _make_harness(window_s=30.0, mdw=_mdw(d_10s=-10.5, d_30s=-15.0))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -15.0


def test_60s_window_returns_drift_60s():
    h = _make_harness(window_s=60.0, mdw=_mdw(d_30s=-15.0, d_60s=-20.0))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -20.0


def test_high_window_clamps_to_60s():
    """Any window > 45 maps to drift_60s."""
    h = _make_harness(window_s=60.0, mdw=_mdw(d_60s=-7.0))
    eff_q = _eff_q(mid_return_250ms_bps=-0.5)
    assert Bot._select_trend_drift_signal(h, eff_q) == -7.0


# ---------------------------------------------------------------------------
# Sign handling
# ---------------------------------------------------------------------------


def test_positive_uptrend_signal_returned_as_positive():
    """Uptrend (positive drift) propagates through unchanged."""
    h = _make_harness(window_s=10.0, mdw=_mdw(d_10s=+8.5))
    eff_q = _eff_q(mid_return_250ms_bps=-0.1)
    assert Bot._select_trend_drift_signal(h, eff_q) == +8.5


def test_zero_drift_returns_zero_not_none():
    """drift_10s=0.0 is a valid value (flat market), not warmup —
    returned as 0.0, not falling through to fallback chain."""
    h = _make_harness(window_s=10.0, mdw=_mdw(d_10s=0.0))
    eff_q = _eff_q(mid_return_250ms_bps=+5.0)
    assert Bot._select_trend_drift_signal(h, eff_q) == 0.0


# ---------------------------------------------------------------------------
# v1.5.154-260526-074029 reproducer: the snapshot moment
# ---------------------------------------------------------------------------


def test_reproducer_v1_5_154_260526_074029_snapshot_moment():
    """At the snapshot moment:
      * drift_500ms_bps = -5.26
      * drift_5s_bps   = -10.5
      * drift_10s_bps  = -10.5
      * drift_30s_bps  = None (warming up after regime transition)
      * mid_return_250ms_bps in eff_q ≈ -5.26 (~same as 500ms drift in
        that moment because the move was very recent)

    Pre-v1.5.155 the constructive trend skew saw -5.26 bps and
    produced 0.30 × -5.26 = -1.58 bps midpoint shift (matches the
    ``trend_drift_shift_bps`` in the snapshot's last_quote_breakdown).

    Post-v1.5.155 with window=10.0 and alpha=1.0 the same moment
    produces -10.5 bps drift → 1.0 × -10.5 = -10.5 bps midpoint
    shift — ~7× stronger lean into the downtrend.
    """
    h = _make_harness(
        window_s=10.0,
        mdw=_mdw(d_500ms=-5.26, d_5s=-10.5, d_10s=-10.5, d_30s=None),
    )
    eff_q = _eff_q(mid_return_250ms_bps=-5.26, mid_return_500ms_bps=-5.26)
    selected = Bot._select_trend_drift_signal(h, eff_q)
    assert selected == -10.5

    # With pre-v1.5.155 default (window=0.25) we'd get the weak signal.
    h_legacy = _make_harness(
        window_s=0.25,
        mdw=_mdw(d_500ms=-5.26, d_5s=-10.5, d_10s=-10.5, d_30s=None),
    )
    selected_legacy = Bot._select_trend_drift_signal(h_legacy, eff_q)
    assert selected_legacy == -5.26
    # The ratio matters: post-v1.5.155 sees ~2× the trend strength
    # in this snapshot moment, and at higher alpha (1.0 vs 0.30) the
    # midpoint shift grows by ~7× overall.
    assert abs(selected) > abs(selected_legacy)
