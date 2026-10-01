"""v1.4.116 Phase 1E.3.f — per-quoting-feature firing-rate tracker.

Bot Stats → Detectors card section 5 ("Quoting features") needs to
show *how often* a per-side suppression feature is firing, not just
whether it's currently active. ON/OFF doesn't capture the regime —
``post_fill_cooldown_bid`` flipping ON 30×/min during a churn-heavy
session is a different operating regime than the same feature ON
once every 5 min during a calm session.

The tracker is a deque-of-timestamps per feature key. Each
``note_fire(name, ts_mono)`` appends a monotonic timestamp; entries
older than ``MAX_RETENTION_SECONDS`` (default 120 s — 2× the 60 s
window the dashboard shows) are pruned at append time so memory
stays bounded regardless of fire frequency.

``rate_in_window(name, window_seconds, now_mono)`` returns the
count of fires within the last ``window_seconds``. Caller picks the
window — typical use is 60 s for the Detectors card display.

Thread-safety: single-threaded ownership (bot's main quote loop is
the sole writer, called from ``_apply_regime_gates`` after the
observability flags are refreshed). The publisher reads under the
GIL — torn reads on the deque can produce a stale count, never
undefined behaviour.

Why a backend tracker rather than client-side diff:

* Client-side diff over the 5 s live_stats poll cadence loses
  transitions that happen between polls (e.g. a feature that fires,
  clears, and fires again all within 5 s shows up as one event).
* Client-side diff loses ALL history across dashboard reloads.
* The backend tracker is cheap (~6 features × 200 timestamps =
  trivial memory) and authoritative.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Optional


# Each deque holds at most this many seconds of history. The
# dashboard typically asks for 60 s windows; 120 s keeps a small
# safety margin so a slow publisher can still answer the 60 s
# question without recent fires being already evicted.
MAX_RETENTION_SECONDS: float = 120.0

# Hard cap on deque length per feature. Acts as a safety net if a
# feature fires faster than once per 600 ms (which would only
# happen under a pathological state-flapping bug). At 200 entries
# we hold ~2 minutes of normal firings.
_PER_FEATURE_MAXLEN: int = 200


class FeatureFiringRateTracker:
    """Records per-feature fire timestamps for the Detectors card's
    Quoting-features section.

    Features tracked today (Phase 1 closure scope):
    * ``post_fill_cooldown_bid`` / ``..._ask`` — per-side cooldown
      arms after a fill on that side.
    * ``at_touch_adverse_pause_bid`` / ``..._ask`` — per-side pause
      when recent at-touch fills cluster adverse on that side.

    Additional features (``hard_skew_*``, ``inventory_exec_bias_*``)
    are NOT tracked here yet — they're per-tick reason-string
    markers without explicit ON/OFF state. Adding them is a small
    follow-up once their state representation is cleaned up.
    """

    def __init__(self) -> None:
        # Map: feature name → deque of monotonic timestamps. Each
        # deque pruned at append time so its length is bounded by
        # MAX_RETENTION_SECONDS × fire rate.
        self._fires: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_PER_FEATURE_MAXLEN)
        )

    def note_fire(self, name: str, now_mono: float) -> None:
        """Record a single fire event for ``name`` at ``now_mono``.

        Idempotent within the same tick — if the feature is already
        active, do NOT call this every tick; call it only on the
        edge transition (False → True). Caller's job to detect the
        edge; this function appends unconditionally.

        Prunes entries older than ``MAX_RETENTION_SECONDS`` from the
        head of the deque before appending. Linear-time but the
        deques never grow past ~200 entries in practice.
        """
        if not name:
            return
        if name not in self._fires:
            self._fires[name]  # trigger defaultdict factory
        dq = self._fires[name]
        cutoff = float(now_mono) - MAX_RETENTION_SECONDS
        while dq and dq[0] < cutoff:
            dq.popleft()
        dq.append(float(now_mono))

    def rate_in_window(
        self, name: str, window_seconds: float, now_mono: float
    ) -> int:
        """Return the count of fires for ``name`` within the last
        ``window_seconds``. Window is half-open: ``(now − window,
        now]`` — a fire AT now_mono is included; a fire exactly at
        ``now − window`` is excluded.

        Does NOT prune the deque — the publisher path stays cheap
        (no mutation under read). Pruning only happens at
        ``note_fire`` time so the deque is bounded by retention.
        """
        if name not in self._fires:
            return 0
        dq = self._fires[name]
        if not dq:
            return 0
        cutoff = float(now_mono) - float(window_seconds)
        # Linear walk from the tail. Acceptable because each deque
        # is capped at _PER_FEATURE_MAXLEN (~200) — sub-microsecond
        # iteration even on a deep deque.
        count = 0
        # ``deque`` doesn't index efficiently from the right; iterate
        # forward and short-circuit once we cross the cutoff.
        for ts in dq:
            if ts > cutoff:
                count += 1
        return count

    def all_rates(
        self, window_seconds: float, now_mono: float
    ) -> dict[str, int]:
        """Convenience: return a dict ``{name: count_in_window}`` for
        every tracked feature. Used by the live_stats publisher to
        emit the section-5 payload."""
        return {
            name: self.rate_in_window(name, window_seconds, now_mono)
            for name in self._fires
        }

    def feature_names(self) -> list[str]:
        """List of feature keys ever seen. Stable order (insertion)."""
        return list(self._fires.keys())


def detect_edge_transitions(
    *,
    tracker: FeatureFiringRateTracker,
    now_mono: float,
    prior_flags: dict[str, Optional[bool]],
    current_flags: dict[str, Optional[bool]],
) -> dict[str, Optional[bool]]:
    """Helper for the bot's tick path: given the prior + current
    ``observability_gate_flags`` snapshots, record a fire for every
    feature that transitioned False → True. Returns the new prior
    snapshot for the caller to stash.

    ``None`` values (uninitialised / feature disabled) are treated
    as False for edge detection — a feature transitioning from
    None to True records one fire.

    Caller is responsible for stashing the returned snapshot so the
    next tick can detect the next edge. Keep state external to the
    tracker so the tracker stays focused on the fire-timestamp
    bookkeeping.
    """
    for key, cur_val in current_flags.items():
        cur_active = bool(cur_val) if cur_val is not None else False
        prior_val = prior_flags.get(key)
        prior_active = bool(prior_val) if prior_val is not None else False
        if cur_active and not prior_active:
            tracker.note_fire(key, now_mono)
    # Return a shallow copy so the caller's stash is decoupled from
    # the live current_flags dict.
    return dict(current_flags)
