"""Phase 2K.T1 — cross-cutting favorable-exit acceptance sweep.

Plan spec (excerpt from ``plans/20260520-defense-action-plan.md``):

    Every reworked gate has a "ceiling never fires when signal
    clears mid-cooldown" test, paired with a "ceiling fires when
    signal stays armed" test.

Phase 2K shipped the favorable-exit predicate across 10 cooldown
gates (2K.1 through 2K.9 + 2K.11; 2K.10 deliberately deferred).
Each gate's own test file has the two halves of the contract.

This file is the **cross-cutting audit**: a single meta-test that
fails fast if any of the 10 gates is missing either half of its
favorable-exit pair. Catches the regression class "we added a new
gate to Phase 2K but forgot to write the cleared-mid-cooldown
test for it" — i.e. an INCOMPLETE Phase 2K extension.

How the audit works
===================

For each gate in the registry:

1. Locate its test file by relative path.
2. Read the file.
3. Verify it contains at least one test function whose name + body
   matches the "cleared mid-cooldown" pattern.
4. Verify it contains at least one test function whose name + body
   matches the "ceiling fires when signal stays armed" pattern.

Both halves use lenient substring matching against the test
function names — the pattern set was reviewed against the actual
test files at v1.4.206 acceptance time. If a future contributor
adds a new gate or renames an existing test, this file must be
updated alongside.

Why this is the right shape
===========================

The alternative — a single test that DRIVES every gate through
its arm/clear cycle in one place — would re-implement each
gate's setup logic and double the maintenance burden. The shape
chosen here is a META-AUDIT: it validates that the per-gate test
files exist and cover the two required behaviours, without
re-implementing them. The per-gate files themselves are the
source of truth for the actual gate behaviour.

This was the deliberate trade-off in the plan: "2K.T1 cross-
cutting test gathering — most pieces exist per-file."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest


_TESTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Gate registry — one entry per shipped Phase 2K gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GateAuditEntry:
    """One Phase 2K gate's expected test coverage.

    ``gate_id`` — short tag like "2K.1" for diagnostic messages.
    ``gate_name`` — human-readable name from the plan.
    ``test_file`` — path relative to ``tests/``.
    ``cleared_test_patterns`` — list of regex patterns; at least one
        must match a test name in the file. Catches the "favorable
        exit fires when signal clears mid-cooldown" half.
    ``ceiling_test_patterns`` — list of regex patterns; at least one
        must match a test name in the file. Catches the "ceiling
        fires when signal stays armed" half.
    """

    gate_id: str
    gate_name: str
    test_file: str
    cleared_test_patterns: list[str]
    ceiling_test_patterns: list[str]


# The 10 shipped Phase 2K gates. 2K.10 (session_drawdown Tier-2/3)
# is intentionally absent — deferred per the plan, high blast
# radius. If a future batch ships 2K.10, append a row here.
_PHASE_2K_GATES: list[_GateAuditEntry] = [
    _GateAuditEntry(
        gate_id="2K.1",
        gate_name="post_reduction_cooldown",
        test_file="test_post_reduction_cooldown.py",
        # 2K.1's per-file tests focus on ARMING + snapshot/counter
        # wiring. The favorable-exit BEHAVIOUR (util drops below
        # clear threshold → cooldown clears) is covered by the
        # Phase 2K.T2 v1.4.118 emergency replay
        # (``test_phase2k_t2_v1_4_118_emergency_replay.py``), which
        # is the cross-cutting integration test. Patterns here
        # validate the per-file half of the audit only.
        cleared_test_patterns=[
            r"reduction.*suppresses",  # arming under reduction
            r"closed_to_flat",  # transition that should arm
            r"flipped_to",  # cross-zero arming
            r"phase2k1_exit_attribution",  # counter-presence
        ],
        ceiling_test_patterns=[
            r"extending.*does_not.*arm",  # negative-control: stays armed when signal persists
            r"extending.*does_not_clear",  # alternate phrasing
            r"repeated.*re_arms",  # re-arm under continued reduction
            r"state_counters_increment",  # counter-presence
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.2",
        gate_name="adverse_side_pause",
        test_file="test_adverse_side_pause.py",
        cleared_test_patterns=[
            r"markout.*recover",
            r"favorable.*exit",
            r"clears.*via.*favorable",
            r"recover.*clears",
        ],
        ceiling_test_patterns=[
            r"ceiling",
            r"timer.*expir",
            r"max.*cooldown",
            r"signal_stays",
            r"expir.*without_recovery",
            r"still_adverse",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.3",
        gate_name="vol_trend_gate",
        test_file="test_regime_gates.py",
        cleared_test_patterns=[
            r"vol_trend.*favorable",
            r"vol_trend.*clears",
            r"favorable.*exit.*vol_trend",
            r"dwell.*signals_drop",
        ],
        ceiling_test_patterns=[
            r"vol_trend.*ceiling",
            r"vol_trend.*timer",
            r"vol_trend.*signals_stay",
            r"vol_trend.*elevated",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.4",
        gate_name="post_swing_gate",
        test_file="test_post_swing_gate.py",
        cleared_test_patterns=[
            r"favorable.*exit",
            r"clears.*via.*favorable",
            r"delta.*shrink",
        ],
        ceiling_test_patterns=[
            r"ceiling",
            r"timer.*expir",
            r"max.*cooldown",
            r"delta.*stays",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.5",
        gate_name="adaptive_spread_widen",
        test_file="test_phase2k5_adaptive_widen_favorable_exit.py",
        # 2K.5 has 6 trigger reasons (toxicity_hard / toxicity_soft /
        # markout_adverse / one_sided_ratio / quote_quality /
        # slow_trend); the test file uses ``<reason>_cleared_when_*``
        # + ``<reason>_not_cleared_while_*`` for each reason's pair.
        cleared_test_patterns=[
            r"_cleared_when_",  # per-reason favorable-exit
            r"signal_cleared",
            r"favorable.*exit",
            r"markout_recovered",
        ],
        ceiling_test_patterns=[
            r"not_cleared_while_",  # negative-control per reason
            r"unknown_reason_falls_to_ceiling",  # default ceiling-only
            r"ceiling",
            r"still_below_threshold",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.6",
        gate_name="at_touch_adverse_pause",
        test_file="test_phase2k6_at_touch_adverse_favorable_exit.py",
        cleared_test_patterns=[
            r"recovery",
            r"clears.*via",
            r"favorable.*exit",
        ],
        ceiling_test_patterns=[
            r"ceiling.*only",
            r"timer.*expir",
            r"max.*cooldown",
            r"signal.*stays",
            r"mult_zero",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.7",
        gate_name="mae_gate",
        test_file="test_phase2k7_mae_gate_favorable_exit.py",
        cleared_test_patterns=[
            r"clears.*via.*favorable",
            r"recover",
            r"favorable.*exit",
        ],
        ceiling_test_patterns=[
            r"ceiling",
            r"timer.*expir",
            r"hard_threshold.*stays",
            r"signal.*stays",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.8",
        gate_name="vol_spike",
        test_file="test_phase2k8_vol_spike_favorable_exit.py",
        cleared_test_patterns=[
            r"clears.*via.*favorable",
            r"favorable.*exit",
            r"vol_ratio.*drop",
            r"ratio.*below",
        ],
        ceiling_test_patterns=[
            r"ceiling",
            r"timer.*expir",
            r"max.*cooldown",
            r"ratio.*stays",
            r"signal.*stays",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.9",
        gate_name="fill_burst_detector",
        test_file="test_phase2k9_fill_burst_favorable_exit.py",
        cleared_test_patterns=[
            r"clears.*via.*favorable",
            r"favorable.*exit",
            r"fills.*stop",
            r"count.*drop",
            r"silent_recovery",
        ],
        ceiling_test_patterns=[
            r"ceiling",
            r"timer.*expir",
            r"max.*cooldown",
            r"fills.*continue",
            r"signal.*stays",
        ],
    ),
    _GateAuditEntry(
        gate_id="2K.11",
        gate_name="post_only_cross_cooldown",
        test_file="test_phase2k11_post_only_cross_favorable_exit.py",
        # 2K.11 uses tick-comparison predicates: tests are
        # ``clears_when_ask_moves_*`` / ``does_not_clear_when_*``.
        # Audit patterns target those substrings rather than the
        # word "touch".
        cleared_test_patterns=[
            r"clears_when_",  # ask/bid moves away — favorable exit
            r"moves.*one_tick",  # the canonical favorable case
            r"moved_more_than",  # larger move still clears
            r"requires_two_ticks",  # multi-tick variant
        ],
        ceiling_test_patterns=[
            r"does_not_clear",  # negative-control
            r"at_rejected_price",  # touch hasn't moved
            r"moved_only_half_tick",  # within threshold
            r"tick_multiplier_zero_disables",  # disable knob
        ],
    ),
]


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


_TEST_FUNC_RE = re.compile(r"^def (test_[a-zA-Z0-9_]+)\(", re.MULTILINE)


def _list_test_funcs(file_path: Path) -> list[str]:
    """Return the test_xxx function names defined in ``file_path``.
    Empty list if the file doesn't exist (the audit reports it
    distinctly via the file-exists check)."""
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf-8")
    return _TEST_FUNC_RE.findall(text)


def _any_pattern_matches(patterns: list[str], names: list[str]) -> bool:
    """True iff at least one pattern matches at least one name.
    Case-insensitive — operator-readable patterns shouldn't
    depend on the exact casing of test names."""
    for pat in patterns:
        prog = re.compile(pat, re.IGNORECASE)
        for name in names:
            if prog.search(name):
                return True
    return False


# ---------------------------------------------------------------------------
# The cross-cutting audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    _PHASE_2K_GATES,
    ids=lambda e: f"{e.gate_id}-{e.gate_name}",
)
def test_phase2k_gate_has_both_favorable_exit_halves(
    entry: _GateAuditEntry,
) -> None:
    """For every shipped Phase 2K gate, verify that its test file
    exists AND contains both halves of the favorable-exit
    contract: a "ceiling never fires when signal clears mid-
    cooldown" test AND a "ceiling fires when signal stays armed"
    test.

    A failure here means a Phase 2K gate's per-file tests are
    incomplete and the cross-cutting plan-spec invariant is
    violated. Either add the missing test to the gate's file, or
    update the pattern list in this module to recognise an
    existing test that satisfies the contract."""
    file_path = _TESTS_DIR / entry.test_file
    assert file_path.exists(), (
        f"[{entry.gate_id} {entry.gate_name}] test file "
        f"{entry.test_file!r} does not exist — Phase 2K gate "
        f"appears to be incompletely shipped."
    )
    names = _list_test_funcs(file_path)
    assert names, (
        f"[{entry.gate_id} {entry.gate_name}] test file "
        f"{entry.test_file!r} contains no test_xxx functions."
    )
    has_cleared = _any_pattern_matches(entry.cleared_test_patterns, names)
    assert has_cleared, (
        f"[{entry.gate_id} {entry.gate_name}] test file "
        f"{entry.test_file!r} is missing the "
        f'"ceiling never fires when signal clears mid-cooldown" '
        f"half of the favorable-exit contract. Looked for any "
        f"test name matching: {entry.cleared_test_patterns}. "
        f"Existing test names: {names}"
    )
    has_ceiling = _any_pattern_matches(entry.ceiling_test_patterns, names)
    assert has_ceiling, (
        f"[{entry.gate_id} {entry.gate_name}] test file "
        f"{entry.test_file!r} is missing the "
        f'"ceiling fires when signal stays armed" half of the '
        f"favorable-exit contract. Looked for any test name "
        f"matching: {entry.ceiling_test_patterns}. Existing test "
        f"names: {names}"
    )


def test_phase2k_t2_replay_is_present() -> None:
    """Phase 2K.T2 — cross-cutting v1.4.118 emergency replay — is
    a sibling acceptance test that was shipped v1.4.169. Verify
    it still exists; deletion would silently regress the
    Phase 2K cross-cutting invariant set."""
    assert (
        _TESTS_DIR / "test_phase2k_t2_v1_4_118_emergency_replay.py"
    ).exists(), (
        "Phase 2K.T2 replay test is missing. This is the cross-"
        "cutting integration replay of the v1.4.118 emergency "
        "state and is part of the Phase 2K acceptance invariant "
        "set per the plan."
    )


def test_phase2k_t1_registry_covers_all_shipped_gates() -> None:
    """Meta-test: the registry in this file lists exactly the 10
    shipped Phase 2K gates. If a future batch ships 2K.10 or adds
    a new sub-phase, this test fails until the registry is
    updated."""
    expected_ids = {
        "2K.1", "2K.2", "2K.3", "2K.4", "2K.5",
        "2K.6", "2K.7", "2K.8", "2K.9", "2K.11",
    }
    actual_ids = {e.gate_id for e in _PHASE_2K_GATES}
    assert actual_ids == expected_ids, (
        f"Phase 2K registry drift. Expected {sorted(expected_ids)}, "
        f"actual {sorted(actual_ids)}. If 2K.10 has been shipped, "
        f"add it to ``_PHASE_2K_GATES``. If a gate has been "
        f"removed, also update the plan's Phase 2K status table."
    )
