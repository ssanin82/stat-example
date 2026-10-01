/**
 * Regime-axis bucketing — TypeScript port of
 * ``scripts/regime_summary.py``.
 *
 * Single source of truth for the dashboard's Regimes tab. Every
 * threshold value here must match the Python file's constants
 * exactly — a drift means the dashboard's per-bucket aggregates
 * stop matching the snapshot's pre-aggregated JSON the operator
 * cross-checks against.
 *
 * **WHEN UPDATING A THRESHOLD**: update both this file AND
 * ``scripts/regime_summary.py``. The pure-fn tests in
 * ``tests/test_regime_summary.py`` cover the Python side; the
 * dashboard's spot-check against
 * ``regime_summary_fill.by_axis`` covers the TS side.
 *
 * **DESIGN CHOICES** (mirrored from the Python file):
 *
 * 1. Absolute thresholds for axes with natural breakpoints —
 *    basis, inventory, quote age, aggressiveness, mode, trend,
 *    spread. These have meaningful boundaries that shouldn't be
 *    rebased per session.
 *
 * 2. Per-session quintiles for unconstrained axes — vol, toxicity.
 *    Absolute thresholds would silently mis-bucket symbols whose
 *    natural range differs from the calibration symbol (TON's vol
 *    distribution is not BTC's). Quintiles auto-calibrate per
 *    session.
 *
 * 3. The ``sampleSizeLabel`` thresholds (30 / 75 / 200) are Codex's
 *    flat heuristics. Block-bootstrap CIs (the Python file's more-
 *    rigorous metric) are deferred to a later phase — this tab
 *    relies on the flat labels for now.
 */

import type { FillSinceRow, ExposureBarRow } from "./dashboard_state";

// ---------------------------------------------------------------------------
// Bucket constants — keep in sync with regime_summary.py
// ---------------------------------------------------------------------------

export const BASIS_BUCKETS = [
  "NEG_STRETCHED",
  "NEG",
  "FLAT",
  "POS",
  "POS_STRETCHED",
] as const;
export type BasisBucket = (typeof BASIS_BUCKETS)[number];

export const INVENTORY_BUCKETS = ["FLAT", "LOW", "MED", "HIGH"] as const;
export type InventoryBucket = (typeof INVENTORY_BUCKETS)[number];

export const QUOTE_AGE_BUCKETS = [
  "<250ms",
  "250-1000ms",
  "1-2s",
  "2-5s",
  "5s+",
] as const;
export type QuoteAgeBucket = (typeof QUOTE_AGE_BUCKETS)[number];

export const AGGRESSIVENESS_BUCKETS = [
  "AT_TOUCH",
  "BEHIND_1",
  "BEHIND_2PLUS",
  "INSIDE",
  "AGED_TIGHTENED",
  "UNKNOWN",
] as const;
export type AggressivenessBucket = (typeof AGGRESSIVENESS_BUCKETS)[number];

export const MODE_BUCKETS = [
  "BOTH",
  "BID_ONLY",
  "ASK_ONLY",
  "HOLD_ALL",
] as const;
export type ModeBucket = (typeof MODE_BUCKETS)[number];

export const TREND_BUCKETS = ["UP_DRIFT", "DOWN_DRIFT", "FLAT"] as const;
export type TrendBucket = (typeof TREND_BUCKETS)[number];

export const VOL_BUCKETS = ["Q1_LOW", "Q2", "Q3", "Q4", "Q5_HIGH"] as const;
export type VolBucket = (typeof VOL_BUCKETS)[number];

export const TOXICITY_BUCKETS = [
  "Q1_LOW",
  "Q2",
  "Q3",
  "Q4",
  "Q5_HIGH",
] as const;
export type ToxicityBucket = (typeof TOXICITY_BUCKETS)[number];

export const SPREAD_BUCKETS = ["1tick", "2tick", "3tick+"] as const;
export type SpreadBucket = (typeof SPREAD_BUCKETS)[number];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

// ---------------------------------------------------------------------------
// Absolute-threshold bucketers
// ---------------------------------------------------------------------------

/** Five-way split from ``basis_regime_sign`` + ``binance_basis_ewma``.
 *  EWMA is decimal (0.0005 = 5 bp); converted to bp internally.
 *  Stretched threshold: |ewma_bps| > 5 bp. Matches
 *  ``regime_summary.py::basis_bucket_from_sign_and_ewma``. */
export function basisBucket(
  sign: number | null | undefined,
  ewmaDecimal: number | null | undefined,
): BasisBucket | null {
  if (sign === null || sign === undefined) return null;
  const s = Number(sign);
  if (!Number.isFinite(s)) return null;
  if (s === 0) return "FLAT";
  if (!isFiniteNum(ewmaDecimal)) {
    return s < 0 ? "NEG" : "POS";
  }
  const ewmaBps = Math.abs(ewmaDecimal) * 10_000;
  const stretched = ewmaBps > 5.0;
  if (s < 0) return stretched ? "NEG_STRETCHED" : "NEG";
  return stretched ? "POS_STRETCHED" : "POS";
}

/** ``utilization`` is a signed fraction in [-1, +1] roughly. We use
 *  the magnitude to bucket so a +0.4 long and a -0.4 short land in
 *  the same MED bucket. Matches ``regime_summary.py``. */
export function inventoryBucket(
  utilization: number | null | undefined,
): InventoryBucket | null {
  if (!isFiniteNum(utilization)) return null;
  const u = Math.abs(utilization);
  if (u < 0.1) return "FLAT";
  if (u < 0.35) return "LOW";
  if (u < 0.65) return "MED";
  return "HIGH";
}

export function quoteAgeBucket(
  ageMs: number | null | undefined,
): QuoteAgeBucket | null {
  if (!isFiniteNum(ageMs)) return null;
  if (ageMs < 250) return "<250ms";
  if (ageMs < 1000) return "250-1000ms";
  if (ageMs < 2000) return "1-2s";
  if (ageMs < 5000) return "2-5s";
  return "5s+";
}

/** Combines ``quote_aggressiveness`` label + ``quote_distance_to_
 *  touch_ticks_at_placement`` to distinguish "1 tick behind" from
 *  "2+ ticks behind". */
export function aggressivenessBucket(
  aggLabel: string | null | undefined,
  distTicks: number | null | undefined,
): AggressivenessBucket | null {
  if (aggLabel === null || aggLabel === undefined) return null;
  const a = String(aggLabel).toLowerCase();
  if (a === "at_touch") return "AT_TOUCH";
  if (a === "inside") return "INSIDE";
  if (a === "aged_tightened") return "AGED_TIGHTENED";
  if (a === "behind_touch") {
    if (isFiniteNum(distTicks) && distTicks > 1.5) return "BEHIND_2PLUS";
    return "BEHIND_1";
  }
  return "UNKNOWN";
}

/** Prefer ``quote_eligibility`` (actual outcome) over the requested
 *  ``active_sides`` when both are present. HOLD_ALL wins everything.
 *  Normalizes OKX-style ``QUOTE_BUY_ONLY`` / ``QUOTE_SELL_ONLY`` to
 *  the internal BID/ASK naming (bug fix carried from the Python
 *  side 2026-05-13). */
export function modeBucket(
  activeSides: string | null | undefined,
  quoteEligibility: string | null | undefined,
): ModeBucket | null {
  const elig = String(quoteEligibility || "").toUpperCase();
  if (elig === "HOLD_ALL") return "HOLD_ALL";
  if (
    elig === "QUOTE_BID_ONLY" ||
    elig === "QUOTE_BUY_ONLY" ||
    elig === "BID_ONLY" ||
    elig === "BUY_ONLY"
  ) {
    return "BID_ONLY";
  }
  if (
    elig === "QUOTE_ASK_ONLY" ||
    elig === "QUOTE_SELL_ONLY" ||
    elig === "ASK_ONLY" ||
    elig === "SELL_ONLY"
  ) {
    return "ASK_ONLY";
  }
  if (elig === "QUOTE_BOTH" || elig === "BOTH") return "BOTH";
  // Fall back to ``active_sides`` if eligibility isn't set.
  const av = String(activeSides || "").toUpperCase();
  if (av === "BOTH" || av === "BID_ONLY" || av === "ASK_ONLY") {
    return av as ModeBucket;
  }
  if (av === "HOLD_ALL" || av === "NONE") return "HOLD_ALL";
  return null;
}

/** Forward-looking trend from per-fill mid return. Python uses
 *  ``mid_return_500ms_bps_at_fill`` (the coarsest fwd horizon we
 *  capture); same here. */
export function trendBucket(
  midReturn500msBps: number | null | undefined,
): TrendBucket | null {
  if (!isFiniteNum(midReturn500msBps)) return null;
  if (midReturn500msBps > 2.0) return "UP_DRIFT";
  if (midReturn500msBps < -2.0) return "DOWN_DRIFT";
  return "FLAT";
}

/** Number-of-ticks bucket. ``priceTickBps`` is the bps width of one
 *  tick at the bar's mid. When unavailable, falls back to absolute
 *  bps bands (5 / 10 / 20 bp) — reasonable defaults for crypto perps,
 *  same fallback path the Python file uses. */
export function spreadBucket(
  spreadBps: number | null | undefined,
  priceTickBps: number | null | undefined,
): SpreadBucket | null {
  if (!isFiniteNum(spreadBps)) return null;
  if (!isFiniteNum(priceTickBps) || priceTickBps <= 0) {
    if (spreadBps < 5) return "1tick";
    if (spreadBps < 10) return "2tick";
    return "3tick+";
  }
  const ticks = spreadBps / priceTickBps;
  if (ticks < 1.5) return "1tick";
  if (ticks < 2.5) return "2tick";
  return "3tick+";
}

// ---------------------------------------------------------------------------
// Per-session quintile bucketing for vol + toxicity
// ---------------------------------------------------------------------------

/** Compute 4 breakpoints that partition ``values`` into 5
 *  equal-count quintiles. Returns ``null`` when n < 10 (too small
 *  for meaningful quintiles — matches Python). Uses the same
 *  index-based partition rule as Python's
 *  ``finite[len(finite) * i // 5]``. */
export function quintileBreakpoints(values: number[]): number[] | null {
  const finite = values
    .filter((v): v is number => isFiniteNum(v))
    .sort((a, b) => a - b);
  if (finite.length < 10) return null;
  return [1, 2, 3, 4].map((i) =>
    finite[Math.floor((finite.length * i) / 5)],
  );
}

/** Map a value to one of Q1_LOW / Q2 / Q3 / Q4 / Q5_HIGH using
 *  pre-computed breakpoints. Returns ``null`` when breakpoints
 *  weren't computable (insufficient sample) or value isn't finite. */
export function assignQuintile(
  value: number | null | undefined,
  breaks: number[] | null,
): VolBucket | null {
  if (breaks === null || !isFiniteNum(value)) return null;
  for (let i = 0; i < breaks.length; i++) {
    if (value < breaks[i]) return VOL_BUCKETS[i];
  }
  return VOL_BUCKETS[4];
}

// ---------------------------------------------------------------------------
// Sample-size labeller (Codex flat tiers — block-bootstrap CIs
// deferred to later phase)
// ---------------------------------------------------------------------------

export const SAMPLE_SIZE_TIERS = [
  "anecdote",
  "hypothesis",
  "directional",
  "trustworthy",
] as const;
export type SampleSizeTier = (typeof SAMPLE_SIZE_TIERS)[number];

export function sampleSizeLabel(n: number): SampleSizeTier {
  if (n < 30) return "anecdote";
  if (n < 75) return "hypothesis";
  if (n < 200) return "directional";
  return "trustworthy";
}

// ---------------------------------------------------------------------------
// All axes the Regimes tab renders. Order matches the spec's
// operator-priority section ordering.
// ---------------------------------------------------------------------------

export const REGIME_AXES = [
  "side",
  "basis",
  "inventory",
  "mode",
  "quote_age",
  "aggressiveness",
  "vol",
  "toxicity",
  "trend",
  "spread",
] as const;
export type RegimeAxis = (typeof REGIME_AXES)[number];

/** The axes that are also published on exposure bars — used as the
 *  denominator for ``fills per hour of exposure`` and per-regime
 *  minutes-exposed. Bars don't carry every fill-only axis (e.g.,
 *  quote_age, aggressiveness, trend are fill-specific). */
export const EXPOSURE_AXES: ReadonlySet<RegimeAxis> = new Set([
  "side", // not on bars; handled as "all bars" denominator
  "basis",
  "inventory",
  "mode",
  "vol",
  "toxicity",
  "spread",
]);

// ---------------------------------------------------------------------------
// Per-row bucketing entry points (compute every axis for one row)
// ---------------------------------------------------------------------------

export interface RowBuckets {
  basis: BasisBucket | null;
  inventory: InventoryBucket | null;
  quote_age: QuoteAgeBucket | null;
  aggressiveness: AggressivenessBucket | null;
  mode: ModeBucket | null;
  trend: TrendBucket | null;
  spread: SpreadBucket | null;
  vol: VolBucket | null;
  toxicity: ToxicityBucket | null;
}

/** Compute every bucket for one fill row. Quintile breaks for vol /
 *  toxicity must be passed in (caller computes them once per session
 *  from the full fill set — see ``regime_aggregator.ts``). */
export function bucketsForFill(
  fill: FillSinceRow,
  vol_breaks: number[] | null,
  toxicity_breaks: number[] | null,
): RowBuckets {
  // ``spread_bps_at_fill`` + ``mid_return_500ms_bps_at_fill`` are
  // schema-additive columns not in the typed FillSinceRow surface —
  // read via the catch-all index signature.
  const f = fill as Record<string, unknown>;
  const spreadAtFill = isFiniteNum(f["spread_bps_at_fill"])
    ? (f["spread_bps_at_fill"] as number)
    : null;
  const midRet500ms = isFiniteNum(f["mid_return_500ms_bps_at_fill"])
    ? (f["mid_return_500ms_bps_at_fill"] as number)
    : null;
  return {
    basis: basisBucket(
      fill.basis_regime_sign,
      fill.binance_basis_ewma_at_decision,
    ),
    inventory: inventoryBucket(fill.inventory_utilization_before_fill),
    quote_age: quoteAgeBucket(fill.quote_age_at_fill_ms),
    aggressiveness: aggressivenessBucket(
      fill.quote_aggressiveness,
      fill.quote_distance_to_touch_ticks_at_placement,
    ),
    mode: modeBucket(
      fill.active_sides_at_decision,
      fill.quote_eligibility_state,
    ),
    trend: trendBucket(midRet500ms),
    spread: spreadBucket(spreadAtFill, null),
    vol: assignQuintile(fill.vol_estimate_at_decision, vol_breaks),
    toxicity: assignQuintile(
      fill.toxicity_score_at_decision,
      toxicity_breaks,
    ),
  };
}

/** Same as ``bucketsForFill`` but for one exposure-bar row. Bars
 *  carry instantaneous state (not at-fill), so only the subset of
 *  axes that exist on the bars table are populated. */
export interface BarBuckets {
  basis: BasisBucket | null;
  inventory: InventoryBucket | null;
  mode: ModeBucket | null;
  spread: SpreadBucket | null;
  vol: VolBucket | null;
  toxicity: ToxicityBucket | null;
}

export function bucketsForBar(
  bar: ExposureBarRow,
  vol_breaks: number[] | null,
  toxicity_breaks: number[] | null,
): BarBuckets {
  return {
    basis: basisBucket(bar.basis_regime_sign, bar.binance_basis_ewma),
    inventory: inventoryBucket(bar.inventory_utilization),
    mode: modeBucket(bar.active_sides, bar.quote_eligibility),
    spread: spreadBucket(bar.spread_bps, null),
    vol: assignQuintile(bar.vol_estimate, vol_breaks),
    toxicity: assignQuintile(bar.toxicity_score, toxicity_breaks),
  };
}
