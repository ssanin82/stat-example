/**
 * Regime aggregation — TypeScript port of
 * ``regime_summary.py::_aggregate_fill_slice`` + ``regime_summary_
 * fill`` + ``regime_summary_exposure``, plus the per-row bucket
 * caching that lets us walk fills + bars ONCE and emit every axis's
 * aggregates.
 *
 * Pure function: takes the dashboard's ``fills_since`` +
 * ``exposure_since`` rows, returns a nested record:
 *
 *   axis → side → bucket → RegimeRow
 *
 * One ``RegimeRow`` carries the 13 columns the Regimes-tab table
 * renders (n, confidence, notional, minutes_exposed, fills/h,
 * mean MO 5s, median MO 5s, adverse%, rebate$, close PnL$,
 * net edge bp).
 *
 * Independent of UI — unit-testable by passing synthetic fill / bar
 * arrays. Drift mitigation against ``regime_summary.py``: spot-check
 * the dashboard's output against a snapshot's pre-aggregated
 * ``regime_summary_fill.by_axis`` and confirm cell-for-cell parity.
 */

import {
  bucketsForBar,
  bucketsForFill,
  quintileBreakpoints,
  sampleSizeLabel,
  type RegimeAxis,
  type RowBuckets,
  type BarBuckets,
  type SampleSizeTier,
} from "./regime_buckets";
import type {
  ExposureBarRow,
  FillSinceRow,
} from "./dashboard_state";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RegimeRow {
  /** Number of fills in this (axis, side, bucket) cell. */
  n: number;
  /** Codex flat sample-size tier — drives row contrast + colour
   *  suppression on n<15. */
  confidence: SampleSizeTier;
  /** Σ notional_usd over the cell. */
  notional_usd: number;
  /** Minutes the venue spent in this bucket (per-bar denominator).
   *  Only populated for axes also published on exposure bars
   *  (``EXPOSURE_AXES``). ``null`` otherwise. */
  minutes_exposed: number | null;
  /** Fills per hour-of-exposure, ``n / (minutes / 60)``. ``null``
   *  when ``minutes_exposed`` is null or zero. */
  fills_per_hour: number | null;
  /** Mean markout at 1s / 5s windows over the cell's fills. */
  mean_markout_1s_bps: number | null;
  mean_markout_5s_bps: number | null;
  /** v1.4.101 — mean markout at extended diagnostic horizons.
   *  Populated when ``FillSinceRow`` carries the matching field
   *  (post-v1.4.98 fills). Used by the Regime-tab page-level horizon
   *  selector so the operator can pivot the headline MO column to a
   *  longer holding period. */
  mean_markout_15s_bps: number | null;
  mean_markout_30s_bps: number | null;
  mean_markout_60s_bps: number | null;
  mean_markout_120s_bps: number | null;
  /** Median markout at 5s — more robust to one bad fill than mean. */
  median_markout_5s_bps: number | null;
  /** v1.4.101 — medians at extended horizons (same population /
   *  resolution semantics as the means above). */
  median_markout_15s_bps: number | null;
  median_markout_30s_bps: number | null;
  median_markout_60s_bps: number | null;
  median_markout_120s_bps: number | null;
  /** Fraction of fills with markout_5s_bps < 0, as a percentage. */
  adverse_fill_pct: number | null;
  /** Σ rebate $. Convention: the ``fee`` column is negative when
   *  the bot received a rebate, so ``rebate_usd = -Σ fee``. */
  rebate_usd: number;
  /** Σ closed_pnl_usd over the cell. */
  closed_pnl_usd: number;
  /** Net edge in bp:
   *    net$ = rebate_usd + (notional * mean_markout_5s_bps / 1e4)
   *           + closed_pnl_usd
   *    net edge bp = net$ / notional * 1e4
   *  Matches ``regime_summary_net_edge.json``. */
  net_edge_bp: number | null;
}

/** Aggregated payload: axis → side → bucket → RegimeRow.
 *  The 'side' axis is special: it has only buckets, not (side × bucket)
 *  — keyed via the sentinel "ALL" side so the call site can render
 *  uniformly. */
export type RegimeAggregates = Record<
  RegimeAxis,
  Record<string, Record<string, RegimeRow>>
>;

export interface RegimeAggregationResult {
  axes: readonly RegimeAxis[];
  by_axis: RegimeAggregates;
  /** Session-wide totals for the header strip. */
  totals: {
    n_fills: number;
    notional_usd: number;
    rebate_usd: number;
    closed_pnl_usd: number;
    n_bars: number;
    minutes_exposed: number;
    bar_cadence_seconds: number;
    /** Cell-count rollups for the header. ``slices_*`` = number of
     *  filled cells (n>=1) at each confidence tier across every
     *  axis × side × bucket. Helps the operator see at a glance
     *  how much of the section is signal vs noise. */
    slices_anecdote: number;
    slices_hypothesis: number;
    slices_directional: number;
    slices_trustworthy: number;
  };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Same cadence as the bot's ``exposure_bar_emitter``. Used to
 *  convert ``n_bars`` → minutes. Matches Python's default. */
const BAR_CADENCE_SECONDS = 5.0;

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

function isFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function meanOrNull(values: number[]): number | null {
  if (values.length === 0) return null;
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

function medianOrNull(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function round(v: number | null, digits: number): number | null {
  if (v === null) return null;
  const mult = 10 ** digits;
  return Math.round(v * mult) / mult;
}

// ---------------------------------------------------------------------------
// Per-cell aggregation
// ---------------------------------------------------------------------------

function aggregateFillSlice(
  fills: FillSinceRow[],
  minutes_exposed: number | null,
): RegimeRow {
  const n = fills.length;
  if (n === 0) {
    return {
      n: 0,
      confidence: "anecdote",
      notional_usd: 0,
      minutes_exposed,
      fills_per_hour:
        minutes_exposed !== null && minutes_exposed > 0 ? 0 : null,
      mean_markout_1s_bps: null,
      mean_markout_5s_bps: null,
      mean_markout_15s_bps: null,
      mean_markout_30s_bps: null,
      mean_markout_60s_bps: null,
      mean_markout_120s_bps: null,
      median_markout_5s_bps: null,
      median_markout_15s_bps: null,
      median_markout_30s_bps: null,
      median_markout_60s_bps: null,
      median_markout_120s_bps: null,
      adverse_fill_pct: null,
      rebate_usd: 0,
      closed_pnl_usd: 0,
      net_edge_bp: null,
    };
  }
  const markout1s: number[] = [];
  const markout5s: number[] = [];
  // v1.4.101 — extended horizons. Read defensively via the
  // ``[key: string]: unknown`` indexer on FillSinceRow because the
  // backend `/fills/since` endpoint serves columns the FillSinceRow
  // interface didn't originally name.
  const markout15s: number[] = [];
  const markout30s: number[] = [];
  const markout60s: number[] = [];
  const markout120s: number[] = [];
  const notionals: number[] = [];
  const fees: number[] = [];
  const closedPnls: number[] = [];
  for (const f of fills) {
    if (isFiniteNum(f.markout_1s_bps)) markout1s.push(f.markout_1s_bps);
    if (isFiniteNum(f.markout_5s_bps)) markout5s.push(f.markout_5s_bps);
    const m15 = (f as unknown as Record<string, unknown>).markout_15s_bps;
    const m30 = (f as unknown as Record<string, unknown>).markout_30s_bps;
    const m60 = (f as unknown as Record<string, unknown>).markout_60s_bps;
    const m120 = (f as unknown as Record<string, unknown>).markout_120s_bps;
    if (isFiniteNum(m15)) markout15s.push(m15);
    if (isFiniteNum(m30)) markout30s.push(m30);
    if (isFiniteNum(m60)) markout60s.push(m60);
    if (isFiniteNum(m120)) markout120s.push(m120);
    if (isFiniteNum(f.notional)) notionals.push(f.notional);
    if (isFiniteNum(f.fee)) fees.push(f.fee);
    if (isFiniteNum(f.closed_pnl)) closedPnls.push(f.closed_pnl);
  }
  const notional_usd = notionals.reduce((s, x) => s + x, 0);
  const mean_mo_5s = meanOrNull(markout5s);
  const rebate_usd = -fees.reduce((s, x) => s + x, 0);
  const closed_pnl_usd = closedPnls.reduce((s, x) => s + x, 0);
  const adverse = markout5s.filter((m) => m < 0).length;
  // Net edge in $: rebate + markout-$-impact + closed_pnl. Mirrors
  // ``regime_summary_net_edge.json``'s definition.
  let net_edge_bp: number | null = null;
  if (mean_mo_5s !== null && notional_usd > 0) {
    const markoutDollars = (notional_usd * mean_mo_5s) / 10_000;
    const netDollars = rebate_usd + markoutDollars + closed_pnl_usd;
    net_edge_bp = (netDollars / notional_usd) * 10_000;
  }
  const fills_per_hour =
    minutes_exposed !== null && minutes_exposed > 0
      ? n / (minutes_exposed / 60)
      : null;
  return {
    n,
    confidence: sampleSizeLabel(n),
    notional_usd: round(notional_usd, 4) ?? 0,
    minutes_exposed: round(minutes_exposed, 2),
    fills_per_hour: round(fills_per_hour, 2),
    mean_markout_1s_bps: round(meanOrNull(markout1s), 4),
    mean_markout_5s_bps: round(mean_mo_5s, 4),
    mean_markout_15s_bps: round(meanOrNull(markout15s), 4),
    mean_markout_30s_bps: round(meanOrNull(markout30s), 4),
    mean_markout_60s_bps: round(meanOrNull(markout60s), 4),
    mean_markout_120s_bps: round(meanOrNull(markout120s), 4),
    median_markout_5s_bps: round(medianOrNull(markout5s), 4),
    median_markout_15s_bps: round(medianOrNull(markout15s), 4),
    median_markout_30s_bps: round(medianOrNull(markout30s), 4),
    median_markout_60s_bps: round(medianOrNull(markout60s), 4),
    median_markout_120s_bps: round(medianOrNull(markout120s), 4),
    adverse_fill_pct:
      markout5s.length > 0
        ? round((adverse / markout5s.length) * 100, 2)
        : null,
    rebate_usd: round(rebate_usd, 6) ?? 0,
    closed_pnl_usd: round(closed_pnl_usd, 6) ?? 0,
    net_edge_bp: round(net_edge_bp, 4),
  };
}

// ---------------------------------------------------------------------------
// Top-level aggregator
// ---------------------------------------------------------------------------

/** Compute per-(axis × side × bucket) aggregates for the Regimes
 *  tab. Single pass over fills + bars; O(N × axes).
 *
 *  - vol / toxicity quintile breakpoints are derived from the FILL
 *    distribution (matches the snapshot pipeline's behaviour — see
 *    ``regime_summary.py``'s ``generate_summaries`` which passes
 *    the fill-side distribution to ``assign_buckets_to_*``).
 *  - The 'side' axis aggregates ALL fills (no side filter) keyed
 *    under the sentinel "ALL".
 *  - Bars carry minutes_exposed denominators for the per-axis
 *    "fills per hour" + "minutes" columns. Axes not on bars (e.g.,
 *    quote_age, aggressiveness, trend) get null minutes; their
 *    fills_per_hour column will render n/a.
 */
export function aggregateRegimes(
  fills: FillSinceRow[],
  bars: ExposureBarRow[],
): RegimeAggregationResult {
  // Compute quintile breakpoints from the fill-side vol / toxicity
  // distributions. Same rule as ``regime_summary.py``.
  const volValues: number[] = [];
  const toxValues: number[] = [];
  for (const f of fills) {
    if (isFiniteNum(f.vol_estimate_at_decision)) {
      volValues.push(f.vol_estimate_at_decision);
    }
    if (isFiniteNum(f.toxicity_score_at_decision)) {
      toxValues.push(f.toxicity_score_at_decision);
    }
  }
  const vol_breaks = quintileBreakpoints(volValues);
  const toxicity_breaks = quintileBreakpoints(toxValues);

  // Pre-bucket every fill once (memo on row).
  type FillWithBuckets = { fill: FillSinceRow; buckets: RowBuckets };
  const fb: FillWithBuckets[] = fills.map((f) => ({
    fill: f,
    buckets: bucketsForFill(f, vol_breaks, toxicity_breaks),
  }));
  const bb: { bar: ExposureBarRow; buckets: BarBuckets }[] = bars.map(
    (b) => ({
      bar: b,
      buckets: bucketsForBar(b, vol_breaks, toxicity_breaks),
    }),
  );

  // Per-bucket bar counts → minutes_exposed denominator per axis ×
  // bucket. Axes not on bars produce empty maps; cells under those
  // axes fall through to ``minutes_exposed: null``.
  function barCountsForAxis(axis: RegimeAxis): Map<string, number> {
    const counts = new Map<string, number>();
    if (axis === "side") {
      // Bars don't carry a 'side' — total bars apply to every side.
      // Caller handles via the sentinel: "ALL" → total bars; per-
      // side breakdowns get the same denominator.
      counts.set("ALL", bb.length);
      return counts;
    }
    for (const { buckets } of bb) {
      const key = (buckets as unknown as Record<string, string | null>)[axis];
      if (key === null || key === undefined) continue;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }

  // For each axis: side → bucket → fill slice.
  const by_axis: Record<string, Record<string, Record<string, RegimeRow>>> = {};
  for (const axis of [
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
  ] as const) {
    const grouped: Record<string, Record<string, FillSinceRow[]>> = {};
    for (const { fill, buckets } of fb) {
      let bucket: string | null;
      let side: string;
      if (axis === "side") {
        bucket = String(fill.side || "").toUpperCase();
        side = "ALL";
        if (bucket !== "BUY" && bucket !== "SELL") continue;
      } else {
        const b = (buckets as unknown as Record<string, string | null>)[axis];
        if (b === null || b === undefined) continue;
        bucket = b;
        side = String(fill.side || "").toUpperCase();
        if (side !== "BUY" && side !== "SELL") continue;
      }
      if (!grouped[side]) grouped[side] = {};
      if (!grouped[side][bucket]) grouped[side][bucket] = [];
      grouped[side][bucket].push(fill);
    }
    const barCounts = barCountsForAxis(axis);
    const axisOut: Record<string, Record<string, RegimeRow>> = {};
    for (const [side, bucketMap] of Object.entries(grouped)) {
      axisOut[side] = {};
      for (const [bucket, slice] of Object.entries(bucketMap)) {
        const count = barCounts.get(bucket) ?? null;
        const minutes_exposed =
          count !== null ? (count * BAR_CADENCE_SECONDS) / 60 : null;
        axisOut[side][bucket] = aggregateFillSlice(slice, minutes_exposed);
      }
    }
    by_axis[axis] = axisOut;
  }

  // Session totals + slice-tier histograms for the header.
  let total_notional = 0;
  let total_rebate = 0;
  let total_closed_pnl = 0;
  for (const f of fills) {
    if (isFiniteNum(f.notional)) total_notional += f.notional;
    if (isFiniteNum(f.fee)) total_rebate -= f.fee;
    if (isFiniteNum(f.closed_pnl)) total_closed_pnl += f.closed_pnl;
  }
  let slices_anecdote = 0;
  let slices_hypothesis = 0;
  let slices_directional = 0;
  let slices_trustworthy = 0;
  for (const sides of Object.values(by_axis)) {
    for (const buckets of Object.values(sides)) {
      for (const row of Object.values(buckets)) {
        if (row.n === 0) continue;
        switch (row.confidence) {
          case "anecdote":
            slices_anecdote++;
            break;
          case "hypothesis":
            slices_hypothesis++;
            break;
          case "directional":
            slices_directional++;
            break;
          case "trustworthy":
            slices_trustworthy++;
            break;
        }
      }
    }
  }

  return {
    axes: [
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
    ],
    by_axis: by_axis as RegimeAggregates,
    totals: {
      n_fills: fills.length,
      notional_usd: round(total_notional, 4) ?? 0,
      rebate_usd: round(total_rebate, 6) ?? 0,
      closed_pnl_usd: round(total_closed_pnl, 6) ?? 0,
      n_bars: bars.length,
      minutes_exposed: round((bars.length * BAR_CADENCE_SECONDS) / 60, 2) ?? 0,
      bar_cadence_seconds: BAR_CADENCE_SECONDS,
      slices_anecdote,
      slices_hypothesis,
      slices_directional,
      slices_trustworthy,
    },
  };
}
