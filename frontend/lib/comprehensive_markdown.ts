/* Shared comprehensive-markdown builder + helpers.
 *
 * Moved from `frontend/app/page.tsx` on 2026-05-20 so the
 * function can be invoked from BOTH:
 *   1. The dashboard's "Download stats" button (client-side
 *      use, via the existing import in page.tsx).
 *   2. A server-side Next.js API route that returns the
 *      markdown over HTTP so the snapshot script can save it
 *      alongside `stats/*.json`.
 *
 * ONE source of truth. Edits land here and BOTH consumers
 * pick them up automatically.
 *
 * v1.5.135 (Codex bug #9) — ``@ts-nocheck`` REMOVED. The directive
 * was masking exactly the kind of field-name drift that produced
 * Codex bug #3 (the route reading the wrong account-risk field on
 * snapshot markdown export). The type aliases below stay ``any``
 * for the dashboard payload shapes that originate in ``page.tsx``
 * inline interfaces — a future refactor can move those into a
 * shared module — but the FILE as a whole is now checked, so the
 * exported function signatures + their internal logic are
 * statically protected.
 */

import {
  aggregateRegimes,
  type RegimeRow,
} from "./regime_aggregator";
import {
  basisBucket,
  inventoryBucket,
  modeBucket,
  sampleSizeLabel,
  type SampleSizeTier,
} from "./regime_buckets";
import type {
  DashboardStateResult,
  ExposureBarRow,
  FillSinceRow,
  OrdersLifecycleRow,
} from "./dashboard_state";
import type { ConfigResult } from "./types";

// Type aliases for shapes that live in page.tsx as inline
// interfaces. Using `any` is pragmatic: the function builds
// a markdown STRING, no static-type guarantees are needed.
// If/when these get factored into a shared types module,
// these aliases can be replaced with real imports.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type AccountState = any;
export type LiveStatsPayload = any;
export type BotStatusInfo = any;
export type VolumeResponse = any;
export type VenueInfo = any;
export type AccountRiskThresholds = any;
export type HistoryResponse = any;
export type EquityHistoryPayload = any;
export type Order = any;
export type Fill = any;

/** Convert ``basis.ewma`` (published by the bot as a raw price
 *  difference: ``okx_mid − binance_mid``, smoothed) into basis points
 *  relative to the OKX mid. Returns null when either input is missing
 *  or the mid is non-positive.
 *
 *  Why this helper exists: the dashboard previously had FOUR different
 *  interpretations of ``basis.ewma`` (treated as bp, treated as
 *  fraction, treated as raw price, …) in different places — causing
 *  the same value to read as "−0.00 bp / ALIGNED" on the Cross-venue
 *  card AND "NEG_STRETCHED" on the Observability — Regime card. All
 *  call sites now route through this function.
 *
 *  Note: divides by the OKX mid; the difference vs dividing by Binance
 *  mid is ≪ 1 % at any realistic cross-venue spread, irrelevant for
 *  display + bucketing thresholds. */
export function basisEwmaToBps(
  ewma: number | null | undefined,
  okxMid: number | null | undefined,
): number | null {
  if (ewma === null || ewma === undefined) return null;
  if (okxMid === null || okxMid === undefined) return null;
  if (!Number.isFinite(ewma) || !Number.isFinite(okxMid) || okxMid <= 0) {
    return null;
  }
  return (ewma / okxMid) * 10_000;
}


export function exIsFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}


/** Per-fill predicate for "Fills wrong-side of eligibility" — bot's
 *  most recent decision did NOT cover this fill's side. See spec
 *  Block C card #5. */
export function fillWrongSideOfEligibility(f: FillSinceRow): boolean {
  const elig = String(f.quote_eligibility_state || "").toUpperCase();
  if (elig === "HOLD_ALL") return true;
  const side = String(f.side || "").toUpperCase();
  if (side === "BUY") {
    return (
      elig === "QUOTE_ASK_ONLY" ||
      elig === "ASK_ONLY" ||
      elig === "QUOTE_SELL_ONLY" ||
      elig === "SELL_ONLY"
    );
  }
  if (side === "SELL") {
    return (
      elig === "QUOTE_BID_ONLY" ||
      elig === "BID_ONLY" ||
      elig === "QUOTE_BUY_ONLY" ||
      elig === "BUY_ONLY"
    );
  }
  return false;
}


/** Aggregate count + mean markout + net$ for a fill subset. Net$
 *  uses the same formula as ``regime_aggregator.netEdge``:
 *  rebate$ + markout-$-impact + closed-PnL. */
export function aggregateLeakage(fills: FillSinceRow[]): {
  count: number;
  meanMarkoutBps: number | null;
  netUsd: number | null;
} {
  if (fills.length === 0) {
    return { count: 0, meanMarkoutBps: null, netUsd: null };
  }
  let sumMarkout = 0;
  let nMarkout = 0;
  let notional = 0;
  let rebate = 0;
  let closed = 0;
  for (const f of fills) {
    if (exIsFiniteNum(f.markout_5s_bps)) {
      sumMarkout += f.markout_5s_bps;
      nMarkout++;
    }
    if (exIsFiniteNum(f.notional)) notional += f.notional;
    if (exIsFiniteNum(f.fee)) rebate -= f.fee;
    if (exIsFiniteNum(f.closed_pnl)) closed += f.closed_pnl;
  }
  const meanMarkout = nMarkout > 0 ? sumMarkout / nMarkout : null;
  const markoutDollars =
    meanMarkout !== null ? (notional * meanMarkout) / 10_000 : 0;
  const netUsd =
    meanMarkout !== null ? rebate + markoutDollars + closed : null;
  return { count: fills.length, meanMarkoutBps: meanMarkout, netUsd };
}


export interface GateRow {
  name: string;
  /** Short tooltip describing what fires this gate. */
  tooltip: string;
  /** True when the gate is active on the most recent bar (or as
   *  reported by live_stats for live-stats-only gates). */
  active_now: boolean;
  /** Seconds where the gate was active (≈ N_bars × 5 s). */
  time_active_s: number;
  time_active_pct: number;
  /** Bar-edge 0→1 transitions over the session. */
  fire_count: number;
  /** Number of fills landed while the gate was active at decision. */
  fills_during: number;
  mean_markout_5s_bps: number | null;
  median_markout_5s_bps: number | null;
  /** ``rebate$ + markout-$ + closed-PnL`` over filtered fills, ``null``
   *  when no filtered fill carries a finite markout. */
  net_dollars: number | null;
  confidence: SampleSizeTier;
}


export const GATE_BAR_CADENCE_S = 5.0;


export type BarFlag = (b: ExposureBarRow) => boolean;


export type FillFlag = (f: FillSinceRow) => boolean;


export interface GateDef {
  name: string;
  tooltip: string;
  barFlag: BarFlag;
  /** ``null`` when no per-fill counterpart exists (e.g. recovery_cooldown
   *  doesn't stamp on fills). When null, ``fills_during`` is computed
   *  from "bar time window where active" instead of an at-decision flag. */
  fillFlag: FillFlag | null;
}


export const GATE_DEFS: GateDef[] = [
  {
    name: "adaptive_widen",
    tooltip:
      "Widens spread when adverse markout streak exceeds threshold. Reverts when streak clears.",
    barFlag: (b) => b.adaptive_widen_active === 1,
    fillFlag: (f) => f.adaptive_widen_active_at_decision === 1,
  },
  {
    name: "post_fill_cooldown_bid",
    tooltip:
      "Pause bid quoting briefly after a buy fill to avoid stacking through a sustained adverse move.",
    barFlag: (b) => b.post_fill_cooldown_active_bid === 1,
    fillFlag: (f) => f.post_fill_cooldown_active_bid_at_decision === 1,
  },
  {
    name: "post_fill_cooldown_ask",
    tooltip:
      "Pause ask quoting briefly after a sell fill — mirror of post_fill_cooldown_bid.",
    barFlag: (b) => b.post_fill_cooldown_active_ask === 1,
    fillFlag: (f) => f.post_fill_cooldown_active_ask_at_decision === 1,
  },
  {
    name: "at_touch_adverse_pause_bid",
    tooltip:
      "Pause bid quoting when at-touch fills are landing adversely — prevents repeat pickoffs.",
    barFlag: (b) => b.at_touch_adverse_pause_bid === 1,
    fillFlag: (f) => f.at_touch_adverse_pause_bid_at_decision === 1,
  },
  {
    name: "at_touch_adverse_pause_ask",
    tooltip:
      "Pause ask quoting when at-touch fills are landing adversely — mirror of bid variant.",
    barFlag: (b) => b.at_touch_adverse_pause_ask === 1,
    fillFlag: (f) => f.at_touch_adverse_pause_ask_at_decision === 1,
  },
  {
    name: "hold_all",
    tooltip:
      "All quoting paused — typically driven by safety signals (data staleness, kill, drawdown tier).",
    barFlag: (b) =>
      b.hold_all_active === 1 ||
      String(b.quote_eligibility || "").toUpperCase() === "HOLD_ALL",
    fillFlag: (f) =>
      String(f.quote_eligibility_state || "").toUpperCase() === "HOLD_ALL",
  },
  {
    name: "recovery_cooldown",
    tooltip:
      "Sub-second post-fill recovery pause. Bar-only flag (no per-fill stamp — by design, fills shouldn't land in this state).",
    barFlag: (b) => b.recovery_cooldown_active === 1,
    fillFlag: null,
  },
  {
    name: "vol_trend",
    tooltip:
      "Cooldown after a volatility-trend trigger fires. Suppresses re-entry until the trend signal abates.",
    barFlag: (b) => b.vol_trend_active === 1,
    fillFlag: (f) => f.vol_trend_active_at_decision === 1,
  },
  {
    name: "post_swing",
    tooltip:
      "Cooldown after a large recent price swing — protects against post-swing mean-reversion picks.",
    barFlag: (b) => b.post_swing_active === 1,
    fillFlag: (f) => f.post_swing_active_at_decision === 1,
  },
  {
    name: "session_drawdown",
    tooltip:
      "Active when session drawdown is in a non-CLEAR tier (WARN/CAP1/CAP2). Tier-2 backend landed v1.3.32.",
    barFlag: (b) =>
      typeof b.session_drawdown_tier === "string" &&
      b.session_drawdown_tier !== "" &&
      b.session_drawdown_tier !== "CLEAR",
    fillFlag: (f) =>
      typeof f.session_drawdown_tier_at_decision === "string" &&
      f.session_drawdown_tier_at_decision !== "" &&
      f.session_drawdown_tier_at_decision !== "CLEAR",
  },
  // Derived gates — proxies from bar fields, no per-fill stamp.
  // Approximate; sub-bar transitions invisible.
  {
    name: "basis_IC (derived)",
    tooltip:
      "Derived from bar.basis_regime_sign === 0 — proxy for the bot's |IC| < 0.05 gate. Approximate; the real gate also requires the basis_regime classifier to be warmed_up.",
    barFlag: (b) => b.basis_regime_sign === 0,
    fillFlag: null,
  },
  {
    name: "microprice (derived)",
    tooltip:
      "Derived from |bar.imbalance_top| ≥ 0.5 — the bot's microprice-imbalance defensive gate. Direct read from the bar field, but sub-bar transitions are invisible.",
    barFlag: (b) =>
      typeof b.imbalance_top === "number" &&
      Number.isFinite(b.imbalance_top) &&
      Math.abs(b.imbalance_top) >= 0.5,
    fillFlag: null,
  },
];


export function gateIsFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}


export function computeGateRow(
  def: GateDef,
  bars: ExposureBarRow[],
  fills: FillSinceRow[],
): GateRow {
  const totalBars = bars.length;
  // Time-active + fire-count via single pass over sorted bars.
  // bars from publisher arrive sorted by ts_bar ASC but defensively
  // sort here so the result is correct on any input.
  const sortedBars = [...bars].sort((a, b) =>
    String(a.ts_bar).localeCompare(String(b.ts_bar)),
  );
  let activeBars = 0;
  let fireCount = 0;
  let lastFlag = false;
  for (const bar of sortedBars) {
    const flag = def.barFlag(bar);
    if (flag) activeBars++;
    if (flag && !lastFlag) fireCount++;
    lastFlag = flag;
  }
  // "Active now" — last bar's flag value (or false if no bars).
  const activeNow =
    sortedBars.length > 0 ? def.barFlag(sortedBars[sortedBars.length - 1]) : false;

  // Filter fills by the gate's per-decision flag. For derived gates
  // (fillFlag === null), use bar-time-window join: count fills whose
  // ts falls inside a bar where the flag was active. Approximate at
  // 5 s granularity; documented in the tooltip.
  let filteredFills: FillSinceRow[] = [];
  if (def.fillFlag !== null) {
    filteredFills = fills.filter(def.fillFlag);
  } else if (sortedBars.length > 0) {
    // Build set of active-bar ts ranges. Each bar covers [ts_bar,
    // ts_bar + 5s). Match fill if its ts falls in any active bar.
    const activeStarts: number[] = [];
    for (const bar of sortedBars) {
      if (!def.barFlag(bar)) continue;
      const t = Date.parse(String(bar.ts_bar));
      if (Number.isFinite(t)) activeStarts.push(t);
    }
    if (activeStarts.length > 0) {
      activeStarts.sort((a, b) => a - b);
      const cadenceMs = GATE_BAR_CADENCE_S * 1000;
      // Binary-search the active-starts for each fill — O(F log B).
      const isFillInActive = (fillTs: number): boolean => {
        let lo = 0;
        let hi = activeStarts.length - 1;
        while (lo <= hi) {
          const mid = (lo + hi) >> 1;
          const start = activeStarts[mid];
          if (fillTs >= start && fillTs < start + cadenceMs) return true;
          if (fillTs < start) hi = mid - 1;
          else lo = mid + 1;
        }
        return false;
      };
      filteredFills = fills.filter((f) => {
        const t = Date.parse(String(f.ts_fill));
        return Number.isFinite(t) && isFillInActive(t);
      });
    }
  }

  // Aggregate markouts + net $.
  const markouts: number[] = [];
  let notional = 0;
  let rebate = 0;
  let closedPnl = 0;
  for (const f of filteredFills) {
    if (gateIsFiniteNum(f.markout_5s_bps)) markouts.push(f.markout_5s_bps);
    if (gateIsFiniteNum(f.notional)) notional += f.notional;
    if (gateIsFiniteNum(f.fee)) rebate -= f.fee;
    if (gateIsFiniteNum(f.closed_pnl)) closedPnl += f.closed_pnl;
  }
  const meanMarkout =
    markouts.length > 0
      ? markouts.reduce((a, b) => a + b, 0) / markouts.length
      : null;
  const sortedMo = [...markouts].sort((a, b) => a - b);
  const medianMarkout =
    sortedMo.length > 0
      ? sortedMo.length % 2 === 0
        ? (sortedMo[sortedMo.length / 2 - 1] + sortedMo[sortedMo.length / 2]) / 2
        : sortedMo[Math.floor(sortedMo.length / 2)]
      : null;
  const markoutDollars =
    meanMarkout !== null ? (notional * meanMarkout) / 10_000 : 0;
  const netDollars =
    meanMarkout !== null ? rebate + markoutDollars + closedPnl : null;

  return {
    name: def.name,
    tooltip: def.tooltip,
    active_now: activeNow,
    time_active_s: activeBars * GATE_BAR_CADENCE_S,
    time_active_pct: totalBars > 0 ? (activeBars / totalBars) * 100 : 0,
    fire_count: fireCount,
    fills_during: filteredFills.length,
    mean_markout_5s_bps: meanMarkout,
    median_markout_5s_bps: medianMarkout,
    net_dollars: netDollars,
    confidence: sampleSizeLabel(filteredFills.length),
  };
}


/** Sorted-array percentile helper. Returns ``null`` for empty input.
 *  ``percentile=0.5`` returns the median, 0.95 the p95, etc. */
export function _percentile(sorted: number[], percentile: number): number | null {
  if (sorted.length === 0) return null;
  if (percentile <= 0) return sorted[0];
  if (percentile >= 1) return sorted[sorted.length - 1];
  const idx = Math.min(
    sorted.length - 1,
    Math.floor(percentile * sorted.length),
  );
  return sorted[idx];
}


/** min / median / p95 / max — the four stats the operator wants on every
 *  latency series (dropping mean per 1.3.82 operator preference). */
export interface LatencyStats {
  min: number | null;
  median: number | null;
  p95: number | null;
  max: number | null;
}


export function _latencyStats(latencies: number[]): LatencyStats {
  if (latencies.length === 0) {
    return { min: null, median: null, p95: null, max: null };
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  return {
    min: sorted[0],
    median: _percentile(sorted, 0.5),
    p95: _percentile(sorted, 0.95),
    max: sorted[sorted.length - 1],
  };
}


export interface CancelReasonRow {
  reason: string;
  count: number;
  pct: number;
  latency: LatencyStats;
  first_seen_ts: string | null;
  last_seen_ts: string | null;
}


export interface PlaceOutcomeRow {
  outcome: string;
  count: number;
  pct: number;
  latency: LatencyStats; // ts_sent -> ts_place_response (place RTT)
}


export interface ConnectivityFooter {
  orderAge: LatencyStats; // ts_closed - ts_ack (resting age)
  totalOrders: number;
  totalAcked: number;
  totalCancels: number;
  totalPlaceResponses: number;
}


/** Reject-detail row: the venue-side rejection reason (OKX sCode +
 *  sMsg) bucketed with count + %. ``benign=true`` when the rejection
 *  is operationally fine (post-only-would-cross, order-already-gone)
 *  vs ``false`` for attention-worthy failures (insufficient margin,
 *  rate limit, auth). */
export interface RejectDetailRow {
  detail: string;
  count: number;
  pct: number;
  benign: boolean;
}


/** Classify a place-response detail as benign vs attention.
 *  ``post_only_would_cross`` is the only common benign rejection on the
 *  place side — it just means the bot's price crossed the book and
 *  the venue refused; the operator-configured cooldown re-arms. */
export function _isBenignPlaceDetail(detail: string): boolean {
  const d = detail.toLowerCase();
  return d.includes("post_only_would_cross") || d.includes("51604");
}


/** Classify a cancel-response detail as benign vs attention. The
 *  benign codes are 51400/51401/51402/51503 — all variants of
 *  "order doesn't exist anymore" which is fine on a cancel attempt
 *  (the order had already filled / been cancelled / never existed).
 *  Anything else is attention-worthy (rate limit, auth, network). */
export function _isBenignCancelDetail(detail: string): boolean {
  const d = detail.toLowerCase();
  return (
    d.includes("51400") ||
    d.includes("51401") ||
    d.includes("51402") ||
    d.includes("51503") ||
    d.includes("has been filled") ||
    d.includes("has been canceled") ||
    d.includes("has been cancelled") ||
    d.includes("does not exist")
  );
}


/**
 * Pure aggregator for the Connectivity tab. Lifted out of the
 * ``ConnectivityPanel`` useMemo on 2026-05-16 (v1.3.90) so the
 * Download-Stats markdown builder can produce the SAME tables the
 * dashboard shows without re-implementing the same passes. Used in
 * exactly two places: the panel's useMemo, and
 * ``buildComprehensiveMarkdown``. Pure function on
 * ``OrdersLifecycleRow[]``; safe to call repeatedly.
 */
/** Sticky reject summary shape published by the backend in v1.4.6+.
 *  Source: BotState.reject_summary_snapshot() — accumulates over the
 *  whole session and NEVER decays. Used to populate the
 *  "Place rejects by venue reason" + "Cancel rejects by venue reason"
 *  rows so the 5000-row orders_lifecycle window can't wipe exchange
 *  errors from operator visibility. */
export interface StickyRejectSummary {
  place: Record<
    string,
    {
      count: number;
      outcome: string;
      is_benign: boolean;
      first_seen_ts: string;
      last_seen_ts: string;
    }
  >;
  cancel: Record<
    string,
    {
      count: number;
      outcome: string;
      is_benign: boolean;
      first_seen_ts: string;
      last_seen_ts: string;
    }
  >;
  place_total: number;
  cancel_total: number;
}


/** Per-outcome cumulative aggregate published in v1.4.6+.
 *  Source: BotState.outcome_aggregates_snapshot(). Survives the
 *  5000-row window eviction — drives total counts on the connectivity
 *  header so "N placements · X% ack rate" is correct for the whole
 *  session, not just the recent slice. */
export interface OutcomeAggregates {
  place: Record<
    string,
    {
      count: number;
      min_ms: number | null;
      max_ms: number | null;
      mean_ms: number | null;
      samples_with_latency: number;
    }
  >;
  cancel: Record<
    string,
    {
      count: number;
      min_ms: number | null;
      max_ms: number | null;
      mean_ms: number | null;
      samples_with_latency: number;
    }
  >;
}


export function computeConnectivityAggregates(
  rows: OrdersLifecycleRow[],
  sticky?: StickyRejectSummary | null,
  aggregates?: OutcomeAggregates | null,
): {
  cancelRows: CancelReasonRow[];
  placeRows: PlaceOutcomeRow[];
  footer: ConnectivityFooter;
  cancelRejects: RejectDetailRow[];
  placeRejects: RejectDetailRow[];
} {
  // ----- Cancels --------------------------------------------------------
  const byReason = new Map<
    string,
    {
      count: number;
      latencies: number[];
      first_ts: string | null;
      last_ts: string | null;
    }
  >();
  let totalCancels = 0;
  for (const r of rows) {
    const reasonRaw = (r as Record<string, unknown>).cancel_reason;
    if (reasonRaw === null || reasonRaw === undefined || reasonRaw === "") {
      continue;
    }
    const reason = String(reasonRaw);
    totalCancels++;
    const existing = byReason.get(reason) ?? {
      count: 0,
      latencies: [] as number[],
      first_ts: null as string | null,
      last_ts: null as string | null,
    };
    existing.count += 1;
    if (
      typeof r.cancel_to_close_ms === "number" &&
      Number.isFinite(r.cancel_to_close_ms) &&
      r.cancel_to_close_ms >= 0
    ) {
      existing.latencies.push(r.cancel_to_close_ms);
    }
    const ts = r.ts_cancel_requested || r.ts_closed;
    if (ts) {
      if (
        !existing.first_ts ||
        String(ts).localeCompare(existing.first_ts) < 0
      ) {
        existing.first_ts = String(ts);
      }
      if (
        !existing.last_ts ||
        String(ts).localeCompare(existing.last_ts) > 0
      ) {
        existing.last_ts = String(ts);
      }
    }
    byReason.set(reason, existing);
  }
  const cancelOut: CancelReasonRow[] = [];
  for (const [reason, g] of byReason) {
    cancelOut.push({
      reason,
      count: g.count,
      pct: totalCancels > 0 ? (g.count / totalCancels) * 100 : 0,
      latency: _latencyStats(g.latencies),
      first_seen_ts: g.first_ts,
      last_seen_ts: g.last_ts,
    });
  }
  cancelOut.sort((a, b) => b.count - a.count);

  // ----- Placements -----------------------------------------------------
  const byOutcome = new Map<string, { count: number; rtts: number[] }>();
  let totalPlaceResp = 0;
  for (const r of rows) {
    const oc = (r as Record<string, unknown>).place_response_outcome;
    const tsSent = r.ts_sent ? String(r.ts_sent) : null;
    const tsResp = (r as Record<string, unknown>).ts_place_response as
      | string
      | null
      | undefined;
    let outcomeStr: string;
    if (oc) {
      outcomeStr = String(oc);
    } else if (tsSent && !tsResp && r.status && r.status !== "PENDING") {
      outcomeStr = "no_response";
    } else if (!tsSent) {
      continue;
    } else {
      continue;
    }
    totalPlaceResp += 1;
    const existing = byOutcome.get(outcomeStr) ?? {
      count: 0,
      rtts: [] as number[],
    };
    existing.count += 1;
    if (tsSent && tsResp) {
      const dtMs =
        new Date(String(tsResp)).getTime() - new Date(tsSent).getTime();
      if (Number.isFinite(dtMs) && dtMs >= 0) {
        existing.rtts.push(dtMs);
      }
    }
    byOutcome.set(outcomeStr, existing);
  }
  // 1.4.6: when the backend ships ``outcome_aggregates``, prefer
  // those for total + per-outcome counts so the connectivity header
  // reflects the WHOLE session, not just the most-recent 5000 rows.
  // Latency stats stay from the row-window — by design those are
  // "current performance" rather than "lifetime."
  const placeOut: PlaceOutcomeRow[] = [];
  const aggPlace = aggregates?.place;
  if (aggPlace && Object.keys(aggPlace).length > 0) {
    let aggTotal = 0;
    for (const g of Object.values(aggPlace)) aggTotal += g.count;
    totalPlaceResp = aggTotal;
    for (const [outcome, g] of Object.entries(aggPlace)) {
      // Use recent-row latency samples for the per-outcome
      // distribution. If no samples in the window (e.g. older
      // outcomes evicted), fall back to the backend's lifetime
      // mean/min/max from the aggregate (no median/p95 there —
      // streaming percentiles weren't worth the histogram cost).
      const windowSamples = byOutcome.get(outcome);
      const latency =
        windowSamples && windowSamples.rtts.length > 0
          ? _latencyStats(windowSamples.rtts)
          : {
              min: g.min_ms,
              median: g.mean_ms, // approximation when no recent samples
              p95: g.max_ms,
              max: g.max_ms,
            };
      placeOut.push({
        outcome,
        count: g.count,
        pct: aggTotal > 0 ? (g.count / aggTotal) * 100 : 0,
        latency,
      });
    }
  } else {
    // Legacy per-row aggregation (older bot build).
    for (const [outcome, g] of byOutcome) {
      placeOut.push({
        outcome,
        count: g.count,
        pct: totalPlaceResp > 0 ? (g.count / totalPlaceResp) * 100 : 0,
        latency: _latencyStats(g.rtts),
      });
    }
  }
  placeOut.sort((a, b) => b.count - a.count);

  // ----- Order-age-at-market footer ------------------------------------
  const ageMs: number[] = [];
  let totalAcked = 0;
  for (const r of rows) {
    if (!r.ts_ack || !r.ts_closed) continue;
    totalAcked += 1;
    const ackT = new Date(String(r.ts_ack)).getTime();
    const closedT = new Date(String(r.ts_closed)).getTime();
    const dt = closedT - ackT;
    if (Number.isFinite(dt) && dt >= 0) {
      ageMs.push(dt);
    }
  }

  // ----- Reject-detail breakdowns --------------------------------------
  //
  // 1.4.6: prefer the sticky cumulative summary published by the
  // backend over per-row aggregation. The 5000-row orders_lifecycle
  // window evicts older rows under churn, which silently wiped
  // exchange errors from the dashboard (operator observation
  // 2026-05-17). The sticky summary survives across the eviction.
  //
  // Fallback to per-row aggregation when the sticky summary is
  // absent (older bot build pre-1.4.6) or empty (no rejections yet
  // recorded in this session).
  const cancelRejectsOut: RejectDetailRow[] = [];
  const placeRejectsOut: RejectDetailRow[] = [];

  const stickyHasPlace =
    sticky &&
    sticky.place &&
    Object.keys(sticky.place).length > 0;
  const stickyHasCancel =
    sticky &&
    sticky.cancel &&
    Object.keys(sticky.cancel).length > 0;

  if (stickyHasPlace) {
    const total = sticky!.place_total || 0;
    for (const [detail, g] of Object.entries(sticky!.place)) {
      placeRejectsOut.push({
        detail,
        count: g.count,
        pct: total > 0 ? (g.count / total) * 100 : 0,
        benign: !!g.is_benign,
      });
    }
  } else {
    // Legacy per-row fallback. Kept verbatim so older snapshots / older
    // bot builds without the sticky summary still surface rejections
    // (just subject to the 5000-row window-eviction quirk).
    const placeRejectMap = new Map<
      string,
      { count: number; benign: boolean }
    >();
    let totalPlaceRejects = 0;
    for (const r of rows) {
      const oc = (r as Record<string, unknown>).place_response_outcome;
      if (!oc || oc === "accepted") continue;
      const detail =
        ((r as Record<string, unknown>).place_response_detail as
          | string
          | null
          | undefined) || `(${String(oc)} / no detail)`;
      const benign = _isBenignPlaceDetail(String(detail));
      totalPlaceRejects += 1;
      const existing = placeRejectMap.get(String(detail)) ?? {
        count: 0,
        benign,
      };
      existing.count += 1;
      placeRejectMap.set(String(detail), existing);
    }
    for (const [detail, g] of placeRejectMap) {
      placeRejectsOut.push({
        detail,
        count: g.count,
        pct: totalPlaceRejects > 0 ? (g.count / totalPlaceRejects) * 100 : 0,
        benign: g.benign,
      });
    }
  }
  placeRejectsOut.sort((a, b) => b.count - a.count);

  if (stickyHasCancel) {
    const total = sticky!.cancel_total || 0;
    for (const [detail, g] of Object.entries(sticky!.cancel)) {
      cancelRejectsOut.push({
        detail,
        count: g.count,
        pct: total > 0 ? (g.count / total) * 100 : 0,
        benign: !!g.is_benign,
      });
    }
  } else {
    const cancelRejectMap = new Map<
      string,
      { count: number; benign: boolean }
    >();
    let totalCancelRejects = 0;
    for (const r of rows) {
      const oc = (r as Record<string, unknown>).cancel_response_outcome;
      if (!oc || oc === "success") continue;
      const detail =
        ((r as Record<string, unknown>).cancel_response_detail as
          | string
          | null
          | undefined) || `(${String(oc)} / no detail)`;
      const benign = _isBenignCancelDetail(String(detail));
      totalCancelRejects += 1;
      const existing = cancelRejectMap.get(String(detail)) ?? {
        count: 0,
        benign,
      };
      existing.count += 1;
      cancelRejectMap.set(String(detail), existing);
    }
    for (const [detail, g] of cancelRejectMap) {
      cancelRejectsOut.push({
        detail,
        count: g.count,
        pct:
          totalCancelRejects > 0 ? (g.count / totalCancelRejects) * 100 : 0,
        benign: g.benign,
      });
    }
  }
  cancelRejectsOut.sort((a, b) => b.count - a.count);

  return {
    cancelRows: cancelOut,
    placeRows: placeOut,
    footer: {
      orderAge: _latencyStats(ageMs),
      totalOrders: rows.length,
      totalAcked,
      totalCancels: (() => {
        // 1.4.6: lifetime cancel count from aggregates when available.
        const aggCancel = aggregates?.cancel;
        if (aggCancel && Object.keys(aggCancel).length > 0) {
          let s = 0;
          for (const g of Object.values(aggCancel)) s += g.count;
          return s;
        }
        return totalCancels;
      })(),
      totalPlaceResponses: totalPlaceResp,
    },
    cancelRejects: cancelRejectsOut,
    placeRejects: placeRejectsOut,
  };
}


export interface BasisBucketRow {
  bucket: string;
  n_bars: number;
  minutes: number;
  pct_of_session: number;
}


export const BASIS_BAR_CADENCE_S = 5.0;


export const BASIS_ORDER = [
  "NEG_STRETCHED",
  "NEG",
  "FLAT",
  "POS",
  "POS_STRETCHED",
] as const;


/** Block E: time per basis bucket. */
export function computeBasisDwell(bars: ExposureBarRow[]): BasisBucketRow[] {
  const counts = new Map<string, number>();
  for (const b of bars) {
    const bucket = basisBucket(b.basis_regime_sign, b.binance_basis_ewma);
    if (bucket === null) continue;
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }
  const total = bars.length;
  return BASIS_ORDER.map((bucket) => {
    const n = counts.get(bucket) ?? 0;
    return {
      bucket,
      n_bars: n,
      minutes: (n * BASIS_BAR_CADENCE_S) / 60,
      pct_of_session: total > 0 ? (n / total) * 100 : 0,
    };
  });
}


export interface InvBucketRow {
  bucket: string;
  n_bars: number;
  minutes: number;
  pct_of_session: number;
}


export interface InvTransitionRow {
  from: string;
  to: string;
  count: number;
  /** Average duration spent in ``from`` state before this transition,
   *  in seconds. */
  avg_duration_before_s: number;
}


export interface InvActiveSidesRow {
  active_sides: string;
  n: number;
  mean_markout_5s_bps: number | null;
  net_usd: number | null;
  confidence: SampleSizeTier;
}


export const INV_BAR_CADENCE_S = 5.0;


export function invIsFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}


/** Compute Block A: time-in-bucket. Buckets bars by
 *  ``inventoryBucket(inventory_utilization)`` and counts minutes
 *  exposed per bucket. */
export function computeInventoryDwell(bars: ExposureBarRow[]): InvBucketRow[] {
  const order = ["FLAT", "LOW", "MED", "HIGH"];
  const counts = new Map<string, number>();
  for (const b of bars) {
    const bucket = inventoryBucket(b.inventory_utilization);
    if (bucket === null) continue;
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }
  const total = bars.length;
  return order.map((bucket) => {
    const n = counts.get(bucket) ?? 0;
    return {
      bucket,
      n_bars: n,
      minutes: (n * INV_BAR_CADENCE_S) / 60,
      pct_of_session: total > 0 ? (n / total) * 100 : 0,
    };
  });
}


/** Compute the one-sided dwell table — same shape as Block A but
 *  bucketed by ``quote_eligibility`` (fall back to ``active_sides``). */
export function computeOneSidedDwell(bars: ExposureBarRow[]): InvBucketRow[] {
  const order = ["BOTH", "BID_ONLY", "ASK_ONLY", "HOLD_ALL"];
  const counts = new Map<string, number>();
  for (const b of bars) {
    const mode = modeBucket(b.active_sides, b.quote_eligibility);
    if (mode === null) continue;
    counts.set(mode, (counts.get(mode) ?? 0) + 1);
  }
  const total = bars.length;
  return order.map((bucket) => {
    const n = counts.get(bucket) ?? 0;
    return {
      bucket,
      n_bars: n,
      minutes: (n * INV_BAR_CADENCE_S) / 60,
      pct_of_session: total > 0 ? (n / total) * 100 : 0,
    };
  });
}


/** Compute Block B: quote-eligibility transitions. Walks bars in
 *  ts order, records (prev_state → curr_state) transitions, sums
 *  duration spent in the prev_state between transitions. */
export function computeTransitions(bars: ExposureBarRow[]): InvTransitionRow[] {
  if (bars.length < 2) return [];
  // Sort by ts_bar ascending.
  const sorted = [...bars].sort((a, b) =>
    String(a.ts_bar).localeCompare(String(b.ts_bar)),
  );
  // Walk: track current state + when it started; on state-change emit
  // transition record.
  let currentState: string | null = null;
  let stateStartBarIdx = 0;
  const byKey = new Map<
    string,
    { count: number; total_duration_bars: number }
  >();
  for (let i = 0; i < sorted.length; i++) {
    const mode = modeBucket(sorted[i].active_sides, sorted[i].quote_eligibility);
    if (mode === null) continue;
    if (currentState === null) {
      currentState = mode;
      stateStartBarIdx = i;
      continue;
    }
    if (mode !== currentState) {
      const key = `${currentState}|${mode}`;
      const dur = i - stateStartBarIdx;
      const existing = byKey.get(key) ?? { count: 0, total_duration_bars: 0 };
      existing.count += 1;
      existing.total_duration_bars += dur;
      byKey.set(key, existing);
      currentState = mode;
      stateStartBarIdx = i;
    }
  }
  const rows: InvTransitionRow[] = [];
  for (const [key, agg] of byKey) {
    const [from, to] = key.split("|");
    rows.push({
      from,
      to,
      count: agg.count,
      avg_duration_before_s:
        (agg.total_duration_bars / agg.count) * INV_BAR_CADENCE_S,
    });
  }
  // Sort: most common first, then alphabetical.
  rows.sort((a, b) => b.count - a.count || a.from.localeCompare(b.from));
  return rows;
}


/** Compute Block D: fills by ``active_sides_at_decision``. */
export function computeFillsByActiveSides(fills: FillSinceRow[]): InvActiveSidesRow[] {
  const order = ["BOTH", "BID_ONLY", "ASK_ONLY", "HOLD_ALL"];
  const byMode = new Map<string, FillSinceRow[]>();
  for (const f of fills) {
    const mode = modeBucket(
      f.active_sides_at_decision,
      f.quote_eligibility_state,
    );
    if (mode === null) continue;
    if (!byMode.has(mode)) byMode.set(mode, []);
    byMode.get(mode)!.push(f);
  }
  return order.map((mode) => {
    const slice = byMode.get(mode) ?? [];
    if (slice.length === 0) {
      return {
        active_sides: mode,
        n: 0,
        mean_markout_5s_bps: null,
        net_usd: null,
        confidence: "anecdote",
      };
    }
    const markouts: number[] = [];
    let notional = 0;
    let rebate = 0;
    let closedPnl = 0;
    for (const f of slice) {
      if (invIsFiniteNum(f.markout_5s_bps)) markouts.push(f.markout_5s_bps);
      if (invIsFiniteNum(f.notional)) notional += f.notional;
      if (invIsFiniteNum(f.fee)) rebate -= f.fee;
      if (invIsFiniteNum(f.closed_pnl)) closedPnl += f.closed_pnl;
    }
    const meanMarkout =
      markouts.length > 0
        ? markouts.reduce((a, b) => a + b, 0) / markouts.length
        : null;
    const markoutDollars =
      meanMarkout !== null ? (notional * meanMarkout) / 10_000 : 0;
    const netUsd =
      meanMarkout !== null ? rebate + markoutDollars + closedPnl : null;
    return {
      active_sides: mode,
      n: slice.length,
      mean_markout_5s_bps: meanMarkout,
      net_usd: netUsd,
      confidence: sampleSizeLabel(slice.length),
    };
  });
}


/** Build a comprehensive markdown export of the dashboard's current
 *  state — not just the Bot Stats panel, but session counters, account
 *  balances, position, rolling volume windows, and live-stats.
 *  Triggered by the Download button in the Bot Stats header.
 *  Filename: ``bot-stats-<YYYYMMDDHHMMSS>.md``.
 *
 *  Sections (in order):
 *    1. Header (profile, captured-at, bot version, run state)
 *    2. Session activity (PnL, fills, new, actions, volume)
 *    3. Account (equity, withdrawable, unrealized PnL)
 *    4. Position (qty, notional, avg entry, unrealized)
 *    5. Rolling volume windows (1d/7d/15d/30d, fill count + notional)
 *    6. Venue config (leverage, margin mode, position mode)
 *    7. Live stats (working orders, venue/reference, basis, inventory,
 *       skew, markouts, book signals, PnL attribution, fill buckets)
 *    8. Risk thresholds (current caps from profile env)
 *
 *  Numeric fields preserve on-screen precision so the snapshot is
 *  round-trippable for operator review.
 */
export function buildComprehensiveMarkdown(args: {
  state: AccountState;
  liveStats: LiveStatsPayload | null;
  botStatus: BotStatusInfo | null;
  volume: VolumeResponse | null;
  venueInfo: VenueInfo | null;
  accountRisk: AccountRiskThresholds | undefined;
  history: HistoryResponse | null;
  // 2026-05-13 v1.3.25: dashboard-wide download. These two were
  // previously omitted because the report was Bot-Stats-only; now
  // included so the export covers every surface the dashboard renders.
  equityHistory: EquityHistoryPayload | null;
  configResult: ConfigResult | null;
  sessionStartedAtUtc: string | null;
  // 2026-05-16 v1.3.90: include the Regime tab's per-axis tables.
  // The operator asked for the Download to mirror EVERY panel; the
  // Regime tab was the major gap. We re-run ``aggregateRegimes``
  // inside the builder rather than passing pre-computed results so
  // the export reflects exactly the data the dashboard is reading
  // at click-time, not a stale memo.
  regimeFills: DashboardStateResult<FillSinceRow> | null;
  regimeBars: DashboardStateResult<ExposureBarRow> | null;
  // 2026-05-16 v1.3.90: include the Connectivity tab tables.
  // ``computeConnectivityAggregates`` runs on this in the builder
  // so the markdown matches the dashboard's panel exactly.
  ordersLifecycle: DashboardStateResult<OrdersLifecycleRow> | null;
}): string {
  const {
    state,
    liveStats: p,
    botStatus,
    volume,
    venueInfo,
    accountRisk,
    history,
    equityHistory,
    configResult,
    sessionStartedAtUtc,
    regimeFills,
    regimeBars,
    ordersLifecycle,
  } = args;
  const lines: string[] = [];
  const fmt = (v: number | null | undefined, dp = 4): string =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "—"
      : v.toFixed(dp);
  const fmtBp = (v: number | null | undefined): string =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "—"
      : `${v.toFixed(2)} bp`;
  const fmtUsd = (v: number | null | undefined): string =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "—"
      : `$${v.toFixed(4)}`;
  const fmtUsd2 = (v: number | null | undefined): string =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "—"
      : `$${v.toFixed(2)}`;
  const fmtPct = (v: number | null | undefined): string =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "—"
      : `${(v * 100).toFixed(1)}%`;

  // ----- Header ---------------------------------------------------
  lines.push(`# Dashboard snapshot — ${state.account_id}`);
  lines.push("");
  lines.push(`- Captured: \`${new Date().toISOString()}\``);
  lines.push(`- Venue: \`${state.venue.toUpperCase()}\` · Symbol: \`${state.symbol}\``);
  if (botStatus) {
    if (botStatus.bot_version) {
      lines.push(`- Bot version: \`${botStatus.bot_version}\``);
    }
    if (botStatus.bot_status) {
      lines.push(`- Bot status: \`${botStatus.bot_status}\`${botStatus.bot_trading_enabled === false ? " (trading disabled)" : ""}`);
    }
    if (botStatus.bot_killed) {
      lines.push(`- KILLED: \`${botStatus.bot_kill_reason ?? "(unknown reason)"}\` at \`${botStatus.bot_kill_timestamp_utc ?? "?"}\``);
    }
    if (botStatus.bot_manual_pause) {
      lines.push(`- PAUSED: \`${botStatus.bot_pause_reason ?? "(unknown)"}\` at \`${botStatus.bot_pause_timestamp_utc ?? "?"}\``);
    }
    if (botStatus.bot_session_started_at_utc) {
      lines.push(`- Session started: \`${botStatus.bot_session_started_at_utc}\``);
    }
    if (botStatus.bot_heartbeat_at_utc) {
      lines.push(`- Last heartbeat: \`${botStatus.bot_heartbeat_at_utc}\``);
    }
  }
  if (state.errors && state.errors.length > 0) {
    lines.push(`- Errors: ${state.errors.length}`);
    for (const e of state.errors) lines.push(`  - ${e}`);
  }
  lines.push("");

  // ----- Analysis context (sign conventions + session age) --------
  // Surfaced near the top because this snapshot is intended for
  // off-line LLM cross-analysis (Codex review). The LLM needs the
  // sign-convention reminder before it interprets any "fee" /
  // "markout" / "drawdown" numbers, and the session-age framing
  // lets it correctly scale session-cumulative counters.
  {
    const nowMs = Date.now();
    const sessionStartMs = sessionStartedAtUtc
      ? new Date(sessionStartedAtUtc).getTime()
      : NaN;
    const sessionAgeS = Number.isFinite(sessionStartMs)
      ? Math.max(0, (nowMs - sessionStartMs) / 1000)
      : null;
    const fmtAgeS = (s: number): string => {
      if (s < 60) return `${s.toFixed(0)} s`;
      if (s < 3600) return `${(s / 60).toFixed(1)} min`;
      return `${(s / 3600).toFixed(2)} h`;
    };
    lines.push("## Analysis context");
    lines.push("");
    lines.push(
      `- Snapshot intent: off-line cross-analysis (LLM-readable, not a live ops tool).`,
    );
    lines.push(
      `- Capture clock: \`${new Date().toISOString()}\` (UTC)`,
    );
    if (sessionAgeS !== null) {
      lines.push(
        `- Session age: \`${fmtAgeS(sessionAgeS)}\` (started \`${sessionStartedAtUtc}\`). All "session-cumulative" counters below scale to this age.`,
      );
    }
    lines.push("");
    lines.push("**Sign conventions** (critical — the bot's house style):");
    lines.push("");
    lines.push(
      `- \`fee\` (per-fill and aggregate): POSITIVE = cost paid, NEGATIVE = rebate received. Net negative is GOOD (we made rebate income on net).`,
    );
    lines.push(
      `- \`rebate_usd\` (regime tables, PnL attribution): pre-flipped so POSITIVE = rebate income (intuitive). Equals \`-Σfee\` over the bucket.`,
    );
    lines.push(
      `- \`markout_*_bps\` (1s / 5s / 30s): SIGNED FROM THE BOT'S P-AND-L PERSPECTIVE. POSITIVE = favourable post-fill drift, NEGATIVE = adverse (picked off). Median > 0 is GOOD.`,
    );
    lines.push(
      `- \`closed_pnl\` and \`realized_pnl_usd\`: standard sign — POSITIVE = profit.`,
    );
    lines.push(
      `- \`drawdown_usd\` / \`drawdown\`: distance below session-peak equity. POSITIVE magnitude, zero at peak.`,
    );
    lines.push(
      `- \`basis_okx_minus_binance_bps\`: signed difference in bp. POSITIVE = target above reference. (Field name retains \`okx_minus_binance\` for backend-compat; semantically: target_mid − reference_mid.)`,
    );
    lines.push(
      `- \`skew.bps\`: SIGNED reservation-price offset. POSITIVE = bias to sell (long inventory); NEGATIVE = bias to buy (short inventory).`,
    );
    lines.push(
      `- \`norm_inventory\`: ratio in [-1, +1]. Sign matches position; magnitude is utilization.`,
    );
    lines.push(
      `- \`adverse_pct_5s\`: percentage in [0, 100], NOT a 0..1 fraction.`,
    );
    lines.push("");
  }

  // ----- Session activity (matches the headline chips) -----------
  if (botStatus) {
    lines.push("## Session activity");
    lines.push(`- Fills: ${botStatus.bot_session_fill_count ?? "—"}`);
    lines.push(`- New orders (place attempts): ${botStatus.bot_session_place_attempt_count ?? "—"}`);
    lines.push(`- Actions: ${botStatus.bot_session_action_count ?? "—"}`);
    lines.push(`- Traded notional: ${fmtUsd2(botStatus.bot_session_traded_notional_usd)}`);
    if (botStatus.bot_execution_idle_seconds !== null && botStatus.bot_execution_idle_seconds !== undefined) {
      lines.push(`- Execution idle: ${botStatus.bot_execution_idle_seconds.toFixed(0)} s`);
    }
    if (botStatus.bot_suppression_rate_60s !== null && botStatus.bot_suppression_rate_60s !== undefined) {
      lines.push(`- Suppression rate (60s): ${(botStatus.bot_suppression_rate_60s * 100).toFixed(1)}%`);
    }
    lines.push("");
  }

  // ----- Derived performance ratios (LLM-friendly headline) -------
  // Numbers an analyst LLM frequently wants but that don't ship
  // directly on any payload. Computed inline from the existing
  // session counters + recent fills + venue config so the LLM
  // doesn't have to re-derive them (and can't get the formula
  // subtly wrong). Each ratio carries a one-line interpretation
  // hint so the LLM doesn't have to guess what "high" / "low"
  // means.
  if (botStatus) {
    const fills = botStatus.bot_session_fill_count ?? null;
    const places = botStatus.bot_session_place_attempt_count ?? null;
    const traded = botStatus.bot_session_traded_notional_usd ?? null;
    const sessionStartMs = sessionStartedAtUtc
      ? new Date(sessionStartedAtUtc).getTime()
      : NaN;
    const ageHours = Number.isFinite(sessionStartMs)
      ? Math.max(0.0001, (Date.now() - sessionStartMs) / 3_600_000)
      : null;
    // Buy/sell split from publisher fills (BotStatus doesn't break
    // it down). Skip if publisher data is missing.
    let buyN = 0;
    let sellN = 0;
    let buyNotional = 0;
    let sellNotional = 0;
    const fillsForSide = regimeFills?.payload?.rows ?? [];
    for (const f of fillsForSide) {
      const s = String(f.side || "").toUpperCase();
      const n = Number.isFinite(f.notional) ? (f.notional ?? 0) : 0;
      if (s === "BUY") {
        buyN++;
        buyNotional += n;
      } else if (s === "SELL") {
        sellN++;
        sellNotional += n;
      }
    }
    // Net rebate + total fee accounting from publisher fills.
    // ``fee`` sign convention: positive = paid, negative = rebate.
    let totalFeePaid = 0;
    let totalRebateReceived = 0;
    let totalClosed = 0;
    for (const f of fillsForSide) {
      const fee = Number.isFinite(f.fee) ? (f.fee ?? 0) : 0;
      if (fee > 0) totalFeePaid += fee;
      else if (fee < 0) totalRebateReceived += -fee;
      const cp = Number.isFinite(f.closed_pnl) ? (f.closed_pnl ?? 0) : 0;
      totalClosed += cp;
    }
    const netFeeFlow = totalFeePaid - totalRebateReceived;
    const grossFee = totalFeePaid + totalRebateReceived;

    lines.push("## Derived performance ratios");
    lines.push("");
    // ----- Activity ratios
    if (fills !== null && places !== null && places > 0) {
      lines.push(
        `- **Fill rate**: ${((fills / places) * 100).toFixed(2)}% (${fills} fills / ${places} place attempts). _Lower means the bot is placing but quickly cancelling/missing — could be queue priority issue or too-aggressive cancel logic._`,
      );
    }
    if (ageHours !== null && fills !== null) {
      lines.push(
        `- **Fills per hour**: ${(fills / ageHours).toFixed(1)} (${fills} fills over ${ageHours.toFixed(2)} h)`,
      );
    }
    if (ageHours !== null && botStatus.bot_session_action_count !== null && botStatus.bot_session_action_count !== undefined) {
      lines.push(
        `- **Actions per hour**: ${(botStatus.bot_session_action_count / ageHours).toFixed(1)} (cancel + place + amend events). _Compares to OKX MM-tier 600 req/s headroom — see venue rate limit._`,
      );
    }
    if (fills !== null && fills > 0 && traded !== null) {
      lines.push(
        `- **Avg fill notional**: ${fmtUsd2(traded / fills)} (${fmtUsd2(traded)} / ${fills} fills)`,
      );
    }
    // ----- Inventory turnover
    if (
      traded !== null &&
      accountRisk &&
      accountRisk.max_position_notional_usd &&
      accountRisk.max_position_notional_usd > 0
    ) {
      const turnover = traded / accountRisk.max_position_notional_usd;
      lines.push(
        `- **Inventory turnover**: ${turnover.toFixed(2)}× of max-position cap (${fmtUsd2(traded)} traded / ${fmtUsd2(accountRisk.max_position_notional_usd)} cap). _High turnover at low net position = healthy market-making churn._`,
      );
    }
    // ----- Fee economics (from publisher fills, not session_pnl)
    if (fillsForSide.length > 0) {
      lines.push(
        `- **Fee economics** (over ${fillsForSide.length} publisher fills): rebate received ${fmtUsd(totalRebateReceived)} · fees paid ${fmtUsd(totalFeePaid)} · NET fee flow ${fmtUsd(netFeeFlow)} (NEGATIVE = net rebate income, GOOD).`,
      );
      if (grossFee > 0) {
        lines.push(
          `- **Rebate share of gross fees**: ${((totalRebateReceived / grossFee) * 100).toFixed(1)}% (${fmtUsd(totalRebateReceived)} rebate / ${fmtUsd(grossFee)} gross). _Higher = bot fills are mostly maker (good)._`,
        );
      }
      if (fillsForSide.length > 0) {
        lines.push(
          `- **Avg rebate per fill**: ${fmtUsd(totalRebateReceived / fillsForSide.length)} · **Avg net fee flow per fill**: ${fmtUsd(netFeeFlow / fillsForSide.length)}`,
        );
      }
    }
    // ----- Buy/sell imbalance
    if (buyN + sellN > 0) {
      const totalN = buyN + sellN;
      const buyPct = (buyN / totalN) * 100;
      const sellPct = (sellN / totalN) * 100;
      const skew = buyN - sellN;
      const skewPct =
        totalN > 0 ? Math.abs(skew) / totalN * 100 : 0;
      lines.push(
        `- **Buy/sell fill split**: ${buyN} buys (${buyPct.toFixed(1)}%) vs ${sellN} sells (${sellPct.toFixed(1)}%) — net ${skew > 0 ? "+" : ""}${skew} (skew ${skewPct.toFixed(1)}%). _Large skew = bot is getting hit on one side disproportionately (adverse selection signal)._`,
      );
      lines.push(
        `- **Buy/sell notional split**: ${fmtUsd2(buyNotional)} / ${fmtUsd2(sellNotional)} (delta ${fmtUsd2(buyNotional - sellNotional)})`,
      );
    }
    // ----- Net realized closed-position PnL from publisher fills
    if (fillsForSide.length > 0) {
      lines.push(
        `- **Closed-PnL from publisher fills**: ${fmtUsd(totalClosed)}. _Sum of \`closed_pnl\` field per fill — what the bot has realised from round-trips. Distinct from \`session_pnl.realized_usd\` which includes funding._`,
      );
    }
    lines.push("");
  }

  // ----- Account --------------------------------------------------
  lines.push("## Account");
  lines.push(`- Equity: ${fmtUsd2(state.account.equity_usd)}`);
  lines.push(`- Cash: ${fmtUsd2(state.account.cash_usd)}`);
  lines.push(`- Withdrawable: ${fmtUsd2(state.account.withdrawable_usd)}`);
  lines.push(`- Unrealized PnL: ${fmtUsd(state.account.unrealized_pnl_usd)}`);
  lines.push("");

  // ----- Position -------------------------------------------------
  lines.push("## Position");
  lines.push(`- Symbol: \`${state.position.symbol}\``);
  lines.push(`- Qty: ${fmt(state.position.qty)}`);
  lines.push(`- Notional: ${fmtUsd2(state.position.notional_usd)}`);
  lines.push(`- Avg entry: ${fmt(state.position.avg_entry_price)}`);
  lines.push(`- Mark price: ${fmt(state.position.mark_price)}`);
  lines.push(`- Unrealized PnL: ${fmtUsd(state.position.unrealized_pnl_usd)}`);
  lines.push("");

  // ----- Rolling volume windows ----------------------------------
  if (volume && volume.windows.length > 0) {
    lines.push("## Rolling volume");
    lines.push("");
    lines.push("| Window | Fills | Notional |");
    lines.push("|---|---:|---:|");
    for (const w of volume.windows) {
      lines.push(`| ${w.label} | ${w.fill_count} | ${fmtUsd2(w.notional_usd)} |`);
    }
    if (!volume.complete) {
      lines.push("");
      lines.push(`> Note: pagination did not reach the cutoff (${volume.total_fills_scanned} fills scanned, oldest \`${volume.oldest_fill_utc ?? "?"}\`). Older windows may under-count.`);
    }
    lines.push("");
  }

  // ----- Venue config --------------------------------------------
  if (venueInfo && (venueInfo.leverage || venueInfo.margin_mode || venueInfo.position_mode)) {
    lines.push("## Venue config");
    if (venueInfo.leverage) lines.push(`- Leverage: \`${venueInfo.leverage}\``);
    if (venueInfo.margin_mode) lines.push(`- Margin mode: \`${venueInfo.margin_mode}\``);
    if (venueInfo.position_mode) lines.push(`- Position mode: \`${venueInfo.position_mode}\``);
    if (venueInfo.account_level) lines.push(`- Account level: \`${venueInfo.account_level}\``);
    lines.push("");
  }

  // ----- Risk thresholds (current bot caps) ----------------------
  if (accountRisk) {
    lines.push("## Risk thresholds (current)");
    lines.push(`- MAX_ORDER_NOTIONAL_USD: ${fmtUsd2(accountRisk.max_order_notional_usd)}`);
    lines.push(`- MAX_POSITION_NOTIONAL_USD: ${fmtUsd2(accountRisk.max_position_notional_usd)}`);
    lines.push(`- MAX_ABS_POSITION: ${fmt(accountRisk.max_abs_position, 4)}`);
    lines.push(`- QUOTE_NOTIONAL_USD: ${fmtUsd2(accountRisk.quote_notional_usd)}`);
    lines.push("");
  }

  // ----- History counts (just the rollup; full history isn't dumped) ----
  if (history) {
    const opens = history.orders.filter((o: Order) => o.status === "OPEN" || o.status === "ACKED").length;
    lines.push("## History");
    lines.push(`- Open orders (live snapshot): ${state.open_orders.length}`);
    lines.push(`- Recent orders (last 7 days): ${history.orders.length} (${opens} live)`);
    lines.push(`- Recent fills (last 7 days): ${history.fills.length}`);
    lines.push("");
  }

  // ----- Open orders detail (from venue state) -------------------
  // The Open tab on the dashboard shows this table; include it in the
  // report so the operator can paste a snapshot into an incident log
  // without screenshotting the dashboard.
  if (state.open_orders.length > 0) {
    lines.push("## Open orders (live snapshot)");
    lines.push("");
    lines.push("| Side | Price | Qty | Order ID | Client ID | State | Age (ms) |");
    lines.push("|---|---:|---:|---|---|---|---:|");
    const now = Date.now();
    for (const o of state.open_orders) {
      const ageMs = o.time_ms ? now - o.time_ms : null;
      lines.push(
        `| ${o.side} | ${fmt(o.price, 6)} | ${fmt(o.qty, 4)} | \`${o.order_id}\` | \`${o.client_order_id || "—"}\` | ${o.state} | ${ageMs === null ? "—" : ageMs.toFixed(0)} |`,
      );
    }
    lines.push("");
  }

  // ----- Recent orders + fills tail (last 10 each) ----------------
  // The Order History / Fill History tabs show these; include the
  // tail in the report so the operator sees the most recent activity
  // alongside the rollup counts above.
  if (history && history.orders.length > 0) {
    lines.push("## Recent orders (last 10)");
    lines.push("");
    lines.push("| Time (UTC) | Side | Price | Qty | Filled | Status | Notional |");
    lines.push("|---|---|---:|---:|---:|---|---:|");
    for (const o of history.orders.slice(0, 10)) {
      const ts = new Date(o.time_ms).toISOString();
      lines.push(
        `| ${ts} | ${o.side} | ${fmt(o.price, 6)} | ${fmt(o.orig_qty, 4)} | ${fmt(o.filled_qty, 4)} | ${o.status} | ${fmtUsd2(o.notional_usd)} |`,
      );
    }
    lines.push("");
  }
  if (history && history.fills.length > 0) {
    lines.push("## Recent fills (last 10)");
    lines.push("");
    lines.push("| Time (UTC) | Side | Price | Qty | Notional | Fee | Maker? | Realized |");
    lines.push("|---|---|---:|---:|---:|---:|---|---:|");
    for (const f of history.fills.slice(0, 10)) {
      const ts = new Date(f.time_ms).toISOString();
      lines.push(
        `| ${ts} | ${f.side} | ${fmt(f.price, 6)} | ${fmt(f.qty, 4)} | ${fmtUsd2(f.notional_usd)} | ${fmt(f.fee, 6)} ${f.fee_ccy} | ${f.is_maker ? "Y" : "N"} | ${fmtUsd(f.realized_pnl)} |`,
      );
    }
    lines.push("");
  }

  // ----- Latency panel (right rail) -------------------------------
  // The Latency right-rail card surfaces order-RTT + WS exchange→
  // receive delay + per-venue gap stats. Include here in the same
  // shape (median + p95 + sample count) for completeness.
  if (botStatus && botStatus.latency) {
    const lat = botStatus.latency;
    const wsStats = lat.exchange_to_local_receive_ms ?? null;
    // v1.4.58 todo-037: unified TX → ack. Pre-v1.4.58 fallback rows
    // removed in v1.4.81 — they only flickered on during the new
    // bot's startup window and confused operators.
    const txRttStats = lat.tx_submit_rtt_ms ?? null;
    if (wsStats || txRttStats) {
      lines.push("## Latency");
      if (wsStats) {
        lines.push(
          `- Exchange → receive: median ${fmt(wsStats.median_ms, 1)} ms · p95 ${fmt(wsStats.p95_ms, 1)} ms · n=${wsStats.sample_count}`,
        );
      }
      if (txRttStats) {
        lines.push(
          `- Tx send → ack (place + amend + cancel): median ${fmt(txRttStats.median_ms, 1)} ms · p95 ${fmt(txRttStats.p95_ms, 1)} ms · n=${txRttStats.sample_count}`,
        );
      }
      lines.push("");
    }
  }
  if (p && p.gap_stats) {
    const g = p.gap_stats;
    lines.push("### Trading venue book-update gaps");
    lines.push(
      `- median ${fmt(g.median_gap_ms, 1)} ms · p95 ${fmt(g.p95_gap_ms, 1)} ms · min ${fmt(g.min_gap_ms, 1)} ms · max ${fmt(g.max_gap_ms, 1)} ms · n=${g.update_count}`,
    );
    if (g.last_gap_ms !== null) {
      lines.push(`- Last gap: ${fmt(g.last_gap_ms, 1)} ms`);
    }
    lines.push("");
  }
  if (p && p.gap_stats_reference) {
    const g = p.gap_stats_reference;
    lines.push("### Reference venue book-update gaps");
    lines.push(
      `- median ${fmt(g.median_gap_ms, 1)} ms · p95 ${fmt(g.p95_gap_ms, 1)} ms · min ${fmt(g.min_gap_ms, 1)} ms · max ${fmt(g.max_gap_ms, 1)} ms · n=${g.update_count}`,
    );
    lines.push("");
  }

  // ----- Live stats payload (everything below was bot-stats-only earlier) ---
  if (!p) {
    lines.push("## Live stats");
    lines.push("");
    lines.push("> Live stats payload not available (bot may be on a version that doesn't publish, or the S3 read failed).");
    lines.push("");
    return lines.join("\n");
  }

  lines.push("## Live stats");
  lines.push(`- Profile: \`${p.profile}\``);
  lines.push(`- Bot version: \`${p.version}\``);
  lines.push(`- Captured: \`${p.captured_at_utc}\``);
  lines.push(`- Interval: ${p.interval_seconds.toFixed(1)} s`);
  if (sessionStartedAtUtc) {
    lines.push(`- Session started: \`${sessionStartedAtUtc}\``);
  }
  lines.push("");

  lines.push("### Working orders");
  if (p.working_bid) {
    lines.push(
      `- BID: ${fmt(p.working_bid.size)} @ ${fmt(p.working_bid.price)} (${p.working_bid.status ?? "?"})`,
    );
  } else {
    lines.push("- BID: —");
  }
  if (p.working_ask) {
    lines.push(
      `- ASK: ${fmt(p.working_ask.size)} @ ${fmt(p.working_ask.price)} (${p.working_ask.status ?? "?"})`,
    );
  } else {
    lines.push("- ASK: —");
  }
  lines.push("");

  lines.push("### Trading venue");
  lines.push(
    `- Best bid / ask: ${fmt(p.trading_venue.best_bid)} / ${fmt(p.trading_venue.best_ask)}`,
  );
  lines.push(
    `- Bid / ask sizes: ${fmt(p.trading_venue.bid_size)} / ${fmt(p.trading_venue.ask_size)}`,
  );
  lines.push(`- Mid: ${fmt(p.trading_venue.mid)}`);
  lines.push(`- Microprice: ${fmt(p.trading_venue.microprice)}`);
  lines.push(`- Spread: ${fmtBp(p.trading_venue.spread_bps)}`);
  lines.push("");

  lines.push(`### Reference venue (${p.reference_venue.name})`);
  lines.push(`- Symbol: \`${p.reference_venue.symbol}\``);
  lines.push(
    `- Best bid / ask: ${fmt(p.reference_venue.best_bid)} / ${fmt(p.reference_venue.best_ask)}`,
  );
  lines.push(
    `- Bid / ask sizes: ${fmt(p.reference_venue.bid_size)} / ${fmt(p.reference_venue.ask_size)}`,
  );
  lines.push(`- Mid: ${fmt(p.reference_venue.mid)}`);
  lines.push(`- Microprice: ${fmt(p.reference_venue.microprice)}`);
  lines.push("");

  lines.push("### Basis & fair value");
  lines.push(`- Target − reference: ${fmtBp(p.basis.okx_minus_binance_bps)}`);
  // basis.ewma is a raw price difference. Convert to bp using OKX mid
  // (same convention as the live dashboard cards). The 6-decimal raw
  // value is also kept after, so a snapshot reader can see both.
  {
    const ewmaBps = basisEwmaToBps(p.basis.ewma, p.trading_venue.mid);
    lines.push(
      `- Basis EWMA: ${
        ewmaBps !== null ? `${ewmaBps.toFixed(2)} bp` : "—"
      } (raw ${fmt(p.basis.ewma, 6)})`,
    );
  }
  lines.push(`- Fair value: ${fmt(p.basis.fair_value)}`);
  lines.push("");

  lines.push("### Inventory (live-stats view)");
  lines.push(`- Position qty: ${fmt(p.inventory.qty)}`);
  lines.push(`- Notional: ${fmtUsd(p.inventory.notional_usd)}`);
  lines.push(`- Max notional cap: ${fmtUsd(p.inventory.max_notional_usd)}`);
  lines.push(`- Utilization: ${fmtPct(p.inventory.utilization_pct)}`);
  lines.push("");

  lines.push("### Skew");
  lines.push(`- Skew: ${fmtBp(p.skew.bps)}`);
  lines.push(`- Norm inventory: ${fmt(p.skew.norm_inventory, 3)}`);
  lines.push("");

  lines.push("### Markouts (recent fills)");
  lines.push(`- Recent fills considered: ${p.markouts.n_recent_fills}`);
  lines.push(`- Median 1s: ${fmtBp(p.markouts.median_1s_bps)}`);
  lines.push(`- Median 5s: ${fmtBp(p.markouts.median_5s_bps)}`);
  // 1.2.38: backend ``adverse_pct_{1,5}s`` is already a percentage
  // (e.g. 86.0 for 86 %). ``fmtPct`` multiplies by 100 which gave
  // "8600.0 %" in earlier downloads — render directly instead.
  lines.push(
    `- Adverse % 1s: ${
      p.markouts.adverse_pct_1s === null ||
      !Number.isFinite(p.markouts.adverse_pct_1s)
        ? "—"
        : `${p.markouts.adverse_pct_1s.toFixed(1)}%`
    }`,
  );
  lines.push(
    `- Adverse % 5s: ${
      p.markouts.adverse_pct_5s === null ||
      !Number.isFinite(p.markouts.adverse_pct_5s)
        ? "—"
        : `${p.markouts.adverse_pct_5s.toFixed(1)}%`
    }`,
  );
  lines.push("");

  lines.push("### Book signals");
  lines.push(
    `- OB imbalance EWMA: ${fmt(p.book_signals.ob_imbalance_ewma, 3)}`,
  );
  lines.push(`- Vol: ${fmtBp(p.book_signals.vol_bps)}`);
  lines.push(
    `- Toxicity score: ${fmt(p.book_signals.toxicity_score, 3)}`,
  );
  lines.push("");

  if (p.pnl_attribution && p.pnl_attribution.fill_count > 0) {
    const a = p.pnl_attribution;
    lines.push("### PnL Attribution");
    lines.push(
      `- Fills: ${a.fill_count} (maker ${a.maker_count} / taker ${a.taker_count})`,
    );
    lines.push(`- Horizon: ${a.horizon_s} s`);
    lines.push(`- Realized PnL: ${fmtUsd(a.realized_pnl_usd)}`);
    lines.push(`- Rebate income: ${fmtUsd(a.rebate_income_usd)}`);
    lines.push(
      `- Markout dollar impact: ${fmtUsd(a.markout_dollar_impact_usd)}`,
    );
    lines.push(`- Residual: ${fmtUsd(a.residual_usd)}`);
    lines.push(
      `- Rebate bps of notional: ${fmtBp(a.rebate_bps_of_notional)}`,
    );
    lines.push(`- Mean markout: ${fmtBp(a.mean_markout_bps)}`);
    lines.push(`- Median markout: ${fmtBp(a.median_markout_bps)}`);
    lines.push(
      `- Win rate: ${fmtPct(a.win_rate)} (${a.favorable_count}/${a.adverse_count + a.favorable_count})`,
    );
    lines.push("");
  }

  if (p.fill_buckets && p.fill_buckets.session_total_fills > 0) {
    const b = p.fill_buckets;
    lines.push("### Fill buckets (quote age)");
    lines.push(
      `- Window: ${b.fills_in_window} / ${b.window_size} · session total ${b.session_total_fills}` +
        (b.unknown_age_count > 0 ? ` · ${b.unknown_age_count} unknown-age` : ""),
    );
    lines.push("");
    lines.push("| Bucket | Count | B/S | Mean markout | Median markout | Notional |");
    lines.push("|---|---:|---:|---:|---:|---:|");
    for (const row of b.buckets) {
      lines.push(
        `| ${row.label} | ${row.count} | ${row.buy_count}/${row.sell_count} | ${fmtBp(row.mean_markout_bps)} | ${fmtBp(row.median_markout_bps)} | $${row.notional_usd.toFixed(2)} |`,
      );
    }
    lines.push("");
  }

  // ----- Session PnL block (above the chart on Bot Stats) ---------
  if (p.session_pnl) {
    const sp = p.session_pnl;
    lines.push("### Session PnL");
    lines.push(`- Total: ${fmtUsd(sp.total_usd)}`);
    lines.push(`- Realized: ${fmtUsd(sp.realized_usd)}`);
    lines.push(`- Unrealized: ${fmtUsd(sp.unrealized_usd)}`);
    lines.push(`- Fees: ${fmtUsd(sp.fees_usd)} (signed: positive = paid, negative = rebate)`);
    lines.push(`- Drawdown: ${fmtUsd(sp.drawdown_usd)}`);
    lines.push(`- Session peak equity: ${fmtUsd(sp.session_peak_equity_usd)}`);
    lines.push("");
  }

  // ----- Trading mode strip (header + AccountView mode-strip) -----
  // The trading-mode strip shows current eligibility + reason.
  // Also on the Engine state card's "mode" row.
  if (p.quote_breakdown) {
    const qb = p.quote_breakdown;
    lines.push("### Trading mode");
    lines.push(`- Eligibility: \`${qb.quote_eligibility ?? "—"}\``);
    if (qb.active_sides) {
      lines.push(`- Active sides: \`${qb.active_sides}\``);
    }
    if (qb.quote_eligibility_reason) {
      // Pipe-string; render as a bulleted list of reasons.
      const parts = qb.quote_eligibility_reason
        .split("|")
        .map((s: string) => s.trim())
        .filter((s: string) => s && !s.startsWith("ok") && !s.endsWith("_ok"));
      if (parts.length > 0) {
        lines.push("- Restrictive clauses:");
        for (const r of parts) lines.push(`  - ${r}`);
      } else {
        lines.push(
          `- Reason string: \`${qb.quote_eligibility_reason}\` (no restrictive clauses)`,
        );
      }
    }
    lines.push("");
  }

  // ----- Engine state (Market tab "Engine state" card) ------------
  lines.push("### Engine state");
  if (p.basis_regime) {
    const br = p.basis_regime;
    const ic = br.last_ic;
    const sign = br.last_regime_sign;
    const signLabel =
      sign === null
        ? "—"
        : sign > 0
        ? "+1 (trend)"
        : sign < 0
        ? "−1 (mean-revert)"
        : "0 (off)";
    lines.push(
      `- Basis regime: ${signLabel} · IC ${ic === null ? "—" : ic.toFixed(3)} · pairs=${br.pair_count} · warmed_up=${br.warmed_up}`,
    );
  }
  if (p.flow_score) {
    const fs = p.flow_score;
    lines.push(
      `- Flow score: TFI ${fs.tfi_signed_normalised === null ? "—" : fs.tfi_signed_normalised.toFixed(3)} · streaks ${fs.streak_buy_count}↑ / ${fs.streak_sell_count}↓ · buy_tox ${fmt(fs.buy_toxic_score, 3)} · sell_tox ${fmt(fs.sell_toxic_score, 3)}`,
    );
  }
  if (p.desync) {
    const dy = p.desync;
    lines.push(
      `- Order desync: ${dy.any ? `${dy.buy ? "BUY " : ""}${dy.sell ? "SELL " : ""}(phase=${dy.phase ?? "?"})` : "OK"}`,
    );
  }
  if (p.adaptive_widen) {
    const aw = p.adaptive_widen;
    lines.push(
      `- Adaptive widen: ${aw.active ? `ON ${aw.seconds_remaining.toFixed(0)}s${aw.reason ? ` (${aw.reason})` : ""}` : "off"}`,
    );
  }
  lines.push("");

  // ----- v1.4.116 Phase 1E.5.b — Spread composition (rewritten
  //       from the legacy "Gates" section). One row per bps
  //       contributor with bid/ask split where asymmetric;
  //       effective_total row ends the section. Safety gates moved
  //       to the new "Detectors — safety gates" section below.
  lines.push("### Spread composition (bps contributors)");
  const qb_md =
    (p as unknown as Record<string, unknown>).quote_breakdown ?? null;
  const composition_md = (qb_md as Record<string, unknown> | null)
    ?.spread_composition as Record<string, number | boolean> | undefined;
  if (composition_md === undefined) {
    lines.push(
      "- spread_composition not in payload — bot on pre-v1.4.13 build or quote_breakdown disabled.",
    );
  } else {
    const fmtBpsMd = (v: number | undefined): string => {
      if (v === undefined || !Number.isFinite(v)) return "—";
      const r = Math.abs(v) < 1e-9 ? 0 : v;
      return `${r >= 0 ? "+" : ""}${r.toFixed(2)} bp`;
    };
    const fmtBidAskMd = (
      bid: number | undefined,
      ask: number | undefined,
    ): string => {
      const b = bid === undefined || !Number.isFinite(bid) ? 0 : bid;
      const a = ask === undefined || !Number.isFinite(ask) ? 0 : ask;
      if (Math.abs(b - a) < 1e-9) return fmtBpsMd(b);
      return `bid ${fmtBpsMd(b)} · ask ${fmtBpsMd(a)}`;
    };
    lines.push(
      `- econ_floor: ${fmtBpsMd(composition_md.econ_floor_bps as number)}`,
    );
    lines.push(
      `- toxicity_bump: ${fmtBpsMd(composition_md.toxicity_bps as number)}`,
    );
    lines.push(
      `- adverse_overlay: ${fmtBpsMd(composition_md.adverse_overlay_bps as number)}`,
    );
    const gateRows: Array<[string, string, string]> = [
      ["vol_trend", "vol_trend_bid_bps", "vol_trend_ask_bps"],
      ["momentum", "momentum_bid_bps", "momentum_ask_bps"],
      ["post_swing", "post_swing_bid_bps", "post_swing_ask_bps"],
      ["microprice", "microprice_bid_bps", "microprice_ask_bps"],
      ["basis_regime", "basis_bid_bps", "basis_ask_bps"],
      ["freshness_one_sided", "freshness_bid_bps", "freshness_ask_bps"],
      [
        "recovery_cooldown",
        "recovery_cooldown_bid_bps",
        "recovery_cooldown_ask_bps",
      ],
      ["slow_trend", "slow_trend_bid_bps", "slow_trend_ask_bps"],
      [
        "inventory_drift",
        "inventory_drift_bid_bps",
        "inventory_drift_ask_bps",
      ],
    ];
    for (const [label, bidKey, askKey] of gateRows) {
      lines.push(
        `- ${label}: ${fmtBidAskMd(
          composition_md[bidKey] as number | undefined,
          composition_md[askKey] as number | undefined,
        )}`,
      );
    }
    const sumSideMd = (side: "bid" | "ask"): number => {
      const f = (k: string): number => {
        const v = composition_md[`${k}_${side}_bps`];
        return typeof v === "number" && Number.isFinite(v) ? v : 0;
      };
      let total = 0;
      for (const k of ["econ_floor", "toxicity", "adverse_overlay"]) {
        const sym = composition_md[`${k}_bps`];
        if (typeof sym === "number" && Number.isFinite(sym)) total += sym;
      }
      for (const k of [
        "vol_trend",
        "momentum",
        "post_swing",
        "microprice",
        "basis",
        "freshness",
        "recovery_cooldown",
        "slow_trend",
        "inventory_drift",
      ])
        total += f(k);
      return total;
    };
    const capBidMd = Boolean(composition_md.capped_at_max_bid);
    const capAskMd = Boolean(composition_md.capped_at_max_ask);
    lines.push(
      `- **effective total**: bid ${fmtBpsMd(sumSideMd("bid"))}${capBidMd ? " (CAPPED at MAX)" : ""} · ask ${fmtBpsMd(sumSideMd("ask"))}${capAskMd ? " (CAPPED at MAX)" : ""}`,
    );
  }
  lines.push("");

  // ----- Detectors — safety gates + regime-response gates ---------
  // Mirrors the dashboard's Detectors card layout. Safety gates
  // moved out of the legacy Gates section because the rewritten
  // Spread composition section above is bps-only.
  lines.push("### Detectors — safety gates");
  if (p.session_drawdown) {
    const sd = p.session_drawdown;
    const active = sd.tier !== "CLEAR";
    lines.push(
      `- session_drawdown: tier=${sd.tier}${active && sd.cooldown_seconds_remaining > 0 ? ` · ${sd.cooldown_seconds_remaining.toFixed(0)}s remaining` : ""}${sd.fire_count > 0 ? ` · fired ${sd.fire_count}×` : ""}${sd.last_trigger_pnl_usd !== null ? ` · last trigger pnl ${fmtUsd(sd.last_trigger_pnl_usd)}` : ""}`,
    );
  }
  if (p.shock_gate) {
    const sg = p.shock_gate;
    lines.push(
      `- shock_gate (1B): ${sg.active ? `LOCKED · ${(sg.seconds_in_lock ?? 0).toFixed(0)}s · side=${sg.locked_side ?? "—"}` : "off"}${sg.fire_count > 0 ? ` · fired ${sg.fire_count}×` : ""}${sg.active && sg.last_trigger_drift_bps ? ` · drift_${sg.last_trigger_window ?? "?"}=${fmtBp(sg.last_trigger_drift_bps)}` : ""}`,
    );
  }
  if (p.post_reduction_cooldown) {
    const prc = p.post_reduction_cooldown;
    lines.push(
      `- post_reduction_cooldown (1D): ${!prc.enabled ? "disabled" : prc.active ? `ON · ${prc.seconds_remaining.toFixed(0)}s · suppress=${prc.suppressed_side ?? "—"}` : "armed_idle"}${prc.fire_count > 0 ? ` · fired ${prc.fire_count}×` : ""}`,
    );
  }
  lines.push("");

  lines.push("### Detectors — regime-response gates");
  if (p.vol_trend_gate) {
    const vt = p.vol_trend_gate;
    lines.push(
      `- vol_trend: ${vt.active ? `ON ${vt.seconds_remaining.toFixed(0)}s` : "off"}${vt.fire_count > 0 ? ` · fired ${vt.fire_count}×` : ""}${vt.last_trigger_drift_bps !== null ? ` · last drift ${fmtBp(vt.last_trigger_drift_bps)}` : ""}`,
    );
  }
  if (p.post_swing_gate) {
    const ps = p.post_swing_gate;
    lines.push(
      `- post_swing: ${ps.active ? `ON ${ps.seconds_remaining.toFixed(0)}s` : "off"}${ps.fire_count > 0 ? ` · fired ${ps.fire_count}×` : ""}${ps.last_trigger_reason ? ` · last: ${ps.last_trigger_reason}` : ""}`,
    );
  }
  if (p.quote_eligibility_recovery) {
    const er = p.quote_eligibility_recovery;
    lines.push(
      `- recovery_cooldown: ${er.active ? `ON ${er.seconds_remaining.toFixed(0)}s${er.floor ? ` (floor=${er.floor})` : ""}` : "off"}`,
    );
  }
  if (p.adaptive_widen) {
    const aw = p.adaptive_widen;
    lines.push(
      `- adaptive_widen: ${aw.active ? `ON ${aw.seconds_remaining.toFixed(0)}s${aw.reason ? ` (${aw.reason})` : ""}` : "off"}`,
    );
  }
  // Derived gates — basis_IC + microprice
  {
    const br = p.basis_regime;
    const ic = br?.last_ic ?? null;
    const basisIcActive =
      br?.warmed_up === true && ic !== null && Math.abs(ic) < 0.05;
    lines.push(
      `- basis_IC: ${basisIcActive ? `ON · |IC|=${ic !== null ? Math.abs(ic).toFixed(3) : "—"} < 0.05` : "off"}`,
    );
  }
  {
    const obi = p.book_signals.ob_imbalance_ewma;
    const mpActive =
      obi !== null && Number.isFinite(obi) && Math.abs(obi) >= 0.5;
    lines.push(
      `- microprice: ${mpActive ? `ON · imb=${obi !== null ? obi.toFixed(3) : "—"} (|imb| ≥ 0.5)` : "off"}`,
    );
  }
  lines.push("");

  // ----- v1.4.116 Phase 1E.3.f — quoting-features firing rates ----
  if (p.feature_firing_rates) {
    const ffr = p.feature_firing_rates;
    const w = ffr.window_seconds;
    lines.push(`### Detectors — quoting features (${w}s rate)`);
    const knownKeys: Array<[string, string]> = [
      ["post_fill_cooldown_bid", "post_fill_cd (bid)"],
      ["post_fill_cooldown_ask", "post_fill_cd (ask)"],
      ["at_touch_adverse_pause_bid", "at_touch_pause (bid)"],
      ["at_touch_adverse_pause_ask", "at_touch_pause (ask)"],
    ];
    for (const [key, label] of knownKeys) {
      const count = Number(ffr.rates[key] ?? 0);
      lines.push(`- ${label}: ${count} / ${w}s`);
    }
    lines.push("");
  }

  // ----- v1.4.115 Phase 1E.5.c — Regime mode (Phase 1C FSM) -------
  // Carries the operator-facing mode + dwell + transition reason +
  // session-cumulative time-in-mode counters from the
  // regime_controller. Falls through with a "—" placeholder when
  // the bot is on a pre-Phase-1C build.
  lines.push("### Regime mode (Phase 1C FSM)");
  if (p.regime_mode) {
    const rm = p.regime_mode;
    const fmtDwell = (s: number): string => {
      if (!Number.isFinite(s)) return "—";
      if (s < 60) return `${Math.floor(s)}s`;
      if (s < 3600) return `${Math.floor(s / 60)}m`;
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `${h}h${String(m).padStart(2, "0")}m`;
    };
    lines.push(
      `- Mode: ${rm.mode} · in mode ${fmtDwell(rm.seconds_in_mode)}${rm.last_transition_reason ? ` · last transition: ${rm.last_transition_reason}` : ""}`,
    );
    lines.push(
      `- Time in mode (session): NORMAL ${fmtDwell(rm.time_in_normal_seconds)} · DEFENSIVE ${fmtDwell(rm.time_in_defensive_seconds)} · SHOCK ${fmtDwell(rm.time_in_shock_seconds)} · transitions ${rm.transition_count}`,
    );
    if (rm.recent_transitions.length > 0) {
      // v1.4.116 Phase 1E.3 — render the FULL session log per
      // operator request, not just last 3. The bot caps the deque
      // at 1024 entries as a memory safety net; in practice a
      // session has << 100 transitions thanks to the 15 s entry /
      // 30 s exit hysteresis.
      lines.push(
        `- Mode transitions (${rm.recent_transitions.length} this session, newest last):`,
      );
      for (const t of rm.recent_transitions) {
        lines.push(`  - ${t.from} → ${t.to}: ${t.reason}`);
      }
    }
  } else {
    lines.push("- Mode: — (regime_controller not present — pre-v1.4.112 build)");
  }
  lines.push("");

  // ----- Alerts (right rail Alerts card) --------------------------
  // The Alerts card shows live aggregate alerts derived from the
  // same live_stats fields above; include the formatted readout so
  // the operator's report carries the same "at-a-glance" view they
  // see in the dashboard.
  lines.push("### Alerts (right-rail card)");
  // v1.4.115 Phase 1E — the legacy "Active gates (N): …" mirror is
  // gone; the AlertsCard now ends with the Regime-mode chip (the
  // line below) and the full gate enumeration lives in the Gates
  // section earlier in this report. The mode chip captures the
  // FSM aggregate the operator reads at a glance.
  lines.push(`- Mean MO 5s: ${fmtBp(p.pnl_attribution?.mean_markout_bps ?? null)}`);
  lines.push(`- Median MO 5s: ${fmtBp(p.markouts.median_5s_bps)}`);
  lines.push(
    `- Adverse % 5s: ${p.markouts.adverse_pct_5s === null ? "—" : `${p.markouts.adverse_pct_5s.toFixed(0)}%`}`,
  );
  lines.push(
    `- Eligibility: \`${p.quote_breakdown?.quote_eligibility ?? "—"}\``,
  );
  // v1.4.115 Phase 1E.5.c — Regime mode chip replaces the legacy
  // "Active gates" line in the Alerts mirror (matches the new
  // AlertsCard layout). Full gate enumeration lives in the Gates
  // section above; this line is the FSM aggregate.
  if (p.regime_mode) {
    const rm = p.regime_mode;
    const fmtDwellMd = (s: number): string => {
      if (!Number.isFinite(s)) return "—";
      if (s < 60) return `${Math.floor(s)}s`;
      if (s < 3600) return `${Math.floor(s / 60)}m`;
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `${h}h${String(m).padStart(2, "0")}m`;
    };
    lines.push(`- Regime mode: ${rm.mode} · ${fmtDwellMd(rm.seconds_in_mode)}`);
  } else {
    lines.push(`- Regime mode: — (pre-v1.4.112 build)`);
  }
  lines.push(`- Basis vs reference: ${fmtBp(p.basis.okx_minus_binance_bps)}`);
  lines.push(`- Toxicity: ${fmt(p.book_signals.toxicity_score, 3)}`);
  lines.push(
    `- Inventory util: ${p.inventory.utilization_pct === null ? "—" : `${p.inventory.utilization_pct.toFixed(1)}%`}`,
  );
  lines.push("");

  // ----- Quote breakdown (Spread tab data) ------------------------
  // The Spread tab shows the per-cycle quote decomposition. Include
  // the headline numbers (reservation, target half-spread, size mult)
  // so the report captures what the bot decided this cycle.
  if (p.quote_breakdown) {
    const qb = p.quote_breakdown;
    lines.push("### Quote breakdown (per-cycle, Spread tab)");
    lines.push(`- Cycle ts: \`${qb.ts}\``);
    lines.push(`- Mid: ${fmt(qb.mid_price, 6)}`);
    lines.push(`- Reservation: ${fmt(qb.reservation_price, 6)} (${fmtBp(qb.reservation_delta_from_mid_bps)} from mid)`);
    lines.push(`- Target half-spread: ${fmtBp(qb.target_half_spread_bps)}`);
    lines.push(`- BASE half-spread: ${fmtBp(qb.base_half_spread_bps)}`);
    lines.push(`- Raw half-spread (before clamp): ${fmtBp(qb.raw_half_spread_bps)}`);
    lines.push(`- Effective notional / side: $${qb.effective_notional_usd.toFixed(2)} (× ${qb.final_size_mult.toFixed(3)})`);
    lines.push("");
  }

  // ----- Soft-flatten attribution (Bot Stats subsection) ----------
  // Pre-2026-05-16: only the latest event was emitted. The LLM
  // cross-analysis needs ALL events to spot patterns (clustered
  // SF triggers = something is repeatedly stalling the bot). Cap
  // at 50 most-recent in case the session is huge.
  if (p.soft_flatten_attribution && p.soft_flatten_attribution.events.length > 0) {
    const sfa = p.soft_flatten_attribution;
    lines.push("### Soft-flatten attribution");
    lines.push(`- Events this session: ${sfa.events.length}`);
    lines.push("");
    const SF_CAP = 50;
    const evs = sfa.events.slice(0, SF_CAP);
    lines.push(
      "| # | Start (UTC) | End (UTC) | Trigger | Exit reason | Force phase | Entry qty | Entry mid | Exit phase | Attributed orders | Attributed fills | Attributed notional |",
    );
    lines.push(
      "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    );
    for (const ev of evs) {
      lines.push(
        `| ${ev.id} | \`${ev.ts_start}\` | \`${ev.ts_end ?? "(open)"}\` | ${ev.trigger_reason ?? "—"} | ${ev.exit_reason ?? "—"} | ${ev.initial_force_phase ?? "—"} | ${ev.entry_position_qty === null ? "—" : fmt(ev.entry_position_qty, 4)} | ${fmt(ev.entry_mid_price, 6)} | ${ev.exit_phase_reached ?? "—"} | ${ev.attributed_orders_count} | ${ev.attributed_fills_count} | ${fmtUsd2(ev.attributed_fills_notional_usd)} |`,
      );
    }
    lines.push("");
    if (sfa.events.length > SF_CAP) {
      lines.push(
        `_… plus ${sfa.events.length - SF_CAP} older events not shown._`,
      );
      lines.push("");
    }
  }

  // ----- Equity history summary (Session PnL chart data) ----------
  // The chart shows the full samples array; for the report we
  // include the headline numbers (count, first/last, peak, trough)
  // rather than the raw points (would be hundreds of KB).
  if (equityHistory && equityHistory.samples.length > 0) {
    const samples = equityHistory.samples;
    const first = samples[0];
    const last = samples[samples.length - 1];
    const equities = samples
      .map((s: { equity_usd: number | null }) => s.equity_usd)
      .filter((v: number | null): v is number => v !== null && Number.isFinite(v));
    const peak = equities.length > 0 ? Math.max(...equities) : null;
    const trough = equities.length > 0 ? Math.min(...equities) : null;
    // Identify the peak-equity sample index for "time since peak" calc.
    let peakIdx = -1;
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].equity_usd === peak) {
        peakIdx = i;
        break;
      }
    }
    const peakTs = peakIdx >= 0 ? samples[peakIdx].ts : null;
    const minutesSincePeak =
      peakTs && last.ts
        ? Math.max(
            0,
            (new Date(last.ts).getTime() - new Date(peakTs).getTime()) /
              60000,
          )
        : null;
    lines.push("## Session PnL chart (equity history)");
    lines.push(`- Samples: ${samples.length}`);
    lines.push(`- First sample: \`${first.ts ?? "?"}\` · equity ${fmtUsd2(first.equity_usd)}`);
    lines.push(`- Last sample: \`${last.ts ?? "?"}\` · equity ${fmtUsd2(last.equity_usd)}`);
    lines.push(`- Session peak equity: ${fmtUsd2(peak)}${peakTs ? ` (at \`${peakTs}\`${minutesSincePeak !== null ? `, ${minutesSincePeak.toFixed(1)} min ago` : ""})` : ""}`);
    lines.push(`- Session trough equity: ${fmtUsd2(trough)}`);
    if (peak !== null && last.equity_usd !== null && Number.isFinite(last.equity_usd)) {
      const currentDdFromPeak = peak - last.equity_usd;
      lines.push(
        `- Current drawdown from peak: ${fmtUsd2(currentDdFromPeak)} (${peak > 0 ? `${((currentDdFromPeak / peak) * 100).toFixed(2)}%` : "—"} of peak equity)`,
      );
    }
    lines.push(`- Max drawdown cap: ${fmtUsd2(equityHistory.max_drawdown_usd_cap)}`);
    if (equityHistory.killed) {
      lines.push(
        `- KILL: \`${equityHistory.kill_reason ?? "(unknown)"}\` at \`${equityHistory.kill_timestamp_utc ?? "?"}\``,
      );
    }
    lines.push("");
    // Equity / position / vol trajectory tail. Last 30 samples
    // chosen as "enough to see the shape, small enough to keep the
    // markdown a reasonable size at the session-PnL publisher's
    // ~60s cadence" → ~30 min window. The LLM uses this to read
    // trend direction, drawdown shape, and vol/position
    // co-movement that the headline numbers can't capture.
    const TAIL_N = 30;
    const tail = samples.slice(-TAIL_N);
    lines.push(
      `### Equity / position / vol trajectory (newest ${tail.length} samples)`,
    );
    lines.push("");
    lines.push(
      "| Time (UTC) | Equity $ | Realized $ | Unrealized $ | Fees $ | Drawdown $ | Position qty | Vol bp | Basis EWMA |",
    );
    lines.push(
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    );
    for (const s of tail) {
      const basisBp =
        s.binance_basis_ewma !== null &&
        s.binance_basis_ewma !== undefined &&
        s.mid_price !== null &&
        s.mid_price !== undefined &&
        s.mid_price > 0
          ? (s.binance_basis_ewma / s.mid_price) * 10_000
          : null;
      lines.push(
        `| \`${s.ts ?? "?"}\` | ${fmtUsd2(s.equity_usd)} | ${fmtUsd(s.realized_pnl_usd)} | ${fmtUsd(s.unrealized_pnl_usd)} | ${fmtUsd(s.fees_usd)} | ${fmtUsd2(s.drawdown_usd)} | ${s.position_qty === null || s.position_qty === undefined ? "—" : fmt(s.position_qty, 4)} | ${fmtBp(s.vol_bps ?? null)} | ${fmtBp(basisBp)} |`,
      );
    }
    lines.push("");
    if (samples.length > TAIL_N) {
      lines.push(
        `_… plus ${samples.length - TAIL_N} earlier samples not shown (typically 60 s cadence; full series visible on the Session tab)._`,
      );
      lines.push("");
    }
    // Soft-flatten events from the equity-history payload — full
    // table (the live-stats SF section above only shows the latest).
    if (equityHistory.sf_events && equityHistory.sf_events.length > 0) {
      lines.push(
        `### Soft-flatten events (${equityHistory.sf_events.length} this session)`,
      );
      lines.push("");
      lines.push(
        "| # | First fill (UTC) | Last fill (UTC) | Side | Fills | Notional |",
      );
      lines.push("|---:|---|---|---|---:|---:|");
      for (const ev of equityHistory.sf_events) {
        lines.push(
          `| ${ev.event_id} | \`${ev.ts_first_fill ?? "?"}\` | \`${ev.ts_last_fill ?? "?"}\` | ${ev.side ?? "—"} | ${ev.fill_count} | ${fmtUsd2(ev.notional_usd)} |`,
        );
      }
      lines.push("");
    }
  }

  // ----- Bot Stats — Aggressiveness bar chart (second sub-table) --
  // The Bot Stats panel's Fill-Buckets card renders TWO bar charts:
  // 1) by quote age (already emitted above as "Fill buckets (quote
  //    age)"), 2) by place-time aggressiveness. Add the second one
  //    so the export covers all bar charts in plain text.
  if (
    p &&
    p.fill_buckets &&
    p.fill_buckets.aggressiveness &&
    p.fill_buckets.aggressiveness.length > 0
  ) {
    const aggro = p.fill_buckets.aggressiveness;
    lines.push("### Fill buckets (place-time aggressiveness)");
    lines.push("");
    lines.push(
      "| Bucket | Count | B/S | Mean markout | Median markout | Notional |",
    );
    lines.push("|---|---:|---:|---:|---:|---:|");
    for (const row of aggro) {
      lines.push(
        `| ${row.label} | ${row.count} | ${row.buy_count}/${row.sell_count} | ${fmtBp(row.mean_markout_bps)} | ${fmtBp(row.median_markout_bps)} | $${row.notional_usd.toFixed(2)} |`,
      );
    }
    lines.push("");
  }

  // ----- Bot Stats — Execution Quality / Leakage summary ----------
  // The ExecutionQualityCard renders 5 leakage cards (stale, blocked,
  // wanted-out, missing-ref, wrong-side). Latency / distance
  // histograms don't tabulate well, but the 5 leakage counts +
  // mean-markout + net-$ tabulate cleanly. Same filter predicates the
  // on-screen card uses (lines ~13101-13135).
  if (regimeFills?.payload?.rows && regimeFills.payload.rows.length > 0) {
    const allFills = regimeFills.payload.rows;
    const staleFills = allFills.filter(
      (f) =>
        exIsFiniteNum(f.quote_age_at_fill_ms) &&
        f.quote_age_at_fill_ms > 2000,
    );
    const blockedStateFills = allFills.filter((f) => {
      const elig = String(f.quote_eligibility_state || "").toUpperCase();
      return (
        elig === "HOLD_ALL" ||
        f.adaptive_widen_active_at_decision === 1 ||
        f.post_fill_cooldown_active_bid_at_decision === 1 ||
        f.post_fill_cooldown_active_ask_at_decision === 1 ||
        f.at_touch_adverse_pause_bid_at_decision === 1 ||
        f.at_touch_adverse_pause_ask_at_decision === 1
      );
    });
    const wantedOutFills = allFills.filter((f) => {
      const fdc = (f as Record<string, unknown>)[
        "fill_during_quote_cooldown"
      ];
      return f.cancel_requested_before_fill === 1 || fdc === 1;
    });
    const degradedRefFills = allFills.filter(
      (f) =>
        f.book_reference_quality !== null &&
        f.book_reference_quality !== "exact_or_prior",
    );
    const wrongSideFills = allFills.filter(fillWrongSideOfEligibility);

    const total = allFills.length;
    const pctOf = (n: number, d: number): string =>
      d > 0 ? `${((n / d) * 100).toFixed(1)}%` : "—";
    const aggRow = (
      label: string,
      subset: FillSinceRow[],
    ): string => {
      const a = aggregateLeakage(subset);
      return `| ${label} | ${a.count} | ${total} | ${pctOf(a.count, total)} | ${fmtBp(a.meanMarkoutBps)} | ${fmtUsd(a.netUsd)} |`;
    };
    lines.push("### Execution quality — leakage summary (Bot Stats)");
    lines.push("");
    lines.push(
      "| Category | Count | Total fills | % | Mean MO 5s | Net $ |",
    );
    lines.push("|---|---:|---:|---:|---:|---:|");
    lines.push(aggRow("Stale fills (>2s quote age)", staleFills));
    lines.push(aggRow("Fills during defensive gates", blockedStateFills));
    lines.push(
      aggRow("Fills after wanted-out (cancel race)", wantedOutFills),
    );
    lines.push(aggRow("Missing/degraded reference", degradedRefFills));
    lines.push(aggRow("Wrong-side of eligibility", wrongSideFills));
    lines.push("");
  }

  // ----- History — Connectivity sub-tab ---------------------------
  // Replicates the four tables the ConnectivityPanel renders.
  // ``computeConnectivityAggregates`` is the SAME function the panel
  // uses (lifted to module scope on 2026-05-16), so the export is
  // guaranteed-consistent with what's on screen.
  if (
    ordersLifecycle?.payload?.rows &&
    ordersLifecycle.payload.rows.length > 0
  ) {
    const conn = computeConnectivityAggregates(
      ordersLifecycle.payload.rows,
      // 1.4.6: sticky reject summary + outcome aggregates survive
      // the row-buffer eviction. Same source as the live
      // ConnectivityPanel uses. v1.5.135: the explicit-unknown
      // hop is what TypeScript wants for the
      // DashboardStatePayload→Record cast since the two types
      // don't structurally overlap.
      (ordersLifecycle.payload as unknown as Record<string, unknown>)
        .sticky_reject_summary as StickyRejectSummary | undefined,
      (ordersLifecycle.payload as unknown as Record<string, unknown>)
        .outcome_aggregates as OutcomeAggregates | undefined,
    );
    lines.push("## Connectivity (History → Connectivity sub-tab)");
    lines.push("");
    lines.push(
      `- Total orders: ${conn.footer.totalOrders} · ` +
        `acked: ${conn.footer.totalAcked} · ` +
        `cancels: ${conn.footer.totalCancels} · ` +
        `place responses: ${conn.footer.totalPlaceResponses}`,
    );
    const oa = conn.footer.orderAge;
    lines.push(
      `- Order age at market (ack → close, ms): ` +
        `min ${fmt(oa.min, 1)} · ` +
        `median ${fmt(oa.median, 1)} · ` +
        `p95 ${fmt(oa.p95, 1)} · ` +
        `max ${fmt(oa.max, 1)} · ` +
        `n=${conn.footer.totalAcked}`,
    );
    lines.push("");
    lines.push("### Cancels by reason");
    if (conn.cancelRows.length === 0) {
      lines.push("");
      lines.push("_no cancels in window_");
      lines.push("");
    } else {
      lines.push("");
      lines.push(
        "| Reason | Count | % | Latency min | Median | p95 | Max | First seen | Last seen |",
      );
      lines.push("|---|---:|---:|---:|---:|---:|---:|---|---|");
      for (const r of conn.cancelRows) {
        lines.push(
          `| ${r.reason} | ${r.count} | ${r.pct.toFixed(1)}% | ` +
            `${fmt(r.latency.min, 1)} | ${fmt(r.latency.median, 1)} | ` +
            `${fmt(r.latency.p95, 1)} | ${fmt(r.latency.max, 1)} | ` +
            `\`${r.first_seen_ts ?? "—"}\` | \`${r.last_seen_ts ?? "—"}\` |`,
        );
      }
      lines.push("");
    }
    lines.push("### Cancel rejects by venue detail");
    if (conn.cancelRejects.length === 0) {
      lines.push("");
      lines.push("_no cancel rejects in window_");
      lines.push("");
    } else {
      lines.push("");
      lines.push("| Detail | Count | % | Category |");
      lines.push("|---|---:|---:|---|");
      for (const r of conn.cancelRejects) {
        lines.push(
          `| ${r.detail} | ${r.count} | ${r.pct.toFixed(1)}% | ${r.benign ? "benign" : "attention"} |`,
        );
      }
      lines.push("");
    }
    lines.push("### New-order outcomes");
    if (conn.placeRows.length === 0) {
      lines.push("");
      lines.push("_no place responses in window_");
      lines.push("");
    } else {
      lines.push("");
      lines.push(
        "| Outcome | Count | % | RTT min | Median | p95 | Max |",
      );
      lines.push("|---|---:|---:|---:|---:|---:|---:|");
      for (const r of conn.placeRows) {
        lines.push(
          `| ${r.outcome} | ${r.count} | ${r.pct.toFixed(1)}% | ` +
            `${fmt(r.latency.min, 1)} | ${fmt(r.latency.median, 1)} | ` +
            `${fmt(r.latency.p95, 1)} | ${fmt(r.latency.max, 1)} |`,
        );
      }
      lines.push("");
    }
    lines.push("### Place rejects by venue detail");
    if (conn.placeRejects.length === 0) {
      lines.push("");
      lines.push("_no place rejects in window_");
      lines.push("");
    } else {
      lines.push("");
      lines.push("| Detail | Count | % | Category |");
      lines.push("|---|---:|---:|---|");
      for (const r of conn.placeRejects) {
        lines.push(
          `| ${r.detail} | ${r.count} | ${r.pct.toFixed(1)}% | ${r.benign ? "benign" : "attention"} |`,
        );
      }
      lines.push("");
    }
  }

  // ----- History — Fill Drill-Down sub-tab ------------------------
  // The drilldown panel shows per-fill detail. The "Recent fills
  // (last 10)" section above gives the venue-fills view; this
  // section gives the publisher-fills view with the bot-side
  // decision/conditions/outcome columns. Cap at 25 newest to keep
  // the markdown a reasonable size.
  if (regimeFills?.payload?.rows && regimeFills.payload.rows.length > 0) {
    const sorted = [...regimeFills.payload.rows].sort((a, b) =>
      String(b.ts_fill).localeCompare(String(a.ts_fill)),
    );
    const top = sorted.slice(0, 25);
    lines.push(
      "## Fill Drill-Down (History → Fill Drill-Down sub-tab, newest 25)",
    );
    lines.push("");
    lines.push(
      "| Time | Side | Price | Qty | Notional | MO 5s | Quote age (ms) | Eligibility | Active sides | Inv util | Basis EWMA | Net $ |",
    );
    lines.push(
      "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
    );
    for (const f of top) {
      const fee = Number.isFinite(f.fee) ? f.fee ?? 0 : 0;
      const notional = Number.isFinite(f.notional) ? f.notional ?? 0 : 0;
      const mo5 = f.markout_5s_bps;
      const closed = f.closed_pnl ?? 0;
      const netUsd =
        mo5 !== null && Number.isFinite(mo5)
          ? -fee + (notional * mo5) / 1e4 + closed
          : null;
      lines.push(
        `| \`${f.ts_fill}\` | ${f.side} | ${fmt(f.price, 6)} | ` +
          `${fmt(f.qty, 4)} | ${fmtUsd2(notional)} | ` +
          `${fmtBp(mo5)} | ${fmt(f.quote_age_at_fill_ms, 0)} | ` +
          `\`${f.quote_eligibility_state ?? "—"}\` | ` +
          `\`${f.active_sides_at_decision ?? "—"}\` | ` +
          `${f.inventory_utilization_before_fill === null ? "—" : `${(f.inventory_utilization_before_fill * 100).toFixed(1)}%`} | ` +
          `${fmtBp(f.binance_basis_ewma_at_decision)} | ` +
          `${netUsd === null ? "—" : fmtUsd(netUsd)} |`,
      );
    }
    lines.push("");
    if (sorted.length > 25) {
      lines.push(
        `_… plus ${sorted.length - 25} older fills not shown (open the Fill Drill-Down tab for the full list)_`,
      );
      lines.push("");
    }
  }

  // ----- Market tab — Gate Effectiveness Table --------------------
  // 12 rows (one per gate in GATE_DEFS). ``computeGateRow`` is the
  // SAME helper the GateEffectivenessTable uses on screen.
  if (
    regimeBars?.payload?.rows &&
    regimeFills?.payload?.rows &&
    (regimeBars.payload.rows.length > 0 ||
      regimeFills.payload.rows.length > 0)
  ) {
    const bRows = regimeBars.payload.rows;
    const fRows = regimeFills.payload.rows;
    lines.push("## Gate effectiveness (Market tab)");
    lines.push("");
    lines.push(
      "| Gate | Time active (s) | % of session | Fires | Fills during | Confidence | Mean MO 5s | Median MO 5s | Net $ |",
    );
    lines.push(
      "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    );
    const totalBarSec = bRows.length * GATE_BAR_CADENCE_S;
    for (const def of GATE_DEFS) {
      const gr = computeGateRow(def, bRows, fRows);
      const pctSession =
        totalBarSec > 0
          ? `${((gr.time_active_s / totalBarSec) * 100).toFixed(1)}%`
          : "—";
      lines.push(
        `| ${gr.name} | ${gr.time_active_s.toFixed(1)} | ${pctSession} | ` +
          `${gr.fire_count} | ${gr.fills_during ?? "—"} | ` +
          `${gr.confidence ?? "—"} | ` +
          `${fmtBp(gr.mean_markout_5s_bps)} | ` +
          `${fmtBp(gr.median_markout_5s_bps)} | ` +
          `${fmtUsd(gr.net_dollars)} |`,
      );
    }
    lines.push("");
  }

  // ----- Market tab — Cross-venue regime (basis dwell) ------------
  if (regimeBars?.payload?.rows && regimeBars.payload.rows.length > 0) {
    const bRows = regimeBars.payload.rows;
    const dwell = computeBasisDwell(bRows);
    if (dwell.length > 0) {
      lines.push("## Basis regime dwell (Market tab — cross-venue)");
      lines.push("");
      lines.push("| Bucket | Minutes | % of session | n bars |");
      lines.push("|---|---:|---:|---:|");
      for (const r of dwell) {
        lines.push(
          `| ${r.bucket} | ${r.minutes.toFixed(1)} | ${r.pct_of_session.toFixed(1)}% | ${r.n_bars} |`,
        );
      }
      lines.push("");
    }
  }

  // ----- Inventory tab — 4 tables ---------------------------------
  // 1) Inventory dwell (4 buckets: FLAT/LOW/MED/HIGH)
  // 2) One-sided dwell (4 buckets: BOTH/BID_ONLY/ASK_ONLY/HOLD_ALL)
  // 3) Quote-eligibility transitions
  // 4) Fills by active sides at decision
  // (Per-side × inventory-bucket markout breakdown is already in
  //  the Regime "inventory" axis table — no need to duplicate.)
  if (
    (regimeBars?.payload?.rows && regimeBars.payload.rows.length > 0) ||
    (regimeFills?.payload?.rows && regimeFills.payload.rows.length > 0)
  ) {
    const bRows = regimeBars?.payload?.rows ?? [];
    const fRows = regimeFills?.payload?.rows ?? [];
    lines.push("## Inventory behaviour (Inventory tab)");
    lines.push("");
    if (bRows.length > 0) {
      const dwell = computeInventoryDwell(bRows);
      const oneSided = computeOneSidedDwell(bRows);
      const transitions = computeTransitions(bRows);
      lines.push("### Inventory utilization dwell");
      lines.push("");
      lines.push("| Bucket | Minutes | % of session | n bars |");
      lines.push("|---|---:|---:|---:|");
      for (const r of dwell) {
        lines.push(
          `| ${r.bucket} | ${r.minutes.toFixed(1)} | ${r.pct_of_session.toFixed(1)}% | ${r.n_bars} |`,
        );
      }
      lines.push("");
      lines.push("### One-sided / both-sides dwell");
      lines.push("");
      lines.push("| Bucket | Minutes | % of session | n bars |");
      lines.push("|---|---:|---:|---:|");
      for (const r of oneSided) {
        lines.push(
          `| ${r.bucket} | ${r.minutes.toFixed(1)} | ${r.pct_of_session.toFixed(1)}% | ${r.n_bars} |`,
        );
      }
      lines.push("");
      if (transitions.length > 0) {
        lines.push("### Quote-eligibility transitions");
        lines.push("");
        lines.push("| From → To | Count | Avg duration before (s) |");
        lines.push("|---|---:|---:|");
        for (const r of transitions) {
          lines.push(
            `| ${r.from} → ${r.to} | ${r.count} | ${r.avg_duration_before_s === null ? "—" : r.avg_duration_before_s.toFixed(1)} |`,
          );
        }
        lines.push("");
      }
    }
    if (fRows.length > 0) {
      const activeSides = computeFillsByActiveSides(fRows);
      if (activeSides.length > 0) {
        lines.push("### Fills by active sides at decision");
        lines.push("");
        lines.push(
          "| Active sides | n | Confidence | Mean MO 5s | Net $ |",
        );
        lines.push("|---|---:|---|---:|---:|");
        for (const r of activeSides) {
          lines.push(
            `| ${r.active_sides} | ${r.n} | ${r.confidence ?? "—"} | ${fmtBp(r.mean_markout_5s_bps)} | ${fmtUsd(r.net_usd)} |`,
          );
        }
        lines.push("");
      }
    }
  }

  // ----- Config (Config tab — full dump) --------------------------
  // 2026-05-16 v1.3.90: dump the entire env file + every resolved
  // setting. Operator wants the export to mirror the Config tab,
  // and the Config tab IS the full env. Truncated only if the raw
  // env exceeds 64 KB (unlikely — profiles are typically 8-20 KB).
  if (configResult && configResult.payload) {
    const c = configResult.payload;
    lines.push("## Config (Config tab — full)");
    lines.push(`- Profile: \`${c.profile}\``);
    lines.push(`- Symbol: \`${c.symbol ?? "—"}\``);
    lines.push(`- Bot version: \`${c.version ?? "—"}\``);
    lines.push(`- Captured: \`${c.captured_at_utc}\``);
    lines.push(`- Session id: \`${c.session_id ?? "—"}\``);
    lines.push(`- Session started: \`${c.session_started_at_utc}\``);
    lines.push(`- Env file path: \`${c.env_file_path}\``);
    if (c.env_file_read_error) {
      lines.push(`- ⚠ env file read error: ${c.env_file_read_error}`);
    }
    lines.push("");
    // Raw env file body (when present). Hard cap at 64 KB to keep
    // the markdown manageable; truncation footer makes it obvious.
    if (c.env_file_raw) {
      const RAW_CAP = 64 * 1024;
      const raw =
        c.env_file_raw.length > RAW_CAP
          ? c.env_file_raw.slice(0, RAW_CAP) +
            `\n# … truncated, ${c.env_file_raw.length - RAW_CAP} more bytes omitted`
          : c.env_file_raw;
      lines.push("### Raw env file");
      lines.push("");
      lines.push("```env");
      lines.push(raw);
      lines.push("```");
      lines.push("");
    }
    // Resolved settings table — sorted alphabetically by key, same
    // order the ConfigPanel's ResolvedSettingsList fallback uses.
    // Booleans/numbers print as-is; strings passthrough; objects
    // get JSON-stringified to one line.
    const keys = Object.keys(c.resolved_settings).sort();
    if (keys.length > 0) {
      lines.push(`### Resolved settings (${keys.length} keys)`);
      lines.push("");
      lines.push("| Key | Value |");
      lines.push("|---|---|");
      for (const k of keys) {
        const v = c.resolved_settings[k];
        let formatted: string;
        if (v === null || v === undefined) {
          formatted = "—";
        } else if (typeof v === "boolean" || typeof v === "number") {
          formatted = String(v);
        } else if (typeof v === "string") {
          // Escape table-breaking characters.
          formatted = v.replace(/\|/g, "\\|").replace(/\n/g, " ");
          if (formatted.length > 200) {
            formatted = formatted.slice(0, 200) + "…";
          }
        } else {
          try {
            formatted = JSON.stringify(v);
            if (formatted.length > 200) {
              formatted = formatted.slice(0, 200) + "…";
            }
          } catch {
            formatted = String(v);
          }
        }
        lines.push(`| \`${k}\` | ${formatted} |`);
      }
      lines.push("");
    }
  }

  // ----- Regime tables (Regime tab) -------------------------------
  // Reproduce the per-axis (side × bucket × metrics) tables shown
  // in the Regime tab. One markdown subsection per axis, each
  // rendered as a sortable-by-column table that matches the
  // on-screen layout. We aggregate inside the builder so the
  // export is point-in-time correct: the dashboard's memo could
  // be using slightly older fills/bars, but the operator pressing
  // Download wants the freshest read.
  //
  // Schema parity with REGIME_COLS:
  //   Side · Bucket · n · Confidence · Notional · Minutes ·
  //   Fills/h · Mean MO 5s · Median MO 5s · Adverse % · Rebate$ ·
  //   Close PnL$ · Net edge bp
  //
  // Empty axes (no fills landed in any bucket) are still emitted
  // with a single "no fills yet" line so the operator can see the
  // axis exists rather than wonder if it's been removed.
  const fillsRowsForRegime = regimeFills?.payload?.rows ?? [];
  const barsRowsForRegime = regimeBars?.payload?.rows ?? [];
  if (fillsRowsForRegime.length > 0 || barsRowsForRegime.length > 0) {
    const regime = aggregateRegimes(
      fillsRowsForRegime,
      barsRowsForRegime,
    );
    lines.push("## Regime tables (Regime tab)");
    lines.push("");
    const t = regime.totals;
    lines.push(
      `- Totals: \`${t.n_fills}\` fills · ` +
        `notional \`$${t.notional_usd.toFixed(2)}\` · ` +
        `rebate \`$${t.rebate_usd.toFixed(3)}\` · ` +
        `close PnL \`$${t.closed_pnl_usd.toFixed(3)}\` · ` +
        `exposure \`${t.minutes_exposed.toFixed(1)} min\` ` +
        `(${t.n_bars} bars · ${t.bar_cadence_seconds.toFixed(1)} s cadence)`,
    );
    lines.push(
      `- Slices by confidence: ` +
        `\`trustworthy ${t.slices_trustworthy}\` · ` +
        `\`directional ${t.slices_directional}\` · ` +
        `\`hypothesis ${t.slices_hypothesis}\` · ` +
        `\`anecdote ${t.slices_anecdote}\``,
    );
    lines.push("");
    // Tiny formatters local to the regime section to mirror the
    // on-screen rgmFmt* helpers exactly.
    const rgFmtNum = (
      v: number | null,
      digits: number,
    ): string =>
      v === null || !Number.isFinite(v) ? "n/a" : v.toFixed(digits);
    const rgFmtUsd = (
      v: number | null,
      digits: number,
    ): string =>
      v === null || !Number.isFinite(v)
        ? "n/a"
        : `$${v.toFixed(digits)}`;
    const rgFmtBp = (
      v: number | null,
      digits: number,
    ): string =>
      v === null || !Number.isFinite(v)
        ? "n/a"
        : `${v >= 0 ? "+" : ""}${v.toFixed(digits)} bp`;
    const rgFmtPct = (
      v: number | null,
      digits: number,
    ): string =>
      v === null || !Number.isFinite(v)
        ? "n/a"
        : `${v.toFixed(digits)}%`;
    for (const axis of regime.axes) {
      const byAxis = regime.by_axis[axis] ?? {};
      // Materialise rows: (side, bucket) keys preserved so the
      // table reads in the same row order as the dashboard.
      const rows: Array<{
        side: string;
        bucket: string;
        r: RegimeRow;
      }> = [];
      for (const [side, bucketMap] of Object.entries(byAxis)) {
        for (const [bucket, r] of Object.entries(bucketMap)) {
          if (r.n === 0) continue;
          rows.push({ side, bucket, r });
        }
      }
      lines.push(`### Axis: \`${axis}\``);
      if (rows.length === 0) {
        lines.push("");
        lines.push("_no fills landed in this axis yet_");
        lines.push("");
        continue;
      }
      lines.push("");
      lines.push(
        "| Side | Bucket | n | Confidence | Notional | Minutes | Fills/h | Mean MO 5s | Median MO 5s | Adverse % | Rebate $ | Close PnL $ | Net edge bp |",
      );
      lines.push(
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
      );
      for (const { side, bucket, r } of rows) {
        lines.push(
          `| ${side} | ${bucket} | ${r.n} | ${r.confidence} | ` +
            `${rgFmtUsd(r.notional_usd, 2)} | ` +
            `${rgFmtNum(r.minutes_exposed, 1)} | ` +
            `${rgFmtNum(r.fills_per_hour, 2)} | ` +
            `${rgFmtBp(r.mean_markout_5s_bps, 2)} | ` +
            `${rgFmtBp(r.median_markout_5s_bps, 2)} | ` +
            `${rgFmtPct(r.adverse_fill_pct, 1)} | ` +
            `${rgFmtUsd(r.rebate_usd, 3)} | ` +
            `${rgFmtUsd(r.closed_pnl_usd, 3)} | ` +
            `${rgFmtBp(r.net_edge_bp, 2)} |`,
        );
      }
      lines.push("");
    }
  } else {
    lines.push("## Regime tables (Regime tab)");
    lines.push("");
    lines.push(
      "_no Regime data available — publisher hasn't shipped " +
        "fills_since/exposure_since yet, or this profile has " +
        "`OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED=false`_",
    );
    lines.push("");
  }

  return lines.join("\n");
}


/** ``YYYYMMDDHHMMSS`` (UTC) for filenames. */
export function compactUtcStamp(d: Date = new Date()): string {
  const z = (n: number, w = 2) => n.toString().padStart(w, "0");
  return (
    `${d.getUTCFullYear()}${z(d.getUTCMonth() + 1)}${z(d.getUTCDate())}` +
    `${z(d.getUTCHours())}${z(d.getUTCMinutes())}${z(d.getUTCSeconds())}`
  );
}

