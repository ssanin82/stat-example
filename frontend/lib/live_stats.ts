/**
 * Reads the bot's ``live_stats/<profile>.json`` from S3 -- richer
 * trading-internals payload than the heartbeat, written every ~5s.
 * Powers the dashboard's Bot Stats tab.
 *
 * Same instance-resolution + ``aws s3 cp`` shell-out path the bot-
 * status fetcher uses. The bot's IAM role already has s3:PutObject;
 * the dashboard's host AWS creds have GetObject on the same bucket.
 */

import { _resolveBotInstance, _awsS3CpToStdout } from "./bot_status";

export interface LiveStatsWorkingOrder {
  price: number | null;
  size: number | null;
  status: string | null;
  exchange_oid: string | null;
}

export interface LiveStatsTradingVenue {
  best_bid: number | null;
  best_ask: number | null;
  mid: number | null;
  microprice: number | null;
  spread_bps: number | null;
  bid_size: number | null;
  ask_size: number | null;
}

export interface LiveStatsReferenceVenue {
  name: string;
  symbol: string;
  best_bid: number | null;
  best_ask: number | null;
  mid: number | null;
  microprice: number | null;
  bid_size: number | null;
  ask_size: number | null;
}

export interface LiveStatsBasis {
  ewma: number | null;
  fair_value: number | null;
  okx_minus_binance_bps: number | null;
}

export interface LiveStatsInventory {
  qty: number | null;
  notional_usd: number | null;
  max_notional_usd: number | null;
  utilization_pct: number | null;
}

export interface LiveStatsSkew {
  bps: number | null;
  norm_inventory: number | null;
}

export interface LiveStatsMarkouts {
  n_recent_fills: number;
  median_1s_bps: number | null;
  median_5s_bps: number | null;
  adverse_pct_1s: number | null;
  adverse_pct_5s: number | null;
}

export interface LiveStatsBookSignals {
  ob_imbalance_ewma: number | null;
  vol_bps: number | null;
  toxicity_score: number | null;
  toxicity_avg_adverse_markout_bps: number | null;
}

/** PnL attribution decomposition (5s horizon by default).
 *
 *   realized = rebate_income + markout_dollar_impact + residual
 *
 * Rolling window over the bot's ``state.recent_fills`` (capped at
 * 200). For typical sub-200-fill sessions this is the full session
 * total; for longer sessions it's a rolling window which is more
 * useful for live tuning ("how is the bot doing right now?") than
 * a session-cumulative average. */
export interface LiveStatsPnlAttribution {
  horizon_s: number;
  fill_count: number;
  /** Total realized PnL — the number we're decomposing. */
  realized_pnl_usd: number | null;
  /** Maker rebates received (signed: negative = paid as fees,
   *  positive = received as rebates). */
  rebate_income_usd: number | null;
  /** Markout-dollar contribution: ``Σ markout_bps × notional / 10000``.
   *  Negative = adverse selection drag; positive = favourable fills. */
  markout_dollar_impact_usd: number | null;
  /** What's left after rebate + markout — typically inventory PnL
   *  drift past the markout horizon. ``null`` until we have a
   *  realized_pnl_total to compute against. */
  residual_usd: number | null;
  /** Rebate income / total notional, in bps. */
  rebate_bps_of_notional: number | null;
  /** Mean / median markout in bps over the window. */
  mean_markout_bps: number | null;
  median_markout_bps: number | null;
  /** Win-rate (favorable / (favorable+adverse)) at the chosen horizon. */
  adverse_count: number;
  favorable_count: number;
  win_rate: number | null;
  /** Liquidity-flag breakdown — informs whether rebates are real
   *  maker income or a mix with taker fills. */
  maker_count: number;
  taker_count: number;
}

/** One soft-flatten episode row from
 *  ``soft_flatten_events`` joined with attribution counts from
 *  ``orders`` / ``fills``. Plan ref:
 *  plans/20260507-sf-frontend.md Phase 2. */
export interface SoftFlattenEvent {
  id: number;
  ts_start: string;
  ts_end: string | null;
  trigger_reason: string | null;
  initial_force_phase: number | null;
  taker_fallback_ticks: number | null;
  entry_position_qty: number | null;
  entry_mid_price: number | null;
  exit_phase_reached: number | null;
  exit_reason: string | null;
  attributed_orders_count: number;
  attributed_fills_count: number;
  attributed_fills_notional_usd: number;
}

/** Tag map: {order_id_exchange: soft_flatten_event_id} for orders
 *  in the recent window that have a non-null FK. The UI uses these
 *  to map venue REST rows (which don't know about our internal FK)
 *  back to their SF episode. */
export interface SoftFlattenAttribution {
  events: SoftFlattenEvent[];
  order_tags: Record<string, number>;
  fill_tags: Record<string, number>;
}

/** One take-profit episode row from ``tp_events`` joined with
 *  attribution counts. Mirror of ``SoftFlattenEvent`` but with TP-
 *  specific entry fields (trigger uPnL, threshold, disarm margin)
 *  and outcome fields (exit_upnl_bps). v1.5.33. */
export interface TPEvent {
  id: number;
  ts_start: string;
  ts_end: string | null;
  trigger_upnl_bps: number | null;
  trigger_threshold_bps: number | null;
  disarm_margin_bps: number | null;
  entry_position_qty: number | null;
  entry_mid_price: number | null;
  entry_target_price: number | null;
  close_side: string | null;
  exit_reason: string | null;
  exit_upnl_bps: number | null;
  attributed_orders_count: number;
  attributed_fills_count: number;
  attributed_fills_notional_usd: number;
}

/** Tag map for TP: {order_id_exchange: tp_event_id}. Same shape as
 *  the SF attribution block. v1.5.33. */
export interface TPAttribution {
  events: TPEvent[];
  order_tags: Record<string, number>;
  fill_tags: Record<string, number>;
}

/** TP telemetry counters surfaced for Bot Stats card. v1.5.33. */
export interface TPTelemetry {
  enabled: boolean;
  trigger_bps: number;
  disarm_margin_bps: number;
  active: boolean;
  armed_total: number;
  filled_total: number;
  exited_unfilled_total: number;
  exited_sf_takeover_total: number;
}

/** Adaptive spread-widen cooldown surface (1.1.130+).
 *  ``active`` = overlay currently gating the engine.
 *  ``reason`` = categorical trigger (toxicity_hard / toxicity_soft /
 *  markout_adverse / one_sided_ratio / quote_quality). Older bot
 *  builds may omit this block — defensive ``?`` on the field. */
export interface LiveStatsAdaptiveWiden {
  active: boolean;
  seconds_remaining: number;
  reason: string | null;
  quote_quality_latched: boolean;
}

/** Per-side execution-desync detail (1.1.130+). The ``any`` field
 *  is the OR of buy/sell — same as the legacy single boolean. */
export interface LiveStatsDesync {
  any: boolean;
  buy: boolean;
  sell: boolean;
  phase: string | null;
}

/** Basis-regime classifier snapshot (1.2.2+). ``last_regime_sign``
 *  is -1 / 0 / +1. ``last_ic`` is the rolling Pearson IC; ``null``
 *  when fewer than ``min_pair_samples`` paired observations exist
 *  yet (warmup). ``warmed_up`` flips true when pair_count >= min. */
export interface LiveStatsBasisRegime {
  last_regime_sign: number | null;
  last_ic: number | null;
  pair_count: number;
  warmed_up: boolean;
}

/** Flow-score signals (1.2.2+). TFI signed normalised + streak
 *  counts + per-side toxicity scores. */
export interface LiveStatsFlowScore {
  tfi_signed_normalised: number | null;
  streak_buy_count: number;
  streak_sell_count: number;
  buy_toxic_score: number | null;
  sell_toxic_score: number | null;
}

/** Vol×trend conjunction gate state (1.2.2+). */
export interface LiveStatsVolTrendGate {
  active: boolean;
  seconds_remaining: number;
  fire_count: number;
  last_trigger_vol_ratio: number | null;
  last_trigger_drift_bps: number | null;
}

/** Tiered session-PnL drawdown ladder (1.2.1+). */
export interface LiveStatsSessionDrawdown {
  tier: string;
  cooldown_seconds_remaining: number;
  test_resume_fills_remaining: number;
  last_trigger_pnl_usd: number | null;
  fire_count: number;
}

/** v1.5.202 — SF-event-count fatigue ladder. Same five-tier
 *  vocabulary as ``LiveStatsSessionDrawdown`` but keyed on SF episode
 *  count in a rolling window (vs PnL drawdown). Orthogonal to
 *  session_drawdown — either ladder firing pauses quoting. */
export interface LiveStatsSfFatigue {
  enabled?: boolean;
  tier: string;
  events_in_window: number;
  fire_count: number;
  cooldown_seconds_remaining: number;
  window_seconds?: number;
  tier1_widen_count?: number;
  tier2_pause_short_count?: number;
  tier3_pause_long_count?: number;
  tier4_kill_count?: number;
}

/** v1.4.20 rate-limit-observability Phase 1+2 + v1.4.22 per-second
 *  stats extension: per-endpoint pool rate snapshot. Populated by
 *  the bot's heartbeat handler from the OKX adapter's
 *  `rest_runtime_counters()`. */
export interface RateLimitPoolStats {
  current_2s: number;
  peak_2s_60s: number;
  total: number;
  cap: number | null;
  pct_of_cap: number | null;
  /** v1.4.22 per-second stats. Null when fewer than 2 finalized
   *  1-second samples exist. */
  rate_per_sec_min?: number | null;
  rate_per_sec_max?: number | null;
  rate_per_sec_median?: number | null;
  rate_per_sec_p95?: number | null;
  rate_per_sec_samples?: number;
}

export interface LiveStatsPayload {
  schema_version: number;
  profile: string;
  symbol: string;
  version: string;
  captured_at_utc: string;
  interval_seconds: number;
  working_bid: LiveStatsWorkingOrder | null;
  working_ask: LiveStatsWorkingOrder | null;
  trading_venue: LiveStatsTradingVenue;
  reference_venue: LiveStatsReferenceVenue;
  basis: LiveStatsBasis;
  inventory: LiveStatsInventory;
  skew: LiveStatsSkew;
  markouts: LiveStatsMarkouts;
  book_signals: LiveStatsBookSignals;
  /** Cross-venue cancel + amend session counters (v1.4.21+). */
  session_cross_venue_cancel_count?: number;
  session_cross_venue_amend_count?: number;
  /** Per-pool rate snapshot (v1.4.20+). Keys are pool names like
   *  `place_batch`, `cancel_batch`, `amend_batch`, `place_single`,
   *  `cancel_single`, `reads`, `aggregate`. */
  okx_rate_window_per_pool?: Record<string, RateLimitPoolStats>;
  /** New 1.1.130+ blocks. All optional so the dashboard renders
   *  cleanly against older bot builds. */
  adaptive_widen?: LiveStatsAdaptiveWiden | null;
  desync?: LiveStatsDesync | null;
  /** New 1.2.2+ blocks. Optional for the same backward-compat reason. */
  basis_regime?: LiveStatsBasisRegime | null;
  flow_score?: LiveStatsFlowScore | null;
  vol_trend_gate?: LiveStatsVolTrendGate | null;
  /** New 1.2.1+ block. */
  session_drawdown?: LiveStatsSessionDrawdown | null;
  /** v1.5.202 — SF-event-count fatigue ladder. Optional for
   *  backward-compat with pre-v1.5.202 bot builds. */
  sf_fatigue?: LiveStatsSfFatigue | null;
  /** PnL decomposition over the rolling fill window. Null on older
   *  bot builds that don't publish this block. */
  pnl_attribution?: LiveStatsPnlAttribution | null;
  /** Soft-flatten episode attribution. Optional — older bot
   *  builds (pre-Phase-2 of plans/20260507-sf-frontend.md) don't
   *  publish this block. When present, the UI uses
   *  ``order_tags`` / ``fill_tags`` to tint history rows that
   *  belong to an SF episode and ``events`` for the per-episode
   *  drawer. */
  soft_flatten_attribution?: SoftFlattenAttribution | null;
  /** v1.5.33 — take-profit episode attribution. Same shape as the SF
   *  block. ``events`` powers the PnL sub-band TP#N markers and the
   *  Stats/History TP row tinting. */
  tp_attribution?: TPAttribution | null;
  /** v1.5.33 — TP telemetry counters surfaced for the Bot Stats
   *  card. Optional for forward compat. */
  tp_telemetry?: TPTelemetry | null;
}

export interface LiveStatsResult {
  profile: string;
  payload: LiveStatsPayload | null;
  fetched_at_utc: string;
  /** Recommended client poll cadence (ms). Matches the bot's
   *  publish interval -- polling faster wastes S3 GETs. */
  poll_interval_ms: number;
  errors: string[];
}

export async function fetchLiveStats(
  profile: string
): Promise<LiveStatsResult> {
  const errors: string[] = [];
  const out: LiveStatsResult = {
    profile,
    payload: null,
    fetched_at_utc: new Date().toISOString(),
    // Match the bot's default 5s cadence; the actual interval is
    // also embedded in the payload so the client can tighten the
    // polling cadence if the operator dropped it lower at the bot.
    poll_interval_ms: 5_000,
    errors,
  };

  let loc;
  try {
    loc = await _resolveBotInstance(profile);
  } catch (e) {
    errors.push(`resolve_failed: ${String(e)}`);
    return out;
  }
  if (!loc) {
    errors.push(`no instance found with tag BotProfile=${profile}`);
    return out;
  }
  if (!loc.logsBucket) {
    errors.push("no logs bucket derivable from EC2 Name tag");
    return out;
  }

  const body = await _awsS3CpToStdout(
    loc.logsBucket,
    `live_stats/${profile}.json`,
    loc.region
  );
  if (body === null) {
    // Common reasons: bot hasn't shipped a v with live-stats yet,
    // bot down, IAM mismatch. Leave payload null; UI shows "no
    // recent live-stats" placeholder.
    return out;
  }
  try {
    const parsed = JSON.parse(body) as LiveStatsPayload;
    out.payload = parsed;
    if (parsed.interval_seconds && parsed.interval_seconds > 0) {
      out.poll_interval_ms = Math.round(parsed.interval_seconds * 1000);
    }
  } catch (e) {
    errors.push(`parse_failed: ${String(e)}`);
  }
  return out;
}
