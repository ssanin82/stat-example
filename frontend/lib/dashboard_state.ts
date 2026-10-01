/**
 * Readers for the dashboard's session-scoped S3 objects, written by
 * the bot-side ``app/dashboard_state_publisher.py`` daemon every 30s.
 *
 * Three objects, one per consuming panel set:
 *
 * - ``dashboard/exposure_since_<profile>.json`` — exposure_bars rows
 *   (todo-027 gates, todo-028 inventory).
 * - ``dashboard/fills_since_<profile>.json`` — fills rows with full
 *   Phase 1-4c enrichment (todo-028 inventory, todo-029 cross-venue,
 *   todo-031 regimes, todo-032 execution-quality).
 * - ``dashboard/orders_lifecycle_since_<profile>.json`` —
 *   orders_lifecycle rows including derived
 *   ``placement_to_ack_ms`` / ``cancel_to_close_ms`` / ``lifetime_ms``
 *   (todo-032 execution-quality latency histograms).
 *
 * Same S3-cp pattern the other panels use. Resolution via
 * ``_resolveBotInstance`` so colo profiles short-circuit to their
 * direct S3 bucket without an EC2-tag lookup (bug-023 fix).
 *
 * Each fetcher returns ``{rows, fetched_at_utc, errors}``. The UI
 * decides what to do on errors (typically: keep the prior data
 * cached, render an inline error chip near the panel header).
 */

import { _resolveBotInstance, _awsS3CpToStdout } from "./bot_status";

/** Common envelope written by ``DashboardStatePublisher`` around each
 *  rows array. Carries session metadata so the dashboard can detect
 *  cross-session boundaries without an extra fetch. */
export interface DashboardStatePayload<TRow> {
  schema_version: number;
  profile: string;
  symbol: string | null;
  version: string;
  captured_at_utc: string;
  session_id: string;
  session_started_at_utc: string;
  table: string;
  row_limit: number;
  row_count: number;
  rows: TRow[];
}

/** Result wrapper the API routes return to the dashboard. Mirrors
 *  the shape ``fetchLiveStats`` uses (payload | errors | poll
 *  interval) so the dashboard code can treat all three fetchers
 *  uniformly. */
export interface DashboardStateResult<TRow> {
  profile: string;
  payload: DashboardStatePayload<TRow> | null;
  fetched_at_utc: string;
  poll_interval_ms: number;
  errors: string[];
}

/** Exposure-bar row. Mirrors ``exposure_bars`` table columns plus
 *  the few derived fields the snapshot pipeline already returns. */
export interface ExposureBarRow {
  session_id: string;
  ts_bar: string;
  symbol: string | null;
  mid: number | null;
  spread_bps: number | null;
  microprice: number | null;
  imbalance_top: number | null;
  bid_size_top: number | null;
  ask_size_top: number | null;
  inventory_qty: number | null;
  inventory_utilization: number | null;
  active_sides: string | null;
  quote_eligibility: string | null;
  toxicity_score: number | null;
  vol_estimate: number | null;
  binance_basis_ewma: number | null;
  basis_regime_sign: number | null;
  bid_live: number | null;
  ask_live: number | null;
  bid_distance_ticks: number | null;
  ask_distance_ticks: number | null;
  quoted_spread_bps: number | null;
  adaptive_widen_active: number | null;
  hold_all_active: number | null;
  recovery_cooldown_active: number | null;
  post_fill_cooldown_active_bid: number | null;
  post_fill_cooldown_active_ask: number | null;
  at_touch_adverse_pause_bid: number | null;
  at_touch_adverse_pause_ask: number | null;
  // 2026-05-14 todo-027 Tier 2.
  vol_trend_active: number | null;
  post_swing_active: number | null;
  session_drawdown_tier: string | null;
}

/** Fills-since row. Mirrors ``fills`` table columns including the
 *  full Phase 1-4c enrichment. Optional fields are NULL on rows
 *  written by pre-1.3.x bot builds. */
export interface FillSinceRow {
  fill_id: string;
  ts_fill: string;
  symbol: string;
  side: string;
  // v1.5.234 — the bot's `fills` SQLite table column is `size`
  // (confirmed against snapshot v1.5.230-260529-095319 `bot_db/
  // fills.jsonl.gz` dump: every row has size=<float> and qty=null).
  // The `qty` alias here is null on every current-build wire-format
  // row — kept declared for back-compat with any old reader, but
  // ALL new code should read `size`. The chart's fill markers /
  // inventory anchors silently lost the value (rendered "?" qty
  // and "inv -3.0 → -3.0" because `qty` was undefined / 0) before
  // this fix.
  qty: number | null;
  size: number | null;
  price: number | null;
  notional: number | null;
  fee: number | null;
  rebate_usd: number | null;
  closed_pnl: number | null;
  // Phase 1: decision-state propagation
  quote_eligibility_state: string | null;
  quote_eligibility_reason: string | null;
  inventory_qty_before_fill: number | null;
  inventory_utilization_before_fill: number | null;
  active_sides_at_decision: string | null;
  toxicity_score_at_decision: number | null;
  vol_estimate_at_decision: number | null;
  binance_basis_ewma_at_decision: number | null;
  adaptive_widen_active_at_decision: number | null;
  post_fill_cooldown_active_bid_at_decision: number | null;
  post_fill_cooldown_active_ask_at_decision: number | null;
  at_touch_adverse_pause_bid_at_decision: number | null;
  at_touch_adverse_pause_ask_at_decision: number | null;
  quote_distance_to_touch_ticks_at_placement: number | null;
  // 2026-05-14 todo-027 Tier 2.
  vol_trend_active_at_decision: number | null;
  post_swing_active_at_decision: number | null;
  session_drawdown_tier_at_decision: string | null;
  // Phase 1: cancel-race
  cancel_requested_before_fill: number | null;
  ms_cancel_request_to_fill: number | null;
  // Phase 3: latency
  quote_age_at_fill_ms: number | null;
  effective_book_age_at_fill_ms: number | null;
  effective_book_age_at_last_decision_ms: number | null;
  book_age_seconds_at_fill: number | null;
  book_reference_quality: string | null;
  book_snapshot_quality: string | null;
  // Markouts. v1.4.98 added extended diagnostic horizons (15 s, 60 s,
  // 120 s); v1.4.101 adds them to the typed shape so the dashboard
  // doesn't have to fall through the ``[key: string]: unknown``
  // indexer. Backend persists all horizons on the fills row; values
  // are null on legacy fills + on fills younger than the horizon.
  markout_1s_bps: number | null;
  markout_5s_bps: number | null;
  markout_15s_bps: number | null;
  markout_30s_bps: number | null;
  markout_60s_bps: number | null;
  markout_120s_bps: number | null;
  // Phase 4c: MAE/MFE
  mae_5s_bps: number | null;
  mfe_5s_bps: number | null;
  mae_30s_bps: number | null;
  mfe_30s_bps: number | null;
  // Phase 4c follow-up
  time_to_flat_seconds: number | null;
  // Phase 4a
  expected_net_edge_bps_at_decision: number | null;
  // Regime
  basis_regime_sign: number | null;
  // Soft-flatten attribution
  soft_flatten_event_id: number | null;
  // v1.5.33 — take-profit attribution
  tp_event_id: number | null;
  // Quote quality
  target_half_spread_bps: number | null;
  quote_aggressiveness: string | null;
  // v1.5.190 Phase 8A Option C — per-fill AS attribution. Snapshot of
  // the parent order's ``base_half_spread_bps`` at decision time.
  // NULL on legacy fills.
  as_base_half_spread_bps_at_decision: number | null;
  // v1.5.306 audit §5 P0 #2 — per-fill AQC attribution. The PI
  // aggression output [0,1] + the markout safety-floor flag (1/0) at
  // the moment the parent order was placed. NULL on legacy fills + on
  // fills whose parent was placed while AQC was disabled.
  aqc_aggression_level_at_decision: number | null;
  aqc_safety_floor_engaged_at_decision: number | null;
  // Other observability fields the schema exposes (allow unknown
  // fields so the dashboard doesn't crash on additive columns)
  [key: string]: unknown;
}

/** Orders-lifecycle row. Mirrors ``orders`` table plus the three
 *  SELECT-time-computed latency fields ``orders_lifecycle_since``
 *  derives. */
export interface OrdersLifecycleRow {
  order_id_local: string | null;
  order_id_exchange: string | null;
  symbol: string;
  side: string;
  ts_created: string;
  ts_sent: string | null;
  ts_ack: string | null;
  ts_cancel_requested: string | null;
  ts_closed: string | null;
  status: string | null;
  price: number | null;
  size: number | null;
  // Derived at SELECT time by storage.orders_lifecycle_since
  lifetime_ms: number | null;
  cancel_to_close_ms: number | null;
  placement_to_ack_ms: number | null;
  // 1.4.0 cancel-prio Phase 0.5 — cancel-latency decomposition.
  // ``cancel_decision_to_send_ms`` is the bot-side leg (decision →
  // HTTP wire); ``cancel_send_to_ack_ms`` is the pure transport leg
  // (HTTP wire → ack). Both NULL on rows where the HTTP cancel
  // path wasn't reached (e.g. order-already-gone races, where
  // ``ts_cancel_acked`` is intentionally NOT stamped) and on
  // pre-v1.4.0 rows.
  cancel_decision_to_send_ms: number | null;
  cancel_send_to_ack_ms: number | null;
  // Phase 1 decision-state stamps (subset)
  toxicity_score_at_decision: number | null;
  vol_estimate_at_decision: number | null;
  active_sides_at_decision: string | null;
  binance_basis_ewma_at_decision: number | null;
  adaptive_widen_active_at_decision: number | null;
  quote_distance_to_touch_ticks_at_placement: number | null;
  // 2026-05-14 todo-027 Tier 2.
  vol_trend_active_at_decision: number | null;
  post_swing_active_at_decision: number | null;
  session_drawdown_tier_at_decision: string | null;
  // Quote quality
  target_half_spread_bps: number | null;
  quote_aggressiveness: string | null;
  // v1.5.190 Phase 8A Option C — per-order AS attribution.
  as_base_half_spread_bps_at_decision: number | null;
  // v1.5.306 audit §5 P0 #2 — per-order AQC attribution (PI aggression
  // output + markout safety-floor flag at place-time).
  aqc_aggression_level_at_decision: number | null;
  aqc_safety_floor_engaged_at_decision: number | null;
  // Soft-flatten attribution
  soft_flatten_event_id: number | null;
  // v1.5.33 — take-profit attribution
  tp_event_id: number | null;
  // 1.3.82 connectivity diagnostics. ``cancel_trigger_reason`` is the
  // local-side decision that issued the cancel (reprice_replace /
  // hard_age_cap / side_suppressed / soft_flatten / binance_cross_venue
  // / etc.). ``ts_place_response`` is when the place HTTP response was
  // interpreted — NULL on phantom-place rows where the response never
  // arrived. ``place_response_outcome`` is the venue-side place
  // classification (accepted / exchange_rejected / transport_rejected /
  // unconfirmed). ``cancel_response_outcome`` is the cancel-HTTP
  // classification (success / benign_missing / transport / error).
  cancel_trigger_reason: string | null;
  ts_place_response: string | null;
  place_response_outcome: string | null;
  cancel_response_outcome: string | null;
  // 1.3.83 venue-side detail strings — the OKX sCode + sMsg or
  // equivalent. NULL when the outcome was accepted/success (no
  // rejection detail to record).
  place_response_detail: string | null;
  cancel_response_detail: string | null;
  // Allow unknown fields for the same forward-compat reason as fills.
  [key: string]: unknown;
}

async function _fetchPayload<TRow>(
  profile: string,
  s3Key: string,
  errors: string[]
): Promise<DashboardStatePayload<TRow> | null> {
  let loc;
  try {
    loc = await _resolveBotInstance(profile);
  } catch (e) {
    errors.push(`resolve_failed: ${String(e)}`);
    return null;
  }
  if (!loc) {
    errors.push(`no instance found with tag BotProfile=${profile}`);
    return null;
  }
  if (!loc.logsBucket) {
    errors.push("no logs bucket derivable from EC2 Name tag");
    return null;
  }

  const body = await _awsS3CpToStdout(loc.logsBucket, s3Key, loc.region);
  if (body === null) {
    // Common when the publisher hasn't run yet, or this profile has
    // OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED=false. Leave payload
    // null; the UI shows an empty-state placeholder. NOT an error
    // from the route's perspective.
    return null;
  }
  try {
    return JSON.parse(body) as DashboardStatePayload<TRow>;
  } catch (e) {
    errors.push(`parse_failed: ${String(e)}`);
    return null;
  }
}

/** Default poll cadence: 30 s, matches the publisher's interval.
 *  Polling faster than the publisher's cadence wastes S3 GETs. */
const DEFAULT_POLL_INTERVAL_MS = 30_000;

export async function fetchExposureSince(
  profile: string
): Promise<DashboardStateResult<ExposureBarRow>> {
  const errors: string[] = [];
  const payload = await _fetchPayload<ExposureBarRow>(
    profile,
    `dashboard/exposure_since_${profile}.json`,
    errors
  );
  return {
    profile,
    payload,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
    errors,
  };
}

export async function fetchFillsSince(
  profile: string
): Promise<DashboardStateResult<FillSinceRow>> {
  const errors: string[] = [];
  const payload = await _fetchPayload<FillSinceRow>(
    profile,
    `dashboard/fills_since_${profile}.json`,
    errors
  );
  return {
    profile,
    payload,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
    errors,
  };
}

export async function fetchOrdersLifecycleSince(
  profile: string
): Promise<DashboardStateResult<OrdersLifecycleRow>> {
  const errors: string[] = [];
  const payload = await _fetchPayload<OrdersLifecycleRow>(
    profile,
    `dashboard/orders_lifecycle_since_${profile}.json`,
    errors
  );
  return {
    profile,
    payload,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
    errors,
  };
}
