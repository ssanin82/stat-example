/**
 * Reads the bot's ``equity_history/<profile>.json`` from S3 — a
 * session-wide time series of equity samples that powers the Bot
 * Stats panel's PnL chart (path + HWM + drawdown trough).
 *
 * Mirrors the pattern of ``frontend/lib/live_stats.ts``:
 *   - resolve the EC2 instance for the profile (EC2 Name tag → logs bucket)
 *   - aws s3 cp s3://<bucket>/equity_history/<profile>.json -
 *   - parse and return
 *
 * Cadence: bot publishes every ~60 s. Dashboard polls at the same
 * cadence (the response advertises the recommended interval so the
 * client can stay in sync if the bot's interval changes later).
 */

import { _resolveBotInstance, _awsS3CpToStdout } from "./bot_status";

export interface EquitySample {
  ts: string | null;
  equity_usd: number | null;
  realized_pnl_usd: number | null;
  unrealized_pnl_usd: number | null;
  fees_usd: number | null;
  drawdown_usd: number | null;
  // v22-v25 sub-band feeds. NULL on rows written by pre-1.2.9 bot
  // builds (column was added incrementally on the bot side).
  // SessionPnlSubBands reads these directly.
  mid_price?: number | null;
  vol_bps?: number | null;
  session_traded_notional_usd?: number | null;
  position_qty?: number | null;
  // v1.3.31 / todo-030: cross-venue basis EWMA (raw price diff,
  // okx_mid - binance_mid). Convert to bp via basisEwmaToBps(value,
  // mid_price) at render time. NULL when Binance feed disabled or
  // before EWMA seeded.
  binance_basis_ewma?: number | null;
}

export interface EquityHistoryPayload {
  schema_version: number;
  profile: string;
  symbol: string | null;
  version: string | null;
  captured_at_utc: string;
  session_id: string | null;
  session_started_at_utc: string;
  killed: boolean;
  kill_reason: string | null;
  kill_timestamp_utc: string | null;
  /** ``MAX_DRAWDOWN_USD`` cap from the bot's profile env. The
   *  dashboard renders a horizontal threshold line in the chart so
   *  the operator sees how close drawdown is to the kill cap. */
  max_drawdown_usd_cap: number;
  samples: EquitySample[];
}

export interface EquityHistoryResult {
  profile: string;
  payload: EquityHistoryPayload | null;
  fetched_at_utc: string;
  /** Recommended client poll cadence in ms. Default 60s, matches
   *  the bot's publish interval. */
  poll_interval_ms: number;
  errors: string[];
}

export async function fetchEquityHistory(
  profile: string
): Promise<EquityHistoryResult> {
  const errors: string[] = [];
  const out: EquityHistoryResult = {
    profile,
    payload: null,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: 60_000,
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
    `equity_history/${profile}.json`,
    loc.region
  );
  if (body === null) {
    // Common reasons: bot hasn't published yet (just-started session),
    // bot is on a version older than 1.1.48 which didn't ship this
    // publisher, or IAM permission glitch. Leave payload null; UI
    // shows a neutral "no equity history available" placeholder.
    return out;
  }
  try {
    const parsed = JSON.parse(body) as EquityHistoryPayload;
    out.payload = parsed;
  } catch (e) {
    errors.push(`parse_failed: ${String(e)}`);
  }
  return out;
}
