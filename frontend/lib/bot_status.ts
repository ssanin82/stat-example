/**
 * Per-bot EC2 + SSM reachability check.
 *
 * Uses the AWS SDK for JavaScript v3 (@aws-sdk/client-{s3,ec2,ssm,sts})
 * to:
 *   1. Resolve `BotProfile=<name>` tag -> InstanceId + Region
 *      across all opted-in regions (cached after first hit).
 *   2. Get EC2 instance state.
 *   3. Get SSM ping status.
 *   4. Read the bot's heartbeat object from S3.
 *
 * Combined into a single `summary` value the UI can render as a chip.
 *
 * History: previously shelled out to `aws.exe` via child_process.
 * That spawned a fresh Python process per call (~200-500ms warm,
 * 800+ ms cold) and burned ~25% of one core sustained when the
 * dashboard was active. Switched to SDK for ~30-80ms per call and
 * no spawn overhead. Credentials still resolve from the operator's
 * existing ``~/.aws/config`` via the default provider chain.
 */

import {
  EC2Client,
  DescribeRegionsCommand,
  DescribeInstancesCommand,
} from "@aws-sdk/client-ec2";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import {
  SSMClient,
  DescribeInstanceInformationCommand,
} from "@aws-sdk/client-ssm";
import { STSClient, GetCallerIdentityCommand } from "@aws-sdk/client-sts";
import { getColoProfileTarget } from "@/lib/profile_targets";

/** Per-call timeout. EC2/SSM describe calls are typically <500ms;
 * 8s gives slack for cross-region latency without letting frontend
 * polls stack up if the call hangs. */
const AWS_CALL_TIMEOUT_MS = 8_000;

/** Region used for region-agnostic calls (STS, EC2 describe-regions).
 *  us-east-1 is the historical default and is always opted-in. */
const DEFAULT_AWS_REGION = "us-east-1";

interface InstanceLocation {
  instanceId: string;
  region: string;
  /** Logs bucket derived from the EC2's Name tag (``<name_prefix>-bot``)
   *  + the AWS account id. Same convention the laptop ops scripts use
   *  (``scripts/_bot_resolve.ps1``). */
  logsBucket: string | null;
}

// Per-region SDK client caches. Reuse keeps the underlying HTTP
// keep-alive pool warm across polls (one of the wins over the old
// shell-out path, which had to handshake fresh every call).
const _ec2Clients = new Map<string, EC2Client>();
const _s3Clients = new Map<string, S3Client>();
const _ssmClients = new Map<string, SSMClient>();
let _stsClient: STSClient | null = null;

function ec2Client(region: string): EC2Client {
  let c = _ec2Clients.get(region);
  if (!c) {
    c = new EC2Client({ region, requestHandler: { requestTimeout: AWS_CALL_TIMEOUT_MS } });
    _ec2Clients.set(region, c);
  }
  return c;
}

function s3Client(region: string): S3Client {
  let c = _s3Clients.get(region);
  if (!c) {
    c = new S3Client({ region, requestHandler: { requestTimeout: AWS_CALL_TIMEOUT_MS } });
    _s3Clients.set(region, c);
  }
  return c;
}

function ssmClient(region: string): SSMClient {
  let c = _ssmClients.get(region);
  if (!c) {
    c = new SSMClient({ region, requestHandler: { requestTimeout: AWS_CALL_TIMEOUT_MS } });
    _ssmClients.set(region, c);
  }
  return c;
}

function stsClient(): STSClient {
  if (!_stsClient) {
    _stsClient = new STSClient({
      region: DEFAULT_AWS_REGION,
      requestHandler: { requestTimeout: AWS_CALL_TIMEOUT_MS },
    });
  }
  return _stsClient;
}

/** Cached AWS account id; one ``GetCallerIdentity`` per dev-server session. */
let _accountIdCache: string | null = null;
async function getAccountIdCached(): Promise<string | null> {
  if (_accountIdCache !== null) return _accountIdCache;
  try {
    const r = await stsClient().send(new GetCallerIdentityCommand({}));
    _accountIdCache = r.Account ?? null;
    return _accountIdCache;
  } catch {
    return null;
  }
}

/**
 * Lifetime cache of profile -> instance location. Keyed by profile
 * name. Instance IDs don't change without re-provisioning, so a cache
 * that lives for the dev-server lifetime is fine. The first poll
 * after `npm run dev` does a region scan (~5s); subsequent polls hit
 * the cache instantly. A null entry means "scan failed" -- we re-try
 * those on subsequent calls.
 */
const _resolveCache = new Map<string, InstanceLocation>();
/** Promise cache to dedupe concurrent first-poll resolutions. */
const _resolvePending = new Map<string, Promise<InstanceLocation | null>>();

/** Exported wrapper for sibling modules (live_stats.ts). The
 *  default export remains the cached resolveInstance for the
 *  status-chip path. */
export async function _resolveBotInstance(
  profile: string
): Promise<InstanceLocation | null> {
  return resolveInstance(profile);
}

/** Exported s3-get helper for sibling modules. Returns the object
 *  body decoded as UTF-8, or null on any error (missing key, IAM,
 *  network, etc.). */
export async function _awsS3CpToStdout(
  bucket: string,
  key: string,
  region: string
): Promise<string | null> {
  try {
    const r = await s3Client(region).send(
      new GetObjectCommand({ Bucket: bucket, Key: key })
    );
    if (!r.Body) return null;
    // SDK v3 stream body has `transformToString()` on Node.
    return await r.Body.transformToString("utf-8");
  } catch {
    return null;
  }
}

async function resolveInstanceImpl(
  profile: string
): Promise<InstanceLocation | null> {
  // 2026-05-14 BUG-023 fix: colo-hosted profiles short-circuit the
  // AWS-EC2-tag resolver. Pre-fix, the colo branch only existed in
  // ``fetchBotStatus`` (below), so sibling modules that imported
  // ``_resolveBotInstance`` directly (``live_stats.ts``,
  // ``equity_history.ts``, ``config.ts``) fell through to the AWS
  // path, found no EC2 instance for the colo profile, and returned
  // the "no instance found with tag BotProfile=…" error. TON happened
  // to "work" because a phantom EC2 was tagged with its name; SUI
  // had no such phantom, so every Bot-Stats / Config / Equity
  // surface broke after the 2026-05-14 switch. Moving the colo
  // check into the resolver itself fixes all four consumers in
  // one place and makes the AWS phantom-EC2 unnecessary for colo
  // profiles going forward.
  const colo = getColoProfileTarget(profile);
  if (colo) {
    return {
      instanceId: `colo:${colo.host}`,
      region: colo.log_region,
      logsBucket: colo.logs_bucket,
    };
  }
  // List opted-in regions only -- skipping disabled regions saves
  // ~30s of failed describe-instances calls.
  let regions: string[];
  try {
    const r = await ec2Client(DEFAULT_AWS_REGION).send(
      new DescribeRegionsCommand({})
    );
    regions = (r.Regions ?? [])
      .filter(
        (rg) =>
          rg.OptInStatus === "opted-in" ||
          rg.OptInStatus === "opt-in-not-required"
      )
      .map((rg) => rg.RegionName!)
      .filter(Boolean);
  } catch {
    return null;
  }
  for (const region of regions) {
    let resp;
    try {
      resp = await ec2Client(region).send(
        new DescribeInstancesCommand({
          Filters: [
            { Name: "tag:BotProfile", Values: [profile] },
            {
              Name: "instance-state-name",
              Values: ["running", "stopped", "pending", "stopping"],
            },
          ],
        })
      );
    } catch {
      continue;
    }
    const inst = (resp.Reservations ?? [])
      .flatMap((r) => r.Instances ?? [])
      .find((i) => i.InstanceId);
    if (!inst || !inst.InstanceId) continue;
    const nameTag =
      inst.Tags?.find((t) => t.Key === "Name")?.Value ?? "";
    // Name tag has shape ``<name_prefix>-bot``. Strip ``-bot`` to
    // recover the prefix used in the logs-bucket convention.
    const namePrefix = /^(.+)-bot$/.exec(nameTag)?.[1] ?? "dtc-mm-as";
    const accountId = await getAccountIdCached();
    const logsBucket = accountId ? `${namePrefix}-logs-${accountId}` : null;
    return { instanceId: inst.InstanceId, region, logsBucket };
  }
  return null;
}

async function resolveInstance(
  profile: string
): Promise<InstanceLocation | null> {
  if (_resolveCache.has(profile)) return _resolveCache.get(profile)!;
  if (_resolvePending.has(profile)) return _resolvePending.get(profile)!;
  const p = resolveInstanceImpl(profile).then((loc) => {
    _resolvePending.delete(profile);
    if (loc) _resolveCache.set(profile, loc);
    return loc;
  });
  _resolvePending.set(profile, p);
  return p;
}

export type BotStatusSummary =
  | "up"
  | "starting"
  | "stopping"
  | "stopped"
  | "terminated"
  | "killed"
  | "paused"
  | "soft_flattening"
  | "stale"
  | "disconnected"
  | "unknown";

export interface BotStatusResult {
  profile: string;
  instance_id: string | null;
  region: string | null;
  /** EC2 instance state name, e.g. "running" / "stopped" / "pending". */
  ec2_state: string | null;
  /** SSM agent ping status: "Online" / "ConnectionLost" / null. */
  ssm_ping: string | null;
  /** Operator-friendly summary. See the chip-rendering code in page.tsx. */
  summary: BotStatusSummary;
  /**
   * EC2 ``LaunchTime`` (ISO 8601). AWS updates this field on every
   * start (after a stop). Reset by ``ops stop`` + start cycle but
   * NOT by ``ops deploy`` (which restarts only the bot service).
   * For "how long has the bot been quoting" use ``bot_session_*``
   * below instead.
   */
  ec2_launch_time_utc: string | null;
  /**
   * Bot session start (= last bot process / systemd restart, e.g.
   * the most recent ``ops deploy``). Resets on every deploy. This
   * is the "true" bot uptime the operator usually wants. Sourced
   * from the heartbeat JSON the bot writes to S3 every ~30s
   * (``app/heartbeat.py``); null when the heartbeat file is
   * missing or unreadable (operator hasn't set ``LOGS_BUCKET``,
   * IAM issue, fresh deploy still booting, etc.).
   */
  bot_session_started_at_utc: string | null;
  /** Heartbeat freshness -- when the bot last refreshed the file.
   *  Stale (>60s) means the bot may have crashed mid-session. */
  bot_heartbeat_at_utc: string | null;
  /** Bot version reported in the heartbeat. Distinct from the
   *  dashboard's local-repo version chip; useful for confirming
   *  what's ACTUALLY running on EC2 right now. */
  bot_version: string | null;
  /** Bot's reported operational status from heartbeat. Mirrors
   *  ``BotState.bot_status`` (e.g. ``running`` / ``starting`` /
   *  ``halting`` / ``halted``). Distinct from EC2 reachability:
   *  the box can be ``up`` while the bot is ``halted`` due to a
   *  kill-switch trip. */
  bot_status: string | null;
  /** Kill-switch flag from heartbeat. ``true`` when the bot has
   *  tripped a hard halt (drawdown, exec errors, etc.). */
  bot_killed: boolean | null;
  /** Manual-pause flag from heartbeat. ``true`` when the operator
   *  has paused via Telegram ``/pause`` or ``/halt``. */
  bot_manual_pause: boolean | null;
  /** Free-form kill-reason string the bot emitted when it last
   *  tripped a hard halt (``execution_errors``, ``max_drawdown``,
   *  ``manual_kill``, etc.). ``null`` when the bot is healthy or
   *  has never killed in this session. Sourced from the heartbeat. */
  bot_kill_reason: string | null;
  /** UTC ISO 8601 timestamp of the kill, when present. */
  bot_kill_timestamp_utc: string | null;
  /** Free-form pause-reason string emitted by the bot when it
   *  transitioned into PAUSED (e.g. ``manual_pause``,
   *  ``reconcile_stall``, ``post_flatten_timed_out``). ``null``
   *  while the bot is RUNNING / KILLED / SHUTTING_DOWN. */
  bot_pause_reason: string | null;
  /** UTC ISO 8601 timestamp of the most recent pause. */
  bot_pause_timestamp_utc: string | null;
  /** ``TRADING_ENABLED`` setting from the bot's loaded config. */
  bot_trading_enabled: boolean | null;
  /** Venue-side leverage for the trading symbol (e.g. "10"). Fetched
   *  once at bot startup. ``null`` for venues / configurations where
   *  this isn't available. */
  bot_venue_leverage: string | null;
  /** Venue-side margin mode: "cross" / "isolated". */
  bot_venue_margin_mode: string | null;
  /** Account-level position mode: "net_mode" / "long_short_mode". */
  bot_venue_position_mode: string | null;
  /** Latency stats from the heartbeat. ``null`` when the bot is too
   *  freshly booted to have any samples, or when timing trackers are
   *  disabled. */
  latency: BotLatencyStats | null;
  /** Session-scoped fill count. Mirrors the ``fills`` field shown
   *  by Telegram ``/status``. ``null`` when the heartbeat hasn't
   *  shipped this field yet (older bot builds). */
  bot_session_fill_count: number | null;
  /** Session-scoped place-attempt count. Mirrors the ``new_orders``
   *  field on Telegram ``/status``. */
  bot_session_place_attempt_count: number | null;
  /** v1.4.27: session-scoped amend-intent emission count. Mirrors
   *  ``amend_intents_emitted_total``. Shown as a separate column in
   *  the dashboard header so the operator distinguishes fresh place
   *  attempts (queue-position-losing) from amends (queue-preserving). */
  bot_session_amend_intents_emitted_total: number | null;
  /** v1.4.28: session-scoped cancel-intent emission count. Parallel
   *  to ``bot_session_place_attempt_count``. Incremented once per
   *  successful enqueue of a cancel intent in the executor's quote
   *  path. Operator-facing dashboard pairs this with
   *  ``place + amend`` to render the FILLS|ORDERS|CANCEL|VOLUME card
   *  — gives a clean read of "fresh order traffic" vs "withdrawal
   *  traffic" without the combined ``action_count`` blurring the two. */
  bot_session_cancel_attempt_count: number | null;
  /** Session-scoped exchange-action count (placements + cancels +
   *  amends). Mirrors the ``actions`` field on Telegram ``/status``.
   *  Post-v1.4.27 the dashboard ALSO shows fills/new/amend separately
   *  because amend volume often dwarfs new+cancel, making the combined
   *  ``actions`` counter uninformative. */
  bot_session_action_count: number | null;
  /** Cumulative session-scoped traded notional in USD. Sum of
   *  ``abs(fill.notional)`` across every session fill — never
   *  decremented. Operator-visible volume measure. ``null`` on
   *  older bot builds that don't publish it. */
  bot_session_traded_notional_usd: number | null;
  /** Phase 5 ("healthy but not trading" indicator). Seconds since the
   *  bot's last place-order attempt — same field the deadlock watchdog
   *  uses internally (kills at 600 s). The dashboard surfaces this
   *  alongside ``suppression_rate_60s`` to flash an early-warning chip
   *  before the watchdog fires. ``null`` when the bot has issued zero
   *  place attempts (fresh boot) or when the heartbeat doesn't carry
   *  the field yet (older bot builds). */
  bot_execution_idle_seconds: number | null;
  /** Phase 5: fraction of the last 60 s of quote cycles where any
   *  suppressor fired. ``0.0`` = no suppression (bot quoting freely).
   *  ``1.0`` = every cycle suppressed (bot deciding but not executing —
   *  the silent-deadlock signature). ``null`` when no quote cycles have
   *  happened in the rolling window or on older bot builds. */
  bot_suppression_rate_60s: number | null;
  /** v1.5.2: backtest recorder state, surfaced from the bot's
   *  heartbeat ``recording`` block. ``null`` when the bot is too
   *  freshly booted to have published heartbeat, or running an older
   *  build without recording-status support. */
  recording: BotRecordingStatus | null;
  /** Suggested poll cadence; the frontend reads this from the response. */
  poll_interval_ms: number;
  fetched_at_utc: string;
  errors: string[];
}

/** Recording-orchestration status published in the bot's heartbeat.
 *  See ``app/recording_status.py`` for the writer.
 *
 *  Field semantics:
 *  - ``enabled``: ``RECORDING_ENABLED`` setting (config flag)
 *  - ``active``: pointer file exists on colo + session dir is real
 *  - ``session_name`` / ``session_path``: only when ``active=true``
 *  - ``bytes`` / ``files``: cumulative size of the session dir;
 *    snapshotted at heartbeat cadence, so lags real-time by ~30s
 *  - ``profile_resolved``: which bot profile name was used to
 *    look up the pointer file (debug aid; empty when bot couldn't
 *    resolve its profile, e.g. local dev without APP_ENV_FILE) */
export interface BotRecordingStatus {
  enabled: boolean;
  active: boolean;
  session_name: string | null;
  session_path: string | null;
  bytes: number;
  files: number;
  profile_resolved: string;
}

/** Compact latency block surfaced via the bot's heartbeat JSON.
 *  See ``app/heartbeat.py:_build_latency_block`` for the writer. */
export interface BotLatencyStats {
  /** Public-WS event-time → local-receive wall-time gap. */
  exchange_to_local_receive_ms: BotLatencyDistribution | null;
  /** v1.4.58 todo-037 unified outbound transport RTT: pooled samples
   *  across ``place`` + ``amend`` + ``cancel`` op kinds. Single
   *  dashboard row instead of the pre-v1.4.58 per-op pair. Optional
   *  for cross-version compatibility — older heartbeats omit this
   *  field, in which case the dashboard falls back to rendering the
   *  per-op rows below. */
  tx_submit_rtt_ms?: BotLatencyDistribution | null;
  /** Place + amend send → ack RTT from the executor. Pre-v1.4.58 this
   *  was the dashboard's "Place send → ack" row; v1.4.58+ it's only
   *  used as a fallback when ``tx_submit_rtt_ms`` is unavailable.
   *  Still useful for the Telegram /latency command and postmortem. */
  order_submit_rtt_ms: BotLatencyDistribution | null;
  /** Cancel send → ack RTT (v1.4.0 Phase 0.5). PURE TRANSPORT
   *  measurement: time between the bot sending the cancel HTTP
   *  request and receiving a successful response. Companion to
   *  ``order_submit_rtt_ms`` for places. Sample feed: every
   *  successfully-ack'd cancel feeds the OrderRttTracker with
   *  ``op="cancel"``. NULL until the first cancel lands. */
  cancel_submit_rtt_ms: BotLatencyDistribution | null;
}

export interface BotLatencyDistribution {
  min_ms: number | null;
  median_ms: number | null;
  p95_ms: number | null;
  max_ms: number | null;
  sample_count: number;
}

/** Heartbeat is considered stale if older than this. The bot writes
 *  every ~30s, so 90s leaves headroom for a single missed publish
 *  + clock skew before we assume the bot is wedged. */
const HEARTBEAT_STALE_MS = 90_000;

function summarize(
  ec2State: string | null,
  ssmPing: string | null,
  heartbeat: {
    status: string | null;
    killed: boolean | null;
    manualPause: boolean | null;
    heartbeatAtUtc: string | null;
  }
): BotStatusSummary {
  if (ec2State === null) return "unknown";
  if (ec2State === "terminated") return "terminated";
  if (ec2State === "stopped") return "stopped";
  if (ec2State === "stopping" || ec2State === "shutting-down") return "stopping";
  if (ec2State === "pending") return "starting";
  if (ec2State !== "running") return "unknown";

  // EC2 says running. Lean on the heartbeat for the bot's own view
  // of its operational state -- the host can be up while the bot
  // process is halted (kill-switch) or paused.
  const hbFresh =
    heartbeat.heartbeatAtUtc !== null &&
    Date.now() - Date.parse(heartbeat.heartbeatAtUtc) < HEARTBEAT_STALE_MS;
  const ssmOnline = ssmPing === "Online";
  if (hbFresh) {
    if (heartbeat.killed) return "killed";
    if (heartbeat.manualPause) return "paused";
    // Heartbeats published by older bot builds wrote the prefixed
    // enum repr ("BotStatus.RUNNING"); strip the prefix before
    // comparing so this code works during a partial-deploy window.
    const raw = (heartbeat.status || "").replace(/^BotStatus\./, "");
    const s = raw.toLowerCase();
    if (s === "killed") return "killed";
    if (s === "paused") return "paused";
    if (s === "soft_flattening") return "soft_flattening";
    if (s === "starting" || s === "recovering_market_data") return "starting";
    // SHUTTING_DOWN is the new shutdown-marker (was overloaded onto
    // PAUSED, which made every deploy fire a stale "Bot PAUSED"
    // toast). Mapped to "stopping" so the chip pulses amber but the
    // toast trigger (which only fires for killed/paused/soft_flat)
    // stays silent.
    if (s === "flattening" || s === "shutting_down") return "stopping";
    if (s === "running") return "up";
    // v1.5.24 -- handle the v1.5.23 STOPPED tombstone heartbeat
    // (written by systemd's ExecStopPost on the colo path; also
    // reachable on AWS if/when we add a parallel tombstone there).
    if (s === "stopped") return "stopped";
    // Unknown bot_status string -- fall back to ssm reachability.
  }
  // No fresh heartbeat (or unrecognised bot_status string).
  if (ssmOnline) {
    // Host is reachable but bot hasn't written a heartbeat lately.
    // Two sub-cases:
    //   • heartbeat ever published, now stale: stale path is broken
    //     or the bot is wedged. Show 'stale' (amber).
    //   • heartbeat never published: bot is still booting (or
    //     LOGS_BUCKET unset). Show 'starting' (amber).
    return heartbeat.heartbeatAtUtc !== null ? "stale" : "starting";
  }
  // EC2 running, SSM offline, no fresh heartbeat. We've completely
  // lost contact with the bot -- can't determine RUNNING/KILLED/
  // PAUSED. Distinct from 'stopped' (host is up) and 'starting'
  // (boot still in progress). Render in grey.
  return "disconnected";
}

/**
 * Colo-profile summarizer. No EC2 state to lean on — derive
 * everything from heartbeat freshness + the bot's own status string.
 *
 *   • Fresh heartbeat (< 90 s) + ``bot_status``:
 *       running         → "up"
 *       starting/recov  → "starting"
 *       killed          → "killed"
 *       paused          → "paused"
 *       soft_flattening → "soft_flattening"
 *       shutting_down   → "stopping"
 *   • Fresh heartbeat with no recognised status → "up" (defensive
 *     default; the bot is publishing, that's what matters).
 *   • Stale heartbeat (>90 s but file exists)   → "stale" — bot may
 *     have crashed or systemd has stopped it.
 *   • No heartbeat file at all                  → "stopped" — operator
 *     stopped the service cleanly, or it's never been deployed.
 *
 * Note: "stopped" is the right value when the heartbeat is absent
 * because that's exactly when ``systemctl stop dtc-bot`` is the
 * cause. The dashboard uses this to colour the chip and to gate the
 * uptime display.
 */
function summarizeColo(heartbeat: {
  status: string | null;
  killed: boolean | null;
  manualPause: boolean | null;
  heartbeatAtUtc: string | null;
}): BotStatusSummary {
  if (heartbeat.heartbeatAtUtc === null) {
    // No heartbeat published at all.
    return "stopped";
  }
  const hbFresh =
    Date.now() - Date.parse(heartbeat.heartbeatAtUtc) < HEARTBEAT_STALE_MS;
  if (!hbFresh) {
    return "stale";
  }
  if (heartbeat.killed) return "killed";
  if (heartbeat.manualPause) return "paused";
  const raw = (heartbeat.status || "").replace(/^BotStatus\./, "");
  const s = raw.toLowerCase();
  if (s === "killed") return "killed";
  if (s === "paused") return "paused";
  if (s === "soft_flattening") return "soft_flattening";
  if (s === "starting" || s === "recovering_market_data") return "starting";
  if (s === "flattening" || s === "shutting_down") return "stopping";
  if (s === "running") return "up";
  // v1.5.24 -- "stopped" comes from the v1.5.23 STOPPED tombstone
  // heartbeat that the systemd unit's ExecStopPost writes to S3
  // when the bot exits. Pre-fix this fell through to the default
  // "up" branch (since BotStatus has no STOPPED enum value and
  // the original cases only covered the live-bot enum values),
  // producing a misleading "running (obs)" chip after every clean
  // stop. Now mapped explicitly so the chip + RecordingIndicator
  // both treat the tombstone as a stop signal.
  if (s === "stopped") return "stopped";
  // Heartbeat is fresh but bot_status unrecognised — bot is alive
  // (publishing), so default to "up" rather than "unknown".
  return "up";
}

/**
 * Parse a raw heartbeat JSON string and copy the known fields onto
 * the result object. Used by both the AWS path (after S3 fetch) and
 * the colo path (which has its own bucket). Silent failure: a bad
 * JSON body just leaves the bot_* fields at their default null.
 */
function _applyHeartbeatToResult(
  heartbeatBody: string,
  out: BotStatusResult
): void {
  try {
    const parsed = JSON.parse(heartbeatBody) as {
      session_started_at_utc?: string;
      heartbeat_at_utc?: string;
      version?: string;
      bot_status?: string;
      killed?: boolean;
      kill_reason?: string | null;
      kill_timestamp_utc?: string | null;
      pause_reason?: string | null;
      pause_timestamp_utc?: string | null;
      manual_pause?: boolean;
      trading_enabled?: boolean;
      venue_leverage?: string | null;
      venue_margin_mode?: string | null;
      venue_position_mode?: string | null;
      latency?: {
        exchange_to_local_receive_ms?: BotLatencyDistribution | null;
        // v1.4.58 todo-037 unified outbound TX → ack across all ops.
        tx_submit_rtt_ms?: BotLatencyDistribution | null;
        order_submit_rtt_ms?: BotLatencyDistribution | null;
        cancel_submit_rtt_ms?: BotLatencyDistribution | null;
      } | null;
      session_fill_count?: number | null;
      session_place_attempt_count?: number | null;
      session_amend_intents_emitted_total?: number | null;
      session_cancel_attempt_count?: number | null;
      session_action_count?: number | null;
      session_traded_notional_usd?: number | null;
      execution_idle_seconds?: number | null;
      suppression_rate_60s?: number | null;
      // v1.5.2: backtest recorder status. Defensive — every field
      // optional in case an older bot omits the block.
      recording?: {
        enabled?: boolean;
        active?: boolean;
        session_name?: string | null;
        session_path?: string | null;
        bytes?: number;
        files?: number;
        profile_resolved?: string;
      } | null;
    };
    out.bot_session_started_at_utc =
      parsed.session_started_at_utc ?? null;
    out.bot_heartbeat_at_utc = parsed.heartbeat_at_utc ?? null;
    out.bot_version = parsed.version ?? null;
    out.bot_status = parsed.bot_status ?? null;
    out.bot_killed =
      typeof parsed.killed === "boolean" ? parsed.killed : null;
    out.bot_manual_pause =
      typeof parsed.manual_pause === "boolean" ? parsed.manual_pause : null;
    out.bot_trading_enabled =
      typeof parsed.trading_enabled === "boolean"
        ? parsed.trading_enabled
        : null;
    out.bot_kill_reason =
      typeof parsed.kill_reason === "string" ? parsed.kill_reason : null;
    out.bot_kill_timestamp_utc =
      typeof parsed.kill_timestamp_utc === "string"
        ? parsed.kill_timestamp_utc
        : null;
    out.bot_pause_reason =
      typeof parsed.pause_reason === "string" ? parsed.pause_reason : null;
    out.bot_pause_timestamp_utc =
      typeof parsed.pause_timestamp_utc === "string"
        ? parsed.pause_timestamp_utc
        : null;
    out.bot_venue_leverage =
      typeof parsed.venue_leverage === "string"
        ? parsed.venue_leverage
        : null;
    out.bot_venue_margin_mode =
      typeof parsed.venue_margin_mode === "string"
        ? parsed.venue_margin_mode
        : null;
    out.bot_venue_position_mode =
      typeof parsed.venue_position_mode === "string"
        ? parsed.venue_position_mode
        : null;
    if (parsed.latency && typeof parsed.latency === "object") {
      out.latency = {
        exchange_to_local_receive_ms:
          parsed.latency.exchange_to_local_receive_ms ?? null,
        // v1.4.58 todo-037: unified TX → ack across all op kinds.
        // Optional on the wire for cross-version compatibility.
        tx_submit_rtt_ms: parsed.latency.tx_submit_rtt_ms ?? null,
        order_submit_rtt_ms: parsed.latency.order_submit_rtt_ms ?? null,
        cancel_submit_rtt_ms: parsed.latency.cancel_submit_rtt_ms ?? null,
      };
    }
    out.bot_session_fill_count =
      typeof parsed.session_fill_count === "number"
        ? parsed.session_fill_count
        : null;
    out.bot_session_place_attempt_count =
      typeof parsed.session_place_attempt_count === "number"
        ? parsed.session_place_attempt_count
        : null;
    out.bot_session_amend_intents_emitted_total =
      typeof parsed.session_amend_intents_emitted_total === "number"
        ? parsed.session_amend_intents_emitted_total
        : null;
    out.bot_session_cancel_attempt_count =
      typeof parsed.session_cancel_attempt_count === "number"
        ? parsed.session_cancel_attempt_count
        : null;
    out.bot_session_action_count =
      typeof parsed.session_action_count === "number"
        ? parsed.session_action_count
        : null;
    out.bot_session_traded_notional_usd =
      typeof parsed.session_traded_notional_usd === "number"
        ? parsed.session_traded_notional_usd
        : null;
    out.bot_execution_idle_seconds =
      typeof parsed.execution_idle_seconds === "number"
        ? parsed.execution_idle_seconds
        : null;
    out.bot_suppression_rate_60s =
      typeof parsed.suppression_rate_60s === "number"
        ? parsed.suppression_rate_60s
        : null;
    // v1.5.2: recording-status block. Defensive parse — every field
    // optional, fall back to a stable empty shape if the bot is on
    // an older build or the block is malformed.
    if (parsed.recording && typeof parsed.recording === "object") {
      const r = parsed.recording;
      out.recording = {
        enabled: r.enabled === true,
        active: r.active === true,
        session_name:
          typeof r.session_name === "string" ? r.session_name : null,
        session_path:
          typeof r.session_path === "string" ? r.session_path : null,
        bytes: typeof r.bytes === "number" ? r.bytes : 0,
        files: typeof r.files === "number" ? r.files : 0,
        profile_resolved:
          typeof r.profile_resolved === "string" ? r.profile_resolved : "",
      };
    }
  } catch {
    // JSON parse failed -- bucket has a non-JSON object at that
    // key, or heartbeat schema changed. Treat as missing.
  }
}

export async function fetchBotStatus(
  profile: string
): Promise<BotStatusResult> {
  const errors: string[] = [];
  const out: BotStatusResult = {
    profile,
    instance_id: null,
    region: null,
    ec2_state: null,
    ssm_ping: null,
    summary: "unknown",
    ec2_launch_time_utc: null,
    bot_session_started_at_utc: null,
    bot_heartbeat_at_utc: null,
    bot_version: null,
    bot_status: null,
    bot_killed: null,
    bot_manual_pause: null,
    bot_kill_reason: null,
    bot_kill_timestamp_utc: null,
    bot_pause_reason: null,
    bot_pause_timestamp_utc: null,
    bot_trading_enabled: null,
    bot_venue_leverage: null,
    bot_venue_margin_mode: null,
    bot_venue_position_mode: null,
    latency: null,
    bot_session_fill_count: null,
    bot_session_place_attempt_count: null,
    bot_session_amend_intents_emitted_total: null,
    bot_session_cancel_attempt_count: null,
    bot_session_action_count: null,
    bot_session_traded_notional_usd: null,
    bot_execution_idle_seconds: null,
    bot_suppression_rate_60s: null,
    recording: null,
    // 10s polling: instance state changes are slow (EC2 transitions
    // take ~30s), and we don't want to hammer aws CLI on every render.
    poll_interval_ms: 10_000,
    fetched_at_utc: new Date().toISOString(),
    errors,
  };

  // -----------------------------------------------------------------
  // Colo-profile early branch (S3 heartbeat only; no EC2/SSM).
  // -----------------------------------------------------------------
  // When the profile is declared as ``kind: colo-ssh`` in
  // ``scripts/profile_targets.json``, there is no EC2 instance and
  // no SSM agent. Skip those calls entirely. Reachability is derived
  // from heartbeat freshness on S3 (which the bot publishes every
  // ~30s from the colo box).
  const colo = getColoProfileTarget(profile);
  if (colo) {
    out.instance_id = `colo:${colo.host}`;
    out.region = colo.log_region;
    // No EC2 or SSM concept on colo. Leave both null.
    let heartbeatBody: string | null = null;
    try {
      heartbeatBody = await _awsS3CpToStdout(
        colo.logs_bucket,
        `heartbeat/${profile}.json`,
        colo.log_region
      );
    } catch (e) {
      errors.push(`heartbeat_s3: ${String(e)}`);
    }
    if (heartbeatBody !== null) {
      _applyHeartbeatToResult(heartbeatBody, out);
    }
    out.summary = summarizeColo({
      status: out.bot_status,
      killed: out.bot_killed,
      manualPause: out.bot_manual_pause,
      heartbeatAtUtc: out.bot_heartbeat_at_utc,
    });
    return out;
  }

  let loc: InstanceLocation | null;
  try {
    loc = await resolveInstance(profile);
  } catch (e) {
    errors.push(`resolve_failed: ${String(e)}`);
    return out;
  }
  if (!loc) {
    errors.push(`no instance found with tag BotProfile=${profile}`);
    return out;
  }
  out.instance_id = loc.instanceId;
  out.region = loc.region;

  // Run all three reads in parallel. The third is the bot's
  // heartbeat object on S3 -- skipped when we couldn't derive a
  // bucket (missing Name tag / sts call).
  const heartbeatPromise = (async (): Promise<string | null> => {
    if (!loc.logsBucket) return null;
    return _awsS3CpToStdout(
      loc.logsBucket,
      `heartbeat/${profile}.json`,
      loc.region
    );
  })();
  const describeInstancePromise = ec2Client(loc.region).send(
    new DescribeInstancesCommand({ InstanceIds: [loc.instanceId] })
  );
  const ssmPromise = ssmClient(loc.region).send(
    new DescribeInstanceInformationCommand({
      Filters: [{ Key: "InstanceIds", Values: [loc.instanceId] }],
    })
  );
  const [stateRes, pingRes, heartbeatRes] = await Promise.allSettled([
    describeInstancePromise,
    ssmPromise,
    heartbeatPromise,
  ]);

  if (stateRes.status === "fulfilled") {
    const inst = (stateRes.value.Reservations ?? [])
      .flatMap((r) => r.Instances ?? [])
      .find((i) => i.InstanceId);
    out.ec2_state = inst?.State?.Name ?? null;
    // LaunchTime comes back as a Date in the SDK; serialize to ISO
    // for the wire format the frontend already expects.
    out.ec2_launch_time_utc = inst?.LaunchTime
      ? new Date(inst.LaunchTime).toISOString()
      : null;
  } else {
    errors.push(`ec2_describe: ${String(stateRes.reason)}`);
  }
  if (pingRes.status === "fulfilled") {
    // Empty result = instance not yet registered with SSM (still booting,
    // or stopped). Treat as null rather than empty-string.
    const info = pingRes.value.InstanceInformationList?.[0];
    out.ssm_ping = info?.PingStatus ?? null;
  } else {
    errors.push(`ssm_describe: ${String(pingRes.reason)}`);
  }
  // Heartbeat parse. Strictly best-effort: missing/unreadable file
  // silently leaves the bot_session_* fields null. The dashboard's
  // chip falls back to host uptime when these are null.
  if (
    heartbeatRes.status === "fulfilled" &&
    heartbeatRes.value !== null
  ) {
    _applyHeartbeatToResult(heartbeatRes.value, out);
  }

  out.summary = summarize(out.ec2_state, out.ssm_ping, {
    status: out.bot_status,
    killed: out.bot_killed,
    manualPause: out.bot_manual_pause,
    heartbeatAtUtc: out.bot_heartbeat_at_utc,
  });
  return out;
}
