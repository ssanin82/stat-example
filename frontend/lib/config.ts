/**
 * Reads the bot's ``config/<profile>.json`` from S3 — a one-shot
 * snapshot of the bot's effective configuration written at startup.
 * Powers the Bot Stats panel's CONFIG tab.
 *
 * Mirrors the pattern of ``frontend/lib/equity_history.ts``:
 *   - resolve the EC2 instance for the profile
 *   - aws s3 cp s3://<bucket>/config/<profile>.json -
 *   - parse and return
 *
 * Cadence: bot writes once per deploy. The dashboard fetches once
 * per page load (no polling — the file doesn't change in-flight).
 */

import { _resolveBotInstance, _awsS3CpToStdout } from "./bot_status";

export interface ConfigPayload {
  schema_version: number;
  profile: string;
  symbol: string | null;
  version: string | null;
  captured_at_utc: string;
  session_id: string | null;
  session_started_at_utc: string;
  /** Absolute path of the env file the bot loaded (server-side). */
  env_file_path: string;
  /** Verbatim env-file contents with secret values masked as ``***``.
   *  Comments and blank lines pass through unchanged so the dashboard
   *  can attach each ``# ...`` block as a tooltip on the ``KEY=value``
   *  line that follows it. */
  env_file_raw: string;
  env_file_read_error: string | null;
  /** ``Settings.sanitized_dict()`` — the resolved view after Pydantic
   *  defaults / coercions / validators. Keys are UPPER_SNAKE_CASE. */
  resolved_settings: Record<string, unknown>;
}

export interface ConfigResult {
  profile: string;
  payload: ConfigPayload | null;
  fetched_at_utc: string;
  errors: string[];
}

export async function fetchConfig(profile: string): Promise<ConfigResult> {
  const errors: string[] = [];
  const out: ConfigResult = {
    profile,
    payload: null,
    fetched_at_utc: new Date().toISOString(),
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
    `config/${profile}.json`,
    loc.region,
  );
  if (body === null) {
    return out;
  }
  try {
    const parsed = JSON.parse(body) as ConfigPayload;
    out.payload = parsed;
  } catch (e) {
    errors.push(`parse_failed: ${String(e)}`);
  }
  return out;
}
