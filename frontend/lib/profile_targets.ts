/**
 * Reads ``scripts/profile_targets.json`` (the deployment-target
 * registry that ``scripts/ops.ps1`` also consumes) and exposes the
 * per-profile target descriptor.
 *
 * Single source of truth: the same JSON file drives both the
 * laptop's PowerShell ops dispatcher AND the dashboard's bot-status
 * resolver, so adding a new colo deployment is one config edit.
 *
 * Profiles not listed in the JSON, or listed with ``kind: aws-ssm``,
 * are routed through the existing EC2 tag-based resolver (the
 * dashboard's ``resolveInstance`` in ``lib/bot_status.ts``). Only
 * ``kind: colo-ssh`` profiles get the SSH/S3-only treatment.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

export interface ColoProfileTarget {
  kind: "colo-ssh";
  /** Hostname or IP (e.g. ``8.217.74.202``). */
  host: string;
  /** SSH login user (typically ``root`` on Alibaba colo). */
  user: string;
  /** Optional explicit SSH key path; ``null`` to use the default. */
  ssh_key: string | null;
  /** S3 bucket where the bot publishes heartbeat / live_stats. */
  logs_bucket: string;
  /** AWS region of the logs bucket (``ap-east-1`` for HK). */
  log_region: string;
  /** systemd unit name on the colo box. */
  systemd_unit: string;
  /** Path of the cloned repo on the colo box. */
  repo_path: string;
  /** Path of the bot's Python venv on the colo box. */
  venv_path: string;
  /** Human label for UI / logs. */
  label: string;
}

export interface AwsSsmProfileTarget {
  kind: "aws-ssm";
  /** Optional label; AWS profiles otherwise resolve via the
   *  ``BotProfile`` EC2 tag. */
  label?: string;
}

export type ProfileTarget = ColoProfileTarget | AwsSsmProfileTarget;

/** Repo-root-relative path to the registry JSON. ``process.cwd()``
 *  is set to ``frontend/`` by ``next dev`` per the README. Going up
 *  one level reaches the repo root, then into ``scripts/``. */
function registryPath(): string {
  return resolve(process.cwd(), "..", "scripts", "profile_targets.json");
}

/** Module-lifetime cache. The registry is small (<1 KB) and the
 *  parse is microseconds, but the readFileSync is cheap-but-not-free
 *  and the dashboard polls per-profile every 10 s. Cache for the
 *  duration of a dev-server session; operator restart picks up
 *  changes. */
let _cachedRegistry: Record<string, ProfileTarget> | null = null;

function loadRegistry(): Record<string, ProfileTarget> {
  if (_cachedRegistry !== null) return _cachedRegistry;
  const p = registryPath();
  if (!existsSync(p)) {
    _cachedRegistry = {};
    return _cachedRegistry;
  }
  try {
    const raw = readFileSync(p, "utf-8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    // Drop the ``_schema`` documentation entry; it isn't a profile.
    delete parsed["_schema"];
    _cachedRegistry = parsed as Record<string, ProfileTarget>;
    return _cachedRegistry;
  } catch {
    _cachedRegistry = {};
    return _cachedRegistry;
  }
}

/**
 * Returns the colo-ssh target for a profile, or ``null`` if the
 * profile is not registered as colo-ssh (i.e. defaults to the
 * AWS/EC2/SSM resolver).
 */
export function getColoProfileTarget(
  profile: string
): ColoProfileTarget | null {
  const reg = loadRegistry();
  const entry = reg[profile];
  if (!entry || entry.kind !== "colo-ssh") return null;
  return entry;
}
