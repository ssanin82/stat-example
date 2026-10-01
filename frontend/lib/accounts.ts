/**
 * Account registry: scans `config/profiles/*.env` and returns the
 * EXCHANGE + SYMBOL fields for each, mapped to a stable `id` =
 * profile filename without the `.env` suffix (same convention as
 * the bot's `BotProfile` EC2 tag).
 *
 * Why read profile files instead of hardcoding accounts: the
 * dashboard should follow the bot's source-of-truth for "what
 * accounts are ACTIVE" -- adding/removing a profile in
 * `config/profiles/` should immediately surface (or hide) it in
 * the dashboard with no code or restart.
 *
 * Convention (informal, agreed 2026-05-04 with operator): everything
 * in `config/profiles/` is an active trading account. Inactive /
 * archived profiles live in a sibling directory, typically
 * `config/profiles_unused/`, which the dashboard does NOT scan.
 *
 * No caching: reading ~5 small env files per request is rounding-
 * error cost, and a stale cache silently breaks "operator just
 * moved a file" UX.
 *
 * Path resolution: we resolve from `process.cwd()`. Next.js dev
 * sets cwd to the directory `npm run dev` was launched from --
 * which is `frontend/` per the README. Going up one level reaches
 * the repo root.
 */

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";

export type SupportedVenue = "okx" | "binance";
/** Includes the explicit ``"unsupported"`` sentinel so unrecognised
 *  venues surface in the dropdown but get rejected downstream by the
 *  state route, instead of silently routing through Binance code
 *  paths (the previous buggy behaviour). */
export type Venue = SupportedVenue | "unsupported";

export interface Account {
  /** `prod.okx.sui.usdt.perp` -- the filename without `.env`. */
  id: string;
  /** Human-readable label shown in the dropdown. */
  label: string;
  /** Lowercase venue key. */
  venue: Venue;
  /** Trading symbol in the venue's native form. */
  symbol: string;
  /** Per-account risk thresholds parsed from the profile env. The
   *  dashboard uses these for INDEPENDENT (frontend-side) violation
   *  checks on orders / fills as a defense in depth -- if a value
   *  exceeds these caps, the row gets a red background regardless of
   *  what the bot did or didn't do. ``null`` when unset in the
   *  profile (cap not enforced; dashboard skips that check). */
  risk: AccountRiskThresholds;
}

export interface AccountRiskThresholds {
  /** Per-order USD notional cap (``MAX_ORDER_NOTIONAL_USD``). */
  max_order_notional_usd: number | null;
  /** Per-position USD notional cap (``MAX_POSITION_NOTIONAL_USD``). */
  max_position_notional_usd: number | null;
  /** Per-position base-units cap (``MAX_ABS_POSITION``). */
  max_abs_position: number | null;
  /** Configured quote notional, used as the EXPECTED size baseline. */
  quote_notional_usd: number | null;
}

function profilesDir(): string {
  return resolve(process.cwd(), "..", "config", "profiles");
}

export function listAccounts(): Account[] {
  const dir = profilesDir();
  if (!existsSync(dir)) {
    return [];
  }
  const out: Account[] = [];
  for (const name of readdirSync(dir)) {
    if (!name.endsWith(".env")) continue;
    const id = name.slice(0, -".env".length);
    const fields = parseProfileEnv(resolve(dir, name));
    const venueRaw = (fields.EXCHANGE || "").toLowerCase().trim();
    const symbol = (fields.SYMBOL || "").trim();
    if (!venueRaw || !symbol) continue;
    // The dashboard currently has adapters for OKX + Binance. If a
    // profile in this directory carries a different venue, surface
    // it in the dropdown anyway (with the raw venue label) so the
    // operator notices it exists -- but tag the descriptor with the
    // explicit ``"unsupported"`` venue so the state endpoint can
    // return a clear "unsupported venue" error for it. Previously
    // the code coerced the venue to ``"binance"`` which silently
    // routed unsupported profiles through the Binance code paths.
    const venue: Venue = isSupportedVenue(venueRaw) ? venueRaw : "unsupported";
    out.push({
      id,
      label: humanLabel(venueRaw, symbol),
      venue,
      symbol,
      risk: {
        max_order_notional_usd: _parseFloatOrNull(fields.MAX_ORDER_NOTIONAL_USD),
        max_position_notional_usd: _parseFloatOrNull(fields.MAX_POSITION_NOTIONAL_USD),
        max_abs_position: _parseFloatOrNull(fields.MAX_ABS_POSITION),
        quote_notional_usd: _parseFloatOrNull(fields.QUOTE_NOTIONAL_USD),
      },
    });
  }
  // Stable sort: alphabetical by id so the dropdown order doesn't
  // jump around between page loads.
  out.sort((a, b) => a.id.localeCompare(b.id));
  return out;
}

export function getAccount(id: string): Account | null {
  return listAccounts().find((a) => a.id === id) ?? null;
}

function isSupportedVenue(s: string): s is SupportedVenue {
  return s === "okx" || s === "binance";
}

function _parseFloatOrNull(v: string | undefined): number | null {
  if (v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function humanLabel(venue: string, symbol: string): string {
  return `${venue.toUpperCase()} ${symbol}`;
}

function parseProfileEnv(path: string): Record<string, string> {
  const out: Record<string, string> = {};
  let text: string;
  try {
    text = readFileSync(path, "utf-8");
  } catch {
    return out;
  }
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith("#") || !s.includes("=")) continue;
    const eq = s.indexOf("=");
    const k = s.slice(0, eq).trim();
    let v = s.slice(eq + 1).trim();
    if (
      v.length >= 2 &&
      (v[0] === '"' || v[0] === "'") &&
      v[v.length - 1] === v[0]
    ) {
      v = v.slice(1, -1);
    }
    if (k) out[k] = v;
  }
  return out;
}
