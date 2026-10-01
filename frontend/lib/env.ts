/**
 * Loads venue API credentials from `tests/integration/.env` at the
 * repo root. Same file the bot's Python integration tests use --
 * gitignored, lives only on the operator's laptop.
 *
 * Process env wins over the file; this lets a one-off CI / shell
 * override beat the file without editing.
 */

import { readFileSync, existsSync, statSync } from "node:fs";
import { resolve } from "node:path";

// mtime-keyed cache: re-parse the file only when it's been modified
// since the last load. This honours the state route's comment that
// "the operator can edit tests/integration/.env between page polls
// and see the buttons enable/disable on the next refresh" -- the
// previous module-level boolean cache silently broke that promise
// until the next dev-server restart.
//
// We also track keys we wrote into process.env on the last load so
// that on reload we can clear them and re-apply the file's current
// contents. Process-env values that did NOT originate from the file
// (set by the shell / CI override) still win and are left untouched.
let _cachedMtimeMs: number | null = null;
let _keysFromFile: Set<string> = new Set();

export function loadIntegrationEnv(): void {
  // process.cwd() resolves to the frontend/ directory when running
  // `npm run dev`. The .env file lives two levels up.
  const envPath = resolve(process.cwd(), "..", "tests", "integration", ".env");
  let currentMtimeMs: number;
  try {
    if (!existsSync(envPath)) {
      // File disappeared since last load: clear file-sourced keys so
      // the dashboard stops reporting stale capability.
      if (_cachedMtimeMs !== null) {
        for (const k of _keysFromFile) {
          delete process.env[k];
        }
        _keysFromFile = new Set();
      }
      _cachedMtimeMs = 0;
      return;
    }
    currentMtimeMs = statSync(envPath).mtimeMs;
  } catch {
    // Unreadable file -- behave like file-missing.
    if (_cachedMtimeMs !== null) {
      for (const k of _keysFromFile) {
        delete process.env[k];
      }
      _keysFromFile = new Set();
    }
    _cachedMtimeMs = 0;
    return;
  }
  if (_cachedMtimeMs === currentMtimeMs) return;
  // File has changed (or first load). Clear previously-applied keys
  // before re-parsing so deleted entries actually leave process.env.
  for (const k of _keysFromFile) {
    delete process.env[k];
  }
  _keysFromFile = new Set();
  let text: string;
  try {
    text = readFileSync(envPath, "utf-8");
  } catch {
    _cachedMtimeMs = currentMtimeMs;
    return;
  }
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith("#") || !s.includes("=")) continue;
    const eq = s.indexOf("=");
    const k = s.slice(0, eq).trim();
    let v = s.slice(eq + 1).trim();
    // Strip matching surrounding quotes only if they match on both
    // ends (same rule as the Python loader; mismatched quotes are
    // a typo we don't try to fix here -- caller will see the bad
    // value).
    if (
      v.length >= 2 &&
      (v[0] === '"' || v[0] === "'") &&
      v[v.length - 1] === v[0]
    ) {
      v = v.slice(1, -1);
    }
    if (k && process.env[k] === undefined) {
      process.env[k] = v;
      _keysFromFile.add(k);
    }
  }
  _cachedMtimeMs = currentMtimeMs;
}

export interface CredentialFingerprint {
  len: number;
  head: string;
  tail: string;
}

export function fingerprintCred(value: string): CredentialFingerprint | null {
  if (!value) return null;
  const n = value.length;
  if (n <= 4) return { len: n, head: "***", tail: "" };
  return { len: n, head: value.slice(0, 2), tail: value.slice(-2) };
}
