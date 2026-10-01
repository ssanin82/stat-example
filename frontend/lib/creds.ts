/**
 * Per-venue credential resolver with READONLY preference.
 *
 * Convention (introduced 2026-05-04):
 *   For each venue we keep TWO credential sets in
 *   `tests/integration/.env`, distinguished by a ``_READONLY`` suffix:
 *
 *     <VENUE>_API_KEY               -- trading-grade key (the bot uses)
 *     <VENUE>_API_SECRET            -- trading-grade secret
 *     <VENUE>_API_KEY_READONLY      -- read-only key (the dashboard uses)
 *     <VENUE>_API_SECRET_READONLY   -- read-only secret
 *
 *   The dashboard ALWAYS prefers the ``_READONLY`` pair when present.
 *   Reasons:
 *     * Trading keys have IP whitelists (the bot's EC2 EIP). Calling
 *       from a residential IP that rotates with the ISP triggers
 *       endless re-whitelisting churn.
 *     * Read-only keys can be unrestricted -- "read your balance"
 *       isn't a meaningful attack surface even if leaked.
 *     * Lets the operator keep both flavours of keys side by side in
 *       the same `.env` file, no shell shuffling.
 *
 * Atomic pairing: KEY and SECRET MUST come from the same variant. We
 * never mix the read-only KEY with the trading SECRET (would just
 * fail signing with a confusing error). Either both ``_READONLY``
 * vars are set, or we fall back to the trading pair.
 *
 * For OKX, the third secret (PASSPHRASE) follows the same rule.
 */

const _loggedSet = new Set<string>();

export interface ResolvedCreds {
  values: string[];
  /** Which variant got picked, for diagnostics. */
  variant: "readonly" | "trading";
}

/**
 * Resolve a credential pair (or triple, for OKX). Pass the trading-
 * grade env-var names. We probe `<NAME>_READONLY` first, fall back
 * to `<NAME>` if any READONLY field is missing.
 *
 * Throws if neither variant is fully populated.
 */
export function resolveVenueCreds(
  label: string,
  names: string[]
): ResolvedCreds {
  // Try READONLY first -- all-or-nothing per the atomicity rule.
  const ro = names.map((n) => (process.env[`${n}_READONLY`] || "").trim());
  if (ro.every((v) => v.length > 0)) {
    logOnce(label, "readonly", names, ro);
    return { values: ro, variant: "readonly" };
  }
  // Fall back to trading.
  const tr = names.map((n) => (process.env[n] || "").trim());
  if (tr.every((v) => v.length > 0)) {
    logOnce(label, "trading", names, tr);
    return { values: tr, variant: "trading" };
  }
  // Neither full set is populated. Compose a useful error.
  const missing: string[] = [];
  for (let i = 0; i < names.length; i++) {
    if (!tr[i] && !ro[i]) missing.push(names[i]);
    else if (!tr[i]) missing.push(`${names[i]} (or ${names[i]}_READONLY)`);
  }
  throw new Error(
    `${label} credentials missing: ${missing.join(", ")}`
  );
}

function logOnce(
  label: string,
  variant: string,
  names: string[],
  values: string[]
): void {
  // Log once per (label, variant) tuple so the operator sees a line
  // for the read flow (readonly variant) AND a line for the first
  // write action (trading variant), without spamming on every poll.
  const k = `${label}|${variant}`;
  if (_loggedSet.has(k)) return;
  const parts = names.map((n, i) => `${n}=${fingerprint(values[i])}`);
  // eslint-disable-next-line no-console
  console.error(
    `[creds] ${label} -> using ${variant} variant (${parts.join(", ")})`
  );
  _loggedSet.add(k);
}

/**
 * Resolve creds for WRITE actions (cancel order, close position, etc.).
 * Always uses the trading-grade variant -- read-only keys cannot
 * mutate state, so this never falls back to ``_READONLY``. Throws
 * with a clear message if the trading pair isn't fully populated.
 */
export function resolveTradingCreds(
  label: string,
  names: string[]
): ResolvedCreds {
  const tr = names.map((n) => (process.env[n] || "").trim());
  if (tr.every((v) => v.length > 0)) {
    logOnce(label, "trading", names, tr);
    return { values: tr, variant: "trading" };
  }
  const missing = names.filter((n) => !(process.env[n] || "").trim());
  throw new Error(
    `${label} write requires trading-grade credentials. ` +
    `Missing in process env: ${missing.join(", ")}. ` +
    `(Read-only ${label}_*_READONLY keys cannot place / cancel / close.)`
  );
}

/**
 * Quick capability probe: does the operator have trading-grade
 * credentials for this venue in the env right now? Returned in the
 * /state response so the UI can grey out write buttons rather than
 * letting the operator click and discover the failure.
 */
export function hasTradingCreds(names: string[]): boolean {
  return names.every((n) => (process.env[n] || "").trim().length > 0);
}

function fingerprint(v: string): string {
  if (!v) return "<empty>";
  const n = v.length;
  if (n <= 4) return `len=${n}, ***`;
  return `len=${n}, ${v.slice(0, 2)}***${v.slice(-2)}`;
}
