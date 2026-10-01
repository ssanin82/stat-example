/**
 * GET /api/accounts/[id]/volume
 *
 * Returns true rolling-window traded volume (1d / 7d / 15d / 30d)
 * by paginating the venue's fills-history endpoint back 30 days.
 * The legacy ``/history?limit=N`` route caps at N fills, which on
 * an active bot covers only a few hours and makes the dashboard's
 * Volume tiles show the same number across every window. This
 * route does the right thing -- paginate to the cutoff, sum
 * notional_usd per window, return totals.
 *
 * Cost / freshness: paginated fetches hit the venue ~5-50× per call.
 * Use a server-side LRU cache (5-minute TTL) to keep the dashboard
 * polling cadence loose. Volume changes slowly anyway.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { fetchOkxFillsHistoryPaginatedFull } from "@/lib/okx";
import { fetchBinanceFillsHistoryPaginatedFull } from "@/lib/binance";

export const dynamic = "force-dynamic";

interface VolumeWindow {
  label: string;
  days: number;
  notional_usd: number;
  fill_count: number;
}

/** Per-token 30d volume row, sorted desc by notional. Computed
 *  alongside the time-window rollups from the SAME paginated fills
 *  walk so the dashboard's Tokens-tab and Total-Volume tile share
 *  one venue REST cost rather than each driving its own walk. */
interface TokenVolumeRow {
  symbol: string;
  notional_usd: number;
  fill_count: number;
}

interface VolumeResponse {
  account_id: string;
  venue: string;
  symbol: string;
  windows: VolumeWindow[];
  /** Per-token breakdown over the full 30d window (the longest one
   *  the route paginates back to). Same fills source as ``windows``.
   *  Sorted desc by ``notional_usd`` so the largest contributor is
   *  index 0. Empty array when no fills landed in the window. */
  tokens: TokenVolumeRow[];
  total_fills_scanned: number;
  oldest_fill_utc: string | null;
  /** True iff the pagination retrieved every fill the venue knows
   *  about (or we crossed the requested cutoff). False if it
   *  stopped early due to a safety cap -- in that case ALL windows
   *  are potentially under-counted. The frontend uses this in
   *  preference to per-window heuristics so a fresh / small account
   *  doesn't see spurious ⚠ warnings. */
  complete: boolean;
  fetched_at_utc: string;
  /** Provenance of the served payload. ``fresh`` = just-paginated.
   *  ``stale_complete`` = the latest paginated walk was partial
   *  (probably an OKX rate-limit blip), but we still have a recent
   *  complete walk in cache and are serving that instead so the UI
   *  doesn't flap to a partial result. ``cache_hot`` = served from
   *  the within-TTL cache without re-fetching. */
  served_from: "fresh" | "stale_complete" | "cache_hot";
  /** Server-recommended client poll cadence. Volume tiles don't
   *  need fast updates; 1 min keeps venue REST cost predictable. */
  poll_interval_ms: number;
  errors: string[];
}

interface CacheEntry {
  ts_ms: number;
  payload: VolumeResponse;
}

// Hot cache: any fresh fetch (complete or not) for this many ms.
// v1.4.43 rate-limit Tier 1.4: raised from 60 s to 5 min. Volume
// is a postmortem-grade aggregate (1d/7d/15d/30d realized notional
// + fills count). It does not need sub-minute freshness — the
// operator looks at it once or twice per session, not every minute.
// Pre-v1.4.43 the 60 s TTL meant every cache-miss kicked off a
// ~15-page OKX paginated walk (~15 reads of the `reads` pool in a
// ~7-15 s window). At 60 s TTL we paid that burst every minute;
// at 5 min TTL we pay it every 5 minutes, cutting the worst single
// contributor to the dashboard's peak / 2 s by 5×. Frontend
// receives this TTL via ``poll_interval_ms`` and matches its poll
// cadence to it.
const CACHE_TTL_MS = 5 * 60 * 1000;
// Stale-while-revalidate window for the LAST KNOWN COMPLETE walk.
// When a fresh fetch returns ``complete=false`` (typically because
// OKX rate-limited mid-pagination and the paginator exited early),
// the volume tiles would flap to "all four windows show the same
// 500-fill total ⚠". That's worse than serving a 5-minute-old
// complete result. 30 min is generous enough to ride out a few
// consecutive rate-limit blips while still surfacing the warning
// quickly if the venue is genuinely down.
const COMPLETE_STALE_TTL_MS = 30 * 60 * 1000;
// Two-tier cache: ``latest`` (any payload, hot-window) and
// ``latest_complete`` (only payloads with complete=true, used as
// fallback when the fresh fetch is partial).
const _cache_latest = new Map<string, CacheEntry>();
const _cache_latest_complete = new Map<string, CacheEntry>();

// Single-flight dedup. Without this, concurrent callers (React
// strict-mode double-mount in dev, two browser tabs, a refresh
// landing while the previous fetch is still walking pages) ALL
// trigger pagination simultaneously. OKX's per-endpoint rate limit
// budget is shared across them, so each parallel walk gets fewer
// successful pages before hitting 50011 and exiting partial.
// Keying by ``cacheKey`` means concurrent callers attach to the
// in-flight Promise instead of issuing their own pagination.
const _inflight = new Map<string, Promise<VolumeResponse>>();

const WINDOWS = [
  { label: "1d", days: 1 },
  { label: "7d", days: 7 },
  { label: "15d", days: 15 },
  { label: "30d", days: 30 },
];

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { error: `unknown account: ${id}` },
      { status: 404 }
    );
  }
  loadIntegrationEnv();

  // Hot-cache check: any payload less than CACHE_TTL_MS old gets
  // served as-is. The two-tier fallback below only kicks in when we
  // actually re-paginate.
  const cacheKey = `${account.venue}:${account.id}`;
  const cachedHot = _cache_latest.get(cacheKey);
  const now = Date.now();
  if (cachedHot && now - cachedHot.ts_ms < CACHE_TTL_MS) {
    return NextResponse.json(
      { ...cachedHot.payload, served_from: "cache_hot" as const },
      { headers: { "Cache-Control": "no-store" } }
    );
  }

  // Single-flight: if a pagination is already in flight for this
  // account, attach to it instead of starting a second one. Without
  // this, two concurrent callers (dev mode double-mount, two tabs,
  // refresh during pagination) BOTH paginate, sharing OKX's per-
  // endpoint rate-limit budget — both end up partial.
  const inflight = _inflight.get(cacheKey);
  if (inflight) {
    const payload = await inflight;
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
    });
  }

  const promise = _runPagination(account, now);
  _inflight.set(cacheKey, promise);
  let payload: VolumeResponse;
  try {
    payload = await promise;
  } finally {
    _inflight.delete(cacheKey);
  }

  // Cache the fresh result regardless of complete flag (so hot
  // polls within CACHE_TTL_MS don't re-paginate). Track the latest
  // COMPLETE result separately for the stale-while-revalidate path.
  if (payload.errors.length === 0) {
    _cache_latest.set(cacheKey, { ts_ms: now, payload });
    if (payload.complete) {
      _cache_latest_complete.set(cacheKey, { ts_ms: now, payload });
    }
  }

  // Stale-while-revalidate: if the fresh walk came back partial,
  // prefer a recent complete walk over flapping the UI to a partial
  // result. Empirically the partial cases are transient OKX rate-
  // limit blips on the /fills-history endpoint that resolve within
  // 1-2 minutes; serving the previous complete totals during that
  // window keeps the dashboard stable.
  if (
    !payload.complete &&
    payload.errors.length === 0 &&
    _cache_latest_complete.has(cacheKey)
  ) {
    const last = _cache_latest_complete.get(cacheKey)!;
    if (now - last.ts_ms < COMPLETE_STALE_TTL_MS) {
      return NextResponse.json(
        {
          ...last.payload,
          // The served payload is genuinely complete (it was when
          // captured) so leave ``complete: true`` from the cached
          // copy. Provenance flag tells the operator the totals
          // they see are slightly older than ``fetched_at_utc``
          // would imply.
          served_from: "stale_complete" as const,
        },
        { headers: { "Cache-Control": "no-store" } }
      );
    }
  }

  return NextResponse.json(payload, {
    headers: { "Cache-Control": "no-store" },
  });
}

/** The actual paginated walk, factored out so the GET handler can
 *  wrap it in single-flight dedup. */
async function _runPagination(
  account: ReturnType<typeof getAccount> & {},
  now: number
): Promise<VolumeResponse> {
  const errors: string[] = [];
  const out: VolumeResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    windows: WINDOWS.map((w) => ({
      label: w.label,
      days: w.days,
      notional_usd: 0,
      fill_count: 0,
    })),
    tokens: [],
    total_fills_scanned: 0,
    oldest_fill_utc: null,
    complete: false,
    fetched_at_utc: new Date().toISOString(),
    served_from: "fresh",
    poll_interval_ms: CACHE_TTL_MS,
    errors,
  };

  const since_ms = now - 30 * 86400_000;
  // Carrying ``symbol`` through alongside time + notional so we can
  // group by token without a second pagination pass. The full
  // FillHistoryRecord has more fields but those are the three the
  // rollup + breakdown need.
  let fills: Array<{
    time_ms: number;
    notional_usd: number;
    symbol: string;
  }> = [];
  try {
    if (account.venue === "okx") {
      // Account-wide: pass ``null`` symbol → fetches every SWAP
      // fill on the account regardless of instId. The the partner rebate
      // tier is calculated on the account's total maker volume,
      // not per-symbol, so the operator wants visibility on the
      // sum across all symbols ever traded (e.g. SUI fills before
      // the switch + TON fills after).
      const r = await fetchOkxFillsHistoryPaginatedFull(
        null,
        since_ms
      );
      fills = r.fills;
      out.complete = r.complete;
    } else if (account.venue === "binance") {
      // Binance ``/fapi/v1/userTrades`` requires a symbol per
      // request, so the account-wide query needs a known symbol
      // list. For now keep single-symbol behaviour on Binance —
      // when the operator goes multi-symbol on Binance we can
      // extend this to iterate over the configured symbol list.
      const r = await fetchBinanceFillsHistoryPaginatedFull(
        account.symbol,
        since_ms
      );
      fills = r.fills;
      out.complete = r.complete;
    } else {
      errors.push(`unsupported venue: ${account.venue}`);
    }
  } catch (e) {
    errors.push(`fetch_failed: ${String(e)}`);
  }

  out.total_fills_scanned = fills.length;
  if (fills.length > 0) {
    const oldest = Math.min(...fills.map((f) => Number(f.time_ms || 0)));
    out.oldest_fill_utc = new Date(oldest).toISOString();
  }

  for (const w of out.windows) {
    const cutoff = now - w.days * 86400_000;
    let total = 0;
    let count = 0;
    for (const f of fills) {
      if (Number(f.time_ms || 0) >= cutoff) {
        total += Math.abs(Number(f.notional_usd) || 0);
        count++;
      }
    }
    w.notional_usd = total;
    w.fill_count = count;
  }

  // Per-token rollup. Same input fills, just grouped by symbol.
  // Empty-string symbol (very old fills before the parser carried
  // instId) lands under "(unknown)" so they don't silently disappear.
  const byToken = new Map<string, TokenVolumeRow>();
  for (const f of fills) {
    const sym = f.symbol || "(unknown)";
    let row = byToken.get(sym);
    if (!row) {
      row = { symbol: sym, notional_usd: 0, fill_count: 0 };
      byToken.set(sym, row);
    }
    row.notional_usd += Math.abs(Number(f.notional_usd) || 0);
    row.fill_count += 1;
  }
  out.tokens = Array.from(byToken.values()).sort(
    (a, b) => b.notional_usd - a.notional_usd,
  );

  return out;
}
