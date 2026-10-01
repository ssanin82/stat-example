/**
 * GET /api/accounts/[id]/pnl-history?days=30
 *
 * Fetches fills from the last ``days`` days via the venue's
 * archived-fills endpoint (OKX `/trade/fills-history`, Binance
 * `/fapi/v1/userTrades`), aggregates by calendar day (UTC), and
 * returns daily-bucketed realized PnL + fees + net.
 *
 * Cost: paginated. 30 days at our typical fill rate is ~15 OKX
 * pages or ~2 Binance pages -- a few seconds. The dashboard
 * caches the response client-side and re-fetches infrequently
 * (the operator's PnL-history chart doesn't need to track tick-
 * by-tick activity).
 *
 * Caveats:
 *   - OKX archived endpoint covers the past ~3 months. Requesting
 *     `days=180` will return only 90 days of data.
 *   - Binance covers ~7 days on ``/userTrades`` for default IP.
 *     The bot's own DB has full history if longer windows matter
 *     (this dashboard endpoint stays venue-direct).
 *   - Day buckets are UTC date boundaries. No timezone preference
 *     until the operator asks.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { fetchOkxFillsHistoryPaginated } from "@/lib/okx";
import { fetchBinanceFillsHistoryPaginated } from "@/lib/binance";

export const dynamic = "force-dynamic";

interface DailyBucket {
  date: string; // UTC YYYY-MM-DD
  realized_pnl_usd: number;
  fees_usd: number;
  net_usd: number;
  fill_count: number;
}

interface PnlHistoryResponse {
  account_id: string;
  venue: string;
  symbol: string;
  days_requested: number;
  days_covered: number; // actual span of data (oldest fill -> now)
  fills_count: number;
  fetched_at_utc: string;
  /** Server-suggested poll cadence; PnL history doesn't change tick-by-tick. */
  poll_interval_ms: number;
  daily: DailyBucket[]; // ascending by date
  totals: {
    realized_pnl_usd: number;
    fees_usd: number;
    net_usd: number;
    fill_count: number;
  };
  errors: string[];
}

function utcDateString(ms: number): string {
  const d = new Date(ms);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// v1.4.43 rate-limit Tier 1.4: server-side cache + single-flight.
// Pre-v1.4.43 this route was uncached server-side: each call to
// the dashboard kicked off a ~15-page OKX paginated walk
// (`/trade/fills-history`), 15 reads against the `reads` rate-limit
// pool. With two operator tabs open (no client coordination) that
// doubled. Now we cache the result for 5 min and collapse
// concurrent calls into one in-flight Promise.
//
// Cache key: ``account.id + ":" + days``. Different windows (e.g.
// 30d vs 90d) cache independently. In practice the dashboard only
// requests 30d.
const PNL_CACHE_TTL_MS = 5 * 60 * 1000;
const _pnl_cache: Map<string, { ts_ms: number; payload: PnlHistoryResponse }> =
  new Map();
const _pnl_inflight: Map<string, Promise<PnlHistoryResponse>> = new Map();

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  loadIntegrationEnv();
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { error: `unknown account: ${id}` },
      { status: 404 }
    );
  }

  const url = new URL(req.url);
  const rawDays = parseInt(url.searchParams.get("days") || "30", 10);
  // Cap at 180 (OKX archive limit anyway). Floor at 1.
  const days = Math.min(
    Math.max(Number.isFinite(rawDays) ? rawDays : 30, 1),
    180
  );

  // v1.4.43 Tier 1.4: cache + single-flight. Check hot cache before
  // doing any venue REST work. Cache key includes the ``days``
  // parameter so different window sizes don't share entries.
  const cacheKey = `${account.id}:${days}`;
  const now_ms = Date.now();
  const cached = _pnl_cache.get(cacheKey);
  if (cached && now_ms - cached.ts_ms < PNL_CACHE_TTL_MS) {
    return NextResponse.json(cached.payload, {
      headers: { "Cache-Control": "no-store" },
    });
  }
  // Attach to in-flight if one is already running. The cache miss
  // → pagination walk takes several seconds; without single-flight,
  // a second concurrent caller (different tab, dev-server double-
  // mount, etc.) would kick off its own parallel walk and double
  // the rate-limit cost.
  const inflight = _pnl_inflight.get(cacheKey);
  if (inflight) {
    const payload = await inflight;
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
    });
  }

  // Cache miss + nothing in-flight: do the fetch behind a single-
  // flight Promise so concurrent callers attach.
  const fetchPromise = (async (): Promise<PnlHistoryResponse> => {
    return _doFetchPnlHistory(account, days);
  })();
  _pnl_inflight.set(cacheKey, fetchPromise);
  let payload: PnlHistoryResponse;
  try {
    payload = await fetchPromise;
  } finally {
    _pnl_inflight.delete(cacheKey);
  }
  _pnl_cache.set(cacheKey, { ts_ms: Date.now(), payload });
  return NextResponse.json(payload, {
    headers: { "Cache-Control": "no-store" },
  });
}

async function _doFetchPnlHistory(
  account: NonNullable<ReturnType<typeof getAccount>>,
  days: number,
): Promise<PnlHistoryResponse> {
  const since_ms = Date.now() - days * 86400_000;

  const errors: string[] = [];
  let fills: Array<{
    realized_pnl: number;
    fee: number;
    notional_usd: number;
    time_ms: number;
  }> = [];

  try {
    if (account.venue === "okx") {
      fills = await fetchOkxFillsHistoryPaginated(account.symbol, since_ms);
    } else if (account.venue === "binance") {
      fills = await fetchBinanceFillsHistoryPaginated(
        account.symbol,
        since_ms
      );
    } else {
      errors.push(`unsupported venue: ${account.venue}`);
    }
  } catch (e) {
    errors.push(`fetch_failed: ${String(e)}`);
  }

  // Aggregate per UTC day. Use a Map for stable ascending sort.
  const buckets = new Map<string, DailyBucket>();
  for (const f of fills) {
    const date = utcDateString(f.time_ms);
    let b = buckets.get(date);
    if (!b) {
      b = {
        date,
        realized_pnl_usd: 0,
        fees_usd: 0,
        net_usd: 0,
        fill_count: 0,
      };
      buckets.set(date, b);
    }
    b.realized_pnl_usd += Number(f.realized_pnl) || 0;
    // Frontend's fee is positive=paid, negative=rebate. Same
    // convention everywhere on this side of the API.
    b.fees_usd += Number(f.fee) || 0;
    b.fill_count += 1;
  }
  for (const b of buckets.values()) {
    // Net = realized - fee. Positive fee = we paid, so subtract.
    // Negative fee = rebate received, subtracting a negative adds it.
    b.net_usd = b.realized_pnl_usd - b.fees_usd;
  }
  const daily = Array.from(buckets.values()).sort((a, b) =>
    a.date.localeCompare(b.date)
  );

  const totals = daily.reduce(
    (acc, b) => ({
      realized_pnl_usd: acc.realized_pnl_usd + b.realized_pnl_usd,
      fees_usd: acc.fees_usd + b.fees_usd,
      net_usd: acc.net_usd + b.net_usd,
      fill_count: acc.fill_count + b.fill_count,
    }),
    { realized_pnl_usd: 0, fees_usd: 0, net_usd: 0, fill_count: 0 }
  );

  // Coverage = days from oldest fill to now (or 0 if no fills).
  const oldestMs =
    fills.length > 0
      ? Math.min(...fills.map((f) => f.time_ms))
      : Date.now();
  const days_covered = Math.max(
    0,
    (Date.now() - oldestMs) / 86400_000
  );

  const out: PnlHistoryResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    days_requested: days,
    days_covered: Math.min(days, days_covered),
    fills_count: fills.length,
    fetched_at_utc: new Date().toISOString(),
    // 5 minutes -- this view changes very slowly relative to the
    // 1m-5s-tick view of the live chart. Operator can manually
    // refresh by re-selecting the window. v1.4.43: matches
    // PNL_CACHE_TTL_MS — frontend polls at the same cadence and
    // hits the hot cache on every subsequent call.
    poll_interval_ms: PNL_CACHE_TTL_MS,
    daily,
    totals,
    errors,
  };

  return out;
}
