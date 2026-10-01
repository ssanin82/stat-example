/**
 * GET /api/accounts/[id]/activity-calendar?bar=1H[&cursor=<ts>]
 *
 * Backs the Market Activity Calendar modal. ONE OKX page per call —
 * the client paginates explicitly so it can render real progress
 * (N / M pages) while the fetch is in flight. v1.5.249 originally
 * shipped this as a single all-pages-in-one server call; v1.5.255
 * split it so the operator sees honest progress instead of an
 * unbounded "fetching..." label.
 *
 * Query params:
 *   bar     -- "15m" / "30m" / "1H". Only "1H" is currently sent by
 *              the modal; the other two are kept in the allowlist
 *              for a hypothetical future re-add of the granularity
 *              toggle without a route round-trip.
 *   cursor  -- Optional OKX ``after`` ts (in ms). When present, OKX
 *              returns the page of candles immediately BEFORE this
 *              ts. When absent, OKX returns the most recent page.
 *
 * Response:
 *   {
 *     account_id, venue, symbol, bar,
 *     fetched_at_utc,        -- this call's wall time (UTC ISO)
 *     candles: Candle[],     -- one page, chronological order
 *     next_cursor:           -- pass back as ``cursor`` for the next
 *       string | null          page. ``null`` = no more pages.
 *     expected_pages: number -- the client's best estimate of total
 *                              pages for a full window; lets the
 *                              progress bar size itself before
 *                              pagination finishes.
 *     error: string | null
 *   }
 *
 * OKX-only (per the v1.5.249 spec). Binance support would add a
 * branch and isn't on the roadmap.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";

export const dynamic = "force-dynamic";

const OKX_REST = "https://www.okx.com";
// PAGE_SIZE = what we REQUEST from OKX per call (their hard max is 300).
// ESTIMATED_PER_PAGE = what we ASSUME comes back when estimating the
// total page count for the progress bar.
//
// v1.5.259 calibration: OKX's /market/candles returns 100–135 rows per
// call when walking back in time (well below the 300 we request — the
// rolling-window endpoint throttles per-call sizes the deeper we go).
// We deliberately OVER-estimate the total here by using a low per-page
// value: the client stops fetching as soon as it has enough data for
// the calendar (target_candles = DAYS × 24 + buffer), so the bar fills
// to ~70–90 % then disappears. Far better UX than the previous flow
// where the client widened mid-fetch and the operator saw the total
// shift from "5/5" to "5/7" with no explanation.
const PAGE_SIZE = 300;
const ESTIMATED_PER_PAGE = 100;
const DAYS = 28;
const ALLOWED_BARS = new Set(["15m", "30m", "1H"]);

interface Candle {
  time_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ActivityCalendarResponse {
  account_id: string;
  venue: string;
  symbol: string;
  bar: string;
  fetched_at_utc: string;
  candles: Candle[];
  next_cursor: string | null;
  expected_pages: number;
  error: string | null;
}

function parseFloat0(s: string | undefined): number {
  if (s == null) return 0;
  const n = Number(s);
  return Number.isFinite(n) ? n : 0;
}

function expectedPagesFor(bar: string): number {
  // Total bars in a full DAYS-day window ÷ ESTIMATED_PER_PAGE (NOT
  // PAGE_SIZE — see the constant block above for the
  // request-vs-response distinction). Rounded up. The actual loop
  // terminates on ``next_cursor == null``, so this is purely the
  // total displayed in the client progress bar.
  let totalBars: number;
  if (bar === "15m") totalBars = 24 * 4 * DAYS + 4;
  else if (bar === "30m") totalBars = 24 * 2 * DAYS + 2;
  else totalBars = 24 * DAYS + 2; // 1H + buffer
  return Math.max(1, Math.ceil(totalBars / ESTIMATED_PER_PAGE));
}

/** One page of OKX market candles. Returns the rows + the oldest ts
 *  on this page (the client uses that as the next cursor). */
async function fetchOnePage(
  symbol: string,
  bar: string,
  cursor: string,
): Promise<{ candles: Candle[]; oldestTs: number | null }> {
  const url = new URL(`${OKX_REST}/api/v5/market/candles`);
  url.searchParams.set("instId", symbol);
  url.searchParams.set("bar", bar);
  url.searchParams.set("limit", String(PAGE_SIZE));
  if (cursor) {
    url.searchParams.set("after", cursor);
  }
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`okx_candles: HTTP ${r.status}`);
  }
  const j = (await r.json()) as {
    code: string;
    msg: string;
    data: string[][];
  };
  if (j.code !== "0") {
    throw new Error(`okx_candles: ${j.msg || j.code}`);
  }
  const rows = j.data || [];
  const candles: Candle[] = rows
    .map((row) => ({
      time_ms: Number(row[0] || 0),
      open: parseFloat0(row[1]),
      high: parseFloat0(row[2]),
      low: parseFloat0(row[3]),
      close: parseFloat0(row[4]),
      volume: parseFloat0(row[5]),
    }))
    .sort((a, b) => a.time_ms - b.time_ms);
  // OKX returns newest-first; oldest is at the end of the original
  // ``rows`` (== beginning of our sorted output). When the page is
  // empty there's no next cursor.
  const oldestTs = candles.length > 0 ? candles[0].time_ms : null;
  return { candles, oldestTs };
}

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { error: `unknown account: ${id}` },
      { status: 404 },
    );
  }
  if (account.venue !== "okx") {
    return NextResponse.json(
      {
        error:
          "activity-calendar route is OKX-only; " +
          `venue=${account.venue} not supported`,
      },
      { status: 400 },
    );
  }

  const url = new URL(req.url);
  const bar = (url.searchParams.get("bar") || "1H").trim();
  if (!ALLOWED_BARS.has(bar)) {
    return NextResponse.json(
      {
        error: `bar=${bar} not allowed; choose one of ${[...ALLOWED_BARS].join(", ")}`,
      },
      { status: 400 },
    );
  }
  const cursor = (url.searchParams.get("cursor") || "").trim();

  const out: ActivityCalendarResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    bar,
    fetched_at_utc: new Date().toISOString(),
    candles: [],
    next_cursor: null,
    expected_pages: expectedPagesFor(bar),
    error: null,
  };

  try {
    const page = await fetchOnePage(account.symbol, bar, cursor);
    out.candles = page.candles;
    // Next cursor = oldest ts of THIS page, which becomes OKX's
    // ``after`` for the next page (returns records earlier than
    // this ts). ``null`` when this page was empty → loop stops.
    out.next_cursor = page.oldestTs == null ? null : String(page.oldestTs);
  } catch (e) {
    out.error = e instanceof Error ? e.message : String(e);
  }

  return NextResponse.json(out);
}
