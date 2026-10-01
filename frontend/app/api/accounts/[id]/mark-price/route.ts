/**
 * GET /api/accounts/[id]/mark-price?since=<ms>&bar=<interval>
 *
 * Returns the venue's public mark-price candles for the requested
 * window. Used by the dashboard's mark-price overlay on the Session-
 * PnL chart's Mid sub-band.
 *
 * Source: target-venue public REST (no auth). For OKX this is
 * ``/api/v5/market/mark-price-candles`` (rolling 3-month archive).
 * For other venues, this route returns an empty result — the
 * overlay simply doesn't render.
 *
 * Why this route exists separately from the existing klines/state
 * routes: mark price differs from mid (mark is the venue's smoothed
 * computation used for liquidation valuation; mid is raw best-bid/
 * best-ask midpoint). The operator wants to see them overlaid so a
 * drift between bot's quoted mid and the venue's mark is visually
 * obvious.
 *
 * Cache-Control: no-store. Public mark-price candles update on the
 * venue's bar cadence (~1 min); the dashboard polls at the same
 * cadence as the equity-history publisher (~60 s) so we never lag
 * by more than the bar interval.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchOkxMarkPriceCandles } from "@/lib/okx";

export const dynamic = "force-dynamic";

/** Polling cadence advertised back to the client. Matches the
 *  equity-history publisher's ~60 s rhythm — pulling mark candles
 *  faster than that wastes the venue's request budget without
 *  surfacing a finer-grained value (the venue itself only updates
 *  the candle every bar interval). */
const POLL_INTERVAL_MS = 60_000;

/** Default lookback if the caller doesn't supply ``since``. Six
 *  hours of 1-minute bars = 360 rows = 4 OKX pages, ~1 s end-to-end.
 *  Generous enough to cover most operator-visible sessions without
 *  forcing them to pass a query param. */
const DEFAULT_LOOKBACK_MS = 6 * 60 * 60 * 1000;

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

  const url = new URL(req.url);
  const sinceParam = url.searchParams.get("since");
  const barParam = url.searchParams.get("bar") || "1m";
  const nowMs = Date.now();
  let sinceMs = nowMs - DEFAULT_LOOKBACK_MS;
  if (sinceParam) {
    const parsed = Number(sinceParam);
    if (Number.isFinite(parsed) && parsed > 0 && parsed < nowMs) {
      sinceMs = parsed;
    }
  }

  const errors: string[] = [];
  let candles: Array<{ time_ms: number; mark: number }> = [];

  if (account.venue === "okx") {
    try {
      const rows = await fetchOkxMarkPriceCandles(
        account.symbol,
        barParam,
        sinceMs,
      );
      // Trim to the requested window — fetchOkxMarkPriceCandles
      // may return a few rows older than sinceMs because the
      // pagination cursor stops one page past the boundary.
      candles = rows.filter((c) => c.time_ms >= sinceMs);
    } catch (e) {
      errors.push(`fetch_okx_mark_price: ${String(e)}`);
    }
  } else {
    // Bluefin / Hyperliquid / Binance: not wired yet. Empty
    // ``candles`` ⇒ the overlay simply doesn't render. Filed as a
    // small follow-up when those venues come online.
    errors.push(`mark_price_not_implemented_for_venue: ${account.venue}`);
  }

  return NextResponse.json(
    {
      profile: account.id,
      venue: account.venue,
      symbol: account.symbol,
      bar: barParam,
      since_ms: sinceMs,
      fetched_at_utc: new Date().toISOString(),
      poll_interval_ms: POLL_INTERVAL_MS,
      candles,
      errors,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
