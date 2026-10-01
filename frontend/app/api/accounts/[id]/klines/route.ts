/**
 * GET /api/accounts/[id]/klines?bar=1m&limit=100
 *
 * Public market-data klines for the account's symbol, dispatched
 * to OKX or Binance. Unauthenticated upstream -- both venues expose
 * candles without API keys -- so the dashboard pulls from a single
 * proxy endpoint regardless of which venue the operator selected.
 *
 * Query params:
 *   bar    -- "1m" / "5m" / "15m" / "1h" / "4h" / "1d". OKX uses
 *             uppercase 'H'/'D' but we accept either and normalise.
 *   limit  -- number of candles (default 100, hard cap 500).
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import {
  fetchOkxKlines,
  fetchOkxTicker24h,
  fetchOkxOpenInterest,
  type Candle,
} from "@/lib/okx";
import {
  fetchBinanceKlines,
  fetchBinanceTicker24h,
  fetchBinanceOpenInterest,
} from "@/lib/binance";

export const dynamic = "force-dynamic";

interface MarketSummary {
  /** USDT-notional 24h volume on the venue. */
  vol_ccy_quote_24h: number | null;
  /** 24h price change in bps. */
  change_24h_bps: number | null;
  /** Open interest in USD notional (OKX provides directly; for
   *  Binance we expose only contract count + ``oi_base``). */
  oi_usd: number | null;
  /** Open interest in base-asset units (e.g. SUI count). */
  oi_base: number | null;
}

interface KlineResponse {
  account_id: string;
  venue: string;
  symbol: string;
  bar: string;
  fetched_at_utc: string;
  /** Suggested client poll cadence. Klines update on bar-close cadence,
   *  so a 5s refresh is fine even for 1m bars (the latest bar gets
   *  redrawn ~12 times before it closes; cheap on the venue API). */
  poll_interval_ms: number;
  candles: Candle[];
  /** Companion market-data: 24h volume + open interest for the same
   *  symbol on the same venue. Independent of candle bar; same poll
   *  cadence. Either field may be null on partial venue failure. */
  market_summary: MarketSummary;
  error: string | null;
}

/** Normalize the operator's bar input across venues. */
function normalizeBar(raw: string, venue: string): string {
  const s = raw.trim();
  // OKX accepts: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M
  // Binance: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M
  // Difference: hour/day suffix case. OKX wants H/D, Binance wants h/d.
  if (venue === "okx") return s.replace(/([0-9]+)h$/i, "$1H").replace(/([0-9]+)d$/i, "$1D");
  return s.replace(/([0-9]+)H$/i, "$1h").replace(/([0-9]+)D$/i, "$1d");
}

export async function GET(
  req: Request,
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
  const url = new URL(req.url);
  const rawBar = url.searchParams.get("bar") || "1m";
  const rawLimit = parseInt(url.searchParams.get("limit") || "100", 10);
  const limit = Math.min(
    Math.max(Number.isFinite(rawLimit) ? rawLimit : 100, 1),
    500
  );
  const bar = normalizeBar(rawBar, account.venue);

  const out: KlineResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    bar,
    fetched_at_utc: new Date().toISOString(),
    // 15s: chart shows the in-progress bar; updating faster gives
    // the operator no real signal but does cause a re-fetch of the
    // full 100-candle window every poll. With React.memo on the
    // chart the re-render is cheap, but the network/CPU still adds
    // up over a working day. Bumped from 5s with the perf pass.
    poll_interval_ms: 15_000,
    candles: [],
    market_summary: {
      vol_ccy_quote_24h: null,
      change_24h_bps: null,
      oi_usd: null,
      oi_base: null,
    },
    error: null,
  };

  // Fan out the three reads in parallel — candles + 24h ticker + OI.
  // ``Promise.allSettled`` so a single venue endpoint hiccup doesn't
  // sink the whole panel; we report what landed and leave the rest
  // null. Total wall time = max of the three (typically ~250 ms).
  try {
    if (account.venue === "okx") {
      const [klinesR, tickerR, oiR] = await Promise.allSettled([
        fetchOkxKlines(account.symbol, bar, limit),
        fetchOkxTicker24h(account.symbol),
        fetchOkxOpenInterest(account.symbol),
      ]);
      if (klinesR.status === "fulfilled") {
        out.candles = klinesR.value;
      } else {
        out.error = String(klinesR.reason);
      }
      if (tickerR.status === "fulfilled") {
        out.market_summary.vol_ccy_quote_24h =
          tickerR.value.vol_ccy_quote_24h;
        out.market_summary.change_24h_bps = tickerR.value.change_24h_bps;
      }
      if (oiR.status === "fulfilled") {
        out.market_summary.oi_usd = oiR.value.oi_usd;
        out.market_summary.oi_base = oiR.value.oi_base;
      }
    } else if (account.venue === "binance") {
      const [klinesR, tickerR, oiR] = await Promise.allSettled([
        fetchBinanceKlines(account.symbol, bar, limit),
        fetchBinanceTicker24h(account.symbol),
        fetchBinanceOpenInterest(account.symbol),
      ]);
      if (klinesR.status === "fulfilled") {
        out.candles = klinesR.value;
      } else {
        out.error = String(klinesR.reason);
      }
      if (tickerR.status === "fulfilled") {
        out.market_summary.vol_ccy_quote_24h =
          tickerR.value.vol_ccy_quote_24h;
        out.market_summary.change_24h_bps = tickerR.value.change_24h_bps;
      }
      if (oiR.status === "fulfilled") {
        // Binance OI: contracts==base for USDT-margined perps.
        // USD = base × last; if we have last from ticker, compute it.
        out.market_summary.oi_base = oiR.value.oi_base;
        if (tickerR.status === "fulfilled" && tickerR.value.last > 0) {
          out.market_summary.oi_usd =
            oiR.value.oi_base * tickerR.value.last;
        }
      }
    } else {
      out.error = `unsupported venue: ${account.venue}`;
    }
  } catch (e) {
    out.error = String(e);
  }

  return NextResponse.json(out, {
    headers: { "Cache-Control": "no-store" },
  });
}
