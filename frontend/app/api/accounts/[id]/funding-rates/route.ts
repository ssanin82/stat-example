/**
 * GET /api/accounts/[id]/funding-rates?since=<ms>
 *
 * Returns 8 h funding-rate history for both the target venue (where
 * the bot trades) and the cross-reference venue (currently Binance
 * USD-M perp for every active profile — see below). Powers the
 * Funding sub-band on the Session-PnL chart.
 *
 * Source: each venue's public REST endpoint. No auth, no bot
 * involvement. Fetched server-side here so the browser never sees a
 * cross-origin call.
 *
 * Symbol resolution: the target symbol is whatever the account
 * config says (e.g. ``SUI-USDT-SWAP`` on OKX, ``DOGEUSDT`` on
 * Binance-native). The cross-reference symbol is the corresponding
 * USD-M perp on Binance, derived by stripping OKX-style dashes /
 * ``-SWAP`` suffix. If the cross-reference symbol ever needs to
 * differ structurally (e.g. a venue that lists ``SUI-PERP`` without
 * a USDT pair), the bot's ``live_stats.reference_venue.symbol``
 * payload could thread it through here — defer until that case
 * actually appears.
 *
 * Polling cadence: 5 minutes. Funding rates only change every 8 h
 * (00:00 / 08:00 / 16:00 UTC on both venues); polling faster wastes
 * the venues' public-REST budget without surfacing new data.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchOkxFundingRateHistory } from "@/lib/okx";
import { fetchBinanceFundingRateHistory } from "@/lib/binance";

export const dynamic = "force-dynamic";

/** 5 minutes. Funding only flips every 8 h; polling faster wastes
 *  the venues' public-REST budget for zero new info. */
const POLL_INTERVAL_MS = 5 * 60_000;

/** Default lookback if the caller doesn't supply ``since``. 24 h
 *  comfortably covers 3 funding ticks on either venue, enough for
 *  the band to render a step shape even on a short session. */
const DEFAULT_LOOKBACK_MS = 24 * 60 * 60 * 1000;

/** Convert an OKX-style instId (``SUI-USDT-SWAP``) to the equivalent
 *  Binance USD-M perp symbol (``SUIUSDT``). Strips trailing
 *  ``-SWAP`` and any remaining dashes. No-op for symbols already in
 *  Binance format. */
function toBinancePerpSymbol(targetSymbol: string): string {
  return targetSymbol
    .toUpperCase()
    .replace(/-SWAP$/, "")
    .replace(/-/g, "");
}

interface FundingTick {
  time_ms: number;
  rate: number;
}

interface VenueBlock {
  venue: string;
  symbol: string;
  rates: FundingTick[];
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

  const url = new URL(req.url);
  const sinceParam = url.searchParams.get("since");
  const nowMs = Date.now();
  let sinceMs = nowMs - DEFAULT_LOOKBACK_MS;
  if (sinceParam) {
    const parsed = Number(sinceParam);
    if (Number.isFinite(parsed) && parsed > 0 && parsed < nowMs) {
      sinceMs = parsed;
    }
  }

  const errors: string[] = [];
  // ``target`` is only mutated via ``.rates = ...`` inside the async
  // closure below; ``reference`` may be replaced wholesale in the
  // binance-native fallback after both fetches resolve.
  const target: VenueBlock = {
    venue: account.venue,
    symbol: account.symbol,
    rates: [],
  };
  let reference: VenueBlock = {
    venue: "binance",
    symbol: toBinancePerpSymbol(account.symbol),
    rates: [],
  };

  // Kick off both fetches in parallel so the route's latency is
  // bounded by the slower venue, not the sum. ``Promise.allSettled``
  // so one venue failing doesn't suppress the other's data.
  const fetchTarget = (async () => {
    try {
      if (account.venue === "okx") {
        const rates = await fetchOkxFundingRateHistory(
          account.symbol,
          sinceMs,
        );
        target.rates = rates.filter((r) => r.time_ms >= sinceMs);
      } else if (account.venue === "binance") {
        const rates = await fetchBinanceFundingRateHistory(
          account.symbol,
          sinceMs,
        );
        target.rates = rates;
      } else {
        errors.push(
          `target_funding_not_implemented_for_venue: ${account.venue}`,
        );
      }
    } catch (e) {
      errors.push(`fetch_target_funding: ${String(e)}`);
    }
  })();

  const fetchReference = (async () => {
    // Binance-native target: target and reference are the SAME
    // venue + symbol. Skip the second fetch and reuse the result
    // after fetchTarget resolves.
    if (account.venue === "binance") {
      return;
    }
    try {
      const rates = await fetchBinanceFundingRateHistory(
        reference.symbol,
        sinceMs,
      );
      reference.rates = rates;
    } catch (e) {
      errors.push(`fetch_reference_funding: ${String(e)}`);
    }
  })();

  await Promise.allSettled([fetchTarget, fetchReference]);

  // Binance-native target: mirror target → reference so the band
  // still has both lines (they overlap, which is the correct
  // visual answer to "is there a basis-driving funding spread?").
  if (account.venue === "binance" && target.rates.length > 0) {
    reference = {
      venue: target.venue,
      symbol: target.symbol,
      rates: target.rates,
    };
  }

  return NextResponse.json(
    {
      profile: account.id,
      target,
      reference,
      since_ms: sinceMs,
      fetched_at_utc: new Date().toISOString(),
      poll_interval_ms: POLL_INTERVAL_MS,
      errors,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
