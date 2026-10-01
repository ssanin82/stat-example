/**
 * GET /api/accounts/[id]/funding-rate-estimate/history?since=<ms>
 *
 * Returns reconstructed HISTORICAL predicted-funding-rate samples for
 * target + reference venues over the requested window. Used by the
 * dashboard's funding-band dashed trace to seed the predicted-rate
 * curve back to session start — so the curve never goes blank after
 * a tab switch / page refresh.
 *
 * Each venue's historical reconstruction uses the public premium-
 * index trajectory (the same proxy each exchange's own UI uses for
 * the visual display):
 *
 *   • Binance — single REST call to ``/fapi/v1/premiumIndexKlines``.
 *     Each kline's ``close`` is the premium index at that minute.
 *   • OKX — two parallel REST calls to ``mark-price-candles`` +
 *     ``index-candles``, joined per minute to compute
 *     ``premium = (mark - index) / index``.
 *
 * Distinct from the SISTER live-snapshot route at
 * ``/api/accounts/[id]/funding-rate-estimate``, which returns the
 * exact TWAP+clamp predicted rate for the IMMEDIATE next settlement
 * (~1 bp accuracy vs venue's authoritative). The history route's
 * samples are the instantaneous premium proxy; they converge to the
 * live route's value at each funding settlement.
 *
 * Architecture: same dispatch pattern as the live route. Adding a
 * new venue = one ``fetch<Venue>FundingRateEstimateHistory`` helper
 * in ``lib/<venue>.ts`` + one ``case`` here.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchOkxFundingRateEstimateHistory } from "@/lib/okx";
import { fetchBinanceFundingRateEstimateHistory } from "@/lib/binance";

export const dynamic = "force-dynamic";

/** Default lookback when caller doesn't supply ``since``. 24 h covers
 *  3 funding intervals plus headroom — enough to keep the dashed
 *  trace populated across any reasonable session length. */
const DEFAULT_LOOKBACK_MS = 24 * 60 * 60 * 1000;

interface HistorySample {
  time_ms: number;
  rate: number;
}

interface VenueHistory {
  venue: string;
  symbol: string;
  samples: HistorySample[];
}

function toBinancePerpSymbol(targetSymbol: string): string {
  return targetSymbol
    .toUpperCase()
    .replace(/-SWAP$/, "")
    .replace(/-/g, "");
}

async function fetchVenueHistory(
  venue: string,
  symbol: string,
  sinceMs: number,
): Promise<HistorySample[]> {
  switch (venue) {
    case "okx":
      return fetchOkxFundingRateEstimateHistory(symbol, sinceMs);
    case "binance":
      return fetchBinanceFundingRateEstimateHistory(symbol, sinceMs);
    default:
      throw new Error(
        `funding_estimate_history_not_implemented_for_venue: ${venue}`,
      );
  }
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
  const target: VenueHistory = {
    venue: account.venue,
    symbol: account.symbol,
    samples: [],
  };
  let reference: VenueHistory = {
    venue: "binance",
    symbol: toBinancePerpSymbol(account.symbol),
    samples: [],
  };

  const fetchTarget = (async () => {
    try {
      target.samples = await fetchVenueHistory(
        account.venue,
        account.symbol,
        sinceMs,
      );
    } catch (e) {
      errors.push(`fetch_target_history: ${String(e)}`);
    }
  })();

  const fetchReference = (async () => {
    if (account.venue === "binance") {
      return; // mirror target after both resolve
    }
    try {
      reference.samples = await fetchVenueHistory(
        "binance",
        reference.symbol,
        sinceMs,
      );
    } catch (e) {
      errors.push(`fetch_reference_history: ${String(e)}`);
    }
  })();

  await Promise.allSettled([fetchTarget, fetchReference]);

  if (account.venue === "binance" && target.samples.length > 0) {
    reference = {
      venue: target.venue,
      symbol: target.symbol,
      samples: target.samples,
    };
  }

  return NextResponse.json(
    {
      profile: account.id,
      target,
      reference,
      since_ms: sinceMs,
      fetched_at_utc: new Date().toISOString(),
      errors,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
