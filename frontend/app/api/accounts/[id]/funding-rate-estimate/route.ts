/**
 * GET /api/accounts/[id]/funding-rate-estimate
 *
 * Returns the LIVE predicted next-period funding rate for both
 * target and cross-reference venues. Powers the predicted-rate
 * traces on the Session-PnL chart's Funding sub-band — the "what
 * will be paid next" complement to the settled-rate steps from the
 * sibling ``/funding-rates`` route.
 *
 * Architecture: each ``lib/<venue>.ts`` exports a venue-specific
 * ``fetch<Venue>FundingRateEstimate(symbol)`` that returns the
 * same ``FundingRateEstimate`` shape. This route dispatches on
 * ``account.venue`` (target) and currently uses Binance USD-M as
 * the fixed cross-reference (future venues plug in by extending
 * the switch in ``fetchVenueEstimate``).
 *
 * Why each venue needs its own implementation:
 *
 *   • **OKX** exposes the predicted rate directly via
 *     ``/api/v5/public/funding-rate`` (``nextFundingRate`` field).
 *     Single call.
 *
 *   • **Binance** does NOT expose the predicted rate. The web UI
 *     reconstructs it from premium-index klines + the published
 *     funding-rate formula (see ``fetchBinanceFundingRateEstimate``
 *     in ``lib/binance.ts``). Two calls.
 *
 * Poll cadence: 1 minute. Predicted rate updates ~per second on
 * Binance's UI as the premium index moves, but 1 min sampling is
 * plenty for trajectory-watching during a multi-hour session
 * without burning REST budget.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchOkxFundingRateEstimate } from "@/lib/okx";
import { fetchBinanceFundingRateEstimate } from "@/lib/binance";

export const dynamic = "force-dynamic";

/** 1 minute. Binance UI updates ~per second; we sample slower
 *  because the trajectory matters more than the per-tick noise. */
const POLL_INTERVAL_MS = 60_000;

interface VenueEstimate {
  venue: string;
  symbol: string;
  predicted_rate: number;
  next_funding_ms: number;
  funding_interval_ms: number;
  method: string;
  fetched_at_ms: number;
}

/** Convert an OKX-style instId (``SUI-USDT-SWAP``) to the equivalent
 *  Binance USD-M perp symbol (``SUIUSDT``). Mirrors the helper in
 *  the sibling ``funding-rates`` route — kept local so this route
 *  is self-contained. */
function toBinancePerpSymbol(targetSymbol: string): string {
  return targetSymbol
    .toUpperCase()
    .replace(/-SWAP$/, "")
    .replace(/-/g, "");
}

/** Venue-agnostic dispatch. Add new venues here by importing their
 *  ``fetchXxxFundingRateEstimate`` and extending the switch. */
async function fetchVenueEstimate(
  venue: string,
  symbol: string,
): Promise<VenueEstimate> {
  switch (venue) {
    case "okx": {
      const r = await fetchOkxFundingRateEstimate(symbol);
      return { venue, symbol, ...r };
    }
    case "binance": {
      const r = await fetchBinanceFundingRateEstimate(symbol);
      return { venue, symbol, ...r };
    }
    default:
      throw new Error(
        `funding_estimate_not_implemented_for_venue: ${venue}`,
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

  const errors: string[] = [];
  let target: VenueEstimate | null = null;
  let reference: VenueEstimate | null = null;

  // Reference venue: currently always Binance USD-M for non-Binance
  // targets. When target IS Binance, we deliberately don't fetch
  // twice — set ``reference = target`` post-resolution so the band
  // still shows both traces overlapping (the visual answer to "is
  // there a cross-venue spread?" is "no, same venue").
  const refVenue = "binance";
  const refSymbol =
    account.venue === "binance"
      ? account.symbol
      : toBinancePerpSymbol(account.symbol);

  const fetchTarget = fetchVenueEstimate(account.venue, account.symbol)
    .then((r) => {
      target = r;
    })
    .catch((e) => {
      errors.push(`fetch_target_estimate: ${String(e)}`);
    });

  const fetchReference =
    account.venue === "binance"
      ? Promise.resolve()
      : fetchVenueEstimate(refVenue, refSymbol)
          .then((r) => {
            reference = r;
          })
          .catch((e) => {
            errors.push(`fetch_reference_estimate: ${String(e)}`);
          });

  await Promise.allSettled([fetchTarget, fetchReference]);

  // Binance-native target: mirror so the band still has both lines.
  if (account.venue === "binance" && target !== null) {
    reference = target;
  }

  return NextResponse.json(
    {
      profile: account.id,
      target,
      reference,
      fetched_at_utc: new Date().toISOString(),
      poll_interval_ms: POLL_INTERVAL_MS,
      errors,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
