/**
 * POST /api/accounts/[id]/close-position
 *
 * Closes (flattens) the account's open position on its symbol, via
 * a reduce-only MARKET order. Trading-grade credentials required.
 *
 * Returns ``{ outcome, detail }`` per ``ActionResult``. ``noop`` if
 * already flat.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { closeBinancePosition, hasBinanceTradingCreds } from "@/lib/binance";
import { closeOkxPosition, hasOkxTradingCreds } from "@/lib/okx";

export const dynamic = "force-dynamic";

export async function POST(
  _req: Request,
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

  if (account.venue === "binance" && !hasBinanceTradingCreds()) {
    return NextResponse.json(
      {
        outcome: "error",
        detail:
          "Binance trading-grade credentials not in env. Read-only keys cannot close positions.",
      },
      { status: 412 }
    );
  }
  if (account.venue === "okx" && !hasOkxTradingCreds()) {
    return NextResponse.json(
      {
        outcome: "error",
        detail:
          "OKX trading-grade credentials not in env. Read-only keys cannot close positions.",
      },
      { status: 412 }
    );
  }

  try {
    const result =
      account.venue === "binance"
        ? await closeBinancePosition(account.symbol)
        : account.venue === "okx"
        ? await closeOkxPosition(account.symbol)
        : { outcome: "error" as const, detail: `unsupported venue: ${account.venue}` };

    // eslint-disable-next-line no-console
    console.error(
      `[action] close-position account=${account.id} venue=${account.venue} ` +
        `symbol=${account.symbol} outcome=${result.outcome} ` +
        `detail="${result.detail}"`
    );

    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { outcome: "error", detail: String(e) },
      { status: 500 }
    );
  }
}
