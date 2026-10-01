/**
 * POST /api/accounts/[id]/cancel-all
 *
 * Cancels every open order on the account's symbol. Trading-grade
 * credentials required (read-only keys cannot mutate state).
 *
 * Returns ``{ outcome, detail, affected? }`` per ``ActionResult``.
 * Failure modes are SURFACED in the response body, not raised --
 * the dashboard renders the result either way.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { cancelAllBinanceOrders, hasBinanceTradingCreds } from "@/lib/binance";
import { cancelAllOkxOrders, hasOkxTradingCreds } from "@/lib/okx";

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

  // Capability gate: write actions need trading-grade creds. The UI
  // already greys the button when this is false, but we re-check
  // server-side -- never trust the client.
  if (account.venue === "binance" && !hasBinanceTradingCreds()) {
    return NextResponse.json(
      {
        outcome: "error",
        detail:
          "Binance trading-grade credentials not in env (BINANCE_API_KEY / BINANCE_API_SECRET). " +
          "Read-only keys cannot cancel orders.",
      },
      { status: 412 }
    );
  }
  if (account.venue === "okx" && !hasOkxTradingCreds()) {
    return NextResponse.json(
      {
        outcome: "error",
        detail:
          "OKX trading-grade credentials not in env. Read-only keys cannot cancel orders.",
      },
      { status: 412 }
    );
  }

  try {
    const result =
      account.venue === "binance"
        ? await cancelAllBinanceOrders(account.symbol)
        : account.venue === "okx"
        ? await cancelAllOkxOrders(account.symbol)
        : { outcome: "error" as const, detail: `unsupported venue: ${account.venue}` };

    // Audit-log to stderr -- the operator's npm-run-dev terminal is
    // the running record of mutating actions. No persistent log needed
    // at v1 scale (single operator, ~1 action per session).
    // eslint-disable-next-line no-console
    console.error(
      `[action] cancel-all account=${account.id} venue=${account.venue} ` +
        `symbol=${account.symbol} outcome=${result.outcome} ` +
        `detail="${result.detail}" affected=${result.affected ?? "n/a"}`
    );

    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { outcome: "error", detail: String(e) },
      { status: 500 }
    );
  }
}
