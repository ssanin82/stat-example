/**
 * GET /api/accounts/[id]/history
 *
 * Returns recent order + fill history for the account's symbol,
 * normalised across venues. Read-only -- prefers READONLY creds
 * for the venue.
 *
 * Query params:
 *   limit -- max rows per category (default 50, hard cap 200)
 *
 * Response shape:
 *   {
 *     account_id, venue, symbol, fetched_at_utc,
 *     orders: OrderHistoryRecord[],
 *     fills:  FillHistoryRecord[],
 *     errors: string[]      -- per-tier failure messages, non-fatal
 *   }
 *
 * Failure modes are SURFACED in `errors[]`, not raised. The
 * dashboard renders whichever tier landed and shows the errors
 * inline.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import {
  fetchBinanceFillsHistory,
  fetchBinanceOrdersHistory,
} from "@/lib/binance";
import {
  fetchOkxFillsHistory,
  fetchOkxOrdersHistory,
} from "@/lib/okx";

export const dynamic = "force-dynamic";

interface HistoryResponse {
  account_id: string;
  venue: string;
  symbol: string;
  fetched_at_utc: string;
  orders: Array<{
    order_id: string;
    client_order_id: string;
    side: string;
    type: string;
    price: number;
    orig_qty: number;
    filled_qty: number;
    status: string;
    time_ms: number;
    notional_usd: number;
  }>;
  fills: Array<{
    trade_id: string;
    order_id: string;
    side: string;
    price: number;
    qty: number;
    notional_usd: number;
    fee: number;
    fee_ccy: string;
    is_maker: boolean;
    realized_pnl: number;
    time_ms: number;
  }>;
  errors: string[];
}

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

  // Parse limit with sane bounds. Operator-tunable but not unbounded
  // (saving venues from a 1000-row query that nobody will read).
  const url = new URL(req.url);
  const rawLimit = parseInt(url.searchParams.get("limit") || "50", 10);
  const limit = Math.min(Math.max(Number.isFinite(rawLimit) ? rawLimit : 50, 1), 200);

  const errors: string[] = [];
  const out: HistoryResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    fetched_at_utc: new Date().toISOString(),
    orders: [],
    fills: [],
    errors,
  };

  try {
    if (account.venue === "binance") {
      const [orders, fills] = await Promise.allSettled([
        fetchBinanceOrdersHistory(account.symbol, limit),
        fetchBinanceFillsHistory(account.symbol, limit),
      ]);
      if (orders.status === "fulfilled") out.orders = orders.value;
      else errors.push(`orders: ${String(orders.reason)}`);
      if (fills.status === "fulfilled") out.fills = fills.value;
      else errors.push(`fills: ${String(fills.reason)}`);
    } else if (account.venue === "okx") {
      const [orders, fills] = await Promise.allSettled([
        fetchOkxOrdersHistory(account.symbol, limit),
        fetchOkxFillsHistory(account.symbol, limit),
      ]);
      if (orders.status === "fulfilled") out.orders = orders.value;
      else errors.push(`orders: ${String(orders.reason)}`);
      if (fills.status === "fulfilled") out.fills = fills.value;
      else errors.push(`fills: ${String(fills.reason)}`);
    } else {
      errors.push(`unsupported venue: ${account.venue}`);
    }
  } catch (e) {
    errors.push(String(e));
  }

  return NextResponse.json(out, {
    headers: { "Cache-Control": "no-store" },
  });
}
