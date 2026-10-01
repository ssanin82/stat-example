/**
 * GET /api/accounts/[id]/open-orders
 *
 * Slim, fast-poll endpoint that returns ONLY the currently-resting
 * orders for the account's symbol. Designed to be polled at 1s so
 * the operator gets a live "the bot is quoting" heartbeat as orders
 * flow in/out of the book.
 *
 * Why a separate endpoint vs. a flag on /state:
 *   - The full /state response includes account/positions/etc which
 *     are slow-drifting and not worth fetching every second
 *   - Splitting cleanly bounds the rate-limit cost of fast polling:
 *
 *     OKX (MM-tier, 1200 req/2s):
 *       1 req/s * 2 = 2 req/2s = 0.17% of budget
 *
 *     Binance Futures (1200 weight/min, 1 weight per call):
 *       1 weight/s = 60 weight/min = 5% of budget
 *
 *   - Both venues are comfortably safe at 1s cadence on this single
 *     endpoint. Combined with /state at 5s and the bot's own usage,
 *     total dashboard load is well under any limit.
 *
 * Read-only -- prefers READONLY creds for the venue when present.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { fetchBinanceOpenOrders } from "@/lib/binance";
import { fetchOkxOpenOrders } from "@/lib/okx";

export const dynamic = "force-dynamic";

interface OpenOrdersResponse {
  account_id: string;
  venue: string;
  symbol: string;
  fetched_at_utc: string;
  poll_interval_ms: number;
  open_orders: Array<{
    side: string;
    price: number;
    qty: number;
    order_id: string;
    client_order_id: string;
    state: string;
    time_ms: number;
  }>;
  /**
   * True iff ``open_orders`` was returned by the venue (zero or more
   * rows). False when the lib threw (venue error, rate-limit, network).
   * Dashboard MUST gate "no orders to cancel" affordances on this flag --
   * pre-2026-05-16 the fail-open lib silently returned ``[]`` and the
   * dashboard could not distinguish "venue says empty" from "we don't
   * know". Codex review #3.
   */
  open_orders_ok: boolean;
  errors: string[];
}

export async function GET(
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

  const errors: string[] = [];
  const out: OpenOrdersResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: 1000,
    open_orders: [],
    // Optimistic default: flipped to false on any caught error below.
    open_orders_ok: true,
    errors,
  };

  try {
    if (account.venue === "okx") {
      const opens = await fetchOkxOpenOrders(account.symbol);
      out.open_orders = opens.map((o) => ({
        side: o.side,
        price: o.px,
        qty: o.sz_contracts,
        order_id: o.ord_id,
        client_order_id: o.cl_ord_id,
        state: o.state,
        time_ms: o.c_time_ms,
      }));
    } else if (account.venue === "binance") {
      const opens = await fetchBinanceOpenOrders(account.symbol);
      out.open_orders = opens.map((o) => ({
        side: o.side,
        price: o.price,
        qty: o.orig_qty,
        order_id: String(o.order_id),
        client_order_id: o.client_order_id,
        state: o.status,
        time_ms: o.time_ms,
      }));
    } else {
      errors.push(`unsupported venue: ${account.venue}`);
      out.open_orders_ok = false;
    }
  } catch (e) {
    errors.push(String(e));
    out.open_orders_ok = false;
  }

  // Status: 502 when the only data this endpoint serves failed to
  // fetch -- callers should NOT treat the empty ``open_orders`` field
  // as ground truth. The body still carries diagnostic ``errors``.
  // Codex review #3, 2026-05-16.
  const status = out.open_orders_ok ? 200 : 502;
  return NextResponse.json(out, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}
