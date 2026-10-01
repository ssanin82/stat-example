/**
 * GET /api/accounts/[id]/state
 *
 * Returns a venue-agnostic shape with current account state +
 * position + open orders for the given account id. Dispatches by
 * venue to the right adapter (`lib/okx.ts` or `lib/binance.ts`).
 *
 * Cache-Control: no-store -- the bot's account state must always be
 * the freshest available. The dashboard's auto-refresh handles
 * cadence on the client side.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import {
  fetchOkxAccount,
  fetchOkxPositions,
  hasOkxTradingCreds,
} from "@/lib/okx";
import {
  fetchBinanceAccount,
  fetchBinancePosition,
  hasBinanceTradingCreds,
} from "@/lib/binance";

export const dynamic = "force-dynamic";

interface NormalizedState {
  account_id: string;
  venue: string;
  symbol: string;
  fetched_at_utc: string;
  /**
   * Recommended client poll interval in milliseconds for the FULL
   * state response. 5000ms across both venues -- slow-drifting fields
   * (equity, cash, withdrawable, position) don't need faster cadence
   * and the rate-limit cost would be wasteful. Open orders, which DO
   * change every quote cycle, are served by a separate fast endpoint
   * at /api/accounts/[id]/open-orders polled at 1s.
   */
  poll_interval_ms: number;
  /** Capabilities the dashboard UI uses to gate write buttons. */
  capabilities: {
    can_cancel_orders: boolean;
    can_close_position: boolean;
  };
  account: {
    equity_usd: number | null;
    cash_usd: number | null;
    withdrawable_usd: number | null;
    unrealized_pnl_usd?: number;
    extra?: Record<string, unknown>;
  };
  position: {
    symbol: string;
    qty: number;
    avg_entry_price: number | null;
    mark_price: number | null;
    notional_usd: number;
    unrealized_pnl_usd: number;
  };
  /**
   * v1.4.43 rate-limit-normalization Tier 1.1: open-orders are no
   * longer fetched in /state. The dedicated /open-orders endpoint
   * (polled every 5 s by the dashboard) is the single source of truth
   * for the orders table. Pre-1.4.43 /state did a parallel fan-out of
   * 3 OKX reads (balance + positions + open-orders) every 5 s, with
   * the open-orders read fully duplicated by the dedicated loop —
   * burning 0.2 reads/s of the OKX `reads` rate-limit pool for no
   * informational gain. The field is kept (always empty + ok=true)
   * so existing consumers don't crash; they should read open-orders
   * from the /open-orders endpoint's response instead.
   */
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
   * True iff the open-orders sub-call returned data. v1.4.43: always
   * ``true`` here because the sub-call no longer runs in this route.
   * The dashboard's cancel-all affordance gating reads
   * ``open_orders_ok`` from the /open-orders endpoint's response
   * (spliced into the same state object via ``refreshOpenOrders`` in
   * `page.tsx`), so the gate continues to work end-to-end.
   */
  open_orders_ok: boolean;
  errors: string[];
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  // Load credentials from tests/integration/.env before any signed
  // call. Idempotent + cheap on the cached path.
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
  // Capabilities reflect the env at request time -- the operator can
  // edit tests/integration/.env between page polls and see the
  // buttons enable/disable on the next refresh. ``loadIntegrationEnv``
  // uses an mtime-keyed cache, so it reparses the file iff it changed
  // since the last load; saved edits propagate without a dev-server
  // restart.
  const hasTrading =
    account.venue === "binance"
      ? hasBinanceTradingCreds()
      : account.venue === "okx"
      ? hasOkxTradingCreds()
      : false;
  // The full state endpoint polls slowly (5s) -- equity / cash /
  // withdrawable / position drift on minute timescales, no operator
  // value in faster cadence. The fast-changing data (open orders)
  // moved to its own endpoint at /api/accounts/[id]/open-orders
  // polled at 1s. See its route handler for rationale.
  const pollIntervalMs = 5000;
  const out: NormalizedState = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    fetched_at_utc: new Date().toISOString(),
    poll_interval_ms: pollIntervalMs,
    capabilities: {
      can_cancel_orders: hasTrading,
      can_close_position: hasTrading,
    },
    account: {
      equity_usd: null,
      cash_usd: null,
      withdrawable_usd: null,
    },
    position: {
      symbol: account.symbol,
      qty: 0,
      avg_entry_price: null,
      mark_price: null,
      notional_usd: 0,
      unrealized_pnl_usd: 0,
    },
    open_orders: [],
    // Optimistic; flipped to false below when the open-orders sub-call
    // rejects or the venue is unsupported.
    open_orders_ok: true,
    errors,
  };

  try {
    if (account.venue === "okx") {
      // v1.4.43 rate-limit Tier 1.1: open-orders fetch removed from
      // this route. Only account + positions are fetched here now.
      // Open-orders comes from /api/accounts/[id]/open-orders polled
      // on its own faster cadence. Savings: ~0.2 reads/s in the OKX
      // `reads` pool per dashboard tab.
      const [acc, pos] = await Promise.allSettled([
        fetchOkxAccount(),
        fetchOkxPositions(account.symbol),
      ]);
      if (acc.status === "fulfilled") {
        out.account.equity_usd = acc.value.equity_usd;
        out.account.cash_usd = acc.value.cash_usd;
        out.account.withdrawable_usd = acc.value.withdrawable_usd;
        out.account.extra = {
          uid: acc.value.uid,
          details: acc.value.details,
        };
      } else {
        errors.push(`account: ${String(acc.reason)}`);
      }
      if (pos.status === "fulfilled") {
        out.position = {
          symbol: pos.value.symbol,
          qty: pos.value.position_qty,
          avg_entry_price: pos.value.avg_entry_price,
          mark_price: pos.value.mark_price,
          notional_usd: pos.value.notional_usd,
          unrealized_pnl_usd: pos.value.unrealized_pnl_usd,
        };
      } else {
        errors.push(`position: ${String(pos.reason)}`);
      }
    } else if (account.venue === "binance") {
      // v1.4.43 rate-limit Tier 1.1: open-orders fetch removed
      // (mirror of OKX change above). Binance has its own per-IP
      // rate budget but the same dashboard split benefits both
      // venues — single source of truth for orders is the dedicated
      // /open-orders endpoint.
      const [acc, pos] = await Promise.allSettled([
        fetchBinanceAccount(),
        fetchBinancePosition(account.symbol),
      ]);
      if (acc.status === "fulfilled") {
        out.account.equity_usd = acc.value.equity_usd;
        out.account.cash_usd = acc.value.cash_usd;
        out.account.withdrawable_usd = acc.value.withdrawable_usd;
        out.account.unrealized_pnl_usd = acc.value.unrealized_pnl_usd;
        out.account.extra = { assets: acc.value.assets };
      } else {
        errors.push(`account: ${String(acc.reason)}`);
      }
      if (pos.status === "fulfilled") {
        out.position = {
          symbol: pos.value.symbol,
          qty: pos.value.position_qty,
          avg_entry_price: pos.value.avg_entry_price,
          mark_price: pos.value.mark_price,
          notional_usd: pos.value.notional_usd,
          unrealized_pnl_usd: pos.value.unrealized_pnl_usd,
        };
      } else {
        errors.push(`position: ${String(pos.reason)}`);
      }
    } else {
      errors.push(`unsupported venue: ${account.venue}`);
      // v1.4.43: open_orders_ok used to flip false here as the
      // "anything else is suspect" signal. Now that this route doesn't
      // serve open-orders, leave the flag at its initialized ``true``
      // — the /open-orders endpoint owns that flag. Account / position
      // accuracy is reflected in the ``errors`` array.
    }
  } catch (e) {
    errors.push(String(e));
    // v1.4.43: catch-all path no longer flips open_orders_ok — that
    // flag is owned by the /open-orders endpoint now. account /
    // position consumers should read ``errors`` for fault signals.
  }

  return NextResponse.json(out, {
    headers: { "Cache-Control": "no-store" },
  });
}
