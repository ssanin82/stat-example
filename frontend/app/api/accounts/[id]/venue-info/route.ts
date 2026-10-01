/**
 * GET /api/accounts/[id]/venue-info
 *
 * One-shot venue-side configuration snapshot for the dashboard's
 * Position panel: leverage, margin mode, account-level position
 * mode. Designed to be called once per page load (not polled) --
 * these values change rarely (operator action via the venue UI),
 * so the dashboard can cache them client-side for the session.
 *
 * Currently OKX-only. Binance / Bluefin / GRVT / Hyperliquid would
 * each need their own venue-side fetch; for those venues we return
 * nulls so the panel renders "—".
 *
 * Cache-Control: no-store anyway -- the operator may change leverage
 * via the OKX UI mid-session and click Refresh to pick up the new
 * value.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { loadIntegrationEnv } from "@/lib/env";
import { fetchOkxVenueInfo } from "@/lib/okx";

export const dynamic = "force-dynamic";

interface VenueInfoResponse {
  account_id: string;
  venue: string;
  symbol: string;
  leverage: string | null;
  margin_mode: string | null;
  position_mode: string | null;
  account_level: string | null;
  // v1.5.260 — instrument metadata for the Reference Data banner.
  // Populated for OKX, null for other venues (per-venue work TBD).
  tick_size: number | null;
  lot_size: number | null;
  min_size: number | null;
  contract_value: number | null;
  fetched_at_utc: string;
  errors: string[];
}

export async function GET(
  _req: Request,
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
  loadIntegrationEnv();

  const errors: string[] = [];
  const out: VenueInfoResponse = {
    account_id: account.id,
    venue: account.venue,
    symbol: account.symbol,
    leverage: null,
    margin_mode: null,
    position_mode: null,
    account_level: null,
    tick_size: null,
    lot_size: null,
    min_size: null,
    contract_value: null,
    fetched_at_utc: new Date().toISOString(),
    errors,
  };

  if (account.venue === "okx") {
    try {
      const info = await fetchOkxVenueInfo(account.symbol);
      out.leverage = info.leverage;
      out.margin_mode = info.margin_mode;
      out.position_mode = info.position_mode;
      out.account_level = info.account_level;
      out.tick_size = info.tick_size;
      out.lot_size = info.lot_size;
      out.min_size = info.min_size;
      out.contract_value = info.contract_value;
    } catch (e) {
      errors.push(`fetch_okx_venue_info: ${String(e)}`);
    }
  } else {
    // Other venues: return nulls (panel renders "—"). Future:
    // Binance leverage via /fapi/v1/leverageBracket etc.
  }

  return NextResponse.json(out, {
    headers: { "Cache-Control": "no-store" },
  });
}
