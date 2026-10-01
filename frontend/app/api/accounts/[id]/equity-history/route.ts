/**
 * GET /api/accounts/[id]/equity-history
 *
 * Returns the bot's most recent ``equity_history/<profile>.json``
 * from S3 — a session-wide time series of equity samples (PnL
 * components + drawdown per sample) that powers the Bot Stats
 * panel's PnL chart.
 *
 * See ``app/equity_history_publisher.py`` (writer) and
 * ``frontend/lib/equity_history.ts`` (reader) for the schema.
 *
 * Cache-Control: no-store. Bot publishes every ~60s.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchEquityHistory } from "@/lib/equity_history";

export const dynamic = "force-dynamic";

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
  const result = await fetchEquityHistory(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
