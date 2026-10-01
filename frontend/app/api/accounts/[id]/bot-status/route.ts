/**
 * GET /api/accounts/[id]/bot-status
 *
 * Returns the EC2 + SSM reachability of the bot's host. Distinct from
 * /api/accounts/[id]/state, which queries OKX/Binance for ground-truth
 * account state -- this endpoint asks AWS "is the box even alive?".
 *
 * Cache-Control: no-store -- the dashboard wants the freshest "up/down"
 * read on every poll. The 10s client cadence is the rate-limit gate.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchBotStatus } from "@/lib/bot_status";

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
  const result = await fetchBotStatus(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
