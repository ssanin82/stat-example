/**
 * GET /api/accounts/[id]/live-stats
 *
 * Returns the bot's most recent ``live_stats/<profile>.json`` from
 * S3 -- richer trading-internals payload than the heartbeat. See
 * ``app/live_stats.py`` (writer) and ``frontend/lib/live_stats.ts``
 * (reader) for the schema.
 *
 * Cache-Control: no-store. The bot writes every ~5s; the dashboard
 * polls at the same cadence (the response carries the recommended
 * interval so the client can dial in if the bot's cadence changed).
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchLiveStats } from "@/lib/live_stats";

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
  const result = await fetchLiveStats(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
