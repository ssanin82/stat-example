/**
 * GET /api/accounts/[id]/fills-since
 *
 * Returns the bot's most recent
 * ``dashboard/fills_since_<profile>.json`` from S3 -- session-scoped
 * fills with full Phase 1-4c enrichment. Powers the inventory-
 * conditional markout / regime-attribution / execution-quality
 * panels.
 *
 * See ``app/dashboard_state_publisher.py`` (writer) and
 * ``frontend/lib/dashboard_state.ts`` (reader).
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchFillsSince } from "@/lib/dashboard_state";

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
  const result = await fetchFillsSince(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
