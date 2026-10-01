/**
 * GET /api/accounts/[id]/exposure-since
 *
 * Returns the bot's most recent
 * ``dashboard/exposure_since_<profile>.json`` from S3 -- session-
 * scoped exposure-bar rows (5s cadence) the dashboard's gate /
 * inventory panels read. See ``app/dashboard_state_publisher.py``
 * (writer) and ``frontend/lib/dashboard_state.ts`` (reader).
 *
 * Cache-Control: no-store. Publisher cadence is 30s; dashboard
 * polls at the same cadence (response carries poll_interval_ms).
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchExposureSince } from "@/lib/dashboard_state";

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
  const result = await fetchExposureSince(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
