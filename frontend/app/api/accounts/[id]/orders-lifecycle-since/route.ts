/**
 * GET /api/accounts/[id]/orders-lifecycle-since
 *
 * Returns the bot's most recent
 * ``dashboard/orders_lifecycle_since_<profile>.json`` from S3 --
 * session-scoped order rows including the SELECT-time-computed
 * ``placement_to_ack_ms`` / ``cancel_to_close_ms`` /
 * ``lifetime_ms`` latency derivations. Powers todo-032's execution-
 * quality latency histograms and cancel-race diagnostics.
 *
 * See ``app/dashboard_state_publisher.py`` (writer) and
 * ``frontend/lib/dashboard_state.ts`` (reader).
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchOrdersLifecycleSince } from "@/lib/dashboard_state";

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
  const result = await fetchOrdersLifecycleSince(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
