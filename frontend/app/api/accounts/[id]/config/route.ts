/**
 * GET /api/accounts/[id]/config
 *
 * Returns the bot's most recent ``config/<profile>.json`` from S3 —
 * a one-shot snapshot of the bot's effective config written at
 * startup. Powers the Bot Stats panel's CONFIG tab.
 *
 * See ``app/config_publisher.py`` (writer) and
 * ``frontend/lib/config.ts`` (reader) for the schema.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchConfig } from "@/lib/config";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { error: `unknown account: ${id}` },
      { status: 404 },
    );
  }
  const result = await fetchConfig(account.id);
  return NextResponse.json(result, {
    headers: { "Cache-Control": "no-store" },
  });
}
