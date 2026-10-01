/**
 * GET /api/accounts
 *
 * Returns the list of bot profiles available to the dashboard,
 * derived from `config/profiles/*.env` at the repo root. Each
 * entry has the stable `id` (filename without `.env`), a
 * human-readable label, the venue, and the configured symbol.
 *
 * No credentials are read or surfaced -- this is the discovery
 * endpoint the picker dropdown polls at startup.
 */

import { NextResponse } from "next/server";
import { listAccounts } from "@/lib/accounts";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json({ accounts: listAccounts() });
  } catch (e) {
    return NextResponse.json(
      { accounts: [], error: String(e) },
      { status: 500 }
    );
  }
}
