/**
 * POST /api/accounts/[id]/start
 *
 * Starts the bot's EC2 instance (auto-resumes the trading service
 * via systemd). Wraps ``./scripts/ops.ps1 <profile> start``.
 *
 * Mutating: the bot will start trading once the EC2 is up and the
 * service finishes startup reconciliation. Confirm in the UI
 * before invoking.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { runOpsCommand } from "@/lib/ops";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { outcome: "error", detail: `unknown account: ${id}` },
      { status: 404 }
    );
  }
  const result = await runOpsCommand(account.id, "start");
  return NextResponse.json(result);
}
