/**
 * POST /api/accounts/[id]/deploy
 *
 * Pulls the latest commit on the EC2 and restarts the trading
 * service. Wraps ``./scripts/ops.ps1 <profile> deploy``.
 *
 * Mutating: the bot picks up new code (binary, config, schema).
 * Brief gap in trading during the systemd restart. Confirm in
 * the UI before invoking. Push your commit before clicking.
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
  const result = await runOpsCommand(account.id, "deploy");
  return NextResponse.json(result);
}
