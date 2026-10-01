/**
 * POST /api/accounts/[id]/stop
 *
 * Stops the bot's EC2 instance (cheap pause; EIP / EBS preserved).
 * Wraps ``./scripts/ops.ps1 <profile> stop``.
 *
 * Mutating: the bot stops trading. Open positions remain on the
 * venue — operator must flatten manually if desired before stop.
 * Confirm in the UI before invoking.
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
  const result = await runOpsCommand(account.id, "stop");
  return NextResponse.json(result);
}
