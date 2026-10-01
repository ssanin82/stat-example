/**
 * POST /api/accounts/[id]/snapshot
 *
 * Pulls a full diagnostic snapshot from the bot's EC2 to the
 * laptop's ``snapshots/`` directory. Wraps
 * ``./scripts/ops.ps1 <profile> snapshot``.
 *
 * v1.4.150 — accepts an OPTIONAL ``{ markdown: string }`` POST body.
 * When provided, the markdown is written to the snapshot folder as
 * ``bot-stats-<profile>-<UTC>.md``. The dashboard generates this
 * markdown CLIENT-SIDE from its already-loaded React state (via the
 * shared ``buildComprehensiveMarkdown`` lib) — same code, same data,
 * but it executes in the browser at sub-second speed because it
 * never re-fetches anything. PowerShell-only invocations (without
 * the body) still work; they just produce a snapshot folder
 * without the markdown rollup, same as v1.4.149.
 *
 * Read-only operation against the bot — no trading impact.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { runOpsCommand } from "@/lib/ops";
import * as fs from "node:fs/promises";
import * as path from "node:path";

export const dynamic = "force-dynamic";
// Snapshots can take 30-60 s. Default Next.js API timeout is OK
// but signal the platform we want longer execution if hosted.
export const maxDuration = 120;

/** Parse the snapshot folder path from ops.ps1 stdout.
 *
 * Both PowerShell variants (colo + ec2 paths) print exactly:
 *   ``Snapshot complete: snapshots\v1.4.x-YYMMDD-HHMMSS-<profile>``
 * on success. Returning the LAST match defensively handles a future
 * variant that emits multiple "complete" lines. Returns ``null``
 * when nothing matches.
 */
function parseSnapshotDir(stdout: string): string | null {
  // Match either Windows backslash or POSIX forward-slash variants
  // (the operator might be on either, plus future PowerShell-Core
  // builds for non-Windows). The path captured runs from
  // ``snapshots`` up to the end of the line.
  const re = /^Snapshot complete:\s+(snapshots[\\/][^\r\n]+)$/gm;
  let last: string | null = null;
  let m: RegExpExecArray | null;
  while ((m = re.exec(stdout)) !== null) {
    last = m[1].trim();
  }
  return last;
}

/** Compact-UTC stamp matching the dashboard button's filename
 *  convention (``YYYYMMDDHHMMSS``). */
function compactUtcStamp(d: Date = new Date()): string {
  const z = (n: number) => String(n).padStart(2, "0");
  return (
    d.getUTCFullYear().toString() +
    z(d.getUTCMonth() + 1) +
    z(d.getUTCDate()) +
    z(d.getUTCHours()) +
    z(d.getUTCMinutes()) +
    z(d.getUTCSeconds())
  );
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const account = getAccount(id);
  if (!account) {
    return NextResponse.json(
      { outcome: "error", detail: `unknown account: ${id}` },
      { status: 404 },
    );
  }

  // Optional pre-built markdown body. When present, it's written to
  // the snapshot folder after ops.ps1 succeeds. Body shape is loose
  // so curl / PowerShell / Postman can all hit this endpoint with
  // just `{ "markdown": "..." }`. Missing or non-string `markdown`
  // → behave like the pre-v1.4.150 endpoint (no markdown saved).
  let markdownBody: string | null = null;
  try {
    const ct = req.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const body = (await req.json()) as { markdown?: unknown };
      if (typeof body?.markdown === "string" && body.markdown.length > 0) {
        markdownBody = body.markdown;
      }
    }
  } catch {
    // Bad JSON / no body — fall through to no-markdown path.
  }

  const result = await runOpsCommand(account.id, "snapshot");

  // If ops.ps1 succeeded AND the operator pre-built a markdown,
  // write the markdown into the snapshot folder. Best-effort: any
  // failure here gets surfaced in the response but doesn't fail
  // the overall snapshot (the JSON files are already on disk).
  let markdownSavedPath: string | null = null;
  let markdownSaveError: string | null = null;
  if (result.outcome === "success" && markdownBody !== null) {
    const dir = parseSnapshotDir(result.stdout || "");
    if (!dir) {
      markdownSaveError =
        "could not parse 'Snapshot complete: <path>' from ops.ps1 stdout";
    } else {
      try {
        // Resolve from the repo root (one level up from frontend/).
        const repoRoot = path.resolve(process.cwd(), "..");
        const absDir = path.resolve(repoRoot, dir);
        // Defensive: make sure the parsed path is within
        // ``snapshots/`` and the folder exists — refuse to write
        // anywhere else.
        const snapshotsRoot = path.resolve(repoRoot, "snapshots");
        if (!absDir.startsWith(snapshotsRoot + path.sep)) {
          markdownSaveError = `parsed dir outside snapshots/: ${absDir}`;
        } else {
          await fs.access(absDir);
          const filename = `bot-stats-${id}-${compactUtcStamp()}.md`;
          const fullPath = path.join(absDir, filename);
          await fs.writeFile(fullPath, markdownBody, "utf-8");
          markdownSavedPath = path.relative(repoRoot, fullPath);
        }
      } catch (e) {
        markdownSaveError = `write failed: ${
          e instanceof Error ? e.message : String(e)
        }`;
      }
    }
  }

  return NextResponse.json({
    ...result,
    markdown_saved_path: markdownSavedPath,
    markdown_save_error: markdownSaveError,
  });
}
