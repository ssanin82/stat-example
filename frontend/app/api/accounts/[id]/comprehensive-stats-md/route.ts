/**
 * GET /api/accounts/[id]/comprehensive-stats-md
 *
 * Returns the same markdown the dashboard's "Download stats" button
 * produces — but over HTTP, so it can be saved by the snapshot
 * pipeline (`scripts/colo_fetch_bot_snapshot.ps1` /
 * `scripts/fetch_bot_snapshot.ps1`) immediately after a snapshot
 * completes.
 *
 * Implementation: in-parallel call every data source the dashboard
 * uses (existing API routes, served by this same Next.js server),
 * then invoke the SHARED `buildComprehensiveMarkdown` function from
 * `@/lib/comprehensive_markdown` — the SAME function the browser-
 * side button calls. ONE source of truth.
 *
 * Response: `text/markdown` body, attachment Content-Disposition
 * with the canonical `bot-stats-<profile>-<UTC>.md` filename.
 */

import { NextResponse } from "next/server";
import { getAccount } from "@/lib/accounts";
import { fetchBotStatus } from "@/lib/bot_status";
import { fetchLiveStats } from "@/lib/live_stats";
import { fetchEquityHistory } from "@/lib/equity_history";
import { fetchConfig } from "@/lib/config";
import {
  fetchExposureSince,
  fetchFillsSince,
  fetchOrdersLifecycleSince,
} from "@/lib/dashboard_state";
import {
  buildComprehensiveMarkdown,
  compactUtcStamp,
} from "@/lib/comprehensive_markdown";

export const dynamic = "force-dynamic";
// Pulling ~10 sources in parallel takes 2-5 s on the colo path;
// give a generous ceiling so a single slow S3 GET doesn't fail
// the whole markdown render.
export const maxDuration = 60;

async function _fetchJsonOrNull<T>(
  origin: string,
  path: string,
): Promise<T | null> {
  try {
    const r = await fetch(`${origin}${path}`, {
      cache: "no-store",
      // Internal localhost call — no auth needed in the dev/local
      // server context where this route runs.
    });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    // Network glitches / route returning 5xx: surface as null so
    // the markdown builder's "data missing" fallback fires per-
    // section instead of failing the whole rollup.
    return null;
  }
}

export async function GET(
  req: Request,
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

  // Same origin we're running on — internal fetch path. URL form
  // works because Next.js mounts every route on this same server.
  const origin = new URL(req.url).origin;

  // Parallel fetch of every data source the dashboard's
  // onDownloadStats callback reads (see frontend/app/page.tsx,
  // search for `onDownloadStats`). Each one is independent so
  // they run concurrently. Failed fetches return null and the
  // markdown builder handles them gracefully per-section.
  const [
    state,
    botStatus,
    liveStatsResult,
    equityHistoryResult,
    configResult,
    history,
    volume,
    venueInfo,
    regimeFills,
    regimeBars,
    ordersLifecycle,
  ] = await Promise.all([
    _fetchJsonOrNull<any>(origin, `/api/accounts/${id}/state`),
    fetchBotStatus(id),
    fetchLiveStats(id),
    fetchEquityHistory(id),
    fetchConfig(id),
    _fetchJsonOrNull<any>(origin, `/api/accounts/${id}/history`),
    _fetchJsonOrNull<any>(origin, `/api/accounts/${id}/volume`),
    _fetchJsonOrNull<any>(origin, `/api/accounts/${id}/venue-info`),
    fetchFillsSince(id),
    fetchExposureSince(id),
    fetchOrdersLifecycleSince(id),
  ]);

  // The button computes `accountRisk` from the Account record's
  // ``risk`` field (parsed from the profile env on
  // /api/accounts boot). Same source here.
  //
  // v1.5.135 (Codex bug #3) — pre-fix this read
  // ``(account as any).riskThresholds`` which always resolved to
  // ``undefined`` (the field is named ``risk``, see
  // ``frontend/lib/accounts.ts``'s ``Account`` interface). Net
  // effect was the downloaded markdown silently dropped the risk
  // block even though the live dashboard shows it. The ``any``
  // cast was both unnecessary and exactly the kind of thing
  // ``@ts-nocheck`` in ``comprehensive_markdown.ts`` was masking
  // (fixed under bug #9 in the same change).
  const accountRisk = account.risk ?? null;

  // Defensive: the builder dereferences `state.account_id` /
  // `state.position` early; can't build anything if /state itself
  // failed. Return a stub markdown rather than crashing.
  if (!state) {
    const stub =
      `# Snapshot markdown — unavailable\n\n` +
      `> The dashboard's \`/api/accounts/${id}/state\` endpoint did ` +
      `not return data. Check that the bot is reachable from this host.\n`;
    return new Response(stub, {
      status: 503,
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  }

  const sessionStartedAtUtc =
    botStatus?.bot_session_started_at_utc ?? null;

  // Call the SHARED function — same one the dashboard's button uses.
  const md = buildComprehensiveMarkdown({
    state,
    liveStats: liveStatsResult?.payload ?? null,
    botStatus,
    volume,
    venueInfo,
    accountRisk,
    history,
    equityHistory: equityHistoryResult?.payload ?? null,
    configResult,
    sessionStartedAtUtc,
    regimeFills,
    regimeBars,
    ordersLifecycle,
  });

  // Canonical filename: `bot-stats-<profile>-<compactUtcStamp>.md`
  // — same shape the dashboard's button writes when the operator
  // clicks Download. Snapshot script will save the response body
  // under this name directly.
  const filename = `bot-stats-${id}-${compactUtcStamp()}.md`;

  return new Response(md, {
    status: 200,
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Cache-Control": "no-store",
      "X-Suggested-Filename": filename,
    },
  });
}
