/**
 * CONFIG tab — display-only listing of every parameter the bot is
 * currently running with, sourced from the env file the bot recorded
 * at startup (``app/config_publisher.py``).
 *
 * Extracted from ``app/page.tsx`` 2026-05-13 so it can be code-split
 * via ``next/dynamic`` — see plans/frontend.md Rec #3. The panel is
 * only ever rendered when the operator clicks the CONFIG tab, so
 * shipping it in the initial bundle was paying ~10-15 KB of gzipped
 * JS the operator usually never executes.
 *
 * Self-contained: every helper used by ConfigPanel lives in this
 * file. The only inputs are:
 *   - ``ConfigResult`` type (re-exported from ``app/page.tsx``)
 *   - the ``React`` runtime
 *
 * Resilient default: when ``env_file_raw`` is missing (bot fed
 * config via process env instead of a file), the panel falls back to
 * the ``resolved_settings`` dict — same one used by Telegram /status.
 */

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ConfigResult } from "@/lib/types";

interface ConfigRow {
  key: string;
  value: string;
  comment: string;
}

/** Display-only CONFIG tab: lists every parameter from the bot's
 *  loaded env file with its preceding comment block as a tooltip.
 *  Falls back to ``resolved_settings`` if the raw file is missing
 *  (e.g. APP_ENV_FILE not set; bot fed config via process env). */
export default function ConfigPanel({
  configResult,
}: {
  configResult: ConfigResult | null;
}) {
  const [filter, setFilter] = useState("");
  if (!configResult) {
    return (
      <div className="text-sm text-gray-500 font-mono">loading…</div>
    );
  }
  const payload = configResult.payload;
  if (!payload) {
    const why =
      configResult.errors.length > 0
        ? configResult.errors[0]
        : "no config file published yet (bot may be on a version older than 1.1.87 or hasn't started since deploy)";
    return (
      <div className="text-sm text-gray-400 font-mono">
        Config not available — {why}
      </div>
    );
  }
  const allRows = parseEnvFileRaw(payload.env_file_raw || "");
  const resolved = payload.resolved_settings ?? {};
  const needle = filter.trim().toLowerCase();
  const rows = needle
    ? allRows.filter(
        (r) =>
          r.key.toLowerCase().includes(needle) ||
          r.value.toLowerCase().includes(needle),
      )
    : allRows;
  return (
    <div className="flex flex-col gap-3 text-xs font-mono">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-gray-500">
        <span>
          profile:{" "}
          <span className="text-gray-300">{payload.profile}</span>
        </span>
        <span>
          version:{" "}
          <span className="text-gray-300">{payload.version ?? "—"}</span>
        </span>
        <span>
          symbol:{" "}
          <span className="text-gray-300">{payload.symbol ?? "—"}</span>
        </span>
        <span title={payload.env_file_path}>
          env_file:{" "}
          <span className="text-gray-300">
            {payload.env_file_path
              ? payload.env_file_path.split(/[\\/]/).pop()
              : "—"}
          </span>
        </span>
        <span>
          captured:{" "}
          <span className="text-gray-300">
            {new Date(payload.captured_at_utc).toLocaleString()}
          </span>
        </span>
      </div>
      {payload.env_file_read_error && (
        <div className="text-accent-amber">
          env_file read error: {payload.env_file_read_error} (showing
          resolved settings only)
        </div>
      )}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter (key or value)…"
          className="bg-bg-card border border-bg-border rounded px-2 py-1 text-xs font-mono text-gray-100 w-64 focus:outline-none focus:border-accent-amber/60"
        />
        <span className="text-gray-500">
          {rows.length}
          {needle ? ` of ${allRows.length}` : ""} parameter
          {rows.length === 1 ? "" : "s"}
        </span>
      </div>
      {allRows.length === 0 ? (
        <ResolvedSettingsList resolved={resolved} filter={needle} />
      ) : (
        <ConfigRowList rows={rows} />
      )}
    </div>
  );
}

function parseEnvFileRaw(raw: string): ConfigRow[] {
  const rows: ConfigRow[] = [];
  let pendingComment: string[] = [];
  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    const stripped = line.trim();
    if (!stripped) {
      pendingComment = [];
      continue;
    }
    if (stripped.startsWith("#")) {
      // Strip the leading ``#`` (and one optional space) so tooltips
      // read as plain prose.
      const text = stripped.replace(/^#\s?/, "");
      pendingComment.push(text);
      continue;
    }
    const eq = stripped.indexOf("=");
    if (eq <= 0) {
      pendingComment = [];
      continue;
    }
    const key = stripped.slice(0, eq).trim();
    const value = stripped.slice(eq + 1).trim();
    rows.push({ key, value, comment: pendingComment.join("\n") });
    pendingComment = [];
  }
  return rows;
}

/** Hook: resizable key-column width with localStorage persistence.
 *  ``storageKey`` lets each list keep its own preferred width.
 *
 *  Returned ``dividerProps`` go on a small <div> between the key
 *  column and the value column. Pointer events use setPointerCapture
 *  so the drag continues even if the cursor leaves the divider —
 *  the standard pattern for a column splitter. */
function useResizableColumn(
  initialPx: number,
  storageKey: string,
  minPx: number = 120,
  maxPx: number = 720,
): {
  width: number;
  dividerProps: React.HTMLAttributes<HTMLDivElement>;
} {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === "undefined") return initialPx;
    try {
      const v = window.localStorage.getItem(storageKey);
      const n = v ? parseInt(v, 10) : NaN;
      if (Number.isFinite(n) && n >= minPx && n <= maxPx) return n;
    } catch {
      // best-effort
    }
    return initialPx;
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(storageKey, String(width));
    } catch {
      // best-effort
    }
  }, [storageKey, width]);

  // Drag-state ref: the currently-pointer-captured element + its
  // ID + the (clientX, width) baseline at pointerdown. Stored on
  // the hook (not on each divider) so the same drag mechanics work
  // for ANY of the per-row dividers — the user might grab any row.
  const dragStartRef = useRef<{
    el: HTMLDivElement;
    pointerId: number;
    x: number;
    w: number;
  } | null>(null);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const el = e.currentTarget;
      el.setPointerCapture(e.pointerId);
      dragStartRef.current = {
        el,
        pointerId: e.pointerId,
        x: e.clientX,
        w: width,
      };
      e.preventDefault();
    },
    [width],
  );
  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const start = dragStartRef.current;
      if (!start) return;
      const next = Math.min(maxPx, Math.max(minPx, start.w + (e.clientX - start.x)));
      setWidth(next);
    },
    [minPx, maxPx],
  );
  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const start = dragStartRef.current;
      if (start && start.el.hasPointerCapture(start.pointerId)) {
        start.el.releasePointerCapture(start.pointerId);
      }
      dragStartRef.current = null;
    },
    [],
  );

  return {
    width,
    dividerProps: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
      role: "separator",
      "aria-orientation": "vertical",
      title: "Drag to resize",
      className:
        "shrink-0 w-1 -mx-0.5 cursor-col-resize bg-bg-border/40 " +
        "hover:bg-accent-blue/60 active:bg-accent-blue/80 " +
        "transition-colors self-stretch",
    },
  };
}

/** Flex-row layout: fixed-width key column on the left so values
 *  begin right next to it instead of being pushed to the far right
 *  by a stretched table cell. Hover any row to see the .env comment
 *  block in a tooltip. The column is resizable via a draggable
 *  divider; preferred width persists in localStorage. */
function ConfigRowList({ rows }: { rows: ConfigRow[] }) {
  const { width: keyColWidth, dividerProps } = useResizableColumn(
    320,
    "dtc-mm-as.config-row-key-width",
  );
  if (rows.length === 0) {
    return (
      <div className="text-gray-500">No parameters match the filter.</div>
    );
  }
  return (
    <div className="flex flex-col">
      {rows.map((r, idx) => {
        const tooltip = r.comment || undefined;
        return (
          <div
            key={idx}
            title={tooltip}
            className="flex items-stretch gap-3 py-1 border-b border-bg-border/50 hover:bg-bg-card/40"
          >
            <div
              style={{ width: keyColWidth }}
              className="shrink-0 text-gray-300 break-all leading-tight"
            >
              {r.key}
              {tooltip && (
                <span className="ml-1 text-gray-600" aria-hidden="true">
                  ⓘ
                </span>
              )}
            </div>
            <div {...dividerProps} />
            <div className="flex-1 text-gray-100 break-all">
              {r.value || <span className="text-gray-600">(empty)</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResolvedSettingsList({
  resolved,
  filter,
}: {
  resolved: Record<string, unknown>;
  filter: string;
}) {
  const { width: keyColWidth, dividerProps } = useResizableColumn(
    320,
    "dtc-mm-as.resolved-settings-key-width",
  );
  const allKeys = Object.keys(resolved).sort();
  const keys = filter
    ? allKeys.filter(
        (k) =>
          k.toLowerCase().includes(filter) ||
          formatResolvedValue(resolved[k]).toLowerCase().includes(filter),
      )
    : allKeys;
  if (keys.length === 0) {
    return (
      <div className="text-gray-500">
        {filter ? "No parameters match the filter." : "No settings found."}
      </div>
    );
  }
  return (
    <div className="flex flex-col">
      {keys.map((k) => (
        <div
          key={k}
          className="flex items-stretch gap-3 py-1 border-b border-bg-border/50 hover:bg-bg-card/40"
        >
          <div
            style={{ width: keyColWidth }}
            className="shrink-0 text-gray-300 break-all leading-tight"
          >
            {k}
          </div>
          <div {...dividerProps} />
          <div className="flex-1 text-gray-100 break-all">
            {formatResolvedValue(resolved[k])}
          </div>
        </div>
      ))}
    </div>
  );
}

function formatResolvedValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean" || typeof v === "number") return String(v);
  if (typeof v === "string") return v === "" ? "(empty)" : v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
