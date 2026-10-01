/**
 * Shared dashboard types extracted from ``app/page.tsx``.
 *
 * Why a separate module: Next.js's app-router page files restrict
 * which symbols can be exported (basically just ``default`` and a
 * fixed allowlist of route helpers — ``metadata``, ``generateMetadata``,
 * etc.). Re-exporting our domain types from ``page.tsx`` would
 * trigger build warnings or be unsupported in future Next versions.
 * A dedicated module keeps things clean and lets feature modules
 * (e.g. ``components/dashboard/ConfigPanel.tsx``) consume the types
 * without depending on the page itself.
 *
 * Currently scoped to types referenced from extracted lazy-loaded
 * panels. More types can move here over time as additional pieces of
 * ``page.tsx`` get split out (see plans/frontend.md Rec #2).
 */

export interface ConfigPayload {
  schema_version: number;
  profile: string;
  symbol: string | null;
  version: string | null;
  captured_at_utc: string;
  session_id: string | null;
  session_started_at_utc: string;
  env_file_path: string;
  env_file_raw: string;
  env_file_read_error: string | null;
  resolved_settings: Record<string, unknown>;
}

export interface ConfigResult {
  profile: string;
  payload: ConfigPayload | null;
  fetched_at_utc: string;
  errors: string[];
}
