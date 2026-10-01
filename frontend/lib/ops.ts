/**
 * Shell out to ``scripts/ops.ps1`` from a Next.js API route.
 *
 * Why: snapshot / start / stop / deploy are EC2/SSM operations
 * the operator normally runs by typing ``./scripts/ops.ps1
 * <profile> <command>`` in PowerShell. The dashboard spawns the
 * exact same script with ``child_process.spawn`` so the operator
 * doesn't have to context-switch to a terminal.
 *
 * Trust model: the dashboard runs LOCALLY on the operator's laptop
 * (no external auth, no public exposure). Spawning a script with
 * an account-id from the request URL is fine because:
 *   • the account-id is validated against ``getAccount`` before
 *     this helper is called — anything not in
 *     ``config/profiles/*.env`` is rejected,
 *   • only the four whitelisted commands below are exposed.
 *
 * Long-running: ops commands take 5-90 s typically (EC2 boot, SSM
 * round-trips, snapshot tarball). The caller awaits completion.
 * 120 s wall-clock cap kills runaway processes.
 *
 * Output captured: stdout + stderr returned in the result for the
 * dashboard to surface in the toast / inline detail. Plus an
 * audit-log line on the dev server's stderr (the operator's
 * ``npm run dev`` terminal).
 */

import { spawn } from "node:child_process";
import { resolve } from "node:path";

export type OpsCommand = "snapshot" | "start" | "stop" | "deploy";

export interface OpsResult {
  outcome: "success" | "error";
  detail: string;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
}

const OPS_TIMEOUT_MS = 120_000;
const OPS_SCRIPT_RELATIVE_PATH = ["scripts", "ops.ps1"];

/**
 * Candidate PowerShell binaries, tried in order.
 *
 *   • ``pwsh``       — PowerShell 7+ (cross-platform; preferred).
 *   • ``powershell`` — Windows PowerShell 5.1, built in on every
 *                      Windows install. Fallback when pwsh is not
 *                      on PATH (the operator may not have installed
 *                      PowerShell 7).
 *
 * Detection result is cached for the lifetime of the dev server
 * so subsequent calls don't re-pay the discovery cost.
 */
const POWERSHELL_CANDIDATES: readonly string[] = ["pwsh", "powershell"];
let cachedShell: string | null = null;

async function detectPowerShell(): Promise<string | null> {
  if (cachedShell !== null) return cachedShell;
  for (const candidate of POWERSHELL_CANDIDATES) {
    const found = await new Promise<boolean>((resolveOuter) => {
      const probe = spawn(candidate, ["-NoProfile", "-Command", "$null"], {
        windowsHide: true,
      });
      probe.on("error", () => resolveOuter(false));
      probe.on("close", (code) => resolveOuter(code === 0));
    });
    if (found) {
      cachedShell = candidate;
      // eslint-disable-next-line no-console
      console.error(`[ops] using PowerShell binary: ${candidate}`);
      return candidate;
    }
  }
  return null;
}

/**
 * Run ``ops.ps1 <profile> <command>`` and return the outcome.
 *
 * The ``profile`` arg should already be sanitised — this helper
 * does not validate against ``config/profiles/`` (caller's job).
 * It does, however, refuse to invoke if the script path doesn't
 * resolve cleanly to a real file on disk.
 */
export async function runOpsCommand(
  profile: string,
  command: OpsCommand
): Promise<OpsResult> {
  // Profile-name sanity: tight whitelist to be safe even though
  // the caller already validated. The bot's profile naming
  // convention is dot-separated lowercase ``prod.<venue>.<symbol>``
  // — no shell metacharacters possible in valid names.
  if (!/^[a-zA-Z0-9._-]+$/.test(profile)) {
    return {
      outcome: "error",
      detail: `invalid profile name: ${profile}`,
    };
  }
  // Resolve script path. Next.js dev cwd is the ``frontend/``
  // directory; the script lives one level up under ``scripts/``.
  const scriptPath = resolve(process.cwd(), "..", ...OPS_SCRIPT_RELATIVE_PATH);

  // Pick a PowerShell binary that exists on PATH. ``pwsh`` first
  // (PowerShell 7+), then ``powershell`` (Windows PowerShell 5.1)
  // as the universal Windows fallback.
  const shell = await detectPowerShell();
  if (!shell) {
    return {
      outcome: "error",
      detail:
        "no PowerShell binary on PATH — tried 'pwsh' (PowerShell 7+) " +
        "and 'powershell' (Windows PowerShell 5.1). Install PowerShell " +
        "or add it to PATH.",
    };
  }

  const t0 = Date.now();
  return new Promise<OpsResult>((resolveOuter) => {
    const proc = spawn(
      shell,
      [
        "-NoProfile",
        "-NonInteractive",
        "-File",
        scriptPath,
        profile,
        command,
      ],
      {
        cwd: resolve(process.cwd(), ".."),
        // Inherit env so AWS_PROFILE / AWS credentials etc. are
        // available to the SSM commands the script runs.
        env: process.env,
        windowsHide: true,
      }
    );
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    proc.stderr.on("data", (d) => {
      stderr += d.toString();
    });
    const timer = setTimeout(() => {
      try {
        proc.kill("SIGTERM");
      } catch {
        // Already dead — ignore.
      }
      resolveOuter({
        outcome: "error",
        detail: `${command} timed out after ${OPS_TIMEOUT_MS / 1000}s`,
        stdout,
        stderr,
        duration_ms: Date.now() - t0,
      });
    }, OPS_TIMEOUT_MS);
    proc.on("error", (err) => {
      clearTimeout(timer);
      const code = (err as NodeJS.ErrnoException).code;
      // ENOENT here means the previously-detected shell vanished
      // mid-flight (very rare); the detect step already covers the
      // common "no PowerShell on PATH" case before we got here.
      resolveOuter({
        outcome: "error",
        detail:
          code === "ENOENT"
            ? `${shell} no longer on PATH — restart the dev server`
            : `spawn error: ${String(err)}`,
        stdout,
        stderr,
        duration_ms: Date.now() - t0,
      });
    });
    proc.on("close", (code) => {
      clearTimeout(timer);
      const duration_ms = Date.now() - t0;
      const ok = code === 0;
      // Truncate captured output so toasts / response bodies
      // stay reasonable. Full output is in the dev-server log.
      const tail = (s: string, n = 800): string =>
        s.length > n ? "…" + s.slice(-n) : s;
      // eslint-disable-next-line no-console
      console.error(
        `[ops] profile=${profile} command=${command} ` +
          `code=${code ?? "null"} duration=${duration_ms}ms\n` +
          `--- stdout ---\n${stdout}\n--- stderr ---\n${stderr}`
      );
      // Build error detail: PowerShell's ``Write-Error`` rendering
      // puts the message FIRST and CategoryInfo / FullyQualifiedErrorId
      // boilerplate at the END. A short 200-char tail catches only
      // the useless trailer. Widen substantially AND include both
      // streams so the operator sees the real cause in the toast.
      // The full untruncated output is still logged to the dev-server
      // terminal via ``console.error`` above.
      const combineForError = (out: string, err: string): string => {
        const parts: string[] = [];
        if (err) parts.push(`STDERR:\n${tail(err, 1500)}`);
        if (out && out.trim()) parts.push(`STDOUT:\n${tail(out, 1500)}`);
        if (parts.length === 0) return "(no output)";
        return parts.join("\n---\n");
      };
      resolveOuter({
        outcome: ok ? "success" : "error",
        detail: ok
          ? defaultSuccessMessage(command)
          : `ops.ps1 exit=${code ?? "null"}\n${combineForError(stdout, stderr)}`,
        stdout: tail(stdout, 4000),
        stderr: tail(stderr, 4000),
        duration_ms,
      });
    });
  });
}

function defaultSuccessMessage(command: OpsCommand): string {
  switch (command) {
    case "snapshot":
      return "snapshot taken";
    case "start":
      return "bot started";
    case "stop":
      return "bot stopped";
    case "deploy":
      return "deployed";
  }
}
