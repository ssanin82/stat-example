/**
 * Global market-event annotations for the kline chart overlay.
 *
 * Two event types:
 *   * **lines** — exact recurring timestamps (e.g. London cash open,
 *     US cash open). Rendered as thin vertical strokes with a short
 *     label at the top and a hover-tooltip with the full reason.
 *   * **bands** — regime windows (e.g. Asia core session, US/Europe
 *     overlap). Rendered as low-opacity full-height background
 *     rectangles.
 *
 * All times are configured in the event's NATIVE timezone (IANA tz
 * id, e.g. ``Europe/London`` or ``America/New_York``). The DST-aware
 * resolver below converts each occurrence to a precise UTC instant
 * for the calendar dates overlapping the visible chart window.
 *
 * v1.5.161 — initial config. Operator-curated based on ChatGPT
 * suggestions, with the OKX-funding lines DELIBERATELY OMITTED
 * (operator: "I am not sure that OKX funding is exclusively
 * important... maybe, don't add funding lines here").
 *
 * Importance scale (1-5): higher = more impactful regime boundary.
 * Default render filter is ``>= 4``; lower values surface in
 * "debug" mode (operator follow-up).
 */

export type IsoWeekday = 1 | 2 | 3 | 4 | 5 | 6 | 7;  // 1 = Mon, 7 = Sun

export interface MarketEventLine {
  id: string;
  label: string;          // short, ≤ 24 chars — what appears on the chart
  reason: string;         // full description for the hover tooltip
  category: string;
  importance: 1 | 2 | 3 | 4 | 5;
  timezone: string;       // IANA tz id, DST-aware via Intl
  time: string;           // "HH:MM" 24h in the event's native tz
  days: IsoWeekday[];     // 1-7, ISO weekday convention
  // Optional cross-reference for paired session bounds. Populated
  // so the tooltip can answer "this opens at HH:MM — *when does it
  // close*?" without the operator having to scroll the chart into
  // the future to find the matching close line. Both fields are
  // ``"HH:MM"`` interpreted in this event's OWN ``timezone`` (the
  // common case — cash sessions open and close in their home tz).
  // Set ``sessionEnd`` on open events; ``sessionStart`` on close
  // events. Leave unset for one-off events (UTC roll, CME reopen).
  sessionEnd?: string;
  sessionStart?: string;
}

export interface MarketEventBand {
  id: string;
  label: string;
  reason: string;
  category: string;
  importance: 1 | 2 | 3 | 4 | 5;
  start: { timezone: string; time: string };
  end: { timezone: string; time: string };
  days: IsoWeekday[];
  opacity?: number;       // band fill opacity (default 0.05)
}

/* ------------------------------------------------------------------
 * Event data
 * ------------------------------------------------------------------
 *
 * No OKX-specific funding lines per operator preference. If the
 * operator later wants venue-specific funding marks, they belong in
 * a separate venue-events config keyed by symbol/profile so the
 * dashboard doesn't tie cross-venue session boundaries to one
 * exchange's settle cadence.
 */

export const LINE_EVENTS: MarketEventLine[] = [
  {
    id: "utc_daily_roll",
    label: "UTC roll",
    reason:
      "New UTC day boundary. Many systems, daily candles, stats, " +
      "risk limits, and cron-driven jobs reset around this time.",
    category: "crypto_structure",
    importance: 4,
    timezone: "UTC",
    time: "00:00",
    days: [1, 2, 3, 4, 5, 6, 7],
  },
  {
    id: "tokyo_seoul_open",
    label: "TOK/SEO open",
    reason:
      "Start of major East Asian cash-market activity (Tokyo + " +
      "Seoul). Korea is especially relevant for crypto participation.",
    category: "asia_session",
    importance: 4,
    timezone: "Asia/Tokyo",
    time: "09:00",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "15:00",  // TSE close
  },
  {
    id: "hk_shanghai_open",
    label: "HK/SH open",
    reason:
      "Hong Kong / Shanghai cash open. Important Asia liquidity and " +
      "risk boundary; useful proxy for China/HK/offshore Asia flow. " +
      "Note: OKX is HK-based — this is roughly when native flow ramps.",
    category: "asia_session",
    importance: 5,
    timezone: "Asia/Hong_Kong",
    time: "09:30",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "16:00",  // HKEX close (lunch break 12:00-13:00 ignored)
  },
  {
    id: "india_open",
    label: "IND open",
    reason:
      "Start of Indian cash equity session (NSE/BSE). Good proxy " +
      "for South Asian trader activity ramp. TON has heavy South-" +
      "Asian retail base — relevant for the operator's symbol.",
    category: "south_asia_session",
    importance: 4,
    timezone: "Asia/Kolkata",
    time: "09:15",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "15:30",  // NSE/BSE close
  },
  {
    id: "europe_london_open",
    label: "LON open",
    reason:
      "London cash open — major global liquidity and regime " +
      "boundary. DST-aware (BST in summer, GMT in winter).",
    category: "europe_session",
    importance: 5,
    timezone: "Europe/London",
    time: "08:00",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "16:30",  // LSE close
  },
  {
    id: "europe_frankfurt_open",
    label: "FFM open",
    reason:
      "Frankfurt / Xetra cash open. Main continental Europe equity " +
      "open; often overlaps closely with London in UTC terms.",
    category: "europe_session",
    importance: 4,
    timezone: "Europe/Berlin",
    time: "09:00",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "17:30",  // Xetra close
  },
  {
    id: "us_open",
    label: "US open",
    reason:
      "New York cash open. Most important global risk-session " +
      "boundary; strong effect on liquidity, volatility, and " +
      "directional flow. DST-aware (EDT in summer, EST in winter).",
    category: "us_session",
    importance: 5,
    timezone: "America/New_York",
    time: "09:30",
    days: [1, 2, 3, 4, 5],
    sessionEnd: "16:00",  // NYSE/Nasdaq close
  },
  {
    id: "europe_london_close",
    label: "LON close",
    reason:
      "London cash close. Can shift liquidity and positioning as " +
      "Europe exits and US remains active.",
    category: "europe_session",
    importance: 3,
    timezone: "Europe/London",
    time: "16:30",
    days: [1, 2, 3, 4, 5],
    sessionStart: "08:00",  // LSE open
  },
  {
    id: "us_close",
    label: "US close",
    reason:
      "New York cash close. Major liquidity and risk-transfer " +
      "boundary; US equity close often changes crypto regime.",
    category: "us_session",
    importance: 5,
    timezone: "America/New_York",
    time: "16:00",
    days: [1, 2, 3, 4, 5],
    sessionStart: "09:30",  // NYSE/Nasdaq open
  },
  {
    id: "cme_globex_reopen",
    label: "CME reopen",
    reason:
      "US futures Globex reopens after the daily maintenance " +
      "break. Can matter for BTC/ETH and macro-linked crypto flow.",
    category: "futures_session",
    importance: 3,
    timezone: "America/Chicago",
    time: "17:00",
    days: [7, 1, 2, 3, 4],  // Sun, Mon, Tue, Wed, Thu
  },
];

export const BAND_EVENTS: MarketEventBand[] = [
  {
    id: "asia_core_session",
    label: "Asia core",
    reason:
      "Tokyo, Seoul, Hong Kong, Shanghai, Singapore, and early " +
      "South Asia activity. Broad Asia regime window.",
    category: "asia_session",
    importance: 5,
    start: { timezone: "UTC", time: "00:00" },
    end: { timezone: "UTC", time: "05:30" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.06,
  },
  {
    id: "europe_preopen",
    label: "Europe pre-open",
    reason:
      "European desks arrive before official cash open. Often " +
      "visible in liquidity and volatility before the bell.",
    category: "europe_session",
    importance: 5,
    start: { timezone: "Europe/London", time: "07:00" },
    end: { timezone: "Europe/London", time: "08:00" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.08,
  },
  {
    id: "europe_cash_session",
    label: "Europe session",
    reason: "Main European cash trading session. Broad regime context.",
    category: "europe_session",
    importance: 4,
    start: { timezone: "Europe/London", time: "08:00" },
    end: { timezone: "Europe/London", time: "16:30" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.04,
  },
  {
    id: "us_preopen",
    label: "US pre-open",
    reason:
      "US macro/equity desks become active before cash open; can " +
      "affect crypto liquidity and direction.",
    category: "us_session",
    importance: 4,
    start: { timezone: "America/New_York", time: "08:00" },
    end: { timezone: "America/New_York", time: "09:30" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.07,
  },
  {
    id: "us_cash_session",
    label: "US session",
    reason:
      "Main US cash session. Usually the most important global " +
      "risk/liquidity regime.",
    category: "us_session",
    importance: 5,
    start: { timezone: "America/New_York", time: "09:30" },
    end: { timezone: "America/New_York", time: "16:00" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.05,
  },
  {
    id: "us_europe_overlap",
    label: "US/EU overlap",
    reason:
      "Highest-quality institutional overlap window in normal " +
      "markets. DST-aware via the two anchor timezones.",
    category: "overlap_session",
    importance: 5,
    start: { timezone: "America/New_York", time: "09:30" },
    end: { timezone: "Europe/London", time: "16:30" },
    days: [1, 2, 3, 4, 5],
    opacity: 0.09,
  },
  {
    id: "weekend_crypto",
    label: "Weekend",
    reason:
      "Weekend crypto regime: thinner institutional liquidity, " +
      "different adverse-selection and liquidation behavior.",
    category: "crypto_structure",
    importance: 4,
    start: { timezone: "UTC", time: "00:00" },
    end: { timezone: "UTC", time: "23:59" },
    days: [6, 7],  // Sat, Sun
    opacity: 0.07,
  },
];

/* ------------------------------------------------------------------
 * Human-friendly timezone formatting
 * ------------------------------------------------------------------ */

/** Map an IANA timezone id to a short city label suitable for inline
 *  tooltips (e.g. ``"America/New_York"`` → ``"New York"``). For
 *  ``UTC`` returns ``"UTC"``. Unknown ids fall back to the last
 *  path segment with underscores replaced by spaces. */
export function tzCityLabel(iana: string): string {
  if (iana === "UTC") return "UTC";
  const seg = (iana.split("/").pop() || iana).replace(/_/g, " ");
  // Two minor refinements where the IANA segment isn't the trading
  // city: the Berlin tz id stands in for Frankfurt's open in the
  // ``europe_frankfurt_open`` event, and Kolkata stands in for the
  // Indian session anchored on NSE/BSE.
  if (iana === "Europe/Berlin") return "Frankfurt";
  if (iana === "Asia/Kolkata") return "Mumbai";
  return seg;
}

/** Tooltip-ready time string for a line event: ``"09:30 New York"``
 *  or ``"00:00 UTC"``. */
export function formatLineTiming(ev: MarketEventLine): string {
  return `${ev.time} ${tzCityLabel(ev.timezone)}`;
}

/** Natural-language phrase for a line event's timing, picking the
 *  verb from the event id and pairing both bounds of the session
 *  when available:
 *    ``"Opens at 09:30 New York · Closes at 16:00 New York"``
 *    ``"Closes at 16:00 New York · Opened at 09:30 New York"``
 *    ``"Reopens at 17:00 Chicago"``
 *    ``"Rolls at 00:00 UTC"``
 *  Paired times come from ``ev.sessionEnd`` / ``ev.sessionStart``
 *  (both interpreted in the event's own timezone). For one-off
 *  events without a pair, falls back to a single bound. */
export function formatLineTimingPhrase(ev: MarketEventLine): string {
  const id = ev.id.toLowerCase();
  const tz = tzCityLabel(ev.timezone);
  const t = `${ev.time} ${tz}`;
  if (id.includes("reopen")) return `Reopens at ${t}`;
  if (id.includes("close")) {
    const opens = ev.sessionStart
      ? ` · Opened at ${ev.sessionStart} ${tz}`
      : "";
    return `Closes at ${t}${opens}`;
  }
  if (id.includes("open")) {
    const closes = ev.sessionEnd
      ? ` · Closes at ${ev.sessionEnd} ${tz}`
      : "";
    return `Opens at ${t}${closes}`;
  }
  if (id.includes("roll")) return `Rolls at ${t}`;
  return `At ${t}`;
}

/** Tooltip-ready start → end window for a band event:
 *  ``"09:30 New York → 16:30 London"`` (DST-aware two-anchor bands
 *  read naturally as the cross-region window they actually are). */
export function formatBandTiming(ev: MarketEventBand): string {
  const start = `${ev.start.time} ${tzCityLabel(ev.start.timezone)}`;
  const end = `${ev.end.time} ${tzCityLabel(ev.end.timezone)}`;
  return `${start} → ${end}`;
}

/* ------------------------------------------------------------------
 * Browser-local-time formatting
 * ------------------------------------------------------------------
 *
 * Converts a (HH:MM, IANA tz) anchor to the same instant expressed in
 * the user's BROWSER local timezone. Used to add a "(17:30 local
 * time)" suffix to tooltips so the operator doesn't have to mentally
 * convert "NY 09:30 EDT" → Dubai every time.
 *
 * Anchored on today's calendar date in the source tz so DST state is
 * correct for the current occurrence. If you're hovering this tooltip
 * around a DST changeover, the local time shifts by 1h between hover
 * sessions — that's the right behaviour.
 */

function parseHHMMTuple(s: string): [number, number] {
  const m = /^(\d{1,2}):(\d{2})$/.exec(s);
  if (!m) return [0, 0];
  return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

function ymdInTzInternal(d: Date, tz: string): [number, number, number] {
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(d);
  const obj: Record<string, string> = {};
  for (const p of parts) obj[p.type] = p.value;
  return [
    parseInt(obj.year || "1970", 10),
    parseInt(obj.month || "1", 10),
    parseInt(obj.day || "1", 10),
  ];
}

/** Given an ``HH:MM`` clock face in ``timezone`` (an IANA id),
 *  return the corresponding ``HH:MM`` clock face in the browser's
 *  local timezone, anchored on today's date in the source tz so DST
 *  state is current. SSR-safe: ``new Date()`` works on Node too.
 *
 *  Returns ``"--:--"`` on malformed input rather than throwing,
 *  matching the fail-soft posture of ``parseHHMM`` / ``tzToUtcMs``. */
export function formatLocalTime(time: string, timezone: string): string {
  const [h, m] = parseHHMMTuple(time);
  if (!Number.isFinite(h) || !Number.isFinite(m)) return "--:--";
  const now = new Date();
  const [y, mo, d] = ymdInTzInternal(now, timezone);
  const utcMs = tzToUtcMs(y, mo, d, h, m, timezone);
  const fmt = new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  return fmt.format(new Date(utcMs));
}

/** Local-time bracket for a line event with paired session bound.
 *  Returns ``"08:00 → 17:30 local time"`` style strings. When the
 *  event has no paired bound (utc roll, CME reopen), returns just
 *  the single time, e.g. ``"04:00 local time"``. */
export function formatLineLocalTiming(ev: MarketEventLine): string {
  const main = formatLocalTime(ev.time, ev.timezone);
  // For close events, pair with sessionStart (in same tz).
  if (ev.sessionStart) {
    const start = formatLocalTime(ev.sessionStart, ev.timezone);
    return `${start} → ${main} local time`;
  }
  // For open events, pair with sessionEnd (in same tz).
  if (ev.sessionEnd) {
    const end = formatLocalTime(ev.sessionEnd, ev.timezone);
    return `${main} → ${end} local time`;
  }
  return `${main} local time`;
}

/** Local-time bracket for a band event: ``"17:30 → 20:30 local
 *  time"``. Start and end may live in different timezones (US/EU
 *  overlap), each resolved to the browser's local clock face. */
export function formatBandLocalTiming(ev: MarketEventBand): string {
  const start = formatLocalTime(ev.start.time, ev.start.timezone);
  const end = formatLocalTime(ev.end.time, ev.end.timezone);
  return `${start} → ${end} local time`;
}

/* ------------------------------------------------------------------
 * Timezone-aware resolution
 * ------------------------------------------------------------------ */

/** Parse "HH:MM" → [hours, minutes]; returns [0, 0] on malformed
 *  input rather than throwing — fail-soft so a config typo doesn't
 *  blow up the chart. */
function parseHHMM(s: string): [number, number] {
  const m = /^(\d{1,2}):(\d{2})$/.exec(s);
  if (!m) return [0, 0];
  return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

/** Resolve (year, month, day, hours, minutes, IANA tz) → exact UTC
 *  instant in milliseconds. Uses Intl.DateTimeFormat to query the
 *  target tz's UTC offset for that specific calendar moment — DST
 *  transitions handled correctly by the platform.
 *
 *  ``year`` / ``month`` / ``day`` are in the TARGET timezone's
 *  calendar (not the local browser tz).
 */
export function tzToUtcMs(
  year: number,
  month: number,
  day: number,
  hours: number,
  minutes: number,
  tz: string,
): number {
  // Start with the naive UTC interpretation: assume the target tz
  // has zero offset. We'll then correct using Intl.
  let utcMs = Date.UTC(year, month - 1, day, hours, minutes, 0, 0);
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  // Format the naive UTC instant in the target tz; the difference
  // between what it READS and what we WANTED is the timezone offset
  // (with DST applied).
  const parts = fmt.formatToParts(new Date(utcMs));
  const partsObj: Record<string, string> = {};
  for (const p of parts) partsObj[p.type] = p.value;
  const seenHours = parseInt(partsObj.hour || "0", 10);
  const seenMins = parseInt(partsObj.minute || "0", 10);
  // Day may also differ on either side of midnight tz boundary.
  const seenDay = parseInt(partsObj.day || String(day), 10);
  const seenMonth = parseInt(partsObj.month || String(month), 10);
  const seenYear = parseInt(partsObj.year || String(year), 10);
  // Compute the absolute minute difference between requested and seen.
  // Walk via UTC-day math so a DST-spring-forward gap doesn't loop.
  const requestedUtcMin = Date.UTC(year, month - 1, day, hours, minutes) / 60_000;
  const seenUtcMin = Date.UTC(seenYear, seenMonth - 1, seenDay, seenHours, seenMins) / 60_000;
  const offMin = seenUtcMin - requestedUtcMin;
  utcMs = utcMs - offMin * 60_000;
  return utcMs;
}

/* Calendar-day iteration helpers. */

function isoWeekday(d: Date): IsoWeekday {
  // ``Date.prototype.getUTCDay`` returns 0=Sun..6=Sat; remap to
  // 1=Mon..7=Sun (ISO).
  const u = d.getUTCDay();
  return (u === 0 ? 7 : u) as IsoWeekday;
}

/** Days that overlap [t0Ms, tNMs]. Returns one Date per
 *  midnight-UTC anchor inside the window plus a 1-day buffer either
 *  side (event times can shift the actual UTC instant up to ±1
 *  day across timezone offsets). */
function utcDaysOverlapping(t0Ms: number, tNMs: number): Date[] {
  const out: Date[] = [];
  const startMs = t0Ms - 86_400_000;
  const endMs = tNMs + 86_400_000;
  const startDate = new Date(startMs);
  const startDay = Date.UTC(
    startDate.getUTCFullYear(),
    startDate.getUTCMonth(),
    startDate.getUTCDate(),
  );
  for (let t = startDay; t <= endMs; t += 86_400_000) {
    out.push(new Date(t));
  }
  return out;
}

/* ------------------------------------------------------------------
 * Resolve to overlapping window
 * ------------------------------------------------------------------ */

export interface ResolvedLine {
  event: MarketEventLine;
  utcMs: number;
}

export interface ResolvedBand {
  event: MarketEventBand;
  startUtcMs: number;
  endUtcMs: number;
}

/** Compute the calendar Y/M/D, IN THE TARGET TZ, for a given UTC
 *  Date. Returns the [year, month-1-indexed, day] tuple in that tz. */
function ymdInTz(
  d: Date,
  tz: string,
): [number, number, number] {
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(d);
  const obj: Record<string, string> = {};
  for (const p of parts) obj[p.type] = p.value;
  return [
    parseInt(obj.year || "1970", 10),
    parseInt(obj.month || "1", 10),
    parseInt(obj.day || "1", 10),
  ];
}

/** Compute the ISO weekday of the calendar date (year, month-1, day)
 *  as if it were a date — uses Date.UTC + getUTCDay so we don't
 *  drift due to browser tz. */
function isoWeekdayOf(year: number, month1Indexed: number, day: number): IsoWeekday {
  const d = new Date(Date.UTC(year, month1Indexed - 1, day));
  return isoWeekday(d);
}

/** All occurrences of ``LINE_EVENTS`` that fall inside
 *  [t0Ms, tNMs], filtered by ``minImportance`` (inclusive). */
export function resolveLineEvents(
  t0Ms: number,
  tNMs: number,
  minImportance: number,
): ResolvedLine[] {
  const out: ResolvedLine[] = [];
  const seedDays = utcDaysOverlapping(t0Ms, tNMs);
  for (const ev of LINE_EVENTS) {
    if (ev.importance < minImportance) continue;
    const [h, m] = parseHHMM(ev.time);
    for (const dUtc of seedDays) {
      // Compute the event's calendar date in its native timezone for
      // this UTC-day anchor. Then check weekday + resolve to UTC.
      const [y, mo, d] = ymdInTz(dUtc, ev.timezone);
      const wd = isoWeekdayOf(y, mo, d);
      if (!ev.days.includes(wd)) continue;
      const utcMs = tzToUtcMs(y, mo, d, h, m, ev.timezone);
      if (utcMs < t0Ms || utcMs > tNMs) continue;
      // Avoid duplicate entries when the seed-day loop produces the
      // same calendar date twice (can happen at tz boundaries).
      if (out.some((r) => r.event.id === ev.id && r.utcMs === utcMs)) continue;
      out.push({ event: ev, utcMs });
    }
  }
  out.sort((a, b) => a.utcMs - b.utcMs);
  return out;
}

/** Bands whose UTC window contains the given instant. Pulls
 *  occurrences from a 36-hour window around ``nowMs`` (well past the
 *  longest configured band) and filters to those bracketing the
 *  moment. Used by the "Ongoing events" panel so the operator can
 *  see exactly which chart shadings they're currently inside,
 *  sourced from the same config that drives the chart overlay. */
export function getActiveBands(
  nowMs: number,
  minImportance: number = 4,
): ResolvedBand[] {
  const margin = 36 * 60 * 60_000;
  const all = resolveBandEvents(nowMs - margin, nowMs + margin, minImportance);
  return all.filter(
    (b) => b.startUtcMs <= nowMs && nowMs <= b.endUtcMs,
  );
}

/** All occurrences of ``BAND_EVENTS`` that overlap [t0Ms, tNMs]. */
export function resolveBandEvents(
  t0Ms: number,
  tNMs: number,
  minImportance: number,
): ResolvedBand[] {
  const out: ResolvedBand[] = [];
  const seedDays = utcDaysOverlapping(t0Ms, tNMs);
  for (const ev of BAND_EVENTS) {
    if (ev.importance < minImportance) continue;
    const [sh, sm] = parseHHMM(ev.start.time);
    const [eh, em] = parseHHMM(ev.end.time);
    for (const dUtc of seedDays) {
      // Both endpoints reference the SAME calendar day in their own
      // native timezones — for US/EU overlap (NY 09:30 → London
      // 16:30) this means: pick the calendar date in the *start*
      // timezone, compute start; then the *end* time is on the
      // same calendar day in the end timezone (close enough at
      // mid-Atlantic).
      const [y, mo, d] = ymdInTz(dUtc, ev.start.timezone);
      const wd = isoWeekdayOf(y, mo, d);
      if (!ev.days.includes(wd)) continue;
      const startUtc = tzToUtcMs(y, mo, d, sh, sm, ev.start.timezone);
      // The end uses the end timezone's calendar date — typically
      // the same date but can differ across the international date
      // line. Use the SAME y/mo/d for simplicity; if end < start
      // (which would mean a midnight wrap), bump end by a day.
      let endUtc = tzToUtcMs(y, mo, d, eh, em, ev.end.timezone);
      if (endUtc <= startUtc) endUtc += 86_400_000;
      // Clip to window for partial-overlap bands.
      const clipStart = Math.max(t0Ms, startUtc);
      const clipEnd = Math.min(tNMs, endUtc);
      if (clipEnd <= clipStart) continue;
      if (
        out.some(
          (r) =>
            r.event.id === ev.id &&
            r.startUtcMs === startUtc &&
            r.endUtcMs === endUtc,
        )
      ) continue;
      out.push({ event: ev, startUtcMs: startUtc, endUtcMs: endUtc });
    }
  }
  out.sort((a, b) => a.startUtcMs - b.startUtcMs);
  return out;
}
