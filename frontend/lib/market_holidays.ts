/**
 * US (NYSE) trading-holiday calendar — 2026-2027.
 *
 * Used by the Market Activity Calendar modal to chip each UTC
 * calendar cell with the matching closure.
 *
 * Tag rule (decided in the v1.5.249 handover): NYSE's local trading
 * window is 09:30-16:00 EST = 14:30-21:00 UTC. The closure for
 * holiday H falls entirely within UTC day H — no spillover into
 * the adjacent UTC day. So a UTC calendar day == the holiday date
 * directly. Simple lookup by ``YYYY-MM-DD``.
 *
 * Multi-exchange (LSE / TSE / HKEX / etc.) is OUT OF SCOPE for
 * this release. The operator may extend this file with additional
 * tables later — the consumer ``usHolidayForUtcDate`` only reads
 * the ``US_TRADING_HOLIDAYS`` symbol so adding more without
 * touching the calendar component is straightforward.
 *
 * Source: NYSE official 2026 + 2027 closure schedule. Observed
 * dates are listed (e.g. 2026-07-03 for July 4 falling on a
 * Saturday) — those are the days NYSE is actually closed.
 */

export interface MarketHoliday {
  /** ISO date in ``YYYY-MM-DD`` form (no time / zone — calendar-day). */
  date: string;
  /** Human-readable name; rendered verbatim in the cell chip. */
  name: string;
}

export const US_TRADING_HOLIDAYS: MarketHoliday[] = [
  // 2026
  { date: "2026-01-01", name: "New Year's Day" },
  { date: "2026-01-19", name: "Martin Luther King Jr. Day" },
  { date: "2026-02-16", name: "Presidents' Day" },
  { date: "2026-04-03", name: "Good Friday" },
  { date: "2026-05-25", name: "Memorial Day" },
  { date: "2026-06-19", name: "Juneteenth" },
  { date: "2026-07-03", name: "Independence Day" },
  { date: "2026-09-07", name: "Labor Day" },
  { date: "2026-11-26", name: "Thanksgiving" },
  { date: "2026-12-25", name: "Christmas" },
  // 2027
  { date: "2027-01-01", name: "New Year's Day" },
  { date: "2027-01-18", name: "Martin Luther King Jr. Day" },
  { date: "2027-02-15", name: "Presidents' Day" },
  { date: "2027-03-26", name: "Good Friday" },
  { date: "2027-05-31", name: "Memorial Day" },
  { date: "2027-06-18", name: "Juneteenth" },
  { date: "2027-07-05", name: "Independence Day" },
  { date: "2027-09-06", name: "Labor Day" },
  { date: "2027-11-25", name: "Thanksgiving" },
  { date: "2027-12-24", name: "Christmas" },
];

// Index by ISO date for O(1) lookup. Materialised at module load.
const HOLIDAY_BY_DATE: Map<string, string> = new Map(
  US_TRADING_HOLIDAYS.map((h) => [h.date, h.name] as const),
);

/**
 * Return the US holiday name for the UTC calendar day containing
 * ``date``, or null if no holiday matches.
 *
 * The match is by ``YYYY-MM-DD`` in UTC — caller passes any
 * ``Date`` instance; we extract its UTC date string and look it
 * up. No timezone conversion past UTC.
 */
export function usHolidayForUtcDate(date: Date): string | null {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;
  const y = date.getUTCFullYear();
  const m = String(date.getUTCMonth() + 1).padStart(2, "0");
  const d = String(date.getUTCDate()).padStart(2, "0");
  const iso = `${y}-${m}-${d}`;
  return HOLIDAY_BY_DATE.get(iso) ?? null;
}
