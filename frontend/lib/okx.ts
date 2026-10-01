/**
 * OKX V5 signing + signed REST helpers.
 *
 * Mirrors the Python implementation in `app/exchange/okx_client.py` --
 * same payload shape, same headers, same endpoints. Cross-checked
 * by running both against the live the partner sub-account on 2026-05-04
 * (the operator's preflight + integration tests passed; this TS
 * version targets the same surface).
 */

import { createHmac } from "node:crypto";
import {
  resolveVenueCreds,
  resolveTradingCreds,
  hasTradingCreds,
} from "./creds";

const OKX_REST = "https://www.okx.com";
const OKX_TRADING_NAMES = [
  "OKX_API_KEY",
  "OKX_API_SECRET",
  "OKX_API_PASSPHRASE",
];

interface OkxCreds {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
}

function readCreds(): OkxCreds {
  // Prefer OKX_API_KEY_READONLY / SECRET_READONLY / PASSPHRASE_READONLY
  // when ALL three are present. the partner's primary keys already work for
  // the dashboard (no IP whitelist on that sub-account by the partner's
  // design), so READONLY is mostly relevant for self-funded OKX
  // accounts in the future.
  const { values } = resolveVenueCreds("OKX", OKX_TRADING_NAMES);
  return {
    apiKey: values[0],
    apiSecret: values[1],
    passphrase: values[2],
  };
}

function readTradingCreds(): OkxCreds {
  const { values } = resolveTradingCreds("OKX", OKX_TRADING_NAMES);
  return {
    apiKey: values[0],
    apiSecret: values[1],
    passphrase: values[2],
  };
}

export function hasOkxTradingCreds(): boolean {
  return hasTradingCreds(OKX_TRADING_NAMES);
}

/**
 * OKX requires ISO 8601 with exactly millisecond precision and a
 * literal 'Z' suffix. JS's Date.toISOString() already produces this
 * shape: `2026-05-04T08:30:00.000Z`. Verified against the Python
 * adapter's output.
 */
function okxIsoTimestamp(): string {
  return new Date().toISOString();
}

function okxSign(
  apiSecret: string,
  ts: string,
  method: string,
  requestPath: string,
  body: string
): string {
  const prehash = `${ts}${method.toUpperCase()}${requestPath}${body}`;
  return createHmac("sha256", apiSecret).update(prehash).digest("base64");
}

interface OkxResponse<T = unknown> {
  code: string;
  msg: string;
  data: T[];
}

async function okxSignedRequest<T = unknown>(
  method: "GET" | "POST",
  path: string,
  opts: {
    params?: Record<string, string>;
    body?: unknown;
    write?: boolean;
  } = {}
): Promise<OkxResponse<T>> {
  // Use trading creds for explicit writes; reads default to the
  // venue resolver (which prefers READONLY when present).
  const creds = opts.write ? readTradingCreds() : readCreds();
  const queryString =
    opts.params && Object.keys(opts.params).length > 0
      ? "?" + new URLSearchParams(opts.params).toString()
      : "";
  const requestPath = path + queryString;
  const bodyStr =
    opts.body === undefined ? "" : JSON.stringify(opts.body);
  const ts = okxIsoTimestamp();
  const sign = okxSign(creds.apiSecret, ts, method, requestPath, bodyStr);
  const url = OKX_REST + requestPath;
  const headers: Record<string, string> = {
    "OK-ACCESS-KEY": creds.apiKey,
    "OK-ACCESS-SIGN": sign,
    "OK-ACCESS-TIMESTAMP": ts,
    "OK-ACCESS-PASSPHRASE": creds.passphrase,
  };
  if (bodyStr) headers["Content-Type"] = "application/json";
  const resp = await fetch(url, {
    method,
    headers,
    body: bodyStr || undefined,
    cache: "no-store",
  });
  return (await resp.json()) as OkxResponse<T>;
}

// Compatibility shim for existing read paths.
async function okxSignedGet<T = unknown>(
  path: string,
  params: Record<string, string> = {}
): Promise<OkxResponse<T>> {
  return okxSignedRequest<T>("GET", path, { params });
}

// ---------------------------------------------------------------------------
// OKX instrument metadata cache (public endpoint).
// ---------------------------------------------------------------------------
//
// Why: OKX's `pos`, `sz`, `fillSz`, `accFillSz` fields are all in
// CONTRACTS, not base-asset units. To present user-meaningful base
// quantities + correct USD notionals on the dashboard we need
// `ctVal` (base-asset value of one contract) per symbol. Most
// USDT-M perps have `ctVal=1` (e.g. SUI, TON — 1 contract = 1
// base unit), but several major listings have `ctVal != 1`:
// HYPE-USDT-SWAP is 0.1 (so 1 contract = 0.1 HYPE). Without
// ctVal the dashboard shows contract counts and over-states
// notional by 1/ctVal, producing false-positive risk-violation
// flags on the orders / fills tables.
//
// Caching: 5-minute TTL is safe — instrument metadata changes
// only when OKX adds/delists products (rare). The `instruments`
// endpoint returns ~250 SWAP instruments in one call, so a
// per-process cache costs nothing in memory.

interface InstrumentMeta {
  ctVal: number;
  tickSz: number;
  lotSz: number;
  minSz: number;
  state: string;
}

const INSTRUMENTS_TTL_MS = 5 * 60 * 1000;
let _instrumentsCache: {
  ts_ms: number;
  byId: Map<string, InstrumentMeta>;
} | null = null;
let _instrumentsInflight: Promise<Map<string, InstrumentMeta>> | null = null;

async function fetchOkxInstrumentsMap(): Promise<Map<string, InstrumentMeta>> {
  const now = Date.now();
  if (_instrumentsCache && now - _instrumentsCache.ts_ms < INSTRUMENTS_TTL_MS) {
    return _instrumentsCache.byId;
  }
  if (_instrumentsInflight) return _instrumentsInflight;
  _instrumentsInflight = (async () => {
    // Public endpoint — no auth needed. Use the unsigned fetch
    // path so we don't burn a signed call when ctVal lookups are
    // very frequent (every state poll).
    const url = OKX_REST + "/api/v5/public/instruments?instType=SWAP";
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`okx_instruments: HTTP ${resp.status}`);
    }
    const j = (await resp.json()) as {
      code: string;
      msg: string;
      data: Array<{
        instId?: string;
        ctVal?: string;
        tickSz?: string;
        lotSz?: string;
        minSz?: string;
        state?: string;
      }>;
    };
    if (j.code !== "0") {
      throw new Error(`okx_instruments: ${j.msg || j.code}`);
    }
    const byId = new Map<string, InstrumentMeta>();
    for (const r of j.data || []) {
      if (!r.instId) continue;
      byId.set(r.instId, {
        ctVal: parseFloat0(r.ctVal),
        tickSz: parseFloat0(r.tickSz),
        lotSz: parseFloat0(r.lotSz),
        minSz: parseFloat0(r.minSz),
        state: String(r.state || ""),
      });
    }
    _instrumentsCache = { ts_ms: now, byId };
    return byId;
  })();
  try {
    return await _instrumentsInflight;
  } finally {
    _instrumentsInflight = null;
  }
}

/**
 * Look up `ctVal` for a single OKX SWAP instrument. Falls back to
 * 1.0 if the symbol isn't found (treats unknown instruments as
 * "1 contract = 1 base unit" — matches the most common case and
 * preserves legacy behaviour for tests / scripts that exercise
 * symbols outside the instruments cache). Errors fall back to 1.0
 * defensively rather than throwing — a misclassified ctVal is a
 * display issue, not a trading-correctness issue.
 */
export async function fetchOkxCtVal(symbol: string): Promise<number> {
  try {
    const map = await fetchOkxInstrumentsMap();
    const meta = map.get(symbol);
    if (meta && meta.ctVal > 0 && Number.isFinite(meta.ctVal)) {
      return meta.ctVal;
    }
  } catch {
    // Best-effort. Caller still gets 1.0 = legacy behaviour.
  }
  return 1.0;
}

// ---------------------------------------------------------------------------
// Public account-state surface
// ---------------------------------------------------------------------------

export interface AccountSnapshot {
  /** Total equity in USD (incl. unrealised). */
  equity_usd: number | null;
  /** Free / withdrawable margin. */
  withdrawable_usd: number | null;
  /** Cash balance (sum of stablecoin cashBal across the account). */
  cash_usd: number | null;
  /** Per-currency balance details (raw passthrough of OKX's ``details``). */
  details: Array<{
    ccy: string;
    cashBal: number;
    eq: number;
    availBal: number;
  }>;
  /** Sub-account UID for traceability. */
  uid?: string;
}

export interface PositionSnapshot {
  symbol: string;
  position_qty: number;
  avg_entry_price: number | null;
  mark_price: number | null;
  unrealized_pnl_usd: number;
  /** Notional in USD (qty * mark). */
  notional_usd: number;
}

export interface OpenOrder {
  ord_id: string;
  cl_ord_id: string;
  side: "buy" | "sell";
  px: number;
  sz_contracts: number;
  state: string;
  c_time_ms: number;
}

function parseFloat0(v: unknown): number {
  if (v === null || v === undefined || v === "") return 0;
  const f = Number(v);
  return Number.isFinite(f) ? f : 0;
}

/**
 * Per-symbol leverage + margin-mode + account-level position-mode.
 * Fetched once per page load for the dashboard's Position panel.
 *
 * Tries cross mode first; if that returns no rows, retries with
 * isolated. ``leverage`` and ``marginMode`` are null when neither
 * mode has a row (symbol genuinely not configured). ``positionMode``
 * comes from the separate /account/config call.
 */
export interface OkxVenueInfo {
  leverage: string | null; // e.g. "10"
  margin_mode: string | null; // "cross" | "isolated"
  position_mode: string | null; // "net_mode" | "long_short_mode"
  account_level: string | null; // "1"-"4" per OKX acctLv
  // v1.5.260 — instrument metadata (static-per-symbol, populated from
  // the already-cached /public/instruments map so this adds 0 OKX
  // calls when the cache is warm). Drives the dashboard's Reference
  // Data banner. ``null`` when the symbol isn't in the instruments
  // map (e.g. delisted) — banner renders "—" for those rows.
  tick_size: number | null;       // price tick (quote ccy)
  lot_size: number | null;        // size lot (contracts)
  min_size: number | null;        // minimum order size (contracts)
  contract_value: number | null;  // base units per contract (``ctVal``)
}

export async function fetchOkxVenueInfo(
  symbol: string
): Promise<OkxVenueInfo> {
  const result: OkxVenueInfo = {
    leverage: null,
    margin_mode: null,
    position_mode: null,
    account_level: null,
    tick_size: null,
    lot_size: null,
    min_size: null,
    contract_value: null,
  };

  // v1.5.260 — instrument metadata. Pulled from the cached
  // ``/public/instruments`` map (5 min TTL, shared across the
  // process). Cost when warm: a Map.get() — effectively free.
  try {
    const map = await fetchOkxInstrumentsMap();
    const meta = map.get(symbol);
    if (meta) {
      result.tick_size = meta.tickSz;
      result.lot_size = meta.lotSz;
      result.min_size = meta.minSz;
      result.contract_value = meta.ctVal;
    } else {
      // The /public/instruments call worked but the symbol isn't
      // in the map. Surface this loudly because it means either
      // (a) the symbol was delisted, or (b) the env profile points
      // at a symbol that OKX doesn't have. Either way the dashboard
      // banner would render dashes — log so the operator can see
      // why in the Next.js dev console.
      console.warn(
        `[venue-info] OKX instruments map has ${map.size} entries ` +
        `but no match for symbol=${JSON.stringify(symbol)}. ` +
        `Sample keys: ${[...map.keys()].slice(0, 3).join(", ")}`,
      );
    }
  } catch (e) {
    // The /public/instruments call itself failed (network / OKX
    // outage / unexpected payload shape). Log so the operator can
    // see it in the dev console; keep returning nulls so the
    // dashboard panel renders dashes rather than crashing.
    console.warn(
      `[venue-info] fetchOkxInstrumentsMap failed for ${symbol}: ${String(e)}`,
    );
  }

  // Position mode + account level (account-wide, single call).
  try {
    const cfgResp = await okxSignedGet<{
      posMode?: string;
      acctLv?: string;
    }>("/api/v5/account/config");
    if (cfgResp.code === "0") {
      const row = (cfgResp.data || [])[0];
      if (row) {
        result.position_mode = row.posMode || null;
        result.account_level = row.acctLv || null;
      }
    }
  } catch {
    // Best-effort; leave nulls.
  }

  // Portfolio-margin accounts (acctLv=4) have no per-symbol leverage
  // setting. The /account/leverage-info endpoint behaves
  // asymmetrically for these accounts: it rejects mgnMode=cross
  // queries with code 59111 ("Leverage query isn't supported in
  // portfolio margin account mode") but still returns a value for
  // mgnMode=isolated queries — likely a legacy / default record
  // that has NO effect on actual trading (PM computes required
  // margin via portfolio stress scenarios, not per-symbol leverage).
  //
  // Surfacing that ghost value as "3x isolated" misleads the
  // operator into thinking it constrains their bot. Skip the probe
  // entirely for PM accounts; leave leverage / margin_mode as null
  // so the dashboard's Position panel renders "—". The bot's
  // ``MAX_*`` config knobs and the PM risk model are what actually
  // constrain trading.
  if (result.account_level === "4") {
    return result;
  }

  // Per-symbol leverage + margin mode. Try cross first (the bot's
  // standard setup), fall back to isolated when cross returns nothing.
  for (const mode of ["cross", "isolated"] as const) {
    try {
      const lvResp = await okxSignedGet<{
        instId?: string;
        mgnMode?: string;
        lever?: string;
        posSide?: string;
      }>("/api/v5/account/leverage-info", {
        instId: symbol,
        mgnMode: mode,
      });
      if (lvResp.code === "0" && (lvResp.data || []).length > 0) {
        const row = lvResp.data[0];
        if (row.lever) {
          result.leverage = row.lever;
          result.margin_mode = row.mgnMode || mode;
          break;
        }
      }
    } catch {
      // Try the next mode.
    }
  }

  return result;
}

export async function fetchOkxAccount(): Promise<AccountSnapshot> {
  const resp = await okxSignedGet<{
    totalEq?: string;
    adjEq?: string;
    uid?: string;
    details?: Array<{
      ccy?: string;
      cashBal?: string;
      eq?: string;
      availBal?: string;
    }>;
  }>("/api/v5/account/balance");
  if (resp.code !== "0") {
    throw new Error(`OKX account/balance: code=${resp.code} msg=${resp.msg}`);
  }
  const row = resp.data?.[0] || {};
  const totalEq = parseFloat0(row.totalEq);
  const adjEq = parseFloat0(row.adjEq);
  let cash = 0;
  const details: AccountSnapshot["details"] = [];
  for (const d of row.details || []) {
    const ccy = String(d.ccy || "").toUpperCase();
    const cashBal = parseFloat0(d.cashBal);
    const eq = parseFloat0(d.eq);
    const availBal = parseFloat0(d.availBal);
    details.push({ ccy, cashBal, eq, availBal });
    if (ccy === "USDT" || ccy === "USDC" || ccy === "USD") {
      cash += cashBal;
    }
  }
  return {
    equity_usd: totalEq > 0 ? totalEq : null,
    cash_usd: cash > 0 ? cash : null,
    withdrawable_usd: adjEq > 0 ? adjEq : null,
    details,
    uid: row.uid,
  };
}

export async function fetchOkxPositions(
  symbol: string
): Promise<PositionSnapshot> {
  const ctVal = await fetchOkxCtVal(symbol);
  const resp = await okxSignedGet<{
    instId?: string;
    pos?: string;
    avgPx?: string;
    markPx?: string;
    upl?: string;
  }>("/api/v5/account/positions", { instId: symbol });
  if (resp.code !== "0") {
    throw new Error(
      `OKX account/positions: code=${resp.code} msg=${resp.msg}`
    );
  }
  // Convert OKX `pos` (CONTRACTS) → base-asset units up front so
  // every downstream consumer sees consistent units. ``ctVal=1``
  // for SUI / TON (no change); ``ctVal=0.1`` for HYPE etc. shrinks
  // ``position_qty`` to base-asset HYPE. ``notional_usd`` already
  // uses pos × price which after the conversion equals USD
  // notional directly (price is in USD/base, qty is in base).
  let posContracts = 0;
  let entry = 0;
  let mark = 0;
  let upl = 0;
  for (const r of resp.data || []) {
    if (r.instId !== symbol) continue;
    posContracts = parseFloat0(r.pos);
    entry = parseFloat0(r.avgPx);
    mark = parseFloat0(r.markPx);
    upl = parseFloat0(r.upl);
    break;
  }
  const pos = posContracts * ctVal;
  return {
    symbol,
    position_qty: pos,
    avg_entry_price: entry > 0 ? entry : null,
    mark_price: mark > 0 ? mark : null,
    unrealized_pnl_usd: upl,
    notional_usd: Math.abs(pos) * (mark > 0 ? mark : entry),
  };
}

export async function fetchOkxOpenOrders(symbol: string): Promise<OpenOrder[]> {
  const ctVal = await fetchOkxCtVal(symbol);
  const resp = await okxSignedGet<{
    instId?: string;
    ordId?: string;
    clOrdId?: string;
    side?: string;
    px?: string;
    sz?: string;
    state?: string;
    cTime?: string;
  }>("/api/v5/trade/orders-pending", { instType: "SWAP", instId: symbol });
  // FAIL CLOSED on non-zero OKX code. Pre-2026-05-16 we returned ``[]``
  // here, which made consumers indistinguishable between "no open
  // orders" and "OKX errored, we don't know". ``cancelAllOkxOrders``
  // then misreported "no open orders to cancel" when the venue was
  // unreachable -- the operator pressed Cancel, saw a green tick,
  // believed their orders had been pulled, and walked away. Throwing
  // here forces the caller (route handlers + cancel paths) to make
  // the partial-knowledge decision explicitly instead of guessing.
  // Codex review #3, 2026-05-16.
  if (resp.code !== "0") {
    throw new Error(
      `OKX orders-pending: code=${resp.code} msg=${resp.msg || ""}`,
    );
  }
  const out: OpenOrder[] = [];
  for (const r of resp.data || []) {
    if (r.instId !== symbol) continue;
    out.push({
      ord_id: String(r.ordId || ""),
      cl_ord_id: String(r.clOrdId || ""),
      side: r.side === "buy" ? "buy" : "sell",
      px: parseFloat0(r.px),
      // ``sz_contracts`` field name preserved for backward compat,
      // but the value now reflects BASE-ASSET quantity (contracts ×
      // ctVal). Most OKX USDT-M perps have ctVal=1 (SUI, TON), so
      // the value is unchanged; HYPE etc. with ctVal=0.1 now show
      // 0.1× the contract count, matching the rest of the
      // dashboard's base-asset convention.
      sz_contracts: parseFloat0(r.sz) * ctVal,
      state: String(r.state || ""),
      c_time_ms: Number(r.cTime || 0),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Write actions (cancel + flatten). Trading-grade creds only.
// ---------------------------------------------------------------------------

export interface ActionResult {
  outcome: "success" | "noop" | "error";
  detail: string;
  affected?: number;
}

/**
 * Cancel ALL open orders on `symbol`. OKX does not have a single-call
 * "cancel-all" endpoint analogous to Binance, so we fetch open orders
 * and POST to /api/v5/trade/cancel-batch-orders in chunks of 20
 * (OKX batch limit). Mirrors `app/exchange/okx_client.py::cancel_all_open_orders`.
 */
export async function cancelAllOkxOrders(
  symbol: string
): Promise<ActionResult> {
  // Pre-flight: list opens. ``fetchOkxOpenOrders`` now THROWS on
  // non-zero OKX code (Codex #3, 2026-05-16), so we wrap it in a
  // try/catch and surface the venue error as ``outcome: "error"``.
  // The previous fail-open behaviour silently returned ``noop`` on a
  // venue blip, falsely telling the operator "no orders to cancel" --
  // a dangerous lie if the bot was actively quoting.
  let opens: OpenOrder[];
  try {
    opens = await fetchOkxOpenOrders(symbol);
  } catch (e) {
    return {
      outcome: "error",
      detail: `cancel pre-flight failed (could not list opens): ${String(e)}`,
    };
  }
  if (opens.length === 0) {
    return { outcome: "noop", detail: "no open orders to cancel", affected: 0 };
  }
  const CHUNK = 20;
  let succeeded = 0;
  let lastErr = "";
  for (let i = 0; i < opens.length; i += CHUNK) {
    const slice = opens.slice(i, i + CHUNK);
    const body = slice.map((o) => ({ instId: symbol, ordId: o.ord_id }));
    const resp = await okxSignedRequest<{
      sCode?: string;
      sMsg?: string;
      ordId?: string;
    }>("POST", "/api/v5/trade/cancel-batch-orders", { body, write: true });
    if (resp.code !== "0") {
      lastErr = `code=${resp.code} msg=${resp.msg || ""}`;
      continue;
    }
    for (const row of resp.data || []) {
      if (String(row.sCode || "0") === "0") succeeded += 1;
    }
  }
  if (succeeded === 0) {
    return {
      outcome: "error",
      detail: `cancel batch failed: ${lastErr || "unknown"}`,
    };
  }
  if (succeeded < opens.length) {
    return {
      outcome: "error",
      detail: `partial: canceled ${succeeded} of ${opens.length} (${lastErr})`,
      affected: succeeded,
    };
  }
  return {
    outcome: "success",
    detail: `canceled ${succeeded} order(s)`,
    affected: succeeded,
  };
}

/**
 * Close (flatten) the open position on `symbol` via OKX dedicated
 * /api/v5/trade/close-position endpoint (reduce-only market-execution
 * in one call -- safer than building a market-IOC manually). Mirrors
 * `app/exchange/okx_client.py::market_close`.
 */
export async function closeOkxPosition(
  symbol: string
): Promise<ActionResult> {
  const pos = await fetchOkxPositions(symbol);
  if (Math.abs(pos.position_qty) <= 0) {
    return { outcome: "noop", detail: "position already flat" };
  }

  // 2026-05-22: OKX's ``/api/v5/trade/close-position`` rejects with
  // code=51115 "Cancel all pending close-orders before liquidation"
  // when there are open reduce-only orders on the symbol — common in
  // operator-incident scenarios where the bot is still trying to
  // place passive flatten orders. Cancel them first, then close.
  // The cancel-all is idempotent and a no-op when the book is clean,
  // so this adds at most one extra REST roundtrip in the normal
  // (already-flat-book) case.
  const cancelResult = await cancelAllOkxOrders(symbol);
  if (cancelResult.outcome === "error") {
    // Don't block the close attempt — the operator hit this button
    // because they want the position flat NOW. Cancel may have
    // partially succeeded; OKX may accept close-position anyway if
    // the remaining orders are not reduce-only-blocking. Log and
    // proceed; if close-position still fails we surface BOTH errors.
    // eslint-disable-next-line no-console
    console.error(
      `[close-position] pre-cancel failed: ${cancelResult.detail}`,
    );
  }

  const resp = await okxSignedRequest<{
    instId?: string;
    posSide?: string;
  }>("POST", "/api/v5/trade/close-position", {
    body: {
      instId: symbol,
      mgnMode: "cross",
      posSide: "net",
    },
    write: true,
  });
  if (resp.code !== "0") {
    const cancelNote =
      cancelResult.outcome === "error"
        ? ` (pre-cancel also failed: ${cancelResult.detail})`
        : cancelResult.outcome === "success"
        ? ` (after pre-cancel: ${cancelResult.detail})`
        : "";
    return {
      outcome: "error",
      detail: `okx code=${resp.code} msg=${resp.msg || ""}${cancelNote}`,
    };
  }

  const cancelPrefix =
    cancelResult.outcome === "success" && cancelResult.detail
      ? `${cancelResult.detail}; `
      : "";
  return {
    outcome: "success",
    detail: `${cancelPrefix}closed ${pos.position_qty} contracts on ${symbol}`,
  };
}


// ---------------------------------------------------------------------------
// History (orders + fills). Read-only -- prefers READONLY creds.
// ---------------------------------------------------------------------------

export interface OrderHistoryRecord {
  order_id: string;
  client_order_id: string;
  side: "BUY" | "SELL";
  type: string;
  price: number;
  orig_qty: number;
  filled_qty: number;
  status: string;
  time_ms: number;
  notional_usd: number;
}

export interface FillHistoryRecord {
  trade_id: string;
  order_id: string;
  side: "BUY" | "SELL";
  price: number;
  qty: number;
  notional_usd: number;
  fee: number;
  fee_ccy: string;
  is_maker: boolean;
  realized_pnl: number;
  time_ms: number;
  /** Venue instId / symbol of the fill. Carried through so the
   *  account-wide path (``symbol === null``) can group by token for
   *  the per-token volume tab. Single-symbol callers just see the
   *  symbol they queried. Empty string when the venue omits it. */
  symbol: string;
}

/**
 * Last N orders on `symbol`. OKX `/api/v5/trade/orders-history` covers
 * the past 7 days; deeper history would need `orders-history-archive`
 * + pagination, which is v2.
 *
 * Returns BASE-ASSET quantities (not contracts) — applies the
 * symbol's ctVal at parse time so all downstream consumers
 * (display, risk-violation checks, notional rollups) see consistent
 * units. ctVal lookup is cached per process for 5 min.
 */
export async function fetchOkxOrdersHistory(
  symbol: string,
  limit: number = 50
): Promise<OrderHistoryRecord[]> {
  const ctVal = await fetchOkxCtVal(symbol);
  const resp = await okxSignedRequest<{
    instId?: string;
    ordId?: string;
    clOrdId?: string;
    side?: string;
    ordType?: string;
    px?: string;
    sz?: string;
    accFillSz?: string;
    state?: string;
    uTime?: string;
    cTime?: string;
  }>("GET", "/api/v5/trade/orders-history", {
    params: { instType: "SWAP", instId: symbol, limit: String(limit) },
  });
  if (resp.code !== "0") return [];
  return (resp.data || [])
    .map((r) => {
      const price = parseFloat0(r.px);
      const origContracts = parseFloat0(r.sz);
      const filledContracts = parseFloat0(r.accFillSz);
      // Convert contracts → base units up-front. price × base ==
      // USD notional directly, no further multiplication needed.
      const origBase = origContracts * ctVal;
      const filledBase = filledContracts * ctVal;
      // OKX state values: live, partially_filled, filled, canceled,
      // mmp_canceled. Normalise to upper for consistency with Binance.
      const stateRaw = String(r.state || "").toLowerCase();
      const status = stateRaw === "live" ? "NEW" :
        stateRaw === "partially_filled" ? "PARTIALLY_FILLED" :
        stateRaw === "filled" ? "FILLED" :
        stateRaw === "canceled" || stateRaw === "mmp_canceled" ? "CANCELED" :
        stateRaw.toUpperCase();
      return {
        order_id: String(r.ordId || ""),
        client_order_id: String(r.clOrdId || ""),
        side: r.side === "buy" ? "BUY" : ("SELL" as "BUY" | "SELL"),
        type: String(r.ordType || ""),
        price,
        orig_qty: origBase,
        filled_qty: filledBase,
        status,
        time_ms: Number(r.uTime || r.cTime || 0),
        notional_usd: price * origBase,
      };
    })
    .sort((a, b) => b.time_ms - a.time_ms);
}

/**
 * Last N fills (executions) on `symbol`. Includes `fee`, `feeCcy`,
 * and `execType` (T = taker, M = maker).
 *
 * Returns BASE-ASSET quantities (not contracts) — applies the
 * symbol's ctVal at parse time. See ``fetchOkxOrdersHistory`` for
 * the unit-convention rationale.
 *
 * 2026-05-14 BUG-024 fix: pulls from BOTH OKX fill endpoints in
 * parallel and merges the result so the dashboard shows fills
 * older than 3 days. Pre-fix this called only ``/api/v5/trade/fills``
 * which has a 3-day retention window, so after stopping a bot for
 * 4+ days and restarting on a different profile, the Fill History
 * tab appeared empty even though the venue still had ~3 months of
 * history available via ``/api/v5/trade/fills-history`` (archive).
 *
 * Cost: two OKX REST calls per fetch (vs one before). Both rate-
 * limited tiers are well under the dashboard's 15 s polling cadence.
 */
function _parseOkxFillRow(
  r: {
    instId?: string;
    ordId?: string;
    tradeId?: string;
    side?: string;
    fillSz?: string;
    fillPx?: string;
    fee?: string;
    feeCcy?: string;
    execType?: string;
    fillPnl?: string;
    ts?: string;
  },
  ctVal: number,
  fallbackSymbol: string,
): FillHistoryRecord {
  const price = parseFloat0(r.fillPx);
  const qtyContracts = parseFloat0(r.fillSz);
  const qty = qtyContracts * ctVal; // base-asset units
  // OKX ``fee`` is signed: NEGATIVE means rebate (we received), POSITIVE
  // means cost. Negate so the dashboard's convention (positive=paid,
  // negative=rebate-received) reads naturally on screen.
  const fee = parseFloat0(r.fee);
  return {
    trade_id: String(r.tradeId || ""),
    order_id: String(r.ordId || ""),
    side: r.side === "buy" ? "BUY" : ("SELL" as "BUY" | "SELL"),
    price,
    qty,
    notional_usd: price * qty,
    fee: -fee,
    fee_ccy: String(r.feeCcy || ""),
    is_maker: r.execType === "M",
    realized_pnl: parseFloat0(r.fillPnl),
    time_ms: Number(r.ts || 0),
    symbol: String(r.instId || fallbackSymbol || ""),
  };
}

export async function fetchOkxFillsHistory(
  symbol: string,
  limit: number = 50
): Promise<FillHistoryRecord[]> {
  const ctVal = await fetchOkxCtVal(symbol);
  // The two endpoints' response rows have identical shape (verified
  // against OKX V5 docs: /trade/fills and /trade/fills-history return
  // the same fields). Parser is shared via _parseOkxFillRow above.
  const sharedParams = {
    instType: "SWAP",
    instId: symbol,
    limit: String(limit),
  };
  // Call both in parallel. If either fails, fall back to the other's
  // results rather than returning empty — partial visibility beats
  // a broken tab.
  const [recentResult, archiveResult] = await Promise.allSettled([
    okxSignedRequest<{
      instId?: string;
      ordId?: string;
      tradeId?: string;
      side?: string;
      fillSz?: string;
      fillPx?: string;
      fee?: string;
      feeCcy?: string;
      execType?: string;
      fillPnl?: string;
      ts?: string;
    }>("GET", "/api/v5/trade/fills", { params: sharedParams }),
    okxSignedRequest<{
      instId?: string;
      ordId?: string;
      tradeId?: string;
      side?: string;
      fillSz?: string;
      fillPx?: string;
      fee?: string;
      feeCcy?: string;
      execType?: string;
      fillPnl?: string;
      ts?: string;
    }>("GET", "/api/v5/trade/fills-history", { params: sharedParams }),
  ]);
  const merged = new Map<string, FillHistoryRecord>();
  if (recentResult.status === "fulfilled" && recentResult.value.code === "0") {
    for (const r of recentResult.value.data || []) {
      const parsed = _parseOkxFillRow(r, ctVal, symbol);
      if (parsed.trade_id) merged.set(parsed.trade_id, parsed);
    }
  }
  if (archiveResult.status === "fulfilled" && archiveResult.value.code === "0") {
    for (const r of archiveResult.value.data || []) {
      const parsed = _parseOkxFillRow(r, ctVal, symbol);
      // Newer endpoint wins on dupes (it has lower latency for fills
      // within its 3-day window; archive may lag by ~1 min).
      if (parsed.trade_id && !merged.has(parsed.trade_id)) {
        merged.set(parsed.trade_id, parsed);
      }
    }
  }
  return Array.from(merged.values())
    .sort((a, b) => b.time_ms - a.time_ms)
    .slice(0, limit);
}

/**
 * Public market candles for ``symbol`` at ``bar`` interval.
 * Unauthenticated -- ``/api/v5/market/candles`` is public.
 *
 * Bar shorthand: "1m", "3m", "5m", "15m", "30m", "1H", "2H",
 * "4H", "1D", etc. OKX's parameter is ``bar``.
 */
export interface Candle {
  time_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function fetchOkxKlines(
  symbol: string,
  bar: string = "1m",
  limit: number = 100
): Promise<Candle[]> {
  const url = `${OKX_REST}/api/v5/market/candles?instId=${encodeURIComponent(
    symbol
  )}&bar=${encodeURIComponent(bar)}&limit=${limit}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`okx_klines: HTTP ${r.status}`);
  const j = (await r.json()) as {
    code: string;
    msg: string;
    data: string[][];
  };
  if (j.code !== "0") throw new Error(`okx_klines: ${j.msg || j.code}`);
  // OKX returns rows as [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
  // -- string-typed. Newest first; reverse so chart x-axis is left=old.
  return (j.data || [])
    .map((row) => ({
      time_ms: Number(row[0] || 0),
      open: parseFloat0(row[1]),
      high: parseFloat0(row[2]),
      low: parseFloat0(row[3]),
      close: parseFloat0(row[4]),
      volume: parseFloat0(row[5]),
    }))
    .sort((a, b) => a.time_ms - b.time_ms);
}

/**
 * Paginated klines walk — assembles ``totalBars`` candles oldest-first.
 *
 * The single-shot ``fetchOkxKlines`` is capped at ~300 rows per call
 * by OKX V5. Anything longer (e.g. 14 days × 15m = 1344 bars for the
 * Market Activity Calendar) must page backwards using the ``after``
 * query parameter — OKX docs:
 *
 *     "after" — Pagination of data to return records earlier than
 *               the requested ts. Default: returns the most recent
 *               records.
 *
 * So with no ``after`` we get the newest 300; pass the oldest row's
 * ts as ``after`` next call to walk further back; repeat.
 *
 * Used by ``/api/accounts/[id]/activity-calendar``. Pagination
 * pattern identical to ``fetchOkxMarkPriceCandles`` above.
 *
 * @param totalBars  Target window in bars. Function may return up to
 *                   one page (~300) MORE than requested to avoid
 *                   truncating mid-day at the boundary.
 * @param maxPages   Safety cap. Default 10 (= up to 3000 bars at
 *                   page=300) which covers ≥ 14 days at 15m.
 */
export async function fetchOkxKlinesPaginated(
  symbol: string,
  bar: string = "1m",
  totalBars: number = 1500,
  maxPages: number = 10,
): Promise<Candle[]> {
  const pageSize = 300;
  const out: Candle[] = [];
  let afterTs = "";
  for (let page = 0; page < maxPages; page++) {
    const url = new URL(`${OKX_REST}/api/v5/market/candles`);
    url.searchParams.set("instId", symbol);
    url.searchParams.set("bar", bar);
    url.searchParams.set("limit", String(pageSize));
    if (afterTs) {
      url.searchParams.set("after", afterTs);
    }
    const r = await fetch(url.toString(), { cache: "no-store" });
    if (!r.ok) {
      throw new Error(`okx_klines_paginated: HTTP ${r.status}`);
    }
    const j = (await r.json()) as {
      code: string;
      msg: string;
      data: string[][];
    };
    if (j.code !== "0") {
      throw new Error(`okx_klines_paginated: ${j.msg || j.code}`);
    }
    const rows = j.data || [];
    if (rows.length === 0) break;
    // Row schema same as ``fetchOkxKlines`` above:
    // [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    for (const row of rows) {
      out.push({
        time_ms: Number(row[0] || 0),
        open: parseFloat0(row[1]),
        high: parseFloat0(row[2]),
        low: parseFloat0(row[3]),
        close: parseFloat0(row[4]),
        volume: parseFloat0(row[5]),
      });
    }
    // Stop early once we have enough — caller doesn't need more.
    if (out.length >= totalBars) break;
    // Newest-first within each page; the oldest row of this page is
    // the earliest ts seen so far. Use it as the next page's anchor.
    const oldestTsThisPage = Number(rows[rows.length - 1][0] || 0);
    if (!Number.isFinite(oldestTsThisPage) || oldestTsThisPage <= 0) break;
    afterTs = String(oldestTsThisPage);
  }
  // Chronological order + dedupe (defensive — adjacent pages can
  // briefly overlap during a fill).
  out.sort((a, b) => a.time_ms - b.time_ms);
  const dedup: Candle[] = [];
  let lastTs = -Infinity;
  for (const c of out) {
    if (c.time_ms === lastTs) continue;
    dedup.push(c);
    lastTs = c.time_ms;
  }
  return dedup;
}

/** Mark-price candle: ts + close. We deliberately drop o/h/l because
 *  the dashboard's overlay use case only needs the per-bar mark, not
 *  the candle's range. */
export interface OkxMarkPriceCandle {
  time_ms: number;
  mark: number;
}

/**
 * Public mark-price-candles paginated fetch.
 * Endpoint: ``/api/v5/market/mark-price-candles`` (NOT ``/public/...`` —
 * OKX keeps the candles surface under ``/market/`` despite the
 * underlying mark-price spot value living at ``/public/mark-price``.
 * Verified live 2026-05-14: ``/public/mark-price-candles`` returns
 * a 404 page, while the ``/market/`` variant returns the candle
 * data documented in OKX V5).
 *
 * Public / unauthenticated — no API key needed. Rolling 3-month
 * archive. Same row shape as ``/api/v5/market/candles`` but for the
 * venue's COMPUTED mark price (smoothed, used for liquidations &
 * position valuation), not the trade tape.
 *
 * Pagination: OKX caps each response at 100 rows; newest-first. We
 * pull pages backward (the OKX ``after`` parameter means "records
 * EARLIER than ts") until we've covered ``sinceMs`` or hit the
 * safety cap.
 *
 * Used by the dashboard's mark-price overlay on the Session-PnL
 * chart's Mid sub-band (Q1 from the 2026-05-14 funding-rate / mark-
 * price thread).
 */
export async function fetchOkxMarkPriceCandles(
  symbol: string,
  bar: string = "1m",
  sinceMs: number,
  maxPages: number = 25,
): Promise<OkxMarkPriceCandle[]> {
  // ``sinceMs`` is the OLDEST ts we want covered (inclusive). The
  // OKX endpoint returns newest-first, so we walk backwards using
  // ``after`` until the response's oldest row is <= sinceMs OR we
  // hit maxPages (safety against unbounded loops).
  const out: OkxMarkPriceCandle[] = [];
  // Empty string = no `after` parameter on the first page (returns
  // most recent 100). Subsequent pages pass the oldest ts from the
  // last response.
  let afterTs = "";
  for (let page = 0; page < maxPages; page++) {
    const url = new URL(`${OKX_REST}/api/v5/market/mark-price-candles`);
    url.searchParams.set("instId", symbol);
    url.searchParams.set("bar", bar);
    url.searchParams.set("limit", "100");
    if (afterTs) {
      url.searchParams.set("after", afterTs);
    }
    const r = await fetch(url.toString(), { cache: "no-store" });
    if (!r.ok) {
      throw new Error(`okx_mark_price_candles: HTTP ${r.status}`);
    }
    const j = (await r.json()) as {
      code: string;
      msg: string;
      data: string[][];
    };
    if (j.code !== "0") {
      throw new Error(`okx_mark_price_candles: ${j.msg || j.code}`);
    }
    const rows = j.data || [];
    if (rows.length === 0) break;
    // Each row: [ts, o, h, l, c, confirm] — confirm = "0" (still
    // building) or "1" (closed). We take ``close`` (index 4).
    for (const row of rows) {
      const tsMs = Number(row[0] || 0);
      const close = parseFloat0(row[4]);
      if (!Number.isFinite(tsMs) || tsMs <= 0) continue;
      if (!Number.isFinite(close) || close <= 0) continue;
      out.push({ time_ms: tsMs, mark: close });
    }
    // Newest-first → the last (oldest) row of this page is the
    // earliest ts we've seen so far.
    const oldestTsThisPage = Number(rows[rows.length - 1][0] || 0);
    if (!Number.isFinite(oldestTsThisPage) || oldestTsThisPage <= 0) break;
    // Stop once we've covered the requested window. Allow one
    // extra buffer-page-worth of slack so the chart's left edge
    // doesn't look truncated.
    if (oldestTsThisPage <= sinceMs) break;
    afterTs = String(oldestTsThisPage);
  }
  // Sort chronological for the chart's left=old / right=new axis.
  out.sort((a, b) => a.time_ms - b.time_ms);
  // Dedup by ts in case overlapping pages leaked a row.
  const dedup: OkxMarkPriceCandle[] = [];
  let lastTs = -Infinity;
  for (const c of out) {
    if (c.time_ms === lastTs) continue;
    dedup.push(c);
    lastTs = c.time_ms;
  }
  return dedup;
}

/** Index-price candle row from ``/api/v5/market/index-candles``.
 *  Same shape + same pagination semantics as ``OkxMarkPriceCandle``;
 *  used together with the mark-price candles to reconstruct
 *  ``premium = (mark - index) / index`` per minute for the dashboard's
 *  funding-rate predicted trace (1.3.60 todo-funding-history). */
export interface OkxIndexCandle {
  time_ms: number;
  index: number;
}

/**
 * Public index-price-candles paginated fetch. Mirror of
 * ``fetchOkxMarkPriceCandles`` for the index price feed. Same
 * pagination pattern: ``after`` walks backward in time, max 100 rows
 * per response, ``maxPages`` is the safety cap.
 *
 * Used by the dashboard's funding-rate-estimate history route to
 * reconstruct historical ``premium = (mark - index) / index`` per
 * minute when the bot first comes up (or after a refresh) so the
 * predicted-rate dashed line on the Funding sub-band has a populated
 * curve back to session start instead of an empty trace.
 */
export async function fetchOkxIndexCandles(
  symbol: string,
  bar: string = "1m",
  sinceMs: number,
  maxPages: number = 25,
): Promise<OkxIndexCandle[]> {
  const out: OkxIndexCandle[] = [];
  let afterTs = "";
  for (let page = 0; page < maxPages; page++) {
    const url = new URL(`${OKX_REST}/api/v5/market/index-candles`);
    url.searchParams.set("instId", symbol);
    url.searchParams.set("bar", bar);
    url.searchParams.set("limit", "100");
    if (afterTs) {
      url.searchParams.set("after", afterTs);
    }
    const r = await fetch(url.toString(), { cache: "no-store" });
    if (!r.ok) {
      throw new Error(`okx_index_candles: HTTP ${r.status}`);
    }
    const j = (await r.json()) as {
      code: string;
      msg: string;
      data: string[][];
    };
    if (j.code !== "0") {
      throw new Error(`okx_index_candles: ${j.msg || j.code}`);
    }
    const rows = j.data || [];
    if (rows.length === 0) break;
    for (const row of rows) {
      const tsMs = Number(row[0] || 0);
      const close = parseFloat0(row[4]);
      if (!Number.isFinite(tsMs) || tsMs <= 0) continue;
      if (!Number.isFinite(close) || close <= 0) continue;
      out.push({ time_ms: tsMs, index: close });
    }
    const oldestTsThisPage = Number(rows[rows.length - 1][0] || 0);
    if (!Number.isFinite(oldestTsThisPage) || oldestTsThisPage <= 0) break;
    if (oldestTsThisPage <= sinceMs) break;
    afterTs = String(oldestTsThisPage);
  }
  out.sort((a, b) => a.time_ms - b.time_ms);
  const dedup: OkxIndexCandle[] = [];
  let lastTs = -Infinity;
  for (const c of out) {
    if (c.time_ms === lastTs) continue;
    dedup.push(c);
    lastTs = c.time_ms;
  }
  return dedup;
}

/** A historical predicted-funding-rate sample. ``rate`` is the decimal
 *  fraction (0.0001 = 0.01%). Computed offline from venue public
 *  candle data; sample cadence matches the underlying candle ``bar``
 *  (typically 1m). Used by the dashboard to seed the funding-band's
 *  predicted-rate trace back to session start so the curve never goes
 *  blank on tab switch / page refresh. */
export interface FundingRateEstimateSample {
  time_ms: number;
  rate: number;
}

/**
 * Historical predicted-funding-rate reconstruction for an OKX perp.
 *
 * OKX doesn't expose historical ``nextFundingRate`` values, so we
 * reconstruct the trajectory from the public mark-price + index-price
 * candle feeds: ``premium = (mark - index) / index`` per minute. This
 * is the *instantaneous premium*, not the TWAP-adjusted predicted
 * rate that ``fetchOkxFundingRateEstimate`` returns for the live tail
 * — same proxy Binance's UI uses for the visual display. Visually
 * conveys the same trajectory; converges to the settled rate at each
 * funding settlement.
 *
 * Two REST calls in parallel (mark + index candles, each paginating
 * 100 rows per page). 24 h coverage = ~1440 rows per stream = ~15
 * pages each; safety-capped at 25 pages per stream.
 */
export async function fetchOkxFundingRateEstimateHistory(
  symbol: string,
  sinceMs: number,
): Promise<FundingRateEstimateSample[]> {
  // OKX endpoints take DIFFERENT symbols:
  //   • mark-price-candles → perp instId, e.g. ``SUI-USDT-SWAP``
  //   • index-candles      → index symbol, e.g. ``SUI-USDT``
  //                          (the spot pair that the perp tracks)
  // Strip the ``-SWAP`` suffix to derive the index symbol — works
  // for every USDT-denominated OKX perp we currently support.
  const indexSymbol = symbol.toUpperCase().replace(/-SWAP$/, "");
  const [marks, indexes] = await Promise.all([
    fetchOkxMarkPriceCandles(symbol, "1m", sinceMs),
    fetchOkxIndexCandles(indexSymbol, "1m", sinceMs),
  ]);
  const indexByTs = new Map<number, number>();
  for (const c of indexes) indexByTs.set(c.time_ms, c.index);
  // 1.3.62: compute per-minute predicted rate as a TWAP of premium
  // over the elapsed portion of the current funding interval, plus
  // the standard funding-rate clamp. Earlier 1.3.60 implementation
  // plotted the INSTANTANEOUS premium per minute, which flickers
  // minute-to-minute and bears no resemblance to the smoothed
  // predicted-rate value Binance / OKX UIs display. The TWAP form
  // matches each venue's published formula (Binance is exact to
  // ~1 bp; OKX uses the same clamp constants in practice).
  const perMinute: Array<{ time_ms: number; premium: number }> = [];
  for (const m of marks) {
    if (m.time_ms < sinceMs) continue;
    const idx = indexByTs.get(m.time_ms);
    if (idx === undefined || idx <= 0) continue;
    const premium = (m.mark - idx) / idx;
    if (!Number.isFinite(premium)) continue;
    perMinute.push({ time_ms: m.time_ms, premium });
  }
  return _twapPerInterval(perMinute);
}

/** TWAP-and-clamp helper shared by both venues' history reconstructions.
 *  Walks per-minute premium samples in chronological order, maintains a
 *  running sum within each 8 h funding interval (reset at UTC 00:00 /
 *  08:00 / 16:00 boundaries), applies the standard funding-rate
 *  formula ``predicted = TWAP + clamp(I - TWAP, ±0.05%)`` per minute.
 *  Output is one sample per input minute, with the predicted rate that
 *  would have been paid had settlement happened at that minute. */
function _twapPerInterval(
  perMinute: Array<{ time_ms: number; premium: number }>,
): FundingRateEstimateSample[] {
  const INTERVAL_MS = 8 * 60 * 60 * 1000;
  // Standard 8 h Binance / OKX constants. Interest rate per
  // interval = 0.01 %; symmetric clamp ±0.05 % per interval.
  const INTEREST_RATE = 0.0001;
  const CLAMP_LIMIT = 0.0005;
  const sorted = [...perMinute].sort((a, b) => a.time_ms - b.time_ms);
  const out: FundingRateEstimateSample[] = [];
  let runningSum = 0;
  let runningCount = 0;
  let currentPeriodStart = -1;
  for (const p of sorted) {
    const periodStart = Math.floor(p.time_ms / INTERVAL_MS) * INTERVAL_MS;
    if (periodStart !== currentPeriodStart) {
      runningSum = 0;
      runningCount = 0;
      currentPeriodStart = periodStart;
    }
    runningSum += p.premium;
    runningCount += 1;
    const twap = runningSum / runningCount;
    const clamped = Math.max(
      -CLAMP_LIMIT,
      Math.min(CLAMP_LIMIT, INTEREST_RATE - twap),
    );
    out.push({ time_ms: p.time_ms, rate: twap + clamped });
  }
  return out;
}

// Exported for the Binance-side history reconstruction in
// ``lib/binance.ts`` — same TWAP+clamp math applies regardless of
// which venue produced the per-minute premium stream.
export { _twapPerInterval as twapFundingRateEstimatePerInterval };

/** Funding-rate tick. Funding intervals on OKX perp are 8 h (00:00 /
 *  08:00 / 16:00 UTC). ``rate`` is the decimal fraction (e.g.
 *  ``-0.0000559`` = -0.00559 % per 8 h). Conversion to bp / % happens
 *  at the dashboard layer. */
export interface OkxFundingRateTick {
  time_ms: number;
  rate: number;
}

/**
 * Public funding-rate history paginated fetch.
 * Endpoint: ``/api/v5/public/funding-rate-history``.
 *
 * Public / unauthenticated — no API key needed. Rolling archive.
 * OKX returns rows newest-first; we paginate backwards using
 * ``after`` until we've covered ``sinceMs`` OR hit ``maxPages``.
 *
 * Used by the dashboard's funding-rate sub-band on the Session-PnL
 * chart (Q2 from the 2026-05-14 funding-rate / mark-price thread).
 */
export async function fetchOkxFundingRateHistory(
  symbol: string,
  sinceMs: number,
  maxPages: number = 10,
): Promise<OkxFundingRateTick[]> {
  const out: OkxFundingRateTick[] = [];
  let afterTs = "";
  for (let page = 0; page < maxPages; page++) {
    const url = new URL(
      `${OKX_REST}/api/v5/public/funding-rate-history`,
    );
    url.searchParams.set("instId", symbol);
    url.searchParams.set("limit", "100");
    if (afterTs) {
      url.searchParams.set("after", afterTs);
    }
    const r = await fetch(url.toString(), { cache: "no-store" });
    if (!r.ok) {
      throw new Error(`okx_funding_rate_history: HTTP ${r.status}`);
    }
    const j = (await r.json()) as {
      code: string;
      msg: string;
      data: Array<{
        fundingTime?: string;
        realizedRate?: string;
        fundingRate?: string;
      }>;
    };
    if (j.code !== "0") {
      throw new Error(
        `okx_funding_rate_history: ${j.msg || j.code}`,
      );
    }
    const rows = j.data || [];
    if (rows.length === 0) break;
    for (const row of rows) {
      const tsMs = Number(row.fundingTime || 0);
      // Prefer ``realizedRate`` (what actually got paid) over the
      // predicted ``fundingRate`` when both are present. OKX
      // sometimes caps the predicted rate; the realized value is
      // the operator-facing truth.
      const rateStr = row.realizedRate || row.fundingRate || "";
      const rate = parseFloat0(rateStr);
      if (!Number.isFinite(tsMs) || tsMs <= 0) continue;
      if (!Number.isFinite(rate)) continue;
      out.push({ time_ms: tsMs, rate });
    }
    const oldestTsThisPage = Number(
      rows[rows.length - 1].fundingTime || 0,
    );
    if (!Number.isFinite(oldestTsThisPage) || oldestTsThisPage <= 0) break;
    if (oldestTsThisPage <= sinceMs) break;
    afterTs = String(oldestTsThisPage);
  }
  out.sort((a, b) => a.time_ms - b.time_ms);
  return out;
}

/** Venue-agnostic predicted-funding-rate estimate. Each venue's
 *  fetcher returns this same shape so the dashboard route can stitch
 *  target + reference together without per-venue branching downstream.
 *  Rate is the decimal fraction Binance / OKX use (0.0001 = 0.01 %).
 *  ``method`` is opaque provenance for the tooltip — different venues
 *  source the estimate differently (OKX exposes nextFundingRate
 *  directly; Binance has to be TWAP-reconstructed). */
export interface FundingRateEstimate {
  predicted_rate: number;
  next_funding_ms: number;
  funding_interval_ms: number;
  method: string;
  fetched_at_ms: number;
}

/**
 * Predicted next-period funding rate for an OKX perpetual.
 * Endpoint: ``/api/v5/public/funding-rate``.
 *
 * OKX computes the estimate itself and exposes it as
 * ``nextFundingRate`` — no client-side reconstruction needed. The
 * value updates approximately every 8 minutes server-side as the
 * premium index moves and is the same number the OKX web UI shows.
 *
 * Public / unauthenticated. Single call. Used by the dashboard's
 * funding-rate sub-band predicted-rate trace.
 */
export async function fetchOkxFundingRateEstimate(
  symbol: string,
): Promise<FundingRateEstimate> {
  const url = new URL(`${OKX_REST}/api/v5/public/funding-rate`);
  url.searchParams.set("instId", symbol);
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`okx_funding_rate: HTTP ${r.status}`);
  }
  const j = (await r.json()) as {
    code: string;
    msg: string;
    data: Array<{
      fundingRate?: string;
      nextFundingRate?: string;
      fundingTime?: string;
      nextFundingTime?: string;
    }>;
  };
  if (j.code !== "0" || !j.data || j.data.length === 0) {
    throw new Error(`okx_funding_rate: ${j.msg || j.code}`);
  }
  const row = j.data[0];
  // OKX field semantics (per /docs-v5 #public-data-rest-api-get-
  // funding-rate):
  //
  //   • ``fundingTime``     = settlement time of the CURRENT period
  //                            — the next upcoming settlement.
  //   • ``nextFundingTime`` = settlement time of the period AFTER
  //                            the current one — one interval beyond.
  //   • ``fundingRate``     = predicted rate for the upcoming
  //                            settlement.
  //   • ``nextFundingRate`` = predicted rate for the settlement
  //                            AFTER that.
  //
  // For the dashboard's countdown + predicted-rate trace we want
  // the IMMEDIATE next settlement, i.e. ``fundingTime`` and
  // ``fundingRate``. (Original v1.3.51 implementation used the
  // ``next*`` pair, which made every OKX countdown show ~12-16 h
  // instead of 0-8 h on an 8 h-cadence pair.)
  const nextFundingMs = Number(row.fundingTime || 0);
  const subsequentFundingMs = Number(row.nextFundingTime || 0);
  const intervalMs =
    subsequentFundingMs > 0 && nextFundingMs > 0
      ? subsequentFundingMs - nextFundingMs
      : 8 * 3_600_000;
  const predictedRate = parseFloat0(row.fundingRate || "");
  if (!Number.isFinite(nextFundingMs) || nextFundingMs <= 0) {
    throw new Error("okx_funding_rate: missing fundingTime");
  }
  return {
    predicted_rate: predictedRate,
    next_funding_ms: nextFundingMs,
    funding_interval_ms: intervalMs,
    method: "okx_current_period_funding_rate",
    fetched_at_ms: Date.now(),
  };
}

/** 24h ticker summary — last price, 24h volume (base + USD), high / low.
 *  Public unauthenticated endpoint; same source the OKX UI shows. */
export interface OkxTicker24h {
  last: number;
  high24h: number;
  low24h: number;
  /** Base-currency volume over the trailing 24 hours (e.g. SUI count). */
  vol24h: number;
  /** Notional 24h volume in the QUOTE currency (USDT). For perp swaps
   *  ``volCcy24h`` is contract×ctVal — already in base units; OKX's
   *  ``volCcyQuote24h`` is the USDT figure we typically want for the
   *  display. We surface both so the caller can pick. */
  vol_ccy_quote_24h: number;
  /** Price change over the last 24h, in bps of last price. */
  change_24h_bps: number;
}

export async function fetchOkxTicker24h(symbol: string): Promise<OkxTicker24h> {
  const url = `${OKX_REST}/api/v5/market/ticker?instId=${encodeURIComponent(symbol)}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`okx_ticker24h: HTTP ${r.status}`);
  const j = (await r.json()) as {
    code: string;
    msg: string;
    data: Array<Record<string, string>>;
  };
  if (j.code !== "0") throw new Error(`okx_ticker24h: ${j.msg || j.code}`);
  const row = (j.data && j.data[0]) || {};
  const last = parseFloat0(row.last || row.lastPx || "0");
  const open24h = parseFloat0(row.open24h || "0");
  const change_bps =
    open24h > 0 && last > 0 ? ((last - open24h) / open24h) * 10_000 : 0;
  return {
    last,
    high24h: parseFloat0(row.high24h || "0"),
    low24h: parseFloat0(row.low24h || "0"),
    vol24h: parseFloat0(row.vol24h || "0"),
    vol_ccy_quote_24h: parseFloat0(row.volCcy24h || "0"),
    change_24h_bps: change_bps,
  };
}

/** Open-interest snapshot (perp / futures only — spot returns empty).
 *  ``oi`` is the contract count; ``oiCcy`` is the base-asset notional;
 *  ``oiUsd`` is the USD notional. We expose the USD figure as the
 *  default since it's directly comparable across symbols. */
export interface OkxOpenInterest {
  oi_contracts: number;
  oi_base: number;
  oi_usd: number;
}

export async function fetchOkxOpenInterest(
  symbol: string,
): Promise<OkxOpenInterest> {
  const url = `${OKX_REST}/api/v5/public/open-interest?instId=${encodeURIComponent(symbol)}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`okx_open_interest: HTTP ${r.status}`);
  const j = (await r.json()) as {
    code: string;
    msg: string;
    data: Array<Record<string, string>>;
  };
  if (j.code !== "0") throw new Error(`okx_open_interest: ${j.msg || j.code}`);
  const row = (j.data && j.data[0]) || {};
  return {
    oi_contracts: parseFloat0(row.oi || "0"),
    oi_base: parseFloat0(row.oiCcy || "0"),
    oi_usd: parseFloat0(row.oiUsd || "0"),
  };
}

/**
 * Paginated long-window fills fetch via OKX's archived endpoint
 * ``/api/v5/trade/fills-history`` (rolling 3 months). Used by the
 * dashboard's PnL-history chart -- ``fetchOkxFillsHistory`` above
 * uses ``/trade/fills`` which only covers a few days.
 *
 * Pagination: uses ``after`` (NOT ``before`` -- OKX semantic is
 * "records after this trade-id", meaning OLDER). We loop:
 *   1. fetch newest 100
 *   2. note oldest tradeId in batch
 *   3. fetch next batch with after=<that tradeId>
 *   4. stop when batch is empty OR oldest ts < cutoff
 *
 * Limit: hard cap on iterations to avoid runaway loops.
 *
 * 2026-05-15: bumped from 100 → 300. The the partner sub-account on OKX
 * now does ~700-1000 fills/day (was ~50/day when the original cap
 * was set). At 100 pages × 100 fills = 10k fills, the 30d window
 * saturated at the cap — the dashboard's "Total Volume" tile showed
 * 15d and 30d at the SAME number (10000 fills, $165k) for days. At
 * 300 pages = 30k fills cap, covers ~30 days at 1000 fills/day.
 *
 * Walk time at 500ms inter-page delay = 150s max on cold-cache.
 * Acceptable: blocks only the first caller, single-flight dedup
 * means concurrent callers attach to the same promise. After one
 * successful complete walk, the route's 30-min stale-while-revalidate
 * cache absorbs subsequent 60s cache misses.
 *
 * If 30k starts saturating too (rare, would require sustained
 * >1000 fills/day for >30 days), raise to 500. Walk time scales
 * linearly: 500 pages = 250s.
 */
const PNL_HISTORY_MAX_ITERATIONS = 300;

export interface PaginatedFillsResult {
  fills: FillHistoryRecord[];
  /** True when we have ALL fills the venue knows about between
   *  ``since_ms`` and now. Two ways this is True:
   *    1. The venue returned an empty page (no fills older than
   *       what we already saw).
   *    2. The pagination crossed the ``since_ms`` cutoff (we got
   *       everything we asked for; older fills may exist BEFORE
   *       the cutoff but they're outside the requested window).
   *  False only when pagination stopped early due to a safety
   *  cap (iteration count) or a missing pagination cursor in the
   *  response -- those are the cases where the result is truly
   *  under-counted. */
  complete: boolean;
}

// Inter-page delay for the OKX fills-history paginator. The
// ``/api/v5/trade/fills-history`` endpoint enforces ~5 req / 2 s
// per UID (empirically — OKX's docs are vague). 100ms between
// pages was 2× too aggressive and consistently tripped 50011 Too
// Many Requests on page 6 (5 successful pages × 100ms = 500ms,
// crossing the 5/2s window). 500ms between pages → 2 req/s, well
// under the limit, and a 30-page (~3000-fill) walk finishes in
// ~15s — still below the volume route's 60s cache TTL so the
// dashboard's polling cadence is unchanged for hot reads.
const PAGINATOR_INTER_PAGE_DELAY_MS = 500;

function _sleep(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

export async function fetchOkxFillsHistoryPaginatedFull(
  symbol: string | null,
  since_ms: number
): Promise<PaginatedFillsResult> {
  // ``symbol === null`` requests ALL fills on the account (any
  // SWAP instrument). Used by the Total Volume panel which
  // aggregates account-wide rebate-earning volume across all
  // symbols the operator has traded (TON now, SUI before, etc.).
  // ``symbol !== null`` filters to that one instId — the legacy
  // single-symbol path used by per-account state queries.
  //
  // ctVal lookup per fill: each row may belong to a different
  // instId (account-wide path), so we resolve ctVal per row from
  // the cached instruments map. Single map fetch up front; per-row
  // lookup is just a hashtable hit. Falls back to 1.0 for any
  // instId not in the cache.
  let instruments: Map<string, InstrumentMeta>;
  try {
    instruments = await fetchOkxInstrumentsMap();
  } catch {
    // Defensive: empty map → all rows fall back to ctVal=1.0
    // (legacy behaviour). Volume aggregates would be slightly
    // wrong in that case, but it's a rare edge case (the public
    // endpoint is highly available).
    instruments = new Map();
  }
  const ctValFor = (instId: string | undefined): number => {
    if (!instId) return 1.0;
    const m = instruments.get(instId);
    if (!m || !(m.ctVal > 0) || !Number.isFinite(m.ctVal)) return 1.0;
    return m.ctVal;
  };
  const out: FillHistoryRecord[] = [];
  let after: string | undefined = undefined;
  let complete = false;
  for (let i = 0; i < PNL_HISTORY_MAX_ITERATIONS; i++) {
    if (i > 0) {
      await _sleep(PAGINATOR_INTER_PAGE_DELAY_MS);
    }
    const params: Record<string, string> = {
      instType: "SWAP",
      limit: "100",
    };
    if (symbol) {
      params.instId = symbol;
    }
    if (after) params.after = after;
    const resp = await okxSignedRequest<{
      instId?: string;
      ordId?: string;
      tradeId?: string;
      // OKX pagination key for ``/trade/fills-history`` is ``billId``.
      // Each fill row carries one. The ``after`` request param must
      // be the billId from the OLDEST row of the previous response;
      // OKX then returns rows OLDER than that billId. Using
      // ``tradeId`` here (legacy bug) returned the next page empty
      // and pagination terminated after the first batch.
      billId?: string;
      side?: string;
      fillSz?: string;
      fillPx?: string;
      fee?: string;
      feeCcy?: string;
      execType?: string;
      fillPnl?: string;
      ts?: string;
    }>("GET", "/api/v5/trade/fills-history", { params });
    if (resp.code !== "0") {
      // Log + exit early, leaving ``complete=false``. The volume
      // route's stale-while-revalidate cache catches this case and
      // serves the last known complete walk so the UI doesn't
      // flap. Common causes: rate-limit blip (50011), tier-not-
      // permitted for >7d history (50112), session token expiry
      // (60012). Visible in the dev-server console.
      console.warn(
        "fetchOkxFillsHistoryPaginatedFull early exit on OKX error",
        {
          page: i + 1,
          fills_so_far: out.length,
          code: resp.code,
          msg: resp.msg,
        }
      );
      break;
    }
    const batch = resp.data || [];
    if (batch.length === 0) {
      // Venue returned no more rows -- we have everything it knows
      // for this account+symbol. Mark complete so the UI doesn't
      // warn about under-counting on a small / new account.
      complete = true;
      break;
    }
    let oldestTs = Number.MAX_SAFE_INTEGER;
    let oldestBillId: string | undefined;
    for (const r of batch) {
      const price = parseFloat0(r.fillPx);
      const qtyContracts = parseFloat0(r.fillSz);
      const qty = qtyContracts * ctValFor(r.instId); // base-asset units
      const fee = parseFloat0(r.fee);
      const ts = Number(r.ts || 0);
      out.push({
        trade_id: String(r.tradeId || ""),
        order_id: String(r.ordId || ""),
        side: r.side === "buy" ? "BUY" : ("SELL" as "BUY" | "SELL"),
        price,
        qty,
        notional_usd: price * qty,
        // Same sign normalization as fetchOkxFillsHistory: positive
        // = paid, negative = rebate received (matches dashboard
        // convention).
        fee: -fee,
        fee_ccy: String(r.feeCcy || ""),
        is_maker: r.execType === "M",
        realized_pnl: parseFloat0(r.fillPnl),
        time_ms: ts,
        symbol: String(r.instId || ""),
      });
      if (ts < oldestTs) {
        oldestTs = ts;
        oldestBillId = String(r.billId || "");
      }
    }
    // Stop when the oldest fill in the batch is past our cutoff.
    // We got everything WITHIN the requested window; there may be
    // older fills beyond ``since_ms`` but the caller didn't ask
    // for them, so this is also "complete" from the caller's POV.
    if (oldestTs <= since_ms) {
      complete = true;
      break;
    }
    // Continue with the oldest billId as the next page's anchor.
    // ``complete`` stays False if we exit here -- missing cursor
    // means we couldn't continue, so the picture may be partial.
    if (!oldestBillId) break;
    after = oldestBillId;
  }
  // Trim anything strictly older than the cutoff (the last batch may
  // straddle it) and sort newest-first for consistency with the
  // single-page fetcher.
  const fills = out
    .filter((f) => f.time_ms >= since_ms)
    .sort((a, b) => b.time_ms - a.time_ms);
  return { fills, complete };
}

/** Backwards-compatible wrapper that returns just the fills array,
 *  preserved so existing PnL-history callers don't have to change.
 *  New callers (like the volume route) should use the ``Full``
 *  variant above to get the ``complete`` flag. */
export async function fetchOkxFillsHistoryPaginated(
  symbol: string,
  since_ms: number
): Promise<FillHistoryRecord[]> {
  const r = await fetchOkxFillsHistoryPaginatedFull(symbol, since_ms);
  return r.fills;
}
