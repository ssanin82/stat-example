/**
 * Binance Futures USDM signing + signed REST helpers.
 *
 * Mirrors `app/exchange/binance_client.py` -- HMAC-SHA256 over the
 * urlencoded query string, signature appended as `signature=...`.
 */

import { createHmac } from "node:crypto";
import {
  resolveVenueCreds,
  resolveTradingCreds,
  hasTradingCreds,
} from "./creds";

const BINANCE_REST = "https://fapi.binance.com";
const RECV_WINDOW_MS = 5000;
const BINANCE_TRADING_NAMES = ["BINANCE_API_KEY", "BINANCE_API_SECRET"];

interface BinanceCreds {
  apiKey: string;
  apiSecret: string;
}

function readCreds(): BinanceCreds {
  // Prefer BINANCE_API_KEY_READONLY / BINANCE_API_SECRET_READONLY when
  // BOTH are present (the dashboard's expected setup; the read-only
  // key is unrestricted IP-wise so it survives ISP IP rotations).
  // Fall back to the trading-grade BINANCE_API_KEY / BINANCE_API_SECRET
  // when the readonly pair is incomplete.
  const { values } = resolveVenueCreds("Binance", BINANCE_TRADING_NAMES);
  return { apiKey: values[0], apiSecret: values[1] };
}

/**
 * Trading-grade creds (no readonly fallback). Required for any
 * write action -- cancel, close, modify. A read-only key would hit
 * a venue auth error trying to mutate.
 */
function readTradingCreds(): BinanceCreds {
  const { values } = resolveTradingCreds("Binance", BINANCE_TRADING_NAMES);
  return { apiKey: values[0], apiSecret: values[1] };
}

export function hasBinanceTradingCreds(): boolean {
  return hasTradingCreds(BINANCE_TRADING_NAMES);
}

function binanceSign(apiSecret: string, queryString: string): string {
  return createHmac("sha256", apiSecret).update(queryString).digest("hex");
}

async function binanceSignedRequest<T>(
  method: "GET" | "POST" | "DELETE" | "PUT",
  path: string,
  params: Record<string, string> = {},
  opts: { write?: boolean } = {}
): Promise<T> {
  // Use trading creds for explicit writes; reads default to the
  // venue resolver (which prefers READONLY when present).
  const creds = opts.write ? readTradingCreds() : readCreds();
  const allParams = {
    ...params,
    timestamp: Date.now().toString(),
    recvWindow: String(RECV_WINDOW_MS),
  };
  const queryString = new URLSearchParams(allParams).toString();
  const sign = binanceSign(creds.apiSecret, queryString);
  const url = `${BINANCE_REST}${path}?${queryString}&signature=${sign}`;
  const resp = await fetch(url, {
    method,
    headers: { "X-MBX-APIKEY": creds.apiKey },
    cache: "no-store",
  });
  return (await resp.json()) as T;
}

// Compatibility shim for the existing read paths (unchanged callers).
async function binanceSignedGet<T>(
  path: string,
  params: Record<string, string> = {}
): Promise<T> {
  return binanceSignedRequest<T>("GET", path, params);
}

// ---------------------------------------------------------------------------
// Public account-state surface
// ---------------------------------------------------------------------------

export interface BinanceAccountSnapshot {
  equity_usd: number | null;
  withdrawable_usd: number | null;
  cash_usd: number | null;
  unrealized_pnl_usd: number;
  /** Raw per-asset details for the dashboard's drill-down view. */
  assets: Array<{
    asset: string;
    walletBalance: number;
    availableBalance: number;
    unrealizedProfit: number;
  }>;
}

export interface BinancePositionSnapshot {
  symbol: string;
  position_qty: number;
  avg_entry_price: number | null;
  mark_price: number | null;
  unrealized_pnl_usd: number;
  notional_usd: number;
}

export interface BinanceOpenOrder {
  order_id: number;
  client_order_id: string;
  side: "BUY" | "SELL";
  price: number;
  orig_qty: number;
  status: string;
  time_ms: number;
}

function parseFloat0(v: unknown): number {
  if (v === null || v === undefined || v === "") return 0;
  const f = Number(v);
  return Number.isFinite(f) ? f : 0;
}

export async function fetchBinanceAccount(): Promise<BinanceAccountSnapshot> {
  // /fapi/v2/account returns a single object (not wrapped in a list
  // like OKX). Fields:
  //   totalWalletBalance     -- collateral deposited
  //   availableBalance       -- free margin
  //   totalUnrealizedProfit  -- sum of open-position MTM
  //   assets[]               -- per-asset balance rows
  const body = (await binanceSignedGet<{
    totalWalletBalance?: string;
    availableBalance?: string;
    totalUnrealizedProfit?: string;
    assets?: Array<{
      asset?: string;
      walletBalance?: string;
      availableBalance?: string;
      unrealizedProfit?: string;
    }>;
    code?: number;
    msg?: string;
  }>("/fapi/v2/account")) || {};
  if (typeof body.code === "number" && body.code < 0) {
    throw new Error(`Binance account: code=${body.code} msg=${body.msg}`);
  }
  const wallet = parseFloat0(body.totalWalletBalance);
  const avail = parseFloat0(body.availableBalance);
  const upl = parseFloat0(body.totalUnrealizedProfit);
  const assets: BinanceAccountSnapshot["assets"] = [];
  for (const a of body.assets || []) {
    if (!a.asset) continue;
    const wb = parseFloat0(a.walletBalance);
    const ab = parseFloat0(a.availableBalance);
    const up = parseFloat0(a.unrealizedProfit);
    // Filter out zero rows so the UI isn't a wall of dust.
    if (wb === 0 && ab === 0 && up === 0) continue;
    assets.push({
      asset: a.asset,
      walletBalance: wb,
      availableBalance: ab,
      unrealizedProfit: up,
    });
  }
  return {
    equity_usd: wallet > 0 ? wallet + upl : null,
    cash_usd: wallet > 0 ? wallet : null,
    withdrawable_usd: avail > 0 ? avail : null,
    unrealized_pnl_usd: upl,
    assets,
  };
}

export async function fetchBinancePosition(
  symbol: string
): Promise<BinancePositionSnapshot> {
  const arr = (await binanceSignedGet<
    Array<{
      symbol?: string;
      positionAmt?: string;
      entryPrice?: string;
      markPrice?: string;
      unRealizedProfit?: string;
      // Error envelope variant -- /fapi/v2/positionRisk returns a
      // top-level object {code, msg} on auth failure rather than
      // an array. We accept both shapes.
    }>
  >("/fapi/v2/positionRisk", { symbol })) as
    | Array<{
        symbol?: string;
        positionAmt?: string;
        entryPrice?: string;
        markPrice?: string;
        unRealizedProfit?: string;
      }>
    | { code?: number; msg?: string };
  if (!Array.isArray(arr)) {
    throw new Error(
      `Binance positionRisk: code=${arr.code} msg=${arr.msg}`
    );
  }
  let qty = 0;
  let entry = 0;
  let mark = 0;
  let upl = 0;
  for (const r of arr) {
    if (r.symbol !== symbol) continue;
    qty = parseFloat0(r.positionAmt);
    entry = parseFloat0(r.entryPrice);
    mark = parseFloat0(r.markPrice);
    upl = parseFloat0(r.unRealizedProfit);
    break;
  }
  return {
    symbol,
    position_qty: qty,
    avg_entry_price: entry > 0 ? entry : null,
    mark_price: mark > 0 ? mark : null,
    unrealized_pnl_usd: upl,
    notional_usd: Math.abs(qty) * (mark > 0 ? mark : entry),
  };
}

export async function fetchBinanceOpenOrders(
  symbol: string
): Promise<BinanceOpenOrder[]> {
  const arr = (await binanceSignedGet<
    Array<{
      orderId?: number;
      clientOrderId?: string;
      side?: string;
      price?: string;
      origQty?: string;
      status?: string;
      time?: number;
    }>
  >("/fapi/v1/openOrders", { symbol })) as
    | Array<{
        orderId?: number;
        clientOrderId?: string;
        side?: string;
        price?: string;
        origQty?: string;
        status?: string;
        time?: number;
      }>
    | { code?: number; msg?: string };
  if (!Array.isArray(arr)) return [];
  return arr.map((r) => ({
    order_id: Number(r.orderId || 0),
    client_order_id: String(r.clientOrderId || ""),
    side: r.side === "BUY" ? "BUY" : "SELL",
    price: parseFloat0(r.price),
    orig_qty: parseFloat0(r.origQty),
    status: String(r.status || ""),
    time_ms: Number(r.time || 0),
  }));
}

// ---------------------------------------------------------------------------
// Write actions (cancel + flatten). Always use trading-grade creds.
// Returns a normalized result the dashboard can render uniformly.
// ---------------------------------------------------------------------------

export interface ActionResult {
  outcome: "success" | "noop" | "error";
  detail: string;
  /** Number of orders affected, when applicable. */
  affected?: number;
}

/**
 * Cancel ALL open orders on `symbol`. Single-call endpoint
 * (`DELETE /fapi/v1/allOpenOrders`); Binance returns 200 with
 * `code: 200, msg: "The operation of cancel all open order is done."`
 * on success, even if there were no orders to cancel.
 */
export async function cancelAllBinanceOrders(
  symbol: string
): Promise<ActionResult> {
  // Pre-check: how many orders are out there now? Lets the UI show
  // "canceled N orders" rather than just "success".
  const before = await fetchBinanceOpenOrders(symbol);
  if (before.length === 0) {
    return { outcome: "noop", detail: "no open orders to cancel", affected: 0 };
  }
  const resp = await binanceSignedRequest<{
    code?: number;
    msg?: string;
  }>("DELETE", "/fapi/v1/allOpenOrders", { symbol }, { write: true });
  // Binance returns code: 200 on success (positive). Negative code
  // means error. Missing code with HTTP 200 also means success.
  if (resp && typeof resp.code === "number" && resp.code < 0) {
    return {
      outcome: "error",
      detail: `binance code=${resp.code} msg=${resp.msg || ""}`,
    };
  }
  return {
    outcome: "success",
    detail: `canceled ${before.length} order(s)`,
    affected: before.length,
  };
}

/**
 * Close (flatten) the open position on `symbol` via a reduce-only
 * MARKET IOC order. Mirrors `app/exchange/binance_client.py::market_close`.
 * Returns noop if the position is already flat.
 */
export async function closeBinancePosition(
  symbol: string
): Promise<ActionResult> {
  const pos = await fetchBinancePosition(symbol);
  if (Math.abs(pos.position_qty) <= 0) {
    return { outcome: "noop", detail: "position already flat" };
  }
  const isBuy = pos.position_qty < 0; // closing short -> buy back
  const qty = Math.abs(pos.position_qty);
  const params: Record<string, string> = {
    symbol,
    side: isBuy ? "BUY" : "SELL",
    type: "MARKET",
    quantity: String(qty),
    reduceOnly: "true",
    timeInForce: "IOC",
    newOrderRespType: "RESULT",
  };
  const resp = await binanceSignedRequest<{
    code?: number;
    msg?: string;
    orderId?: number;
    executedQty?: string;
    avgPrice?: string;
  }>("POST", "/fapi/v1/order", params, { write: true });
  if (resp && typeof resp.code === "number" && resp.code < 0) {
    return {
      outcome: "error",
      detail: `binance code=${resp.code} msg=${resp.msg || ""}`,
    };
  }
  const executed = parseFloat0(resp.executedQty);
  const avgPx = parseFloat0(resp.avgPrice);
  return {
    outcome: "success",
    detail:
      `closed via MARKET ${isBuy ? "BUY" : "SELL"} ${qty}; ` +
      `executed=${executed} avgPx=${avgPx > 0 ? avgPx.toFixed(6) : "n/a"}`,
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
  /** Notional in quote-currency USD. Computed from price * qty. */
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
  /** Venue symbol of the fill — parity with the OKX side. Binance's
   *  ``/fapi/v1/userTrades`` requires a single symbol per request, so
   *  this is always the symbol the caller passed in. Carried through
   *  for the per-token volume aggregator. */
  symbol: string;
}

/**
 * Last N orders for `symbol`. Includes filled, canceled, expired,
 * etc. Newest first. Binance default cap is 500 -- 50 is plenty
 * for an at-a-glance view.
 */
export async function fetchBinanceOrdersHistory(
  symbol: string,
  limit: number = 50
): Promise<OrderHistoryRecord[]> {
  const arr = (await binanceSignedGet<
    Array<{
      orderId?: number;
      clientOrderId?: string;
      side?: string;
      type?: string;
      price?: string;
      origQty?: string;
      executedQty?: string;
      status?: string;
      time?: number;
      updateTime?: number;
    }>
  >("/fapi/v1/allOrders", { symbol, limit: String(limit) })) as
    | Array<{
        orderId?: number;
        clientOrderId?: string;
        side?: string;
        type?: string;
        price?: string;
        origQty?: string;
        executedQty?: string;
        status?: string;
        time?: number;
        updateTime?: number;
      }>
    | { code?: number; msg?: string };
  if (!Array.isArray(arr)) return [];
  // Binance returns oldest-first; reverse for newest-first display.
  return arr
    .map((r) => {
      const price = parseFloat0(r.price);
      const orig = parseFloat0(r.origQty);
      return {
        order_id: String(r.orderId || ""),
        client_order_id: String(r.clientOrderId || ""),
        side: r.side === "BUY" ? "BUY" : ("SELL" as "BUY" | "SELL"),
        type: String(r.type || ""),
        price,
        orig_qty: orig,
        filled_qty: parseFloat0(r.executedQty),
        status: String(r.status || ""),
        time_ms: Number(r.updateTime || r.time || 0),
        notional_usd: price * orig,
      };
    })
    .sort((a, b) => b.time_ms - a.time_ms);
}

/**
 * Last N fills (executions) for `symbol`. Includes per-fill `commission`
 * (fee), `commissionAsset` (fee currency), and `maker` (bool).
 */
export async function fetchBinanceFillsHistory(
  symbol: string,
  limit: number = 50
): Promise<FillHistoryRecord[]> {
  const arr = (await binanceSignedGet<
    Array<{
      id?: number;
      orderId?: number;
      side?: string;
      price?: string;
      qty?: string;
      quoteQty?: string;
      commission?: string;
      commissionAsset?: string;
      time?: number;
      maker?: boolean;
      realizedPnl?: string;
    }>
  >("/fapi/v1/userTrades", { symbol, limit: String(limit) })) as
    | Array<{
        id?: number;
        orderId?: number;
        side?: string;
        price?: string;
        qty?: string;
        quoteQty?: string;
        commission?: string;
        commissionAsset?: string;
        time?: number;
        maker?: boolean;
        realizedPnl?: string;
      }>
    | { code?: number; msg?: string };
  if (!Array.isArray(arr)) return [];
  return arr
    .map((r) => {
      const price = parseFloat0(r.price);
      const qty = parseFloat0(r.qty);
      // Binance gives us quoteQty (already-computed USD notional) --
      // prefer that to qty * price for currencies where they might
      // differ (multi-collateral accounts).
      const notional = parseFloat0(r.quoteQty) || price * qty;
      return {
        trade_id: String(r.id || ""),
        order_id: String(r.orderId || ""),
        side: r.side === "BUY" ? "BUY" : ("SELL" as "BUY" | "SELL"),
        price,
        qty,
        notional_usd: notional,
        fee: parseFloat0(r.commission),
        fee_ccy: String(r.commissionAsset || ""),
        is_maker: Boolean(r.maker),
        realized_pnl: parseFloat0(r.realizedPnl),
        time_ms: Number(r.time || 0),
        symbol,
      };
    })
    .sort((a, b) => b.time_ms - a.time_ms);
}

/**
 * Public klines for Binance Futures USDM. Unauthenticated --
 * ``/fapi/v1/klines`` is open. Returns rows as [openTime, o, h, l,
 * c, vol, closeTime, ...]; we drop the unused tail.
 *
 * Interval shorthand: "1m", "3m", "5m", "15m", "30m", "1h", "2h",
 * "4h", "1d". (Note Binance lowercase vs OKX uppercase 'H'/'D';
 * normalize at the call site.)
 */
import type { Candle } from "./okx";
import { twapFundingRateEstimatePerInterval } from "./okx";
export type { Candle };

export async function fetchBinanceKlines(
  symbol: string,
  interval: string = "1m",
  limit: number = 100
): Promise<Candle[]> {
  const url = `${BINANCE_REST}/fapi/v1/klines?symbol=${encodeURIComponent(
    symbol
  )}&interval=${encodeURIComponent(interval)}&limit=${limit}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`binance_klines: HTTP ${r.status}`);
  const rows = (await r.json()) as Array<
    [number, string, string, string, string, string, ...unknown[]]
  >;
  return rows.map((row) => ({
    time_ms: Number(row[0] || 0),
    open: parseFloat0(row[1]),
    high: parseFloat0(row[2]),
    low: parseFloat0(row[3]),
    close: parseFloat0(row[4]),
    volume: parseFloat0(row[5]),
  }));
}

/** Funding-rate tick — mirrors ``OkxFundingRateTick`` so the
 *  dashboard route can return both venues' data in a uniform shape.
 *  Funding intervals on Binance USD-M perp are 8 h (same schedule as
 *  OKX). ``rate`` is the decimal fraction (e.g. ``0.00002819`` =
 *  0.002819 % per 8 h). */
export interface BinanceFundingRateTick {
  time_ms: number;
  rate: number;
}

/**
 * Public funding-rate history fetch.
 * Endpoint: ``/fapi/v1/fundingRate``.
 *
 * Public / unauthenticated. ``limit`` caps at 1000; at 8 h intervals
 * that's ~333 days of history per request — far more than any
 * dashboard session window needs. Single request, no pagination.
 *
 * Used by the dashboard's funding-rate sub-band on the Session-PnL
 * chart (Q2 from the 2026-05-14 funding-rate / mark-price thread).
 */
export async function fetchBinanceFundingRateHistory(
  symbol: string,
  sinceMs: number,
): Promise<BinanceFundingRateTick[]> {
  const url = new URL(`${BINANCE_REST}/fapi/v1/fundingRate`);
  url.searchParams.set("symbol", symbol);
  // Always include the boundary explicitly; otherwise Binance
  // returns the most recent 100 records regardless of how far back
  // sinceMs was. Limit 1000 is plenty for any operator session.
  url.searchParams.set("startTime", String(Math.max(0, Math.floor(sinceMs))));
  url.searchParams.set("limit", "1000");
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`binance_funding_rate: HTTP ${r.status}`);
  }
  const rows = (await r.json()) as Array<{
    symbol?: string;
    fundingTime?: number;
    fundingRate?: string;
    markPrice?: string;
  }>;
  const out: BinanceFundingRateTick[] = [];
  for (const row of rows) {
    const tsMs = Number(row.fundingTime || 0);
    const rate = parseFloat0(row.fundingRate || "");
    if (!Number.isFinite(tsMs) || tsMs <= 0) continue;
    if (!Number.isFinite(rate)) continue;
    out.push({ time_ms: tsMs, rate });
  }
  // Binance returns ascending; defensively sort in case the order
  // contract ever changes.
  out.sort((a, b) => a.time_ms - b.time_ms);
  return out;
}

/** Mirrors ``OkxFundingRateEstimate``'s shape exported from
 *  ``lib/okx.ts``. Imported there as ``FundingRateEstimate``; we
 *  re-declare locally to keep ``lib/binance.ts`` import-free from
 *  ``lib/okx.ts``. The route stitches the two together. */
export interface BinanceFundingRateEstimate {
  predicted_rate: number;
  next_funding_ms: number;
  funding_interval_ms: number;
  method: string;
  fetched_at_ms: number;
}

/**
 * Predicted next-period funding rate for a Binance USD-M perp.
 *
 * Binance does NOT expose the predicted rate as a single REST field
 * (only the historical last-paid rate). We reconstruct it the same
 * way Binance's own web UI does:
 *
 *   1. ``/fapi/v1/premiumIndex`` → snapshot of the latest premium,
 *      ``nextFundingTime``, ``interestRate`` (per interval),
 *      ``lastFundingRate``.
 *
 *   2. ``/fapi/v1/premiumIndexKlines`` at 1 m interval covering
 *      ``[nextFundingTime - intervalMs, now]`` — the elapsed portion
 *      of the current funding interval. Each kline's ``close`` is
 *      the premium-index value at that minute.
 *
 *   3. TWAP = arithmetic mean of the kline closes (Binance's
 *      published formula).
 *
 *   4. ``predicted = TWAP + clamp(interestRate - TWAP, ±0.05%)``
 *      — the funding-rate formula Binance publishes. The clamp pins
 *      the rate to the interest rate when premium is small; lets
 *      premium dominate when it's larger than the clamp threshold.
 *
 * Funding interval is derived from the symbol's recent
 * ``/fapi/v1/fundingRate`` history (Binance has many altcoin perps
 * on 4 h instead of 8 h; the value isn't exposed directly in
 * ``premiumIndex``).
 *
 * Two REST calls per refresh — both public and unauthenticated.
 * Weight budget: premiumIndex=1, premiumIndexKlines=1. At a 1 min
 * poll cadence that's 120/h, well inside Binance's 2400/m IP weight
 * budget.
 */
export async function fetchBinanceFundingRateEstimate(
  symbol: string,
): Promise<BinanceFundingRateEstimate> {
  // 1. Snapshot — nextFundingTime, interestRate, current premium.
  const piUrl = `${BINANCE_REST}/fapi/v1/premiumIndex?symbol=${encodeURIComponent(symbol)}`;
  const piResp = await fetch(piUrl, { cache: "no-store" });
  if (!piResp.ok) {
    throw new Error(`binance_premium_index: HTTP ${piResp.status}`);
  }
  const pi = (await piResp.json()) as {
    markPrice?: string;
    indexPrice?: string;
    nextFundingTime?: number;
    interestRate?: string;
    lastFundingRate?: string;
  };
  const nextFundingMs = Number(pi.nextFundingTime || 0);
  if (!Number.isFinite(nextFundingMs) || nextFundingMs <= 0) {
    throw new Error("binance_premium_index: missing nextFundingTime");
  }
  const interestRate = parseFloat0(pi.interestRate || "");

  // 2. Derive funding interval from recent settled history. Binance
  //    has BTC / ETH / etc. on 8 h and many altcoin perps on 4 h;
  //    ``premiumIndex`` doesn't carry it directly.
  const history = await fetchBinanceFundingRateHistory(
    symbol,
    Date.now() - 48 * 3_600_000,
  );
  let intervalMs = 8 * 3_600_000;
  if (history.length >= 2) {
    const gaps: number[] = [];
    for (let i = history.length - 1; i > 0 && gaps.length < 3; i--) {
      const g = history[i].time_ms - history[i - 1].time_ms;
      if (g > 0) gaps.push(g);
    }
    if (gaps.length > 0) {
      gaps.sort((a, b) => a - b);
      intervalMs = gaps[Math.floor(gaps.length / 2)];
    }
  }
  const intervalStartMs = nextFundingMs - intervalMs;

  // 3. Premium-index klines at 1 m, covering the elapsed portion of
  //    the interval. Binance caps to ~1000 entries per response;
  //    8 h × 60 min = 480 minutes, well under the cap.
  const klUrl = new URL(`${BINANCE_REST}/fapi/v1/premiumIndexKlines`);
  klUrl.searchParams.set("symbol", symbol);
  klUrl.searchParams.set("interval", "1m");
  klUrl.searchParams.set("startTime", String(Math.max(0, intervalStartMs)));
  klUrl.searchParams.set("limit", "1000");
  const klResp = await fetch(klUrl.toString(), { cache: "no-store" });
  if (!klResp.ok) {
    throw new Error(`binance_premium_klines: HTTP ${klResp.status}`);
  }
  const klRows = (await klResp.json()) as Array<Array<string | number>>;
  const closes: number[] = [];
  for (const row of klRows) {
    const close = parseFloat0(String(row[4]));
    if (Number.isFinite(close)) closes.push(close);
  }

  let twapPremium: number;
  let method = "binance_premium_index_twap";
  if (closes.length > 0) {
    twapPremium = closes.reduce((s, x) => s + x, 0) / closes.length;
  } else {
    // Fallback: instantaneous (mark - index) / index from the
    // snapshot. Less accurate but always available — used when
    // klines haven't been populated for a brand-new symbol or
    // during transient endpoint outages.
    const mark = parseFloat0(pi.markPrice || "");
    const index = parseFloat0(pi.indexPrice || "");
    twapPremium = index > 0 ? (mark - index) / index : 0;
    method = "binance_instantaneous_premium";
  }

  // 4. Apply Binance's published formula.
  //    F = P + clamp(I - P, ±0.05%)
  //    Clamp limit is 0.05% per interval, unscaled regardless of
  //    funding cadence (4 h / 8 h pairs both use 0.05%).
  const CLAMP_LIMIT = 0.0005;
  const clamped = Math.max(
    -CLAMP_LIMIT,
    Math.min(CLAMP_LIMIT, interestRate - twapPremium),
  );
  const predictedRate = twapPremium + clamped;

  return {
    predicted_rate: predictedRate,
    next_funding_ms: nextFundingMs,
    funding_interval_ms: intervalMs,
    method,
    fetched_at_ms: Date.now(),
  };
}

/** Historical predicted-funding-rate sample. Mirrors the OKX shape
 *  in ``lib/okx.ts`` (re-declared locally to keep this file import-
 *  free from okx.ts). Used by the dashboard's funding-rate-estimate
 *  history route to seed the predicted-rate dashed trace back to
 *  session start. */
export interface BinanceFundingRateEstimateSample {
  time_ms: number;
  rate: number;
}

/**
 * Historical predicted-funding-rate reconstruction for a Binance
 * USD-M perp. Reads ``/fapi/v1/premiumIndexKlines`` at 1 m resolution
 * over the window ``[sinceMs, now]`` and returns one sample per
 * minute. The kline ``close`` IS the premium index at that minute —
 * the same value Binance's UI displays as the instantaneous
 * predicted-rate visual. Different from the TWAP+clamp formula
 * applied for the live tail (``fetchBinanceFundingRateEstimate``)
 * which matches the venue's authoritative value to ~1 bp; the two
 * converge at each funding settlement.
 *
 * Single REST call (premiumIndexKlines paginates only on huge
 * windows; 1440 minutes / 24 h fits in two ``limit=1500`` calls).
 */
export async function fetchBinanceFundingRateEstimateHistory(
  symbol: string,
  sinceMs: number,
): Promise<BinanceFundingRateEstimateSample[]> {
  const url = new URL(`${BINANCE_REST}/fapi/v1/premiumIndexKlines`);
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("interval", "1m");
  url.searchParams.set("startTime", String(Math.max(0, Math.floor(sinceMs))));
  url.searchParams.set("limit", "1500");
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`binance_premium_klines_history: HTTP ${r.status}`);
  }
  const rows = (await r.json()) as Array<Array<string | number>>;
  const perMinute: Array<{ time_ms: number; premium: number }> = [];
  for (const row of rows) {
    const ts = Number(row[0] || 0);
    const close = parseFloat0(String(row[4]));
    if (!Number.isFinite(ts) || ts <= 0) continue;
    if (!Number.isFinite(close)) continue;
    perMinute.push({ time_ms: ts, premium: close });
  }
  // 1.3.62: TWAP-and-clamp the per-minute premium so the historical
  // trace matches the venue UI's predicted-rate value (and the
  // bot's own live-tail TWAP-based prediction). The shared helper
  // lives in ``lib/okx.ts`` to avoid duplicating the math.
  return twapFundingRateEstimatePerInterval(perMinute);
}

/** 24h ticker summary — mirrors ``OkxTicker24h`` so the route can
 *  treat both venues uniformly. ``vol_ccy_quote_24h`` is the USDT
 *  notional figure operators care about for "did this market trade
 *  enough today to be worth quoting?" decisions. */
export interface BinanceTicker24h {
  last: number;
  high24h: number;
  low24h: number;
  vol24h: number;
  vol_ccy_quote_24h: number;
  change_24h_bps: number;
}

export async function fetchBinanceTicker24h(
  symbol: string,
): Promise<BinanceTicker24h> {
  const url = `${BINANCE_REST}/fapi/v1/ticker/24hr?symbol=${encodeURIComponent(symbol)}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`binance_ticker24h: HTTP ${r.status}`);
  const j = (await r.json()) as Record<string, string>;
  const last = parseFloat0(j.lastPrice || "0");
  // ``priceChangePercent`` is already in % — convert to bps.
  const change_bps = parseFloat0(j.priceChangePercent || "0") * 100;
  return {
    last,
    high24h: parseFloat0(j.highPrice || "0"),
    low24h: parseFloat0(j.lowPrice || "0"),
    vol24h: parseFloat0(j.volume || "0"),
    vol_ccy_quote_24h: parseFloat0(j.quoteVolume || "0"),
    change_24h_bps: change_bps,
  };
}

export interface BinanceOpenInterest {
  oi_contracts: number;
  oi_base: number;
  oi_usd: number;
}

export async function fetchBinanceOpenInterest(
  symbol: string,
): Promise<BinanceOpenInterest> {
  const url = `${BINANCE_REST}/fapi/v1/openInterest?symbol=${encodeURIComponent(symbol)}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`binance_open_interest: HTTP ${r.status}`);
  const j = (await r.json()) as Record<string, string>;
  const oi_contracts = parseFloat0(j.openInterest || "0");
  // Binance USD-margined perps: oi is already in base units; we'd
  // need a price multiplier to convert. Caller can compute usd via
  // ``oi_base * last_price`` if needed. Leave usd=0 here.
  return { oi_contracts, oi_base: oi_contracts, oi_usd: 0 };
}

/**
 * Paginated long-window fills via Binance's ``/fapi/v1/userTrades``
 * with ``startTime``-based windowing. Used by the dashboard's PnL-
 * history chart for 30d/90d backfill.
 *
 * Pagination strategy: Binance returns up to 1000 fills per page,
 * sorted ASCENDING by time when ``startTime`` is provided. We loop
 * walking the time cursor forward:
 *   1. fetch from startTime=since_ms, limit=1000
 *   2. note newest ts in batch
 *   3. fetch from startTime=newest_ts+1
 *   4. stop when batch is empty OR < 1000 (last page)
 *
 * Hard iteration cap matches OKX (100 pages).
 */
export interface PaginatedFillsResult {
  fills: FillHistoryRecord[];
  /** Mirrors OKX semantics: True when the venue ran out of newer
   *  fills OR we got everything within the requested window;
   *  False only when pagination stopped early due to safety caps. */
  complete: boolean;
}

export async function fetchBinanceFillsHistoryPaginatedFull(
  symbol: string,
  since_ms: number
): Promise<PaginatedFillsResult> {
  const out: FillHistoryRecord[] = [];
  let cursor = since_ms;
  let complete = false;
  for (let i = 0; i < 100; i++) {
    const arr = (await binanceSignedGet<
      Array<{
        id?: number;
        orderId?: number;
        side?: string;
        price?: string;
        qty?: string;
        quoteQty?: string;
        commission?: string;
        commissionAsset?: string;
        time?: number;
        maker?: boolean;
        realizedPnl?: string;
      }>
    >("/fapi/v1/userTrades", {
      symbol,
      limit: "1000",
      startTime: String(cursor),
    })) as
      | Array<{
          id?: number;
          orderId?: number;
          side?: string;
          price?: string;
          qty?: string;
          quoteQty?: string;
          commission?: string;
          commissionAsset?: string;
          time?: number;
          maker?: boolean;
          realizedPnl?: string;
        }>
      | { code?: number; msg?: string };
    if (!Array.isArray(arr)) break;
    if (arr.length === 0) {
      // Venue returned no more rows -- we have everything.
      complete = true;
      break;
    }
    let newestTs = cursor;
    for (const r of arr) {
      const price = parseFloat0(r.price);
      const qty = parseFloat0(r.qty);
      const notional = parseFloat0(r.quoteQty) || price * qty;
      const ts = Number(r.time || 0);
      out.push({
        trade_id: String(r.id || ""),
        order_id: String(r.orderId || ""),
        side: r.side === "BUY" ? "BUY" : ("SELL" as "BUY" | "SELL"),
        price,
        qty,
        notional_usd: notional,
        fee: parseFloat0(r.commission),
        fee_ccy: String(r.commissionAsset || ""),
        is_maker: Boolean(r.maker),
        realized_pnl: parseFloat0(r.realizedPnl),
        time_ms: ts,
        symbol,
      });
      if (ts > newestTs) newestTs = ts;
    }
    if (arr.length < 1000) {
      // Got fewer rows than the per-request limit -- venue has
      // nothing more newer than ``newestTs``. Complete.
      complete = true;
      break;
    }
    // Advance cursor past the newest fill we've seen. +1ms avoids
    // re-pulling the same boundary fill on the next call.
    cursor = newestTs + 1;
  }
  return {
    fills: out.sort((a, b) => b.time_ms - a.time_ms),
    complete,
  };
}

/** Backwards-compatible wrapper. */
export async function fetchBinanceFillsHistoryPaginated(
  symbol: string,
  since_ms: number
): Promise<FillHistoryRecord[]> {
  const r = await fetchBinanceFillsHistoryPaginatedFull(symbol, since_ms);
  return r.fills;
}
