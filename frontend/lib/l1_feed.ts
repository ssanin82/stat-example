/**
 * Public L1 (top-of-book) WebSocket feeds for the Klines Inspect chart.
 *
 * Two hooks:
 *   * `useOkxL1Mid(symbol)`     — wss://wsaws.okx.com:8443/ws/v5/public, channel "bbo-tbt"
 *   * `useBinanceL1Mid(symbol)` — wss://fstream.binance.com/ws/<sym>@bookTicker
 *
 * Each returns the most recent `(best_bid + best_ask) / 2` as a number, or
 * `null` if no message has arrived yet / the symbol is null. Updates on
 * every L1 change (~tens of ms cadence). No auth, no API keys.
 *
 * Behaviour:
 *   * Auto-reconnect with exponential backoff (1 s → 2 s → 4 s → cap 30 s).
 *     The 30 s cap matches the dashboard's existing reconnect cadence for
 *     other live data sources.
 *   * Pauses when the document tab is hidden (saves bandwidth + matches
 *     the dashboard's existing useDocumentVisible-gated polling pattern).
 *   * Closes cleanly on unmount / symbol change.
 *
 * Why public WS (not the bot's colo feed):
 *   * Operator runs the dashboard on their laptop — no colo network.
 *   * The bot's colo feed and the public feed serve the SAME OKX matching-
 *     engine state, only with different network latencies (milliseconds).
 *     For operator-visible price lines, that drift is invisible.
 *   * No auth → no credential surface.
 *
 * Symbol formats:
 *   * OKX: as-is (`TON-USDT-SWAP`).
 *   * Binance: lowercase, no dashes (`tonusdt`). The hook handles the
 *     transformation — caller passes the same shape as the bot uses
 *     (`TONUSDT` from `live_stats.reference_venue.symbol`).
 */

import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// OKX public WS — bbo-tbt channel
// ---------------------------------------------------------------------------

// PUBLIC endpoint, no authentication, display-only. Per OKX V5 WS
// docs the bbo-tbt channel is on the public WebSocket and "auth-
// orisation is not required"; the VIP-5 tier only affects push
// cadence (10 ms for VIP, ~100 ms otherwise) — channel access is
// universal. If the chart still doesn't paint a tgt-L1 line, check
// the browser console for the diagnostic logs emitted below — they
// trace every state transition the subscription goes through.
//
// 2026-05-20 v1.4.139: tried wsaws.okx.com (AWS-routed) first; some
// networks/regions block direct AWS endpoints. Falls back to the
// global ws.okx.com endpoint, then the EEA endpoint. Each failure
// triggers the next URL on the next reconnect cycle.
const OKX_PUBLIC_WS_URLS = [
  "wss://ws.okx.com:8443/ws/v5/public",      // global (primary)
  "wss://wsaws.okx.com:8443/ws/v5/public",   // AWS-routed (fallback 1)
  "wss://wseea.okx.com:8443/ws/v5/public",   // EEA (fallback 2)
];
const OKX_CHANNEL = "bbo-tbt";

interface OkxBboTbtData {
  asks: [string, string, string, string][];  // [px, sz, _, n_orders]
  bids: [string, string, string, string][];
  ts: string;
}

interface OkxBboTbtMessage {
  arg?: { channel: string; instId: string };
  data?: OkxBboTbtData[];
  event?: string;
  code?: string;
  msg?: string;
}

/**
 * Subscribe to OKX bbo-tbt for `symbol` and return the rolling mid.
 *
 * `symbol` is the OKX instrument id, e.g. "TON-USDT-SWAP". Pass `null`
 * to disable the subscription (returned value will be `null`).
 *
 * Returns the rolling mid `(best_bid + best_ask) / 2`. Updates on every
 * tick (typically 10-50 ms cadence on liquid pairs). `null` until the
 * first message arrives.
 */
export function useOkxL1Mid(
  symbol: string | null,
  enabled: boolean = true,
): { mid: number | null; bid: number | null; ask: number | null } {
  const [snapshot, setSnapshot] = useState<{
    mid: number | null;
    bid: number | null;
    ask: number | null;
  }>({ mid: null, bid: null, ask: null });

  // Visibility gate — pause the WS when the tab is hidden so we don't
  // burn bandwidth on a page no one's looking at.
  const visibleRef = useRef<boolean>(
    typeof document !== "undefined" ? !document.hidden : true,
  );
  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVis = () => {
      visibleRef.current = !document.hidden;
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    if (!enabled || !symbol) {
      setSnapshot({ mid: null, bid: null, ask: null });
      return;
    }
    let ws: WebSocket | null = null;
    let backoffMs = 1000;
    let stopped = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let firstDataTimer: ReturnType<typeof setTimeout> | null = null;
    let dataReceived = false;
    // Rotate through the OKX_PUBLIC_WS_URLS list on each (re)open
    // attempt — handles the case where the primary endpoint is
    // network-blocked but a regional alternative is reachable.
    let urlIdx = 0;

    const open = () => {
      if (stopped) return;
      const url = OKX_PUBLIC_WS_URLS[urlIdx % OKX_PUBLIC_WS_URLS.length];
      urlIdx += 1;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[OKX WS] constructor threw for", url, e);
        scheduleReconnect();
        return;
      }
      // eslint-disable-next-line no-console
      console.info("[OKX WS] connecting to", url);
      ws.onopen = () => {
        if (!ws) return;
        backoffMs = 1000; // reset on successful connect
        const sub = {
          op: "subscribe",
          args: [{ channel: OKX_CHANNEL, instId: symbol }],
        };
        // eslint-disable-next-line no-console
        console.info("[OKX WS] connected; sending subscribe:", sub);
        ws.send(JSON.stringify(sub));
        // If no data lands within 8 seconds after subscribe, log a
        // warning. Possible causes: silent channel-tier restriction,
        // network blocking the WS frames, or wrong symbol id.
        firstDataTimer = setTimeout(() => {
          if (!dataReceived) {
            // eslint-disable-next-line no-console
            console.warn(
              "[OKX WS] no data received within 8s of subscribe — " +
                "check OKX status, network, or try the global endpoint " +
                "(wss://ws.okx.com:8443/ws/v5/public) via dev tools",
            );
          }
        }, 8000);
      };
      ws.onmessage = (ev) => {
        if (!visibleRef.current) return;  // tab hidden — drop updates
        let msg: OkxBboTbtMessage;
        try {
          msg = JSON.parse(ev.data) as OkxBboTbtMessage;
        } catch {
          return;
        }
        if (msg.event === "subscribe") {
          // eslint-disable-next-line no-console
          console.info(
            "[OKX WS] subscribed to",
            msg.arg?.channel,
            msg.arg?.instId,
          );
          return;
        }
        if (msg.event === "error") {
          // OKX sends an error event when the symbol is wrong or the
          // channel is restricted. Logged so the operator can diagnose
          // in the browser console; don't reconnect because reconnecting
          // won't fix a bad subscription.
          // eslint-disable-next-line no-console
          console.warn("[OKX WS] error event:", msg.code, msg.msg);
          return;
        }
        const row = msg.data?.[0];
        if (!row) return;
        const bidPx = parseFloat(row.bids?.[0]?.[0] ?? "NaN");
        const askPx = parseFloat(row.asks?.[0]?.[0] ?? "NaN");
        if (!Number.isFinite(bidPx) || !Number.isFinite(askPx)) return;
        if (!dataReceived) {
          dataReceived = true;
          if (firstDataTimer) clearTimeout(firstDataTimer);
          // eslint-disable-next-line no-console
          console.info(
            "[OKX WS] first L1 received: bid",
            bidPx,
            "ask",
            askPx,
          );
        }
        setSnapshot({ mid: (bidPx + askPx) / 2, bid: bidPx, ask: askPx });
      };
      ws.onerror = (e) => {
        // eslint-disable-next-line no-console
        console.warn("[OKX WS] error event:", e);
      };
      ws.onclose = (e) => {
        // eslint-disable-next-line no-console
        console.info(
          "[OKX WS] closed: code",
          e.code,
          "reason",
          e.reason || "(no reason)",
          "wasClean",
          e.wasClean,
        );
        if (firstDataTimer) clearTimeout(firstDataTimer);
        if (!stopped) scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (stopped) return;
      reconnectTimer = setTimeout(() => {
        open();
        backoffMs = Math.min(30_000, backoffMs * 2);
      }, backoffMs);
    };

    open();

    return () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        // Send unsubscribe before close — polite to OKX. If the socket
        // isn't open yet, OKX will see the silent disconnect; either is
        // fine.
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                op: "unsubscribe",
                args: [{ channel: OKX_CHANNEL, instId: symbol }],
              }),
            );
          }
          ws.close();
        } catch {
          // Already closed — fine.
        }
      }
    };
  }, [symbol, enabled]);

  return snapshot;
}

// ---------------------------------------------------------------------------
// Binance public WS — bookTicker stream
// ---------------------------------------------------------------------------

interface BinanceBookTicker {
  u?: number;
  s?: string;
  b?: string;  // best bid price
  B?: string;  // best bid qty
  a?: string;  // best ask price
  A?: string;  // best ask qty
}

/**
 * Subscribe to Binance futures bookTicker for `symbol` and return the
 * rolling mid. `symbol` is the Binance perpetual symbol (e.g. "TONUSDT").
 * The hook lowercases it for the URL path.
 *
 * URL form: `wss://fstream.binance.com/ws/<symbol>@bookTicker`.
 * Binance pushes a message on every L1 change — typically tens of ms
 * cadence on liquid pairs.
 */
export function useBinanceL1Mid(
  symbol: string | null,
  enabled: boolean = true,
): { mid: number | null; bid: number | null; ask: number | null } {
  const [snapshot, setSnapshot] = useState<{
    mid: number | null;
    bid: number | null;
    ask: number | null;
  }>({ mid: null, bid: null, ask: null });

  const visibleRef = useRef<boolean>(
    typeof document !== "undefined" ? !document.hidden : true,
  );
  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVis = () => {
      visibleRef.current = !document.hidden;
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    if (!enabled || !symbol) {
      setSnapshot({ mid: null, bid: null, ask: null });
      return;
    }
    const url = `wss://fstream.binance.com/ws/${symbol.toLowerCase()}@bookTicker`;
    let ws: WebSocket | null = null;
    let backoffMs = 1000;
    let stopped = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const open = () => {
      if (stopped) return;
      try {
        ws = new WebSocket(url);
      } catch {
        scheduleReconnect();
        return;
      }
      ws.onopen = () => {
        backoffMs = 1000;
        // Binance's per-stream URL form auto-subscribes — no
        // subscribe message needed.
      };
      ws.onmessage = (ev) => {
        if (!visibleRef.current) return;
        let msg: BinanceBookTicker;
        try {
          msg = JSON.parse(ev.data) as BinanceBookTicker;
        } catch {
          return;
        }
        const bidPx = parseFloat(msg.b ?? "NaN");
        const askPx = parseFloat(msg.a ?? "NaN");
        if (!Number.isFinite(bidPx) || !Number.isFinite(askPx)) return;
        setSnapshot({ mid: (bidPx + askPx) / 2, bid: bidPx, ask: askPx });
      };
      ws.onerror = () => {
        // Close handler reconnects.
      };
      ws.onclose = () => {
        if (!stopped) scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (stopped) return;
      reconnectTimer = setTimeout(() => {
        open();
        backoffMs = Math.min(30_000, backoffMs * 2);
      }, backoffMs);
    };

    open();

    return () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        try {
          ws.close();
        } catch {
          // Already closed.
        }
      }
    };
  }, [symbol, enabled]);

  return snapshot;
}
