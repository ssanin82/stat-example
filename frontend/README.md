# dtc-mm-as dashboard (frontend/)

Operator-observation dashboard for the bot fleet. Polls exchange REST
APIs **directly** to show ground-truth account state (equity / cash /
withdrawable / position / open orders) per profile.

Stack: Next.js 15 + React 19 + Tailwind. Single Node process. Dark
theme by default.

## Requirements

* Node ≥ 20 (the host's Node 25 is fine)
* Venue API credentials in `tests/integration/.env` at the repo root
  (gitignored). See `tests/integration/README.md` for format.

## Run

```powershell
cd frontend
npm install   # one-time
npm run dev   # binds 127.0.0.1:8001
```

Open http://localhost:8001

The picker auto-discovers profiles from `config/profiles/*.env`.
Adding a new profile to that directory + restarting `npm run dev`
surfaces it in the dropdown — no code change here. Today: OKX (SUI-
USDT-SWAP) and Binance (DOGEUSDT) are both visible.

## Where credentials are read from

`tests/integration/.env` (one file at the repo root, NOT in `frontend/`).
Format is `KEY=value` per line, no quoting needed (the parser handles
matched quotes; mismatched quotes get a runtime warning). Process env
overrides the file if both are set.

Required keys per venue:

| Venue | Trading-grade keys (the bot uses) | Read-only keys (dashboard prefers when present) |
|---|---|---|
| OKX | `OKX_API_KEY`, `OKX_API_SECRET`, `OKX_API_PASSPHRASE` | `OKX_API_KEY_READONLY`, `OKX_API_SECRET_READONLY`, `OKX_API_PASSPHRASE_READONLY` |
| Binance | `BINANCE_API_KEY`, `BINANCE_API_SECRET` | `BINANCE_API_KEY_READONLY`, `BINANCE_API_SECRET_READONLY` |

**The dashboard auto-prefers `_READONLY` variants when ALL keys of
that variant are populated** for a venue. Atomic per-venue: we never
mix a read-only key with a trading secret. If the readonly set is
incomplete (missing one field), the dashboard falls back to the
trading-grade keys for that venue.

Why both flavours: the trading-grade Binance key is whitelisted to
the EC2's elastic IP (residential laptop IPs rotate; without a
read-only fallback the dashboard breaks every time Etisalat / your
ISP renews your DHCP lease). The read-only key is unrestricted —
"read your balance" isn't a meaningful attack surface even if leaked.

You'll see a one-line redacted log at dashboard startup telling you
which variant got picked per venue, e.g.:

```
[creds] Binance -> using readonly variant (BINANCE_API_KEY=len=64, ab***ef, BINANCE_API_SECRET=len=64, qr***xy)
[creds] OKX     -> using trading variant  (OKX_API_KEY=len=36, 8a***ef, OKX_API_SECRET=len=32, 98***7B, OKX_API_PASSPHRASE=len=16, y$***ip)
```

## Common issue: Binance returns `-2015`

> `Invalid API-key, IP, or permissions for action, request ip: <your-laptop-ip>`

The Binance trading key is whitelisted to the EC2's elastic IP, not the
laptop. **This is intentional** — wide IP allowlists on a trading key
are a bigger risk than dashboard inconvenience. Three fixes:

1. **Skip Binance in the dashboard.** Use Telegram `/status` for the
   Binance bot (this dashboard's primary value-add is the OKX
   sub-account where you have no UI login).
2. **Mint a separate Binance API key, READ-only, with your laptop IP
   in the whitelist.** Replace the dashboard's `BINANCE_API_KEY` /
   `BINANCE_API_SECRET` with this read-only key; the bot's trading
   key stays untouched.
3. **Run the dashboard on the EC2 itself.** Inherits the existing
   whitelist. Cumbersome.

OKX has no whitelist (the partner confirmed it isn't required for the
sub-account), so OKX works from any laptop with the keys.

## Architecture

```
+---------------------+         +--------------------+
| Browser (laptop)    |  HTTP   | Node.js Next.js    |  HTTPS  +-------------+
| http://             | <-----> | http://localhost:  | <-----> | OKX, Binance|
| localhost:8001      |         | 8001               |         | REST APIs   |
+---------------------+         +--------------------+         +-------------+
                                       ^
                                       |
                                  reads creds from
                                  tests/integration/.env
                                  (gitignored, on disk only)
```

Keys never leave the Node process — no HTTP endpoint surfaces them
and the browser never sees them. Browser polls `/api/accounts/[id]/state`
every 5 seconds; that route signs the venue REST call server-side.

## Layout

```
frontend/
├── README.md                          (this file)
├── package.json
├── tsconfig.json + next.config.mjs    (config)
├── postcss.config.js + tailwind.config.ts
├── app/
│   ├── layout.tsx                     (root HTML, dark mode)
│   ├── globals.css                    (Tailwind base + tabular-nums)
│   ├── page.tsx                       (the dashboard UI -- one page)
│   └── api/
│       ├── accounts/route.ts          (GET /api/accounts)
│       └── accounts/[id]/state/route.ts (GET /api/accounts/<id>/state)
└── lib/
    ├── env.ts                         (reads tests/integration/.env)
    ├── accounts.ts                    (scans config/profiles/*.env)
    ├── okx.ts                         (signing + REST helpers)
    └── binance.ts                     (signing + REST helpers)
```

## Safety actions (Cancel All Orders / Close Position)

The dashboard has two **type-to-confirm** action buttons per account:

* **Cancel All Orders** — cancels every open order on the symbol.
  Binance: single `DELETE /fapi/v1/allOpenOrders` call. OKX: batched
  `cancel-batch-orders` (chunks of 20, OKX's batch limit).
* **Close Position** — flattens the open position via reduce-only
  MARKET order. Binance: builds `MARKET reduceOnly IOC` from the
  current `positionAmt`. OKX: dedicated `/api/v5/trade/close-position`
  endpoint.

Both buttons are **disabled when not applicable**: greyed out if
there's nothing to cancel / no position open, or if trading-grade
keys aren't loaded. Hover for the reason.

Confirmation flow: clicking a button opens a modal that:
1. Shows current state (open-order count, position qty, notional)
2. Requires typing the **exact symbol** to confirm (GitHub-style
   "type to delete" pattern; modest friction proportional to
   irreversibility)
3. On confirm: POSTs to the backend, shows the result, auto-refreshes
   the dashboard so you see the post-action reality

**Trading-grade keys are required** for these actions. The dashboard
prefers `_READONLY` for reads but the API routes for cancel/close
ALWAYS use trading-grade keys (no readonly fallback — read-only keys
can't mutate state on the venue side anyway). If only readonly keys
are loaded, the buttons grey out with a clear hint.

Audit trail: each action prints a one-line stderr log to your
`npm run dev` terminal:

```
[action] cancel-all account=prod.okx.sui.usdt.perp venue=okx symbol=SUI-USDT-SWAP outcome=success detail="canceled 3 order(s)" affected=3
[action] close-position account=prod.okx.sui.usdt.perp venue=okx symbol=SUI-USDT-SWAP outcome=success detail="closed 1000 contracts on SUI-USDT-SWAP"
```

Persistent audit log isn't built (single operator, low action
frequency); grep your dev-server output if you need to review.

## Binance write actions and the IP-whitelist gotcha

Your **trading-grade Binance key** is whitelisted to the EC2 EIP
(`54.150.75.21`). Calling cancel/close from your laptop returns
`-2015 Invalid API-key, IP, or permissions for action`. Two paths:

1. **Skip Binance write buttons until the bot is live** — you have
   Telegram /flatten + SSH into EC2 as alternatives. The dashboard
   surfaces the error cleanly so you'll see what happened.
2. **Mint a separate Binance trading key without IP restriction**,
   used only for emergency flatten from your laptop. Drop it into
   `tests/integration/.env` overriding the IP-locked one. (Risk: a
   non-IP-locked key with full trading perms is a bigger target if
   leaked. Acceptable trade-off only if you keep the laptop secure.)

OKX has no such issue — the partner's key has no IP whitelist (their
design), so OKX cancel/close work directly from the laptop.

## What's intentionally NOT here (yet)

This is the v1+v1.5 from `frontend/kickstart.md`. **Not included on
purpose**:

* Cross-account aggregate view (sum equity across all accounts)
* Historical fills / volumes / PnL windows (7d/15d/30d/90d)
* WS push for real-time fills (current cadence: 5s polling)
* Stop / start / deploy / trading-flip actions (those are v3 ops
  control plane territory; cancel/close are the only writes for now)
* Authentication on the local service (binds 127.0.0.1, inherits
  "trust the laptop owner" -- safe because writes are gated by
  type-to-confirm + trading-grade-creds presence)
* Persistent audit log (stderr + npm-run-dev tail is enough at v1)

See `frontend/kickstart.md` for the full v2/v3 trajectory and the
explicit scope fences.

## Adding a new venue

1. Drop a per-venue adapter in `lib/<venue>.ts` exporting
   `fetchXxxAccount` / `fetchXxxPosition` / `fetchXxxOpenOrders`.
2. Wire the venue branch in `app/api/accounts/[id]/state/route.ts`
   (one `else if (account.venue === "<name>")` block).
3. The dashboard now picks up any profile with `EXCHANGE=<name>`
   automatically.
