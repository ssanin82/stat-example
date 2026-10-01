# Integration tests

These tests hit **real exchange APIs** with **real credentials**. They
are **NOT** included in the default `pytest` run -- the project's main
test suite is unit-level and must stay deterministic + offline.

## When to run

* Right after standing up a new venue adapter (smoke).
* Before flipping `TRADING_ENABLED=true` on a venue you haven't traded
  on for a while.
* When debugging a divergence between the bot's view and the venue's
  view of an order or position.

## When NOT to run

* In CI (no creds, would always skip anyway).
* On someone else's machine (creds are operator-personal).
* Concurrently with the live bot trading on the same account
  (these tests place + cancel real orders; there's a
  `scripts/okx_flatten.py` style cleanup if anything is left behind,
  but the safest discipline is "stop the bot, run integration, restart
  the bot").

## How to run

Two ways to supply creds. **Method A (file) is strongly recommended**
because it sidesteps PowerShell / bash quoting bugs that can silently
mangle special characters in API secrets and passphrases.

### Method A: drop a `.env` file (recommended)

Create `tests/integration/.env` (gitignored — see `.gitignore` rule):

```env
OKX_API_KEY=...
OKX_API_SECRET=...
OKX_API_PASSPHRASE=...
# Optional overrides:
# OKX_INTEGRATION_SYMBOL=DOGE-USDT-SWAP
# OKX_DEMO_TRADING=false
```

Then:

```bash
python -m pytest tests/integration/test_okx_live.py -v -s
```

The test module loads `tests/integration/.env` at import time. Process
env still wins if both are present, so a one-off override via shell
`$env:OKX_DEMO_TRADING=true` beats the file without editing.

When creds load successfully, the test prints a redacted fingerprint
of each one (`len=N, head***tail`) at session start. Compare against
what the partner sent — if a length doesn't match, your file got mangled.

### Method B: shell env vars

```bash
# PowerShell — use SINGLE quotes so $, &, etc. aren't interpreted:
$env:OKX_API_KEY='...'
$env:OKX_API_SECRET='...'
$env:OKX_API_PASSPHRASE='...'
python -m pytest tests/integration/test_okx_live.py -v -s

# Bash / WSL:
export OKX_API_KEY='...'
export OKX_API_SECRET='...'
export OKX_API_PASSPHRASE='...'
python -m pytest tests/integration/test_okx_live.py -v -s
```

The `-s` is recommended so you see real-time progress (these tests
intentionally `print()` what they're doing — placing a real order
should never be silent).

### Sanity-checking which value the test actually loaded

If you see `50105: Request header OK-ACCESS-PASSPHRASE incorrect`,
the most likely cause is shell quoting at the env-var-set step. The
session-start fingerprint print tells you the length the test
actually saw. If it differs from what the partner sent, **switch to
Method A** — the file format does no escaping.

### Skipped collection when no creds

If neither the file nor the env vars supply creds, the credentialed
tests are skipped silently and only the public-instruments
connectivity smoke runs. `python -m pytest -q` (no `tests/integration/`
in the path) skips this whole directory entirely — `pytest.ini`'s
`norecursedirs = integration` keeps these out of the default suite.

## What gets placed

The OKX live test suite places **one tiny post-only LIMIT order at a
price that is intentionally far from market** (mid * 0.5 for a buy,
mid * 1.5 for a sell), specifically so it can't fill. It then
verifies the order shows up in `/orders-pending`, cancels it, and
verifies it's gone.

Worst case if the test is interrupted mid-flight: a single
small post-only order is left resting on the book. Run
`python scripts/okx_flatten.py` to clean up.

The tests do NOT exercise the full quote loop, do NOT place crossing
orders, do NOT close positions you didn't put on, and do NOT touch
your real trading.

## Required environment

| Var | Purpose |
|---|---|
| `OKX_API_KEY` | API key |
| `OKX_API_SECRET` | API secret |
| `OKX_API_PASSPHRASE` | API passphrase (third secret) |
| `OKX_REST_URL` (optional) | defaults to `https://www.okx.com` |
| `OKX_PRIVATE_WS_URL` (optional) | defaults to production prod URL |
| `OKX_DEMO_TRADING` (optional) | "true" to use OKX paper-trading |
| `OKX_INTEGRATION_SYMBOL` (optional) | symbol to test on, defaults to `DOGE-USDT-SWAP` |
| `OKX_INTEGRATION_TEST_SIZE_BASE` (optional) | base-asset size for the test order, defaults to `1000` (= 1 contract for DOGE) |
