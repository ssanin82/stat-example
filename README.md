# strat-example — a retired crypto market-making bot

A **previous-generation** single-symbol perpetual market maker, written in Python, that ran
live on OKX against `TON-USDT-SWAP`. It is **no longer in operation**. I am publishing it as a
reference implementation because it is easier to show working code than to describe it.

This is **not** a product, not a tutorial, and not something I maintain. It is a snapshot of a
system that quoted real money, with the strategy logic, the risk gates, the venue connectivity
and the test suite intact, and everything operational removed.

## What was removed, and why

- **All credentials, keys and environment files.** Nothing in this repository is a secret.
- **All live configuration profiles.** The tuned parameter sets carried commentary about
  realised performance and venue behaviour. They are not here.
- **All recorded market data, session results and simulation output.**
- **Deployment, infrastructure, recorder and operations tooling.**
- **Internal research notes, plans and design documents.**
- **A counterparty's name and the commercial arrangement around it**, replaced throughout
  with a neutral placeholder.

What remains is the engine, the backtest harness, the operator dashboard and the tests.

## What it does

A two-sided quoting bot on a single perpetual contract. The core is an
**Avellaneda-Stoikov / GLFT** reservation-price-and-spread formulation, with the quote geometry
fitted per instrument rather than inherited, plus a set of overlays and defensive gates that
exist because live trading produced them.

**Quoting**
- Reservation price and optimal spread from an inventory-aware AS/GLFT formulation
- Multi-level ladder construction with liquidity-adaptive placement
- Queue-position and join-depth control, rather than naive best-bid/best-ask joining
- Expected-edge calculation per quote level, net of fees

**Signals and overlays**
- Order-flow score and fill-burst detection
- Basis-regime classification and gating
- Feature firing-rate accounting, so an overlay that never fires is visible as such

**Risk and defence**, all enforced identically in simulation and in production
- Inventory caps, drift gates and consistency checks
- Drawdown and session-loss kill gates
- Markout-adverse pause and at-touch adverse-selection pause
- Fast-move cancel, staleness gates, pre-trade notional caps

**Execution and venue layer**
- REST and WebSocket clients for OKX, Binance and Bluefin
- Order-book construction from incremental updates, with reconnect and gap-recovery
- Order management under rate limits, with post-restart state reconciliation
- Inbound and market-data timing instrumentation

**Validation**
- A queue-aware fill model rather than an optimistic one
- Event-driven backtest and replay harness under `app/backtest/` — driver, event stream,
  paper executor, scenario and simulation storage, report generation
- Around 400 test modules under `tests/`, including integration tests that exercise the
  gates and the order lifecycle rather than only unit behaviour

**Operator dashboard** (`frontend/`)
- Next.js read-only dashboard over the live account state, written to poll the **exchanges
  directly** rather than the bot, so it still tells the truth when the bot is wedged
- Open orders, positions, fills, equity history, maker volume and rebate-tier progress
- Resolves the host instance and its log bucket by **tag lookup at runtime** rather than
  hardcoded resource names, which is why no infrastructure identifiers appear in this
  repository
- Session reporting and P&L attribution rendered from the recorded exposure bars

## On the engineering

The thing worth looking at is not the alpha. It is that the **risk gates and pre-trade checks
are the same code path in simulation and in production**, so a simulated run cannot pass under
rules the live system would have blocked. Most of the defensive machinery in `app/` exists
because something happened on a live venue and was then written down as a gate and a test.

## Running it

```
pip install -r requirements.txt
pytest

cd frontend && npm install && npm run dev
```

The tests run without credentials or network access. Live operation required configuration
and venue credentials that are deliberately absent from this repository, so it will not trade
as published, by design.

## Status

Retired. Kept public as a code sample. No support, no issues, no roadmap.
