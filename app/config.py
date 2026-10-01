from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import clock as _clock


# Substrings that mark a Settings field as a secret. Matched against
# the UPPER_SNAKE_CASE alias (e.g. "BINANCE_API_SECRET"). Used by
# Settings.sanitized_dict() to redact values before they leak via
# /config or scripts/stats_snapshot.py output.
#
# *_FILE keys (e.g. HL_SECRET_KEY_FILE) are paths to secrets, not the
# secrets themselves -- not redacted.
_SENSITIVE_NAME_SUBSTRINGS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSPHRASE",
    "PRIVATE_KEY",
    "API_KEY",
)


def _is_sensitive_key_name(name: str) -> bool:
    if name.endswith("_FILE"):
        return False
    return any(s in name for s in _SENSITIVE_NAME_SUBSTRINGS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Operator-set tag for cross-session A/B / experiment tracking
    # (analysis-day instrumentation N7, 2026-05-10). Free-form: a
    # config hash, a human label like ``"cap-4x-2026-05-09"``, or
    # whatever lets postmortem aggregations group runs without
    # juggling timestamps. Surfaced verbatim through
    # ``session_summary.json``. Empty string when unset (reads as
    # ``None`` downstream).
    experiment_id: str = Field(default="", alias="EXPERIMENT_ID")

    # Venue selector. Drives adapter + stream construction in ``app.exchange.factory``.
    # Allowed values: ``"hyperliquid"`` (default, production-today), ``"grvt"``
    # (scaffolded; see ``app/exchange/grvt_*.py``) and ``"bluefin"`` (Sui
    # appchain, see ``app/exchange/bluefin_*.py``). No hardcoding outside the factory.
    exchange: str = Field(default="hyperliquid", alias="EXCHANGE")

    hl_secret_key: str = Field(default="", alias="HL_SECRET_KEY")
    hl_secret_key_file: str = Field(default="", alias="HL_SECRET_KEY_FILE")
    hl_account_address: str = Field(default="", alias="HL_ACCOUNT_ADDRESS")
    hl_base_url: str = Field(
        default="https://api.hyperliquid.xyz",
        alias="HL_BASE_URL",
    )
    hl_ws_url: str = Field(
        default="wss://api.hyperliquid.xyz/ws",
        alias="HL_WS_URL",
    )

    # GRVT credentials / routing.
    grvt_api_key: str = Field(default="", alias="GRVT_API_KEY")
    grvt_api_secret: str = Field(default="", alias="GRVT_API_SECRET")
    grvt_api_secret_file: str = Field(default="", alias="GRVT_API_SECRET_FILE")
    grvt_account_address: str = Field(default="", alias="GRVT_ACCOUNT_ADDRESS")
    grvt_sub_account_id: str = Field(default="", alias="GRVT_SUB_ACCOUNT_ID")
    grvt_env: str = Field(default="prod", alias="GRVT_ENV")
    grvt_edge_url: str = Field(default="", alias="GRVT_EDGE_URL")
    grvt_trade_url: str = Field(default="", alias="GRVT_TRADE_URL")
    grvt_market_data_url: str = Field(default="", alias="GRVT_MARKET_DATA_URL")
    grvt_public_ws_url: str = Field(default="", alias="GRVT_PUBLIC_WS_URL")
    grvt_private_ws_url: str = Field(default="", alias="GRVT_PRIVATE_WS_URL")

    # Bluefin credentials / routing (Sui appchain, off-chain orderbook).
    #
    # Operational model for the first live session:
    #
    #   * Operator enables 1-Click Trading (1CT) in the Bluefin UI, which
    #     mints a session key on-chain. The Bluefin UI exposes the session
    #     key's hex — that goes into ``BLUEFIN_PRIVATE_KEY``.
    #   * ``BLUEFIN_ACCOUNT_ADDRESS`` is the operator's main Sui wallet
    #     (where the USDC collateral and open positions live).
    #   * With ``BLUEFIN_ONE_CT_ENABLED=false`` the configured private key
    #     is used directly (not recommended; exposes wallet key to any
    #     process with env read access).
    #
    # On-chain minting of a fresh 1CT grant (``upsertSubAccount`` in the
    # reference TS SDK) is out of scope for the first adapter ship.
    bluefin_network: str = Field(default="SUI_PROD", alias="BLUEFIN_NETWORK")
    bluefin_private_key: str = Field(default="", alias="BLUEFIN_PRIVATE_KEY")
    bluefin_account_address: str = Field(default="", alias="BLUEFIN_ACCOUNT_ADDRESS")
    bluefin_one_ct_enabled: bool = Field(default=True, alias="BLUEFIN_ONE_CT_ENABLED")
    bluefin_one_ct_duration_hours: float = Field(
        default=24.0, gt=0, alias="BLUEFIN_ONE_CT_DURATION_HOURS"
    )
    # Bluefin moved from a single ``dapi.api.*`` subdomain to three (pro-sdk
    # layout): general API (``api.``), auth (``auth.api.``), trade
    # (``trade.api.``) plus a shared WebSocket (``stream.api.``). Operators
    # that pin a specific colocation endpoint can override individually.
    # The ``bluefin_rest_url`` field continues to name the general/account
    # endpoint for backward compatibility with existing env profiles.
    bluefin_rest_url: str = Field(default="", alias="BLUEFIN_REST_URL")
    bluefin_auth_url: str = Field(default="", alias="BLUEFIN_AUTH_URL")
    bluefin_trade_url: str = Field(default="", alias="BLUEFIN_TRADE_URL")
    bluefin_public_ws_url: str = Field(default="", alias="BLUEFIN_PUBLIC_WS_URL")
    bluefin_private_ws_url: str = Field(default="", alias="BLUEFIN_PRIVATE_WS_URL")

    # Bluefin Pro's ``PUT /api/v1/trade/orders/cancel`` returns 202 Accepted
    # asynchronously — the order is queued for cancellation, not synchronously
    # removed from the matching engine. The authoritative signal is a private
    # WS ``AccountOrderUpdate`` with ``cancellationReason`` set
    # (``OrderCancellationUpdate``). To prevent a cancel-vs-replacement race
    # (observed 2026-04-23: 5× SUI-PERP BUY fills within 2s because the bot
    # placed a replacement before the cancel took effect), the Bluefin
    # adapter tracks in-flight cancels and exposes ``has_pending_cancel``.
    # The execution layer consults this before enqueuing a same-side
    # replacement: if True, skip one tick and retry.
    #
    # ``BLUEFIN_CANCEL_CONFIRM_TIMEOUT_SECONDS`` is the safety ceiling — if
    # the WS confirmation never arrives (dropped frame, stream stall), the
    # pending entry clears and quoting proceeds. Default 3.0s: long enough
    # for normal matching-engine latency, short enough that a hiccup
    # doesn't block quoting for more than a couple of ticks.
    #
    # ``BLUEFIN_CANCEL_CONFIRM_GATE_ENABLED`` is an operator escape hatch.
    # Setting it to false makes ``has_pending_cancel`` always return False,
    # disabling the gate and falling back to the old cancel-then-replace
    # behaviour. Use only if the gate causes unexpected quoting issues.
    bluefin_cancel_confirm_timeout_seconds: float = Field(
        default=3.0, gt=0, alias="BLUEFIN_CANCEL_CONFIRM_TIMEOUT_SECONDS"
    )
    bluefin_cancel_confirm_gate_enabled: bool = Field(
        default=True, alias="BLUEFIN_CANCEL_CONFIRM_GATE_ENABLED"
    )

    # Deadlock watchdog (see app/watchdog.py). The bot has multiple
    # internal gates (pending-cancel confirmation, uncertain-order
    # state, cross-venue-cancel cascade, POST_ONLY_WOULD_TRADE
    # rejections) that each have their own timeout but can latch
    # collectively longer than any single gate's timeout. Observed
    # 2026-04-24: quote engine emitted non-NONE for 1 hour while the
    # execution layer placed zero orders — silent deadlock until manual
    # restart. The watchdog detects this pattern and exits the process
    # so the process supervisor (systemd / container manager / k8s)
    # restarts with clean in-memory state. Turn OFF only if you're
    # intentionally running the bot without auto-restart infrastructure.
    watchdog_enabled: bool = Field(default=True, alias="WATCHDOG_ENABLED")
    # How long the execution layer can go without dispatching an order
    # attempt before the watchdog considers it stuck, PROVIDED the quote
    # engine has been emitting non-NONE decisions in the activity
    # window. Default 600 s (10 min) — generous to avoid false positives
    # in genuinely quiet markets.
    watchdog_no_place_attempt_seconds: float = Field(
        default=600.0, gt=0, alias="WATCHDOG_NO_PLACE_ATTEMPT_SECONDS"
    )
    # The quote engine must have emitted at least one non-NONE decision
    # within this many seconds before the watchdog will fire. If the
    # engine is itself silent (HOLD_ALL / NONE), that's a different
    # problem and we don't want to bounce the container. Default 120 s.
    watchdog_quote_activity_window_seconds: float = Field(
        default=120.0, gt=0, alias="WATCHDOG_QUOTE_ACTIVITY_WINDOW_SECONDS"
    )
    # Watchdog polls this often. Short enough to react within a minute;
    # long enough that the thread costs nothing. Default 30 s.
    watchdog_check_interval_seconds: float = Field(
        default=30.0, gt=0, alias="WATCHDOG_CHECK_INTERVAL_SECONDS"
    )

    # --- Telegram outbound notifications + inbound commands ---
    # See ``app/telegram_notifier.py`` and ``app/telegram_commands.py``.
    # ``TELEGRAM_BOT_TOKEN`` is the only secret; the rest are non-secret
    # routing values that live in ``config/telegram.env`` (gitted).
    # When the token is empty, the entire Telegram subsystem is disabled
    # — both notifier and command poller become no-ops, so local dev /
    # tests work without any Telegram setup.
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")

    # --- Backtest-recorder orchestration (v1.4.239) -------------------------
    #
    # When ``recording_enabled`` is True, the colo ops scripts launch the
    # backtest recorder (`backtesting/recorder/record.py`) as a separate
    # process BEFORE the bot's trading service starts, and shut it down
    # AFTER the bot has cleanly stopped. The finalized session is fetched
    # back to the operator's laptop on stop/redeploy.
    #
    # The bot itself does NOT read or write any recordings; the flag is
    # consumed by `scripts/colo_start_bot.ps1` / `colo_stop_bot.ps1` and
    # surfaced via the periodic S3 status payload so the dashboard can
    # show a recording indicator.
    #
    # Default off (no behavioural change for profiles that don't opt in).
    # Toggled per-profile in `config/profiles/*.env`.
    recording_enabled: bool = Field(default=False, alias="RECORDING_ENABLED")
    # Which recorder profile (`backtesting/recorder/profiles/<name>.yaml`)
    # the orchestration launches when recording_enabled=True. Empty falls
    # back to the bot's exchange + symbol routed to a sensible default
    # (e.g. `okx-colo` for OKX).
    recording_profile: str = Field(default="", alias="RECORDING_PROFILE")
    telegram_ops_chat_id: str = Field(default="", alias="TELEGRAM_OPS_CHAT_ID")
    telegram_trades_chat_id: str = Field(default="", alias="TELEGRAM_TRADES_CHAT_ID")
    # Comma-separated list of integer Telegram user_ids allowed to issue
    # commands. Empty means inbound commands are disabled even if the
    # token is set.
    telegram_allowed_user_ids: str = Field(
        default="", alias="TELEGRAM_ALLOWED_USER_IDS"
    )
    # Optional comma-separated allowlist of chat_ids the inbound poller
    # responds to. Empty means "any chat where an allowed user writes" —
    # which for DM-only setups is fine because DM chat_id == user_id.
    # Tighten this if you ever add a control group/channel.
    telegram_allowed_chat_ids: str = Field(
        default="", alias="TELEGRAM_ALLOWED_CHAT_IDS"
    )
    # Trades-channel throttle: token bucket. Up to ``burst`` messages
    # can be sent immediately, then refills at ``rate_per_second``.
    # Drops over the cap are coalesced into a single "+N more fills
    # suppressed" notice.
    telegram_trade_rate_per_second: float = Field(
        default=0.5, ge=0, alias="TELEGRAM_TRADE_RATE_PER_SECOND"
    )
    telegram_trade_burst: int = Field(
        default=3, ge=1, alias="TELEGRAM_TRADE_BURST"
    )
    # Coalesce window for fill notifications. Fills within this window
    # are batched into a single message (cuts noise during fast bursts).
    # 0 disables coalescing (one message per fill).
    telegram_fill_coalesce_seconds: float = Field(
        default=5.0, ge=0, alias="TELEGRAM_FILL_COALESCE_SECONDS"
    )
    # Long-poll timeout used by the inbound poller's ``getUpdates`` call.
    # Telegram caps this at 50; we use 30 by default. Lower values mean
    # tighter shutdown latency at the cost of more requests.
    telegram_long_poll_timeout_seconds: int = Field(
        default=30, ge=1, le=50, alias="TELEGRAM_LONG_POLL_TIMEOUT_SECONDS"
    )
    # Per-user inbound rate limit. 1 command per N seconds. Stops a
    # stuck-finger from accidentally flooding the bot.
    telegram_command_min_interval_seconds: float = Field(
        default=2.0, ge=0, alias="TELEGRAM_COMMAND_MIN_INTERVAL_SECONDS"
    )
    # Two-step confirmation timeout for destructive commands
    # (/kill, /flatten, /restart). Operator must reply within this many
    # seconds with ``/<cmd> confirm`` or the request is dropped.
    telegram_confirm_timeout_seconds: float = Field(
        default=30.0, gt=0, alias="TELEGRAM_CONFIRM_TIMEOUT_SECONDS"
    )
    # Account volume refresh interval for ``/status``'s 7d/30d figures.
    # The fetch is paginated REST and takes seconds on a busy 30-day
    # window — too slow to do inline on the command path. A background
    # daemon thread refreshes the cache; ``/status`` always reads cached.
    # Default 1800 s = 30 min: volumes don't move that fast and this
    # keeps Bluefin REST load low. Set to 0 to disable the refresher
    # (``/status`` will simply omit the volume rows).
    telegram_volume_refresh_seconds: float = Field(
        default=1800.0, ge=0.0, alias="TELEGRAM_VOLUME_REFRESH_SECONDS"
    )

    # --- Best-effort session resume on restart ---
    # See ``app/session_resume.py``. When true, on startup the bot reads
    # the most-recent ``bot_start`` event from the SQLite ``bot_events``
    # table; if it is within ``BOT_SESSION_CONTINUITY_MAX_HOURS`` of now,
    # it restores cumulative METRIC state (session_id, session_started_at_utc,
    # session_fill_count, realized_pnl_usd, fees_usd, peak_equity_usd) by
    # re-aggregating from the ``fills`` table.
    #
    # CRITICAL safety property: BEHAVIORAL state is NEVER restored
    # (pending cancels, hash-side cache, cancel-confirmation gates,
    # cooldown timers, toxicity window samples, position cache). Restoring
    # those would defeat the watchdog (re-introduce the very deadlock the
    # restart was breaking). Tests guard this property explicitly.
    #
    # Default is OFF: a fresh-start-on-restart bot is the safer default
    # for a market-making process. Enable when you specifically want
    # cumulative-PnL continuity across restarts (e.g. session that
    # spans a watchdog-triggered bounce).
    bot_resume_session_on_restart: bool = Field(
        default=False, alias="BOT_RESUME_SESSION_ON_RESTART"
    )
    bot_session_continuity_max_hours: float = Field(
        default=24.0, gt=0, alias="BOT_SESSION_CONTINUITY_MAX_HOURS"
    )

    # --- Bluefin POST /orders rate-limit defense ---
    # Observed 2026-04-23: Bluefin Pro returned HTTP 429 on ~98% of
    # place-order requests during a burst of ~3 req/sec (bid + ask per
    # 0.5s quote cycle + rapid cancel-and-replace). Root cause is
    # CloudFront rate-limiting in front of POST /api/v1/trade/orders;
    # cancels on PUT /cancel were unaffected by the same burst. Two
    # guards here:
    #
    # ``BLUEFIN_MIN_PLACE_INTERVAL_SECONDS``: preemptive per-process
    #   throttle. Adapter enforces a minimum interval between
    #   POST /orders calls regardless of side or symbol. Simple leaky
    #   bucket with capacity 1. Default 0.2 s (5 req/sec) — conservative
    #   vs. the 18 req/sec peak we observed getting rate-limited.
    #   Set to 0 to disable.
    # ``BLUEFIN_REST_429_MAX_RETRIES``: on a 429 response from any
    #   Bluefin REST call, retry up to this many times. Exponential
    #   backoff plus ``Retry-After`` header (if server provides).
    #   Default 3 — clears transient bursts without masking a
    #   sustained rate-limit problem.
    # ``BLUEFIN_REST_429_BASE_BACKOFF_SECONDS``: initial backoff on
    #   first 429. Doubled each retry. Per-retry sleep capped at 10 s
    #   so the quote loop isn't starved.
    bluefin_min_place_interval_seconds: float = Field(
        default=0.2, ge=0, alias="BLUEFIN_MIN_PLACE_INTERVAL_SECONDS"
    )
    bluefin_rest_429_max_retries: int = Field(
        default=3, ge=0, le=10, alias="BLUEFIN_REST_429_MAX_RETRIES"
    )
    bluefin_rest_429_base_backoff_seconds: float = Field(
        default=0.5, gt=0, alias="BLUEFIN_REST_429_BASE_BACKOFF_SECONDS"
    )

    # ``BLUEFIN_SIGNED_AT_BACKOFF_MS``: milliseconds to subtract from
    #   ``int(_clock.time_seconds() * 1000)`` before using it as ``signedAtMillis``
    #   in a place request. Bluefin's server enforces an undocumented
    #   upper bound ``signedAtMillis <= server_now`` whose rejection is
    #   reported as the misleading "SignedAtUtcMillis must be no earlier
    #   than 1 minute in the past" error. When the local clock is drifted
    #   forward vs Bluefin's clock (commonly ~1 s on Windows, potentially
    #   up to several seconds before NTP sync), every placement 400s.
    #   Subtracting a safety offset keeps ``signedAtMillis`` safely in the
    #   server's past without violating the 60-s freshness lower bound.
    #   Default 2000 ms tolerates up to 2 s of local-ahead drift. Raise
    #   if the operator's host clock drifts more than that; lower (or set
    #   to 0) once Bluefin publishes an explicit tolerance contract.
    #   Verified empirically 2026-04-24 via
    #   ``scripts/bluefin_place_cancel_probe.py``: with backoff=0 every
    #   placement 400s; with backoff in [3000, 30000] every placement is
    #   accepted 202.
    bluefin_signed_at_backoff_ms: int = Field(
        default=2000, ge=0, le=55000, alias="BLUEFIN_SIGNED_AT_BACKOFF_MS"
    )
    # ``BLUEFIN_CANCEL_BY_HASH_WORKAROUND``: when true (default), cancel
    #   requests omit the ``orderHashes`` body key, triggering Bluefin's
    #   documented "cancel all orders for the symbol" behaviour instead
    #   of selective cancel-by-hash. Rationale: as of 2026-04-24, cancel
    #   with a specific ``orderHashes`` list returns HTTP 202 but
    #   silently drops — the orders remain on the book for 60+ seconds
    #   (verified as broken, not merely slow, with isolated polling).
    #   Cancel-all returns 202 and actually removes the orders in ~170
    #   ms. Until Bluefin fixes the selective path, we treat every cancel
    #   as a whole-symbol wipe; the quote loop's next reconcile cycle
    #   re-places any side that was collaterally cancelled. Set to false
    #   once Bluefin confirms the selective path is fixed.
    bluefin_cancel_by_hash_workaround: bool = Field(
        default=True, alias="BLUEFIN_CANCEL_BY_HASH_WORKAROUND"
    )

    # ============================================================
    # Binance Futures USDM trading-venue credentials (added 2026-04-30
    # for plans/20260420-binance-move). Distinct from BINANCE_*_URL /
    # BINANCE_SYMBOL above, which are for cross-venue REFERENCE pricing
    # (Bluefin/HL/GRVT trading vs Binance reference). When Binance is
    # the trading venue, REFERENCE_EXCHANGE auto-overrides to "off"
    # (see ``validate_cross_venue_reference_self_match``); these
    # ``binance_api_*`` and ``binance_rest_url`` fields supply the
    # private trading credentials.
    # ============================================================
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    # Binance Futures USDM REST base URL. The default routes to the
    # production endpoint; ``api.binance.com`` is spot, ``fapi.binance.com``
    # is USDT-margined futures, ``dapi.binance.com`` is COIN-margined.
    binance_rest_url: str = Field(
        default="https://fapi.binance.com",
        alias="BINANCE_REST_URL",
    )
    # Binance Futures private user-data WS endpoint. The bot constructs
    # the full URL as ``{base}/ws/{listenKey}`` after spawning the
    # listenKey via ``POST /fapi/v1/listenKey``.
    binance_private_ws_url: str = Field(
        default="wss://fstream.binance.com",
        alias="BINANCE_PRIVATE_WS_URL",
    )
    # listenKey lifecycle: Binance silently expires keys after 60 min
    # of no PUT keepalive. Default 30 min keepalive interval doubles
    # the safety margin. Failure modes (404 listenKey not found, key
    # rotated by another process) trigger a re-spawn.
    binance_listen_key_keepalive_seconds: float = Field(
        default=1800.0,
        gt=0,
        alias="BINANCE_LISTEN_KEY_KEEPALIVE_SECONDS",
    )
    # Binance signed-request ``recvWindow`` parameter (ms). Server
    # rejects requests where ``timestamp + recvWindow < server_now``,
    # so 5000 ms tolerates ~5s of clock skew. Cap at 60s per Binance
    # docs.
    binance_recv_window_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        alias="BINANCE_RECV_WINDOW_MS",
    )

    # ============================================================
    # OKX Futures (USDT-margined SWAP) trading-venue credentials
    # (added 2026-05-04 for plans/20260504-okx-setup). OKX requires
    # THREE secrets (not two): API key, API secret, and a passphrase
    # set at key-creation time. All three flow as headers on signed
    # requests (OK-ACCESS-KEY / OK-ACCESS-SIGN / OK-ACCESS-PASSPHRASE).
    # See app/exchange/okx_client.py for signing details.
    # ============================================================
    okx_api_key: str = Field(default="", alias="OKX_API_KEY")
    okx_api_secret: str = Field(default="", alias="OKX_API_SECRET")
    okx_api_passphrase: str = Field(default="", alias="OKX_API_PASSPHRASE")
    # OKX REST base URL. Production: https://www.okx.com (the same
    # host serves all market types — SWAP / FUTURES / SPOT — pathed
    # under /api/v5/...). Demo / paper-trading uses the same host
    # but requires the ``x-simulated-trading: 1`` header on every
    # request (handled per-request in okx_client.py).
    okx_rest_url: str = Field(
        default="https://www.okx.com",
        alias="OKX_REST_URL",
    )
    # OKX uses ONE WebSocket host for both public and private channels;
    # the private endpoint is pathed under /v5/private and requires a
    # signed login frame. The bot opens distinct connections for the
    # two channels because keepalive cadences and reconnect semantics
    # differ.
    okx_public_ws_url: str = Field(
        default="wss://ws.okx.com:8443/ws/v5/public",
        alias="OKX_PUBLIC_WS_URL",
    )
    okx_private_ws_url: str = Field(
        default="wss://ws.okx.com:8443/ws/v5/private",
        alias="OKX_PRIVATE_WS_URL",
    )
    # Demo trading: when true, all REST + WS requests carry the
    # ``x-simulated-trading: 1`` header / param. Use during initial
    # smoke testing before flipping to a real funded account.
    okx_demo_trading: bool = Field(
        default=False,
        alias="OKX_DEMO_TRADING",
    )
    # OKX does not have a Binance-style ``recvWindow`` parameter; the
    # signed-request timestamp is freshness-checked server-side with
    # a fixed ~30s tolerance. We keep our local clock NTP-synced and
    # don't expose this as a knob.

    private_ws_enabled: bool = Field(default=True, alias="PRIVATE_WS_ENABLED")
    private_ws_queue_max: int = Field(default=2000, ge=64, alias="PRIVATE_WS_QUEUE_MAX")
    private_ws_reconnect_initial_seconds: float = Field(
        default=0.5, gt=0, alias="PRIVATE_WS_RECONNECT_INITIAL_SECONDS"
    )
    private_ws_reconnect_max_seconds: float = Field(
        default=60.0, gt=0, alias="PRIVATE_WS_RECONNECT_MAX_SECONDS"
    )
    private_ws_recovery_fill_catchup_ticks: int = Field(
        default=5, ge=0, alias="PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS"
    )
    # Application-level JSON keepalive (Hyperliquid ``{"method":"ping"}``); not a substitute for
    # protocol ping, but required because server idle policy does not treat transport ping as activity.
    private_ws_app_keepalive_seconds: float = Field(
        default=22.0,
        gt=0,
        alias="PRIVATE_WS_APP_KEEPALIVE_SECONDS",
    )
    private_ws_inactive_reconnect_seconds: float = Field(
        default=0.75,
        gt=0,
        alias="PRIVATE_WS_INACTIVE_RECONNECT_SECONDS",
    )
    # If no WS text frame received for this long, emit a rate-limited
    # ``grvt_private_ws_idle_warn`` log (observability only — does not
    # reconnect). ``0`` disables the warn log.
    private_ws_idle_warn_seconds: float = Field(
        default=120.0,
        ge=0,
        alias="PRIVATE_WS_IDLE_WARN_SECONDS",
    )
    # If no WS text frame received for this long, force a reconnect
    # (GRVT-only: catches a stuck subscription where TCP-layer ping/pong
    # is healthy but no user events are flowing). Decoupled from
    # ``PRIVATE_WS_IDLE_WARN_SECONDS`` so thin-book symbols (AXS, NEAR)
    # can observe natural fill-silence windows without reconnect-looping.
    # ``0`` disables the force-reconnect. On fast-trading symbols set to
    # e.g. 60 to catch stuck cancels fast; on slow-trading symbols leave
    # at the default 300 (5 min).
    private_ws_idle_reconnect_seconds: float = Field(
        default=300.0,
        ge=0,
        alias="PRIVATE_WS_IDLE_RECONNECT_SECONDS",
    )

    public_ws_enabled: bool = Field(default=True, alias="PUBLIC_WS_ENABLED")
    public_ws_reconnect_initial_seconds: float = Field(
        default=0.5, gt=0, alias="PUBLIC_WS_RECONNECT_INITIAL_SECONDS"
    )
    public_ws_reconnect_max_seconds: float = Field(
        default=60.0, gt=0, alias="PUBLIC_WS_RECONNECT_MAX_SECONDS"
    )
    # Wall-clock age since last public BBO message (or disconnected socket).
    public_ws_stale_warn_seconds: float = Field(
        default=2.0, gt=0, alias="PUBLIC_WS_STALE_WARN_SECONDS"
    )
    public_ws_stale_kill_seconds: float = Field(
        default=8.0, gt=0, alias="PUBLIC_WS_STALE_KILL_SECONDS"
    )
    # SENT + cloid: stop blocking the side after unresolved orderStatus / transport failures.
    #
    # Default: 1.0 s. Calibrated against actual production
    # latency for the OKX colo deployment. Snapshot
    # v1.4.59-260518-173229 dashboard LATENCY panel:
    #
    #   Tx send → ack: min 3.13 ms · med 3.27 ms · p95 3.94 ms · max 4.73 ms (n=1005)
    #
    # 1 s is 250× the MEDIAN ACK latency and 200× the max
    # observed across a thousand samples. If an ACK is genuinely
    # delayed past 1 s, something is broken — either the WS
    # connection, the venue, or the bot's WS handler. Waiting
    # longer doesn't help.
    #
    # History:
    #   pre-v1.4.62: 120 s default. Snapshot v1.4.61-260518-181059
    #     caught a SELL stuck in SENT for 136 s because the WS ACK
    #     didn't process. During those 136 s the reconciler
    #     correctly emitted NoOp:in_flight_wait every tick, but
    #     the operator saw "no SELL on the book" — a 2-minute
    #     wedge from the operator's view.
    #   v1.4.62: reduced to 15 s. Still ~3000× normal ACK latency
    #     on this colo deployment — effectively useless for fast
    #     recovery.
    #   v1.4.63: 1.0 s. Calibrated to actual venue latency.
    #
    # Trade-off: if a legitimate ACK is delayed past 1 s (we've
    # never seen this on OKX colo), the bot transitions the WO to
    # REJECTED and places fresh. The delayed ACK then lands as an
    # orphan, which the reconcile path cleans up via
    # ``_cancel_orphan_remote_order``. The duplicate-orders
    # failure mode is detected (``dup_buy`` / ``dup_sell`` in
    # ``residual_order_audit``) and cleaned up automatically.
    #
    # Operators on cross-region or higher-latency setups should
    # override via ``SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS`` to
    # something like 2-5 s. 1 s is the OKX-colo-aggressive default.
    sent_order_unresolved_timeout_seconds: float = Field(
        default=1.0,
        gt=0,
        alias="SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS",
    )
    # Max consecutive ambiguous REST polls before giving up. Tuned
    # to match the 1 s timeout — at ~100 ms per REST poll worst
    # case, 3 polls = 300 ms, well under the 1 s timeout. The
    # ambiguous-poll counter is the secondary safety net; the
    # timeout is primary.
    sent_order_unresolved_max_ambiguous_polls: int = Field(
        default=3,
        ge=0,
        alias="SENT_ORDER_UNRESOLVED_MAX_AMBIGUOUS_POLLS",
    )
    # Periodic reconcile *request* cadence (wall-clock). This schedules requests only; actual
    # reconcile execution still obeys cooldown/backoff gates in ``OrderManager``.
    open_orders_reconcile_request_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="OPEN_ORDERS_RECONCILE_REQUEST_INTERVAL_SECONDS",
    )
    # Actual open-orders reconcile cooldown between runs (independent of request cadence).
    # Reduced from 30s to 10s so that, after a mismatch/desync repair, we can re-verify
    # exchange state on a human-observable horizon rather than a full half-minute window
    # during which phantom orders could go undetected.
    open_orders_reconcile_cooldown_seconds: float = Field(
        default=10.0,
        gt=0,
        alias="OPEN_ORDERS_RECONCILE_COOLDOWN_SECONDS",
    )
    # Extra cooldown after openOrders 429/rate-limit outcomes.
    open_orders_reconcile_429_backoff_seconds: float = Field(
        default=60.0,
        ge=0,
        alias="OPEN_ORDERS_RECONCILE_429_BACKOFF_SECONDS",
    )
    # v1.4.43 rate-limit Tier 2 open-orders cache REMOVED in v1.4.55
    # (wedge-elimination Phase 1). The cache existed to service the
    # OLD ``cancel_all_orders_for_symbol`` path that iterated the
    # exchange snapshot; Phase 1 rewrote cancel-all to use local WO
    # state + dispatcher, eliminating all consumers. The
    # ``CANCEL_ALL_OPEN_ORDERS_CACHE_TTL_SECONDS`` env var is
    # ignored. If present in a profile, no harm — pydantic-settings
    # silently drops unknown env aliases.
    # Deprecated: old tick-based periodic reconcile request trigger (event-driven runtime no longer
    # uses this on the active path). Kept only for env compatibility.
    open_orders_reconcile_interval_ticks: int = Field(
        default=20,
        ge=1,
        alias="OPEN_ORDERS_RECONCILE_INTERVAL_TICKS",
    )
    cancel_pending_rest_watchdog_ticks: int = Field(
        default=5, ge=1, alias="CANCEL_PENDING_REST_WATCHDOG_TICKS"
    )
    # Side-local cancel-pending convergence timeout; after this, execution triggers guarded
    # recovery/reconcile and keeps same-side placement suppressed until convergence.
    #
    # Default raised from 15 s → 30 s after the overnight-17:22-UTC ETH
    # session showed 6,615 ``cancel_pending_timeout_recovery`` warnings in
    # 31 min — cancels routinely took 15-25 s to ack on GRVT under load,
    # so the 15 s threshold was panicking on normal cancel RTTs and
    # triggering emergency reconciles that piled up state uncertainty.
    # 30 s gives cancels genuine slack before we declare the side stuck.
    cancel_pending_unresolved_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS",
    )
    # v1.4.75 wedge-elimination-cleanup Phase 2B — full-state stale-
    # ghost reaper. Pre-Phase-2B the cancel-pending watchdog
    # (``_maybe_handle_cancel_pending_timeout``) only inspected the
    # inside-rung slots (``working_bid`` / ``working_ask``). Orphan-
    # slot, hydrated, or outer-rung CANCEL_PENDING / DESYNC / stale
    # SENT WOs were NEVER reaped, leaving residual state that drove
    # the SUPPRESSED-state wedge (v1.4.66) and lifecycle duplicate
    # bugs.
    #
    # Phase 2B's ``_reap_stale_ghosts`` walks ``state.all_working_orders()``
    # every tick and force-terminals stale entries. These knobs tune
    # the thresholds.

    # Age (seconds) past which a DESYNC WO is removed from local
    # state entirely (the bot has already issued recovery cancels;
    # the WO is a tombstone). 30 s default gives the cancel time to
    # round-trip while preventing indefinite state buildup.
    desync_reap_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="DESYNC_REAP_TIMEOUT_SECONDS",
    )
    # Safety multiplier applied to ``sent_order_unresolved_timeout_seconds``
    # before the reaper force-terminals a stuck SENT WO. The legitimate
    # SENT timeout path (``_clear_sent_ambiguous_polls`` + side-unresolved
    # latch) handles fast cases; this reaper is the LAST-RESORT
    # safety net for WOs that slipped both paths.
    #
    # At default settings: 120 s × 10 = 1200 s (20 min) — generous
    # so the reaper never preempts a legitimately-slow SENT (e.g.,
    # batch of slow OKX HTTP responses) but catches the genuinely-
    # wedged ones.
    sent_reaper_safety_multiplier: float = Field(
        default=10.0,
        gt=0,
        alias="SENT_REAPER_SAFETY_MULTIPLIER",
    )
    # Master switch (default ON). Set to False to disable reaper
    # entirely — useful for diagnosis if the reaper itself misfires.
    reaper_enabled: bool = Field(
        default=True,
        alias="REAPER_ENABLED",
    )
    # v1.4.92 wedge-elimination-cleanup Phase 4A cutover —
    # ``quote_build_result_strict_consumption`` setting REMOVED along
    # with the Phase 2D runtime audit it gated. The typed
    # ``BuildCommand`` sum-type (``QuoteBoth`` / ``QuoteOneSided`` /
    # ``NoQuote`` / ``ResidualFlatten``) makes the v1.4.66 regression
    # class structurally impossible. If your env var
    # ``QUOTE_BUILD_RESULT_STRICT_CONSUMPTION`` is set, you can
    # safely remove it.
    # During active side uncertainty/desync, allow more frequent (but bounded) forced reconcile
    # attempts than the normal periodic cooldown. 3s (vs previous 5s) keeps fresher
    # convergence checks right after a cancel-replace or exchange-mismatch repair.
    desync_reconcile_interval_seconds: float = Field(
        default=3.0,
        gt=0,
        alias="DESYNC_RECONCILE_INTERVAL_SECONDS",
    )
    desync_reconcile_429_backoff_seconds: float = Field(
        default=15.0,
        ge=0,
        alias="DESYNC_RECONCILE_429_BACKOFF_SECONDS",
    )
    # Failsafe auto-release window for a side convergence latch if local slot is already empty.
    side_unresolved_suppression_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        alias="SIDE_UNRESOLVED_SUPPRESSION_TIMEOUT_SECONDS",
    )
    # ``engine_no_quote_persistent`` diagnostic threshold. Counts
    # consecutive ticks where ``build_quotes`` returned mode="no_quote".
    # When the streak reaches this many ticks, emit a WARNING event
    # naming the most likely cause (residual_flatten gating, latched
    # side_unresolved, sub-spec dust residual). Without this, the bot
    # can silently produce no orders for the full 600 s execution_idle
    # window before the deadlock watchdog fires (reproduced 2026-05-08
    # in snapshot 260507114312). Tuned for the 0.5 s OKX quote loop:
    # 20 ticks ≈ 10 s — early enough to act, long enough to not spam.
    engine_no_quote_diag_streak_ticks: int = Field(
        default=20,
        ge=2,
        alias="ENGINE_NO_QUOTE_DIAG_STREAK_TICKS",
    )
    # Re-emit cadence while the no-quote streak persists past the first
    # warning. Avoids log spam — once stuck, log every N seconds rather
    # than every tick.
    engine_no_quote_diag_relog_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="ENGINE_NO_QUOTE_DIAG_RELOG_SECONDS",
    )
    # v1.4.40 BUG-025: silent-wedge detector. Fires when the engine
    # IS producing valid quotes (mode != 'no_quote') AND eligibility
    # is QUOTE_BOTH AND no behavioural gate is active AND no
    # side_unresolved is set AND ``execution_idle_seconds`` exceeds
    # this threshold. The conjoint condition has no false-positive
    # risk — if all hold and the bot still isn't placing, the bot
    # IS wedged by definition. Distinct from
    # ``engine_no_quote_diag_*`` which catches engine-side stalls;
    # this catches EXECUTOR-side stalls where the engine is fine
    # but downstream dispatch is wedged.
    silent_wedge_detect_enabled: bool = Field(
        default=True,
        alias="SILENT_WEDGE_DETECT_ENABLED",
    )
    silent_wedge_detect_threshold_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        alias="SILENT_WEDGE_DETECT_THRESHOLD_SECONDS",
    )
    # v1.5.159 BUG-031 fix — grace seconds after the engine recovers
    # from an extended ``no_quote`` stretch. The executor's
    # ``last_outbound_attempt_ts_mono`` accumulates idle seconds
    # during no_quote (nothing to place), so right after recovery
    # the wedge detector would see ``idle > threshold`` and false-
    # positive fire. The grace period lets the executor catch up
    # before the wedge check arms again. 10 s default ≫ the 1-2
    # ticks needed for executor to react to engine recovery; small
    # enough that genuine post-recovery wedges still get caught.
    silent_wedge_no_quote_recovery_grace_seconds: float = Field(
        default=10.0,
        ge=0.0,
        le=60.0,
        alias="SILENT_WEDGE_NO_QUOTE_RECOVERY_GRACE_SECONDS",
    )
    # Re-log cadence while a silent wedge persists. Same shape as
    # the engine_no_quote re-log knob.
    silent_wedge_detect_relog_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="SILENT_WEDGE_DETECT_RELOG_SECONDS",
    )
    # v1.4.57 wedge-elimination Phase 3b: when silent_wedge_detected
    # fires, also request an emergency forced reconcile via
    # ``request_open_orders_reconcile``. The reconcile re-fetches
    # exchange ground truth and hydrates / clears local WOs to match
    # — the canonical recovery from the "local says ACKED, exchange
    # says GONE" wedge class observed in snapshots v1.4.52 / v1.4.53.
    # Default True; set False to revert to log-only behaviour.
    silent_wedge_force_reconcile_enabled: bool = Field(
        default=True,
        alias="SILENT_WEDGE_FORCE_RECONCILE_ENABLED",
    )
    # v1.4.50 BUG-025 follow-on: per-tick executor decision trace.
    # When enabled, every call to ``_orchestrate(side, desired, level_idx)``
    # records its decision path (entered branch + exit reason) into:
    #   1) a per-side ring buffer of the last N entries (surfaced in
    #      ``executor_state_snapshot.orchestrate_decision_history``
    #      and inlined into ``executor_silent_wedge_detected`` events);
    #   2) a (side, reason) counter dict (surfaced in
    #      ``executor_state_snapshot.orchestrate_decision_counts``).
    # The buffer + counters are ALWAYS recorded (zero-cost). This flag
    # controls whether each decision ALSO emits an INFO log line. Logs
    # are the only hot-path cost; toggle this OFF after the wedge cause
    # is identified. The flag affects log emission ONLY — the
    # postmortem-visible buffer and counters keep working regardless.
    #
    # Default ON because the 2026-05-18 wedge sequence (three blockages
    # in one day, snapshots v1.4.43 / v1.4.45 / v1.4.47) shows the
    # executor is failing to place valid engine output for >5 minutes
    # at a time without any logged reason. Without per-tick traces we
    # can't pinpoint which silent-return branch is firing.
    executor_decision_trace_enabled: bool = Field(
        default=True,
        alias="EXECUTOR_DECISION_TRACE_ENABLED",
    )
    # v1.4.89 wedge-elimination-cleanup Phase 5C — orchestrate-decision
    # trace mode. Controls when ``_record_orchestrate_decision`` actually
    # writes to the ring buffer:
    #
    #   * "always"        — record every decision (cumulative counters
    #                       + ring buffer + log-if-enabled). v1.4.78
    #                       sampling for high-freq noops still applies.
    #   * "incident_only" — same as "always" EXCEPT high-freq noop actions
    #                       are 100% skipped when no incident is active
    #                       (i.e. risk_exec_state == NORMAL AND no silent-
    #                       wedge fire in the last 5 minutes). Cancels /
    #                       places / amends / errors / non-NORMAL risk
    #                       always trace regardless of mode.
    #   * "off"           — counter still increments, but ring buffer
    #                       and log line are skipped for ALL actions.
    #                       Operator emergency mode for hot-path budget.
    #
    # Default "always" preserves the v1.4.78 behavior. Operator can flip
    # to "incident_only" in prod to reduce ring-buffer pressure during
    # known-healthy sessions. Auto-elevation: when a silent_wedge fires,
    # ``incident_only`` mode behaves as ``always`` for 5 minutes after
    # the fire timestamp — restores full visibility during incidents.
    executor_trace_mode: str = Field(
        default="always",
        alias="EXECUTOR_TRACE_MODE",
        pattern=r"^(always|incident_only|off)$",
    )
    # Ring-buffer size for decision history. Two sides × this many
    # entries. Each entry is a small dict (~10 fields, <300 bytes), so
    # 100 per side = ~60 KB total. Large enough to span a 5-min wedge
    # at the typical 0.5 s quote loop (600 ticks × 2 sides = 1200), but
    # the most-recent N matter for the postmortem.
    executor_decision_trace_buffer_size: int = Field(
        default=200,
        ge=20,
        le=2000,
        alias="EXECUTOR_DECISION_TRACE_BUFFER_SIZE",
    )
    # 1.1.33 added a "Path B" residual-flatten escape that fires when
    # position is BOTH below the dust threshold AND below the venue
    # spec floor on configurations where ``dust_th <= spec_min``. The
    # original motivation was a perceived deadlock from a -1 contract
    # residual that the bot couldn't quote out passively (snapshot
    # 260507114312). On further analysis (snapshot 260507123509) the
    # actual root cause was the ``max_allowed_size`` over-tightening
    # bug in ``_build_side`` self-heal — which 1.1.35 fixes. With the
    # upstream bug fixed, the bot's normal MM cycle can clear sub-spec
    # residuals via overshoot ($10 quote × ~4 contracts = $9.80 order,
    # well above spec) at maker rebate, instead of paying taker fees
    # via Path B's market_close. Default OFF in 1.1.35.
    # Set to ``true`` only on configurations where the upstream bug
    # is suspected to recur or as a defensive belt-and-suspenders.
    # Path A (force-flatten when above ``force_flatten_notional_usd``)
    # remains unchanged and always active.
    residual_flatten_dust_below_spec_enabled: bool = Field(
        default=False,
        alias="RESIDUAL_FLATTEN_DUST_BELOW_SPEC_ENABLED",
    )
    rest_fill_reconcile_interval_ticks: int = Field(
        default=0,
        ge=0,
        alias="REST_FILL_RECONCILE_INTERVAL_TICKS",
    )

    # v1.5.265 fill-rate calibration B. Some venues (notably OKX V5)
    # do NOT publish a per-instrument USD min-notional — they gate
    # on contract-count instead. The bot's per-venue adapters
    # synthesise a conservative USD floor in that case. Historically
    # hardcoded at $5 "to match Binance". For low-priced perps
    # (TON-USDT-SWAP at ~$1.73) that synthetic floor was preventing
    # rung-1 (behind-touch) from ever firing — 68,605 normalize-
    # rejections in a 64-minute snapshot. This knob makes the
    # synthetic floor configurable so the operator can let rung-1
    # fire (lower the floor) or restore tighter rebate-economics
    # (raise the floor) per-profile.
    #
    # Naming: venue-agnostic. Currently consumed by the OKX adapter
    # only; the same hook is appropriate for any venue whose API
    # doesn't expose a USD-denominated min-notional. Venues that
    # DO publish a per-instrument USD floor (Binance, Bluefin via
    # market metadata) ignore this knob and use the venue's value.
    # Default 5.0 preserves legacy behaviour.
    min_venue_notional_usd: float = Field(
        default=5.0,
        gt=0,
        alias="MIN_VENUE_NOTIONAL_USD",
    )

    # v1.5.277 / AQC Phase 1 — Active Quoting Controller.
    # PI controller targeting net edge per minute (rebate + markout).
    # Phase 1: observe-only (controller computes aggression_level but
    # nothing consumes it). Phase 2+: wire to effective multipliers
    # on inventory_skew, inventory_exec_bias, etc. See
    # app/active_quoting_controller.py docstring for the full
    # control-law writeup.
    aqc_enabled: bool = Field(
        default=False, alias="AQC_ENABLED",
    )
    aqc_target_net_edge_per_min_usd: float = Field(
        default=0.020,
        alias="AQC_TARGET_NET_EDGE_PER_MIN_USD",
    )
    aqc_markout_floor_bps: float = Field(
        default=-5.0,
        alias="AQC_MARKOUT_FLOOR_BPS",
    )
    # v1.5.290 — outlier-robust markout floor. The safety brake compares
    # the window MEDIAN markout (not the mean) to AQC_MARKOUT_FLOOR_BPS,
    # and only engages once at least this many distinct fills are in the
    # window. RATIONALE (v1.5.289 incident): a single flash fill
    # (-63.6 bps) dragged the 300 s MEAN to -12.3 < -5 and pinned
    # aggression at 0 for the whole window, though the MEDIAN was -2.8
    # (above the floor). Median + min-fill gate make a lone outlier
    # unable to trip the brake.
    aqc_markout_floor_min_fills: int = Field(
        default=5, ge=1,
        alias="AQC_MARKOUT_FLOOR_MIN_FILLS",
    )
    aqc_pi_kp: float = Field(
        default=0.4, ge=0,
        alias="AQC_PI_KP",
    )
    aqc_pi_ki: float = Field(
        default=0.05, ge=0,
        alias="AQC_PI_KI",
    )
    aqc_window_seconds: float = Field(
        default=300.0, gt=0,
        alias="AQC_WINDOW_SECONDS",
    )
    # AQC Phase 2 (v1.5.281) — wire the controller's aggression_level
    # into the economic min-half-spread floor. Default OFF: when false
    # the controller stays observe-only (Phase 1) and the spread
    # pipeline is byte-identical. When true, ``aggression_level`` in
    # [0,1] lerps the EFFECTIVE base econ floor from its configured
    # value (aggression=0 → unchanged) down toward
    # ``AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS``
    # (aggression=1). Tighten-only: the modulation can never WIDEN the
    # floor above its base, and the per-tick sub-tick / max-rail clamps
    # still apply. See app/active_quoting_controller.py +
    # plans/aqc-execute.md Phase 2.
    aqc_wire_min_half_spread: bool = Field(
        default=False, alias="AQC_WIRE_MIN_HALF_SPREAD",
    )
    aqc_min_half_spread_floor_at_full_aggression_bps: float = Field(
        default=2.0, ge=0,
        alias="AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS",
    )
    # AQC Phase 3 (v1.5.282) — wire the controller's aggression_level
    # into the inventory exec-bias util floor
    # (INVENTORY_EXEC_BIAS_MIN_UTIL_PCT). Default OFF: when false the
    # exec-bias gate uses its configured util floor exactly as today.
    # When true, ``aggression_level`` in [0,1] lerps the EFFECTIVE util
    # floor from its configured value (aggression=0 → unchanged) DOWN
    # toward ``AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT``
    # (aggression=1). Lower floor → exec-bias engages at a lower
    # utilization → the inventory-adding side is suppressed earlier →
    # faster inventory churn (less one-directional accumulation when the
    # controller is being aggressive about chasing fills). Tighten-only:
    # the modulation can only LOWER the floor (a mis-set endpoint above
    # the base is a silent no-op), and the existing
    # ``max(inventory_execution_bias_ratio, floor)`` clamp still applies
    # so the engagement util can never drop below the ratio. See
    # plans/aqc-execute.md Phase 3.
    aqc_wire_inventory: bool = Field(
        default=False, alias="AQC_WIRE_INVENTORY",
    )
    aqc_inventory_util_floor_at_full_aggression_pct: float = Field(
        default=0.10, ge=0.0, le=1.0,
        alias="AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT",
    )
    # AQC Phase 4 (v1.5.283) — wire the controller's aggression_level
    # into the inventory skew coefficient (INVENTORY_SKEW_COEFF_BPS).
    # Default OFF: when false the reservation skew uses its configured
    # coefficient exactly as today. When true, ``aggression_level`` in
    # [0,1] lerps the EFFECTIVE skew coefficient from its configured
    # value (aggression=0 → unchanged) UP toward
    # ``AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS`` (aggression=1). A higher
    # coefficient shifts the reservation price further per unit of
    # inventory → the inventory-REDUCING side quote pulls closer to
    # touch and the adding side pushes away → faster, more constructive
    # inventory flattening when the controller is aggressive. This is
    # the CONSTRUCTIVE inventory lever (safe direction: it churns risk
    # DOWN, never holds it longer). INCREASE-only: the ``max(base,
    # lerped)`` guard makes it impossible to LOWER the coefficient below
    # the operator's configured base (a mis-set endpoint below base is a
    # silent no-op), and the existing ``MAX_RESERVATION_SHIFT_BPS_FROM_
    # MID`` clamp still bounds the total reservation shift so an
    # AQC-raised skew can never push a quote across the book. The trend-
    # skew amplifier still multiplies on top. The plan's companion
    # vol-climbing-widen DISABLE lever is intentionally NOT wired here:
    # disabling a defensive vol gate when aggressive is the less-safe
    # direction and falls under CLAUDE.md Rule 0c (no fix-by-not-
    # trading / loosen-the-defense), so only the constructive skew lever
    # ships. See plans/aqc-execute.md Phase 4.
    aqc_wire_skew: bool = Field(
        default=False, alias="AQC_WIRE_SKEW",
    )
    aqc_skew_coeff_at_full_aggression_bps: float = Field(
        default=40.0, ge=0.0,
        alias="AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS",
    )

    trading_enabled: bool = Field(default=False, alias="TRADING_ENABLED")
    symbol: str = Field(default="ETH", alias="SYMBOL")

    # Max wall-clock wait when no market/private events wake the bot (housekeeping / stale checks).
    # Quote refresh is driven primarily by ``BotState.wake_quote_loop()`` (public BBO + private fills).
    quote_loop_seconds: float = Field(default=1.5, alias="QUOTE_LOOP_SECONDS")
    # Minimum spacing between full REST account/position refreshes when private WS + snapshot are healthy.
    account_rest_min_interval_seconds: float = Field(
        default=8.0,
        gt=0,
        alias="ACCOUNT_REST_MIN_INTERVAL_SECONDS",
    )
    # When order-state uncertainty is active (desync/side convergence), throttle account refresh
    # to avoid contributing to REST 429 cascades from non-account causes.
    order_state_uncertainty_account_rest_interval_seconds: float = Field(
        default=20.0,
        gt=0,
        alias="ORDER_STATE_UNCERTAINTY_ACCOUNT_REST_INTERVAL_SECONDS",
    )
    # When private WS is unhealthy or snapshot health is degraded, use this coarser minimum for
    # account refresh instead of ticking aggressively.
    unhealthy_account_rest_min_interval_seconds: float = Field(
        default=12.0,
        gt=0,
        alias="UNHEALTHY_ACCOUNT_REST_MIN_INTERVAL_SECONDS",
    )

    # 2026-05-16 Codex review #2 fix: paginate the REST fill catch-up
    # past 100 rows. After a private-WS gap or reconnect, OKX returns
    # only the most recent 100 fills per REST call; bursts larger than
    # that were silently truncated, causing session counters / markout
    # attribution / trade-rate accounting to undercount. The fix walks
    # the ``/trade/fills`` endpoint via its ``after=<billId>`` cursor
    # up to this many pages (100 rows per page). Default 10 pages =
    # 1000 fills which covers typical gaps with margin. Raise for very-
    # high-activity accounts that may see >1000-fill gaps.
    okx_fills_rest_max_pages: int = Field(
        default=10,
        ge=1,
        le=100,
        alias="OKX_FILLS_REST_MAX_PAGES",
    )
    # Reference venue selector for the cross-venue fair-value feed. One of:
    # ``binance`` (default, back-compat), ``bybit``, or ``off`` (disable the
    # entire cross-venue path). Chosen at startup — the selected venue's
    # public-WS client writes to ``state.binance_*`` fields (kept under that
    # name for back-compat even when Bybit is the source; see
    # ``tmp/snap_20260419_152651`` for the first session that motivated the
    # alternative: Binance ref latency p50 44 ms vs Bybit p50 3 ms from
    # the Singapore region (Railway era 2026-04). When ``off`` neither
    # stream starts;
    # ``BINANCE_WS_ENABLED=false`` is equivalent. The per-venue
    # ``*_CANCEL_ON_MOVE_BPS`` and ``*_WS_FAIR_VALUE_MAX_AGE_SECONDS``
    # settings below are venue-independent in effect (they govern the
    # shared cross-venue-cancel logic) but kept under the ``binance_*`` name
    # for config continuity — setting them applies to whichever stream is
    # active.
    reference_exchange: str = Field(
        default="binance",
        alias="REFERENCE_EXCHANGE",
    )
    # Binance public-WS cross-venue reference (Level 1: cancel-on-move).
    # When enabled, ``app/exchange/binance_public_ws.py`` subscribes to
    # ``<base_url>/<symbol>@bookTicker`` and maintains ``state.binance_mid``
    # + an EWMA of ``(grvt_mid - binance_mid)`` basis. Execution
    # (``maybe_refresh_quotes``) compares each resting order's price to
    # ``binance_mid + basis_ewma`` and cancels any order whose distance
    # exceeds ``BINANCE_CANCEL_ON_MOVE_BPS`` — gives us a low-latency
    # "arb bot just moved Binance; GRVT is about to follow" signal
    # without waiting on GRVT's own BBO update.
    binance_ws_enabled: bool = Field(
        default=True,
        alias="BINANCE_WS_ENABLED",
    )
    # The Binance SYMBOL (e.g. ETHUSDT) may differ from the GRVT SYMBOL
    # (e.g. ETH_USDT_Perp). We default to ETHUSDT; operator overrides
    # when trading a different base asset.
    binance_symbol: str = Field(
        default="ETHUSDT",
        alias="BINANCE_SYMBOL",
    )
    # ``fstream`` = Binance Futures USDT-margined; closer to GRVT perp
    # than spot (smaller funding-premium basis). Spot base would be
    # ``wss://stream.binance.com:9443/ws``.
    binance_ws_base_url: str = Field(
        default="wss://fstream.binance.com/ws",
        alias="BINANCE_WS_BASE_URL",
    )
    # Threshold (bps) — cancel a resting order when
    # ``|order.price - (binance_mid + basis_ewma)| / fair_value * 10_000
    # > effective_threshold``. v1.4.45: the effective threshold is
    # ``max(BINANCE_CANCEL_ON_MOVE_BPS, target_half_spread_bps +
    # BINANCE_CANCEL_ON_MOVE_BUFFER_BPS)``. The static value below is the
    # FLOOR — minimum distance the bot will let an order rest, regardless
    # of half-spread.
    binance_cancel_on_move_bps: float = Field(
        default=5.0,
        gt=0,
        alias="BINANCE_CANCEL_ON_MOVE_BPS",
    )
    # v1.4.45 — relative-threshold companion knob. When > 0, the
    # cross-venue cancel threshold becomes the LARGER of
    # ``BINANCE_CANCEL_ON_MOVE_BPS`` (static floor) and
    # ``target_half_spread_bps + BINANCE_CANCEL_ON_MOVE_BUFFER_BPS``
    # (dynamic ceiling). This guarantees the bot never quotes a price
    # the same code path would immediately cancel: the cross-venue
    # threshold automatically tracks the engine's current half-spread
    # plus a fixed buffer. Set to 0 to disable (pre-v1.4.45 behaviour;
    # absolute threshold only). 5 bps default — gives any quote
    # roughly 5 bps of room above its natural distance before Binance-
    # led drift triggers a cancel.
    #
    # Background: pre-v1.4.45 the threshold was static-only. When the
    # bot's organic half-spread grew past the static threshold (vol
    # regime up + toxicity bump active), every placed order was born
    # already past the threshold → instant cancel → no fills. Observed
    # 2026-05-18 in snapshot ``v1.4.44-260518-140038``: half-spread
    # 13.7 bps with threshold 10 bps → 30+ minutes of zero trading.
    # v1.4.45 makes the invariant ``threshold > half_spread + buffer``
    # auto-enforced by construction.
    binance_cancel_on_move_buffer_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="BINANCE_CANCEL_ON_MOVE_BUFFER_BPS",
    )
    # If the Binance stream goes silent longer than this, disregard it
    # for cancel decisions (fall back to GRVT-only reprice behaviour).
    # Covers: WS flaps, Binance maintenance, network partitions.
    binance_ws_fair_value_max_age_seconds: float = Field(
        default=15.0,
        gt=0,
        alias="BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS",
    )
    # EWMA smoothing coefficient for the GRVT-vs-Binance basis. At 1
    # update/sec, alpha=0.05 gives ~14-sample (~14 s) effective window
    # — slow enough to ignore single-tick jitter, fast enough to track
    # funding-premium drift over minutes.
    binance_basis_ewma_alpha: float = Field(
        default=0.05,
        gt=0,
        le=1.0,
        alias="BINANCE_BASIS_EWMA_ALPHA",
    )
    # Bybit public-WS cross-venue reference — alternative to Binance.
    # Selected via ``REFERENCE_EXCHANGE=bybit``. Bybit Linear perps run
    # out of co-located Singapore infrastructure: measured p50 3 ms /
    # p95 4 ms one-way latency from Singapore (Railway era; vs Binance
    # Tokyo p50 44 ms / p95 79 ms; see
    # ``C:/Work/DTC/xch-md-measure/tmp/logs.singapore.log``). The
    # ~40 ms edge is meaningful for cutting adverse selection on
    # GRVT's tight ETH book, where latency-advantaged takers pick off
    # stale quotes before we can cancel.
    bybit_symbol: str = Field(
        default="ETHUSDT",
        alias="BYBIT_SYMBOL",
    )
    # ``stream.bybit.com/v5/public/linear`` = USDT-margined perps.
    # Spot stream is ``stream.bybit.com/v5/public/spot``; inverse
    # perps ``.../v5/public/inverse``. Linear is the direct analogue
    # to GRVT's perp.
    bybit_ws_base_url: str = Field(
        default="wss://stream.bybit.com/v5/public/linear",
        alias="BYBIT_WS_BASE_URL",
    )
    # Microprice-based reservation. When ``True`` (default), the strategy
    # computes a queue-imbalance-weighted reference price from top-of-book
    # depth — ``microprice = (bid_size*best_ask + ask_size*best_bid) /
    # (bid_size+ask_size)`` — and uses it as the reservation centerpoint
    # instead of the midpoint. Heavy bid stacks shift the reservation up
    # (price likely to move up next); heavy ask stacks shift it down.
    # Intent: reduce adverse selection on thin-book venues where the
    # midpoint lags fair value. When depth is unavailable (None or zero
    # sizes) the reservation transparently falls back to midprice — a
    # compute_quote_decision call with only ``mid`` still works
    # identically to pre-refactor behaviour.
    microprice_reservation_enabled: bool = Field(
        default=True,
        alias="MICROPRICE_RESERVATION_ENABLED",
    )
    reprice_threshold_bps: float = Field(default=5.0, alias="REPRICE_THRESHOLD_BPS")
    # Volatility-adaptive reprice threshold: when > 0, effective threshold =
    # clip(vol_multiplier * short_vol_bps, REPRICE_THRESHOLD_BPS_MIN, REPRICE_THRESHOLD_BPS).
    # In quiet markets (short_vol_bps → 0) we reprice on tiny moves (tighten queue position);
    # in volatile markets we reprice only on larger moves (avoid cancel/replace churn).
    # Default 0.0 disables — fixed threshold (pre-existing behaviour). Recommend 3.0 for GRVT.
    reprice_threshold_vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        alias="REPRICE_THRESHOLD_VOL_MULTIPLIER",
    )
    reprice_threshold_bps_min: float = Field(
        default=0.5,
        gt=0.0,
        alias="REPRICE_THRESHOLD_BPS_MIN",
    )
    # Aging / distance-to-touch: tighten passive quotes toward BBO without pointless same-price refresh.
    quote_aging_enabled: bool = Field(default=True, alias="QUOTE_AGING_ENABLED")
    quote_aging_max_age_seconds: float = Field(
        default=6.0,
        gt=0,
        alias="QUOTE_AGING_MAX_AGE_SECONDS",
    )
    quote_max_distance_to_touch_ticks: float = Field(
        default=3.0,
        gt=0,
        alias="QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS",
    )
    # Normal two-sided competitive MM: anchor half-spread to live BBO; max distance to touch is
    # enforced inside compute_normal_mm_market_capped_half_spread_bps (not in finalize).
    normal_mm_use_market_spread_anchor: bool = Field(
        default=True,
        alias="NORMAL_MM_USE_MARKET_SPREAD_ANCHOR",
    )
    normal_mm_touch_buffer_ticks: float = Field(
        default=1.0,
        ge=0.0,
        alias="NORMAL_MM_TOUCH_BUFFER_TICKS",
    )
    normal_mm_max_distance_to_touch_ticks: float = Field(
        default=8.0,
        gt=0.0,
        alias="NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS",
    )

    normal_mm_min_competitive_half_spread_bps: float = Field(
        default=0.5,
        ge=0.0,
        alias="NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS",
    )
    # 0 = no extra ceiling on the market-derived max half-spread (before min vs model).
    normal_mm_max_competitive_half_spread_bps: float = Field(
        default=0.0,
        ge=0.0,
        alias="NORMAL_MM_MAX_COMPETITIVE_HALF_SPREAD_BPS",
    )
    quote_aging_tighten_ticks: float = Field(
        default=1.0,
        gt=0,
        alias="QUOTE_AGING_TIGHTEN_TICKS",
    )
    quote_inventory_pressure_pct: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        alias="QUOTE_INVENTORY_PRESSURE_PCT",
    )
    # When |position|/max_abs_position >= QUOTE_INVENTORY_PRESSURE_PCT,
    # push the *adding-side* quote this many ticks BEHIND the touch
    # instead of joining the touch. The adding side is the one that
    # would increase |position| if filled — bid when long, ask when
    # short. Slows accumulation against the wrong side of skew and
    # narrows the BUY-vs-SELL markout asymmetry.
    #
    # Default 0 (off). Worth enabling only when the bot routinely
    # presses position cap (e.g. TON 1.1.29 sat at >=70% of cap for
    # 67% of session). On regimes where the bot cycles inventory well
    # below cap (e.g. SUI 2026-05-09 weekend: max 75%, mean 38%), this
    # buffer mostly kills fill rate without measurable markout gain.
    inventory_high_adding_side_buffer_ticks: float = Field(
        default=0.0,
        ge=0.0,
        alias="INVENTORY_HIGH_ADDING_SIDE_BUFFER_TICKS",
    )
    # Markout-based quote aging — pre-cancel a resting quote whose
    # price has drifted adverse to current mid for at least the
    # configured duration. "Adverse" means: bid above mid (someone
    # selling to us at our bid would have free money) or ask below
    # mid (someone buying from us at our ask would). Both conditions
    # are exactly the toxic-fill setup; the markout-aging gate yanks
    # the quote before the taker arrives.
    #
    # Threshold + duration are deliberately separate: threshold
    # filters noise (single-tick mid wobble shouldn't fire), duration
    # confirms the drift is sustained. Default 0/0 = disabled.
    # Calibration target: set after observing post-colo markout
    # distribution (latency and toxicity profile both shift after the
    # the partner colo move, so calibrating today gives stale thresholds).
    quote_aging_markout_adverse_bps_threshold: float = Field(
        default=0.0,
        ge=0.0,
        alias="QUOTE_AGING_MARKOUT_ADVERSE_BPS_THRESHOLD",
    )
    quote_aging_markout_adverse_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        alias="QUOTE_AGING_MARKOUT_ADVERSE_DURATION_SECONDS",
    )
    # Effective util threshold for touch-relaxing fair cap/floor in quote aging is
    # max(QUOTE_INVENTORY_PRESSURE_PCT, this). Set 0 to use only QUOTE_INVENTORY_PRESSURE_PCT.
    # Calms hyper-aggressive pull-to-touch for small/moderate inventory.
    inventory_fair_touch_relax_min_util_pct: float = Field(
        default=0.52,
        ge=0.0,
        lt=1.0,
        alias="INVENTORY_FAIR_TOUCH_RELAX_MIN_UTIL_PCT",
    )
    # When |position| / max_abs_position >= this ratio, execution prefers the inventory-reducing side first
    # and deprioritizes the adding side until the reducer is viable (0 = disable).
    inventory_execution_bias_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        alias="INVENTORY_EXEC_BIAS_RATIO",
    )
    # Side-priority bias uses max(INVENTORY_EXEC_BIAS_RATIO, this) as the util gate (0 = no extra floor).
    # Prevents panic-style deprioritization at tiny positions (e.g. 2% of max).
    inventory_execution_bias_min_util_pct: float = Field(
        default=0.12,
        ge=0.0,
        lt=1.0,
        alias="INVENTORY_EXEC_BIAS_MIN_UTIL_PCT",
    )
    # Multiply reprice threshold on the non-preferred side when preferred side is not yet viable.
    inventory_execution_bias_nonpreferred_reprice_mult: float = Field(
        default=3.0,
        gt=0,
        alias="INVENTORY_EXEC_BIAS_NONPREFERRED_REPRICE_MULT",
    )
    # Quote bots need a fresh book; tuned for ~1.5s quote loop and small capital.
    stale_data_warn_seconds: float = Field(default=2.0, alias="STALE_DATA_WARN_SECONDS")
    stale_data_kill_seconds: float = Field(default=8.0, alias="STALE_DATA_KILL_SECONDS")
    # Stale book / public WS: bounded reconnect + wait for fresh BBO before permanent KILL.
    #
    # Default MAX_DURATION raised from 15 → 600 s (10 min) after the overnight
    # 2026-04-19 session was killed by a ~38 s GRVT public-feed outage. With
    # the old 15 s window, 5 burst attempts at ~3 s apiece exhausted the
    # budget and escalated to ``stale_data_kill_escalated`` just 1 s before
    # the feed came back. At 10 min we cover the realistic vendor-maintenance
    # envelope while still killing on truly unrecoverable outages.
    market_data_recovery_enabled: bool = Field(default=True, alias="MARKET_DATA_RECOVERY_ENABLED")
    market_data_recovery_max_attempts: int = Field(
        default=3,
        ge=1,
        alias="MARKET_DATA_RECOVERY_MAX_ATTEMPTS",
    )
    # Delay BETWEEN ATTEMPTS WITHIN A SINGLE BURST (legacy name; unchanged).
    market_data_recovery_backoff_seconds: float = Field(
        default=1.0,
        ge=0,
        alias="MARKET_DATA_RECOVERY_BACKOFF_SECONDS",
    )
    market_data_recovery_max_duration_seconds: float = Field(
        default=600.0,
        gt=0,
        alias="MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS",
    )
    # Exponential backoff BETWEEN BURSTS (NEW — separate from the per-attempt
    # delay above). Previously bursts ran back-to-back on every tick
    # (~4/second), hammering the venue during outages. Now each failed burst
    # doubles the wait until the next one, capped. First burst on stale
    # detection runs immediately (no delay).
    market_data_recovery_burst_backoff_initial_seconds: float = Field(
        default=3.0,
        ge=0,
        alias="MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS",
    )
    market_data_recovery_burst_backoff_max_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="MARKET_DATA_RECOVERY_BURST_BACKOFF_MAX_SECONDS",
    )
    market_data_recovery_burst_backoff_multiplier: float = Field(
        default=2.0,
        gt=1.0,
        alias="MARKET_DATA_RECOVERY_BURST_BACKOFF_MULTIPLIER",
    )
    # Instrumentation: alert when bid/ask/mid (+ exchange ts) unchanged across this many OK refreshes.
    market_data_stall_unchanged_threshold: int = Field(
        default=15,
        ge=2,
        alias="MARKET_DATA_STALL_UNCHANGED_THRESHOLD",
    )
    # Min seconds between INFO-level market_data_refresh_success logs (0 = never INFO, use DEBUG only).
    market_data_success_log_interval_seconds: float = Field(
        default=30.0,
        ge=0,
        alias="MARKET_DATA_SUCCESS_LOG_INTERVAL_SECONDS",
    )
    # On BBO fingerprint stall latch, request a public websocket reconnect (no REST book fallback).
    market_data_reset_transport_on_stall: bool = Field(
        default=False,
        alias="MARKET_DATA_RESET_TRANSPORT_ON_STALL",
    )
    # In-memory ring for gap quantiles (median/p95); not full history.
    market_data_gap_ring_buffer_samples: int = Field(
        default=4096,
        ge=64,
        le=65536,
        alias="MARKET_DATA_GAP_RING_BUFFER_SAMPLES",
    )
    market_data_gap_persist_samples: bool = Field(
        default=False,
        alias="MARKET_DATA_GAP_PERSIST_SAMPLES",
    )
    market_data_gap_persist_max_rows: int = Field(
        default=5000,
        ge=100,
        le=500_000,
        alias="MARKET_DATA_GAP_PERSIST_MAX_ROWS",
    )
    # 0 = disabled. Throttled by market_data_gap_large_log_interval_seconds.
    market_data_gap_large_log_threshold_ms: float = Field(
        default=0.0,
        ge=0,
        alias="MARKET_DATA_GAP_LARGE_LOG_THRESHOLD_MS",
    )
    market_data_gap_large_log_interval_seconds: float = Field(
        default=60.0,
        ge=0,
        alias="MARKET_DATA_GAP_LARGE_LOG_INTERVAL_SECONDS",
    )

    # Public WS market data timing (in-memory, rolling window; safe for production).
    market_data_timing_window_enabled: bool = Field(
        default=True,
        alias="MARKET_DATA_TIMING_WINDOW_ENABLED",
    )
    # One sample per applied public WS update; sized for ~60s of data at high rates.
    market_data_timing_max_samples: int = Field(
        default=10_000,
        ge=256,
        le=200_000,
        alias="MARKET_DATA_TIMING_MAX_SAMPLES",
    )
    market_data_timing_raw_endpoint_max_limit: int = Field(
        default=1000,
        ge=50,
        le=20_000,
        alias="MARKET_DATA_TIMING_RAW_ENDPOINT_MAX_LIMIT",
    )

    # --- Quote eligibility (pre-QuoteEngine): freshness + short-horizon drift/jump guards ---
    quote_eligibility_enabled: bool = Field(default=True, alias="QUOTE_ELIGIBILITY_ENABLED")
    # Local receipt age: seconds since last public BBO apply (monotonic), converted to ms.
    quote_hold_max_book_age_ms: float = Field(
        default=500.0,
        gt=0,
        alias="QUOTE_HOLD_MAX_BOOK_AGE_MS",
    )
    quote_one_sided_max_book_age_ms: float = Field(
        default=250.0,
        gt=0,
        alias="QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS",
    )
    # When book age is between one-sided and hold, or gap p95 exceeds this, use one-sided mode.
    quote_one_sided_max_gap_p95_ms: float = Field(
        default=350.0,
        ge=0,
        alias="QUOTE_ONE_SIDED_MAX_GAP_P95_MS",
    )
    quote_hold_max_gap_p95_ms: float = Field(
        default=600.0,
        ge=0,
        alias="QUOTE_HOLD_MAX_GAP_P95_MS",
    )
    # Live-book-fresh override for the p95 gate. When the current book is clearly fresh
    # (``age_ms < max_book_age_ms``) AND the median gap is healthy (``gap_median_ms <
    # max_gap_median_ms``), the p95 hold is overridden and quoting proceeds. This
    # defends against a single outlier gap (e.g., 1.9 s WS silence) pinning the rolling
    # p95 above the threshold and freezing the bot for the rest of the ring's lifetime
    # (observed in ``tmp/snap_20260417_193813``: p95=1094 ms, bot held HOLD_ALL for
    # 49 s while book_age stayed <1 ms). The override does NOT apply to drift/jump/age
    # holds — only to the p95-only case, where the current signal is demonstrably live.
    quote_freshness_live_override_enabled: bool = Field(
        default=True,
        alias="QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED",
    )
    quote_freshness_live_override_max_book_age_ms: float = Field(
        default=150.0,
        ge=0.0,
        alias="QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS",
    )
    quote_freshness_live_override_max_gap_median_ms: float = Field(
        default=500.0,
        ge=0.0,
        alias="QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS",
    )
    # Fallback "feed is alive NOW" signal for the override. The ring-buffer median
    # can be polluted by cold-start / reconnect outliers and take minutes to
    # recover. ``last_gap_ms`` reflects the MOST RECENT inter-update gap —
    # independent of historical pollution. If either the median OR the last-gap
    # path satisfies its threshold, the override fires. Observed in
    # ``tmp/snap_20260418_132744``: 66 s after a deploy-restart, gap_median
    # was 748 ms (polluted by a 3.9 s startup outlier) but last_gap = 416 ms —
    # feed was clearly alive, yet quoting was held for 3-5 min until the ring
    # rotated the outlier out.
    quote_freshness_live_override_max_last_gap_ms: float = Field(
        default=500.0,
        ge=0.0,
        alias="QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS",
    )
    # Minimum gap-sample count before the p95-hold gate is honored. Below this
    # threshold, p95 is statistically unreliable (e.g. a single 4 s outlier in
    # 74 samples produces p95 = 1.8 s). When sample count is below this, we
    # skip the p95 portion of freshness gating entirely — treat as "warming up".
    # Age-based freshness gating is NOT affected; that remains active because
    # book age is a current-state measurement, not a rolling statistic.
    quote_freshness_p95_min_gap_samples: int = Field(
        default=200,
        ge=0,
        le=10000,
        alias="QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES",
    )
    # Mid-price ring buffer (monotonic timestamps) for drift/jump; maxlen and max lookback.
    quote_mid_history_max_samples: int = Field(
        default=256,
        ge=32,
        le=4096,
        alias="QUOTE_MID_HISTORY_MAX_SAMPLES",
    )
    quote_mid_history_max_age_ms: float = Field(
        default=2000.0,
        gt=0,
        alias="QUOTE_MID_HISTORY_MAX_AGE_MS",
    )

    # --- Long-window drift gate (multi-minute trend detection) ---
    # See BUGS/bug-002.md for the failure case that motivated this:
    # a -100 bps slow-grind directional move on SUI over 4.8 h
    # accumulated $0.71 of inventory-carry loss outside the 5 s
    # markout window. The existing DRIFT_BLOCK_*ms gates measure
    # sub-second jumps and don't fire on slow grinds; this gate
    # complements them on the multi-minute horizon.
    #
    # Behavior: if the 5-min mid drift exceeds the threshold,
    # eligibility caps to QUOTE_SELL_ONLY (when falling) or
    # QUOTE_BUY_ONLY (when rising) — i.e. don't accumulate
    # inventory in the trend direction. Same risk-control intent
    # as the toxicity engine, but on a much longer horizon.
    #
    # Set ``DRIFT_BLOCK_LONG_WINDOW_BPS=0`` to disable entirely
    # (default value retains classic pre-fix behaviour). Setting
    # any positive value activates the gate.
    # v1.5.35 bugfix — retention bumped to cover the 15-minute window
    # used by ``mid_drift_windows.compute_mid_drift_windows``. The
    # pre-bump default (300s) was tuned for the slow-grind gate
    # (BUGS/bug-002.md) which uses a 5-minute window; the Phase 2B
    # (v1.5.26) mid_drift_windows feature added 5-min AND 15-min
    # horizons but didn't raise retention. Snapshot v1.5.28-260522-
    # 223401 showed all 30s+ drifts returning None because the
    # ring-buffer (maxlen=512, default) was being EVICTED faster
    # than expected at production tick rates: the quote loop is
    # wake-event-driven (every BBO update wakes it; 47967 BBO events
    # in 6862s ≈ 7 Hz average, bursting to 20+ Hz during
    # shock-gate churn), so 512 samples = 25-72s of coverage.
    # Bumped to 1000s retention so the 15-min window is fully
    # represented.
    drift_long_window_seconds: float = Field(
        default=1000.0, gt=0, alias="DRIFT_LONG_WINDOW_SECONDS"
    )
    drift_block_long_window_bps: float = Field(
        default=50.0, ge=0, alias="DRIFT_BLOCK_LONG_WINDOW_BPS"
    )
    # v1.5.35 bugfix — maxlen sized for HIGH burst tick rates. At 20 Hz
    # (worst observed in shock-gate churn), 1000s retention needs
    # 20000 samples. Memory cost: ~640 KB per deque, negligible vs
    # the bot's ~50 MB resident set. Allow up to 65536 via the
    # operator-tunable upper bound.
    drift_long_window_max_samples: int = Field(
        default=20000, ge=32, le=65536,
        alias="DRIFT_LONG_WINDOW_MAX_SAMPLES",
    )
    drift_long_window_min_samples: int = Field(
        default=30, ge=2, le=1000,
        alias="DRIFT_LONG_WINDOW_MIN_SAMPLES",
    )
    # Fraction of in-window samples used as the smoothed anchor at
    # each end. Median of the first N samples is the "5-min-ago"
    # reference; median of the last N samples is the "now" reference.
    # Robust to single-sample aberrations (a transient spike 5 min ago
    # contributes only 1/N to the median).
    #
    # Default 0.2 = 20% on each end; with 30 samples in a 5-min
    # window that's 6 samples per anchor (~18 s of smoothing). Higher
    # values give more smoothing at the cost of detection latency.
    # 0.0 falls back to single-sample anchoring (oldest vs ``mid_now``).
    drift_long_window_anchor_fraction: float = Field(
        default=0.2, ge=0.0, le=0.5,
        alias="DRIFT_LONG_WINDOW_ANCHOR_FRACTION",
    )
    # Short-horizon return thresholds (bps, signed). Exceeded => block the vulnerable side or hold.
    #
    # Each threshold has two forms: an absolute bps floor AND a vol-multiplier. The
    # effective threshold is ``max(abs_bps, multiplier × short_vol_bps)``. When the
    # multiplier is 0 (default), only the absolute floor applies — preserves the
    # pre-fix behaviour. When the multiplier is > 0, the threshold adapts to
    # observed volatility: tight in quiet markets, relaxed in active ones.
    #
    # Rationale: the absolute thresholds were calibrated for HL-ish books where
    # ``short_vol_bps`` runs 2–5. On GRVT ETH the observed ``short_vol_bps ≈ 0.5``,
    # making the absolute 12 bps threshold effectively 24× vol — never fires on
    # actual informed moves. Vol-scaled with multiplier 3 gives ~1.5 bps (3× vol)
    # at quiet times and ~15 bps (3× vol) at active times. Directional moves of
    # 3+ sigma become the drift-filter trigger — much closer to what adverse
    # selection actually looks like in the markout data.
    drift_block_100ms_bps: float = Field(default=12.0, ge=0, alias="DRIFT_BLOCK_100MS_BPS")
    drift_block_100ms_vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        alias="DRIFT_BLOCK_100MS_VOL_MULTIPLIER",
    )
    drift_block_250ms_bps: float = Field(default=22.0, ge=0, alias="DRIFT_BLOCK_250MS_BPS")
    drift_block_250ms_vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        alias="DRIFT_BLOCK_250MS_VOL_MULTIPLIER",
    )
    drift_hold_500ms_bps: float = Field(default=40.0, ge=0, alias="DRIFT_HOLD_500MS_BPS")
    drift_hold_500ms_vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        alias="DRIFT_HOLD_500MS_VOL_MULTIPLIER",
    )
    jump_hold_250ms_bps: float = Field(default=75.0, ge=0, alias="JUMP_HOLD_250MS_BPS")
    jump_hold_250ms_vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        alias="JUMP_HOLD_250MS_VOL_MULTIPLIER",
    )
    # After HOLD_ALL or one-sided episode, keep elevated conservatism briefly. Defaults
    # sized so cooldowns do not dominate the quote loop period (see snap_20260416_154159
    # forensic: 2000ms hold + 800ms one-sided vs a 500ms loop blocked 4+ cycles per event).
    quote_hold_cooldown_ms: float = Field(default=1000.0, ge=0, alias="QUOTE_HOLD_COOLDOWN_MS")
    quote_one_sided_cooldown_ms: float = Field(
        default=400.0,
        ge=0,
        alias="QUOTE_ONE_SIDED_COOLDOWN_MS",
    )
    # When only freshness tier-2 triggers (no drift), which single side to allow.
    quote_freshness_one_sided_preference: str = Field(
        default="BUY",
        alias="QUOTE_FRESHNESS_ONE_SIDED_PREFERENCE",
    )
    # 2026-05-12 codex-#5: when book freshness degrades into one-sided
    # mode AND we're carrying a position, prefer the inventory-
    # reducing side (asks when long, bids when short) over the static
    # ``QUOTE_FRESHNESS_ONE_SIDED_PREFERENCE`` setting. The reducing
    # side closes exposure, which is the safer choice when our book
    # view is stale. When flat (position ≈ 0) the static config
    # preference still applies. Default true.
    quote_freshness_one_sided_prefer_reducing: bool = Field(
        default=True,
        alias="QUOTE_FRESHNESS_ONE_SIDED_PREFER_REDUCING",
    )
    # Opt-in legacy gating: fold ``effective_staleness_ms`` (wall_now - exchange_ts, which
    # embeds one-way network delay) into the freshness threshold comparison via max(age,
    # eff_stal). Off by default — see quote_eligibility._freshness_eligibility docstring.
    quote_freshness_use_exchange_staleness_for_gating: bool = Field(
        default=False,
        alias="QUOTE_FRESHNESS_USE_EXCHANGE_STALENESS_FOR_GATING",
    )

    # Desync: consecutive ticks with mismatch before KILL; clean ticks required after recovery.
    desync_unrecoverable_after_ticks: int = Field(
        default=48,
        ge=1,
        alias="DESYNC_UNRECOVERABLE_AFTER_TICKS",
    )
    desync_quarantine_ticks: int = Field(
        default=1,
        ge=1,
        alias="DESYNC_QUARANTINE_TICKS",
    )

    # Fill-time book reference: max |snapshot_ts - fill_ts| to accept (else missing_reference).
    book_reference_max_skew_ms: float = Field(
        default=250.0,
        ge=0,
        alias="BOOK_REFERENCE_MAX_SKEW_MS",
    )

    quote_notional_usd: float = Field(default=25.0, alias="QUOTE_NOTIONAL_USD")
    max_abs_position: float = Field(default=0.05, alias="MAX_ABS_POSITION")
    # Caps sized for ~low-hundreds USD equity; raise only after sizing risk consciously.
    max_position_notional_usd: float = Field(default=250.0, alias="MAX_POSITION_NOTIONAL_USD")
    max_order_notional_usd: float = Field(default=75.0, alias="MAX_ORDER_NOTIONAL_USD")
    # Last-resort safety multiplier on top of MAX_ORDER_NOTIONAL_USD.
    # Adapter-level pre-flight check: if the order's notional would
    # exceed ``max_order_notional_usd × max_order_notional_hard_multiplier``,
    # the adapter refuses to send it, raising an exception that the
    # caller surfaces as a place failure. Designed to catch upstream
    # sizing bugs / unit confusion / corrupted state -- the kind that
    # produced 2000+ SUI fills against a $20 configured cap on
    # 2026-05-06.
    #
    # 2x default leaves headroom for legitimate-but-edge cases (engine
    # boost on inventory clear-out, slippage padding) while still
    # blocking 100x runaway sizes. Set to ``1.0`` to make
    # MAX_ORDER_NOTIONAL_USD an absolute hard cap. Set higher (e.g.
    # 5.0) only if you're intentionally running a strategy that
    # legitimately needs large boost factors -- but then reconsider
    # whether MAX_ORDER_NOTIONAL_USD itself is set too low.
    max_order_notional_hard_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        alias="MAX_ORDER_NOTIONAL_HARD_MULTIPLIER",
    )
    # Optional operator-declared collateral (USD) for startup logs when exchange snapshot is missing.
    live_trading_collateral_usd: Optional[float] = Field(
        default=None,
        ge=0,
        alias="LIVE_TRADING_COLLATERAL_USD",
    )
    max_open_orders: int = Field(default=2, alias="MAX_OPEN_ORDERS")

    # Position / quote notional floors (USD). Dust is ignored for emergency residual flatten.
    dust_position_notional_usd: float = Field(
        default=12.0,
        ge=0.0,
        alias="DUST_POSITION_NOTIONAL_USD",
    )
    min_quote_notional_usd: float = Field(
        default=12.0,
        gt=0.0,
        alias="MIN_QUOTE_NOTIONAL_USD",
    )
    force_flatten_notional_usd: float = Field(
        default=25.0,
        ge=0.0,
        alias="FORCE_FLATTEN_NOTIONAL_USD",
    )

    base_half_spread_bps: float = Field(default=8.0, alias="BASE_HALF_SPREAD_BPS")
    min_half_spread_bps: float = Field(default=4.0, alias="MIN_HALF_SPREAD_BPS")
    max_half_spread_bps: float = Field(default=80.0, alias="MAX_HALF_SPREAD_BPS")
    # v1.5.230 — cap on the absolute reservation shift from mid.
    # Closes the cap×shift interaction bug discovered in the
    # v1.5.229 Phase 5-prime snapshot: when the alpha stack
    # (trend_drift + ob_imbalance + flow_score + ofi + basis +
    # inventory_skew) sums to a shift larger than half_spread, the
    # bid lands above best_ask (or ask below best_bid), the
    # post-only order is rejected by the venue, and that side
    # silently stops filling. The MAX_HALF_SPREAD_BPS cap can't fix
    # this — it only caps the spread, not the shift.
    #
    # This knob clamps ``(reservation - mid)`` to ``±value`` bps
    # AFTER all alpha shifts are summed, BEFORE bid/ask prices are
    # computed. Default 0.0 = no clamp (legacy behaviour preserved).
    # Recommended value: half the MAX_HALF_SPREAD_BPS, so the spread
    # always has enough room to keep both quotes on the correct
    # side of the touch even at the worst-case shift.
    max_reservation_shift_bps_from_mid: float = Field(
        default=0.0,
        ge=0.0,
        alias="MAX_RESERVATION_SHIFT_BPS_FROM_MID",
    )
    # v1.5.156 Option A — vol-adaptive base half-spread.
    #
    # The driving observation (overnight ~9h session screenshot,
    # 2026-05-26): in calm regimes (vol < ~6 bp/s on TON) the bot's
    # quoted spread averages 15–26 bps total while the natural touch
    # is ~5 bps. Tighter makers join the inside, bot sits behind, no
    # fills for 56-minute stretches. Pure participation tax — no
    # losses, no rebate income either.
    #
    # The base half-spread is the dominant term in calm regimes
    # (vol_contribution is small when vol_bps is small, so the +4 bp
    # base is the floor). Drop the base specifically in low-vol
    # regimes to let the bot quote closer to the touch.
    #
    # When ``BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=true`` AND
    # ``vol_bps < BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S``:
    #   use ``BASE_HALF_SPREAD_BPS_LOW_VOL`` as the base
    # Otherwise:
    #   use ``BASE_HALF_SPREAD_BPS`` as the base (legacy)
    #
    # The existing ``min_half_spread_bps`` + economic floor still
    # clamp the result, so the low-vol base cannot push the bot
    # below the venue-spec / economic-edge thresholds.
    base_half_spread_vol_adaptive_enabled: bool = Field(
        default=False,
        alias="BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED",
    )

    # Phase 8A (v1.5.185) — Avellaneda-Stoikov-derived adaptive
    # base half-spread. When ``AVELLANEDA_STOIKOV_ENABLED=true``,
    # ``compute_quote_decision`` replaces the constant
    # ``BASE_HALF_SPREAD_BPS`` (or its low-vol variant) with a
    # closed-form value driven by current vol_bps + recent fill
    # rate (k_intensity_per_min). Default-off → no behaviour
    # change for back-compat. The formula and operator dials are
    # documented in ``app/avellaneda_stoikov.py``.
    avellaneda_stoikov_enabled: bool = Field(
        default=False,
        alias="AVELLANEDA_STOIKOV_ENABLED",
    )
    as_gamma_inv: float = Field(
        default=0.05, ge=0.0,
        alias="AS_GAMMA_INV",
    )
    as_gamma_edge: float = Field(
        default=1.0, ge=0.0,
        alias="AS_GAMMA_EDGE",
    )
    as_edge_alpha: float = Field(
        default=1.0, ge=0.0,
        alias="AS_EDGE_ALPHA",
    )
    as_k_floor_per_min: float = Field(
        default=0.1, ge=0.0,
        alias="AS_K_FLOOR_PER_MIN",
    )
    as_base_floor_bps: float = Field(
        default=1.0, ge=0.0,
        alias="AS_BASE_FLOOR_BPS",
    )
    as_min_half_spread_bps: float = Field(
        default=1.5, ge=0.0,
        alias="AS_MIN_HALF_SPREAD_BPS",
    )
    as_max_half_spread_bps: float = Field(
        default=30.0, ge=0.0,
        alias="AS_MAX_HALF_SPREAD_BPS",
    )
    # K-intensity estimator (8A.2). Cache + refresh interval keep
    # the O(N) scan over recent_fills off the per-tick hot path.
    as_k_intensity_window_seconds: float = Field(
        default=3600.0, gt=0.0,
        alias="AS_K_INTENSITY_WINDOW_SECONDS",
    )
    as_k_intensity_refresh_seconds: float = Field(
        default=60.0, gt=0.0,
        alias="AS_K_INTENSITY_REFRESH_SECONDS",
    )
    base_half_spread_bps_low_vol: float = Field(
        default=4.0,
        ge=0.0,
        alias="BASE_HALF_SPREAD_BPS_LOW_VOL",
    )
    base_half_spread_low_vol_threshold_bps_per_s: float = Field(
        default=6.0,
        ge=0.0,
        alias="BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S",
    )

    # v1.5.156 Option B — no-fill–aware gradual spread compression.
    #
    # Even with vol-adaptive base (Option A), the bot's spread can
    # still exceed the natural touch when taker flow is genuinely
    # absent. Track seconds since the last fill (any side); after a
    # generous trigger, gradually compress half-spread at a slow
    # rate until either a fill happens (timer resets, compression
    # clears) or the maximum compression is reached.
    #
    # Per CLAUDE.md Rule 0c this is an ADAPTIVE mechanism:
    #   * Entry condition is a live signal (time since last fill)
    #   * Exit condition is a live signal (next fill resets timer)
    #   * Re-evaluated every tick
    # NOT force-trading — the compression is bounded by max_bps and
    # by ``min_half_spread_bps``, and a single fill on either side
    # clears it.
    #
    # When all safety gates are clear, this provides a slow
    # spread-tightening as a "last resort" when nothing else fired
    # to attract fills. The first fill resets and recompression
    # only re-engages after another full trigger period.
    no_fill_compress_enabled: bool = Field(
        default=False,
        alias="NO_FILL_COMPRESS_ENABLED",
    )
    no_fill_compress_trigger_seconds: float = Field(
        default=600.0,
        ge=0.0,
        alias="NO_FILL_COMPRESS_TRIGGER_SECONDS",
    )
    no_fill_compress_rate_bps_per_minute: float = Field(
        default=1.0,
        ge=0.0,
        alias="NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE",
    )
    no_fill_compress_max_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="NO_FILL_COMPRESS_MAX_BPS",
    )

    # v1.5.248 — no-fill aggression escalator. See
    # app/no_fill_escalator.py for the full design rationale.
    #
    # When ENABLED, takes over the no-fill response:
    #   - Replaces the simple NO_FILL_COMPRESS spread-only path
    #     with a unified aggression-level (0.0-1.0) that drives:
    #     spread compression + microprice widening attenuation +
    #     toxicity bump attenuation + reservation-shift attenuation.
    #   - Aggression level ramps linearly from 0 at TRIGGER_SECONDS
    #     to 1 at TRIGGER_SECONDS + RAMP_SECONDS, then saturates.
    #   - Resets to 0 immediately on first fill.
    #
    # When DISABLED (default), legacy NO_FILL_COMPRESS behavior
    # is preserved exactly. The escalator becomes a no-op.
    #
    # Defaults are conservative; the operator opts in via env.
    no_fill_escalator_enabled: bool = Field(
        default=False,
        alias="NO_FILL_ESCALATOR_ENABLED",
    )
    # Seconds since last fill before the escalator starts ramping.
    no_fill_escalator_trigger_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="NO_FILL_ESCALATOR_TRIGGER_SECONDS",
    )
    # Linear ramp window. 0 = step function at the trigger.
    no_fill_escalator_ramp_seconds: float = Field(
        default=180.0,
        ge=0.0,
        alias="NO_FILL_ESCALATOR_RAMP_SECONDS",
    )
    # Max spread compression in bp at level=1.0. Subtracted from
    # half-spread; floored by MIN_HALF_SPREAD_BPS downstream.
    no_fill_escalator_spread_compress_max_bps: float = Field(
        default=10.0,
        ge=0.0,
        alias="NO_FILL_ESCALATOR_SPREAD_COMPRESS_MAX_BPS",
    )
    # When True, microprice widening (which adds defensive bp on the
    # at-risk side when microprice diverges from mid) is attenuated
    # linearly with the aggression level: mult = 1 - level.
    no_fill_escalator_microprice_attenuate: bool = Field(
        default=True,
        alias="NO_FILL_ESCALATOR_MICROPRICE_ATTENUATE",
    )
    # Same for toxicity soft-trigger half-spread bumps.
    no_fill_escalator_toxicity_attenuate: bool = Field(
        default=True,
        alias="NO_FILL_ESCALATOR_TOXICITY_ATTENUATE",
    )
    # Multiplier on the sum of reservation-shift alphas at level=1.0.
    # 1.0 = no change (alphas keep full strength even at max
    # aggression — useful if alphas are believed to be the right
    # signal regardless of fill rate). 0.5 = halve the alpha pull
    # at full aggression (quotes drift toward mid). 0.0 = full
    # suppression (quote symmetrically around mid).
    no_fill_escalator_reservation_shift_mult_at_full: float = Field(
        default=0.5,
        ge=0.0,
        alias="NO_FILL_ESCALATOR_RESERVATION_SHIFT_MULT_AT_FULL",
    )

    # v1.5.158 Option A — vol-climbing anticipatory widening.
    #
    # Driving observation: v1.5.154-260526-100625 overnight showed
    # 8 of 10 SFs cluster in the Dubai 06:00-10:00 window (UTC 02-06),
    # the Asia-mid-day -> European pre-market overlap. Realized vol
    # climbs visibly in the 15-30 min before the SF cluster begins.
    # If the bot widens its quotes a small amount during that climb,
    # it absorbs the early adverse fills less aggressively and gives
    # the v1.5.155 trend skew more time to lean the inventory in
    # the correct direction.
    #
    # Mechanism (Rule 0c — adaptive, signal-driven, NOT time-driven):
    #   * Maintain a deque of vol_bps samples in BotState.
    #   * Compute short-window MA (e.g. last 5 min) and long-window
    #     MA (e.g. last 30 min) over the deque.
    #   * Fire when short_MA / long_MA >= ratio_threshold AND both
    #     windows have at least min_samples points.
    #   * Effect: add ``widen_bps`` to half-spread (composes via the
    #     existing min/max half-spread clamps).
    #   * Re-evaluated every tick; clears the moment the ratio drops
    #     below ``clear_ratio`` (hysteresis prevents flapping).
    #
    # Defaults are conservative — short MA 5 min, long MA 30 min,
    # arm at 1.5x, clear at 1.1x, widen 1.5 bps. Tune the threshold
    # down if the operator finds the gate doesn't fire ahead of real
    # vol climbs; tune the widening down if fills drop too much.
    vol_climbing_widen_enabled: bool = Field(
        default=False,
        alias="VOL_CLIMBING_WIDEN_ENABLED",
    )
    vol_climbing_widen_short_window_seconds: float = Field(
        default=300.0,  # 5 min
        ge=0.0,
        alias="VOL_CLIMBING_WIDEN_SHORT_WINDOW_SECONDS",
    )
    vol_climbing_widen_long_window_seconds: float = Field(
        default=1800.0,  # 30 min
        ge=0.0,
        alias="VOL_CLIMBING_WIDEN_LONG_WINDOW_SECONDS",
    )
    vol_climbing_widen_arm_ratio: float = Field(
        default=1.5,
        ge=1.0,
        alias="VOL_CLIMBING_WIDEN_ARM_RATIO",
    )
    vol_climbing_widen_clear_ratio: float = Field(
        default=1.1,
        ge=0.0,
        alias="VOL_CLIMBING_WIDEN_CLEAR_RATIO",
    )
    vol_climbing_widen_bps: float = Field(
        default=1.5,
        ge=0.0,
        alias="VOL_CLIMBING_WIDEN_BPS",
    )
    vol_climbing_widen_min_samples: int = Field(
        default=20,
        ge=1,
        alias="VOL_CLIMBING_WIDEN_MIN_SAMPLES",
    )

    # v1.5.158 Option B — funding-settle anticipatory widening.
    #
    # OKX TON-USDT-SWAP perp settles funding at 00:00 / 08:00 / 16:00
    # UTC. Around these times traders rebalance positions to capture
    # / avoid the funding rate, which produces a brief volume spike
    # and price-discovery activity. Operator-approved time-driven
    # widening because:
    #   * The event is known and recurring at fixed cadence.
    #   * The window is short and bounded (default ±15 min).
    #   * The widening is small (1-2 bps) — NOT a HOLD_ALL or
    #     suppression. Bot keeps quoting both sides throughout.
    #   * Cannot recur continuously for hours (only fires ~3x/day,
    #     30 min total per fire).
    # Per the operator's Rule 0c clarification (2026-05-27 chat):
    # "I am ok widen a bit for scheduled 30 minutes interval around
    # or before funding payments".
    #
    # The settle hours list is configurable in case venue cadence
    # changes (some venues use 4h or 1h funding). Default matches
    # OKX standard.
    funding_settle_widen_enabled: bool = Field(
        default=False,
        alias="FUNDING_SETTLE_WIDEN_ENABLED",
    )
    funding_settle_widen_hours_utc: str = Field(
        default="0,8,16",
        alias="FUNDING_SETTLE_WIDEN_HOURS_UTC",
    )
    funding_settle_widen_pre_minutes: float = Field(
        default=15.0,
        ge=0.0,
        alias="FUNDING_SETTLE_WIDEN_PRE_MINUTES",
    )
    funding_settle_widen_post_minutes: float = Field(
        default=15.0,
        ge=0.0,
        alias="FUNDING_SETTLE_WIDEN_POST_MINUTES",
    )
    funding_settle_widen_bps: float = Field(
        default=1.5,
        ge=0.0,
        alias="FUNDING_SETTLE_WIDEN_BPS",
    )

    # v1.5.158 Option C — vol-adaptive position cap.
    #
    # When realized vol exceeds the configured threshold, shrink
    # MAX_ABS_POSITION dynamically. NOT a "stop trading" gate — bot
    # keeps quoting both sides; only the position cap changes. This
    # caps the worst-case bleed during the morning vol cluster.
    # Pure position-management; Rule 0c-aligned.
    #
    # Logic at each tick:
    #   * If vol_climbing_widen has armed (vol_bps in the climbing
    #     regime as detected above), OR if instantaneous vol_bps >=
    #     ``vol_threshold_bps_per_s``, apply the reduced cap.
    #   * Reduced cap = ``MAX_ABS_POSITION × reduction_factor``,
    #     floored at 1 (must allow at least 1 contract or the bot
    #     can't trade at all — that would be Rule 0c violation).
    # Defaults: reduce to 50% when vol > 10 bp/s.
    vol_adaptive_position_cap_enabled: bool = Field(
        default=False,
        alias="VOL_ADAPTIVE_POSITION_CAP_ENABLED",
    )
    vol_adaptive_position_cap_threshold_bps_per_s: float = Field(
        default=10.0,
        ge=0.0,
        alias="VOL_ADAPTIVE_POSITION_CAP_THRESHOLD_BPS_PER_S",
    )
    vol_adaptive_position_cap_reduction_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        alias="VOL_ADAPTIVE_POSITION_CAP_REDUCTION_FACTOR",
    )
    # Hard floor on intentional half-spread (bps per side) after model + inventory mode are known.
    # Neutral two-sided steady state uses the neutral floor; one-sided inventory-reduction uses the
    # inventory floor (may be tighter). Toxicity score adds a linear bump on top of max(min, econ).
    economic_min_half_spread_neutral_bps: float = Field(
        default=8.0,
        ge=0.0,
        alias="ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS",
    )
    economic_min_half_spread_inventory_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS",
    )
    # 2026-05-13 todo-019 Part B: tick-aware one-sided floor (codex
    # #6). On tight-tick symbols the bps-denominated floors above
    # round sub-tick at typical fair values (TON: 1 tick ≈ 1 bp half
    # at $2.30 → ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS=2.0 is barely
    # 2 ticks), so when one-sided the floor doesn't bind and the
    # bot's lone quote sits at or near touch. That collapse-to-touch
    # is the structural cause of the BUY-vs-SELL asymmetry observed
    # in 0512-225450 and 0513-073423 snapshots.
    # ``compute_effective_min_half_spread_bps`` takes the larger of
    # the bps floor and ``(1 + extra_tick) * tick_bps_half``. Extra
    # ticks default to 0 (no-op until configured per symbol).
    one_sided_extra_tick_neutral: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        alias="ONE_SIDED_EXTRA_TICK_NEUTRAL",
    )
    one_sided_extra_tick_inventory: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        alias="ONE_SIDED_EXTRA_TICK_INVENTORY",
    )
    economic_toxicity_score_half_spread_bps: float = Field(
        default=4.0,
        ge=0.0,
        alias="ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS",
    )
    # Coefficient applied to ``toxicity.score`` (0..1) when building the strategy
    # half-spread in ``compute_quote_decision``. Default 12 matches the prior
    # hardcoded value and preserves HL-calibrated behaviour. On tight books (1-tick
    # spread) this must be dropped by ~10× to keep quotes near BBO — see
    # ``config/profiles/prod.grvt.env`` for the GRVT override.
    toxicity_score_half_spread_bps: float = Field(
        default=12.0,
        ge=0.0,
        alias="TOXICITY_SCORE_HALF_SPREAD_BPS",
    )
    # Toxicity-conditional size reduction coefficient. The quote
    # multiplier is computed as ``clip(1.0 - coeff × score, 0.2, 1.0)``
    # in ``compute_quote_decision``. At the legacy default of 0.5,
    # score=0.5 yields multiplier 0.75 (25 % size reduction) and
    # score=1.0 yields 0.50 (50 % reduction). At 1.0 the reduction
    # is twice as aggressive (50 % at score=0.5, capped at 80 % at
    # score≥0.8). Intent: when toxicity is high, smaller fills mean
    # less inventory carried through the adverse-markout window —
    # directly attacks the residual P&L drag observed under TON's
    # high-vol regime (see ``plans/20260507-calibrate.md`` Tier 1).
    toxicity_size_reduction_coeff: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        alias="TOXICITY_SIZE_REDUCTION_COEFF",
    )
    # Additive bps applied on top of the model half-spread when ToxicityEngine
    # raises ``soft_trigger``. Was previously a hardcoded ``+=4.0`` inside
    # ``compute_quote_decision``. Same reasoning as above: 4 bps on a tight book
    # is an order of magnitude too wide.
    toxicity_soft_trigger_half_spread_bump_bps: float = Field(
        default=4.0,
        ge=0.0,
        alias="TOXICITY_SOFT_TRIGGER_HALF_SPREAD_BUMP_BPS",
    )

    # ------------------------------------------------------------------
    # Vol-spike runtime adapter (``app/vol_regime.py``). Shrinks quote
    # size + bumps half-spread floor automatically when realised vol
    # spikes vs baseline; persists the defensive posture for a cooldown
    # after the spike. Default OFF — ship per-profile after the calmer
    # config tuning has had time to settle. See BUGS/todo-009.md for
    # the design + rationale (the 2026-05-07 SUI vol-spike bleed).
    # ------------------------------------------------------------------
    vol_shrink_coeff: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        alias="VOL_SHRINK_COEFF",
    )
    """``shrink_factor = clip(1 - VOL_SHRINK_COEFF * (vol_ratio - 1), VOL_SHRINK_FLOOR, 1.0)``

    With ``VOL_SHRINK_COEFF=0.25``, vol_ratio=3 → shrink=0.5 (size halves,
    quote_notional halves on every NEW order). At ``vol_ratio=1`` → 1.0
    (no change). Default ``0.0`` disables the feature entirely;
    ``shrink_factor`` stays at 1.0, no behaviour change."""

    vol_shrink_floor: float = Field(
        default=0.3,
        ge=0.05,
        le=1.0,
        alias="VOL_SHRINK_FLOOR",
    )
    """Minimum shrink factor — clipping floor for the formula above.
    Prevents the bot from collapsing to zero size under runaway vol;
    a 30 % floor still allows the venue's min_notional self-heal path
    to function."""

    vol_spike_threshold: float = Field(
        default=2.5,
        ge=1.0,
        alias="VOL_SPIKE_THRESHOLD",
    )
    """``vol_ratio`` (current / baseline) at which a spike window opens.
    Default 2.5x mirrors the existing toxicity-score vol-ratio cutoff.
    Set very high (e.g. 100.0) to effectively disable the persistence
    cooldown while keeping ``vol_shrink_coeff`` active."""

    vol_spike_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="VOL_SPIKE_COOLDOWN_SECONDS",
    )
    """How long the spike window remains active after a threshold
    crossing. While active, the shrink factor is held no higher than
    the spike-time level (no immediate relax) and ``half_spread`` gets
    the bump below."""

    # Phase 2K.8 (v1.4.160) — favorable-exit predicate for the vol_spike
    # latch. The ``VOL_SPIKE_COOLDOWN_SECONDS`` timer is the MAX
    # ceiling; the latch also clears EARLY when
    # ``vol_ratio < VOL_SPIKE_THRESHOLD × VOL_SPIKE_CLEAR_BAND_MULT``
    # (e.g. threshold=2.5 and mult=0.7 → clear band 1.75 — vol calmer
    # than the trigger but not all the way back to baseline) and held
    # there for ``VOL_SPIKE_FAVORABLE_EXIT_DWELL_SECONDS``. The dwell
    # is small (default 5 s) — vol_ratio is a per-tick signal so we
    # only need a few quote loops of stability. Set ``ENABLED=False``
    # for legacy pure-timer behaviour.
    vol_spike_favorable_exit_enabled: bool = Field(
        default=True,
        alias="VOL_SPIKE_FAVORABLE_EXIT_ENABLED",
    )
    vol_spike_clear_band_mult: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="VOL_SPIKE_CLEAR_BAND_MULT",
    )
    vol_spike_favorable_exit_dwell_seconds: float = Field(
        default=5.0,
        ge=0.0,
        alias="VOL_SPIKE_FAVORABLE_EXIT_DWELL_SECONDS",
    )

    vol_spike_half_spread_bump_bps: float = Field(
        default=1.0,
        ge=0.0,
        alias="VOL_SPIKE_HALF_SPREAD_BUMP_BPS",
    )
    """Additive bps lift on the half-spread floor while inside the
    spike window. Captures the post-spike continuation risk
    (bounce-and-retest pattern). 0.0 disables the spread effect; the
    inventory/size shrink still applies if ``vol_shrink_coeff > 0``."""

    # ------------------------------------------------------------------
    # Join-depth auto-tune controller (``app/join_depth_controller.py``).
    # Bounded scalar overlay added to ``base_half_spread_bps`` based on
    # observed cross-rejection rate, adverse markouts (1s + 5s
    # horizons), and fill-rate vs target. Closes the loop on
    # "queue value vs pick-off cost". Off by default — opt-in per
    # profile after the simpler config tuning has had time to settle.
    # See ``plans/auto-tune.md`` for the design + rollout plan.
    # ------------------------------------------------------------------
    join_depth_autotune_enabled: bool = Field(
        default=False,
        alias="JOIN_DEPTH_AUTOTUNE_ENABLED",
    )
    # Update interval (seconds). Controller samples + updates the
    # overlay this often; quote engine reads the cached overlay every
    # tick. 30s gives the EWMA enough memory while keeping reaction
    # time well below typical session length.
    join_depth_autotune_update_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS",
    )
    # EWMA blend weight for the new target overlay vs the previous
    # value. 0.3 ≈ 5 update cycles of memory at 30s = 2.5min half-life.
    join_depth_autotune_ewma_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        alias="JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA",
    )
    # Hard bounds on the overlay. Negative = quote tighter; positive =
    # wider. Composes with min/max half-spread clamps in quote_engine,
    # so this can't push the bot below MIN_HALF_SPREAD_BPS or above
    # MAX_HALF_SPREAD_BPS regardless.
    join_depth_autotune_overlay_min_bps: float = Field(
        default=-2.0,
        alias="JOIN_DEPTH_AUTOTUNE_OVERLAY_MIN_BPS",
    )
    join_depth_autotune_overlay_max_bps: float = Field(
        default=6.0,
        alias="JOIN_DEPTH_AUTOTUNE_OVERLAY_MAX_BPS",
    )
    # Per-input contribution coefficients. Each maps a normalized
    # signal magnitude to bps of overlay. See plans/auto-tune.md for
    # the rationale and intended sensitivity.
    join_depth_autotune_alpha_reject: float = Field(
        default=0.5,
        ge=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT",
    )
    join_depth_autotune_alpha_markout_1s: float = Field(
        default=0.3,
        ge=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S",
    )
    join_depth_autotune_alpha_markout_5s: float = Field(
        default=0.2,
        ge=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_5S",
    )
    join_depth_autotune_alpha_underfill: float = Field(
        default=0.1,
        ge=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_ALPHA_UNDERFILL",
    )
    # Target fill rate in fills/min. Fewer than this pushes overlay
    # NEGATIVE (tighter quoting) by ``alpha_underfill × shortfall``.
    join_depth_autotune_target_fills_per_min: float = Field(
        default=1.0,
        ge=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN",
    )
    # Saturation guard: log a warning event when the overlay sits at
    # the max bound continuously for this many seconds. Means market
    # regime has shifted enough that static config needs revisiting.
    join_depth_autotune_saturation_warn_seconds: float = Field(
        default=300.0,
        gt=0.0,
        alias="JOIN_DEPTH_AUTOTUNE_SATURATION_WARN_SECONDS",
    )

    vol_window_samples: int = Field(default=32, alias="VOL_WINDOW_SAMPLES")
    vol_multiplier: float = Field(default=1.0, alias="VOL_MULTIPLIER")

    # v1.5.239 — EWMA-of-|log-return| trend-aware volatility measure
    # (app/vol_abs_ewma.py). Runs unconditionally so the value is
    # always available on the snapshot; consumer wiring is gated.
    vol_abs_ewma_halflife_seconds: float = Field(
        default=20.0, alias="VOL_ABS_EWMA_HALFLIFE_SECONDS"
    )
    # Phase 4.5 A/B flag — when True, the regime classifier's
    # vol_slope criterion computes its slope from
    # ``state.forward_vol_abs_ewma_bps_history`` instead of the
    # legacy ``state.forward_vol_bps_history``. Default False
    # preserves pre-v1.5.239 behaviour exactly.
    regime_forward_use_vol_abs_ewma_for_slope: bool = Field(
        default=False, alias="REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE"
    )

    inventory_skew_coeff_bps: float = Field(default=25.0, alias="INVENTORY_SKEW_COEFF_BPS")
    # Skew shape exponent. Applied as ``sign(norm_inv) * |norm_inv|**exp``
    # so the coefficient above is still the maximum shift at full
    # utilisation, but the ramp between 0 and max is non-linear.
    #
    #   exp=1.0 (default) — linear, back-compat.
    #   exp=3.0 (cubic)   — gentle below ~50 % util, aggressive near the
    #     cap. At exp=3: 50 % util → 12.5 % of max shift (linear gives
    #     50 %); 85 % util → 61 % of max shift (linear 85 %); 100 % util
    #     → 100 % (same as linear).
    #
    # Motivation (``tmp/snap_20260419_152651``): session showed 5672
    # cycles in ``soft_skew_short`` vs 2438 in ``soft_skew_long`` — bot
    # position frequently sat in the 72–85 % util band, pushing the
    # reducing-side quote so far off mid it became invisible, preventing
    # the unwind that would bring us back to neutral. Cubic shape keeps
    # ordinary inventory oscillation (±50 % util) near-neutral while
    # still applying firm pressure when the position gets genuinely
    # stuck near the cap.
    inventory_skew_exponent: float = Field(
        default=1.0,
        gt=0,
        alias="INVENTORY_SKEW_EXPONENT",
    )
    # Reference-venue fair-value blend weight (Tier 1 adaptive lever #7).
    # Formula inside ``compute_quote_decision``:
    #
    #   ref_price_blended = (1 - alpha) * grvt_mid + alpha * (bybit_mid + basis_ewma)
    #
    # At ``alpha=0`` (default) this is the legacy pure-GRVT reservation.
    # At ``alpha=0.5`` the reservation is halfway between GRVT's mid and
    # the reference-venue-anchored fair value (Bybit mid + smoothed
    # basis). When GRVT jumps ahead of Bybit on a thin flow, the blend
    # pulls our quotes back toward the cross-venue consensus — mitigating
    # the "buy high / sell low" pattern seen on ETH and SOL sessions
    # (``tmp/snap_20260419_162938``: 13/13 BUYs adverse in a
    # down-trending market).
    # Requires a reference venue connected (``state.binance_mid`` and
    # ``state.binance_basis_ewma`` populated); when either is None the
    # shift is skipped (no-op fallback).
    reference_venue_fair_blend_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="REFERENCE_VENUE_FAIR_BLEND_ALPHA",
    )
    # Short-term drift bias on the reservation (Tier 1 adaptive lever #1).
    # Shifts the reservation by ``alpha * drift_bps``. Positive drift
    # (market rising) → reservation up → ask moves up (less prone to be
    # lifted into a continuing rise) and bid moves up (catches a fill
    # earlier in the up-move). Symmetric in a downtrend.
    #
    # Pre-v1.5.155 this was wired exclusively to
    # ``eff_q.mid_return_250ms_bps`` (250 ms drift). In any sustained
    # trend (e.g. 10 bps/30 s) the 250 ms drift averages ~0.08 bps, so
    # ``alpha * 0.08`` ≈ 0.024 bps — effectively zero skew in any
    # non-shock regime. The bot quoted symmetrically around mid in
    # trends and got adversely selected on both sides. See
    # ``snapshots/v1.5.154-260526-074029`` for the canonical failure:
    # 67% time in CAUTIOUS, 32% win rate, -2.6 bps mean 5 s markout.
    #
    # v1.5.155 added ``trend_drift_signal_window_seconds`` so the
    # constructive skew can read from longer-horizon drift signals
    # (``state.mid_drift_windows`` 5/10/30/60 s windows). At
    # ``alpha=0`` (default) legacy reservation regardless of window;
    # typical tuned value for the 10 s window is ~1.0 (yields ~3 bps
    # skew in a 10 bps/30 s trend — meaningful but moderate).
    trend_drift_reservation_alpha: float = Field(
        default=0.0,
        ge=0.0,
        alias="TREND_DRIFT_RESERVATION_ALPHA",
    )
    # v1.5.155 — drift-signal window selector for
    # ``trend_drift_reservation_alpha``. Maps approximately to the
    # nearest window in ``state.mid_drift_windows``:
    #
    #   <= 0.3  → ``eff_q.mid_return_250ms_bps`` (legacy)
    #   <= 0.6  → ``eff_q.mid_return_500ms_bps`` (fallback chain to 250ms)
    #   <= 7.5  → ``mid_drift_windows.drift_5s_bps``
    #   <= 20.0 → ``mid_drift_windows.drift_10s_bps``
    #   <= 45.0 → ``mid_drift_windows.drift_30s_bps``
    #   >  45.0 → ``mid_drift_windows.drift_60s_bps``
    #
    # When the chosen window is ``None`` (deque warmup), falls back to
    # progressively shorter windows: chosen → 5 s → 500 ms → 250 ms.
    # Default 0.25 preserves pre-v1.5.155 behaviour for back-compat
    # (tests, other profiles). The OKX TON prod profile sets 10.0
    # together with ``alpha=1.0`` to lean into 10-second drift signals
    # — see v1.5.154-260526-074029 snapshot diagnostic for the
    # motivation.
    trend_drift_signal_window_seconds: float = Field(
        default=0.25,
        ge=0.0,
        le=60.0,
        alias="TREND_DRIFT_SIGNAL_WINDOW_SECONDS",
    )
    # Order-book imbalance bias on the reservation (Priority #1, see
    # ``docs/priorities.md`` + ``deep-research/ob-imbalance-*``).
    #
    # Composition inside ``compute_quote_decision``:
    #   I_L1 = (bid_size - ask_size) / (bid_size + ask_size)    ∈ [-1, 1]
    #   I_ewma_t = (1 - s) * I_ewma_{t-1} + s * I_L1_t          [smoothing]
    #   I_clipped = clip(I_ewma, -CLIP, +CLIP)
    #   α_imb = ALPHA * (half_spread_bps / 2) * I_clipped * ref / 10_000
    #   reservation += α_imb
    #
    # At ``ALPHA=0`` (default) the term is disabled and
    # ``compute_quote_decision`` behaves identically to the pre-imbalance
    # logic. Typical tuned value: 0.3 — shifts reservation up to ~30 % of
    # half-spread in the direction of imbalance. Both research sources
    # (ChatGPT + Gemini, stored under ``deep-research/``) strongly
    # recommend adding as an *alpha term* (alongside skew / basis blend)
    # rather than replacing the mid anchor — which would cause flickering
    # and destroy queue priority.
    ob_imbalance_alpha: float = Field(
        default=0.0,
        ge=0.0,
        alias="OB_IMBALANCE_ALPHA",
    )
    # EWMA step per observation. At GRVT's ~200 ms public-WS cadence,
    # 0.3 gives a half-life of roughly 500 ms (~3 samples) — matches
    # the research-recommended smoothing window for 5 s markout alpha.
    ob_imbalance_smoothing_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        alias="OB_IMBALANCE_SMOOTHING_ALPHA",
    )
    # Hard clip on |I_smoothed|. Values ≥ 0.9 are statistically more
    # likely to reflect spoofing / iceberg flicker than genuine
    # directional pressure; clipping at 0.85 keeps the signal in the
    # "strong but credible" band.
    ob_imbalance_clip: float = Field(
        default=0.85,
        gt=0.0,
        lt=1.0,
        alias="OB_IMBALANCE_CLIP",
    )
    # Minimum total top-of-book notional (bid + ask, USD) required to
    # trust the ratio. Sparse books produce meaningless imbalance values
    # (single small order on one side → |I| near 1 with no information).
    # When `(bid_size + ask_size) * mid < this floor`, the EWMA is not
    # updated for that observation and the existing smoothed value is
    # held constant. 10 USD = 0 on a dry run (we're quoting $25-40 so our
    # own orders provide more than this); adjust upward on venues with
    # large book depth.
    ob_imbalance_depth_floor_usd: float = Field(
        default=10.0,
        ge=0.0,
        alias="OB_IMBALANCE_DEPTH_FLOOR_USD",
    )
    # Cross-venue basis-deviation alpha term (Priority #2, see
    # ``docs/priorities.md`` + ``deep-research/cross-venue-*``).
    #
    # What it measures: how far the instantaneous GRVT–Bybit basis is from
    # its long-run smoothed value. A positive deviation means GRVT is
    # currently "expensive" vs Bybit (after accounting for typical
    # basis); a negative deviation means "cheap". The v1 implementation
    # assumes mean-reversion: when deviation is positive we shift
    # reservation DOWN (expecting GRVT to fall toward Bybit-anchored
    # fair), and vice versa.
    #
    # Composition inside ``compute_quote_decision``:
    #   dev_px = basis_now - basis_ewma      [caller supplies both]
    #   dev_bps = dev_px / ref_price * 10_000
    #   dev_norm = clip(dev_bps / CLIP_BPS, -1, +1)
    #   shift_bps = -ALPHA × (half_spread_bps / 2) × dev_norm
    #   reservation += shift_bps × ref / 10_000
    #
    # Negative sign = mean-reversion. Upgrade to regime-switched
    # (Hurst exponent or EWMA IC) in a future iteration if the simple
    # mean-reversion default misfires during trending regimes.
    #
    # At ``ALPHA=0`` (default) the term is disabled. Typical tuned value
    # 0.3 — same magnitude cap as OB imbalance (~0.3 bps shift at
    # saturation on a 2 bps half-spread).
    basis_deviation_alpha: float = Field(
        default=0.0,
        ge=0.0,
        alias="BASIS_DEVIATION_ALPHA",
    )
    # Deviation magnitude (in bps) at which the normalised signal
    # saturates at ±1. Deviations beyond this are clipped. Default 2.0 =
    # a 2 bps GRVT–Bybit dislocation is "large". Funding jumps and
    # short-term microstructure noise usually stay within this bound on
    # normal SOL flow; outliers past this are treated as clipped rather
    # than passed through raw.
    basis_deviation_clip_bps: float = Field(
        default=2.0,
        gt=0.0,
        alias="BASIS_DEVIATION_CLIP_BPS",
    )
    # Priority #2 v2 — online regime classifier for the basis signal.
    # Computes Pearson IC between lagged ``dev_bps`` and realized future
    # mid-return at ``HORIZON_SECONDS`` lookback. Sign of IC selects the
    # regime: negative → mean-reversion (shift against dev), positive →
    # trend-continuation (shift with dev), |IC| < threshold → undecided
    # (skip the alpha). Guards against the 2026-04-20 afternoon
    # regression where v1's unconditional mean-reversion caused
    # wrong-sign shifts during a trending SOL session.
    basis_deviation_regime_horizon_seconds: float = Field(
        default=2.0,
        gt=0.0,
        alias="BASIS_DEVIATION_REGIME_HORIZON_SECONDS",
    )
    # Number of (lagged_dev, return) pairs retained for the rolling IC.
    # At a typical 2 Hz observation rate, 240 samples ≈ 2 min of data —
    # responsive to regime changes while giving statistical room for
    # the Pearson estimator.
    basis_deviation_regime_window_samples: int = Field(
        default=240,
        ge=10,
        alias="BASIS_DEVIATION_REGIME_WINDOW_SAMPLES",
    )
    # IC magnitude required to declare a regime. With a 240-sample
    # window, 0.15 ≈ 2.3σ under the null of no correlation — strict
    # enough to suppress noise-driven regime flips but loose enough to
    # catch real regimes fast. Raise toward 0.25-0.30 if regime flips
    # too often; drop toward 0.10 to react faster in short sessions.
    basis_deviation_regime_ic_threshold: float = Field(
        default=0.15,
        ge=0.0,
        lt=1.0,
        alias="BASIS_DEVIATION_REGIME_IC_THRESHOLD",
    )
    # Minimum pair count before the classifier yields any sign. Acts as
    # a cold-start guard — prevents the first handful of pairs from
    # driving a premature regime call. At 2 Hz observation, 50 samples
    # ≈ 25 s of warmup.
    basis_deviation_regime_min_pair_samples: int = Field(
        default=50,
        ge=2,
        alias="BASIS_DEVIATION_REGIME_MIN_PAIR_SAMPLES",
    )
    # Priority #3 — flow-direction / toxicity score from public trade
    # prints. See ``docs/priorities.md`` + ``deep-research/flow-direction-*``
    # and ``app/flow_score.py`` for the full design. v1 implementation
    # ships with OBSERVABILITY ONLY: the score is computed and exposed
    # in ``/state/current`` but ``FLOW_SCORE_PAUSE_THRESHOLD=1.1``
    # (unreachable by construction) means no pre-fill suppression
    # actually fires until the operator validates the score's behaviour
    # against live data and lowers the threshold.
    grvt_trade_stream_enabled: bool = Field(
        default=True,
        alias="GRVT_TRADE_STREAM_ENABLED",
    )
    # Public trade-stream primary selector. GRVT's API documentation
    # uses the same ``v1.<primary>`` pattern across feeds; private WS
    # already subscribes to ``v1.fill`` / ``v1.order``, public WS
    # subscribes to ``v1.mini.s``. The trade-prints feed follows the
    # same convention as ``v1.trade``. Overridable via env in case
    # GRVT's exact name differs slightly (e.g. ``v1.trade.s``) and we
    # need to flip without a redeploy.
    grvt_trade_stream_name: str = Field(
        default="v1.trade",
        alias="GRVT_TRADE_STREAM_NAME",
    )
    # Bounded retention of recent trade prints on ``BotState``. 500
    # prints at GRVT SOL trade rates is roughly 5-10 min of history —
    # enough for the longest window in the flow score (30s VPIN
    # buckets). Higher values just cost memory.
    flow_score_recent_trades_maxlen: int = Field(
        default=500,
        ge=50,
        alias="FLOW_SCORE_RECENT_TRADES_MAXLEN",
    )
    # TFI computation window in seconds. Research recommends 250ms-1s
    # for fast decisions; we use 1s as the v1 default — responsive
    # enough to catch aggressor streaks, noisy enough to be
    # statistically meaningful on our typical 1-2 trades/s SOL flow.
    flow_score_tfi_window_seconds: float = Field(
        default=1.0,
        gt=0.0,
        alias="FLOW_SCORE_TFI_WINDOW_SECONDS",
    )
    # Streak feature — number of most-recent prints to scan for
    # consecutive same-side runs. 10 is the simple aggressor-streak
    # window from the research; larger values catch longer trends at
    # the cost of responsiveness.
    flow_score_streak_window_prints: int = Field(
        default=10,
        ge=2,
        alias="FLOW_SCORE_STREAK_WINDOW_PRINTS",
    )
    # Score threshold for side-pause. Score in [0, 1]. v1 default is
    # 1.1 — unreachable by construction, so the pause path never fires
    # (observability-only mode). Drop to 0.75 (per research) or lower
    # once live data validates the score.
    flow_score_pause_threshold: float = Field(
        default=1.1,
        ge=0.0,
        alias="FLOW_SCORE_PAUSE_THRESHOLD",
    )
    # How long each side-pause lasts (seconds) when triggered. 1.0 s
    # matches the upper end of the research recommendation (300-1000 ms).
    flow_score_pause_seconds: float = Field(
        default=1.0,
        gt=0.0,
        alias="FLOW_SCORE_PAUSE_SECONDS",
    )
    # Priority #3 v2 — flow-score as a reservation-price alpha.
    # Composes additively with OB-imbalance, trend-drift, and basis-
    # deviation alphas. Sign: positive TFI (buy-pressure) shifts
    # reservation UP — bid leans into expected continuation, ask
    # widens away from imminent up-move. Streak length and TFI
    # signed magnitude are the two flow features used; both are
    # already published by ``FlowScoreAccumulator.snapshot()``.
    #
    # Composition inside ``compute_quote_decision``:
    #   tfi_clipped = clip(tfi_signed, -CLIP, +CLIP)
    #   streak_signed = signed_streak_count / streak_window_prints
    #   composite = 0.5 * tfi_clipped + 0.5 * streak_signed
    #   shift_bps = ALPHA * (half_spread_bps / 2) * composite
    #   reservation += shift_bps * ref / 10_000
    #
    # At ``ALPHA=0`` (default) the term is disabled. Typical tuned
    # value 0.4 — slightly larger than OB-imbalance because flow-
    # score combines two features and is less prone to spoofing
    # noise than raw L1 size imbalance.
    flow_score_reservation_alpha: float = Field(
        default=0.0,
        ge=0.0,
        alias="FLOW_SCORE_RESERVATION_ALPHA",
    )
    # Hard clip on |composite| before the half-spread scaling, same
    # rationale as ``OB_IMBALANCE_CLIP``: signal at saturation is
    # statistically more likely to be one-off prints than persistent
    # flow.
    flow_score_reservation_clip: float = Field(
        default=0.85,
        gt=0.0,
        lt=1.0,
        alias="FLOW_SCORE_RESERVATION_CLIP",
    )

    # v1.5.209 Phase 8D — Order Flow Imbalance (OFI) directional alpha.
    # The 5th reservation-alpha lever. Default-dormant: ``alpha=0.0``
    # means the OFI accumulator runs + publishes EWMAs to live_stats
    # (for offline calibration via Phase 3F-style attribution) but
    # contributes ZERO shift to reservation. Operator bumps to 0.05 /
    # 0.1 after auditing 24h+ of attribution. Same pattern as the
    # other 4 alphas. See ``app/ofi.py`` for the accumulator + helper.
    ofi_enabled: bool = Field(default=True, alias="OFI_ENABLED")
    ofi_reservation_alpha: float = Field(
        default=0.0,
        ge=0.0,
        alias="OFI_RESERVATION_ALPHA",
    )
    # Halflives for the two EWMAs maintained by the OFI accumulator.
    # 5s is the short-horizon signal that feeds the reservation
    # shift. 30s is the anchor (published to live_stats, not used
    # for shift in v1.5.209 — operator may wire later).
    ofi_halflife_5s_seconds: float = Field(
        default=5.0,
        gt=0.0,
        alias="OFI_HALFLIFE_5S_SECONDS",
    )
    ofi_halflife_30s_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="OFI_HALFLIFE_30S_SECONDS",
    )
    # Tanh normalisation scale. The raw OFI sum is in "shares of net
    # buying pressure" units — scale-dependent. ``ofi_signal =
    # tanh(raw_ewma / scale)``. Calibrate so that "strong" signals
    # land near ±0.8 and noise stays inside ±0.2. TON L1 sizes are
    # typically tens-to-hundreds of contracts; default 100.0 puts a
    # full-size flip near saturation.
    ofi_normalisation_scale: float = Field(
        default=100.0,
        gt=0.0,
        alias="OFI_NORMALISATION_SCALE",
    )
    # Clip on the post-tanh signal before applying alpha. Defensive;
    # tanh already bounds the output, this is belt-and-suspenders.
    ofi_signal_clip: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        alias="OFI_SIGNAL_CLIP",
    )

    # v1.5.209 Phase 8B — Queue-position-aware sizing + inside-spread
    # post. Two independent flags so the operator can A/B-test sizing
    # alone, then sizing + inside-post. See ``app/queue_model.py`` for
    # the helpers + the day plan (Feature 4) for the full design.
    queue_aware_sizing_enabled: bool = Field(
        default=False,
        alias="QUEUE_AWARE_SIZING_ENABLED",
    )
    queue_aware_inside_post_enabled: bool = Field(
        default=False,
        alias="QUEUE_AWARE_INSIDE_POST_ENABLED",
    )
    # Size multiplier shape: ``mult = clamp(floor, 1.0, 1.0 - decay × ratio)``.
    # Defaults match the day plan spec (floor=0.3, decay=0.7).
    queue_aware_size_floor: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        alias="QUEUE_AWARE_SIZE_FLOOR",
    )
    queue_aware_size_decay: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="QUEUE_AWARE_SIZE_DECAY",
    )
    # Inside-post gate: post one tick inside the spread when
    # queue_position_ratio exceeds this threshold AND the inside-
    # post flag is true. 0.7 = "70 % of the inside queue is ahead
    # of us" — bot is deep in the queue, the expected wait is poor.
    queue_aware_inside_post_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="QUEUE_AWARE_INSIDE_POST_THRESHOLD",
    )
    # Arrival-rate EWMA halflife. Read by the bot's per-tick caller;
    # the EWMA itself lives on ``state.queue_arrival_rate``.
    queue_arrival_rate_halflife_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="QUEUE_ARRIVAL_RATE_HALFLIFE_SECONDS",
    )
    # Inside-post step in bps. When inside-post fires, the bot
    # narrows its half-spread by this much (capped at half_spread / 2
    # so the quote never crosses the reservation). For a 1-tick venue
    # this should map approximately to ``1 tick in bps`` — operator
    # tunes for the live tick size. Default 1.0 bp is conservative
    # for TON's tick.
    queue_aware_inside_post_step_bps: float = Field(
        default=1.0,
        gt=0.0,
        alias="QUEUE_AWARE_INSIDE_POST_STEP_BPS",
    )

    inventory_soft_limit_pct: float = Field(default=0.72, alias="INVENTORY_SOFT_LIMIT_PCT")
    inventory_hard_limit_pct: float = Field(default=0.85, alias="INVENTORY_HARD_LIMIT_PCT")

    max_session_loss_usd: float = Field(default=75.0, alias="MAX_SESSION_LOSS_USD")
    max_drawdown_usd: float = Field(default=75.0, alias="MAX_DRAWDOWN_USD")

    toxicity_enabled: bool = Field(default=True, alias="TOXICITY_ENABLED")
    toxicity_markout_soft_bps: float = Field(default=3.0, alias="TOXICITY_MARKOUT_SOFT_BPS")
    toxicity_markout_hard_bps: float = Field(default=12.0, alias="TOXICITY_MARKOUT_HARD_BPS")
    # v1.5.291 — minimum number of resolved-markout fills required before
    # the *markout* hard-trigger path (avg_adv <= -hard_bps) can fire. The
    # one-sided hard path already has its own sample-size floor
    # (``len(fl) >= 8``); the markout path historically had NONE, so a
    # 2-fill window averaging -8.1bp tripped a hard SOFT_FLATTEN. On a
    # sub-min-notional residual that SF can't actually close, this fed the
    # residual_below_min_notional no-op loop that the sf_fatigue ladder
    # escalated to a tier-4 kill (snapshot
    # v1.5.290-260530-223859: 2 fills, -8.1bp, killed in 7 min over a
    # $0.006 loss; the identical loop also killed pre-deploy at 03:35).
    # Statistically you cannot conclude "toxic flow" from 1-2 fills, so
    # gate the markout path on a real sample. Default 0 preserves the
    # legacy (ungated) behaviour so other profiles + the test suite are
    # unchanged; the prod profile sets 4 (matching
    # ``toxicity_one_sided_min_fills``). The gate is adaptive: it is
    # re-evaluated every tick from the CURRENT rolling window, so a
    # genuine toxic burst (which piles up >= min_fills adverse markouts
    # quickly) still trips hard as intended — only the noise-grade
    # 1-3 fill case is suppressed.
    toxicity_markout_hard_min_fills: int = Field(
        default=0,
        ge=0,
        le=50,
        alias="TOXICITY_MARKOUT_HARD_MIN_FILLS",
    )
    toxicity_one_sided_fill_ratio: float = Field(default=0.75, alias="TOXICITY_ONE_SIDED_FILL_RATIO")
    # Minimum fill count before the one-sided-fill-ratio signal can fire. At the
    # old default of 4 fills, a single 3:1 fill split (pure noise at that sample
    # size) arms ``adverse_spread_widen`` and the score component of toxicity.
    # Combined with the self-perpetuation bug (now fixed), this produced 17
    # minutes of invisible quoting in ``tmp/snap_20260418_094415``. Gate it.
    toxicity_one_sided_min_fills: int = Field(
        default=4,
        ge=2,
        le=50,
        alias="TOXICITY_ONE_SIDED_MIN_FILLS",
    )
    toxicity_cooldown_seconds: float = Field(default=30.0, alias="TOXICITY_COOLDOWN_SECONDS")
    # v1.5.197 — time-decay filter for the recent_fills buffer the
    # toxicity engine reads from. Fills older than this many seconds
    # are dropped BEFORE the engine computes its outputs (markout
    # average, one-sided ratio, etc.). Prevents the defensive-deadlock
    # pattern where a single catastrophic-markout fill stays in the
    # buffer permanently after defenses suppress new-fill flow, keeping
    # toxicity_hard / one-sided triggers latched.
    # Default 600 s (10 min) = the toxicity window resets after 10 min
    # of no fills regardless of how adverse the recent burst was.
    # Set to 0 to disable the time-decay (fall back to count-only).
    toxicity_recent_fills_max_age_seconds: float = Field(
        default=600.0,
        ge=0.0,
        alias="TOXICITY_RECENT_FILLS_MAX_AGE_SECONDS",
    )
    # v1.4.155 Phase 2K.5 — favorable-exit knobs for the
    # ``adaptive_spread_widen`` overlay. The overlay is armed by ANY
    # of six trigger reasons (toxicity_hard / toxicity_soft /
    # markout_adverse / one_sided_ratio / quote_quality / slow_trend);
    # each reason has its own per-tick "signal cleared" predicate
    # (e.g. toxicity_hard clears when ``hard_trigger=False`` again,
    # one_sided_ratio clears when the ratio drops back below the
    # configured firing threshold, etc.). When the predicate for the
    # arming reason holds continuously for ``dwell_seconds``, the
    # overlay clears EARLY rather than waiting the full
    # ``toxicity_cooldown_seconds`` deadline. ``dwell_seconds`` =
    # noise filter (a single-tick blip back below threshold doesn't
    # cause premature clearing). Setting it to 0 OR setting
    # ``adaptive_spread_widen_favorable_exit_enabled=false`` disables
    # the predicate entirely → pure-timer behaviour matching pre-2K.5.
    adaptive_spread_widen_favorable_exit_enabled: bool = Field(
        default=True,
        alias="ADAPTIVE_SPREAD_WIDEN_FAVORABLE_EXIT_ENABLED",
    )
    adaptive_spread_widen_favorable_exit_dwell_seconds: float = Field(
        default=10.0,
        ge=0.0,
        alias="ADAPTIVE_SPREAD_WIDEN_FAVORABLE_EXIT_DWELL_SECONDS",
    )
    # v1.5.157 — position-aware favorable-exit knobs for adaptive_widen.
    # Per CLAUDE.md Rule 0c. The existing Phase 2K.5 markout-based
    # exit requires the ARM-TIME signal to clear, which often doesn't
    # happen during sustained adverse regimes (88 ceiling vs 10
    # favorable in the v1.5.154-260526-074029 snapshot). This second
    # exit path fires when:
    #   |position_qty| >= inventory_threshold AND
    #   sign(position_qty) * drift_bps >= drift_threshold_bps
    # I.e. bot has meaningful inventory + current drift is in the
    # SAME direction as that inventory = inventory gaining value =
    # widening is now obstructing a profitable unwind. Reads drift
    # via Bot._select_trend_drift_signal (v1.5.155 selector).
    adaptive_spread_widen_position_favorable_exit_enabled: bool = Field(
        default=True,
        alias="ADAPTIVE_SPREAD_WIDEN_POSITION_FAVORABLE_EXIT_ENABLED",
    )
    adaptive_spread_widen_position_favorable_inventory_threshold: float = Field(
        default=1.0,
        ge=0.0,
        alias="ADAPTIVE_SPREAD_WIDEN_POSITION_FAVORABLE_INVENTORY_THRESHOLD",
    )
    adaptive_spread_widen_position_favorable_drift_threshold_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="ADAPTIVE_SPREAD_WIDEN_POSITION_FAVORABLE_DRIFT_THRESHOLD_BPS",
    )

    # Post-swing PnL cooldown gate (analysis-day 2026-05-10 behavioural change).
    # When session total PnL changes by more than ``POST_SWING_PNL_DELTA_USD``
    # in either direction over a rolling ``POST_SWING_WINDOW_SECONDS`` window,
    # quoting is paused (HOLD_ALL) for ``POST_SWING_COOLDOWN_SECONDS`` while the
    # regime clarifies. Targets the whipsaw pattern where a directional move
    # generates unrealised gains, the bot rotates inventory through the rebound,
    # then the next leg fills the opposite side at fresh adverse prices —
    # observed in snapshot 260510064549 between 00:59 (peak +$1.23) and 01:05
    # (give-back to +$0.02).
    #
    # Set ``POST_SWING_ENABLED=false`` to disable. Tune ``DELTA_USD`` against
    # session-typical PnL noise — a ~$0.50 swing in 60s is the threshold that
    # caught the 2026-05-10 whipsaw without firing on routine drift.
    post_swing_enabled: bool = Field(default=True, alias="POST_SWING_ENABLED")
    post_swing_pnl_delta_usd: float = Field(
        default=0.50, ge=0.0, alias="POST_SWING_PNL_DELTA_USD"
    )
    post_swing_window_seconds: float = Field(
        default=60.0, gt=0.0, alias="POST_SWING_WINDOW_SECONDS"
    )
    post_swing_cooldown_seconds: float = Field(
        default=120.0, ge=0.0, alias="POST_SWING_COOLDOWN_SECONDS"
    )
    # v1.4.154 Phase 2K.4 — favorable-exit predicate knobs for
    # post_swing_gate. When the cooldown is active and the rolling
    # PnL-delta over the visible sample window shrinks below
    # ``post_swing_pnl_delta_usd × clear_band_mult``, start the
    # favorable-exit dwell timer. After ``dwell_seconds`` of
    # continuous sub-clear-band delta, the cooldown clears early.
    # Same dual-track architecture as the rest of Phase 2K — keeps
    # ``post_swing_cooldown_seconds`` as MAX-ceiling safety net.
    # ``clear_band_mult=0`` disables the predicate (pure-timer fallback).
    post_swing_clear_band_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="POST_SWING_CLEAR_BAND_MULT",
    )
    post_swing_favorable_exit_dwell_seconds: float = Field(
        default=15.0,
        ge=0.0,
        alias="POST_SWING_FAVORABLE_EXIT_DWELL_SECONDS",
    )

    # Inventory-aligned-with-momentum gate (analysis-day 2026-05-10 behavioural
    # change). When ``sign(position) == sign(recent mid-return drift)`` AND
    # ``|position|/effective_cap >= POST_SWING_MOMENTUM_INVENTORY_PCT``, refuse
    # to add to that side (forces the adding-side cap on the existing
    # inventory_exec_bias suppression). This is an EXTENSION of the existing
    # inventory-bias logic, not a replacement: today's gate triggers on
    # inventory size alone; this one adds the momentum direction.
    #
    # Targets the snapshot 260510064549 pattern where the bot held a long
    # position into a continuing downtrend at 01:01-01:05 UTC, refilling the
    # long side via inventory_exec_bias permitting the bid. With the momentum
    # condition in place, the gate would refuse the bid because long+downtrend
    # = inventory aligned with adverse direction.
    momentum_gate_enabled: bool = Field(default=True, alias="MOMENTUM_GATE_ENABLED")
    momentum_gate_drift_threshold_bps: float = Field(
        default=2.0, ge=0.0, alias="MOMENTUM_GATE_DRIFT_THRESHOLD_BPS"
    )
    momentum_gate_inventory_pct: float = Field(
        default=0.40, ge=0.0, le=1.0, alias="MOMENTUM_GATE_INVENTORY_PCT"
    )

    # Trend-aware inventory-skew amplifier (1.3.82). Companion to
    # momentum_gate. Same trigger geometry (sign(position) ==
    # sign(drift), |drift| over threshold, util over threshold)
    # multiplies the inventory_skew_coeff_bps by ``factor`` while
    # active, so the REDUCING side quote pulls closer to touch.
    # Default off pending live observation; thresholds match the
    # momentum gate so the two arm together when enabled.
    trend_skew_amplifier_enabled: bool = Field(
        default=False, alias="TREND_SKEW_AMPLIFIER_ENABLED"
    )
    trend_skew_amplifier_drift_threshold_bps: float = Field(
        default=2.0, ge=0.0, alias="TREND_SKEW_AMPLIFIER_DRIFT_THRESHOLD_BPS"
    )
    trend_skew_amplifier_inventory_pct: float = Field(
        default=0.40, ge=0.0, le=1.0, alias="TREND_SKEW_AMPLIFIER_INVENTORY_PCT"
    )
    trend_skew_amplifier_factor: float = Field(
        default=1.5, ge=1.0, le=5.0, alias="TREND_SKEW_AMPLIFIER_FACTOR"
    )

    # 30s-MAE gate (1.3.82). Defensive HOLD_ALL cooldown driven by
    # the rolling-N-fill average of post-fill 30s MAE (max adverse
    # excursion). Parallel to the toxicity engine's hard trigger but
    # at the 30s horizon. Producer is the post-fill-excursion
    # watcher (which already runs for observability stamping); this
    # gate adds a feedback path. Default off pending live tuning.
    #
    # Threshold semantics: ``hard_threshold_bps`` is a positive
    # number; the gate fires when the rolling avg of min(0, mae_30s)
    # crosses BELOW its negation (e.g. 5.0 → fires when avg ≤ -5.0).
    mae_gate_enabled: bool = Field(default=False, alias="MAE_GATE_ENABLED")
    mae_gate_hard_threshold_bps: float = Field(
        default=5.0, ge=0.0, alias="MAE_GATE_HARD_THRESHOLD_BPS"
    )
    mae_gate_fill_window: int = Field(
        default=10, ge=1, le=200, alias="MAE_GATE_FILL_WINDOW"
    )
    mae_gate_cooldown_seconds: float = Field(
        default=180.0, ge=0.0, alias="MAE_GATE_COOLDOWN_SECONDS"
    )
    # Phase 2K.7 (v1.4.158) — favorable-exit predicate for the mae_gate.
    # The ``MAE_GATE_COOLDOWN_SECONDS`` timer is the MAX ceiling; the
    # gate additionally clears EARLY when the rolling N-fill average
    # has recovered above ``-hard_threshold_bps × clear_band_mult``
    # (e.g. hard=5.0 and mult=0.5 → clear band -2.5 bps) and held
    # there for ``favorable_exit_dwell_seconds``. The dwell is small
    # (default 5 s) because each new 30 s-resolved fill is a real
    # signal change — long hysteresis windows just delay the obvious.
    # Set ``ENABLED=False`` for legacy pure-timer behaviour.
    mae_gate_favorable_exit_enabled: bool = Field(
        default=True,
        alias="MAE_GATE_FAVORABLE_EXIT_ENABLED",
    )
    mae_gate_clear_band_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="MAE_GATE_CLEAR_BAND_MULT",
    )
    mae_gate_favorable_exit_dwell_seconds: float = Field(
        default=5.0,
        ge=0.0,
        alias="MAE_GATE_FAVORABLE_EXIT_DWELL_SECONDS",
    )
    # v1.5.155 — position-aware favorable-exit predicate for mae_gate.
    # Operator instruction (CLAUDE.md Rule 0c, 2026-05-26): every
    # timer-based gate must ALSO have a signal-driven conditional
    # exit that considers current bot position. The Phase 2K.7
    # markout-based exit (above) only fires when recent fill bleed
    # stops — but during a cooldown there are no new fills, so it
    # never triggers (v1.5.154-260526-074029 snapshot: 78 cooldowns,
    # 0 favorable, 78 ceiling).
    #
    # The position-aware exit clears the cooldown when:
    #   |position_qty| >= inventory_threshold AND
    #   sign(position_qty) * drift_bps >= drift_threshold_bps
    #
    # I.e. bot has meaningful inventory AND current drift is in the
    # SAME direction as that inventory (inventory is gaining value =
    # good time to unwind). Reads from
    # ``state.mid_drift_windows.drift_10s_bps`` via the v1.5.155
    # signal selector.
    #
    # Defaults are conservative — fires only when inventory is at
    # least 1 contract AND drift is at least ±5 bps over the chosen
    # window. Tune in the per-symbol prod profile.
    mae_gate_position_favorable_inventory_threshold: float = Field(
        default=1.0,
        ge=0.0,
        alias="MAE_GATE_POSITION_FAVORABLE_INVENTORY_THRESHOLD",
    )
    mae_gate_position_favorable_drift_threshold_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="MAE_GATE_POSITION_FAVORABLE_DRIFT_THRESHOLD_BPS",
    )
    # v1.5.197 — idle-decay clearance threshold. When the gate is
    # active AND no fill has arrived for this many seconds, the gate
    # auto-clears. Breaks the defensive-deadlock pattern where the
    # gate suppresses fills (HOLD_ALL) and therefore can never
    # re-evaluate its predicate (which needs fresh fill markouts).
    # See plans/20260527-regime-band-hysteresis.md v1.5.197 section.
    mae_gate_idle_clear_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="MAE_GATE_IDLE_CLEAR_SECONDS",
    )
    # v1.5.197 — same pattern for at_touch_adverse_pause. Per-side
    # pause auto-clears after IDLE_CLEAR_SECONDS of no fills.
    at_touch_adverse_pause_idle_clear_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_IDLE_CLEAR_SECONDS",
    )
    # v1.5.197 — same pattern for realised_edge_side_suppress.
    realised_edge_suppress_idle_clear_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="REALISED_EDGE_SUPPRESS_IDLE_CLEAR_SECONDS",
    )

    # 1.3.85 reconcile hydration guard. When a reconcile cycle sees
    # an exchange-only order (remote has X, local doesn't), check
    # whether the bot recently saw that same (oid, cloid) terminate.
    # If yes within this window, skip the hydration — the REST
    # snapshot is just lagging behind WS, not a genuine orphan.
    # Counter exposed as ``state.hydration_skipped_recently_terminal_total``.
    # 60s is well past typical OKX REST staleness (<5s).
    reconcile_hydration_recent_terminal_window_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="RECONCILE_HYDRATION_RECENT_TERMINAL_WINDOW_SECONDS",
    )

    # 1.3.86: cancel-before-ack defer guard. When True (default),
    # cancels issued against an order whose place HTTP response
    # hasn't returned yet are PARKED rather than dispatched. They
    # flush automatically the moment the order transitions to
    # ACKED. This eliminates the place-cancel race that produced
    # the 50 phantom-place gone_on_exchange events in snapshot
    # 260515-095056. Disable only for diagnostic A/B comparison
    # against the old (bug-prone) behaviour.
    cancel_defer_until_ack_enabled: bool = Field(
        default=True, alias="CANCEL_DEFER_UNTIL_ACK_ENABLED"
    )

    # Tiered session-PnL drawdown gate (1.2.1, snapshot 260510081612
    # follow-up). Catches the chronic-bleed pattern where session PnL
    # erodes slowly without firing any single-event gate. Tier ladder:
    #
    #   tier1 (-tier1_widen_usd): widen spreads (existing
    #     adaptive_widen overlay armed with reason=session_drawdown).
    #   tier2 (-tier2_pause_short_usd): soft-flatten + pause for
    #     pause_short_seconds, then RESUME_TESTING.
    #   tier3 (-tier3_pause_long_usd): soft-flatten + pause for
    #     pause_long_seconds, then RESUME_TESTING.
    #   tier4 (-tier4_kill_usd): hard kill (operator restart).
    #
    # RESUME_TESTING samples the next ``test_resume_sample_fills``
    # fills' mean markout; clears if ≥ -test_resume_max_adverse_bps,
    # escalates to the next tier otherwise. See
    # ``app/session_drawdown_gate.py`` for the algorithm.
    #
    # Defaults are starting points sized to SUI's $1k account; tune
    # per profile. Set ``SESSION_DRAWDOWN_ENABLED=false`` to disable.
    session_drawdown_enabled: bool = Field(
        default=True, alias="SESSION_DRAWDOWN_ENABLED"
    )
    session_drawdown_tier1_widen_usd: float = Field(
        default=0.50, ge=0.0, alias="SESSION_DRAWDOWN_TIER1_WIDEN_USD"
    )
    session_drawdown_tier2_pause_short_usd: float = Field(
        default=1.50, ge=0.0, alias="SESSION_DRAWDOWN_TIER2_PAUSE_SHORT_USD"
    )
    session_drawdown_tier3_pause_long_usd: float = Field(
        default=3.00, ge=0.0, alias="SESSION_DRAWDOWN_TIER3_PAUSE_LONG_USD"
    )
    session_drawdown_tier4_kill_usd: float = Field(
        default=5.00, ge=0.0, alias="SESSION_DRAWDOWN_TIER4_KILL_USD"
    )
    session_drawdown_pause_short_seconds: float = Field(
        default=300.0, gt=0.0, alias="SESSION_DRAWDOWN_PAUSE_SHORT_SECONDS"
    )
    session_drawdown_pause_long_seconds: float = Field(
        default=1800.0, gt=0.0, alias="SESSION_DRAWDOWN_PAUSE_LONG_SECONDS"
    )
    session_drawdown_test_resume_sample_fills: int = Field(
        default=10, ge=1, le=100, alias="SESSION_DRAWDOWN_TEST_RESUME_SAMPLE_FILLS"
    )
    session_drawdown_test_resume_max_adverse_bps: float = Field(
        default=1.5, ge=0.0, alias="SESSION_DRAWDOWN_TEST_RESUME_MAX_ADVERSE_BPS"
    )
    # Phase 2K.10 (v1.5.182) — continuous-test favorable-exit
    # predicate for Tier 2 / Tier 3 pauses. Default OFF for safe
    # rollout; flip to true once acceptance is green. When enabled,
    # session_pnl recovering above (arm_threshold × clear_band_ratio)
    # for ``dwell_seconds`` transitions early to RESUME_TESTING
    # instead of waiting for the fixed pause ceiling.
    session_drawdown_favorable_exit_enabled: bool = Field(
        default=False,
        alias="SESSION_DRAWDOWN_FAVORABLE_EXIT_ENABLED",
    )
    session_drawdown_favorable_exit_clear_band_ratio: float = Field(
        default=0.5, gt=0.0, le=1.0,
        alias="SESSION_DRAWDOWN_FAVORABLE_EXIT_CLEAR_BAND_RATIO",
    )
    session_drawdown_favorable_exit_dwell_seconds: float = Field(
        default=5.0, ge=0.0,
        alias="SESSION_DRAWDOWN_FAVORABLE_EXIT_DWELL_SECONDS",
    )

    # Vol × trend conjunction gate (1.2.2). When realised vol AND
    # directional drift are BOTH elevated for at least
    # ``persistence_seconds``, suppress quoting (HOLD_ALL) for
    # ``cooldown_seconds``. Targets the directional-burst regime
    # where the bot's both-sided quotes get adversely selected
    # by aggressive flow chasing the move.
    vol_trend_gate_enabled: bool = Field(
        default=True, alias="VOL_TREND_GATE_ENABLED"
    )
    vol_trend_gate_vol_multiplier: float = Field(
        default=2.5, ge=1.0, alias="VOL_TREND_GATE_VOL_MULTIPLIER"
    )
    vol_trend_gate_drift_threshold_bps: float = Field(
        default=3.0, ge=0.0, alias="VOL_TREND_GATE_DRIFT_THRESHOLD_BPS"
    )
    vol_trend_gate_persistence_seconds: float = Field(
        default=10.0, ge=0.0, alias="VOL_TREND_GATE_PERSISTENCE_SECONDS"
    )
    vol_trend_gate_cooldown_seconds: float = Field(
        default=120.0, ge=0.0, alias="VOL_TREND_GATE_COOLDOWN_SECONDS"
    )
    # v1.4.153 Phase 2K.3 — favorable-exit predicate knobs.
    #   * ``clear_band_mult`` is the hysteresis-band multiplier.
    #     When the cooldown is active and BOTH ``vol_ratio`` and
    #     ``|drift_bps|`` fall below ``mult × <their trigger
    #     threshold>``, start the favorable-exit dwell timer. With
    #     default mult=0.7, the gate clears when vol drops below
    #     70% of the trigger AND drift drops below 70% of the
    #     trigger — comfortably out of the danger band.
    #   * ``dwell_seconds`` is the time both signals must stay
    #     below the clear band before the cooldown clears early.
    #     10 s by default — short enough to react when conditions
    #     genuinely ease, long enough to filter out single-tick
    #     blips during a stress event.
    # Setting either knob to 0 disables favorable-exit (pure timer
    # behaviour). Cooldown_seconds remains as MAX ceiling.
    vol_trend_gate_clear_band_mult: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="VOL_TREND_GATE_CLEAR_BAND_MULT",
    )
    vol_trend_gate_favorable_exit_dwell_seconds: float = Field(
        default=10.0,
        ge=0.0,
        alias="VOL_TREND_GATE_FAVORABLE_EXIT_DWELL_SECONDS",
    )

    # Basis-regime IC quote gate (1.2.2; size-shrink mode 1.2.8).
    # When |IC| < ``ic_min_quote_threshold`` the regime is
    # signal-absent and the gate fires. Two response modes:
    #
    #   ``hold_all``    — clamp eligibility to HOLD_ALL (legacy).
    #                     Sits out entirely. Throws away rebate.
    #   ``size_shrink`` — keep quoting at reduced size (default
    #                     0.5×). Preserves rebate income during
    #                     quiet regimes; only shrinks exposure.
    #                     Composes via ``min`` with toxicity / vol /
    #                     markout shrinks in ``compute_quote_decision``.
    #
    # The 2026-05-10 wedge (basis IC sat at 0.035 for 13+ minutes
    # while the bot stared at the screen earning nothing) drove the
    # default flip from ``hold_all`` to ``size_shrink``. Operator
    # can flip back via env if the size-shrink response turns out
    # to bleed.
    #
    # Set ``ic_min_quote_threshold = 0.0`` to effectively disable
    # the gate while keeping the feature flag on.
    basis_regime_gate_enabled: bool = Field(
        default=True, alias="BASIS_REGIME_GATE_ENABLED"
    )
    basis_regime_gate_ic_min_quote: float = Field(
        default=0.05, ge=0.0, alias="BASIS_REGIME_GATE_IC_MIN_QUOTE"
    )
    basis_regime_gate_mode: str = Field(
        default="size_shrink",
        alias="BASIS_REGIME_GATE_MODE",
        pattern=r"^(hold_all|size_shrink)$",
    )
    basis_regime_gate_size_mult_signal_absent: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="BASIS_REGIME_GATE_SIZE_MULT_SIGNAL_ABSENT",
    )

    # Multi-level (laddered) quoting (1.2.14, plans/multi-level.md
    # Phase 1 — shadow / instrumentation).
    #
    # When ``num_levels_per_side == 1`` (default) the bot behaves
    # exactly as v1.2.13: one bid + one ask per cycle, no new code
    # paths exercised. Set to 2..5 to ENABLE LADDER COMPUTATION:
    # the per-cycle ``LadderDecision`` is built and persisted to
    # ``quote_decisions`` rows for offline calibration.
    #
    # PHASE 1 IMPORTANT NOTE: outer rungs are NOT placed on the
    # venue yet. Phase 2 (separate plan) wires multi-rung order
    # placement via OKX batch endpoints. With Phase 1 alone, set
    # ``num_levels_per_side > 1`` is "shadow mode" — the operator
    # can analyze what the ladder WOULD have done in snapshots
    # before taking outer-rung risk.
    ladder_num_levels_per_side: int = Field(
        default=1, ge=1, le=5, alias="LADDER_NUM_LEVELS_PER_SIDE"
    )
    # Rung price spacing as a multiple of the cycle's half-spread.
    # Each rung i sits at ``half_spread × (1 + offset_step × i)``
    # away from the reservation price. 1.0 = rung i at i+1 half-
    # spreads (default). Higher = sparser ladder. No-op at N=1.
    ladder_offset_step: float = Field(
        default=1.0, ge=0.0, le=10.0, alias="LADDER_OFFSET_STEP"
    )
    # Phase 2J (v1.4.169) — minimum gap between adjacent ladder rungs
    # measured in TICKS. The legacy bps-math rung-price computation
    # (``half_spread × (1 + offset_step × i)``) can underflow the
    # venue's price tick when ``half_spread`` is small relative to
    # the tick. Without this floor, two rungs whose bps math produces
    # a difference < 1 tick would snap to the same grid cell and the
    # outer rung would be silently DROPPED by the legacy
    # ``grid_collision`` dedup.
    #
    # ``ladder_tick_floor_steps = 1`` (default): rung i sits at
    # least i × 1 × tick away from inside (i.e., at least 1 tick gap
    # per rung step). ``0`` disables the floor (legacy behaviour;
    # collision-dedup is the only safeguard). ``2`` / ``3`` reserve
    # more grid headroom on symbols where 1-tick steps are too
    # aggressive. No-op at ``LADDER_NUM_LEVELS_PER_SIDE=1`` (since
    # there are no outer rungs).
    #
    # Counter: ``BotState.ladder_rung_tick_floor_adjusted_{bid,ask}_total``
    # increments once per rung shift; surfaced in the snapshot so
    # the operator can see how often bps math was about to underflow.
    ladder_tick_floor_steps: int = Field(
        default=1, ge=0, le=10, alias="LADDER_TICK_FLOOR_STEPS"
    )
    # Geometric size decay ratio. Rung i's size is base × decay^i
    # (or decay^max(0, i-1) when inside_full_size=True). 0.7
    # default. 1.0 = flat (discouraged for sweep-day robustness).
    # No-op at N=1.
    ladder_size_decay: float = Field(
        default=0.7, ge=0.1, le=1.0, alias="LADDER_SIZE_DECAY"
    )
    # When True (default), rung 0 (inside) gets full base_size and
    # decay starts at rung 1. No-op at N=1.
    ladder_inside_full_size: bool = Field(
        default=True, alias="LADDER_INSIDE_FULL_SIZE"
    )
    # When True (default), gates can publish ``max_levels_per_side``
    # caps that clamp the effective N below the configured value.
    # When False, gate caps are ignored — useful for shadow-mode
    # calibration where the operator wants to see the unclamped
    # ladder shape.
    ladder_gates_limit_levels: bool = Field(
        default=True, alias="LADDER_GATES_LIMIT_LEVELS"
    )
    # Phase 2 flag — wires multi-rung execution via OKX batch
    # endpoints. Defaults False; flip with the Phase 2 implementation.
    # In Phase 1, this knob is ignored — only the inside rung is
    # placed regardless of value.
    ladder_batch_orders_enabled: bool = Field(
        default=False, alias="LADDER_BATCH_ORDERS_ENABLED"
    )

    # v1.4.99 — Inventory-aware rung pruning (DORMANT BY DEFAULT).
    # When ``LADDER_INVENTORY_AWARE_PRUNING_ENABLED=true`` AND
    # ``|position|/MAX_ABS_POSITION >= LADDER_INVENTORY_AWARE_PRUNING_THRESHOLD_PCT``,
    # the ladder builder clamps the ADDING side's effective N to 1
    # (only rung 0 placed). The REDUCING side is untouched so the
    # bot can still bleed inventory across the full ladder.
    #
    # Default OFF so this can ship to the binary without changing
    # production behaviour. The flag is the SINGULAR gate — the
    # entire pruning computation is skipped when False, so dormant
    # deployments produce snapshot data bit-identical to the
    # pre-v1.4.99 behaviour.
    #
    # Flip to true when calibration data justifies — see
    # ``plans/ladder-observability.md`` for the data-driven
    # decision framework (specifically F2 drop-attribution
    # counters that distinguish inventory-buffer drops from
    # grid/min-notional drops).
    #
    # Rationale: existing ``apply_inventory_high_adding_side_buffer``
    # in ``app/quote_aging.py`` widens the adding-side quote when
    # loaded but doesn't drop rungs explicitly. This adds the
    # deterministic hard-cap layer.
    ladder_inventory_aware_pruning_enabled: bool = Field(
        default=False,
        alias="LADDER_INVENTORY_AWARE_PRUNING_ENABLED",
    )
    # Threshold for engaging the prune (fraction of
    # ``MAX_ABS_POSITION``). Default 0.65 mirrors
    # ``INVENTORY_SOFT_LIMIT_PCT`` — the same point where
    # one-sided clamps already engage. Below this: full ladder
    # both sides. Above: adding side clamped to 1 rung.
    ladder_inventory_aware_pruning_threshold_pct: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        alias="LADDER_INVENTORY_AWARE_PRUNING_THRESHOLD_PCT",
    )

    # Microprice / OB-imbalance passive-adverse-selection gate
    # (1.2.2). When ``ob_imbalance_ewma`` shows a clear depth
    # asymmetry, the thin side faces high adverse-selection risk
    # on the next aggressive print. Suppress that side
    # (one-sided quoting). Default threshold 0.5 ("clear visible
    # asymmetry"). Lower = more sensitive.
    microprice_gate_enabled: bool = Field(
        default=True, alias="MICROPRICE_GATE_ENABLED"
    )
    microprice_gate_ob_imbalance_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="MICROPRICE_GATE_OB_IMBALANCE_THRESHOLD"
    )
    # v1.4.42 BUG-026 / v1.4.47 structural rewrite: inventory-direction
    # suppression of microprice-gate widening on the reducing side.
    #
    # The microprice gate's "widen the side facing adverse selection"
    # logic only makes sense for the ADDING side of current inventory.
    # When the bot is short and the book signals a drop, getting filled
    # on the bid REDUCES the short — that's the bot's intended
    # unwinding, not adverse selection. Widening the bid pushes it out
    # of reach and traps the bot. Same in mirror for long + ask-thin.
    #
    # The knob is now SIGN-encoded for three modes — DEFAULT IS THE
    # STRUCTURAL DIRECTION RULE, no number to tune:
    #
    #   < 0    (DEFAULT, ``-1.0`` recommended) — STRUCTURAL:
    #          direction-only. Position has a sign → reducing side is
    #          fully determined. Suppress widening on whichever side
    #          is reducing. No threshold, no magic number, no tuning.
    #
    #   == 0   DISABLED — pre-v1.4.42 behaviour. Microprice gate
    #          widens unconditionally; reducing side can get pushed
    #          out of reach in a one-sided regime.
    #
    #   > 0    LEGACY THRESHOLD MODE (back-compat for v1.4.42-v1.4.46
    #          deployments that pinned an explicit utilization
    #          fraction). Suppression fires when
    #          ``|position| / abs_cap >= reducing_side_widen_suppress_pct``.
    #          Not recommended for new deployments — the v1.4.47
    #          structural rule beats every fixed threshold across
    #          regimes.
    #
    # History: v1.4.42 0.30 → v1.4.46 coupled-to-0.12 → v1.4.47
    # direction-only (no number). Each iteration was the operator
    # pushing back on yet another magic number; the v1.4.47 rewrite
    # eliminates the concept of a threshold entirely.
    microprice_gate_reducing_side_widen_suppress_pct: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
        alias="MICROPRICE_GATE_REDUCING_SIDE_WIDEN_SUPPRESS_PCT",
    )

    # ------------------------------------------------------------------
    # Tape runtime-feed (shared-memory) consumer knobs — M7/M8.
    #
    # The recorder (`tape`, Rust — repo dtc-tools-tape) publishes a
    # 256-byte runtime-signal frame into a POSIX shared-memory segment
    # (/dev/shm/dtc-tape-runtime-v1) at ~1 Hz. The bot's reader lives in
    # ``app/runtime_recorder_feed.py``. See docs/architecture.md §9-§10
    # and the recorder repo's docs/execution-plan.md Phase C.
    #
    # MASTER GATE — default OFF. The bot NEVER depends on the recorder
    # being up: when this is false the feed is never opened and every
    # downstream consumer (warm-start vol seed, microprice-z widen)
    # no-ops, falling back to the bot's in-process signals. Flip to true
    # only on a host where the recorder is running and has built >24 h of
    # coverage (coverage_valid=1).
    regime_use_runtime_recorder_feed: bool = Field(
        default=False, alias="REGIME_USE_RUNTIME_RECORDER_FEED"
    )
    # Staleness budget (seconds) for runtime-feed reads. The recorder
    # stamps CLOCK_MONOTONIC (system-wide on Linux) so the bot compares
    # directly with no cross-clock skew. A frame older than this is
    # dropped (read_fresh -> None, runtime_feed_stale_count++) and the
    # consumer falls back to in-process signals. 5 s ≈ a few missed 1 Hz
    # publishes — generous enough to ride out a brief recorder hiccup,
    # tight enough that a wedged recorder stops feeding the bot quickly.
    regime_runtime_feed_stale_threshold_s: float = Field(
        default=5.0,
        ge=0.0,
        alias="REGIME_RUNTIME_FEED_STALE_THRESHOLD_S",
    )
    # --- Candidate B (M8): microprice-deviation z-score widen ----------
    # When the runtime feed is live + coverage_valid + fresh, the bot
    # adds an asymmetric widen sourced from the recorder's
    # ``microprice_dev_z_24h`` (a TRUE z-score: signed microprice
    # deviation standardised over the recorder's long-horizon EWMA,
    # clamped to ±10 on the wire). Sign convention matches
    # ``ob_imbalance_ewma``: positive z = microprice above mid =
    # bid-heavy / up-pressure → ASK is thin → widen ASK; negative z →
    # widen BID. Gated by REGIME_USE_RUNTIME_RECORDER_FEED above; when the
    # feed yields None / coverage_valid=0 / stale / NaN the contribution
    # is (0,0) and the in-process microprice (OB-imbalance) gate carries
    # the load unchanged. See app/microprice_gate.py::microprice_z_widening_bps.
    #
    # Threshold in z-score units (standard deviations). 1.0 = "one SD
    # dislocation". For a roughly-normal signal |z|>=1.0 fires on ~32% of
    # ticks — comfortably above the G7 acceptance floor (>=5%) while still
    # selective. Operator iterates against live microprice_dev_z_24h.
    microprice_z_widen_threshold: float = Field(
        default=1.0,
        ge=0.0,
        alias="MICROPRICE_Z_WIDEN_THRESHOLD",
    )
    # Widen magnitude (bps) applied to the threatened side when the z-gate
    # fires. Deliberately NON-sentinel (unlike the -1.0 → full-cap default
    # the binary OB-imbalance / vol_trend / etc. gates use): the z-gate
    # fires far more often than those rare binary gates, so a full-cap
    # default would push the bot effectively dark on the threatened side
    # ~a third of the time the moment the master knob is flipped on. A
    # modest additive widen is the safe demonstration default (precedent:
    # slow_trend 10bp, inventory_drift 15bp are also live-from-day-one
    # non-sentinel defaults). Operator iterates UP toward the cap if the
    # signal proves valuable. -1.0 still falls back to the cap for anyone
    # who wants gate-equivalent magnitude.
    microprice_z_widen_bps: float = Field(
        default=4.0,
        alias="MICROPRICE_Z_WIDEN_BPS",
    )

    # Markout-tier size scaler (1.2.3). Shrinks order sizes
    # directly based on rolling-median 5s markout — same signal
    # the Market-tab "markout" tier label reads. Bypasses the
    # toxicity composite score so dashboard tier and bot behavior
    # align: heavy-adverse markout gets small orders even when
    # toxicity_score has not crossed its 0.45 trigger. Composes
    # multiplicatively with the toxicity-driven size_mult — bot
    # takes the MIN. See ``app/markout_size_scaler.py``.
    markout_size_scaler_enabled: bool = Field(
        default=True, alias="MARKOUT_SIZE_SCALER_ENABLED"
    )
    markout_size_scaler_mild_threshold_bps: float = Field(
        default=0.0, alias="MARKOUT_SIZE_SCALER_MILD_THRESHOLD_BPS"
    )
    markout_size_scaler_moderate_threshold_bps: float = Field(
        default=-1.0, alias="MARKOUT_SIZE_SCALER_MODERATE_THRESHOLD_BPS"
    )
    markout_size_scaler_heavy_threshold_bps: float = Field(
        default=-3.0, alias="MARKOUT_SIZE_SCALER_HEAVY_THRESHOLD_BPS"
    )
    markout_size_scaler_mild_mult: float = Field(
        default=0.85, ge=0.0, le=1.0, alias="MARKOUT_SIZE_SCALER_MILD_MULT"
    )
    markout_size_scaler_moderate_mult: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="MARKOUT_SIZE_SCALER_MODERATE_MULT"
    )
    markout_size_scaler_heavy_mult: float = Field(
        default=0.25, ge=0.0, le=1.0, alias="MARKOUT_SIZE_SCALER_HEAVY_MULT"
    )
    # Per-side adverse markout pause: when recent fills on one side average worse than
    # ``-ADVERSE_SIDE_PAUSE_SOFT_BPS`` (in bps), suppress passive placements on that side
    # for ``ADVERSE_SIDE_PAUSE_SECONDS``. This is a *localized* feedback loop on top of
    # the global hard_trigger/soft_trigger pipeline — in the snap_20260417_183547 session
    # all 3 fills averaged -1.26 bps markout (adverse) but neither the hard (-12 bps) nor
    # soft (-3 bps) thresholds fired, so the bot kept quoting the toxic side without any
    # adaptive response. With default soft threshold 2.0 bps and pause 15 s, a short run
    # of adverse fills on one side yields a temporary withdrawal; the other side remains
    # tradable. Set pause seconds to 0 to disable.
    adverse_side_pause_soft_bps: float = Field(
        default=2.0,
        ge=0.0,
        alias="ADVERSE_SIDE_PAUSE_SOFT_BPS",
    )
    adverse_side_pause_seconds: float = Field(
        default=15.0,
        ge=0.0,
        alias="ADVERSE_SIDE_PAUSE_SECONDS",
    )
    adverse_side_pause_min_fills: int = Field(
        default=2,
        ge=1,
        le=20,
        alias="ADVERSE_SIDE_PAUSE_MIN_FILLS",
    )
    # v1.4.152 Phase 2K.2 — favorable-exit predicate for the per-
    # side adverse pause. Instead of waiting the full
    # ``adverse_side_pause_seconds`` deadline regardless of market
    # state, the pause clears EARLY when the per-side rolling avg
    # markout recovers past
    #     -adverse_side_pause_soft_bps × adverse_side_pause_clear_threshold_mult
    # (i.e. the markout has recovered ``mult`` of the way from
    # `-soft` back to zero). Default 0.5 = clear when markout is
    # halfway recovered. Setting this to 0 disables the early-exit
    # (pure timer mode); setting it to 1.0 makes the pause clear
    # the moment markout goes above zero. The pause seconds remain
    # as a MAX-ceiling safety net for static-markout cases.
    adverse_side_pause_clear_threshold_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="ADVERSE_SIDE_PAUSE_CLEAR_THRESHOLD_MULT",
    )
    # BUG-010: when the adverse-side pause is armed AND the suppressed side is the
    # inventory-reducing side (BUY when short / SELL when long), the bot would
    # otherwise be stranded with directional inventory — the very side needed to
    # reduce it is blocked. Default ON: reducing-side placements bypass the pause.
    # Set to False to restore the pre-fix behaviour (full pause regardless of inventory).
    adverse_side_pause_bypass_reducing_side: bool = Field(
        default=True,
        alias="ADVERSE_SIDE_PAUSE_BYPASS_REDUCING_SIDE",
    )
    # v1.4.60 wedge-elimination Phase 5: mirror BUG-010's bypass on
    # the ``post_only_cross_cooldown`` so the reducing side is never
    # fully suppressed by defensive cooldowns while inventory exists.
    #
    # Snapshot v1.4.59-260518-173229 caught the bot wedged for 90+ s
    # with adverse_side_pause blocking BUY (adding side) AND
    # post_only_cross_cooldown blocking SELL (reducing side). The
    # bot couldn't reduce the +1 long because the only path to
    # flatten was a side blocked by the cooldown.
    #
    # The cost trade-off: when the cooldown is armed (6 events in a
    # 3-minute session), the venue had recently rejected a SELL with
    # 51604 "post-only would cross". Bypassing the cooldown for the
    # reducer means we re-try the SELL right away — risking another
    # 51604. That's CHEAP compared to "bot can't flatten its
    # inventory for 5+ seconds per cooldown arm".
    #
    # Default ON. Same conceptual rule as
    # ``adverse_side_pause_bypass_reducing_side`` — both protections
    # together close the v1.4.59 wedge class.
    post_only_cross_cooldown_bypass_reducing_side: bool = Field(
        default=True,
        alias="POST_ONLY_CROSS_COOLDOWN_BYPASS_REDUCING_SIDE",
    )
    # 2026-05-12 codex-#2: separate maximum-age cap for AT-TOUCH orders.
    # The regular ``QUOTE_AGING_MAX_AGE_SECONDS`` skips at-touch quotes
    # entirely (queue preservation) — they can sit indefinitely until
    # the market walks past them. This second cap fires hard-reprice
    # specifically for at-touch quotes that exceed this longer window,
    # catching pathological "30+ second stale at-touch" cases without
    # sacrificing the normal queue-preservation behaviour.
    # Default 0.0 = DISABLED (preserves pre-1.2.53 behaviour). Set to
    # 30-60 s to engage. Should be at least 5x the regular
    # ``QUOTE_AGING_MAX_AGE_SECONDS`` so it only fires on genuinely
    # stale at-touch orders.
    at_touch_max_age_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=600.0,
        alias="AT_TOUCH_MAX_AGE_SECONDS",
    )
    # 2026-05-13 todo-019 Part C: BEHIND-touch maximum-age cap.
    # The regular ``QUOTE_AGING_MAX_AGE_SECONDS`` already drives
    # tightening + hard-reprice for behind-touch orders, but at the
    # window we typically configure (6 s on TON) it doesn't catch
    # the 0.85-1.75 s adverse-selection window where the worst BUY
    # fills clustered in snapshot ``260513-073423-colo`` (-45/-44/
    # -27/-19/-15 bp behind-touch BUYs aged 850-3390 ms, hit during
    # fast down-moves of -4 bp / 100 ms).
    # This is a SECOND, stricter cap that fires hard-reprice on
    # any behind-touch order whose age exceeds the threshold —
    # independent of the regular age cap. Mirrors
    # ``at_touch_max_age_seconds`` but with the at/not-at gate
    # inverted.
    # Default 0.0 = DISABLED. Set to 1.5-3.0 s on tight-tick
    # symbols where the dominant fill pattern is behind-touch.
    behind_touch_max_age_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=600.0,
        alias="BEHIND_TOUCH_MAX_AGE_SECONDS",
    )
    # 2026-05-12 codex-#1: cancel resting orders when quote eligibility
    # is HOLD_ALL. Before this knob, HOLD_ALL just stopped placing new
    # quotes but left existing ones in the book — and ~10 % of fills
    # in snapshots arrived during HOLD_ALL state. With the knob ON
    # (default), every HOLD_ALL cycle triggers a cancel-resting unless
    # the only reason in ``quote_eligibility_reason`` is in the
    # ``KEEP_RESTING`` allow-list below. Set to false to restore the
    # pre-fix behaviour (not recommended).
    cancel_resting_on_hold_all: bool = Field(
        default=True,
        alias="CANCEL_RESTING_ON_HOLD_ALL",
    )
    # Reasons where HOLD_ALL is brief / transient and the existing
    # orders are likely still fair-priced. Comma-separated list. The
    # canonical case is ``recovery_cooldown`` — a sub-second pause
    # after a fill before re-placing on the just-filled side. Cancel
    # there would just churn the order that's about to be re-placed.
    hold_all_keep_resting_reasons: str = Field(
        default="recovery_cooldown",
        alias="HOLD_ALL_KEEP_RESTING_REASONS",
    )
    # 2026-05-13 todo-019 Part A: NO_QUOTE cancel-policy inversion.
    # Pre-1.2.79 ``_should_cancel_resting_on_no_quote`` used an
    # opt-in severe-data-only whitelist (stale_data_warn etc.) —
    # any other NO_QUOTE reason (toxicity gate, vol regime,
    # drawdown gate, etc.) left existing orders resting while the
    # engine refused to add new ones. Codex's review #1 flagged
    # this as a dominant source of stale-quote toxic fills; the
    # HOLD_ALL half was inverted in v1.2.74, this is the NO_QUOTE
    # half.
    # Behavior: every NO_QUOTE cycle now cancels resting orders by
    # default. Only when every reason in ``risk.reasons`` is in
    # this allow-list does the bot keep resting. Empty/missing
    # reasons → cancel defensively.
    # Canonical safe-hold reasons: ``recovery_cooldown`` (sub-
    # second post-fill pause) and ``trade_rate_limit`` (about to
    # re-emit anyway). Expand iteratively if false-positive
    # cancel-churn is observed.
    no_quote_keep_resting_reasons: str = Field(
        default="recovery_cooldown,trade_rate_limit",
        alias="NO_QUOTE_KEEP_RESTING_REASONS",
    )
    # todo-011: post-fill replace cooldown for the predator window.
    # After a fill on side X (BUY/SELL), suppress new placements on
    # THAT side for this many milliseconds. Intent: dodge the 0-100 ms
    # "predator window" where freshly placed quotes are filled by
    # informed flow at ~85 % adverse-selection rate (overnight 2026-
    # 05-12 snapshot showed -10 bp median markout in this bucket vs
    # -2 bp baseline). The cooldown is PER-SIDE — a BUY fill doesn't
    # affect ASK placements and vice versa. Defaults to 0 (DISABLED)
    # so existing deploys are unchanged; opt in by setting to e.g.
    # 150 ms in the profile env. See ``BUGS/todo-011.md`` for the
    # test plan and expected impact.
    post_fill_replace_cooldown_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=5000.0,
        alias="POST_FILL_REPLACE_COOLDOWN_MS",
    )
    # 2026-05-12 codex-#1 narrow: at-touch adverse pause. When the
    # rolling median 5 s markout of recent AT-TOUCH fills on a given
    # side drops below the threshold (over ``MIN_FILLS`` samples),
    # pause the whole side for ``PAUSE_SECONDS``. Focuses on the
    # specific failure mode where joining the touch is consistently
    # adversely selected. Defaults to 0 bps threshold = DISABLED.
    # Suggested first trial: ``-5.0`` bps + 30 s pause + 3 min fills.
    # See ``app/at_touch_adverse_pause.py`` for mechanics.
    at_touch_adverse_pause_threshold_bps: float = Field(
        default=0.0,
        le=0.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_THRESHOLD_BPS",
    )
    at_touch_adverse_pause_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_SECONDS",
    )
    at_touch_adverse_pause_min_fills: int = Field(
        default=3,
        ge=1,
        le=50,
        alias="AT_TOUCH_ADVERSE_PAUSE_MIN_FILLS",
    )
    # Phase 2K.6 (v1.4.156) — favorable-exit predicate for the at-touch
    # adverse pause. The ``AT_TOUCH_ADVERSE_PAUSE_SECONDS`` timer is the
    # MAX-cooldown ceiling; the gate additionally clears EARLY when the
    # per-side median markout has recovered above
    # ``threshold_bps × clear_band_mult`` (e.g., threshold=-5 and
    # mult=0.5 → clear band -2.5 bps) and held there for
    # ``favorable_exit_dwell_seconds``. clear_band_mult ∈ [0, 1]:
    # smaller mult → stronger recovery required (0 = median must turn
    # positive; 1 = median just above trigger boundary). Set
    # ``ENABLED=False`` to fall back to legacy pure-timer behaviour.
    at_touch_adverse_pause_favorable_exit_enabled: bool = Field(
        default=True,
        alias="AT_TOUCH_ADVERSE_PAUSE_FAVORABLE_EXIT_ENABLED",
    )
    at_touch_adverse_pause_clear_band_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_CLEAR_BAND_MULT",
    )
    at_touch_adverse_pause_favorable_exit_dwell_seconds: float = Field(
        default=5.0,
        ge=0.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_FAVORABLE_EXIT_DWELL_SECONDS",
    )
    # v1.5.157 — position-aware favorable-exit threshold for
    # at_touch_adverse_pause. Per CLAUDE.md Rule 0c. Same template
    # as realised_edge_side_suppress's
    # ``..._position_favorable_inventory_threshold``.
    at_touch_adverse_pause_position_favorable_inventory_threshold: float = Field(
        default=1.0,
        ge=0.0,
        alias="AT_TOUCH_ADVERSE_PAUSE_POSITION_FAVORABLE_INVENTORY_THRESHOLD",
    )
    # Phase 4C.3 mini (v1.4.161) — per-side realised-edge side suppression.
    # Out-of-order pre-release of the maturity-doc's Phase 4C arc. When the
    # per-side rolling N-fill mean of (markout_5s_bps + rebate_bps) drops
    # below ``REALISED_EDGE_SUPPRESS_THRESHOLD_BPS`` (a negative number),
    # suppress placements on that side for ``COOLDOWN_SECONDS``.
    # Phase 2K-style favorable-exit: clears early when the same trailing
    # mean recovers past ``threshold × CLEAR_BAND_MULT`` for ``DWELL_SECONDS``.
    # Targets the failure mode in snapshot v1.4.157-260520-213540 where the
    # SELL side was systematically picked off as TON rallied — the trailing
    # mean would have crossed -4 bps several seconds before util thresholds
    # tripped on the existing position-aware gates. Default 0.0 = disabled.
    # See ``app/realised_edge_side_suppress.py``.
    realised_edge_suppress_threshold_bps: float = Field(
        default=0.0,
        le=0.0,
        alias="REALISED_EDGE_SUPPRESS_THRESHOLD_BPS",
    )
    realised_edge_suppress_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="REALISED_EDGE_SUPPRESS_COOLDOWN_SECONDS",
    )
    realised_edge_suppress_min_fills: int = Field(
        default=4,
        ge=1,
        le=50,
        alias="REALISED_EDGE_SUPPRESS_MIN_FILLS",
    )
    realised_edge_suppress_favorable_exit_enabled: bool = Field(
        default=True,
        alias="REALISED_EDGE_SUPPRESS_FAVORABLE_EXIT_ENABLED",
    )
    realised_edge_suppress_clear_band_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="REALISED_EDGE_SUPPRESS_CLEAR_BAND_MULT",
    )
    realised_edge_suppress_favorable_exit_dwell_seconds: float = Field(
        default=10.0,
        ge=0.0,
        alias="REALISED_EDGE_SUPPRESS_FAVORABLE_EXIT_DWELL_SECONDS",
    )
    # v1.5.155 — position-aware favorable-exit threshold for
    # realised_edge_side_suppress. Per CLAUDE.md Rule 0c: when a per-
    # side suppression is the ONLY way to reduce current adverse
    # inventory, clear it immediately. Operator's "reducing side
    # must always be available" principle.
    #
    # Predicate (per side):
    #   BUY suppression clears when position_qty <= -threshold
    #     (bot is meaningfully SHORT; BUY reduces SHORT)
    #   SELL suppression clears when position_qty >= +threshold
    #     (bot is meaningfully LONG; SELL reduces LONG)
    #
    # Default 1.0 contract — small enough to fire whenever the bot
    # has actual inventory exposure, large enough to ignore
    # sub-tick residuals.
    realised_edge_suppress_position_favorable_inventory_threshold: float = Field(
        default=1.0,
        ge=0.0,
        alias="REALISED_EDGE_SUPPRESS_POSITION_FAVORABLE_INVENTORY_THRESHOLD",
    )
    # Phase 4C.1 + 4C.2 (v1.4.164) — expected-net-edge side
    # suppression. When the per-side ``target_half_spread_bps +
    # maker_rebate_bps - typical_adverse_markout_bps`` drops below
    # ``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE`` (a small negative number),
    # refuse to quote that side. Hysteresis: a refused side does not
    # re-arm until its expected edge is > MIN + recovery margin for
    # ``HYSTERESIS_TICKS`` consecutive ticks.
    #
    # Default 0.0 = DISABLED (no refusal ever fires) so this PR is
    # safe to ship without immediate operator tuning. To enable, set
    # a small negative value such as -1.0 bp (refuse only when expected
    # edge is at least 1 bp underwater).
    #
    # Maker rebate + typical adverse markout priors reuse the existing
    # ``OBSERVABILITY_*`` settings (originally introduced for the
    # ``expected_net_edge_bps_at_decision`` stamp). See
    # ``app/expected_edge.py``.
    min_expected_net_edge_bps_per_side: float = Field(
        default=0.0,
        le=0.0,
        alias="MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE",
    )
    expected_edge_recovery_margin_bps: float = Field(
        default=0.2,
        ge=0.0,
        alias="EXPECTED_EDGE_RECOVERY_MARGIN_BPS",
    )
    expected_edge_hysteresis_ticks: int = Field(
        default=3,
        ge=1,
        le=100,
        alias="EXPECTED_EDGE_HYSTERESIS_TICKS",
    )
    # Phase 4C.2.a (v1.5.146) — dampen band on the economics gate.
    # When the adjusted expected edge falls in the half-open band
    # ``(MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE, EXPECTED_EDGE_DAMPEN_MAX_BPS]``
    # the bot KEEPS QUOTING that side but adds ``EXPECTED_EDGE_DAMPEN_WIDEN_BPS``
    # to its half-spread instead of refusing. The refuse band
    # (below MIN) still takes precedence — the dampen helper returns
    # zero widening when the refuse band would fire.
    #
    # Default DAMPEN_WIDEN_BPS=0.0 keeps the feature DORMANT — set
    # to a positive value (e.g. 2.0 bp) per profile to enable. The
    # dampen ceiling defaults to 0.0 bp so the band is exactly the
    # negative range above the refuse threshold; raise it to allow
    # dampening on slightly-positive economics too (rare; usually
    # leave at 0.0).
    expected_edge_dampen_max_bps: float = Field(
        default=0.0,
        alias="EXPECTED_EDGE_DAMPEN_MAX_BPS",
    )
    expected_edge_dampen_widen_bps: float = Field(
        default=0.0,
        ge=0.0,
        alias="EXPECTED_EDGE_DAMPEN_WIDEN_BPS",
    )
    # v1.5.205 Phase 4C.4 — stale-resting-quote adverse-selection
    # penalty. Coefficient (bps per second of decision-time quote age
    # ABOVE the rolling-window P50). Computed as:
    #
    #     penalty_bps = max(0, age_now - P50_age) × this_coefficient
    #
    # Subtracted from the per-side expected net edge in
    # ``compute_expected_net_edge_bps`` before the refuse/dampen
    # band evaluator compares against thresholds. Older quotes →
    # bigger penalty → side more likely to be refused or dampened.
    #
    # Default ``1.0`` is calibrated for TON where P99 quote_age_at_fill
    # ≈ 2.4 s (from v1.5.200 snapshot data): a P99 fill would lose
    # ~1.3 bp from its expected edge, large enough to matter against
    # the -2 bp typical adverse markout but small enough not to nuke
    # participation. Set to ``0.0`` to disable the feature (preserves
    # pre-v1.5.205 behaviour byte-for-byte).
    #
    # Sister knob ``min_expected_net_edge_bps_per_side`` controls the
    # refuse threshold this feeds into; without that being negative
    # the gate is dormant regardless of this penalty.
    stale_risk_penalty_bps_per_sec: float = Field(
        default=1.0,
        ge=0.0,
        alias="STALE_RISK_PENALTY_BPS_PER_SEC",
    )
    # v1.5.207 Phase 4C.5 — participation score thresholds. See
    # ``app/participation_score.py`` for the score formula. Defaults
    # are intentionally conservative so the score is observability-
    # only in v1.5.207 (the existing refuse/dampen gates still drive
    # behavior). Operator audits ``participation_score_disagreement_
    # {bid,ask}_total`` to see how often the score's action differs
    # from the gates' decision; a future release can flip the
    # behavior driver once those counters look calibrated.
    participation_score_full_edge_bps: float = Field(
        default=3.0,
        ge=0.0,
        alias="PARTICIPATION_SCORE_FULL_EDGE_BPS",
    )
    participation_score_soft_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="PARTICIPATION_SCORE_SOFT_THRESHOLD",
    )
    participation_score_hard_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        alias="PARTICIPATION_SCORE_HARD_THRESHOLD",
    )
    # Phase 4C.3 (v1.5.41) — Side-specific historical edge as quote-
    # decision input. Maintains a rolling 1 h per-side trailing
    # mean+stdev of ``markout_5s_bps + rebate_bps`` and feeds the
    # z-score back into the expected-edge gate as a multiplicative
    # confidence factor:
    #
    #     adjusted_edge = expected_edge × clamp(
    #         1.0 + ZSCORE_COEFF × z_score,
    #         MULT_FLOOR,
    #         MULT_CEIL,
    #     )
    #
    # where ``z_score = trailing_mean / trailing_stdev`` (distance
    # from break-even in stdev units of recent realised edge).
    #
    # When the bot's recent realised edge on side X has degraded
    # relative to its own variability, ``adjusted_edge`` shrinks
    # → the refusal gate (4C.1+4C.2) triggers sooner. When recent
    # edge is strong, the multiplier > 1 → refusal triggers later.
    #
    # Distinct from the existing short-window ``realised_edge_side_
    # suppress`` gate (4C.3-mini, v1.4.161) which uses a fixed-N
    # (20 fills) BINARY cooldown. This module is CONTINUOUS — scales
    # the input to the gate rather than firing its own cooldown.
    #
    # Default DISABLED. Enable by setting ENABLED=true plus tuning
    # WINDOW_SECONDS / MIN_SAMPLES per profile.
    side_edge_history_enabled: bool = Field(
        default=False,
        alias="SIDE_EDGE_HISTORY_ENABLED",
    )
    side_edge_history_window_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        alias="SIDE_EDGE_HISTORY_WINDOW_SECONDS",
    )
    side_edge_history_max_samples: int = Field(
        default=8192,
        ge=128,
        le=65536,
        alias="SIDE_EDGE_HISTORY_MAX_SAMPLES",
    )
    # Minimum samples before z-score is computed. Below this count
    # the multiplier defaults to 1.0 (no effect). Prevents a small
    # number of outlier fills from producing a high-variance stdev
    # that would generate noisy multipliers.
    side_edge_history_min_samples: int = Field(
        default=10,
        ge=2,
        le=1000,
        alias="SIDE_EDGE_HISTORY_MIN_SAMPLES",
    )
    # α in the formula. 0 = no effect; 0.5 = "z=-2 fully zeros out
    # the multiplier with mult_floor=0". Bounded to [0, 5] for
    # safety — values above ~1.0 produce very reactive multipliers
    # which would amplify noise.
    side_edge_history_zscore_coeff: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        alias="SIDE_EDGE_HISTORY_ZSCORE_COEFF",
    )
    # Multiplier clamps. Default [0.0, 2.0]:
    #   * floor 0 → very negative z can fully zero expected_edge
    #     (forces refusal when edge is bad and consistent).
    #   * ceil 2 → very positive z at most doubles expected_edge
    #     (caps amplification on small-stdev windows).
    side_edge_history_mult_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="SIDE_EDGE_HISTORY_MULT_FLOOR",
    )
    side_edge_history_mult_ceil: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        alias="SIDE_EDGE_HISTORY_MULT_CEIL",
    )
    # Phase 4D — Forced-flatten adaptive aggressiveness (v1.4.164).
    # Inserts cross-tick IOC ladders between the existing post-only
    # SF phases and the terminal ``market_close`` taker fallback.
    # Targets the v1.4.157-260520-213540 failure: 1,443 post-only
    # place-cancels in 14 s before the bot timed out into a single
    # market_close at the worst moment of the rally.
    #
    # Phase ladder (extending the existing 2-phase post-only model):
    #   0: post-only at near touch (rebate side) — existing behaviour
    #   1: post-only at far touch +/- 1 tick (current "phase 2")
    #   2: IOC limit crossing 1 tick (NEW)
    #   3: IOC limit crossing 2 ticks (NEW)
    #   4: market_close (existing terminal fallback)
    # Per-phase max-dwell defaults are below. Phase 4 is terminal — no
    # dwell budget.
    sf_phase_ladder_enabled: bool = Field(
        default=False,
        alias="SOFT_FLATTEN_PHASE_LADDER_ENABLED",
    )
    sf_phase_0_duration_seconds: float = Field(
        default=3.0,
        ge=0.0,
        alias="SOFT_FLATTEN_PHASE_0_DURATION_S",
    )
    sf_phase_1_duration_seconds: float = Field(
        default=4.0,
        ge=0.0,
        alias="SOFT_FLATTEN_PHASE_1_DURATION_S",
    )
    sf_phase_2_duration_seconds: float = Field(
        default=4.0,
        ge=0.0,
        alias="SOFT_FLATTEN_PHASE_2_DURATION_S",
    )
    sf_phase_3_duration_seconds: float = Field(
        default=2.0,
        ge=0.0,
        alias="SOFT_FLATTEN_PHASE_3_DURATION_S",
    )
    # Skip-ahead heuristic: when the live touch has moved >= this many
    # ticks past the SF entry mid (in the adverse direction for our
    # remaining inventory), jump directly to phase 2.
    sf_fast_escalate_ticks: float = Field(
        default=3.0,
        ge=0.0,
        alias="SOFT_FLATTEN_FAST_ESCALATE_TICKS",
    )
    # Skip-ahead heuristic: when this many consecutive post-only
    # cancels happen with no fills, jump to the next phase. Detects
    # the "venue is rejecting every post-only" pattern.
    sf_consecutive_rejects_to_escalate: int = Field(
        default=10,
        ge=1,
        le=1000,
        alias="SOFT_FLATTEN_CONSECUTIVE_REJECTS_TO_ESCALATE",
    )
    # v1.5.198 — episode hard-timeout. The absolute cap on total
    # wall time a single SF episode can run. When the episode has
    # been active for this many seconds (regardless of phase, regardless
    # of per-phase budget resets), the phase ladder force-jumps to
    # PHASE_4_MARKET (market_close terminal). Anchored to the original
    # SF entry, NOT reset on phase transitions. Default 60 s — generous
    # enough that a healthy SF episode completes within budget, strict
    # enough to prevent the 30-min pause observed in
    # v1.5.195-260527-165258.
    soft_flatten_episode_max_duration_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="SOFT_FLATTEN_EPISODE_MAX_DURATION_SECONDS",
    )
    # v1.5.198 — re-entry cooldown. After an SF episode exits, refuse
    # to re-enter SF for this many seconds even if the trigger fires
    # again. Prevents the toxicity-hard SF loop pattern observed in
    # v1.5.195-260527-161959 (4708 SF starts in 3 min, each cycle
    # ~25ms). The cooldown is intentionally generous (30s) — a real
    # catastrophic event will still be handled by the FIRST SF episode;
    # consecutive triggers within the cooldown window are
    # statistically redundant (same underlying signal). Set to 0 to
    # disable.
    soft_flatten_reentry_cooldown_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="SOFT_FLATTEN_REENTRY_COOLDOWN_SECONDS",
    )
    # v1.5.202 — SF-event-count fatigue ladder. Orthogonal to
    # session_drawdown (which keys on PnL) — this counts SF episode
    # entries in a rolling window, so storm-cluster SF patterns
    # trigger a brake even when no single episode hit the PnL tier.
    # See ``app/sf_fatigue_gate.py`` for tier semantics. Defaults
    # sized for TON: a healthy session has <2 SF/h; 3 SFs in a 30-min
    # window already signals an adverse regime worth WIDENing on.
    sf_fatigue_gate_enabled: bool = Field(
        default=True, alias="SF_FATIGUE_GATE_ENABLED",
    )
    sf_fatigue_window_seconds: float = Field(
        default=1800.0,
        ge=0.0,
        alias="SF_FATIGUE_WINDOW_SECONDS",
    )
    sf_fatigue_tier1_widen_count: int = Field(
        default=3,
        ge=0,
        alias="SF_FATIGUE_TIER1_WIDEN_COUNT",
    )
    sf_fatigue_tier2_pause_short_count: int = Field(
        default=5,
        ge=0,
        alias="SF_FATIGUE_TIER2_PAUSE_SHORT_COUNT",
    )
    sf_fatigue_tier3_pause_long_count: int = Field(
        default=8,
        ge=0,
        alias="SF_FATIGUE_TIER3_PAUSE_LONG_COUNT",
    )
    sf_fatigue_tier4_kill_count: int = Field(
        default=12,
        ge=0,
        alias="SF_FATIGUE_TIER4_KILL_COUNT",
    )
    sf_fatigue_pause_short_seconds: float = Field(
        default=600.0,
        ge=0.0,
        alias="SF_FATIGUE_PAUSE_SHORT_SECONDS",
    )
    sf_fatigue_pause_long_seconds: float = Field(
        default=1800.0,
        ge=0.0,
        alias="SF_FATIGUE_PAUSE_LONG_SECONDS",
    )
    # Multiplicative widening applied to the half-spread when tier =
    # WIDEN. 1.0 = no-op; 1.3 = 30% wider; higher values harder to
    # justify since at that point PAUSE_SHORT (next tier) is the
    # correct action.
    sf_fatigue_widen_multiplier: float = Field(
        default=1.3,
        ge=1.0,
        alias="SF_FATIGUE_WIDEN_MULTIPLIER",
    )
    # Phase 4E (v1.4.165) — TARGET-venue fast-move cancel.
    # Symmetric defensive cancel using the trading venue's OWN
    # ``mid_return_500ms_bps`` kinematic. Complementary to the legacy
    # ``BINANCE_CANCEL_ON_MOVE_BPS`` (which uses the REFERENCE venue
    # as a leading indicator). Default 0.0 = DISABLED (opt-in per
    # profile). Naming: ``TARGET_VENUE_*`` parallels the existing
    # ``REFERENCE_VENUE_*`` settings; no "OKX" / "Binance" /
    # exchange-specific names — the architecture supports any
    # target/reference pair. See ``app/fast_move_cancel.py``.
    #
    # Suggested first trial: 10.0 bps (catches OKX-led 500 ms moves
    # the reference-venue cancel-on-move at 6 bps does not catch
    # because Binance didn't lead). Lower to 6-8 for tighter
    # protection; raise to 15-20 if the cancel fires spuriously on
    # routine local-venue jitter.
    target_venue_cancel_on_move_bps: float = Field(
        default=0.0,
        ge=0.0,
        alias="TARGET_VENUE_CANCEL_ON_MOVE_BPS",
    )
    # v1.5.157 — drift-signal window selector for the target_venue
    # fast-move cancel gate. Maps approximately to the nearest window
    # in ``state.mid_drift_windows``:
    #
    #   <= 0.6  → state.last_mid_return_500ms_bps (legacy)
    #   <= 7.5  → mid_drift_windows.drift_5s_bps
    #   <= 20.0 → mid_drift_windows.drift_10s_bps
    #
    # Driving observation: v1.5.154-260526-074029 snapshot showed
    # cancel_bid_total=1026 vs cancel_ask_total=85 over the night
    # (12x asymmetric). The 500 ms signal is noisy enough that the
    # cancel gate fires on per-tick jitter rather than on sustained
    # moves. A longer window filters the noise but reacts slower —
    # trade-off operator decides per profile.
    #
    # Default 0.5 preserves legacy behaviour for back-compat. Bug-030
    # Mechanism B analysis (in issues/bug-030.md) notes this is NOT
    # by itself a fix for the asymmetry — the asymmetry is partly a
    # structural reflection of regime direction. v1.5.155 trend skew
    # is the load-bearing fix; this knob is a tuning lever for the
    # cancel rate magnitude.
    target_venue_fast_move_cancel_drift_window_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=20.0,
        alias="TARGET_VENUE_FAST_MOVE_CANCEL_DRIFT_WINDOW_SECONDS",
    )
    # Phase 2I (v1.4.169) — per-order amend rate defence. STRUCTURAL
    # guard against the snapshot v1.4.102-260520-120555 pattern:
    # a single SELL received 244 amend dispatches in ~1 s as the
    # bot's computed price ping-ponged between two adjacent ticks
    # (1.941 ↔ 1.942) every 5-7 ms quote cycle. Per-account aggregate
    # pacing showed plenty of headroom — the breach was per-ORDER only.
    #
    # Two complementary guards:
    #   * TICK_FLICKER_MIN_MS — minimum gap between consecutive amend
    #     dispatches on the SAME order. Default 250 ms. Stops the bot
    #     from generating an amend faster than the venue can ack the
    #     previous one.
    #   * PER_ORDER_MAX_PER_SEC — cap on amend dispatches per order
    #     per 1-second window. Default 8. Hard ceiling regardless of
    #     spacing.
    # Always-on (no enabled flag — these are structural). When a
    # guard suppresses, ``OrderManager._enqueue_amend_quote_path``
    # returns False (the dispatcher receives nothing); a counter +
    # an ``_record_orchestrate_decision`` breadcrumb capture the
    # suppression for postmortem.
    amend_tick_flicker_min_ms: float = Field(
        default=250.0,
        ge=0.0,
        alias="AMEND_TICK_FLICKER_MIN_MS",
    )
    amend_per_order_max_per_sec: int = Field(
        default=8,
        ge=1,
        le=200,
        alias="AMEND_PER_ORDER_MAX_PER_SEC",
    )
    # Phase 4F (v1.4.170) — elevated-vol auto-pause. When the realised
    # vol ratio (current / baseline, computed by the toxicity engine
    # as ``vol_spike_ratio``) stays above ``ARM_RATIO`` for
    # ``ARM_DWELL_SECONDS``, force the bot's eligibility to HOLD_ALL.
    # Clears when the ratio drops below ``CLEAR_RATIO`` for
    # ``CLEAR_DWELL_SECONDS``, OR at ``MAX_SECONDS`` (safety ceiling).
    # Existing defenses (regime_controller, vol_spike, adaptive_widen,
    # shock_gate, 4C side-refuse, 4E fast-move cancel) are SOFT —
    # widen / shrink / suppress one side. 4F is the MACRO defense
    # for when the regime itself is bad enough that quoting is
    # net-negative-edge — sit out entirely.
    #
    # Default ARM_RATIO=0.0 = DISABLED. Opt-in per profile.
    # Suggested first-trial: ARM_RATIO=2.0 (catches elevated regimes
    # without spurious arming on routine spikes), ARM_DWELL=60 s
    # (filters one-spike false positives), CLEAR_RATIO=1.3
    # (relaxed enough to allow gentle recovery), CLEAR_DWELL=120 s
    # (longer than arm — conservative exit), MAX=1800 s (30 min;
    # safety net so stale signal can't pause forever).
    vol_auto_pause_arm_ratio: float = Field(
        default=0.0,
        ge=0.0,
        alias="VOL_AUTO_PAUSE_ARM_RATIO",
    )
    vol_auto_pause_arm_dwell_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="VOL_AUTO_PAUSE_ARM_DWELL_SECONDS",
    )
    vol_auto_pause_clear_ratio: float = Field(
        default=1.3,
        ge=0.0,
        alias="VOL_AUTO_PAUSE_CLEAR_RATIO",
    )
    vol_auto_pause_clear_dwell_seconds: float = Field(
        default=120.0,
        ge=0.0,
        alias="VOL_AUTO_PAUSE_CLEAR_DWELL_SECONDS",
    )
    vol_auto_pause_max_seconds: float = Field(
        default=1800.0,
        ge=0.0,
        alias="VOL_AUTO_PAUSE_MAX_SECONDS",
    )
    # 2026-05-12 codex-#3: fill-burst detector. When N or more fills
    # cluster within ``WINDOW_SECONDS``, apply ``SIZE_MULT`` (default
    # 0.5) for ``COOLDOWN_SECONDS`` — defensive size shrink during
    # what appears to be a multi-fill adverse selection burst. Reacts
    # to the FILL signal directly, not waiting for the per-fill
    # markouts to resolve at 5 s. Composes with other size-mults via
    # ``min`` semantics. Defaults to threshold=0 = DISABLED. Suggested
    # first trial: threshold=3 fills in 30 s, mult 0.5, cooldown 60 s.
    # See ``app/fill_burst_detector.py``.
    fill_burst_threshold: int = Field(
        default=0,
        ge=0,
        le=100,
        alias="FILL_BURST_THRESHOLD",
    )
    fill_burst_window_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="FILL_BURST_WINDOW_SECONDS",
    )
    fill_burst_size_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="FILL_BURST_SIZE_MULT",
    )
    fill_burst_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="FILL_BURST_COOLDOWN_SECONDS",
    )
    # Phase 2K.9 (v1.4.160) — favorable-exit predicate for the
    # fill_burst_detector size-shrink cooldown. The
    # ``FILL_BURST_COOLDOWN_SECONDS`` timer is the MAX ceiling; the
    # cooldown also clears EARLY when the live fill-in-window count
    # has dropped below ``FILL_BURST_THRESHOLD × CLEAR_BAND_MULT``
    # (default mult=0.5 → half the trigger count; e.g. threshold=3
    # clears when live count < 1.5 i.e. 0-1 fills in window) and
    # held there for ``FAVORABLE_EXIT_DWELL_SECONDS`` (default 5 s).
    # Polled on every quote-loop tick (not just on note_fill) so the
    # "burst has stopped" silence is detected without needing a new
    # fill to evaluate. Set ``ENABLED=False`` for legacy behaviour.
    fill_burst_favorable_exit_enabled: bool = Field(
        default=True,
        alias="FILL_BURST_FAVORABLE_EXIT_ENABLED",
    )
    fill_burst_clear_band_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="FILL_BURST_CLEAR_BAND_MULT",
    )
    fill_burst_favorable_exit_dwell_seconds: float = Field(
        default=5.0,
        ge=0.0,
        alias="FILL_BURST_FAVORABLE_EXIT_DWELL_SECONDS",
    )
    # TODO-002: account-data-stale kill gate. If the bot has not seen a
    # successful ``refresh_account_only`` REST roundtrip for this many
    # seconds (default 60), risk.evaluate_risk returns NO_QUOTE/account_data_stale
    # which Bot._should_cancel_resting_on_no_quote promotes to cancel-resting.
    # Set to 0 to disable. Independent of public-WS / market-data freshness;
    # catches "REST stopped refreshing" cases (auth-token expiry, transient
    # network glitch on the account endpoint specifically).
    account_data_stale_kill_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="ACCOUNT_DATA_STALE_KILL_SECONDS",
    )
    # TODO-001: inventory consistency watchdog. Periodically compares
    # ``state.position.position_qty`` (venue truth, refreshed via
    # ``refresh_account_only``) against ``baseline + session_signed_qty_total``
    # (the bot's expectation from the seed + observed session fills). On
    # divergence beyond tolerance, cancels resting orders and arms manual
    # pause + Telegram CRITICAL alert. Catches BUG-005-class divergences
    # (where local state silently drifts from venue) BEFORE they accumulate.
    inventory_consistency_enabled: bool = Field(
        default=True,
        alias="INVENTORY_CONSISTENCY_ENABLED",
    )
    inventory_consistency_check_seconds: float = Field(
        default=60.0,
        ge=5.0,
        alias="INVENTORY_CONSISTENCY_CHECK_SECONDS",
    )
    # Tolerance: max(``_TOLERANCE_QTY``, ``_TOLERANCE_PCT * abs(expected)``)
    # so small books are not flagged on rounding noise and large books are
    # not flagged on tiny absolute drift. Defaults: 0.5 SUI / 1% of position.
    inventory_consistency_tolerance_qty: float = Field(
        default=0.5,
        ge=0.0,
        alias="INVENTORY_CONSISTENCY_TOLERANCE_QTY",
    )
    inventory_consistency_tolerance_pct: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        alias="INVENTORY_CONSISTENCY_TOLERANCE_PCT",
    )
    # Hardening: require N consecutive divergent reads before declaring a
    # real breach. A single bad ``fetch_position`` (transient REST hiccup,
    # mid-update read race) would otherwise lock the bot out for the
    # operator to manually rebase. With N=3 at the default 60s check
    # interval, we need ~3 minutes of *sustained* divergence — long
    # enough to rule out a single-sample glitch, short enough that a real
    # BUG-005-class bug is still caught well before fills accumulate.
    # Set to 1 to restore the pre-2026-04-26 single-shot behaviour.
    # Reproducer for the change: snap_20260426_103520, where a single
    # transient read of -8 SUI (real position -28) cost the operator
    # 1.5 hours of paused trading.
    inventory_consistency_consecutive_breaches_required: int = Field(
        default=3,
        ge=1,
        le=20,
        alias="INVENTORY_CONSISTENCY_CONSECUTIVE_BREACHES_REQUIRED",
    )
    # TODO-003: rolling net-edge-after-fees observability window.
    # Computed over the last ``NET_EDGE_WINDOW_FILLS`` session fills as
    # ``mean_markout_bps - mean_round_trip_fee_bps``. Surfaced in
    # ``status_flags_dict`` and on Telegram ``/status``. Observation only —
    # no gating until calibrated against historical data.
    net_edge_window_fills: int = Field(
        default=50,
        ge=5,
        le=500,
        alias="NET_EDGE_WINDOW_FILLS",
    )
    # Added to economic min half-spread while adaptive widen cooldown is active (0 = disabled).
    # Cooldown length reuses TOXICITY_COOLDOWN_SECONDS; armed from soft/hard toxicity, adverse
    # delayed markout, or one-sided fill ratio (same threshold as toxicity engine).
    adaptive_spread_adverse_overlay_half_spread_bps: float = Field(
        default=6.0,
        ge=0.0,
        alias="ADAPTIVE_SPREAD_ADVERSE_OVERLAY_HALF_SPREAD_BPS",
    )
    # Observational rolling window for GET /toxicity/current and snapshot toxicity_score (not strategy).
    runtime_toxicity_fill_window: int = Field(
        default=50,
        ge=1,
        le=500,
        alias="RUNTIME_TOXICITY_FILL_WINDOW",
    )
    # Rolling window of quote cycles for spread / two-sided telemetry (GET /state/current ``quote_quality``).
    quote_quality_window_samples: int = Field(
        default=400,
        ge=50,
        le=5000,
        alias="QUOTE_QUALITY_WINDOW_SAMPLES",
    )
    # Added to spread_floor overlay while adaptive widen cooldown is active (see quote_quality telemetry).
    quote_quality_overlay_half_spread_bps: float = Field(
        default=2.0,
        ge=0.0,
        le=30.0,
        alias="QUOTE_QUALITY_OVERLAY_HALF_SPREAD_BPS",
    )
    # Minimum quote cycles recorded before spread_widen_signal may arm from one-tick / markout telemetry.
    quote_quality_widen_min_cycles: int = Field(
        default=45,
        ge=15,
        le=2000,
        alias="QUOTE_QUALITY_WIDEN_MIN_CYCLES",
    )
    # Runtime guardrails for decision -> first successful place latency.
    execution_latency_warn_ms: float = Field(
        default=300.0,
        gt=0.0,
        alias="EXECUTION_LATENCY_WARN_MS",
    )
    execution_latency_degrade_ms: float = Field(
        default=500.0,
        gt=0.0,
        alias="EXECUTION_LATENCY_DEGRADE_MS",
    )
    execution_latency_window_samples: int = Field(
        default=20,
        ge=3,
        le=500,
        alias="EXECUTION_LATENCY_WINDOW_SAMPLES",
    )
    execution_latency_degrade_min_breaches: int = Field(
        default=5,
        ge=1,
        le=500,
        alias="EXECUTION_LATENCY_DEGRADE_MIN_BREACHES",
    )

    # Outbound trading path: micro-batch + WS-primary exchange actions + intent coalescing.
    # 1.4.0 cancel-prio Phase 0a: default lowered 6.0 → 0.0 (event-driven
    # dispatch). Same-side coalescing is already done at submit time
    # (``OutboundDispatchCoordinator.submit_{place,cancel}``), so the
    # 6 ms micro-batch window added no value but added 0-6 ms of idle
    # wait per flush cycle — the same magnitude as a full cancel RTT on
    # colo'd OKX (median ~5 ms). Setting > 0 still works for venues
    # where time-based batching matters (legacy GRVT codepaths).
    action_batch_interval_ms: float = Field(
        default=0.0,
        ge=0.0,
        alias="ACTION_BATCH_INTERVAL_MS",
    )
    action_max_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        alias="ACTION_MAX_BATCH_SIZE",
    )
    # 1.3.109 cancel-prio Phase 3: parallel cancel + place worker threads
    # in OutboundDispatchCoordinator. When True (default), two dedicated
    # threads drain the cancel + place lanes concurrently — cancel HTTP
    # no longer waits for a 5-50 ms place HTTP to complete on the SAME
    # worker thread. When False, the legacy single-worker path runs
    # (cancels and places sequentially per flush cycle). The flag exists
    # purely as a rollback lever for staged-release verification —
    # behavior at True matches the pre-Phase-3 contract apart from the
    # latency win, including same-side ordering (cross-lane defer guard
    # prevents place from racing ahead of an in-flight same-side cancel).
    outbound_cancel_worker_enabled: bool = Field(
        default=True,
        alias="OUTBOUND_CANCEL_WORKER_ENABLED",
    )
    action_ws_enabled: bool = Field(default=True, alias="ACTION_WS_ENABLED")
    action_http_fallback_enabled: bool = Field(default=True, alias="ACTION_HTTP_FALLBACK_ENABLED")
    action_ws_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        alias="ACTION_WS_TIMEOUT_SECONDS",
    )
    # Extra suppression vs REPRICE_THRESHOLD_BPS: skip cancel/replace if price within N ticks and
    # relative size change below ratio, and min interval not elapsed (0 disables that gate).
    action_reprice_epsilon_ticks: float = Field(
        default=0.0,
        ge=0.0,
        alias="ACTION_REPRICE_EPSILON_TICKS",
    )
    action_resize_epsilon_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="ACTION_RESIZE_EPSILON_RATIO",
    )
    action_min_replace_interval_ms: float = Field(
        default=0.0,
        ge=0.0,
        alias="ACTION_MIN_REPLACE_INTERVAL_MS",
    )

    flatten_on_kill: bool = Field(default=True, alias="FLATTEN_ON_KILL")
    # BUG-013: ``Bot.kill()`` self-exits with code 42 after this many
    # seconds when the kill reason is environmental (Bluefin WS stall,
    # account-data stale, etc.) so the process supervisor (systemd
    # ``Restart=on-failure`` / container manager) brings the process back
    # automatically. Reasons NOT in ``_AUTO_RESTART_KILL_REASONS``
    # (manual /kill, drawdown, session loss, desync) keep the bot
    # KILLED-alive indefinitely so the operator can investigate
    # before resuming. Set to 0 to disable the auto-restart entirely
    # (legacy behaviour). Default 5 s is enough for the CRITICAL
    # kill event to flush to Telegram and for the post-kill cancel-all
    # REST to land before exit.
    kill_auto_restart_grace_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=60.0,
        alias="KILL_AUTO_RESTART_GRACE_SECONDS",
    )

    # v1.5.152 — crash-snapshot dump on Bot.kill(). When the bot
    # decides to terminate (data-feed stale, drawdown breach, manual
    # kill, ...), it dumps a self-contained diagnostic bundle to
    # ``CRASH_SNAPSHOT_DIR/v<ver>-<YYYYMMDD-HHMMSS>-<reason>/``
    # BEFORE the optional auto-restart os._exit fires. The bundle
    # contains the same JSON files the operator's
    # ``ops.ps1 snapshot`` would produce, but written from inside
    # the bot's own process — so the data exists even if the HTTP
    # API dies seconds later.
    #
    # The default path is on persistent disk (NOT /tmp) so the
    # bundles survive host reboots. Falls back to ``./snapshots_crash/``
    # under the repo if the configured path can't be created.
    #
    # Set to empty string to disable the feature entirely. Default
    # path requires the operator to ``mkdir -p /var/lib/dtc-mm-as``
    # with appropriate ownership ONCE during initial colo setup;
    # the bot creates the leaf ``snapshots_crash/`` automatically.
    crash_snapshot_dir: str = Field(
        default="/var/lib/dtc-mm-as/snapshots_crash",
        alias="CRASH_SNAPSHOT_DIR",
    )
    # Number of bot_events rows to include in the crash bundle.
    # 500 covers ~5-10 minutes of activity at typical event rates;
    # enough to see the lead-up to the kill decision in detail.
    crash_snapshot_event_limit: int = Field(
        default=500,
        ge=10,
        le=10000,
        alias="CRASH_SNAPSHOT_EVENT_LIMIT",
    )
    # BUG-014: slippage cap (bps) used when ``client.market_close``
    # constructs the IOC market-order limit price. Bluefin's MARKET
    # endpoint treats limit_px as a hard slippage cap; a price clipped
    # to ``mark_price`` left the order non-marketable in fast-moving
    # regimes (snap_20260427_055140: 3 attempts × 125 s with no fills).
    # 100 bps = 1 % is permissive enough to fill through any realistic
    # depth at retail size while still rejecting catastrophic-fat-finger
    # fills if mark is briefly stale.
    market_close_slippage_bps: float = Field(
        default=100.0,
        gt=0.0,
        le=1000.0,
        alias="MARKET_CLOSE_SLIPPAGE_BPS",
    )
    # BUG-015: ``BluefinClient.market_close`` uses IOC LIMIT (not true
    # MARKET) by default because Bluefin's MARKET endpoint zeroes the
    # signed ``priceE9`` regardless of the caller's slippage cap, and
    # was silently no-op-ing in production (snap_20260427_095951).
    # IOC LIMIT at ``mark ± MARKET_CLOSE_SLIPPAGE_BPS`` crosses any
    # realistic depth while keeping a hard slippage cap.
    # Set to False to revert to true MARKET (broken on SUI-PERP, but
    # kept available as an escape hatch in case a future Bluefin
    # update fixes the MARKET path or the LIMIT path develops its own
    # quirk).
    bluefin_market_close_use_limit_ioc: bool = Field(
        default=True,
        alias="BLUEFIN_MARKET_CLOSE_USE_LIMIT_IOC",
    )
    cancel_all_on_startup: bool = Field(default=True, alias="CANCEL_ALL_ON_STARTUP")
    cancel_all_on_shutdown: bool = Field(default=True, alias="CANCEL_ALL_ON_SHUTDOWN")
    # Fast-start mode: during startup readiness (BotStatus.STARTING), skip historical fill replay
    # (REST recent fills + private WS isSnapshot userFills) so RUNNING depends only on current
    # account/position snapshot + current open-orders state (and cancel-all if configured).
    fast_start_skip_historical_fill_replay: bool = Field(
        default=False,
        alias="FAST_START_SKIP_HISTORICAL_FILL_REPLAY",
    )

    database_url: str = Field(default="sqlite:///./data/mm.db", alias="DATABASE_URL")
    sqlite_path: str = Field(default="data/mm.db", alias="SQLITE_PATH")

    # ----------------------------------------------------------------
    # POSITION-AWARE DRAWDOWN GATE
    # ----------------------------------------------------------------
    # Closes a stuck losing position via post-only orders before it
    # grows large enough to trip the absolute drawdown / session-loss
    # gates. Trigger: unrealized PnL on the position is ≥ N bps adverse
    # (relative to |position_notional|) and stays there for ≥ M seconds.
    # Action: enter SOFT_FLATTENING mode -- patient post-only-only
    # exit at the best price on the reduce side. No taker IOC is used.
    # Once flat the bot resumes RUNNING; no kill, no operator action
    # required.
    #
    # Sized for small-position, slow-drift situations the absolute
    # drawdown ($10 default) doesn't cover -- e.g. on a $24 position
    # a 50-bps adverse drift = $0.12 loss is well below absolute caps
    # but signals the strategy is stuck and should reduce.
    position_drawdown_gate_enabled: bool = Field(
        default=True,
        alias="POSITION_DRAWDOWN_GATE_ENABLED",
    )
    position_drawdown_gate_threshold_bps: float = Field(
        default=50.0,
        gt=0.0,
        alias="POSITION_DRAWDOWN_GATE_THRESHOLD_BPS",
    )
    position_drawdown_gate_duration_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="POSITION_DRAWDOWN_GATE_DURATION_SECONDS",
    )
    # Re-place threshold for soft-flatten orders. When best price on
    # the reduce side moves more than N ticks from the resting order's
    # price, cancel and re-place at the new best. 1 tick = re-place
    # any time the BBO moves; higher values reduce churn at the cost
    # of slightly stale resting prices.
    soft_flatten_reprice_ticks: int = Field(
        default=1,
        ge=1,
        alias="SOFT_FLATTEN_REPRICE_TICKS",
    )
    # Phase 4D.5 (v1.4.190) — SF action-rate throttle. The SF tick
    # worker runs on the main quote loop, which wakes on every public-
    # WS book update (~100 Hz on active markets). Without a throttle
    # the worker re-evaluates the cancel-and-replace decision on every
    # wake; on a flickering touch (price moves >= 1 tick repeatedly)
    # the worker cancels and replaces 80-100 times per second.
    # Observed in snapshot ``v1.4.180-260521-141619-prod.okx.ton.usdt.perp``:
    # SF#11171 placed 4,902 orders over 56 s (87 / sec sustained) to
    # produce 3 fills. Of those, ~85 of every 87 placements replaced
    # an order at the SAME price — pure churn from WS-driven wake
    # bursts. Mirrors the Phase 2I per-order amend defence (same
    # pathology, different code path).
    #
    # The throttle blocks the SF tick's cancel-replace and place
    # actions when less than ``SF_ACTION_MIN_GAP_MS`` ms have elapsed
    # since the last SF action. The "unwanted side cancel" path is
    # NOT throttled (safety-critical — leaving a stray opposite-side
    # order during SF risks both sides filling and unwinding the
    # close). Set to 0 to disable.
    #
    # Default 250 ms = 4 SF actions / sec maximum. Reduces SF order
    # traffic by ~50× vs the unthrottled 100 Hz pattern with no
    # observable quality cost in the v1.4.180 snapshot (the 3 fills
    # all occurred at the touch independently of the surrounding
    # 4,899 churning place-cancels).
    sf_action_min_gap_ms: float = Field(
        default=250.0,
        ge=0.0,
        alias="SF_ACTION_MIN_GAP_MS",
    )
    # v1.4.192 — SF event-id grace cache window. After
    # ``_exit_soft_flatten`` runs, the fill-ingest path still treats
    # the just-exited SF event id as "valid" for this many seconds so
    # late-arriving WS fills (esp. from the taker paths) get tagged
    # correctly. Default 30 s = well beyond any observed private-WS
    # fill latency (sub-second on OKX), well below typical SF-re-entry
    # cadence (minutes). Set to 0 to disable the grace path
    # (reverts to v1.4.163's structurally-racy behaviour).
    sf_recent_event_id_grace_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="SF_RECENT_EVENT_ID_GRACE_SECONDS",
    )
    # Phase 4D.3 (v1.5.146) — slice the SF close across multiple
    # phase-2/3 IOC sends instead of firing for the whole remaining
    # quantity at once. The per-tick dispatcher already runs on the
    # quote loop; capping each IOC at a notional slice means each
    # slice gets a fresh touch observation, and a Binance pull-back
    # between slices can move the next IOC's limit a tick or two
    # closer to passive — the cross-venue spread saving the plan
    # describes.
    #
    # Default 0.0 = DISABLED. Each phase-2/3 IOC still fires for the
    # full remaining quantity, gated only by the existing
    # ``MAX_ORDER_NOTIONAL_USD`` / ``MAX_ABS_POSITION`` /
    # ``MAX_POSITION_NOTIONAL_USD`` caps (which apply to ALL paths,
    # passive and IOC alike). Set to a positive USD value (e.g. 5.0)
    # to enable slicing — each IOC is then capped at
    # ``slice_notional / target_price`` contracts. A smaller value
    # produces more slices and more chances for the touch to move
    # favourably between them, at the cost of more IOC dispatches
    # (latency + venue rate-limit pressure).
    #
    # ONLY affects phase 2/3 IOC dispatch. Phase 0/1 post-only,
    # phase 4 terminal market-close, and the legacy SF tick path
    # are unchanged (post-only IS its own form of slicing; market
    # close MUST be all-at-once to flatten the residual; legacy
    # path doesn't have a phase concept).
    soft_flatten_slice_notional_usd: float = Field(
        default=0.0,
        ge=0.0,
        alias="SOFT_FLATTEN_SLICE_NOTIONAL_USD",
    )
    # Staged pricing: phase 1 sits at the near-touch (best on the
    # reduce side; passive maker, joining the queue). After this many
    # seconds with no fill, advance to phase 2 -- one tick from the
    # FAR touch into the spread (i.e. the most aggressive post-only
    # price possible: best_ask-1 for BUY, best_bid+1 for SELL).
    # On a 1-tick spread the two phases collapse to the same level.
    # No further escalation: bot stays at phase 2 until filled. The
    # absolute drawdown / session-loss gates remain the hard floor
    # if the price never returns to our level.
    soft_flatten_phase_1_seconds: float = Field(
        default=5.0,
        ge=0.0,
        alias="SOFT_FLATTEN_PHASE_1_SECONDS",
    )
    # Phase-3 escape: when set > 0, the soft-flatten worker tracks
    # adverse mid drift (in ticks) since SF entry. If drift exceeds
    # this many ticks, the worker calls ``client.market_close`` once
    # and exits SF — accepting taker fees as the lesser cost vs. an
    # ever-widening drift. ``0`` disables (post-only forever, the
    # legacy behaviour).
    #
    # Recommended setting (TON, group-2 fees, ~3.5 bp natural
    # spread): 5 ticks ≈ 1.75 bp of adverse drift, comfortably
    # below the ~7-tick taker break-even, with margin for the
    # implicit "if it's drifted this far, P(continues) > P(reverses)".
    # See plans/20260507-sf-frontend.md Phase 6 for the math.
    #
    # This is the operator's GLOBAL ceiling; the per-entry trigger
    # (toxicity / drawdown gate) can request a tighter value but
    # not a looser one. Default 0 keeps existing post-only-only
    # behaviour for callers that don't opt in.
    soft_flatten_taker_fallback_ticks: int = Field(
        default=0,
        ge=0,
        alias="SOFT_FLATTEN_TAKER_FALLBACK_TICKS",
    )

    # v1.5.33 — take-profit (TP) opportunistic harvest mode.
    #
    # When uPnL on the open position rises to ``UPNL_HARVEST_TRIGGER_BPS``
    # bps, the bot enters TP mode: cancels the normal ladder and
    # places a single aggressive post-only close order at one tick
    # inside the far touch (most aggressive maker price possible).
    # Disarms via hysteresis (uPnL falls to ``trigger - disarm_margin``),
    # timeout (``max_dwell_seconds`` without a fill), or position-flat.
    # After disarm, a cooldown of ``arm_cooldown_seconds`` prevents
    # ping-ponging into TP near the threshold.
    #
    # Design note: TP NEVER escalates past post-only. If price escapes,
    # we disarm and resume normal quoting. SF is the safety exit (always
    # closes); TP is the opportunity exit (closes cheaply or not at all).
    # See ``app/take_profit.py`` for the pure helpers.
    #
    # Operator motivation (2026-05-22): the bot's existing ladder lets
    # favorable price spikes pass through without harvesting — we
    # requote ahead of the move every tick and watch the trend reverse
    # before our order fills. TP closes that loop.
    upnl_harvest_enabled: bool = Field(
        default=True,
        alias="UPNL_HARVEST_ENABLED",
    )
    # Threshold at which TP arms. Default 100 bps = ~$0.24 captured
    # on a $24 position. Roughly 2x the SF threshold (50 bps adverse)
    # so the trigger reads as "meaningful favorable swing" on the same
    # scale operators already use for SF. Chosen to fire 2-5 times per
    # active session — enough to generate observation data while not
    # over-clipping the right tail of the PnL distribution.
    #
    # Operator-iteration history:
    #   * Initial code default was 50 bps (mirror of SF threshold).
    #   * Raised to 100 bps for the first live session after pushback
    #     that 50 bps would fire on micro-noise; 300 bps was rejected
    #     as too high (would fire 0 times in a typical session, no
    #     data to evaluate against).
    upnl_harvest_trigger_bps: float = Field(
        default=100.0,
        ge=0.0,
        alias="UPNL_HARVEST_TRIGGER_BPS",
    )
    # Hysteresis margin. Disarm when uPnL falls to
    # ``(trigger_bps - disarm_margin_bps)``. Default 30 bps prevents
    # tight oscillation near the threshold while still keeping the
    # exit threshold (100 - 30 = 70 bps) meaningfully positive. The
    # operator's framing: "if we armed at 100 bps and price retraced
    # to 70 bps, the harvest opportunity is gone; give up and let
    # normal quoting handle the residual."
    upnl_harvest_disarm_margin_bps: float = Field(
        default=30.0,
        ge=0.0,
        alias="UPNL_HARVEST_DISARM_MARGIN_BPS",
    )
    # Max dwell in TP mode (seconds) before unconditional disarm.
    # If the post-only never fills in this window, the harvest
    # opportunity is considered gone and we resume normal quoting.
    upnl_harvest_max_dwell_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="UPNL_HARVEST_MAX_DWELL_SECONDS",
    )
    # Cooldown after disarm before TP can re-arm. Prevents
    # ping-ponging when uPnL hovers near the trigger threshold.
    upnl_harvest_arm_cooldown_seconds: float = Field(
        default=10.0,
        ge=0.0,
        alias="UPNL_HARVEST_ARM_COOLDOWN_SECONDS",
    )

    max_execution_errors: int = Field(default=8, alias="MAX_EXECUTION_ERRORS")
    # Rolling window (seconds) over which execution-error bumps are counted
    # against ``max_execution_errors``. Set to 0 to restore legacy cumulative
    # semantics (lifetime counter vs threshold). See code_reports/execution_kill_audit.md.
    execution_errors_window_seconds: float = Field(
        default=300.0,
        ge=0,
        alias="EXECUTION_ERRORS_WINDOW_SECONDS",
    )
    # After a post-only “would immediately match” rejection, suppress new placements on that side.
    # Set to 0 to disable. Default ~2–3s matches a typical 1.5s quote loop.
    post_only_cross_cooldown_seconds: float = Field(
        default=2.5,
        ge=0,
        alias="POST_ONLY_CROSS_COOLDOWN_SECONDS",
    )
    # Phase 2K.11 (v1.4.160) — favorable-exit predicate for the
    # post_only_cross_cooldown. Clears the side-suppression early
    # when the conflicting touch has moved AT LEAST
    # ``tick_multiplier × tick_size`` away from the rejected price
    # (default 1.0 tick): a re-place at the original price could no
    # longer cross. Lowest-priority of the Phase 2K series because
    # the cooldown is already short (2.5 s) — the upside is sub-minute
    # of saved suppression time per session — but the favorable-exit
    # pattern is now applied consistently across every gate.
    # No dwell needed: the touch-moved-1-tick predicate is exact, not
    # noisy. Set ``ENABLED=False`` for legacy pure-timer behaviour.
    post_only_cross_cooldown_favorable_exit_enabled: bool = Field(
        default=True,
        alias="POST_ONLY_CROSS_COOLDOWN_FAVORABLE_EXIT_ENABLED",
    )
    post_only_cross_cooldown_clear_tick_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        alias="POST_ONLY_CROSS_COOLDOWN_CLEAR_TICK_MULTIPLIER",
    )
    # v1.4.26 — fast-cancel-after-ack inferred-post-only-cross detector.
    # OKX V5 sometimes silently cancels a post-only place AT MATCH TIME
    # via the user-data-WS (no sCode 51604 in the place response). The
    # bot can't tell from the response alone — it sees a normal
    # "accepted" followed by a WS CANCELED event within milliseconds.
    # When we observe this pattern (CANCELED within N ms of ts_ack
    # AND the bot never requested a cancel), treat it as if 51604
    # fired: arm ``post_only_cross_cooldown_seconds`` for the side
    # so the next quote tick suppresses the same crossing place.
    #
    # Without this guard the bot enters a tight place-cancel-place
    # loop at the WS tick rate (~100/sec observed on TON-USDT-SWAP
    # in v1.4.25 snapshots).
    #
    # Default 100ms — safe lower bound for "this can only be a venue
    # silent cancel, no real maker order survives this short". Set
    # to 0 to disable the detector entirely.
    fast_cancel_post_only_cross_threshold_ms: float = Field(
        default=100.0,
        ge=0,
        alias="FAST_CANCEL_POST_ONLY_CROSS_THRESHOLD_MS",
    )
    # Extra price-grid clearance vs the opposite touch before submitting post-only limits.
    # 1 = at least one full tick below best_ask (buy) / above best_bid (sell); 2 adds another
    # tick of slack (less aggressive; helps stale book + HL rounding / immediate-match rejects).
    post_only_touch_buffer_ticks: int = Field(
        default=2,
        ge=1,
        le=12,
        alias="POST_ONLY_TOUCH_BUFFER_TICKS",
    )

    # Exchange HTTP retries (SDK uses synchronous requests).
    exchange_retry_max_attempts: int = Field(default=4, ge=1, alias="EXCHANGE_RETRY_MAX_ATTEMPTS")
    exchange_retry_base_seconds: float = Field(
        default=0.35, gt=0, alias="EXCHANGE_RETRY_BASE_SECONDS"
    )
    exchange_retry_max_backoff_seconds: float = Field(
        default=6.0, gt=0, alias="EXCHANGE_RETRY_MAX_BACKOFF_SECONDS"
    )
    exchange_rate_limit_extra_delay_seconds: float = Field(
        default=2.5, ge=0, alias="EXCHANGE_RATE_LIMIT_EXTRA_DELAY_SECONDS"
    )

    # 1.4.0 cancel-prio Phase 1b: opportunistic batch-cancel. When 2+
    # cancels for distinct sides queue in the same dispatcher flush,
    # the OutboundDispatchCoordinator can dispatch them as ONE call
    # to ``OkxClient.cancel_batch_orders`` instead of N per-side
    # cancel calls. Default ON — pure win on full-reprice cycles
    # where BUY + SELL cancel together. Flip OFF to fall back to the
    # legacy per-cancel HTTP path.
    batch_cancels_enabled: bool = Field(
        default=True,
        alias="BATCH_CANCELS_ENABLED",
    )

    # 1.4.4: opportunistic batch-PLACE. Parallel surface to
    # ``batch_cancels_enabled`` for the place lane. When 2+ places for
    # distinct (side, level_idx) queue in the same dispatcher flush
    # (or 1+ in ``batch_places_always`` mode), the
    # OutboundDispatchCoordinator dispatches them as ONE call to
    # ``OkxClient.batch_place_post_only_limit`` instead of N
    # per-intent /trade/order calls. Two wins: RTT compression
    # (N → 1) AND tapping the separate /trade/batch-orders rate-limit
    # pool (300/2s vs 60/2s on /trade/order). Default ON — directly
    # addresses the v1.4.2 row-level "Rate limit reached" pressure
    # that triggered the 1.4.4 work-unit. Flip OFF only for A/B
    # comparison or as a circuit-breaker if the batch path regresses.
    batch_places_enabled: bool = Field(
        default=True,
        alias="BATCH_PLACES_ENABLED",
    )

    # 1.4.4: always-batch mode. When True, ALL places — including
    # 1-element batches at N=1 — route through
    # ``batch_place_post_only_limit``. Unifies the code path so the
    # single-order endpoint /trade/order is effectively unused on
    # the place hot path. Trade-off: 1-element batches consume the
    # /trade/batch-orders rate-limit pool (300/2s) instead of
    # /trade/order's pool (60/2s). At N=1 the operator preferred
    # consistency over budget-pool predictability — flipping False
    # restores the "batch only when ≥2" behaviour.
    #
    # Operator decision 2026-05-17: True. The /trade/batch-orders
    # pool is larger anyway, so always-batch effectively widens the
    # rate-limit headroom even at N=1.
    batch_places_always: bool = Field(
        default=True,
        alias="BATCH_PLACES_ALWAYS",
    )

    # 1.4.4 Pass 2: opt-in AMEND-on-reprice. When True, the reprice
    # decision in ``OrderManager._orchestrate`` will route an
    # ACKED-order price/size change through
    # ``okx_client.amend_batch_orders`` instead of cancel-then-place.
    # Wins: preserves venue queue position (no queue-loss penalty on
    # tight reprices) AND taps the /trade/amend-batch-orders rate-limit
    # pool (300/2s, separate from both /trade/order and /trade/batch-orders).
    #
    # DEFAULT FALSE — the amend ADAPTER + INTERPRETER are shipped in
    # v1.4.4 but the reprice rewiring is gated off pending its own
    # focused work-unit (the rewiring touches the cancel-before-place
    # ordering machinery and deserves dedicated tests). Flip True only
    # AFTER the wiring lands.
    okx_amend_on_reprice_enabled: bool = Field(
        default=False,
        alias="OKX_AMEND_ON_REPRICE_ENABLED",
    )

    # 1.4.13 gate-widening Phase 2: per-gate widening coefficient
    # (bps the gate contributes to the half-spread when its signal
    # fires). Default ``-1.0`` is a SENTINEL meaning "use
    # MAX_HALF_SPREAD_BPS" — i.e. gate-equivalent magnitude,
    # functionally identical to the binary HOLD_ALL the gate
    # produced pre-cutover.
    #
    # Operator iteration: lower the coefficient toward 0 to make
    # the gate's response less aggressive (bot stays in the market
    # at narrower spread under that gate's signal). Per
    # ``plans/gate-to-widening.md`` Phase 2, iterate one gate at a
    # time, observing per-fill markouts at each step.
    #
    # Bounds: -1.0 (sentinel) or non-negative. Values above
    # MAX_HALF_SPREAD_BPS are clamped inside each gate's
    # ``widening_bps`` so the composition never produces a
    # contribution that bypasses the cap.
    vol_trend_widen_bps: float = Field(
        default=-1.0,
        alias="VOL_TREND_WIDEN_BPS",
    )
    post_swing_widen_bps: float = Field(
        default=-1.0,
        alias="POST_SWING_WIDEN_BPS",
    )
    microprice_widen_bps: float = Field(
        default=-1.0,
        alias="MICROPRICE_WIDEN_BPS",
    )
    basis_regime_widen_bps: float = Field(
        default=-1.0,
        alias="BASIS_REGIME_WIDEN_BPS",
    )
    momentum_widen_bps: float = Field(
        default=-1.0,
        alias="MOMENTUM_WIDEN_BPS",
    )
    freshness_one_sided_widen_bps: float = Field(
        default=-1.0,
        alias="FRESHNESS_ONE_SIDED_WIDEN_BPS",
    )
    recovery_cooldown_widen_bps: float = Field(
        default=-1.0,
        alias="RECOVERY_COOLDOWN_WIDEN_BPS",
    )

    # v1.4.102 — slow-trend gate (defence against the slow-grind
    # accumulation pattern surfaced in snapshot v1.4.92-260520-074215).
    # Complementary to the existing ``drift_block_long_window_bps`` gate
    # (5-min, 50 bp threshold) — this one watches a longer window with
    # a lower threshold to catch slow drifts that average away in
    # 5-min snapshots. See ``app/slow_trend_gate.py``.
    slow_trend_gate_enabled: bool = Field(
        default=False,  # Off by default for back-compat; TON profile enables.
        alias="SLOW_TREND_GATE_ENABLED",
    )
    slow_trend_window_seconds: float = Field(
        default=900.0,  # 15 min
        gt=0,
        alias="SLOW_TREND_WINDOW_SECONDS",
    )
    slow_trend_threshold_bps: float = Field(
        default=25.0,
        ge=0,
        alias="SLOW_TREND_THRESHOLD_BPS",
    )
    slow_trend_min_samples: int = Field(
        default=60,
        gt=0,
        alias="SLOW_TREND_MIN_SAMPLES",
    )
    slow_trend_anchor_fraction: float = Field(
        default=0.2,
        ge=0.0,
        alias="SLOW_TREND_ANCHOR_FRACTION",
    )
    # Unlike the 7 legacy gates this one ships with a NON-sentinel
    # default so it's live from day one without operator iteration.
    # 10 bp on the suppressed side is empirically wide enough to
    # deter fills during a real trend but not so wide that the bot
    # disappears after a single false positive. Operator can tune
    # via ``SLOW_TREND_WIDEN_BPS``; setting ``-1.0`` falls back to
    # ``max_half_spread_bps`` (gate-equivalent dark).
    slow_trend_widen_bps: float = Field(
        default=10.0,
        alias="SLOW_TREND_WIDEN_BPS",
    )

    # v1.4.106 Phase 1A — inventory_drift_gate. Position-aware short-
    # window (10s/30s) anti-aligned drift defence. Different from
    # slow_trend (position-blind, 15 min) and from
    # ``drift_block_long_window_bps`` (position-blind, 5 min, 50 bp):
    # this gate fires only when the bot is already loaded AND the
    # mid is moving in the direction that hurts the held inventory
    # AND it's moving fast enough to clear the short-window threshold.
    # See ``app/inventory_drift_gate.py``.
    inventory_drift_gate_enabled: bool = Field(
        default=False,  # Off by default for back-compat; TON profile enables.
        alias="INVENTORY_DRIFT_GATE_ENABLED",
    )
    inventory_drift_inventory_pct_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        alias="INVENTORY_DRIFT_INVENTORY_PCT_THRESHOLD",
    )
    inventory_drift_threshold_bps_10s: float = Field(
        default=15.0,
        ge=0.0,
        alias="INVENTORY_DRIFT_THRESHOLD_BPS_10S",
    )
    inventory_drift_threshold_bps_30s: float = Field(
        default=30.0,
        ge=0.0,
        alias="INVENTORY_DRIFT_THRESHOLD_BPS_30S",
    )
    # Ships with a non-sentinel default (15 bp) — live from day one
    # without an iteration phase, slightly wider than slow_trend's 10
    # bp because this gate's trigger is more specific (position-aware
    # AND short-window AND anti-aligned).
    inventory_drift_widen_bps: float = Field(
        default=15.0,
        alias="INVENTORY_DRIFT_WIDEN_BPS",
    )

    # v1.4.107 Phase 1B — shock_gate (acute spike defence, binary).
    # Different from `inventory_drift_gate`: thresholds are MUCH higher
    # (20 bp / 10 s, 50 bp / 30 s) so the gate fires only on genuine
    # shock events; and the response is BINARY (eligibility override)
    # rather than widening. At shock magnitudes "stay in the market at
    # wider price" is the wrong response — quote freshness, not spread
    # economics, is the dominant risk.
    shock_gate_enabled: bool = Field(
        default=False,
        alias="SHOCK_GATE_ENABLED",
    )
    shock_inventory_pct_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        alias="SHOCK_INVENTORY_PCT_THRESHOLD",
    )
    shock_threshold_bps_10s: float = Field(
        default=20.0,
        ge=0.0,
        alias="SHOCK_THRESHOLD_BPS_10S",
    )
    shock_threshold_bps_30s: float = Field(
        default=50.0,
        ge=0.0,
        alias="SHOCK_THRESHOLD_BPS_30S",
    )
    # Util threshold below which the soft-clear path activates (paired
    # with drift normalisation). At 0.30 the bot has bled inventory
    # back to near-neutral; safe to re-enable the suppressed side.
    shock_clear_util_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        alias="SHOCK_CLEAR_UTIL_THRESHOLD",
    )
    # Hard ceiling on lock duration — safety net so a feed glitch
    # can't keep the gate locked forever. 5 min is well past the
    # typical shock-event recovery window (the 260520 event resolved
    # in < 2 min).
    shock_max_cooldown_seconds: float = Field(
        default=300.0,
        gt=0.0,
        alias="SHOCK_MAX_COOLDOWN_SECONDS",
    )

    # v1.4.112 Phase 1C — regime_controller FSM. Aggregates the Phase
    # 1A / 1B / existing slow_trend / vol-ratio signals into a single
    # NORMAL / DEFENSIVE / SHOCK mode label, with per-mode knob
    # overlays (size mult, base half-spread mult, etc) that flow into
    # ``compute_quote_decision``. See ``app/regime_controller.py``.
    regime_controller_enabled: bool = Field(
        default=False,
        alias="REGIME_CONTROLLER_ENABLED",
    )
    regime_entry_dwell_seconds: float = Field(
        default=15.0,
        ge=0.0,
        alias="REGIME_ENTRY_DWELL_SECONDS",
    )
    regime_exit_dwell_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="REGIME_EXIT_DWELL_SECONDS",
    )
    regime_util_entry_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        alias="REGIME_UTIL_ENTRY_THRESHOLD",
    )
    regime_util_exit_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        alias="REGIME_UTIL_EXIT_THRESHOLD",
    )
    regime_vol_ratio_entry_threshold: float = Field(
        default=2.0,
        ge=0.0,
        alias="REGIME_VOL_RATIO_ENTRY_THRESHOLD",
    )
    # Phase 2G (v1.4.204) — SHOCK-mode Telegram alert toggles. Three
    # alert classes:
    #   * SHOCK ENTRY (always one-shot per episode)
    #   * SHOCK PERSISTENCE (one-shot when dwell exceeds
    #     ``regime_shock_telegram_persistence_seconds``; the v1.4.189
    #     phase-ladder catastrophe persisted 13 min in SHOCK before
    #     SF fired — an alert at 5 min would have caught it)
    # Daily-summary message (2G.3) is deferred — cron-style scheduling
    # is out of scope for an in-line tick check.
    #
    # Default ON, threshold 300 s (5 minutes) for persistence. Both
    # OFF when the Telegram notifier isn't configured (no-op).
    regime_shock_telegram_alerts_enabled: bool = Field(
        default=True,
        alias="REGIME_SHOCK_TELEGRAM_ALERTS_ENABLED",
    )
    regime_shock_telegram_persistence_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="REGIME_SHOCK_TELEGRAM_PERSISTENCE_SECONDS",
    )

    # v1.5.146 Phase 4G.10 — SHOCK go-dark toggle.
    #
    # Background: ``RegimeKnobs.ladder_levels_max`` is set to 0 for
    # SHOCK in ``app/regime_controller.py`` (the documented design
    # intent — SHOCK should rely on the SF/flatten path, not on
    # passive quoting). However ``app/bot.py`` and ``app/execution.py``
    # both ``max(1, ...)``-floor the cap when consulting the knob so
    # SHOCK actually still quotes 1 rung. The 4G.8 (v1.4.219) entry
    # explicitly tags this as a "bigger behavioural change than the
    # fix scope" deferred to 4G.10.
    #
    # This flag lets the operator A/B test the two designs:
    #   * False (default, current behaviour): SHOCK floors at 1 rung.
    #     The bot keeps a single reducing-side rung under shock_gate's
    #     QuoteEligibility lock — what every prod session before
    #     v1.5.146 has done.
    #   * True (4G.10 design intent): SHOCK honours cap=0 and the
    #     ladder builds ZERO rungs both sides. The bot is fully dark
    #     on passive quoting during SHOCK and relies on the
    #     soft-flatten path to reduce existing inventory. Resting
    #     orders age out via their normal cancel-on-move paths;
    #     ``app/soft_flatten.py`` retains its independent firing
    #     condition and can still trigger during SHOCK.
    #
    # Safe to enable on a low-notional / drawdown-protected profile
    # for one or two sessions to compare. Disable to revert. Has no
    # effect when the bot is not in SHOCK mode.
    shock_ladder_allow_full_dark: bool = Field(
        default=False,
        alias="SHOCK_LADDER_ALLOW_FULL_DARK",
    )

    # v1.5.26 Phase 2G.3 -- daily time-in-mode Telegram summary.
    #
    # Pre-fix this was tagged "DEFERRED, needs cron-style scheduling"
    # at v1.4.204. v1.5.26 implements it without cron: the regime-
    # controller per-tick check on bot.py looks at
    # ``state.regime_daily_summary_last_sent_mono`` and fires when
    # the gap >= ``regime_daily_summary_interval_seconds``. The
    # bot's quote loop runs at ~2 Hz so the cadence check is
    # cheap; the actual send is fire-and-forget via notify_ops.
    #
    # Default ENABLED with 24h cadence. The first summary fires
    # ~24h after process start (because last_sent_mono=0.0 means
    # "never sent" but we compute ``now_mono - 0.0 >= 86400`` which
    # is true once monotonic time has passed 24h since process
    # start). For shorter cadences, set the interval explicitly.
    #
    # Set ENABLED=false to disable entirely (e.g. test profiles).
    regime_daily_summary_enabled: bool = Field(
        default=True,
        alias="REGIME_DAILY_SUMMARY_ENABLED",
    )
    regime_daily_summary_interval_seconds: float = Field(
        default=86400.0,  # 24 hours
        gt=0.0,
        alias="REGIME_DAILY_SUMMARY_INTERVAL_SECONDS",
    )

    # v1.5.26 Phase 2D -- residual-decay trigger for adaptive_widen.
    #
    # Adds a composite-signal arming branch to the existing
    # adaptive_widen gate: when the rolling mean of
    # ``closed_pnl_bps + rebate_bps - markout_5s_bps`` per fill
    # drops below ``residual_decay_widen_threshold_bps`` (negative,
    # default -0.5) and stays below for
    # ``residual_decay_widen_dwell_seconds`` (default 60s), the
    # adaptive_widen gate fires with ``reason=residual_decay``.
    #
    # The signal is "net economic edge after rebates and closed-PnL"
    # -- distinct from the existing single-signal triggers
    # (toxicity hard/soft, markout-only, one-sided, quote-quality,
    # slow-trend) because it captures cases where the bot is
    # bleeding even though no single signal looks anomalous.
    #
    # Disabled by default (threshold = 0.0); enable explicitly per
    # profile with a negative threshold.
    residual_decay_widen_threshold_bps: float = Field(
        default=0.0,
        le=0.0,  # negative or zero only -- positive would be nonsensical
        alias="RESIDUAL_DECAY_WIDEN_THRESHOLD_BPS",
    )
    residual_decay_widen_dwell_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="RESIDUAL_DECAY_WIDEN_DWELL_SECONDS",
    )
    residual_decay_widen_window_fills: int = Field(
        default=50,
        gt=0,
        alias="RESIDUAL_DECAY_WIDEN_WINDOW_FILLS",
    )
    residual_decay_widen_min_fills: int = Field(
        default=10,
        gt=0,
        alias="RESIDUAL_DECAY_WIDEN_MIN_FILLS",
    )

    # ----------------------------------------------------------------
    # Phase 4G.5 (v1.4.211) / 4G.6 (v1.4.212) — forward-looking regime
    # classifier knobs.
    #
    # Master enable flag default flipped ON in v1.4.213 (2026-05-21).
    # Reasoning: the v1.4.200 SF#11175 incident is the WHOLE REASON
    # Phase 4G exists. Shipping with the feature OFF means the very
    # next storm trades with zero forward protection — i.e. the bot
    # eats another SF before the operator gets around to flipping the
    # flag. That defeats the purpose. Default ON, operator overrides
    # to OFF only if they explicitly want to bisect a regression.
    #
    # Safety net is intact: the reactive NORMAL/DEFENSIVE/SHOCK FSM
    # continues to run unchanged. The forward layer can only PROACT
    # (transition NORMAL → CAUTIOUS / CALM ahead of the reactive
    # triggers); it never suppresses the reactive shock_gate or SF
    # gates. False-positive CAUTIOUS costs widened spread + smaller
    # size, NOT actual losses. False-negative CAUTIOUS gracefully
    # falls back to the v1.4.210 reactive behaviour. Either way the
    # bot is at WORST as safe as v1.4.210.
    #
    # The classifier itself is implemented in ``app/regime_forward_signals.py``
    # (Phase 4G.1, shipped v1.4.208). The thresholds below mirror the
    # ``ForwardSignalThresholds`` defaults — keeping them as env-driven
    # settings is the load-bearing tuning surface for the operator,
    # since the v1.4.200 snapshot envelope used to seed the defaults
    # won't generalise to other regimes / symbols.
    # ----------------------------------------------------------------
    regime_forward_enabled: bool = Field(
        default=True,
        alias="REGIME_FORWARD_ENABLED",
    )
    # Dwell timings (legacy, v1.5.191 defaults all zeroed).
    # Pre-v1.5.191: the FSM applied fixed-time hysteresis on top of
    # the classifier's single-threshold output. That was a CLAUDE.md
    # Rule 0c violation (timer-based gate) AND empirically didn't
    # prevent flapping — see issues/bug-034.md + plans/20260527-regime-
    # band-hysteresis.md for the postmortem.
    #
    # v1.5.191 moves all hysteresis into the classifier via Schmitt-
    # trigger band thresholds (``cautious_enter_*`` / ``cautious_exit_*``
    # / ``calm_enter_*`` / ``calm_exit_*`` below). The dwell knobs
    # remain in the FSM signature for backward-compat with anyone who
    # explicitly opts back into time-based dwell, but the defaults are
    # zero so they don't interfere with the band layer. Setting any
    # of these non-zero stacks a confirmation timer on top of the band
    # hysteresis — not recommended in practice; the band layer is
    # already correct.
    regime_forward_cautious_entry_dwell_seconds: float = Field(
        default=0.0,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_ENTRY_DWELL_SECONDS",
    )
    regime_forward_cautious_exit_dwell_seconds: float = Field(
        default=0.0,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_EXIT_DWELL_SECONDS",
    )
    regime_forward_calm_entry_dwell_seconds: float = Field(
        default=0.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_ENTRY_DWELL_SECONDS",
    )
    regime_forward_calm_exit_dwell_seconds: float = Field(
        default=0.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_EXIT_DWELL_SECONDS",
    )
    # Classifier thresholds — CAUTIOUS triggers (v1.5.191 band hysteresis).
    # Any one indicator above its threshold → CAUTIOUS. ENTER thresholds
    # apply when current mode is NOT CAUTIOUS; EXIT thresholds (lower)
    # apply when ALREADY in CAUTIOUS, so the bot stays CAUTIOUS until
    # signals clearly recede. Setting enter == exit reverts to the
    # pre-v1.5.191 single-threshold behaviour for that criterion.
    regime_forward_cautious_enter_vol_slope_bps_per_min: float = Field(
        default=0.5,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_ENTER_VOL_SLOPE_BPS_PER_MIN",
    )
    regime_forward_cautious_exit_vol_slope_bps_per_min: float = Field(
        default=0.25,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_EXIT_VOL_SLOPE_BPS_PER_MIN",
    )
    regime_forward_vol_slope_lookback_seconds: float = Field(
        default=60.0,
        gt=0.0,
        alias="REGIME_FORWARD_VOL_SLOPE_LOOKBACK_SECONDS",
    )
    regime_forward_cautious_enter_drift_magnitude_rising_ratio: float = Field(
        default=1.5,
        gt=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_ENTER_DRIFT_MAGNITUDE_RISING_RATIO",
    )
    regime_forward_cautious_exit_drift_magnitude_rising_ratio: float = Field(
        default=1.25,
        gt=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_EXIT_DRIFT_MAGNITUDE_RISING_RATIO",
    )
    regime_forward_drift_magnitude_lookback_seconds: float = Field(
        default=30.0,
        gt=0.0,
        alias="REGIME_FORWARD_DRIFT_MAGNITUDE_LOOKBACK_SECONDS",
    )
    regime_forward_drift_magnitude_floor_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="REGIME_FORWARD_DRIFT_MAGNITUDE_FLOOR_BPS",
    )
    regime_forward_cautious_enter_ob_imbalance_widening_delta: float = Field(
        default=0.3,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_ENTER_OB_IMBALANCE_WIDENING_DELTA",
    )
    regime_forward_cautious_exit_ob_imbalance_widening_delta: float = Field(
        default=0.15,
        ge=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_EXIT_OB_IMBALANCE_WIDENING_DELTA",
    )
    regime_forward_ob_imbalance_lookback_seconds: float = Field(
        default=60.0,
        gt=0.0,
        alias="REGIME_FORWARD_OB_IMBALANCE_LOOKBACK_SECONDS",
    )
    regime_forward_cautious_enter_basis_stretch_ratio: float = Field(
        default=2.0,
        gt=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_ENTER_BASIS_STRETCH_RATIO",
    )
    regime_forward_cautious_exit_basis_stretch_ratio: float = Field(
        default=1.5,
        gt=0.0,
        alias="REGIME_FORWARD_CAUTIOUS_EXIT_BASIS_STRETCH_RATIO",
    )
    regime_forward_basis_stretch_floor_bps: float = Field(
        default=1.0,
        ge=0.0,
        alias="REGIME_FORWARD_BASIS_STRETCH_FLOOR_BPS",
    )
    # Classifier thresholds — CALM gates (v1.5.191 band hysteresis).
    # ALL must hold for ≥ calm_min_history_seconds. ENTER thresholds
    # apply when current mode is NOT CALM (strict, "be deeply calm to
    # commit"). EXIT thresholds (looser) apply when ALREADY in CALM,
    # so the bot stays CALM until signals clearly drift outside the
    # quiet zone. Setting enter == exit reverts to pre-v1.5.191
    # single-threshold behaviour.
    regime_forward_calm_enter_max_vol_bps: float = Field(
        default=3.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_ENTER_MAX_VOL_BPS",
    )
    regime_forward_calm_exit_max_vol_bps: float = Field(
        default=6.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_EXIT_MAX_VOL_BPS",
    )
    regime_forward_calm_enter_max_drift_magnitude_bps: float = Field(
        default=5.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_ENTER_MAX_DRIFT_MAGNITUDE_BPS",
    )
    regime_forward_calm_exit_max_drift_magnitude_bps: float = Field(
        default=10.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_EXIT_MAX_DRIFT_MAGNITUDE_BPS",
    )
    regime_forward_calm_enter_max_ob_imbalance_magnitude: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        alias="REGIME_FORWARD_CALM_ENTER_MAX_OB_IMBALANCE_MAGNITUDE",
    )
    regime_forward_calm_exit_max_ob_imbalance_magnitude: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        alias="REGIME_FORWARD_CALM_EXIT_MAX_OB_IMBALANCE_MAGNITUDE",
    )
    regime_forward_calm_min_history_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="REGIME_FORWARD_CALM_MIN_HISTORY_SECONDS",
    )
    # History-buffer cap. The classifier needs at most ``max(lookback)``
    # seconds of (ts_mono, value) tuples per indicator. With defaults
    # of 60 s for vol_slope / ob_imbalance and 30 s for drift, plus
    # CALM's 60 s history-span gate, ~180 s is more than enough. At
    # ~2 Hz tick rate that's ~360 entries per buffer — negligible memory.
    regime_forward_history_buffer_seconds: float = Field(
        default=180.0,
        gt=0.0,
        alias="REGIME_FORWARD_HISTORY_BUFFER_SECONDS",
    )

    # v1.5.18 Phase 4G.7 -- the BASIS history buffer has its own
    # (longer) retention window because the classifier's
    # ``basis_stretch_cautious_ratio`` trigger compares current basis
    # to its 30-minute MEDIAN, not to a recent derivative. The other
    # three forward indicators use ~180 s (per
    # ``regime_forward_history_buffer_seconds``) because they're
    # consumed by derivative computations over short windows. Default
    # 1800 s = 30 min matches the classifier's
    # ``binance_basis_30min_median_bps`` semantics.
    regime_forward_basis_median_lookback_seconds: float = Field(
        default=1800.0,
        gt=0.0,
        alias="REGIME_FORWARD_BASIS_MEDIAN_LOOKBACK_SECONDS",
    )

    # ----------------------------------------------------------------
    # Phase 4G.13 (v1.4.228) — Structural-bias auto-throttle.
    #
    # The forward classifier (Phase 4G.5) is a LEADING-indicator
    # system: vol slope rising, drift building, ob_imbalance
    # widening. It can MISS structural-bias accumulation that
    # builds via slow drift below the classifier's per-indicator
    # thresholds. The v1.4.219-260521-214353 snapshot's SF#11183
    # is the textbook failure: bot sat in NORMAL for 285 s between
    # the prior CAUTIOUS exit and SF firing, with adverse drift
    # climbing the whole time but no single indicator triggering
    # CAUTIOUS re-entry.
    #
    # The structural-bias gate is the BACKSTOP. It reads the
    # session-cumulative ``inventory_exec_bias`` suppression
    # counts per side (already on ``quote_quality``) and forces
    # one-sided REDUCING eligibility when the ratio crosses a
    # threshold:
    #
    #   BID-suppressed >> ASK-suppressed → LONG bias → force
    #     QUOTE_SELL_ONLY (bot can only sell to reduce)
    #   ASK-suppressed >> BID-suppressed → SHORT bias → force
    #     QUOTE_BUY_ONLY (bot can only buy to reduce)
    #
    # This breaks the "accumulate → bleed → SF → re-accumulate"
    # cycle by preventing the bot from re-leaning into the same
    # direction immediately after an SF flatten. Stays active
    # until the ratio falls back below the threshold (which
    # happens naturally as the OTHER side fires more suppressions
    # while the bot trades reducing-only).
    #
    # Default ENABLED — this is a safety backstop, not a
    # performance feature. The Bias card (v1.4.220) surfaces the
    # ratio so the operator can see when the throttle is about
    # to engage.
    # ----------------------------------------------------------------
    structural_bias_auto_throttle_enabled: bool = Field(
        default=True,
        alias="STRUCTURAL_BIAS_AUTO_THROTTLE_ENABLED",
    )
    # Ratio threshold for engagement. Default 5.0 mirrors the
    # v1.4.220 Bias card's RED tier. Below this → no action; at-or-
    # above → force reducing-only on the structural lean direction.
    structural_bias_auto_throttle_ratio_threshold: float = Field(
        default=5.0,
        gt=1.0,
        alias="STRUCTURAL_BIAS_AUTO_THROTTLE_RATIO_THRESHOLD",
    )
    # Minimum total suppression-count floor before the gate
    # engages. Without this, a fresh session with 5 BID-suppress
    # / 0 ASK-suppress would compute ratio = inf and throttle
    # immediately — false positive. Wait until enough samples
    # accumulate that the ratio is statistically meaningful.
    structural_bias_auto_throttle_min_samples: int = Field(
        default=20,
        ge=0,
        alias="STRUCTURAL_BIAS_AUTO_THROTTLE_MIN_SAMPLES",
    )
    # v1.5.197 — idle-decay for the ``inventory_exec_bias`` session-
    # cumulative counters that feed this throttle. When no fill has
    # arrived for this many seconds, halve both bid + ask suppression
    # counts. Gives the counter pair a half-life equal to this value
    # so historical-storm asymmetry doesn't lock the bot permanently
    # after the regime changes. See plans/20260527-regime-band-
    # hysteresis.md v1.5.197 section for the defensive-deadlock
    # post-mortem this addresses.
    inventory_exec_bias_idle_decay_seconds: float = Field(
        default=300.0,
        ge=0.0,
        alias="INVENTORY_EXEC_BIAS_IDLE_DECAY_SECONDS",
    )

    # v1.5.2 Phase 4D.3 — post-SF cooldown gate.
    #
    # Problem: after a soft-flatten completes, the bot's quote engine
    # is free to immediately re-accumulate in the SAME direction that
    # caused the SF. The v1.4.219 audit captured 3 SFs in 27 min where
    # the bot oscillated between SHORT and forced flatten because the
    # post-SF window had no friction against re-leaning back to SHORT.
    #
    # When enabled, ``post_sf_cooldown_seconds`` after SF completion,
    # the side that would re-add to the pre-SF position direction is
    # forced into one-sided REDUCING eligibility. E.g. if SF was
    # triggered while LONG, BUY is suppressed during the cooldown.
    # The cooldown clears on its own; the operator can shorten by
    # restarting the bot.
    #
    # Default 60 s — based on the SF-cluster spacing observed in the
    # v1.4.219 audit (3 SFs over 27 min → ~9 min between SFs, but
    # the second SF re-armed within ~3 min of the first completing).
    # 60s is short enough not to suppress legitimate two-sided quoting
    # but long enough to break the immediate-re-accumulate pattern.
    #
    # See ``plans/20260520-defense-action-plan.md`` Phase 4D.3.
    post_sf_cooldown_enabled: bool = Field(
        default=True,
        alias="POST_SF_COOLDOWN_ENABLED",
    )
    post_sf_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        alias="POST_SF_COOLDOWN_SECONDS",
    )

    # v1.4.113 Phase 1D — post-reduction re-entry cooldown. After any
    # fill that reduces |position_qty|, suppress the "adding side"
    # (the side that would add to the current post-fill direction)
    # for ``post_reduction_cooldown_seconds`` IFF the regime_controller
    # is in DEFENSIVE or SHOCK. NORMAL mode keeps round-trip rebate
    # capture intact. Targets the 06:50:05 fast-flip pattern from the
    # 2026-05-20 snapshot: bot unwound LONG and re-opened SHORT within
    # 60 s under shock conditions, getting picked off on both legs.
    # See ``plans/20260520-defense-action-plan.md`` Phase 1D.
    post_reduction_cooldown_seconds: float = Field(
        default=0.0,  # Off by default for back-compat; TON profile enables.
        ge=0.0,
        alias="POST_REDUCTION_COOLDOWN_SECONDS",
    )
    # v1.4.144 Phase 2K.1 — favorable-exit predicate for the post-
    # reduction cooldown. Instead of waiting the full
    # ``post_reduction_cooldown_seconds`` deadline regardless of
    # market state, the bot clears the cooldown EARLY when position
    # util drops below this threshold — i.e. the natural unwind has
    # already happened and the gate's original "don't fast-flip into
    # the side we just exited" intent is satisfied. The cooldown
    # seconds remain as a MAX-ceiling safety net for the case where
    # util doesn't drop (slow unwind / sustained adverse direction).
    #
    # Default 0.30 mirrors ``shock_gate.clear_util_threshold`` — a
    # mid-30 % util means the bot has cleared the bulk of the
    # position that the reducing fill was unwinding. Setting this
    # to 0 effectively disables the early-exit (pure timer mode);
    # setting it >= 1 makes the cooldown ALWAYS early-exit (since
    # util can never exceed 1.0).
    post_reduction_cooldown_clear_util_pct: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        alias="POST_REDUCTION_COOLDOWN_CLEAR_UTIL_PCT",
    )

    # 1.4.0 cancel-prio Phase 0c: cancel-specific HTTP timeout + retry
    # policy. Cancels are time-critical; we'd rather fail-fast and
    # retry sooner than wait the shared 8s read / 350 ms-base policy.
    # Applies to OKX REST ops ``cancel_order``, ``cancel_order_by_cloid``,
    # ``cancel_batch``. Other ops keep the shared defaults above.
    cancel_http_read_timeout_seconds: float = Field(
        default=1.5,
        gt=0,
        alias="CANCEL_HTTP_READ_TIMEOUT_SECONDS",
    )
    cancel_retry_max_attempts: int = Field(
        default=4,
        ge=1,
        alias="CANCEL_RETRY_MAX_ATTEMPTS",
    )
    cancel_retry_base_seconds: float = Field(
        default=0.05,
        gt=0,
        alias="CANCEL_RETRY_BASE_SECONDS",
    )
    cancel_retry_cap_seconds: float = Field(
        default=0.5,
        gt=0,
        alias="CANCEL_RETRY_CAP_SECONDS",
    )

    # 1.3.108 cancel-prio Phase 2: per-op HTTP client + connection pool
    # split inside OkxClient. Two httpx.Client instances — _http_cancel
    # (tight timeouts, small pool budget) and _http_place (current
    # generous defaults) — keep cancel ops on a dedicated pool so a
    # place-pool stall under load cannot block a time-critical cancel
    # waiting on the SAME pool slot. HTTP/2 multiplexes streams on one
    # TLS connection to OKX's REST CDN, removing TCP head-of-line
    # blocking. Default True; flip to False as a one-line revert if
    # an OKX-side compatibility issue surfaces.
    okx_http2_enabled: bool = Field(
        default=True,
        alias="OKX_HTTP2_ENABLED",
    )

    # 1.3.110 cancel-prio Phase 4a: OKX trade-via-WS for CANCEL ops.
    # When True, cancel_order / cancel_order_by_cloid / cancel_batch_orders
    # send the request as a frame on the OKX V5 trade-WS socket
    # (/ws/v5/private with a signed login). HTTP fallback engages when
    # the WS is disconnected, mid-reconnect, or the request times out
    # (and ``action_http_fallback_enabled`` is True — default).
    # Default is False — v1.3.116 flipped it to True after the initial
    # Stage 3 verification looked clean, but v1.3.117 reverted it
    # after a venue-side rejection pattern (sCode 50014
    # ``Parameter instIdCode can not be empty``) appeared on the
    # cancel-retry-with-cloid escalation path and triggered an
    # ``execution_errors`` kill. Root cause TBD — likely either an
    # OKX trade-WS frame contract drift or a colo-WS-specific
    # parameter requirement not documented at v1.3.110 ship time.
    # Do NOT flip back to True until the venue-side spec for the
    # cloid-cancel WS frame is reconfirmed against OKX V5 docs.
    okx_action_ws_cancel_enabled: bool = Field(
        default=False,
        alias="OKX_ACTION_WS_CANCEL_ENABLED",
    )
    # 1.3.117: cloid-escalation in the cancel-pending-retry path.
    # When True (default), after one failed ordId-based retry the bot
    # escalates to cancel-by-cloid (GRVT-era workaround for an
    # ack-but-no-cancel quirk on GRVT's ordId path). When False, the
    # retry loop stays on ordId indefinitely. **Must be False on OKX
    # colo** (alibaba-hk endpoint family): the colo WS expects
    # ``instIdCode`` (numeric) instead of ``instId`` (string) for
    # cloid-lookup, and our WS frame sends ``instId``, so OKX rejects
    # with sCode 50014 ``Parameter instIdCode can not be empty`` →
    # repeats until ``execution_errors`` self-kills the bot. Safe to
    # disable on OKX because Phase 1a's sync ordId binding guarantees
    # ordId is bound by retry time. See ``plans/_DONE/sbe.md`` §2 for
    # the colo wire-format note that documents the ``instIdCode``
    # parameter.
    cancel_pending_cloid_escalation_enabled: bool = Field(
        default=True,
        alias="CANCEL_PENDING_CLOID_ESCALATION_ENABLED",
    )

    # 1.3.120: paranoid-mode for the "order is gone but not via fill"
    # cancel-response codes (51400 / 51401 / 51503). Default is False
    # — these classify as ``unexpected_gone`` which logs WARNING and
    # bumps the dedicated ``cancel_unexpected_gone_total`` counter
    # but does NOT bump ``execution_errors``. Default-False because
    # legitimate cleanup-cancel-all races (incident 2026-05-05 22:35
    # UTC, 8 hits in 5 min) would otherwise self-kill the bot.
    #
    # Flip to True for strict mode: ``unexpected_gone`` reclassifies
    # to ``error`` which DOES bump ``execution_errors``, triggering
    # self-kill on sustained occurrence within the rolling
    # ``execution_errors_window_seconds`` window. Useful when the
    # operator wants maximum alerting and accepts that cleanup races
    # may bounce the process. See
    # ``app/exchange/okx_responses.py:_UNEXPECTED_GONE_CODES`` for
    # the response-code rationale.
    cancel_unexpected_gone_strict_mode: bool = Field(
        default=False,
        alias="CANCEL_UNEXPECTED_GONE_STRICT_MODE",
    )

    # 1.3.121: trust the trade-WS / HTTP sCode 0 success response as
    # TERMINAL proof the order has been cancelled. When True, on cancel
    # success the bot immediately transitions the WO to CANCELED locally
    # (sets working_bid/ask to None, clears side_unresolved) instead of
    # waiting for the inbound user-data WS terminal event.
    #
    # WHY: on OKX colo (snap 2026-05-17), the user-data WS event for a
    # successful cancel sometimes lags 90+ seconds. During that window
    # the side stays unresolved, the cancel-pending watchdog re-fires
    # cancels, and the bot effectively stops quoting. The trade-WS
    # sCode 0 response IS authoritative — once OKX accepts the cancel
    # request, the order is being cancelled at the matching engine.
    # Waiting for the user-data WS event only delays the bot resuming
    # quotes.
    #
    # RISK: tiny race window where sCode 0 returns but a fill landed in
    # the same nanosecond — bot would mark WO CANCELED locally while
    # user-data WS would later push FILLED. WS handler's oid-match path
    # is idempotent (returns silently when working_bid/ask is None), so
    # the late FILLED event is swallowed. **This means a fill in that
    # micro-window would be missed by inventory accounting** until
    # the next reconcile observes the position delta. Acceptable on
    # OKX V5 because matching-engine cancel acceptance is sequential
    # — sCode 0 means the cancel is in the queue ahead of any pending
    # match for that order. On other adapters with ack-but-no-cancel
    # quirks (GRVT historically) this WOULD be unsafe.
    #
    # Default False (preserves legacy behavior). Set True on OKX profile
    # to unblock the trading-stalls-on-WS-lag problem.
    cancel_trust_trade_ws_success_terminal: bool = Field(
        default=False,
        alias="CANCEL_TRUST_TRADE_WS_SUCCESS_TERMINAL",
    )
    # 1.3.110 Phase 4b prep: place-via-WS enabled flag, accepted by
    # config so the staged-release env doesn't need a new bump to flip
    # it on. Plumbing in OkxClient gets wired in Phase 4b; reading the
    # flag today is a harmless no-op.
    okx_action_ws_place_enabled: bool = Field(
        default=False,
        alias="OKX_ACTION_WS_PLACE_ENABLED",
    )
    # Per-frame round-trip timeout for the action WS. Tight default
    # (1.5 s) so a stalled WS doesn't park a cancel beyond the bot's
    # own ``cancel_http_read_timeout_seconds`` budget — fallback to
    # HTTP kicks in within the same wall-clock window. Tune via env
    # for venues with slower trade-WS dispatch.
    okx_action_ws_request_timeout_seconds: float = Field(
        default=1.5,
        gt=0,
        alias="OKX_ACTION_WS_REQUEST_TIMEOUT_SECONDS",
    )
    # Reconnect backoff for the action-WS daemon thread. Initial 0.5s,
    # double each failed attempt up to a 30s cap. Matches the existing
    # read-only OkxPrivateStream cadence so an outage that takes down
    # one channel doesn't reconnect the other 60x faster.
    okx_action_ws_reconnect_initial_seconds: float = Field(
        default=0.5,
        gt=0,
        alias="OKX_ACTION_WS_RECONNECT_INITIAL_SECONDS",
    )
    okx_action_ws_reconnect_cap_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="OKX_ACTION_WS_RECONNECT_CAP_SECONDS",
    )
    # After N identical deterministic validation rejects (same symbol/side/normalized px/sz
    # and same exchange error text), suppress place_post_only for that path until quarantine
    # elapses (monotonic). Emits bot event order_rejection_quarantine_started.
    exchange_validation_quarantine_min_repeats: int = Field(
        default=2,
        ge=1,
        alias="EXCHANGE_VALIDATION_QUARANTINE_MIN_REPEATS",
    )
    exchange_validation_quarantine_seconds: float = Field(
        default=12.0,
        gt=0,
        alias="EXCHANGE_VALIDATION_QUARANTINE_SECONDS",
    )

    # Rolling 60s fill count; 0 disables (no NO_QUOTE from this rule).
    max_trades_per_minute: int = Field(default=120, ge=0, alias="MAX_TRADES_PER_MINUTE")

    flatten_timeout_seconds: float = Field(default=120.0, alias="FLATTEN_TIMEOUT_SECONDS")
    flatten_attempt_delay_seconds: float = Field(
        default=0.35, ge=0, alias="FLATTEN_ATTEMPT_DELAY_SECONDS"
    )
    # Min seconds between persisted position/equity snapshots (0 = persist every tick).
    snapshot_interval_seconds: float = Field(default=10.0, ge=0, alias="SNAPSHOT_INTERVAL_SECONDS")
    # Operator metrics JSON (daily PnL/trade stats, last fill, optional toxicity rollups). Empty = off.
    persistent_runtime_state_path: str = Field(default="", alias="PERSISTENT_RUNTIME_STATE_PATH")
    # 0 disables periodic saves (shutdown save still runs when path is set).
    persistent_runtime_save_interval_seconds: float = Field(
        default=60.0,
        ge=0,
        alias="PERSISTENT_RUNTIME_SAVE_INTERVAL_SECONDS",
    )
    # After loading operator JSON, optionally rebuild day trade count / notional / (maybe) realized PnL from user_fills.
    persistent_runtime_reconcile_from_fills: bool = Field(
        default=False,
        alias="PERSISTENT_RUNTIME_RECONCILE_FROM_FILLS",
    )
    # If the API returns at least this many symbol fills and none are before today UTC, history may be truncated.
    persistent_runtime_reconcile_fill_window_cap: int = Field(
        default=2000,
        ge=1,
        alias="PERSISTENT_RUNTIME_RECONCILE_FILL_WINDOW_CAP",
    )
    control_endpoints_enabled: bool = Field(
        default=False,
        alias="CONTROL_ENDPOINTS_ENABLED",
    )
    # Read-only DB download (``GET /db/download``). Separate from
    # ``CONTROL_ENDPOINTS_ENABLED`` because it's a read-only operation —
    # pulling the DB can't mutate bot state, so it doesn't need the same
    # "operator action" guard that the POST ``/control/*`` endpoints do.
    # Default ``True`` so ``scripts/stats_snapshot.py`` works out of the
    # box. Set to ``false`` on a host whose URL is publicly exposed if
    # you prefer the trading-history file to require an explicit opt-in;
    # the JSON ``/*/since`` endpoints still return the same data, just
    # capped at 10k rows per table.
    db_download_enabled: bool = Field(
        default=True,
        alias="DB_DOWNLOAD_ENABLED",
    )
    # S3 bucket the bot writes its heartbeat to (so the operator
    # dashboard can show bot session uptime distinct from EC2 host
    # uptime). Same bucket used for log / snapshot uploads. Empty
    # string disables the heartbeat -- the bot still trades fine,
    # just the dashboard's "bot session" chip stays empty. Auto-
    # populated by terraform's cloud-init (writes the value derived
    # from ``${name_prefix}-logs-${account_id}``); operators can
    # also set it manually in /opt/dtc-mm-as/secrets/bot.env.
    logs_bucket: str = Field(default="", alias="LOGS_BUCKET")
    # Heartbeat publish cadence. 30s matches the dashboard's poll
    # cadence -- any faster would add S3 PUT cost without operator-
    # visible benefit (the chip ticks once per second locally
    # between server fetches anyway).
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="HEARTBEAT_INTERVAL_SECONDS",
    )
    # ---- Live stats (richer trading internals snapshot) ------------
    # Distinct from the heartbeat: the dashboard's Bot Stats tab needs
    # near-real-time visibility into working bid/ask, microprice,
    # basis, inventory skew, recent markouts. Heartbeat at 30s is too
    # stale; this publisher writes ``live_stats/<profile>.json`` to
    # the same bucket at a faster cadence. 5s default keeps S3 PUT
    # cost ~$2.60/month per bot.
    live_stats_enabled: bool = Field(
        default=True,
        alias="LIVE_STATS_ENABLED",
    )
    live_stats_interval_seconds: float = Field(
        default=5.0,
        gt=0,
        alias="LIVE_STATS_INTERVAL_SECONDS",
    )
    # 2026-05-14 (todo-028 / todo-032 prereq): periodic S3 publisher
    # for the session-scoped tables the dashboard needs to render
    # the Inventory / Gates / Execution-quality / Regimes panels.
    # Three S3 objects per profile, overwritten every cadence:
    #   dashboard/exposure_since_<profile>.json
    #   dashboard/fills_since_<profile>.json
    #   dashboard/orders_lifecycle_since_<profile>.json
    # Same gate/bucket pattern as the other S3 publishers — disabling
    # this flag is the operator's escape hatch if the publisher
    # misbehaves; the bot trades fine without it, the dashboard just
    # loses these specific panels.
    # 2026-05-14 BUG-024 — CRITICAL connectivity-failure guard.
    # When True (default), an ``unconfirmed`` place-response outcome
    # (venue gave neither a clean accept nor a clean reject) triggers
    # an immediate full kill: cancel-all, flatten (if safe), CRITICAL
    # Telegram push, dashboard KILLED chip. Per operator decree
    # (2026-05-14): missing ack/reject is NOT benign — it's a
    # critical connectivity failure. Even ONE occurrence per month
    # warrants stopping the bot.
    #
    # Setting to False restores the legacy silent-leave-in-SENT
    # behaviour that lets reconcile-by-cloid handle transient
    # "pending" states. ONLY appropriate on venues with a robust
    # reconcile-by-cloid path that has been proven not to leak
    # phantom rows (HL / GRVT historically; OKX has been leaking
    # 300-800 phantoms per day on the SUI session — that's exactly
    # the regression this guard catches).
    strict_place_unconfirmed_kill: bool = Field(
        default=True,
        alias="STRICT_PLACE_UNCONFIRMED_KILL",
    )
    observability_dashboard_publish_enabled: bool = Field(
        default=True,
        alias="OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED",
    )
    # 30s cadence chosen by spec (todo-028 § Prerequisites): the
    # inventory / gates / execution panels aggregate over minute-
    # to-hour windows. Sub-30s polling adds S3 PUT cost without
    # operator-visible benefit. Bot-side daemon thread; no impact
    # on the trading hot path.
    observability_dashboard_publish_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300.0,
        alias="OBSERVABILITY_DASHBOARD_PUBLISH_INTERVAL_SECONDS",
    )
    # Row caps per published object. The publisher windows to
    # session-to-date, so these caps protect the S3 payload size
    # when sessions run long (12h+ on TON in v1.3.5+ snapshots
    # showed ~10K exposure bars / ~30K orders-lifecycle rows).
    # Operator can tighten further per-profile via env.
    observability_dashboard_exposure_bars_limit: int = Field(
        default=2000,
        gt=0,
        le=20000,
        alias="OBSERVABILITY_DASHBOARD_EXPOSURE_BARS_LIMIT",
    )
    observability_dashboard_fills_limit: int = Field(
        default=2000,
        gt=0,
        le=20000,
        alias="OBSERVABILITY_DASHBOARD_FILLS_LIMIT",
    )
    observability_dashboard_orders_lifecycle_limit: int = Field(
        default=5000,
        gt=0,
        le=50000,
        alias="OBSERVABILITY_DASHBOARD_ORDERS_LIFECYCLE_LIMIT",
    )
    # 2026-05-13 regime-observability Phase 2: exposure-bar emitter.
    # Periodic snapshot of market + strategy state (independent of
    # whether a fill happened) — provides the "exposure denominator"
    # for per-regime PnL slicing.
    #
    # Feature-flagged so the operator can disable in production if
    # any unforeseen hot-path impact appears. The emitter runs on
    # its own daemon thread and reads state lock-free; in-process
    # impact should be negligible (see plan's hot-path-safety section).
    observability_exposure_bars_enabled: bool = Field(
        default=True,
        alias="OBSERVABILITY_EXPOSURE_BARS_ENABLED",
    )
    # Cadence in seconds. 5s default is the storage/insight trade-off
    # documented in the plan: 5s = ~10.8K rows / 15h session, captures
    # all meaningful regime shifts (no real regime flips faster than
    # ~5s on the symbols we trade). Drop to 1s for deep-dive sessions
    # if needed; raise to 10s+ for very long-running observations.
    observability_exposure_bar_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        alias="OBSERVABILITY_EXPOSURE_BAR_INTERVAL_SECONDS",
    )
    # 2026-05-13 regime-observability Phase 4a: expected_net_edge_bps
    # placeholder formula inputs. These are operator-configurable
    # priors that drive the per-order ``expected_net_edge_bps_at_decision``
    # stamp.
    #
    # IMPORTANT: this value is purely observational. The bot's quote-
    # construction path does NOT read it. See
    # ``plans/regime-observability.md`` Phase 4a scope guard.
    #
    # Real calibration is deferred: the v1 formula is
    #   target_half_spread_bps + maker_rebate_bps
    #     - typical_adverse_markout_bps
    # which is a useful first-order approximation but ignores regime-
    # dependent variation in adverse selection. Operator should
    # calibrate ``typical_adverse_markout_bps`` per symbol from
    # observed session data (the 0513-073423 TON snapshot showed
    # ~2-3 bp mean adverse, so 2.0 is a reasonable starting point).
    observability_maker_rebate_bps: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        alias="OBSERVABILITY_MAKER_REBATE_BPS",
    )
    observability_typical_adverse_markout_bps: float = Field(
        default=2.0,
        ge=0.0,
        le=20.0,
        alias="OBSERVABILITY_TYPICAL_ADVERSE_MARKOUT_BPS",
    )
    # 2026-05-13 regime-observability Phase 4c: MAE/MFE post-fill
    # excursion watcher. New subsystem — disable-by-default for the
    # first deploy so unforeseen impact can't surprise the operator.
    # Enable explicitly per profile once the v2 baseline is stable.
    observability_mae_mfe_enabled: bool = Field(
        default=False,
        alias="OBSERVABILITY_MAE_MFE_ENABLED",
    )
    observability_mae_mfe_poll_interval_seconds: float = Field(
        default=0.5,
        gt=0.0,
        le=10.0,
        alias="OBSERVABILITY_MAE_MFE_POLL_INTERVAL_SECONDS",
    )
    # 2026-05-13 regime-observability Phase 4c follow-up: time-to-flat
    # post-fill watcher. Tracks how long until ``state.position.position_qty``
    # crosses zero after each fill. Combined with markout / MAE / MFE
    # this isolates "how long was inventory exposed" — directly
    # supports residual-loss attribution.
    #
    # Disable-by-default for the first deploy (new subsystem). Same
    # pattern as MAE/MFE.
    observability_time_to_flat_enabled: bool = Field(
        default=False,
        alias="OBSERVABILITY_TIME_TO_FLAT_ENABLED",
    )
    # Cadence. 1s default is sufficient for most use cases — sub-
    # second precision on time-to-flat isn't operationally useful
    # given inter-fill spacing is typically 10s+.
    observability_time_to_flat_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=30.0,
        alias="OBSERVABILITY_TIME_TO_FLAT_POLL_INTERVAL_SECONDS",
    )
    # Max time to wait for a flat-crossing before giving up. Bounds
    # the watcher's in-flight memory. 300s (5 min) is plenty for
    # most markets — fills that don't flatten in 5 min are unusual
    # and the operator can investigate via the regular position
    # trace. Beyond this cap the row stays NULL (signal: "did not
    # flatten within window").
    observability_time_to_flat_max_wait_seconds: float = Field(
        default=300.0,
        gt=0.0,
        le=3600.0,
        alias="OBSERVABILITY_TIME_TO_FLAT_MAX_WAIT_SECONDS",
    )
    # Consecutive bot ticks without a healthy exchange snapshot (market + account when address set)
    # before auto-PAUSED(reconcile_stall). Cleared when snapshot is healthy again (auto-resume) or operator resume.
    exchange_reconcile_stall_ticks: int = Field(
        default=5,
        ge=1,
        alias="EXCHANGE_RECONCILE_STALL_TICKS",
    )

    @field_validator("live_trading_collateral_usd", mode="before")
    @classmethod
    def empty_collateral_to_none(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return v

    @field_validator(
        "hl_account_address",
        "hl_secret_key",
        "hl_secret_key_file",
        "grvt_api_key",
        "grvt_api_secret",
        "grvt_api_secret_file",
        "grvt_account_address",
        "grvt_sub_account_id",
        "grvt_env",
        "grvt_edge_url",
        "grvt_trade_url",
        "grvt_market_data_url",
        "grvt_public_ws_url",
        "grvt_private_ws_url",
        "bluefin_network",
        "bluefin_private_key",
        "bluefin_account_address",
        "bluefin_rest_url",
        "bluefin_auth_url",
        "bluefin_trade_url",
        "bluefin_public_ws_url",
        "bluefin_private_ws_url",
        "binance_api_key",
        "binance_api_secret",
        "binance_rest_url",
        "binance_private_ws_url",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("symbol", mode="before")
    @classmethod
    def symbol_nonempty(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                raise ValueError("SYMBOL must be non-empty")
            return s
        return v

    @field_validator("exchange", mode="before")
    @classmethod
    def exchange_allowed(cls, v: Any) -> Any:
        # Kept tolerant: accept legacy/shorthand "hl" as an alias for "hyperliquid" so
        # existing deployments don't trip if someone sets ``EXCHANGE=hl``.
        if isinstance(v, str):
            s = v.strip().lower()
            if not s:
                return "hyperliquid"
            if s == "hl":
                return "hyperliquid"
            if s not in {"hyperliquid", "grvt", "bluefin", "binance", "okx"}:
                raise ValueError(
                    "EXCHANGE must be one of 'hyperliquid' | 'grvt' | "
                    f"'bluefin' | 'binance' | 'okx' (got {v!r})"
                )
            return s
        return v

    @field_validator("reference_exchange", mode="before")
    @classmethod
    def reference_exchange_allowed(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            if not s:
                return "binance"
            if s not in {"binance", "bybit", "off"}:
                raise ValueError(
                    f"REFERENCE_EXCHANGE must be one of 'binance' | 'bybit' | 'off' (got {v!r})"
                )
            return s
        return v

    @field_validator("grvt_env", mode="before")
    @classmethod
    def grvt_env_allowed(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            if not s:
                return "prod"
            if s in {"prod", "testnet", "staging", "dev"}:
                return s
            raise ValueError(
                "GRVT_ENV must be one of 'prod' | 'testnet' | 'staging' | 'dev'"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def inject_hl_secret_key_from_file(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def _get(*keys: str) -> Any:
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
            return None

        env_key = _get("HL_SECRET_KEY", "hl_secret_key")
        if isinstance(env_key, str) and env_key.strip():
            return data
        path = _get("HL_SECRET_KEY_FILE", "hl_secret_key_file")
        if not isinstance(path, str) or not path.strip():
            return data
        path = path.strip()
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError("Could not read HL_SECRET_KEY_FILE") from e
        key = raw.strip()
        if not key:
            raise ValueError("HL_SECRET_KEY_FILE is empty after stripping whitespace")
        out = dict(data)
        out["HL_SECRET_KEY"] = key
        out["hl_secret_key"] = key
        return out

    @model_validator(mode="before")
    @classmethod
    def inject_grvt_secret_from_file(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def _get(*keys: str) -> Any:
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
            return None

        env_key = _get("GRVT_API_SECRET", "grvt_api_secret")
        if isinstance(env_key, str) and env_key.strip():
            return data
        path = _get("GRVT_API_SECRET_FILE", "grvt_api_secret_file")
        if not isinstance(path, str) or not path.strip():
            return data
        path = path.strip()
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError("Could not read GRVT_API_SECRET_FILE") from e
        key = raw.strip()
        if not key:
            raise ValueError("GRVT_API_SECRET_FILE is empty after stripping whitespace")
        out = dict(data)
        out["GRVT_API_SECRET"] = key
        out["grvt_api_secret"] = key
        return out

    @model_validator(mode="after")
    def validate_cross_venue_reference_self_match(self) -> Settings:
        """When trading on Binance, ``REFERENCE_EXCHANGE=binance`` is
        meaningless — we'd be measuring Binance against itself and the
        basis EWMA would be structurally always 0. Auto-override to
        ``off`` and log a warning so the operator notices the
        misconfig in the startup log.

        Reference docs: ``plans/20260420-binance-move/plan.md`` Phase
        1. The right pairing is:

        * ``EXCHANGE=binance``    → ``REFERENCE_EXCHANGE=off``
        * ``EXCHANGE=hyperliquid`` / ``grvt`` / ``bluefin``
                                  → ``REFERENCE_EXCHANGE=binance`` or ``bybit``
        """
        venue = (self.exchange or "").strip().lower()
        ref = (self.reference_exchange or "").strip().lower()
        if venue == "binance" and ref == "binance":
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "config_validation: EXCHANGE=binance with "
                "REFERENCE_EXCHANGE=binance is a misconfig (would "
                "measure Binance vs itself); auto-overriding "
                "REFERENCE_EXCHANGE=off. Set REFERENCE_EXCHANGE=off "
                "explicitly in your profile to silence this warning."
            )
            # Re-assign via __dict__ to bypass the frozen-after-validation
            # behaviour without re-running validators.
            object.__setattr__(self, "reference_exchange", "off")
        return self

    @model_validator(mode="after")
    def validate_numeric_bounds(self) -> Settings:
        if self.quote_loop_seconds <= 0:
            raise ValueError("QUOTE_LOOP_SECONDS must be positive")
        if self.reprice_threshold_bps <= 0:
            raise ValueError("REPRICE_THRESHOLD_BPS must be positive")
        if self.stale_data_warn_seconds <= 0:
            raise ValueError("STALE_DATA_WARN_SECONDS must be positive")
        if self.stale_data_kill_seconds <= self.stale_data_warn_seconds:
            raise ValueError(
                "STALE_DATA_KILL_SECONDS must be greater than STALE_DATA_WARN_SECONDS"
            )
        if self.min_half_spread_bps > self.max_half_spread_bps:
            raise ValueError("MIN_HALF_SPREAD_BPS cannot exceed MAX_HALF_SPREAD_BPS")
        if self.inventory_soft_limit_pct >= self.inventory_hard_limit_pct:
            raise ValueError("INVENTORY_SOFT_LIMIT_PCT must be < INVENTORY_HARD_LIMIT_PCT")
        if self.max_open_orders < 1:
            raise ValueError("MAX_OPEN_ORDERS must be >= 1")
        if self.vol_window_samples < 4:
            raise ValueError("VOL_WINDOW_SAMPLES must be >= 4")
        if self.max_order_notional_usd > self.max_position_notional_usd:
            raise ValueError(
                "MAX_ORDER_NOTIONAL_USD should not exceed MAX_POSITION_NOTIONAL_USD "
                "(a single fill would breach the position cap)"
            )
        if self.max_session_loss_usd <= 0 or self.max_drawdown_usd <= 0:
            raise ValueError("MAX_SESSION_LOSS_USD and MAX_DRAWDOWN_USD must be positive")
        if self.exchange_retry_max_backoff_seconds < self.exchange_retry_base_seconds:
            raise ValueError(
                "EXCHANGE_RETRY_MAX_BACKOFF_SECONDS must be >= EXCHANGE_RETRY_BASE_SECONDS"
            )
        if self.private_ws_reconnect_max_seconds < self.private_ws_reconnect_initial_seconds:
            raise ValueError(
                "PRIVATE_WS_RECONNECT_MAX_SECONDS must be >= PRIVATE_WS_RECONNECT_INITIAL_SECONDS"
            )
        if self.quote_max_distance_to_touch_ticks <= 0:
            raise ValueError("QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS must be positive")
        if self.quote_aging_tighten_ticks <= 0:
            raise ValueError("QUOTE_AGING_TIGHTEN_TICKS must be positive")
        return self

    def effective_sqlite_path(self) -> str:
        if self.database_url.startswith("sqlite:///"):
            path = self.database_url.replace("sqlite:///", "", 1)
            if path and path != ":memory:":
                return path
        return self.sqlite_path

    def sanitized_dict(self) -> dict[str, Any]:
        # Redact by name pattern, not by per-field enumeration. Anything
        # whose UPPER_SNAKE_CASE alias contains one of these substrings
        # is treated as a secret. The previous per-field allowlist
        # missed TELEGRAM_BOT_TOKEN and BINANCE_API_KEY -- both real
        # auth material that ended up in plaintext when /config was
        # captured by scripts/stats_snapshot.py snapshots. This pattern
        # match also auto-covers any future *_SECRET / *_TOKEN /
        # *_API_KEY field added downstream.
        d = self.model_dump(mode="json", by_alias=True)
        for k, v in list(d.items()):
            if v and _is_sensitive_key_name(k):
                d[k] = "***"
        return d


def require_trading_credentials_when_enabled(settings: Settings) -> None:
    """
    Enforce per-venue signing credentials when live trading is on.

    ``Settings`` parsing does **not** run this check so scripts (e.g.
    ``scripts/local_live_preflight.py``) can construct ``Settings()`` from the process
    environment (after optional ``APP_ENV_FILE`` / ``--env-file`` bootstrap) and print
    their own diagnostics. Call from the app lifespan before signed exchange use.

    Branches on ``settings.exchange``; unknown venues are rejected by the field validator
    before this is reached.
    """
    if not settings.trading_enabled:
        return
    venue = (settings.exchange or "").strip().lower()
    if venue == "hyperliquid":
        if not (settings.hl_secret_key or "").strip():
            raise ValueError(
                "HL_SECRET_KEY or HL_SECRET_KEY_FILE is required when TRADING_ENABLED=true "
                "and EXCHANGE=hyperliquid"
            )
        if not (settings.hl_account_address or "").strip():
            raise ValueError(
                "HL_ACCOUNT_ADDRESS is required when TRADING_ENABLED=true "
                "and EXCHANGE=hyperliquid"
            )
        return
    if venue == "grvt":
        if not (settings.grvt_api_key or "").strip():
            raise ValueError(
                "GRVT_API_KEY is required when TRADING_ENABLED=true and EXCHANGE=grvt"
            )
        if not (settings.grvt_api_secret or "").strip() and not (
            settings.grvt_api_secret_file or ""
        ).strip():
            raise ValueError(
                "GRVT_API_SECRET or GRVT_API_SECRET_FILE is required when "
                "TRADING_ENABLED=true and EXCHANGE=grvt"
            )
        if not (settings.grvt_sub_account_id or "").strip() and not (
            settings.grvt_account_address or ""
        ).strip():
            raise ValueError(
                "GRVT_SUB_ACCOUNT_ID (or legacy GRVT_ACCOUNT_ADDRESS) is required when "
                "TRADING_ENABLED=true and EXCHANGE=grvt"
            )
        return
    if venue == "bluefin":
        if not (settings.bluefin_private_key or "").strip():
            raise ValueError(
                "BLUEFIN_PRIVATE_KEY is required when TRADING_ENABLED=true and EXCHANGE=bluefin "
                "(with 1CT enabled this is the session key; without 1CT it is the main wallet key)"
            )
        if not (settings.bluefin_account_address or "").strip():
            raise ValueError(
                "BLUEFIN_ACCOUNT_ADDRESS is required when TRADING_ENABLED=true and EXCHANGE=bluefin "
                "(the main Sui wallet where USDC collateral / positions live)"
            )
        return
    if venue == "binance":
        if not (settings.binance_api_key or "").strip():
            raise ValueError(
                "BINANCE_API_KEY is required when TRADING_ENABLED=true and EXCHANGE=binance"
            )
        if not (settings.binance_api_secret or "").strip():
            raise ValueError(
                "BINANCE_API_SECRET is required when TRADING_ENABLED=true and EXCHANGE=binance"
            )
        return
    if venue == "okx":
        if not (settings.okx_api_key or "").strip():
            raise ValueError(
                "OKX_API_KEY is required when TRADING_ENABLED=true and EXCHANGE=okx"
            )
        if not (settings.okx_api_secret or "").strip():
            raise ValueError(
                "OKX_API_SECRET is required when TRADING_ENABLED=true and EXCHANGE=okx"
            )
        if not (settings.okx_api_passphrase or "").strip():
            raise ValueError(
                "OKX_API_PASSPHRASE is required when TRADING_ENABLED=true and EXCHANGE=okx "
                "(third secret set at API-key creation time)"
            )
        return
    # Defense in depth: the field validator should prevent this, but keep a readable
    # failure if Settings is constructed bypassing validation (e.g. ``model_construct``).
    raise ValueError(
        f"Unsupported EXCHANGE={settings.exchange!r} "
        "(expected 'hyperliquid' | 'grvt' | 'bluefin' | 'binance' | 'okx')"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
