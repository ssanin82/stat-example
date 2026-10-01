"""One-shot startup log lines for safer live launches (not a subsystem)."""

from __future__ import annotations

import logging

from app.config import Settings
from app.exchange.factory import venue_account_address
from app.exchange.hyperliquid_precision import HL_PERP_LIMIT_PRICE_PIPELINE_ID, HL_PERP_MAX_SIG_FIGS
from app.exchange.mm_client_protocol import HyperliquidMMClient

logger = logging.getLogger(__name__)


def log_startup_readiness(settings: Settings, client: HyperliquidMMClient) -> None:
    write_ok = client.has_write_access()
    # Use the venue-aware address resolver so this works on Bluefin /
    # GRVT / Hyperliquid uniformly. BUG-008: previously hardcoded
    # ``hl_account_address`` which is empty on non-HL venues, making
    # the readiness check silently skip the collateral fetch.
    addr = venue_account_address(settings)
    addr_ok = bool((addr or "").strip())
    logger.info(
        "startup_readiness trading_enabled=%s account_address_configured=%s "
        "exchange_write_access=%s control_endpoints_enabled=%s",
        settings.trading_enabled,
        addr_ok,
        write_ok,
        settings.control_endpoints_enabled,
    )
    logger.info(
        "startup_readiness trading_gate ok_for_writes=%s "
        "(requires trading_enabled=true and exchange_write_access=true to place orders)",
        bool(settings.trading_enabled and write_ok),
    )
    cfg_collat = settings.live_trading_collateral_usd
    logger.info(
        "startup_readiness live_trading_collateral_usd_config=%s",
        cfg_collat if cfg_collat is not None else "not_set",
    )

    snap_eq: float | None = None
    snap_wd: float | None = None
    if addr_ok:
        try:
            snap = client.fetch_account_snapshot(addr.strip())
            snap_eq = snap.equity_usd
            snap_wd = snap.withdrawable_usd
        except Exception as e:
            logger.warning(
                "startup_readiness account_snapshot_unavailable error=%s",
                str(e)[:200],
            )

    basis: float | None = None
    if snap_eq is not None:
        basis = snap_eq
    elif snap_wd is not None:
        basis = snap_wd
    elif cfg_collat is not None:
        basis = cfg_collat

    sufficient: bool | None = None
    if basis is not None:
        sufficient = basis >= settings.max_order_notional_usd

    logger.info(
        "startup_readiness collateral fetched_equity_usd=%s fetched_withdrawable_usd=%s "
        "collateral_basis_usd_for_cap_check=%s max_order_notional_usd=%s "
        "collateral_sufficient_for_max_order_notional=%s",
        snap_eq,
        snap_wd,
        basis,
        settings.max_order_notional_usd,
        sufficient,
    )

    sp = client.symbol_spec
    # The log-line prefix ("hl_symbol_meta", "hl_limit_price_rules", …) is retained
    # when the active venue is Hyperliquid so existing operator dashboards / tests
    # keep matching. For non-HL venues the prefix is venue-specific (e.g. ``grvt_``)
    # and the HL-only sig-figs pipeline line is omitted to avoid misleading
    # operators into thinking HL rules are being enforced.
    venue = (settings.exchange or "").strip().lower() or "hyperliquid"
    if venue in ("", "hl"):
        venue = "hyperliquid"
    prefix = "hl" if venue == "hyperliquid" else venue
    logger.info(
        "startup_readiness %s_symbol_meta symbol=%s meta_decimal_grid_price_tick=%s "
        "meta_decimal_size_step=%s min_size=%s min_notional_usd=%s sz_decimals=%s "
        "spec_source=%s spec_from_exchange=%s",
        prefix,
        settings.symbol,
        sp.price_tick,
        sp.size_step,
        sp.min_size,
        sp.min_notional_usd,
        sp.sz_decimals,
        sp.source,
        client.symbol_spec_fetched_ok,
    )
    if venue == "hyperliquid":
        logger.info(
            "startup_readiness hl_limit_price_rules symbol=%s hl_max_sig_figs_nonint=%s "
            "hl_price_normalize_pipeline=%s note=limit_px_must_satisfy_meta_decimal_grid_and_sigfigs",
            settings.symbol,
            int(HL_PERP_MAX_SIG_FIGS),
            HL_PERP_LIMIT_PRICE_PIPELINE_ID,
        )
    else:
        logger.info(
            "startup_readiness %s_limit_price_rules symbol=%s "
            "price_tick=%s price_normalize_pipeline=venue_tick_grid "
            "note=limit_px_must_satisfy_price_tick_grid",
            prefix,
            settings.symbol,
            sp.price_tick,
        )
    logger.info(
        "startup_readiness %s_order_sizing_floors symbol=%s min_size=%s min_notional_usd=%s",
        prefix,
        settings.symbol,
        sp.min_size,
        sp.min_notional_usd,
    )
    # Sanity: MIN_QUOTE_NOTIONAL_USD is our own internal floor; it MUST NOT be below
    # the venue's ``min_notional_usd`` or we'll silently generate orders the venue
    # rejects (and the quote engine's self-heal treats the venue floor as the true
    # minimum anyway). Emit a WARNING so operators can bump the config.
    try:
        local_floor = float(settings.min_quote_notional_usd)
        venue_floor = float(sp.min_notional_usd)
    except (TypeError, ValueError):
        local_floor = venue_floor = 0.0
    if venue_floor > 0 and local_floor > 0 and local_floor + 1e-9 < venue_floor:
        logger.warning(
            "startup_readiness MIN_QUOTE_NOTIONAL_USD=%s < venue min_notional_usd=%s — "
            "venue floor will be enforced by QuoteEngine self-heal. Raise "
            "MIN_QUOTE_NOTIONAL_USD to at least %s for config clarity.",
            local_floor,
            venue_floor,
            venue_floor,
        )
    # Also sanity: max_order_notional_usd must be >= venue min_notional_usd, otherwise
    # every placement will be infeasible (venue min exceeds per-order cap).
    try:
        max_order = float(settings.max_order_notional_usd)
    except (TypeError, ValueError):
        max_order = 0.0
    if venue_floor > 0 and max_order > 0 and max_order + 1e-9 < venue_floor:
        logger.error(
            "startup_readiness MAX_ORDER_NOTIONAL_USD=%s < venue min_notional_usd=%s — "
            "NO ORDER CAN BE PLACED. Raise MAX_ORDER_NOTIONAL_USD above the venue floor.",
            max_order,
            venue_floor,
        )
    logger.info(
        "startup_readiness caps max_abs_position=%s max_position_notional_usd=%s "
        "max_order_notional_usd=%s max_open_orders=%s max_session_loss_usd=%s "
        "max_drawdown_usd=%s stale_warn_s=%s stale_kill_s=%s",
        settings.max_abs_position,
        settings.max_position_notional_usd,
        settings.max_order_notional_usd,
        settings.max_open_orders,
        settings.max_session_loss_usd,
        settings.max_drawdown_usd,
        settings.stale_data_warn_seconds,
        settings.stale_data_kill_seconds,
    )


def log_runtime_config(settings: Settings) -> None:
    """One structured startup line with effective runtime knobs for run-to-run diffing."""
    logger.info(
        "bot_runtime_config app_env=%s log_level=%s db_path=%s symbol=%s trading_enabled=%s "
        "quote_loop_seconds=%s account_rest_min_interval_seconds=%s "
        "order_state_uncertainty_account_rest_interval_seconds=%s "
        "unhealthy_account_rest_min_interval_seconds=%s "
        "open_orders_reconcile_request_interval_seconds=%s open_orders_reconcile_cooldown_seconds=%s "
        "open_orders_reconcile_429_backoff_seconds=%s desync_reconcile_interval_seconds=%s "
        "desync_reconcile_429_backoff_seconds=%s cancel_pending_unresolved_timeout_seconds=%s "
        "side_unresolved_suppression_timeout_seconds=%s private_ws_keepalive_seconds=%s "
        "private_ws_inactive_reconnect_seconds=%s private_ws_idle_warn_seconds=%s "
        "public_ws_stale_warn_seconds=%s public_ws_stale_kill_seconds=%s "
        "market_data_stale_warn_seconds=%s market_data_stale_kill_seconds=%s "
        "execution_latency_warn_ms=%s execution_latency_degrade_ms=%s max_trades_per_minute=%s "
        "quote_notional_usd=%s min_quote_notional_usd=%s max_abs_position=%s max_open_orders=%s "
        "inventory_skew_coeff_bps=%s inventory_soft_limit_pct=%s inventory_hard_limit_pct=%s "
        "inventory_execution_bias_ratio=%s inventory_execution_bias_min_util_pct=%s "
        "toxicity_enabled=%s toxicity_markout_soft_bps=%s toxicity_markout_hard_bps=%s "
        "toxicity_one_sided_fill_ratio=%s toxicity_cooldown_seconds=%s",
        settings.app_env,
        settings.log_level,
        settings.effective_sqlite_path(),
        settings.symbol,
        settings.trading_enabled,
        settings.quote_loop_seconds,
        settings.account_rest_min_interval_seconds,
        settings.order_state_uncertainty_account_rest_interval_seconds,
        settings.unhealthy_account_rest_min_interval_seconds,
        settings.open_orders_reconcile_request_interval_seconds,
        settings.open_orders_reconcile_cooldown_seconds,
        settings.open_orders_reconcile_429_backoff_seconds,
        settings.desync_reconcile_interval_seconds,
        settings.desync_reconcile_429_backoff_seconds,
        settings.cancel_pending_unresolved_timeout_seconds,
        settings.side_unresolved_suppression_timeout_seconds,
        settings.private_ws_app_keepalive_seconds,
        settings.private_ws_inactive_reconnect_seconds,
        settings.private_ws_idle_warn_seconds,
        settings.public_ws_stale_warn_seconds,
        settings.public_ws_stale_kill_seconds,
        settings.stale_data_warn_seconds,
        settings.stale_data_kill_seconds,
        settings.execution_latency_warn_ms,
        settings.execution_latency_degrade_ms,
        settings.max_trades_per_minute,
        settings.quote_notional_usd,
        settings.min_quote_notional_usd,
        settings.max_abs_position,
        settings.max_open_orders,
        settings.inventory_skew_coeff_bps,
        settings.inventory_soft_limit_pct,
        settings.inventory_hard_limit_pct,
        settings.inventory_execution_bias_ratio,
        settings.inventory_execution_bias_min_util_pct,
        settings.toxicity_enabled,
        settings.toxicity_markout_soft_bps,
        settings.toxicity_markout_hard_bps,
        settings.toxicity_one_sided_fill_ratio,
        settings.toxicity_cooldown_seconds,
    )
