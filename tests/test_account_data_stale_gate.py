"""TODO-002: account-data-stale risk gate.

A healthy public book combined with a stale account view is a dangerous
combination on Bluefin (BUG-006-style transient REST auth failure). This
gate fires when ``refresh_account_only`` hasn't completed for
``ACCOUNT_DATA_STALE_KILL_SECONDS`` and downgrades the action to
``NO_QUOTE/account_data_stale``. ``Bot._should_cancel_resting_on_no_quote``
includes ``account_data_stale`` in its severe set so resting orders are
cancelled.
"""

from __future__ import annotations

from app.bot import Bot
from app.enums import BotStatus, DesyncPhase, RiskAction, Side
from app.models import (
    BestBidAsk,
    PnlSnapshot,
    PositionSnapshot,
    ToxicitySnapshot,
)
from app.risk import evaluate_risk
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
        "MAX_OPEN_ORDERS": 4,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _market() -> BestBidAsk:
    return BestBidAsk(
        symbol="ETH_USDT_Perp",
        best_bid=100.0,
        best_ask=101.0,
        mid_price=100.5,
        spread_bps=100.0,
    )


def _pnl() -> PnlSnapshot:
    return PnlSnapshot(
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
        total_pnl_usd=0.0,
        fees_usd=0.0,
        equity_usd=1000.0,
        drawdown_usd=0.0,
        session_peak_equity_usd=1000.0,
    )


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def _evaluate(account_age_s, **overrides):
    settings = _settings(**overrides.pop("settings_overrides", {}))
    return evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=_market(),
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=_pnl(),
        toxicity=_tox(),
        execution_errors=0,
        desync=False,
        desync_phase=DesyncPhase.OK,
        desync_quarantine_remaining=0,
        trades_last_minute=0,
        reconcile_auto_pause=False,
        public_ws_live_path=False,
        public_ws_connected=True,
        public_ws_seconds_since_message=0.5,
        public_ws_seen_first_bbo=True,
        account_seconds_since_refresh=account_age_s,
        **overrides,
    )


def test_account_data_fresh_passes_through() -> None:
    """A 5-second-old account refresh is fine — no gate fires."""
    r = _evaluate(account_age_s=5.0)
    assert r.action == RiskAction.ALLOW
    assert "account_data_stale" not in r.reasons


def test_account_data_unknown_passes_through() -> None:
    """Pre-startup, the anchor is None — gate must not fire."""
    r = _evaluate(account_age_s=None)
    assert r.action == RiskAction.ALLOW


def test_account_data_stale_returns_no_quote() -> None:
    """Past the kill threshold (default 60s) → NO_QUOTE/account_data_stale."""
    r = _evaluate(account_age_s=70.0)
    assert r.action == RiskAction.NO_QUOTE
    assert "account_data_stale" in r.reasons


def test_account_data_stale_disabled_when_seconds_zero() -> None:
    """Setting the threshold to 0 disables the gate."""
    r = _evaluate(
        account_age_s=999.0,
        settings_overrides={"ACCOUNT_DATA_STALE_KILL_SECONDS": 0.0},
    )
    assert r.action == RiskAction.ALLOW
    assert "account_data_stale" not in r.reasons


def test_account_data_stale_promotes_to_cancel_resting() -> None:
    """``Bot._should_cancel_resting_on_no_quote`` must classify
    ``account_data_stale`` as severe, so the bot cancels resting orders
    rather than leaving them up against an unknown position view.

    Pre-1.2.79 this lived in an opt-in severe whitelist. v1.2.79
    inverted the policy: now ``account_data_stale`` cancels because
    it's NOT in the allow-list of safe-hold reasons (which defaults
    to ``recovery_cooldown,trade_rate_limit``). Either way, the
    user-visible behaviour is unchanged for this reason — it
    cancels.
    """
    assert Bot._should_cancel_resting_on_no_quote(["account_data_stale"]) is True


# ============================================================================
# todo-019 Part A — NO_QUOTE cancel inversion (v1.2.79).
# ============================================================================


def test_no_quote_cancel_empty_reasons_defensive_cancel() -> None:
    """Empty/missing reasons → cancel defensively. Guards against
    a malformed RiskAction.NO_QUOTE with no reasons attached
    (would otherwise vacuously satisfy ``all(...)`` and leave
    orders resting on an unknown trigger)."""
    assert Bot._should_cancel_resting_on_no_quote([]) is True


def test_no_quote_cancel_recovery_cooldown_alone_keeps_resting() -> None:
    """``recovery_cooldown`` is the canonical sub-second post-fill
    pause; existing order is fresh and re-placing would just churn.
    In the default allow-list → keep resting."""
    assert Bot._should_cancel_resting_on_no_quote(["recovery_cooldown"]) is False


def test_no_quote_cancel_trade_rate_limit_alone_keeps_resting() -> None:
    """``trade_rate_limit`` is the canonical 'about to re-emit anyway'
    reason. In the default allow-list → keep resting."""
    assert Bot._should_cancel_resting_on_no_quote(["trade_rate_limit"]) is False


def test_no_quote_cancel_all_allow_list_reasons_keeps_resting() -> None:
    """If every reason is in the allow-list, keep resting."""
    assert (
        Bot._should_cancel_resting_on_no_quote(
            ["recovery_cooldown", "trade_rate_limit"]
        )
        is False
    )


def test_no_quote_cancel_unknown_reason_cancels() -> None:
    """Any non-allow-list reason → cancel."""
    assert Bot._should_cancel_resting_on_no_quote(["toxicity_high"]) is True
    assert Bot._should_cancel_resting_on_no_quote(["vol_regime_off"]) is True
    assert Bot._should_cancel_resting_on_no_quote(["post_swing_active"]) is True


def test_no_quote_cancel_mixed_allow_and_unknown_cancels() -> None:
    """If ANY reason is outside the allow-list, cancel — even when
    one of the reasons would have kept resting on its own. This is
    intentional: a NO_QUOTE state with a real protective trigger
    (e.g. ``account_data_stale``) co-occurring with a benign one
    (e.g. ``recovery_cooldown``) is dominated by the protective
    trigger."""
    assert (
        Bot._should_cancel_resting_on_no_quote(
            ["recovery_cooldown", "account_data_stale"]
        )
        is True
    )


def test_no_quote_cancel_custom_allow_list_override() -> None:
    """The staticmethod accepts a custom allow-list arg (used by the
    bound instance method after reading the operator-configurable
    ``NO_QUOTE_KEEP_RESTING_REASONS`` env)."""
    custom = frozenset({"my_custom_safe_reason"})
    assert (
        Bot._should_cancel_resting_on_no_quote(
            ["my_custom_safe_reason"], custom
        )
        is False
    )
    # recovery_cooldown is NOT in the custom allow-list, so it cancels:
    assert (
        Bot._should_cancel_resting_on_no_quote(
            ["recovery_cooldown"], custom
        )
        is True
    )


def test_no_quote_cancel_empty_custom_allow_list_cancels_everything() -> None:
    """Empty allow-list reduces to pre-inversion behaviour: every
    NO_QUOTE cycle cancels. Used as a defensive fallback when the
    operator configures an empty ``NO_QUOTE_KEEP_RESTING_REASONS``."""
    empty: frozenset[str] = frozenset()
    assert (
        Bot._should_cancel_resting_on_no_quote(["recovery_cooldown"], empty)
        is True
    )
