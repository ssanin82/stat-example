"""Account equity parsing: perps clearinghouse + unified-account spot USDC fallback."""

from __future__ import annotations

from app.exchange.hyperliquid_client import (
    _merge_account_snapshot_balances,
    _perps_equity_and_withdrawable_from_user_state,
    _usdc_from_spot_clearinghouse_payload,
)


def test_perps_prefers_max_of_margin_and_cross() -> None:
    st = {
        "withdrawable": "100.0",
        "marginSummary": {"accountValue": "10.0"},
        "crossMarginSummary": {"accountValue": "50.0"},
    }
    eq, wd = _perps_equity_and_withdrawable_from_user_state(st)
    assert eq == 50.0
    assert wd == 100.0


def test_spot_usdc_parses_total_and_free() -> None:
    raw = {
        "balances": [
            {"coin": "PURR", "total": "1", "hold": "0"},
            {"coin": "USDC", "total": "421.13", "hold": "0.01"},
        ]
    }
    total, free = _usdc_from_spot_clearinghouse_payload(raw)
    assert abs(total - 421.13) < 1e-9
    assert abs(free - 421.12) < 1e-9


def test_perps_zero_cross_present() -> None:
    st = {
        "withdrawable": "0",
        "marginSummary": {"accountValue": "0.0"},
        "crossMarginSummary": {"accountValue": "0"},
    }
    eq, _ = _perps_equity_and_withdrawable_from_user_state(st)
    assert eq == 0.0


def test_merge_prefers_spot_when_perps_understates_unified() -> None:
    eq, wd = _merge_account_snapshot_balances(
        perps_eq=0.0,
        perps_wd=0.0,
        spot_total=421.13,
        spot_free=420.0,
    )
    assert abs(eq - 421.13) < 1e-9
    assert wd == 420.0


def test_merge_keeps_perps_when_higher_standard_account() -> None:
    eq, wd = _merge_account_snapshot_balances(
        perps_eq=5000.0,
        perps_wd=1000.0,
        spot_total=12.5,
        spot_free=12.0,
    )
    assert eq == 5000.0
    assert wd == 1000.0


def test_merge_withdrawable_max_of_both() -> None:
    eq, wd = _merge_account_snapshot_balances(
        perps_eq=100.0,
        perps_wd=0.0,
        spot_total=100.0,
        spot_free=95.5,
    )
    assert eq == 100.0
    assert abs(wd - 95.5) < 1e-9
