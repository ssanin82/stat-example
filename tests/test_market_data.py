from __future__ import annotations

from unittest.mock import MagicMock

from app.market_data import refresh_account_only
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def test_refresh_account_only_skips_exchange_when_address_empty() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    state = BotState(settings)
    client = MagicMock()
    refresh_account_only(client, state, "", None, None)
    client.fetch_best_bid_ask.assert_not_called()
    client.fetch_position.assert_not_called()
    client.fetch_account_snapshot.assert_not_called()
    client.fetch_recent_fills_raw.assert_not_called()
    assert state.position.position_qty == 0.0
    assert state.account is None
