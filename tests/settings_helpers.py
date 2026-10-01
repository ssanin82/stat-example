from pydantic import Field
from pydantic_settings import SettingsConfigDict

from app.config import Settings


class UnitTestSettings(Settings):
    """Settings without reading a dotenv file; env_prefix avoids OS `TRADING_ENABLED` etc."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="HLMMTEST_",
        extra="ignore",
    )
    # Immediate outbound flush (no batching delay) keeps execution tests deterministic.
    action_batch_interval_ms: float = Field(default=0.0, alias="ACTION_BATCH_INTERVAL_MS")
    action_ws_enabled: bool = Field(default=False, alias="ACTION_WS_ENABLED")
    # BUG-013 test-suite hygiene: production default for the kill auto-restart
    # grace is 5 s, but ``threading.Timer(5.0, os._exit)`` scheduled inside a
    # test fires while pytest is still running (the timer is daemon=True so
    # it doesn't BLOCK exit, but it fires on schedule as long as the process
    # is alive — which a long pytest run is). With 5 s default the full
    # suite blew up with EXIT=42 mid-run. Default to 0 in tests (auto-restart
    # disabled); the BUG-013 tests explicitly override to a small grace and
    # inject an ``exit_fn`` recorder to assert on the call.
    kill_auto_restart_grace_seconds: float = Field(
        default=0.0, alias="KILL_AUTO_RESTART_GRACE_SECONDS"
    )
    # 1.4.6: opt-out of always-batch in tests by default so the
    # legacy single-place mock (place_post_only_limit) path is
    # exercised. The batch-place adapter has its own test coverage
    # in tests/test_okx_responses_place_batch.py — execution-layer
    # tests use the HL-shaped MagicMock which doesn't carry a
    # batch_place_post_only_limit stub. Production defaults stay
    # True (in app/config.py).
    batch_places_enabled: bool = Field(default=False, alias="BATCH_PLACES_ENABLED")
    batch_places_always: bool = Field(default=False, alias="BATCH_PLACES_ALWAYS")
