from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.enums import Side
from app.models import Fill
from app.toxicity import ToxicityEngine


def _fill_with_markout(fill_id: str, m5: float) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        markout_1s_bps=-50.0,
        markout_3s_bps=-50.0,
        markout_5s_bps=m5,
    )


def test_toxicity_uses_delayed_markouts_not_mid_at_fill() -> None:
    s = Settings.model_construct(
        toxicity_enabled=True,
        toxicity_markout_soft_bps=3.0,
        toxicity_markout_hard_bps=12.0,
        toxicity_one_sided_fill_ratio=0.75,
    )
    eng = ToxicityEngine(s)
    eng.set_baseline_vol(1.0)
    # Strongly adverse delayed markout but "good" if one wrongly used mid_at_fill vs current mid
    f = _fill_with_markout("a", -100.0)
    snap = eng.snapshot(mid=200.0, current_vol_bps=1.0, fills=[f])
    assert snap.adverse_uses_delayed_markouts is True
    assert snap.delayed_markout_sample_count == 1
    assert snap.avg_adverse_markout_bps < -12.0
    assert snap.hard_trigger is True


def test_toxicity_without_delayed_markouts_skips_markout_leg() -> None:
    s = Settings.model_construct(
        toxicity_enabled=True,
        toxicity_markout_soft_bps=3.0,
        toxicity_markout_hard_bps=12.0,
        toxicity_one_sided_fill_ratio=0.75,
    )
    eng = ToxicityEngine(s)
    eng.set_baseline_vol(1.0)
    f = Fill(
        fill_id="b",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
    )
    snap = eng.snapshot(mid=50.0, current_vol_bps=1.0, fills=[f])
    assert snap.adverse_uses_delayed_markouts is False
    assert snap.delayed_markout_sample_count == 0
    assert snap.avg_adverse_markout_bps == 0.0
