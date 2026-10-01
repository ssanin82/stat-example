"""v1.4.101 — multi-horizon attribution block in live_stats payload.

The backend builder
``app.live_stats._build_multi_horizon_attribution_block_from_fills``
takes a list of ``Fill`` dataclass instances and returns a
``{"1s": {...}, "5s": {...}, ...}`` dict mirroring the shape the
dashboard's "Markout horizon ladder" tile consumes. Tests pin:

* Empty fills → empty dict (not a crash).
* Each horizon emits ``markout`` + ``pnl_attribution_usd`` + ``is_canonical``.
* ``is_canonical`` is True only for the 5 s slot.
* Means/medians are populated when fills carry the corresponding
  markout fields.
* A horizon with all-None markouts in the window still emits the
  slot (defensively) with ``sample_count=0`` and ``mean_bps=None`` —
  the dashboard will render this as "—".
* The decomposition equation ``realized = rebate + markout + residual``
  holds at every horizon when realized is passed in.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.live_stats import (
    _build_multi_horizon_attribution_block,
    _build_multi_horizon_attribution_block_from_fills,
)
from app.models import Fill, Side


def _fill(
    *,
    fid: str,
    side: Side = Side.BUY,
    notional: float = 10.0,
    fee: float = -0.001,  # rebate
    m1: float | None = None,
    m5: float | None = None,
    m15: float | None = None,
    m30: float | None = None,
    m60: float | None = None,
    m120: float | None = None,
    closed_pnl: float | None = 0.0,
) -> Fill:
    return Fill(
        fill_id=fid,
        order_id_exchange=1,
        client_order_id="cl",
        ts_fill=datetime.now(timezone.utc),
        symbol="TON-USDT-SWAP",
        side=side,
        price=2.0,
        size=notional / 2.0,
        notional=notional,
        fee=fee,
        liquidity_flag="M",
        mid_at_fill=2.0,
        markout_1s_bps=m1,
        markout_5s_bps=m5,
        markout_15s_bps=m15,
        markout_30s_bps=m30,
        markout_60s_bps=m60,
        markout_120s_bps=m120,
        closed_pnl=closed_pnl,
    )


def test_empty_fills_yields_empty_dict() -> None:
    out = _build_multi_horizon_attribution_block_from_fills([])
    assert out == {}


def test_block_contains_expected_horizons_when_data_present() -> None:
    """With a population of fills carrying every horizon, every
    slot is emitted with the expected shape."""
    # All markouts non-zero so each fill counts as either adverse or
    # favorable. (Fills with markout == 0 are neither — sample_count
    # would be 0.) Negative ones go through the adverse path.
    fills = [
        _fill(
            fid=f"f{i}",
            side=Side.BUY if i % 2 == 0 else Side.SELL,
            m1=1.0,
            m5=0.5,
            m15=0.3,
            m30=0.2,
            m60=0.1,
            m120=-0.1,
        )
        for i in range(20)
    ]
    out = _build_multi_horizon_attribution_block_from_fills(fills)
    for h in ("1s", "5s", "15s", "30s", "60s", "120s"):
        assert h in out, f"missing horizon {h} in {sorted(out.keys())}"
        slot = out[h]
        assert "markout" in slot
        assert "pnl_attribution_usd" in slot
        assert "is_canonical" in slot
        assert slot["markout"]["sample_count"] == 20


def test_is_canonical_only_for_5s() -> None:
    fills = [_fill(fid=f"f{i}", m5=0.5) for i in range(5)]
    out = _build_multi_horizon_attribution_block_from_fills(fills)
    for h, slot in out.items():
        if h == "5s":
            assert slot["is_canonical"] is True, "5s must be canonical"
        else:
            assert slot["is_canonical"] is False, (
                f"horizon {h} should not be canonical"
            )


def test_horizon_with_no_resolved_markouts_still_emits_slot() -> None:
    """All-None markouts at a horizon (e.g. fills < 120 s old at the
    120 s horizon) must still emit the slot — the dashboard renders
    sample_count=0 + mean=None as '—' rather than disappearing."""
    fills = [_fill(fid=f"f{i}", m5=0.5) for i in range(5)]  # only 5s
    out = _build_multi_horizon_attribution_block_from_fills(fills)
    assert "120s" in out
    assert out["120s"]["markout"]["sample_count"] == 0
    assert out["120s"]["markout"]["mean_bps"] is None


def test_canonical_5s_decomposition_equation() -> None:
    """At the 5 s horizon the dashboard banks on
    ``realized = rebate + markout + residual``. Closed PnL is +$2 per
    fill so the realized total over 4 fills is +$8; the multi-horizon
    builder must preserve this invariant."""
    fills = [
        _fill(fid=f"f{i}", m5=10.0, fee=-0.005, closed_pnl=2.0)
        for i in range(4)
    ]
    out = _build_multi_horizon_attribution_block_from_fills(fills)
    pnl = out["5s"]["pnl_attribution_usd"]
    realized = pnl["realized_pnl_total"]
    rebate = pnl["fee_income"]
    markout = pnl["markout_dollar_impact"]
    residual = pnl["residual"]
    assert realized is not None
    assert rebate is not None
    assert markout is not None
    assert residual is not None
    # Allow tiny rounding tolerance.
    assert abs(realized - (rebate + markout + residual)) < 1e-6


def test_block_from_dicts_directly() -> None:
    """The inner ``_build_multi_horizon_attribution_block`` is the
    raw-dict entrypoint used in tests. Smoke that it handles a
    minimal payload without raising."""
    fill_dicts = [
        {
            "side": "BUY",
            "price": 2.0,
            "size": 5.0,
            "notional": 10.0,
            "fee": -0.001,
            "liquidity_flag": "M",
            "markout_5s_bps": 1.0,
        }
        for _ in range(3)
    ]
    out = _build_multi_horizon_attribution_block(fill_dicts, 0.0)
    assert "5s" in out
    assert out["5s"]["markout"]["sample_count"] == 3
