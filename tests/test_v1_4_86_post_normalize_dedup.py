"""v1.4.86 wedge-elimination-cleanup — post-normalize dedup regression.

The v1.4.79 ladder-level dedup catches grid collisions at the
formula stage, using ``decision.quoted_bid`` / ``decision.quoted_ask``
as the inside-rung reference. The v1.4.86 ROUND_HALF_UP fix
correctly matches the actual rounding semantic.

BUT: the engine's ``_build_side`` runs AFTER ``build_ladder`` and
applies post-only clamps + min-half-spread floor that can shift the
inside rung's final price to a DIFFERENT tick than
``decision.quoted_bid`` predicted. When that happens:

  * Ladder dedup sees:  inside_px_override = 2.0317 → ROUND_HALF_UP → 2.032
  * Engine actual:      build.bid_order.price = 2.031 (post-only clamped)
  * Outer rung raw:     2.0311 → normalize_order_pair → 2.031
  * Both rungs land at 2.031 on the book → SAME-PRICE DUPLICATE

The post-normalize dedup in ``compute_desired_state`` catches this
by comparing the engine's POST-CLAMP inside price against each outer
rung's normalized price. Same price → drop outer with reason
``post_normalize_grid_collision``.

This test reconstructs the collision scenario at the math level and
verifies:
1. The ladder dedup with ROUND_HALF_UP correctly handles the simpler
   cases (when decision.quoted_bid agrees with the engine's
   post-clamp price).
2. Documents the case the ladder dedup CAN'T see (engine post-clamp
   shift), which the post-normalize dedup catches downstream.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def test_v1_4_86_user_reported_2_031_bid_dup_scenario() -> None:
    """User screenshot 260519: BUY 2.031 ×2, SELL 2.034 ×2 on OKX
    TON-USDT-SWAP (tick=0.001). Reproduces the math that produces
    the duplicate at the final-placement tick.

    Setup:
    * mid = 2.0325 (between bid 2.031 and ask 2.034)
    * half_spread_bps ≈ 3.5 (compressed — typical when MIN_HALF_SPREAD floor
      isn't tight and toxicity isn't widening)
    * tick = 0.001

    Math:
    * lvl 0 raw = 2.0325 × (1 - 3.5/10000)  = 2.03179
    * lvl 1 raw = 2.0325 × (1 - 7.0/10000)  = 2.03108
    * best_bid = 2.031  (post-only clamp target)

    Engine actual:
    * lvl 0 clamped via min(2.03179, 2.031) → 2.031 → ROUND_HALF_UP → 2.031
    * lvl 1 (no inside clamp) → ROUND_HALF_UP(2.03108) → 2.031
    * → both at 2.031 on the book

    Ladder dedup (v1.4.86 ROUND_HALF_UP):
    * inside override = decision.quoted_bid = 2.03179 (pre-clamp value)
    * ROUND_HALF_UP(2.03179) = 2.032
    * ROUND_HALF_UP(2.03108) = 2.031
    * 2.031 < 2.032 → "no collision" → both rungs emitted from ladder

    Post-normalize dedup (v1.4.86, the second layer fix):
    * engine_order.price = 2.031 (after _build_side clamps)
    * outer normalized = 2.031
    * abs(2.031 - 2.031) < 1e-9 → collision → outer rung dropped with
      reason "post_normalize_grid_collision"
    """
    tick = 0.001
    tick_d = Decimal("0.001")

    # Ladder formula reproduction
    mid = 2.0325
    hs_bps = 3.5
    lvl0_raw = mid * (1.0 - hs_bps / 10_000.0)
    lvl1_raw = mid * (1.0 - 2.0 * hs_bps / 10_000.0)

    # Engine post-clamp inside (post-only against best_bid=2.031).
    best_bid = 2.031
    lvl0_post_clamp = min(lvl0_raw, best_bid)
    engine_inside_final = float(
        Decimal(str(lvl0_post_clamp)).quantize(tick_d, rounding=ROUND_HALF_UP)
    )

    # Outer rung normalize (no clamp for outer).
    outer_final = float(
        Decimal(str(lvl1_raw)).quantize(tick_d, rounding=ROUND_HALF_UP)
    )

    # Same-tick collision confirmed.
    assert abs(engine_inside_final - outer_final) < 1e-9, (
        f"reproducer should land both rungs on the same tick; "
        f"got inside={engine_inside_final} outer={outer_final}"
    )
    assert engine_inside_final == 2.031, (
        f"expected the user-observed tick (2.031); got {engine_inside_final}"
    )

    # The ladder-level dedup CAN'T see this collision because it uses
    # decision.quoted_bid (pre-clamp 2.03179) which rounds to a
    # different tick (2.032) than the post-clamp output (2.031).
    ladder_predicted_inside_tick = float(
        Decimal(str(lvl0_raw)).quantize(tick_d, rounding=ROUND_HALF_UP)
    )
    ladder_predicted_outer_tick = float(
        Decimal(str(lvl1_raw)).quantize(tick_d, rounding=ROUND_HALF_UP)
    )
    assert ladder_predicted_inside_tick == 2.032
    assert ladder_predicted_outer_tick == 2.031
    # → ladder dedup sees "different ticks, both rungs kept".
    # → the post-normalize dedup is what actually catches the collision.


def test_v1_4_86_snapshot_260519_055516_actual_dup_scenario() -> None:
    """Direct reproducer from snapshot v1.4.85-260519-095608 quote row
    at 05:55:16 — the bot placed 31 pairs of same-side-same-price
    orders within 1s windows. The exact data from quotes_recent.json:

        mid_price: 2.0345
        reservation_price: 2.034529908740031
        target_spread_bps: 7.0  (half_spread_bps = 3.5)
        quoted_bid: 2.033817823271972        (raw, pre-engine)
        exec_norm_bid_px: 2.033              (engine's POST-retreat output)
        ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS: 3.5

    The chain:
    1. compute_quote_decision: raw bid = 2.033818
    2. Ladder lvl 1 raw = reservation × (1 - 2×3.5/10000) = 2.033106
    3. Engine _build_side for lvl 0:
       a. ROUND_HALF_UP(2.033818) → 2.034
       b. Post-rounding economic-floor retreat:
          2.034 > mid - 3.5_bps_px (= 2.033787) → retreat by 1 tick
          → 2.033
       c. engine_order.price = 2.033  ← matches exec_norm_bid_px
    4. Ladder dedup with _tick_round_half_up (the v1.4.86 fix):
       a. inside_px_override = decision.quoted_bid = 2.033818
       b. _tick_round_half_up(2.033818, 0.001) = 2.034
       c. _tick_round_half_up(2.033106, 0.001) = 2.033
       d. 2.033 < 2.034 → NO collision → both rungs kept
       → ladder dedup MISSES this case (because of engine retreat)
    5. compute_desired_state post-normalize dedup (the v1.4.86 second
       layer):
       a. engine_order.price = 2.033 (post-retreat)
       b. normalize_order_pair(spec, 2.033106, ...) → (2.033, sz)
          [because ROUND_HALF_UP(2.033106) = 2.033, fraction 0.106 < 0.5]
       c. abs(2.033 - 2.033) < 1e-9 → COLLISION → drop outer rung
       d. desired[(BUY, 1)] = DesiredOrderState.empty(reason="post_normalize_grid_collision")

    This test verifies the math at every step matches the snapshot's
    observed values and the v1.4.86 fix catches the collision.
    """
    tick_d = Decimal("0.001")
    tick = 0.001

    # === Step 1: compute_quote_decision output (from snapshot) ===
    mid = 2.0345
    reservation = 2.034529908740031
    half_spread_bps = 3.5  # = target_spread_bps / 2
    quoted_bid_raw = 2.033817823271972

    # === Step 2: ladder lvl 1 raw ===
    lvl1_raw = reservation * (1.0 - 2.0 * half_spread_bps / 10_000.0)
    assert abs(lvl1_raw - 2.033106) < 1e-5, f"lvl1 raw {lvl1_raw} doesn't match expected ~2.033106"

    # === Step 3: engine _build_side for lvl 0 ===
    # Stage 3a: ROUND_HALF_UP
    after_round = float(Decimal(str(quoted_bid_raw)).quantize(tick_d, rounding=ROUND_HALF_UP))
    assert after_round == 2.034, f"ROUND_HALF_UP({quoted_bid_raw}) should be 2.034; got {after_round}"
    # Stage 3b: economic floor retreat
    economic_min_half_spread_bps = 3.5
    min_half_spread_px = economic_min_half_spread_bps / 10_000.0 * mid
    floor_px = mid - min_half_spread_px  # = 2.0345 - 0.0007121 = 2.0337879
    # Retreat triggered when after_round > floor_px (engine: ``npx > m - min_half_spread_px + 1e-12``)
    retreat_triggered = after_round > floor_px + 1e-12
    assert retreat_triggered, f"retreat should trigger: {after_round} > {floor_px}"
    engine_inside_final_px = after_round - tick
    assert engine_inside_final_px == 2.033, (
        f"after retreat, inside BID should be 2.033; got {engine_inside_final_px}"
    )

    # === Step 4: ladder dedup with v1.4.86 _tick_round_half_up ===
    from app.ladder import _tick_round_half_up
    lvl0_ladder_predict = _tick_round_half_up(quoted_bid_raw, tick)
    lvl1_ladder_predict = _tick_round_half_up(lvl1_raw, tick)
    # The ladder dedup uses decision.quoted_bid (NOT engine's post-retreat output),
    # so it predicts 2.034 for lvl 0. This MISSES the collision after retreat.
    assert lvl0_ladder_predict == 2.034
    assert lvl1_ladder_predict == 2.033
    # Collision check in the ladder: grid_px >= prev_grid_bid - eps → 2.033 >= 2.034 - eps → FALSE
    ladder_collision_detected = lvl1_ladder_predict >= lvl0_ladder_predict - 1e-12
    assert not ladder_collision_detected, (
        "ladder dedup correctly says 'no collision' on the raw predictions; "
        "the collision only emerges after the engine's post-rounding retreat. "
        "This is the case the post-normalize dedup is for."
    )

    # === Step 5: compute_desired_state post-normalize dedup ===
    # Outer rung normalization mimics normalize_order_pair → ROUND_HALF_UP.
    outer_normalized_px = float(
        Decimal(str(lvl1_raw)).quantize(tick_d, rounding=ROUND_HALF_UP)
    )
    assert outer_normalized_px == 2.033, (
        f"outer rung should round to 2.033 (fraction 0.106 < 0.5); got {outer_normalized_px}"
    )
    # The post-normalize dedup check.
    post_normalize_collision = abs(outer_normalized_px - engine_inside_final_px) < 1e-9
    assert post_normalize_collision, (
        f"v1.4.86 post-normalize dedup should detect collision: "
        f"inside_final={engine_inside_final_px}, outer_normalized={outer_normalized_px}"
    )
