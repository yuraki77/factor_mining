"""
Verified hypotheses from back_lab first-principles research.

Each hypothesis includes:
  - Discovery evidence (IC, walk-forward, regime behavior)
  - Boundary conditions (when to STOP mining this family)
  - Failure modes (what makes this hypothesis invalid)

Boundary condition rules:
  1. MAX_TRIALS: cumulative trials in this family > threshold → deprioritize
  2. FDR_FAIL: FDR-adjusted p-value > 0.10 after N trials → stop
  3. IC_DECAY: walk-forward OOS IC drops below 25% of in-sample → stop
  4. COST_KILLER: break-even cost < 2× actual cost → not tradable
  5. SHORT_FAIL: short-side SR < -2.0 consistently → only long allowed
"""
from factor_mining.models import HypothesisSpec


def discovered_hypotheses() -> list[HypothesisSpec]:
    """Return all verified hypotheses from back_lab first-principles research."""
    return [
        # ── OHLCV-based ─────────────────────────────────────────────
        HypothesisSpec(
            hypothesis_id="h_close_position_trending",
            hypothesis_family="mean_reversion",
            economic_mechanism=(
                "In trending markets, extreme K-line close positions signal counter-trend "
                "exhaustion. When close is near the high in an uptrend, buyers are depleted "
                "→ selling pressure next bar. Effect is 3× stronger in trending vs ranging. "
                "Counter to textbook: mean-reversion works BETTER in trends for BTC 5m."
            ),
            testable_prediction="close_position IC < 0 in trending regime, IC ≈ 0 in ranging regime.",
            null_hypothesis="close_position has zero IC regardless of regime.",
            expected_ic_range=(0.005, 0.015),
            expected_decay_halflife_bars=24,
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_H1_corrected",
        ),
        HypothesisSpec(
            hypothesis_id="h_trend_acceleration",
            hypothesis_family="trend_following",
            economic_mechanism=(
                "Extreme K-line positions in the direction of the 4h trend signal trend "
                "CONTINUATION, not exhaustion. High vol amplifies this effect (IC=+0.023 "
                "in extreme vol vs +0.001 in low vol). The original exhaustion reversal "
                "hypothesis was FALSIFIED — data showed the opposite sign."
            ),
            testable_prediction="cp_directional × vol_ratio IC > 0, stronger in high vol.",
            null_hypothesis="Directional close position has zero IC.",
            expected_ic_range=(0.005, 0.025),
            expected_decay_halflife_bars=48,
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_F1_corrected",
        ),
        HypothesisSpec(
            hypothesis_id="h_panic_bid_5m",
            hypothesis_family="volume_confirmation",
            economic_mechanism=(
                "Extreme volume (>2× daily avg) combined with down move = panic liquidation. "
                "Market makers absorb panic sells → immediate 5m bounce. Effect is strongest "
                "at 5m horizon (IC=+0.012) and decays to zero by 4h. FALSIFIED original "
                "hypothesis that rebound is delayed — it's instantaneous."
            ),
            testable_prediction="vol_ratio > 2 AND ret_5m < 0 → fwd_5m > 0, IC > 0.01.",
            null_hypothesis="Extreme volume has zero predictive power for next-bar return.",
            expected_ic_range=(0.005, 0.020),
            expected_decay_halflife_bars=6,  # decays fast — 5m micro-structure
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_F2_corrected",
        ),
        HypothesisSpec(
            hypothesis_id="h_session_bias",
            hypothesis_family="session_effects",
            economic_mechanism=(
                "BTC trading volume rotates through Asia→Europe→US sessions. "
                "US market close (22 UTC) shows strongest positive bias (+328%/yr annualized) "
                "due to ETF rebalancing flows. US open (14-16 UTC) shows negative bias. "
                "Session timing is a standalone first-order signal — does NOT need K-line "
                "interaction (the CP×session interaction was FALSIFIED as weaker than session-only)."
            ),
            testable_prediction="session-only IC > 0.005 at daily aggregation.",
            null_hypothesis="UTC hour has zero predictive power for BTC returns.",
            expected_ic_range=(0.003, 0.010),
            expected_decay_halflife_bars=288,
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_F3_corrected",
        ),
        # ── Funding Rate (Futures) ──────────────────────────────────
        HypothesisSpec(
            hypothesis_id="h_funding_rate_reversal_daily",
            hypothesis_family="funding_basis",
            economic_mechanism=(
                "Extreme funding rates reflect crowded positioning. When funding is very high "
                "(top 10% of historical), too many traders are long → mean-reversion lower. "
                "When funding is very low/negative (bottom 10%), panic shorts dominate → bounce. "
                "IC = -0.063 at daily horizon (14× stronger than best OHLCV factor). "
                "Walk-forward OOS IC = -0.048, IC ratio = 0.74 (stable). "
                "CRITICAL ASYMMETRY: short side is broken (SR=-1.54) — crypto is structurally "
                "long-biased (84.5% of time FR > 0). Only LONG on extreme negative FR is tradable."
            ),
            testable_prediction="FR level IC < -0.04 daily, OOS IC ratio > 0.5.",
            null_hypothesis="Funding rate has zero predictive power for daily BTC returns.",
            expected_ic_range=(0.03, 0.08),
            expected_decay_halflife_bars=21,  # ~21 days
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_funding_rate",
        ),
        HypothesisSpec(
            hypothesis_id="h_funding_rate_risk_overlay",
            hypothesis_family="funding_basis",
            economic_mechanism=(
                "Funding rate cannot be a standalone alpha source (strategy SR=-0.55 in "
                "post-training validation), but it reduces MaxDD by 51% vs Buy&Hold (30.2% vs "
                "81.6%). Use as a RISK OVERLAY: reduce position size when FR is extreme in "
                "either direction, rather than taking directional bets."
            ),
            testable_prediction="Adding FR-based position scaling to any strategy reduces MaxDD.",
            null_hypothesis="FR-based position scaling does not reduce drawdowns.",
            expected_ic_range=(0.0, 0.01),  # Not an alpha source, a risk tool
            expected_decay_halflife_bars=90,
            symbols=["BTCUSDT", "ETHUSDT"],
            generated_by="back_lab_funding_risk_overlay",
        ),
    ]


def boundary_conditions(hypothesis_family: str) -> dict:
    """Return the stopping boundary for each hypothesis family.

    When ANY boundary is hit, the family should be paused or stopped.
    These prevent infinite mining on dead-end hypotheses.
    """
    boundaries = {
        "mean_reversion": {
            "max_cumulative_trials": 200,
            "min_oos_ic_ratio": 0.25,        # stop if OOS/IS IC < 0.25
            "min_break_even_cost_multiple": 3.0,  # need 3× safety margin
            "short_allowed": True,
            "note": "Mean-reversion works in trends for BTC, not ranges. If rolling IC flips sign in trending regime, stop."
        },
        "trend_following": {
            "max_cumulative_trials": 200,
            "min_oos_ic_ratio": 0.25,
            "min_break_even_cost_multiple": 2.0,
            "short_allowed": True,
            "note": "Trend acceleration is vol-dependent. If |IC| in high-vol regime drops below 0.01, stop."
        },
        "volume_confirmation": {
            "max_cumulative_trials": 100,
            "min_oos_ic_ratio": 0.20,
            "min_break_even_cost_multiple": 3.0,
            "short_allowed": True,
            "note": "Panic bid is a micro-structure signal (5m). If IC at 15m horizon > IC at 5m, signal is not working as hypothesized."
        },
        "session_effects": {
            "max_cumulative_trials": 50,
            "min_oos_ic_ratio": 0.30,
            "min_break_even_cost_multiple": 5.0,  # high bar — session effects can be spurious
            "short_allowed": True,
            "note": "Session patterns may change with ETF flows. If rolling 90-day IC crosses zero, the regime has shifted."
        },
        "funding_basis": {
            "max_cumulative_trials": 150,          # FR is the strongest signal — allow more trials
            "min_oos_ic_ratio": 0.40,              # FR is stable — demand higher OOS consistency
            "min_break_even_cost_multiple": 2.0,
            "short_allowed": False,                 # SHORT SIDE IS BROKEN — never short on high FR
            "note": "Only long on extreme negative FR. Short side (high FR → short) has SR=-1.54. Do not mine short-biased FR strategies."
        },
    }
    if hypothesis_family in boundaries:
        return boundaries[hypothesis_family]
    # Default conservative boundary
    return {
        "max_cumulative_trials": 100,
        "min_oos_ic_ratio": 0.25,
        "min_break_even_cost_multiple": 2.0,
        "short_allowed": True,
        "note": "Default boundary. Tighten for your specific hypothesis."
    }


def should_continue_mining(hypothesis_family: str, cumulative_trials: int,
                           oos_ic_ratio: float, break_even_multiple: float,
                           uses_short: bool = False) -> tuple[bool, str]:
    """Check if mining should continue for this hypothesis family.

    Returns (continue, reason).
    """
    b = boundary_conditions(hypothesis_family)

    if cumulative_trials >= b["max_cumulative_trials"]:
        return False, f"max trials ({b['max_cumulative_trials']}) reached with {cumulative_trials}"

    if oos_ic_ratio < b["min_oos_ic_ratio"]:
        return False, f"OOS/IS IC ratio {oos_ic_ratio:.2f} < {b['min_oos_ic_ratio']}"

    if break_even_multiple < b["min_break_even_cost_multiple"]:
        return False, f"break-even/cost multiple {break_even_multiple:.1f} < {b['min_break_even_cost_multiple']}"

    if uses_short and not b["short_allowed"]:
        return False, "short side is blocked for this hypothesis family"

    return True, "all boundaries satisfied"
