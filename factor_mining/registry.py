from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MethodSpec(BaseModel):
    method_id: str
    family: str
    display_name: str
    status: Literal["implemented", "planned", "blocked_v1"]
    min_universe_size: int = 1
    requires_cross_section: bool = False
    requires_orderbook: bool = False
    v1_schedulable: bool = False
    is_ml: bool = False
    blocked_reason: str | None = None


_V1 = {
    "template_constrained_search",
    "rule_mining",
    "parameter_sweep",
    "grid_search",
    "random_search",
    "condition_combination_search",
    "signal_filter_exit_search",
    "preset_expansion",
    "strategy_template_mutation",
    "factor_scoring",
    "ic_analysis",
    "rank_ic_analysis",
    "factor_decay_analysis",
    "factor_stability_analysis",
    "time_series_supervised_learning",
    "time_series_classification",
    "time_series_regression",
    "funding_rate_event_mining",
    "gate_check_pipeline",
}


def _method(method_id: str, family: str, display_name: str, *, min_universe_size: int = 1,
            cross: bool = False, orderbook: bool = False, ml: bool = False) -> MethodSpec:
    blocked = cross or orderbook or min_universe_size > 2
    status: Literal["implemented", "planned", "blocked_v1"]
    if method_id in _V1 and not blocked:
        status = "implemented"
    elif blocked:
        status = "blocked_v1"
    else:
        status = "planned"
    reason = None
    if blocked:
        reason = "Requires cross-sectional/orderbook/universe size beyond BTC/ETH v1 scope."
    return MethodSpec(
        method_id=method_id,
        family=family,
        display_name=display_name,
        status=status,
        min_universe_size=min_universe_size,
        requires_cross_section=cross,
        requires_orderbook=orderbook,
        v1_schedulable=status == "implemented",
        is_ml=ml,
        blocked_reason=reason,
    )


METHOD_REGISTRY: list[MethodSpec] = [
    _method("template_constrained_search", "template", "Template-constrained Search"),
    _method("rule_mining", "template", "Rule Mining"),
    _method("parameter_sweep", "optimization", "Parameter Sweep"),
    _method("grid_search", "optimization", "Grid Search"),
    _method("random_search", "optimization", "Random Search"),
    _method("condition_combination_search", "template", "Condition Combination Search"),
    _method("signal_filter_exit_search", "template", "Signal + Filter + Exit Search"),
    _method("preset_expansion", "template", "Preset Expansion"),
    _method("strategy_template_mutation", "template", "Strategy Template Mutation"),
    _method("factor_scoring", "statistics", "Factor Scoring"),
    _method("ic_analysis", "statistics", "IC Analysis"),
    _method("rank_ic_analysis", "statistics", "Rank IC Analysis"),
    _method("factor_decay_analysis", "statistics", "Factor Decay Analysis"),
    _method("factor_stability_analysis", "statistics", "Factor Stability Analysis"),
    _method("cross_symbol_validation", "statistics", "Cross-symbol Validation", min_universe_size=3, cross=True),
    _method("cross_timeframe_validation", "statistics", "Cross-timeframe Validation"),
    _method("out_of_sample_test", "statistics", "Out-of-sample Test"),
    _method("walk_forward_analysis", "statistics", "Walk-forward Analysis"),
    _method("sensitivity_analysis", "statistics", "Sensitivity Analysis"),
    _method("factor_correlation_analysis", "statistics", "Factor Correlation Analysis"),
    _method("factor_redundancy_detection", "statistics", "Factor Redundancy Detection"),
    _method("regime_specific_factor_evaluation", "statistics", "Regime-specific Factor Evaluation"),
    _method("time_series_supervised_learning", "ml", "Time-series Supervised Learning", ml=True),
    _method("time_series_classification", "ml", "Time-series Classification Model", ml=True),
    _method("time_series_regression", "ml", "Time-series Regression Model", ml=True),
    _method("xgboost_lightgbm_factor_model", "ml", "XGBoost / LightGBM Factor Model", ml=True),
    _method("random_forest_factor_model", "ml", "Random Forest Factor Model", ml=True),
    _method("logistic_regression_signal_model", "ml", "Logistic Regression Signal Model", ml=True),
    _method("meta_labeling", "ml", "Meta-labeling", ml=True),
    _method("signal_quality_scoring", "ml", "Signal Quality Scoring", ml=True),
    _method("market_regime_clustering", "ml", "Market Regime Clustering", ml=True),
    _method("change_point_detection", "ml", "Change Point Detection", ml=True),
    _method("anomaly_detection", "ml", "Anomaly Detection", ml=True),
    _method("feature_importance_analysis", "ml", "Feature Importance Analysis", ml=True),
    _method("feature_selection", "ml", "Feature Selection", ml=True),
    _method("bayesian_optimization", "optimization", "Bayesian Optimization"),
    _method("genetic_algorithm", "optimization", "Genetic Algorithm"),
    _method("genetic_programming", "optimization", "Genetic Programming"),
    _method("symbolic_regression", "optimization", "Symbolic Regression"),
    _method("evolutionary_strategy_search", "optimization", "Evolutionary Strategy Search"),
    _method("multi_objective_optimization", "optimization", "Multi-objective Optimization"),
    _method("pareto_frontier_search", "optimization", "Pareto Frontier Search"),
    _method("hyperparameter_optimization", "optimization", "Hyperparameter Optimization"),
    _method("complexity_constrained_search", "optimization", "Complexity-constrained Search"),
    _method("top_k_candidate_selection", "optimization", "Top-K Candidate Selection"),
    _method("event_study", "event", "Event Study"),
    _method("breakout_event_mining", "event", "Breakout Event Mining"),
    _method("pullback_event_mining", "event", "Pullback Event Mining"),
    _method("volume_spike_event_mining", "event", "Volume Spike Event Mining"),
    _method("large_trader_flow_mining", "event", "Large Trader Flow Mining", orderbook=True),
    _method("exchange_netflow_mining", "event", "Exchange Netflow Mining"),
    _method("funding_rate_event_mining", "funding_basis", "Funding Rate Event Mining"),
    _method("liquidation_event_mining", "event", "Liquidation Event Mining"),
    _method("social_sentiment_event_mining", "event", "Social Sentiment Event Mining"),
    _method("news_macro_event_study", "event", "News / Macro Event Study"),
    _method("order_flow_mining", "microstructure", "Order Flow Mining", orderbook=True),
    _method("order_book_imbalance_mining", "microstructure", "Order Book Imbalance Mining", orderbook=True),
    _method("trade_imbalance_mining", "microstructure", "Trade Imbalance Mining"),
    _method("liquidity_shock_detection", "microstructure", "Liquidity Shock Detection"),
    _method("spread_regime_detection", "microstructure", "Spread Regime Detection", orderbook=True),
    _method("large_order_impact_analysis", "microstructure", "Large Order Impact Analysis", orderbook=True),
    _method("false_breakout_detection", "microstructure", "False Breakout Detection"),
    _method("stop_hunting_pattern_detection", "microstructure", "Stop-hunting Pattern Detection"),
    _method("strategy_ensemble", "portfolio", "Strategy Ensemble", min_universe_size=3),
    _method("strategy_correlation_mining", "portfolio", "Strategy Correlation Mining", min_universe_size=3),
    _method("signal_overlap_analysis", "portfolio", "Signal Overlap Analysis"),
    _method("drawdown_overlap_analysis", "portfolio", "Drawdown Overlap Analysis"),
    _method("strategy_portfolio_optimization", "portfolio", "Strategy Portfolio Optimization", min_universe_size=3),
    _method("risk_contribution_analysis", "portfolio", "Risk Contribution Analysis", min_universe_size=3),
    _method("market_beta_exposure_analysis", "portfolio", "Market Beta Exposure Analysis"),
    _method("diversification_scoring", "portfolio", "Diversification Scoring", min_universe_size=3),
    _method("gate_check_pipeline", "validation", "Gate Check Pipeline"),
    _method("minimum_trade_count_check", "validation", "Minimum Trade Count Check"),
    _method("drawdown_concentration_check", "validation", "Drawdown Concentration Check"),
    _method("return_concentration_check", "validation", "Return Concentration Check"),
    _method("parameter_stability_check", "validation", "Parameter Stability Check"),
    _method("cross_market_robustness_check", "validation", "Cross-market Robustness Check"),
    _method("cross_period_robustness_check", "validation", "Cross-period Robustness Check"),
    _method("goodhart_detection", "validation", "Goodhart Detection"),
    _method("overfitting_detection", "validation", "Overfitting Detection"),
    _method("backtest_leakage_detection", "validation", "Backtest Leakage Detection"),
    _method("survivorship_bias_check", "validation", "Survivorship Bias Check", min_universe_size=3, cross=True),
    _method("transaction_cost_stress_test", "validation", "Transaction Cost Stress Test"),
    _method("slippage_stress_test", "validation", "Slippage Stress Test"),
]


def schedulable_methods(universe_size: int = 2) -> list[MethodSpec]:
    return [
        method for method in METHOD_REGISTRY
        if method.v1_schedulable and method.min_universe_size <= universe_size and not method.requires_cross_section
    ]


def get_method(method_id: str) -> MethodSpec:
    for method in METHOD_REGISTRY:
        if method.method_id == method_id:
            return method
    raise KeyError(f"Unknown method_id: {method_id}")
