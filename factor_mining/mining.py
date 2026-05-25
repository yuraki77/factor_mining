from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from factor_mining.config import Settings
from factor_mining.llm.providers import hypothesis_system_prompt, provider_from_settings
from factor_mining.models import CandidateStrategySpec, HypothesisSpec
from factor_mining.registry import schedulable_methods


# ── family normalization ────────────────────────────────────────────

# Canonical family names that factor_signal() understands.
# Moved here (from pipeline.py) because hypothesis→signal mapping is a
# mining-layer concern.
FAMILY_ALIASES: dict[str, str] = {
    "momentum": "momentum",
    "trend_following": "momentum",
    "time-series momentum": "momentum",
    "time series momentum": "momentum",
    "trend": "momentum",
    "mean_reversion": "mean_reversion",
    "mean reversion": "mean_reversion",
    "mean-reversion": "mean_reversion",
    "reversion": "mean_reversion",
    "volatility": "volatility",
    "volatility_regime": "volatility",
    "volatility regime": "volatility",
    "volatility feedback": "volatility",
    "vol": "volatility",
    "funding_basis": "funding_basis",
    "funding basis": "funding_basis",
    "funding rate mean reversion": "funding_basis",
    "funding rate reversal": "funding_basis",
    "funding rate": "funding_basis",
    "basis arbitrage": "funding_basis",
    "basis carry": "funding_basis",
    "volume_confirmation": "volume_confirmation",
    "volume confirmation": "volume_confirmation",
    "volume price divergence": "volume_confirmation",
    "volume-price divergence": "volume_confirmation",
    "price volume divergence": "volume_confirmation",
    "price-volume divergence": "volume_confirmation",
    "volume": "volume_confirmation",
}


def normalize_family(hypothesis_family: str) -> str | None:
    """Map any hypothesis family name to a canonical signal-construction family."""
    key = hypothesis_family.lower().strip()
    variants = (
        key,
        key.replace("_", " "),
        key.replace("-", " "),
        key.replace("_", " ").replace("-", " "),
    )
    for variant in variants:
        if variant in FAMILY_ALIASES:
            return FAMILY_ALIASES[variant]
    return None


# ── indicator candidate expansion ───────────────────────────────────

# Canonical hypothesis family → which INDICATOR_META families to search
_CANONICAL_TO_FEATURE_FAMILIES: dict[str, list[str]] = {
    "momentum": ["trend_following"],
    "mean_reversion": ["mean_reversion"],
    "volatility": ["volatility_regime"],
    "funding_basis": ["funding_basis"],
    "volume_confirmation": ["volume_confirmation"],
}

# Lookback grid for factor_signal()-based candidates, per canonical family
_FACTOR_SIGNAL_LOOKBACKS: dict[str, list[int]] = {
    "momentum": [6, 12, 24, 48, 96],
    "mean_reversion": [6, 12, 24, 48],
    "volatility": [12, 24, 48],
    "funding_basis": [12, 48, 96],
}

_SEARCH_VARIANTS: list[dict[str, float | int | str]] = [
    {
        "search_variant": "base",
        "direction_multiplier": 1,
        "signal_threshold": 0.0,
        "smooth_span": 1,
        "position_buffer": 0.05,
    },
    {
        "search_variant": "inverse",
        "direction_multiplier": -1,
        "signal_threshold": 0.0,
        "smooth_span": 1,
        "position_buffer": 0.05,
    },
    {
        "search_variant": "low_turnover",
        "direction_multiplier": 1,
        "signal_threshold": 0.25,
        "smooth_span": 24,
        "position_buffer": 0.15,
    },
]


LAB_MINEABLE_FACTOR_REGISTRY: frozenset[str] = frozenset({
    "rsi14",
    "rsi7",
    "rsi21",
    "rsi14_short",
    "ema50_above",
    "ma_above_200",
    "bbpct",
    "bbands_lower_touch",
    "bbands_percent_b_low",
    "macd_golden",
    "adx_strong",
    "cci_oversold",
    "williams_r_oversold",
    "roc_positive",
    "obv_uptrend",
    "mfi_oversold",
    "cmf_positive",
})

_LAB_FACTOR_FEATURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "rsi14": ("rsi_14",),
    "rsi7": ("rsi_7",),
    "rsi21": ("rsi_21",),
    "rsi14_short": ("rsi_14",),
    "ema50_above": ("ema_50",),
    "ma_above_200": ("sma_200", "price_sma_ratio_200"),
    "bbpct": ("bb_pct_*",),
    "bbands_lower_touch": ("bb_pct_*",),
    "bbands_percent_b_low": ("bb_pct_*",),
    "macd_golden": ("macd_line_*", "macd_hist_*"),
    "adx_strong": ("adx_14",),
    "cci_oversold": ("cci_14",),
    "williams_r_oversold": ("willr_14",),
    "roc_positive": ("roc_*",),
    "obv_uptrend": ("obv",),
    "mfi_oversold": ("mfi_14",),
    "cmf_positive": ("cmf_20",),
}


def _direction_sign(meta_direction: str) -> int | None:
    """Convert INDICATOR_META direction string to a signal sign.

    Returns ``None`` for *neutral* indicators (e.g. volatility gauges)
    that lack inherent long/short semantics and should not be turned into
    simple directional signals.
    """
    if meta_direction == "positive":
        return 1
    if meta_direction in ("negative", "negative_when_high"):
        return -1
    return None


def lab_factor_ids_for_candidate_params(
    params: dict,
    requested_factor_ids: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return Lab factor IDs that this candidate actually tests."""
    if params.get("signal_source") != "feature":
        return []
    indicator_name = str(params.get("indicator_name") or "")
    if not indicator_name:
        return []
    requested = [
        factor_id
        for factor_id in (requested_factor_ids or sorted(LAB_MINEABLE_FACTOR_REGISTRY))
        if factor_id in LAB_MINEABLE_FACTOR_REGISTRY
    ]
    matched: list[str] = []
    for factor_id in requested:
        patterns = _LAB_FACTOR_FEATURE_PATTERNS.get(factor_id, ())
        if any(_feature_pattern_matches(indicator_name, pattern) for pattern in patterns):
            matched.append(factor_id)
    return matched


def filter_candidates_for_lab_factors(
    candidates: list[CandidateStrategySpec],
    factor_ids: list[str] | tuple[str, ...],
) -> list[CandidateStrategySpec]:
    """Restrict candidates to the concrete factor IDs in a Lab direction."""
    requested = [str(item) for item in factor_ids if str(item) in LAB_MINEABLE_FACTOR_REGISTRY]
    if not factor_ids:
        return candidates
    if not requested:
        return []

    scoped: list[CandidateStrategySpec] = []
    for candidate in candidates:
        matched = lab_factor_ids_for_candidate_params(candidate.params, requested)
        if not matched:
            continue
        scoped.append(candidate.model_copy(update={"params": {**candidate.params, "lab_factor_ids": matched}}))
    return scoped


def _feature_pattern_matches(indicator_name: str, pattern: str) -> bool:
    if pattern.endswith("*"):
        return indicator_name.startswith(pattern[:-1])
    return indicator_name == pattern


def build_indicator_candidates(
    hypotheses: list[HypothesisSpec],
    *,
    symbols: list[str],
    feature_meta: dict[str, dict],
    interval: str = "5m",
    max_indicators_per_family: int = 30,
) -> list[CandidateStrategySpec]:
    """Expand hypotheses × indicators into explicitly-parameterized candidates.

    Each candidate's ``params`` carries the full signal specification:
    ``signal_source``, ``indicator_name`` (or ``factor_family`` +
    ``factor_lookback``), ``direction``, and ``transform``.

    Only indicators whose INDICATOR_META family aligns with the hypothesis's
    canonical family are included.  Neutral-direction indicators are skipped
    (they need regime/threshold logic, not a raw directional signal).
    """
    candidates: list[CandidateStrategySpec] = []

    for hypothesis in hypotheses:
        canonical = normalize_family(hypothesis.hypothesis_family)
        if canonical is None:
            continue
        expected_ic_mid = sum(hypothesis.expected_ic_range) / 2.0
        feature_families = _CANONICAL_TO_FEATURE_FAMILIES.get(canonical, [])

        # Collect directional features matching this hypothesis family
        directional_features: list[tuple[str, dict, int]] = []
        for col, meta in feature_meta.items():
            if meta.get("family") not in feature_families:
                continue
            sign = _direction_sign(meta.get("direction", "neutral"))
            if sign is None:
                continue
            directional_features.append((col, meta, sign))

        directional_features = directional_features[:max_indicators_per_family]

        for symbol in symbols:
            # ── Feature-based candidates ────────────────────────────
            # Single method_id: the indicator IS the signal, method adds
            # no information and would only inflate trial count.
            for col, meta, direction in directional_features:
                for variant in _SEARCH_VARIANTS:
                    direction_multiplier = int(variant["direction_multiplier"])
                    candidates.append(CandidateStrategySpec(
                        candidate_id=f"c_{uuid.uuid4().hex[:12]}",
                        hypothesis_id=hypothesis.hypothesis_id,
                        method_id="factor_scoring",
                        hypothesis_family=hypothesis.hypothesis_family,
                        symbol=symbol,
                        interval=interval,
                        params={
                            "signal_source": "feature",
                            "indicator_name": col,
                            "indicator_family": meta.get("family", "hybrid"),
                            "direction": direction * direction_multiplier,
                            "transform": "tanh_zscore",
                            "zscore_window": 288,
                            "tanh_scale": 2.0,
                            "expected_ic_mid": expected_ic_mid,
                            "search_variant": variant["search_variant"],
                            "signal_threshold": variant["signal_threshold"],
                            "smooth_span": variant["smooth_span"],
                            "position_buffer": variant["position_buffer"],
                        },
                    ))

            # ── factor_signal candidates with explicit lookbacks ────
            lookbacks = _FACTOR_SIGNAL_LOOKBACKS.get(canonical, [12])
            for lookback in lookbacks:
                for variant in _SEARCH_VARIANTS:
                    candidates.append(CandidateStrategySpec(
                        candidate_id=f"c_{uuid.uuid4().hex[:12]}",
                        hypothesis_id=hypothesis.hypothesis_id,
                        method_id="factor_scoring",
                        hypothesis_family=hypothesis.hypothesis_family,
                        symbol=symbol,
                        interval=interval,
                        params={
                            "signal_source": "factor_signal",
                            "factor_family": canonical,
                            "factor_lookback": lookback,
                            "direction": int(variant["direction_multiplier"]),
                            "transform": "raw_clip",
                            "expected_ic_mid": expected_ic_mid,
                            "search_variant": variant["search_variant"],
                            "signal_threshold": variant["signal_threshold"],
                            "smooth_span": variant["smooth_span"],
                            "position_buffer": variant["position_buffer"],
                        },
                    ))

    return candidates


def default_hypotheses() -> list[HypothesisSpec]:
    """Return built-in hypotheses + discovered hypotheses from back_lab research."""
    from factor_mining.hypotheses.discovered import discovered_hypotheses

    base = [
        HypothesisSpec(
            hypothesis_id="h_momentum_5m",
            hypothesis_family="momentum",
            economic_mechanism="Short-horizon continuation after strong directional pressure can persist until liquidity replenishes.",
            testable_prediction="Positive 12-bar return predicts positive next-bar return after costs.",
            null_hypothesis="Lagged return has zero IC with next-bar return after costs.",
            expected_ic_range=(0.005, 0.03),
            expected_decay_halflife_bars=24,
        ),
        HypothesisSpec(
            hypothesis_id="h_momentum_volume_confirmed",
            hypothesis_family="momentum",
            economic_mechanism="Directional moves backed by above-normal volume are more likely to represent informed flow than transient noise.",
            testable_prediction="Trend-following indicators conditioned on volume confirmation have positive next-bar IC.",
            null_hypothesis="Volume-confirmed trend signals have zero IC with next-bar return.",
            expected_ic_range=(0.004, 0.025),
            expected_decay_halflife_bars=18,
        ),
        HypothesisSpec(
            hypothesis_id="h_momentum_ema_drift",
            hypothesis_family="momentum",
            economic_mechanism="Persistent EMA/price drift captures slow inventory transfer before liquidity providers fully reprice.",
            testable_prediction="EMA and price/MA trend indicators predict same-direction open-to-open return.",
            null_hypothesis="EMA drift has zero predictive IC after costs.",
            expected_ic_range=(0.003, 0.020),
            expected_decay_halflife_bars=48,
        ),
        HypothesisSpec(
            hypothesis_id="h_momentum_efficiency_breakout",
            hypothesis_family="momentum",
            economic_mechanism="High efficiency ratio regimes indicate directional price discovery with fewer mean-reverting micro moves.",
            testable_prediction="Efficient directional moves have positive continuation IC over the next few bars.",
            null_hypothesis="Directional efficiency has zero predictive IC.",
            expected_ic_range=(0.003, 0.018),
            expected_decay_halflife_bars=36,
        ),
        HypothesisSpec(
            hypothesis_id="h_mean_reversion_rsi_extreme",
            hypothesis_family="mean_reversion",
            economic_mechanism="Short-horizon oscillator extremes often reflect temporary taker imbalance that fades once passive liquidity replenishes.",
            testable_prediction="High RSI/stochastic readings predict lower next-bar returns; low readings predict higher returns.",
            null_hypothesis="Oscillator extremes have zero next-bar IC.",
            expected_ic_range=(0.004, 0.025),
            expected_decay_halflife_bars=12,
        ),
        HypothesisSpec(
            hypothesis_id="h_mean_reversion_close_position",
            hypothesis_family="mean_reversion",
            economic_mechanism="Closes near candle extremes can mark short-lived exhaustion when market orders overrun resting liquidity.",
            testable_prediction="Extreme close position predicts opposite-direction next-bar return.",
            null_hypothesis="Close position has zero IC with next-bar return.",
            expected_ic_range=(0.003, 0.020),
            expected_decay_halflife_bars=8,
        ),
        HypothesisSpec(
            hypothesis_id="h_mean_reversion_bollinger_pct",
            hypothesis_family="mean_reversion",
            economic_mechanism="Price excursions to local distribution tails are often liquidity overshoots in 5m crypto markets.",
            testable_prediction="High Bollinger percent predicts negative next-bar return and low percent predicts positive return.",
            null_hypothesis="Bollinger location has zero next-bar IC.",
            expected_ic_range=(0.003, 0.018),
            expected_decay_halflife_bars=18,
        ),
        HypothesisSpec(
            hypothesis_id="h_mean_reversion_gap_fill",
            hypothesis_family="mean_reversion",
            economic_mechanism="Abrupt open-to-prior-close gaps tend to partially fill as arbitrage and passive liquidity catch up.",
            testable_prediction="Positive gaps predict lower next-bar returns; negative gaps predict higher next-bar returns.",
            null_hypothesis="Gap size has zero IC with next-bar return.",
            expected_ic_range=(0.002, 0.015),
            expected_decay_halflife_bars=6,
        ),
        HypothesisSpec(
            hypothesis_id="h_volatility_risk_premium",
            hypothesis_family="volatility",
            economic_mechanism="Elevated realized volatility raises required compensation for providing liquidity and holding inventory.",
            testable_prediction="Volatility-regime indicators help separate tradable continuation from noisy churn.",
            null_hypothesis="Volatility regime has zero predictive value for next-bar return.",
            expected_ic_range=(0.002, 0.015),
            expected_decay_halflife_bars=48,
        ),
        HypothesisSpec(
            hypothesis_id="h_volatility_squeeze_release",
            hypothesis_family="volatility",
            economic_mechanism="Compressed intraday ranges store directional imbalance that releases when volatility normalizes.",
            testable_prediction="Low-to-rising volatility regimes improve subsequent directional signal quality.",
            null_hypothesis="Volatility compression has zero predictive value.",
            expected_ic_range=(0.002, 0.014),
            expected_decay_halflife_bars=72,
        ),
        HypothesisSpec(
            hypothesis_id="h_volume_panic_bid",
            hypothesis_family="volume_confirmation",
            economic_mechanism="Large sell-pressure bursts can exhaust forced sellers and invite immediate market-maker inventory mean reversion.",
            testable_prediction="Volume spikes combined with weak price action predict short-lived rebound.",
            null_hypothesis="Volume spikes have zero next-bar IC.",
            expected_ic_range=(0.004, 0.022),
            expected_decay_halflife_bars=6,
        ),
        HypothesisSpec(
            hypothesis_id="h_volume_obv_confirmation",
            hypothesis_family="volume_confirmation",
            economic_mechanism="OBV and money-flow confirmation distinguish real accumulation/distribution from price-only noise.",
            testable_prediction="Volume-flow confirmation improves next-bar directional IC.",
            null_hypothesis="Volume-flow indicators have zero next-bar IC.",
            expected_ic_range=(0.003, 0.018),
            expected_decay_halflife_bars=24,
        ),
        HypothesisSpec(
            hypothesis_id="h_funding_basis",
            hypothesis_family="funding_basis",
            economic_mechanism="Extreme funding can reflect crowded positioning that mean-reverts after funding settlement.",
            testable_prediction="High positive funding predicts weaker forward perp returns versus spot.",
            null_hypothesis="Funding rate has zero predictive power for forward perp returns.",
            expected_ic_range=(0.005, 0.025),
            expected_decay_halflife_bars=96,
        ),
        HypothesisSpec(
            hypothesis_id="h_funding_negative_squeeze",
            hypothesis_family="funding_basis",
            economic_mechanism="Extreme negative funding reflects crowded shorts; in structurally long-biased crypto, this creates squeeze risk.",
            testable_prediction="Low/negative funding predicts positive forward returns more reliably than high funding predicts shorts.",
            null_hypothesis="Negative funding extremes have zero predictive IC.",
            expected_ic_range=(0.008, 0.035),
            expected_decay_halflife_bars=96,
        ),
    ]
    # Merge with back_lab discovered hypotheses
    discovered = discovered_hypotheses()
    seen_ids = {h.hypothesis_id for h in base}
    for h in discovered:
        if h.hypothesis_id not in seen_ids:
            base.append(h)
    return base


def generate_hypotheses_with_deepseek(settings: Settings, *, count: int = 5, research_brief: str | None = None) -> list[HypothesisSpec]:
    provider = provider_from_settings("deepseek", settings)
    if not provider.is_configured:
        raise RuntimeError(f"DeepSeek API key is missing. Set {provider.api_key_env} in .env or your shell.")
    prompt = research_brief or (
        "Generate rigorous first-principles BTC/ETH factor hypotheses for Binance spot and USD-M perpetual 5m data. "
        "Only produce time-series or funding/basis hypotheses suitable for N=2 symbols."
    )
    response = provider.chat_json(
        model=settings.llm.deepseek.hypothesis_model,
        messages=[
            {"role": "system", "content": hypothesis_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nReturn JSON with a top-level 'hypotheses' array of exactly {count} objects. "
                    "Each object must include hypothesis_id, hypothesis_family, economic_mechanism, "
                    "testable_prediction, null_hypothesis, expected_ic_range, expected_decay_halflife_bars."
                ),
            },
        ],
    )
    payload = _extract_chat_json(response)
    items = payload.get("hypotheses", payload if isinstance(payload, list) else [])
    hypotheses: list[HypothesisSpec] = []
    for idx, item in enumerate(items[:count]):
        item = dict(item)
        item.setdefault("hypothesis_id", f"h_llm_{idx}_{uuid.uuid4().hex[:8]}")
        item.setdefault("generated_by", "deepseek")
        hypotheses.append(HypothesisSpec.model_validate(item))
    if not hypotheses:
        raise RuntimeError("DeepSeek returned no valid hypotheses")
    return hypotheses


def _extract_chat_json(response: dict) -> dict | list:
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    if content is None:
        raise RuntimeError("LLM response did not include choices[0].message.content")
    if isinstance(content, dict | list):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM response content was not valid JSON: {content[:200]}") from exc


def build_v1_candidates(hypotheses: list[HypothesisSpec], *, symbols: list[str], interval: str = "5m") -> list[CandidateStrategySpec]:
    candidates: list[CandidateStrategySpec] = []
    methods = [m for m in schedulable_methods(universe_size=2) if m.family in {"template", "statistics", "optimization", "funding_basis", "ml"}]
    for hypothesis in hypotheses:
        for symbol in symbols:
            for method in methods:
                if method.is_ml and hypothesis.hypothesis_family not in {"momentum", "mean_reversion", "volatility"}:
                    continue
                expected_mid = sum(hypothesis.expected_ic_range) / 2.0
                candidates.append(
                    CandidateStrategySpec(
                        candidate_id=f"c_{uuid.uuid4().hex[:12]}",
                        hypothesis_id=hypothesis.hypothesis_id,
                        method_id=method.method_id,
                        hypothesis_family=hypothesis.hypothesis_family,
                        symbol=symbol,
                        interval=interval,
                        params={"expected_ic_mid": expected_mid},
                        is_ml=method.is_ml,
                    )
                )
    return candidates


def factor_signal(
    frame: pd.DataFrame, *, family: str, lookback: int = 12, funding_rate: pd.Series | None = None,
) -> pd.Series:
    """Construct a continuous trading signal for *family*.

    All branches return tanh-scaled values in (-1, 1) so that downstream
    ``clip(-1, 1)`` in the backtest engine is a near-no-op while preserving
    the full ranking/intensity information that IC and RankIC need.
    """
    close = pd.Series(frame["close"].to_numpy(dtype=float))
    volume = pd.Series(frame["volume"].to_numpy(dtype=float))
    if family == "momentum":
        ret = close.pct_change(lookback)
        rolling_std = ret.rolling(lookback * 4, min_periods=lookback).std()
        z = (ret / rolling_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return np.tanh(z / 2.0)
    if family == "mean_reversion":
        z = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std().replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return -np.tanh(z / 2.0)
    if family == "volatility":
        vol = close.pct_change().rolling(lookback).std()
        vol_median = vol.rolling(lookback * 4, min_periods=lookback).median()
        vol_iqr = vol.rolling(lookback * 4, min_periods=lookback).apply(
            lambda x: x.quantile(0.75) - x.quantile(0.25) if len(x) > 1 else 1.0,
            raw=False,
        ).replace(0, np.nan)
        z = ((vol - vol_median) / vol_iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return np.tanh(z / 2.0)
    if family == "funding_basis":
        if funding_rate is None:
            raise ValueError("funding_basis factor_signal requires non-zero funding_rate data")
        # funding_rate is an 8h-event z-score aligned to this 5m frame.
        fr = pd.Series(funding_rate.to_numpy(dtype=float), index=frame.index)
        fr = fr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if fr.abs().sum() <= 1e-12:
            raise ValueError("funding_basis factor_signal requires non-zero funding_rate data")
        z = fr.clip(-6, 6)
        # Negative: high funding -> short; Positive: low funding -> long.
        return -np.tanh(z / 2.0)
    # Default: VWAP deviation
    vwap_proxy = (close * volume).rolling(lookback).sum() / volume.rolling(lookback).sum()
    deviation = (close / vwap_proxy.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling_std = deviation.rolling(lookback * 4, min_periods=lookback).std().replace(0, np.nan)
    z = (deviation / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return -np.tanh(z / 2.0)
