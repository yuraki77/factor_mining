"""
Market regime detection — HMM-based with forward prediction.

Uses hmmlearn.GaussianHMM for unsupervised regime inference on:
  - log returns (1, 12, 48 bar lags)
  - realized volatility (20, 50 bar)
  - volume ratio (20 bar)
  - close position (high-low range position)

After fitting, states are post-hoc labeled as bull/bear/sideways/high_vol
based on their mean return and volatility characteristics.

Forward prediction: at bar t, computes P(regime_{t+n} | features_{0:t})
via transition matrix^n.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _diag_covars(covars: np.ndarray) -> np.ndarray:
    if covars.ndim == 3:
        diag = np.array([np.diag(cov) for cov in covars], dtype=float)
    elif covars.ndim == 2:
        diag = covars.astype(float, copy=False)
    else:
        raise ValueError(f"Unsupported covariance shape for diagonal HMM: {covars.shape}")
    return np.clip(diag, 1e-12, None)


def _diag_gaussian_logpdf(x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    diff = means - x
    n_features = means.shape[1]
    return -0.5 * (
        n_features * np.log(2.0 * np.pi)
        + np.log(covars).sum(axis=1)
        + ((diff * diff) / covars).sum(axis=1)
    )


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


class MarkovRegimeDetector:
    """HMM-based regime detection with forward prediction.

    Usage:
        detector = MarkovRegimeDetector(n_states=5)
        detector.fit(frame)
        states = detector.predict(frame)           # in-sample smoothed states
        labels = detector.label_states(states)      # bull/bear/sideways
        # Forward prediction: what regime in 48 bars?
        prob = detector.forward_probability(current_state, horizon=48)
    """

    def __init__(self, n_states: int = 5, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self._model = None
        self._state_labels: dict[int, str] = {}
        self._transition_matrix: np.ndarray | None = None
        self._transition_power_cache: dict[int, np.ndarray] = {}
        self._feature_cols: list[str] = []

    # ── feature extraction ──────────────────────────────────────────

    @staticmethod
    def _extract_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract HMM input features from OHLCV frame."""
        close = frame["close"].to_numpy(dtype=float)
        volume = frame["volume"].to_numpy(dtype=float)
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)

        n = len(close)
        features = {}
        cols: list[str] = []

        # Log returns at multiple lags
        for lag in [1, 12, 48]:
            col = f"log_ret_{lag}"
            ret = np.full(n, np.nan)
            ret[lag:] = np.log(close[lag:] / close[:-lag])
            features[col] = ret
            cols.append(col)

        # Realized volatility
        log_returns = pd.Series(np.log(close)).diff()
        for window in [20, 50]:
            col = f"vol_{window}"
            features[col] = log_returns.rolling(window).std(ddof=0).to_numpy(dtype=float)
            cols.append(col)

        # Volume ratio
        col = "vol_ratio_20"
        vol_ma = pd.Series(volume).rolling(20).mean().to_numpy(dtype=float)
        vol_ratio = np.full(n, np.nan)
        valid = vol_ma > 0
        vol_ratio[valid] = volume[valid] / vol_ma[valid]
        features[col] = vol_ratio
        cols.append(col)

        # Close position within bar
        col = "close_position"
        hl_range = high - low
        cp = np.full(n, 0.5)
        valid = hl_range > 0
        cp[valid] = (close[valid] - low[valid]) / hl_range[valid]
        features[col] = cp
        cols.append(col)

        # Assemble array, drop rows with NaN (initial warm-up)
        arr = np.column_stack([features[c] for c in cols])
        return arr, cols

    # ── fit / predict ───────────────────────────────────────────────

    def fit(self, frame: pd.DataFrame, *, tail: int | None = None) -> "MarkovRegimeDetector":
        """Fit GaussianHMM on OHLCV features."""
        from hmmlearn.hmm import GaussianHMM

        if tail is not None:
            frame = frame.iloc[-tail:]

        X, self._feature_cols = self._extract_features(frame)
        # Drop rows with NaN
        valid = ~np.isnan(X).any(axis=1)
        X = X[valid]

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=50,
            random_state=self.random_state,
            tol=1e-2,
        )
        self._model.fit(X)
        self._transition_matrix = self._model.transmat_.copy()
        self._transition_power_cache.clear()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict most likely hidden state sequence (Viterbi)."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _ = self._extract_features(frame)
        valid = ~np.isnan(X).any(axis=1)
        X_clean = X[valid]
        states_clean = self._model.predict(X_clean)
        # Map back to full length (NaNs → -1)
        states = np.full(len(frame), -1, dtype=int)
        states[valid] = states_clean
        return states

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return full posterior state probabilities per bar.

        This uses hmmlearn's posterior smoother and may use future observations.
        Use ``predict_proba_filtered`` for live trading decisions.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _ = self._extract_features(frame)
        valid = ~np.isnan(X).any(axis=1)
        X_clean = X[valid]
        proba_clean = self._model.predict_proba(X_clean)
        proba = np.zeros((len(frame), self.n_states))
        proba[valid] = proba_clean
        return proba

    def predict_proba_filtered(self, frame: pd.DataFrame) -> np.ndarray:
        """Return causal filtered state probabilities per bar.

        The recursion only uses observations up to the current valid row:
        prior = previous_filtered_probability @ transition_matrix, then a
        Bayesian update with the current emission likelihood.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._transition_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _ = self._extract_features(frame)
        valid = np.isfinite(X).all(axis=1)
        valid_idx = np.flatnonzero(valid)
        proba = np.zeros((len(frame), self.n_states))
        if valid_idx.size == 0:
            return proba

        means = np.asarray(self._model.means_, dtype=float)
        covars = _diag_covars(np.asarray(self._model.covars_, dtype=float))
        start = np.asarray(self._model.startprob_, dtype=float)
        start = start / max(float(start.sum()), 1e-12)
        filtered = start

        for idx in valid_idx:
            log_prior = np.log(np.clip(filtered, 1e-12, None))
            log_emit = _diag_gaussian_logpdf(X[idx], means, covars)
            log_unnormalized = log_prior + log_emit
            filtered = np.exp(log_unnormalized - _logsumexp(log_unnormalized))
            proba[idx] = filtered
            filtered = filtered @ self._transition_matrix
            filtered = filtered / max(float(filtered.sum()), 1e-12)

        return proba

    # ── regime labeling ─────────────────────────────────────────────

    def label_states(
        self,
        state_sequence: np.ndarray,
        frame: pd.DataFrame,
        *,
        annualization_bars: int = 105120,
    ) -> dict[int, str]:
        """Post-hoc label HMM states as bull/bear/sideways/high_vol.

        Ranks states by annualized return to assign labels proportionally:
          - Top return state(s) → bull
          - Bottom return state(s) → bear
          - Middle state(s) → sideways (or high_vol if vol is extreme)

        For n_states=3: bull / sideways / bear
        For n_states=4: bull / mild_bull (sideways) / bear / high_vol
        For n_states=5: bear / mild_bear / sideways / mild_bull / bull,
        with unusually volatile middle states collapsed to high_vol.
        """
        close = frame["close"].to_numpy(dtype=float)
        returns = np.diff(np.log(close))
        labels: dict[int, str] = {}
        state_stats: dict[int, tuple[float, float]] = {}

        for state in sorted(set(state_sequence)):
            if state == -1:
                labels[state] = "unknown"
                continue
            mask = state_sequence[1:] == state
            if mask.sum() < 10:
                labels[state] = "unknown"
                continue
            ann_ret = returns[mask].mean() * annualization_bars
            ann_vol = returns[mask].std() * np.sqrt(annualization_bars)
            state_stats[state] = (ann_ret, ann_vol)

        if not state_stats:
            self._state_labels = labels
            return labels

        # Sort states by annualized return
        ranked = sorted(state_stats.items(), key=lambda x: x[1][0])
        n = len(ranked)

        # global mean vol for thresholding
        global_vol = np.nanmean([v[1] for v in state_stats.values()])

        for rank, (state, (ann_ret, ann_vol)) in enumerate(ranked):
            if n == 2:
                labels[state] = "bull" if rank >= n // 2 else "bear"
            elif n == 3:
                if rank == 0:
                    labels[state] = "bear"
                elif rank == n - 1:
                    labels[state] = "bull"
                else:
                    labels[state] = "high_vol" if ann_vol > global_vol * 1.3 else "sideways"
            elif n == 5:
                if rank == 0:
                    labels[state] = "bear"
                elif rank == n - 1:
                    labels[state] = "bull"
                elif ann_vol > global_vol * 1.25:
                    labels[state] = "high_vol"
                elif rank == 1:
                    labels[state] = "bear"
                elif rank == 3:
                    labels[state] = "bull"
                else:
                    labels[state] = "sideways"
            else:  # 4+
                if rank == 0:
                    labels[state] = "bear"
                elif rank == n - 1:
                    labels[state] = "bull"
                elif ann_vol > global_vol * 1.3:
                    labels[state] = "high_vol"
                else:
                    labels[state] = "sideways"

        self._state_labels = labels
        return labels

    def map_to_labels(self, state_sequence: np.ndarray) -> pd.Series:
        """Convert integer state sequence to human-readable regime labels."""
        if not self._state_labels:
            return pd.Series("sideways", index=range(len(state_sequence)))
        return pd.Series(
            [self._state_labels.get(int(s), "sideways") for s in state_sequence],
            index=range(len(state_sequence)),
        )

    # ── forward prediction ──────────────────────────────────────────

    def forward_probability(self, current_proba: np.ndarray, horizon: int) -> np.ndarray:
        """Predict regime probabilities N bars ahead.

        Args:
            current_proba: Probability distribution over states at current bar
                           (shape: (n_states,) or broadcastable).
            horizon: Number of bars to look ahead.

        Returns:
            Predicted probability distribution at t+horizon (shape: (n_states,)).
        """
        tm_n = self._transition_power(horizon)
        return current_proba @ tm_n

    def rolling_forward_regime(
        self,
        frame: pd.DataFrame,
        *,
        horizon: int = 48,
    ) -> pd.Series:
        """Compute forward-predicted regime at each bar.

        For each bar t, predicts most likely regime at t+horizon using
        the transition matrix^horizon from the current inferred state.

        Returns a Series of regime labels indexed by bar position.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not self._state_labels:
            raise RuntimeError("States not labeled. Call label_states() first.")

        proba = self.predict_proba_filtered(frame)
        regimes = np.full(len(frame), "unknown", dtype=object)
        valid = proba.sum(axis=1) > 0
        if not valid.any():
            return pd.Series(regimes, index=frame.index)

        fwd_proba = proba[valid] @ self._transition_power(horizon)
        fwd_states = np.argmax(fwd_proba, axis=1)
        regimes[valid] = np.array(
            [self._state_labels.get(int(state), "sideways") for state in fwd_states],
            dtype=object,
        )

        return pd.Series(regimes, index=frame.index)

    def _transition_power(self, horizon: int) -> np.ndarray:
        if self._transition_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if horizon not in self._transition_power_cache:
            self._transition_power_cache[horizon] = np.linalg.matrix_power(self._transition_matrix, horizon)
        return self._transition_power_cache[horizon]

    # ── diagnostics ─────────────────────────────────────────────────

    @property
    def transition_matrix(self) -> np.ndarray | None:
        return self._transition_matrix

    @property
    def stationary_distribution(self) -> np.ndarray | None:
        """Stationary distribution of the HMM."""
        if self._transition_matrix is None:
            return None
        # Find eigenvector for eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(self._transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        return stat / stat.sum()

    def regime_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics per regime."""
        states = self.predict(frame)
        labels = self.map_to_labels(states)
        close = frame["close"].to_numpy(dtype=float)
        returns = pd.Series(np.diff(np.log(close)), index=frame.index[1:])

        rows = []
        for r in sorted(set(labels)):
            mask = (labels.iloc[1:].values == r) if r != "unknown" else np.zeros(len(returns), dtype=bool)
            if mask.sum() < 10:
                continue
            ann_factor = 105120
            rets = returns[mask].dropna()
            rows.append({
                "regime": r,
                "pct_time": round(mask.mean() * 100, 1),
                "ann_return": round(rets.mean() * ann_factor, 3),
                "ann_vol": round(rets.std() * np.sqrt(ann_factor), 3),
                "sharpe": round(rets.mean() / rets.std() * np.sqrt(ann_factor), 3) if rets.std() > 0 else 0,
                "n_bars": int(mask.sum()),
            })
        return pd.DataFrame(rows)
