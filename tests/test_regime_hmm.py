import numpy as np
import pandas as pd

from factor_mining.regime.hmm import MarkovRegimeDetector


class FakeDetector(MarkovRegimeDetector):
    def __init__(self, proba: np.ndarray) -> None:
        super().__init__(n_states=2)
        self._model = object()
        self._transition_matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
        self._state_labels = {0: "bear", 1: "bull"}
        self._proba = proba

    def predict_proba_filtered(self, frame: pd.DataFrame) -> np.ndarray:
        return self._proba


def test_rolling_forward_regime_matches_scalar_forward_probability() -> None:
    proba = np.array(
        [
            [0.0, 0.0],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )
    detector = FakeDetector(proba)
    frame = pd.DataFrame({"close": [1.0, 1.1, 1.2, 1.3]})

    vectorized = detector.rolling_forward_regime(frame, horizon=2).tolist()
    scalar = ["unknown"]
    for row in proba[1:]:
        state = int(np.argmax(detector.forward_probability(row, horizon=2)))
        scalar.append(detector._state_labels[state])

    assert vectorized == scalar


def _regime_frame(n: int = 900) -> pd.DataFrame:
    idx = np.arange(n)
    returns = (
        0.0002 * np.sin(idx / 17.0)
        + 0.0008 * np.where((idx // 120) % 2 == 0, 1.0, -1.0)
        + 0.0004 * np.sin(idx / 5.0)
    )
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = 1_000.0 + 50.0 * np.sin(idx / 13.0)
    return pd.DataFrame(
        {
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "volume": volume,
        }
    )


def test_filtered_regime_prediction_is_unchanged_by_future_data() -> None:
    frame = _regime_frame()
    perturbed = frame.copy()
    future = perturbed.index >= 650
    perturbed.loc[future, "close"] = perturbed.loc[future, "close"] * np.linspace(1.0, 3.0, int(future.sum()))
    perturbed.loc[future, "high"] = perturbed.loc[future, "close"] * 1.002
    perturbed.loc[future, "low"] = perturbed.loc[future, "close"] * 0.998
    perturbed.loc[future, "volume"] = perturbed.loc[future, "volume"] * 5.0

    detector = MarkovRegimeDetector(n_states=5, random_state=7)
    fit_frame = frame.iloc[:300]
    detector.fit(fit_frame)
    detector.label_states(detector.predict(fit_frame), fit_frame)

    base_proba = detector.predict_proba_filtered(frame)
    perturbed_proba = detector.predict_proba_filtered(perturbed)
    np.testing.assert_allclose(base_proba[:550], perturbed_proba[:550], atol=0.0)

    base_regime = detector.rolling_forward_regime(frame, horizon=12)
    perturbed_regime = detector.rolling_forward_regime(perturbed, horizon=12)
    assert base_regime.iloc[500] == perturbed_regime.iloc[500]


def test_five_state_labels_collapse_to_canonical_regimes() -> None:
    state_sequence = np.repeat(np.arange(5), 30)
    per_state_return = {
        0: -0.003,
        1: -0.001,
        2: 0.0,
        3: 0.001,
        4: 0.003,
    }
    returns = np.array([per_state_return[int(state)] for state in state_sequence[1:]])
    close = np.r_[100.0, 100.0 * np.exp(np.cumsum(returns))]
    frame = pd.DataFrame({"close": close})

    detector = MarkovRegimeDetector(n_states=5)
    labels = detector.label_states(state_sequence, frame)

    assert labels[0] == "bear"
    assert labels[4] == "bull"
    assert set(labels.values()) <= {"bear", "bull", "sideways", "high_vol"}
