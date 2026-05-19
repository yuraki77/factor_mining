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

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
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
