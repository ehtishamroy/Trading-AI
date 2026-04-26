"""
Model Calibrator — Platt scaling so predicted probabilities match actual win rates.
Without calibration, model.predict_proba() is just a score, not a real probability.
After calibration, 60% predicted = ~60% actual win rate.
"""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import MODELS_DIR


class ProbabilityCalibrator:
    """
    Platt scaling calibrator.
    Fits a sigmoid function to map raw model scores to calibrated probabilities.
    """

    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, raw_probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Fit the calibrator on out-of-sample predictions.

        Args:
            raw_probabilities: Model's predicted P(class=1), shape (n_samples,)
            true_labels: Actual 0/1 labels, shape (n_samples,)
        """
        X = raw_probabilities.reshape(-1, 1)
        self.calibrator.fit(X, true_labels)
        self.is_fitted = True
        logger.info("Probability calibrator fitted")

    def calibrate(self, raw_probability: float) -> float:
        """Map a raw model probability to a calibrated probability."""
        if not self.is_fitted:
            return raw_probability
        X = np.array([[raw_probability]])
        return float(self.calibrator.predict_proba(X)[0, 1])

    def calibrate_batch(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Calibrate a batch of probabilities."""
        if not self.is_fitted:
            return raw_probabilities
        X = raw_probabilities.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]

    def save(self, market: str):
        path = MODELS_DIR / f"calibrator_{market}.pkl"
        joblib.dump(self, path)
        logger.info(f"Calibrator saved -> {path}")

    @staticmethod
    def load(market: str) -> "ProbabilityCalibrator":
        path = MODELS_DIR / f"calibrator_{market}.pkl"
        if not path.exists():
            logger.warning(f"No calibrator for {market} — using uncalibrated probabilities")
            return ProbabilityCalibrator()
        cal = joblib.load(path)
        logger.info(f"Loaded calibrator for {market}")
        return cal
