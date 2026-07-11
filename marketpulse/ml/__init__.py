# marketpulse/ml/__init__.py
# Public API of the ml package.
# Scheduler uses everything; API uses PredictionService only.

from marketpulse.ml.anomaly import (
    load_anomaly_detector,
    save_anomaly_detector,
    train_anomaly_detector,
)
from marketpulse.ml.classifier import (
    SIGNAL_MAP,
    SIGNAL_REVERSE,
    load_classifier,
    save_classifier,
    train_classifier,
)
from marketpulse.ml.features import (
    FEATURE_COLS,
    LOOKAHEAD,
    THRESHOLD,
    build_feature_matrix,
)
from marketpulse.ml.service import PredictionResult, PredictionService

__all__ = [
    # Feature engineering
    "FEATURE_COLS",
    "LOOKAHEAD",
    "THRESHOLD",
    "build_feature_matrix",
    # Classifier
    "SIGNAL_MAP",
    "SIGNAL_REVERSE",
    "train_classifier",
    "save_classifier",
    "load_classifier",
    # Anomaly detector
    "train_anomaly_detector",
    "save_anomaly_detector",
    "load_anomaly_detector",
    # Prediction service
    "PredictionResult",
    "PredictionService",
]
