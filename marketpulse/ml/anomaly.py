# marketpulse/ml/anomaly.py
# Isolation Forest anomaly detector — unsupervised, no labels needed.
# Learns the distribution of "normal" bars and flags deviations.

from __future__ import annotations

import logging
from datetime import UTC, datetime, timezone  # noqa: F401
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy.orm import Session

from marketpulse.config import settings
from marketpulse.db import ModelRegistry

logger = logging.getLogger(__name__)


def train_anomaly_detector(
    X: pd.DataFrame,
    ticker: str,
) -> IsolationForest:
    """
    Fit an Isolation Forest on the feature matrix.

    Uses all available data (no train/test split — unsupervised).
    The contamination parameter sets the fraction of training data
    expected to be anomalous (sets the decision threshold).

    Args:
        X:      Feature matrix from build_feature_matrix() (same as classifier).
        ticker: Used for log messages.

    Returns:
        Fitted IsolationForest. Call .predict(X_new) for inference.
        Returns -1 (anomaly) or 1 (normal).
    """
    iso = IsolationForest(
        n_estimators=100,                                  # number of trees
        contamination=settings.ml_anomaly_contamination,  # expected anomaly fraction
        max_samples="auto",    # use min(256, n_samples) samples per tree
        random_state=42,
        n_jobs=-1,
    )

    iso.fit(X)

    # Sanity check: count anomalies in training data
    predictions = iso.predict(X)
    n_anomalies = int((predictions == -1).sum())
    expected_anomalies = int(len(X) * settings.ml_anomaly_contamination)

    logger.info(
        "Anomaly detector trained for %s: %d/%d training-set anomalies "
        "(contamination=%.2f → expected ~%d)",
        ticker,
        n_anomalies,
        len(X),
        settings.ml_anomaly_contamination,
        expected_anomalies,
    )
    return iso


def save_anomaly_detector(
    iso: IsolationForest,
    ticker: str,
    session: Session,
) -> str:
    """
    Persist the fitted IsolationForest and register it in ModelRegistry.

    Args:
        iso:     Fitted IsolationForest from train_anomaly_detector().
        ticker:  Stock symbol.
        session: Active SQLAlchemy session.

    Returns:
        Absolute path to the saved .pkl file.
    """
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = str(model_dir / f"{ticker}_anomaly.pkl")

    # Bundle with metadata for safe loading
    bundle = {
        "iso": iso,
        "ticker": ticker,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "contamination": settings.ml_anomaly_contamination,
    }
    joblib.dump(bundle, file_path)

    # Deactivate previous anomaly models for this ticker
    session.query(ModelRegistry).filter(
        ModelRegistry.ticker == ticker,
        ModelRegistry.model_type == "anomaly",
        ModelRegistry.is_active.is_(True),
    ).update({"is_active": False})

    # Register new model (no accuracy — unsupervised model)
    session.add(
        ModelRegistry(
            ticker=ticker,
            model_type="anomaly",
            file_path=file_path,
            accuracy=None,   # unsupervised: no ground-truth labels → no accuracy
            is_active=True,
        )
    )
    session.commit()

    logger.info("Saved anomaly detector for %s to %s", ticker, file_path)
    return file_path


def load_anomaly_detector(ticker: str) -> IsolationForest | None:
    """
    Load the active IsolationForest for a ticker from disk.

    Returns None (does not raise) if the model file does not exist.

    Args:
        ticker: Stock symbol.

    Returns:
        Fitted IsolationForest, or None if not yet trained.
    """
    file_path = Path(settings.model_dir) / f"{ticker}_anomaly.pkl"

    if not file_path.exists():
        logger.info(
            "No anomaly detector file for %s at %s",
            ticker, file_path,
        )
        return None

    try:
        bundle: dict = joblib.load(file_path)
    except Exception:
        logger.exception("Failed to load anomaly detector for %s", ticker)
        return None

    logger.debug(
        "Loaded anomaly detector for %s (trained_at=%s)",
        ticker, bundle.get("trained_at", "unknown"),
    )
    return bundle["iso"]
