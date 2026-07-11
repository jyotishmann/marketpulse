# marketpulse/ml/classifier.py
# RandomForest BUY/HOLD/SELL classifier using scikit-learn Pipeline.
# Train → save to disk → load → predict.

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Signal mapping
# Map between integer labels (from feature engineering) and signal strings (API/DB)
SIGNAL_MAP: dict[int, str] = {1: "BUY", 0: "HOLD", -1: "SELL"}
SIGNAL_REVERSE: dict[str, int] = {"BUY": 1, "HOLD": 0, "SELL": -1}


# Training

def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    ticker: str,
) -> tuple[Pipeline, float]:
    """
    Train a RandomForest BUY/HOLD/SELL classifier on the feature matrix.

    Uses a time-series chronological split (80/20): train on older bars,
    test on more recent bars. Never shuffles — that would leak future data.

    Args:
        X:      Feature matrix from build_feature_matrix(). Min 50 rows.
        y:      Integer labels: 1=BUY, 0=HOLD, -1=SELL.
        ticker: Used for log messages only.

    Returns:
        Tuple of:
        - pipeline: Fitted Pipeline(StandardScaler, RandomForestClassifier).
          Call pipeline.predict(X_new) or pipeline.predict_proba(X_new).
        - accuracy: Test-set accuracy (0.0–1.0). Stored in ModelRegistry.

    Raises:
        ValueError: If fewer than 50 rows provided.
    """
    if len(X) < 50:
        raise ValueError(
            f"Insufficient data for {ticker}: {len(X)} rows (minimum 50 required). "
            "Run more ingestion cycles to accumulate data."
        )

    # Chronological 80/20 split — NEVER shuffle time-series data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Training classifier for %s: %d train rows, %d test rows",
        ticker, len(X_train), len(X_test),
    )

    # Log label distribution in training set
    dist = y_train.value_counts().to_dict()
    logger.debug(
        "Train label distribution: BUY=%d HOLD=%d SELL=%d",
        dist.get(1, 0), dist.get(0, 0), dist.get(-1, 0),
    )

    # Build and fit the Pipeline
    pipeline = Pipeline([
        # Step 1: StandardScaler — normalise all features to mean=0, std=1.
        # Fitted on train data only; .transform() applied automatically at predict time.
        ("scaler", StandardScaler()),

        # Step 2: RandomForest — ensemble of 100 decision trees.
        # max_depth=10 and min_samples_leaf=5 prevent overfitting.
        ("clf", RandomForestClassifier(
            n_estimators=100,      # number of trees in the forest
            max_depth=10,          # max tree depth (prevents memorisation)
            min_samples_leaf=5,    # each leaf must have ≥5 samples (smoothing)
            class_weight="balanced",  # handle class imbalance (HOLD often dominates)
            random_state=42,       # reproducible results
            n_jobs=-1,             # use all CPU cores
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on the held-out test set
    accuracy = float(pipeline.score(X_test, y_test))

    # Log class-level predictions for insight
    y_pred = pipeline.predict(X_test)
    pred_series = pd.Series(y_pred)
    pred_dist = pred_series.value_counts().to_dict()
    logger.info(
        "Classifier for %s: accuracy=%.4f | test predictions: "
        "BUY=%d HOLD=%d SELL=%d",
        ticker, accuracy,
        pred_dist.get(1, 0), pred_dist.get(0, 0), pred_dist.get(-1, 0),
    )

    return pipeline, accuracy

# Persistence

import os  # noqa: E402, F401
from datetime import UTC, datetime, timezone  # noqa: E402, F401
from decimal import Decimal  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from marketpulse.config import settings  # noqa: E402
from marketpulse.db import ModelRegistry  # noqa: E402
from marketpulse.ml.features import FEATURE_COLS  # noqa: E402


def save_classifier(
    pipeline: Pipeline,
    ticker: str,
    accuracy: float,
    session: Session,
) -> str:
    """
    Persist the trained pipeline to disk and record it in ModelRegistry.

    Saves a metadata bundle (pipeline + feature_cols + ticker) so the loader
    can verify the model matches the current feature set. Deactivates any
    previously active classifier for this ticker in ModelRegistry.

    Args:
        pipeline: Fitted Pipeline from train_classifier().
        ticker:   Stock symbol.
        accuracy: Test-set accuracy from train_classifier().
        session:  Active SQLAlchemy session.

    Returns:
        Absolute path to the saved .pkl file.
    """
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = str(model_dir / f"{ticker}_classifier.pkl")

    # Bundle model with metadata for safe loading
    bundle = {
        "pipeline": pipeline,
        "feature_cols": FEATURE_COLS,   # which features this model expects
        "ticker": ticker,
        "trained_at": datetime.now(tz=UTC).isoformat(),
    }
    joblib.dump(bundle, file_path)
    logger.info("Saved classifier bundle for %s to %s", ticker, file_path)

    # Update ModelRegistry
    # Deactivate all previously active classifiers for this ticker
    session.query(ModelRegistry).filter(
        ModelRegistry.ticker == ticker,
        ModelRegistry.model_type == "classifier",
        ModelRegistry.is_active.is_(True),
    ).update({"is_active": False})

    # Register the new model as active
    session.add(
        ModelRegistry(
            ticker=ticker,
            model_type="classifier",
            file_path=file_path,
            accuracy=Decimal(str(round(accuracy, 4))),
            is_active=True,
        )
    )
    session.commit()

    logger.info(
        "Registered classifier for %s in ModelRegistry (accuracy=%.4f)",
        ticker, accuracy,
    )
    return file_path


def load_classifier(ticker: str) -> Pipeline | None:
    """
    Load the active classifier for a ticker from disk.

    Returns None (does not raise) if no model file exists yet.
    Validates that the saved feature columns match the current FEATURE_COLS.

    Args:
        ticker: Stock symbol.

    Returns:
        Fitted Pipeline, or None if the model has not been trained yet.
    """
    file_path = Path(settings.model_dir) / f"{ticker}_classifier.pkl"

    if not file_path.exists():
        logger.info(
            "No classifier file for %s at %s — model not yet trained",
            ticker, file_path,
        )
        return None

    try:
        bundle: dict = joblib.load(file_path)
    except Exception:
        logger.exception("Failed to load classifier bundle for %s from %s", ticker, file_path)
        return None

    # Validate feature columns match current FEATURE_COLS
    saved_cols = bundle.get("feature_cols", [])
    if saved_cols != FEATURE_COLS:
        logger.error(
            "Classifier for %s was trained on different features! "
            "Saved: %s | Current: %s. Retrain the model.",
            ticker, saved_cols, FEATURE_COLS,
        )
        return None

    logger.debug(
        "Loaded classifier for %s (trained_at=%s)",
        ticker, bundle.get("trained_at", "unknown"),
    )
    return bundle["pipeline"]
