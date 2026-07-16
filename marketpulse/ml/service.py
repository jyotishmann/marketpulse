# marketpulse/ml/service.py
# PredictionService: lazy-loads trained models and generates signals.
# Single interface for both the scheduler (training) and the API (inference).

from __future__ import annotations

import logging
from datetime import UTC, datetime, timezone  # noqa: F401
from decimal import Decimal
from typing import TypedDict

import pandas as pd
from sqlalchemy.orm import Session

from marketpulse.ml.anomaly import load_anomaly_detector
from marketpulse.ml.classifier import SIGNAL_MAP, load_classifier

logger = logging.getLogger(__name__)


class PredictionResult(TypedDict):
    """
    Typed dict returned by PredictionService.predict().

    All downstream consumers (scheduler, API router, dashboard) depend
    on exactly these keys being present with these types.
    """

    ticker: str
    signal: str  # "BUY", "HOLD", or "SELL"
    confidence: float  # probability of the winning class (0.0–1.0)
    is_anomaly: bool  # True if IsolationForest flagged this bar
    model_version: str  # human-readable version string from ModelRegistry


class PredictionService:
    """
    Generates ML predictions using lazy-loaded, cached models.

    Usage:
        service = PredictionService()  # create once (e.g. in api/main.py)

        # Generate a prediction for the latest bar:
        result = service.predict("AAPL", X_latest)

        # After retraining, clear the cache so models reload on next predict():
        service.invalidate_cache("AAPL")
    """

    def __init__(self) -> None:
        # In-memory model caches — loaded lazily on first .predict() call
        from typing import Any

        self._classifiers: dict[str, Any] = {}  # ticker → fitted Pipeline
        self._anomaly_detectors: dict[str, Any] = {}  # ticker → fitted IsolationForest

    def predict(
        self,
        ticker: str,
        X_latest: pd.DataFrame,
    ) -> PredictionResult | None:
        """
        Generate a BUY/HOLD/SELL signal and anomaly flag for the latest bar.

        Loads models from disk on first call per ticker; uses in-memory
        cache on subsequent calls.

        Args:
            ticker:    Stock symbol.
            X_latest:  Feature DataFrame with FEATURE_COLS columns.
                       Should contain at least the most recent row.
                       The last row (iloc[-1]) is used for prediction.

        Returns:
            PredictionResult dict, or None if:
            - X_latest is empty
            - No trained classifier exists for this ticker (not yet trained)
        """
        if X_latest.empty:
            logger.warning("predict(%s): received empty feature DataFrame", ticker)
            return None

        # Use only the most recent bar — [[...]] keeps it as a 1-row DataFrame
        # (single brackets would give a Series, which predict() rejects)
        X_row = X_latest.iloc[[-1]]

        # ── Lazy load classifier ───────────────────────────────────────────────
        if ticker not in self._classifiers:
            clf = load_classifier(ticker)
            if clf is None:
                logger.warning(
                    "No trained classifier for %s — call run_ml_pipeline() first",
                    ticker,
                )
                return None
            self._classifiers[ticker] = clf
            logger.debug("Cached classifier for %s in memory", ticker)

        clf = self._classifiers[ticker]

        # ── Lazy load anomaly detector (optional — may be None) ───────────────
        if ticker not in self._anomaly_detectors:
            iso = load_anomaly_detector(ticker)
            # Store None explicitly so we don't retry disk load on every call
            self._anomaly_detectors[ticker] = iso
            if iso:
                logger.debug("Cached anomaly detector for %s in memory", ticker)

        iso = self._anomaly_detectors.get(ticker)

        # ── Run classifier ─────────────────────────────────────────────────────
        # predict_proba: [[P(SELL), P(HOLD), P(BUY)]] (order matches clf.classes_)
        proba = clf.predict_proba(X_row)[0]
        pred_int = int(clf.predict(X_row)[0])
        signal = SIGNAL_MAP.get(pred_int, "HOLD")
        confidence = round(float(max(proba)), 4)

        # ── Run anomaly detector ───────────────────────────────────────────────
        is_anomaly = False
        if iso is not None:
            # IsolationForest.predict() returns -1 (anomaly) or 1 (normal)
            iso_pred = int(iso.predict(X_row)[0])
            is_anomaly = iso_pred == -1

        result: PredictionResult = {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "is_anomaly": is_anomaly,
            "model_version": "v1",  # FILE_06 will read this from ModelRegistry
        }

        logger.info(
            "Prediction for %s: %s (conf=%.2f) anomaly=%s",
            ticker,
            signal,
            confidence,
            is_anomaly,
        )
        return result

    # ── Signal persistence and cache management ────────────────────────────────────

    def save_signal(
        self,
        result: PredictionResult,
        session: Session,
    ) -> None:
        """
        Persist a PredictionResult to the ml_signals table.

        Uses ON CONFLICT DO NOTHING so repeated saves within the same
        second do not raise errors.

        Args:
            result:  PredictionResult from .predict().
            session: Active SQLAlchemy session.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from marketpulse.db import MLSignal

        record = {
            "ticker": result["ticker"],
            "timestamp": datetime.now(tz=UTC),  # when prediction was made
            "signal": result["signal"],
            "confidence": Decimal(str(result["confidence"])),
            "is_anomaly": result["is_anomaly"],
            "model_version": result["model_version"],
        }

        try:
            stmt = pg_insert(MLSignal).values([record])
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_ml_signals_ticker_ts",
            )
            session.execute(stmt)
            session.commit()
            logger.debug(
                "Saved signal for %s: %s (conf=%.2f)",
                result["ticker"],
                result["signal"],
                result["confidence"],
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to save signal for %s", result["ticker"])
            raise

    def invalidate_cache(self, ticker: str | None = None) -> None:
        """
        Clear the in-memory model cache to force reload on next predict().

        Call this after retraining models for a ticker so the service
        picks up the new model files from disk.

        Args:
            ticker: If provided, clear cache for this ticker only.
                    If None, clear all cached models.
        """
        if ticker is not None:
            removed_clf = self._classifiers.pop(ticker, None)
            removed_iso = self._anomaly_detectors.pop(ticker, None)
            logger.info(
                "Cache invalidated for %s (classifier=%s, anomaly=%s)",
                ticker,
                "evicted" if removed_clf else "was not cached",
                "evicted" if removed_iso else "was not cached",
            )
        else:
            n_clf = len(self._classifiers)
            n_iso = len(self._anomaly_detectors)
            self._classifiers.clear()
            self._anomaly_detectors.clear()
            logger.info(
                "All model caches cleared (%d classifiers, %d anomaly detectors)",
                n_clf,
                n_iso,
            )
