# tests/unit/test_ml.py
# Unit tests for the ML layer:
# - _make_labels(): NaN pattern, label distribution
# - build_feature_matrix(): insufficient data path (mocked session)
# - train_classifier(): output types, insufficient data guard
# - train_anomaly_detector(): output type, training data check
# - PredictionService: None when no model, cache invalidation

from __future__ import annotations

import os  # noqa: F401
import tempfile  # noqa: F401
from datetime import datetime, timezone  # noqa: F401
from unittest.mock import MagicMock, patch  # noqa: F401

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from marketpulse.ml.features import FEATURE_COLS, LOOKAHEAD, _make_labels

# ══════════════════════════════════════════════════════════════════════════════
# Label generation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeLabels:
    """Tests for the forward-looking label generator."""

    def test_last_n_rows_are_nan(self):
        """Last LOOKAHEAD rows have no future data → must be NaN."""
        close = pd.Series(range(50, 100, dtype=float))  # 50 values, 50–99
        labels = _make_labels(close)
        assert labels.iloc[-LOOKAHEAD:].isna().all()

    def test_non_trailing_rows_have_labels(self):
        """All rows except the last LOOKAHEAD should have a label."""
        close = pd.Series(range(50, 100, dtype=float))
        labels = _make_labels(close)
        non_trailing = labels.iloc[:-LOOKAHEAD]
        assert non_trailing.notna().all()

    def test_labels_only_contain_valid_values(self):
        """Labels must be exactly -1.0, 0.0, or 1.0 (or NaN for trailing)."""
        np.random.seed(7)
        close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))
        labels = _make_labels(close).dropna()
        assert set(labels.unique()).issubset({-1.0, 0.0, 1.0})

    def test_all_gains_produces_buy_labels(self):
        """A strictly rising price series should produce all BUY labels."""
        # Large gains (> 0.5%) on every bar → all BUY
        close = pd.Series([100.0 * (1.01 ** i) for i in range(30)])
        labels = _make_labels(close, threshold=0.005).dropna()
        assert (labels == 1.0).all()

    def test_all_losses_produces_sell_labels(self):
        """A strictly falling price series should produce all SELL labels."""
        close = pd.Series([100.0 * (0.99 ** i) for i in range(30)])
        labels = _make_labels(close, threshold=0.005).dropna()
        assert (labels == -1.0).all()


# ══════════════════════════════════════════════════════════════════════════════
# build_feature_matrix() tests (mocked session)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatureMatrix:
    """Tests for feature matrix construction — uses mocked DB session."""

    def test_empty_db_returns_empty_matrix(self, mock_db):
        """No rows in DB → return empty DataFrames."""
        from marketpulse.ml.features import build_feature_matrix

        X, y, cols = build_feature_matrix("AAPL", mock_db, lookback_days=90)

        assert X.empty
        assert y.empty
        assert cols == FEATURE_COLS

    def test_insufficient_rows_returns_empty(self, mock_db):
        """Fewer than 50 usable feature rows → return empty (enforced in function)."""
        # The mock_db fixture returns [] for .all() calls
        # This simulates no data → feature matrix will be empty
        from marketpulse.ml.features import build_feature_matrix

        X, y, _ = build_feature_matrix("AAPL", mock_db, lookback_days=90)
        assert X.empty


# ══════════════════════════════════════════════════════════════════════════════
# Classifier training tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainClassifier:
    """Tests for RandomForest BUY/HOLD/SELL classifier training."""

    def test_returns_pipeline_and_float(self, feature_df, feature_labels):
        from marketpulse.ml.classifier import train_classifier

        clf, accuracy = train_classifier(feature_df, feature_labels, "AAPL_TEST")

        assert isinstance(clf, Pipeline)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_pipeline_has_predict_and_predict_proba(self, feature_df, feature_labels):
        from marketpulse.ml.classifier import train_classifier

        clf, _ = train_classifier(feature_df, feature_labels, "AAPL_TEST")

        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")

    def test_predict_returns_valid_class(self, feature_df, feature_labels):
        from marketpulse.ml.classifier import SIGNAL_MAP, train_classifier

        clf, _ = train_classifier(feature_df, feature_labels, "AAPL_TEST")
        pred = int(clf.predict(feature_df.iloc[[-1]])[0])

        assert pred in SIGNAL_MAP.keys()  # -1, 0, or 1

    def test_predict_proba_sums_to_one(self, feature_df, feature_labels):
        from marketpulse.ml.classifier import train_classifier

        clf, _ = train_classifier(feature_df, feature_labels, "AAPL_TEST")
        proba = clf.predict_proba(feature_df.iloc[[-1]])[0]

        assert abs(sum(proba) - 1.0) < 1e-6

    def test_raises_on_insufficient_data(self):
        from marketpulse.ml.classifier import train_classifier

        tiny_X = pd.DataFrame(np.random.randn(10, len(FEATURE_COLS)), columns=FEATURE_COLS)
        tiny_y = pd.Series([1, 0, -1, 1, 0, -1, 1, 0, -1, 1])

        with pytest.raises(ValueError, match="Insufficient data"):
            train_classifier(tiny_X, tiny_y, "AAPL")


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly detector tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainAnomalyDetector:
    """Tests for IsolationForest anomaly detector training."""

    def test_returns_isolation_forest(self, feature_df):
        from sklearn.ensemble import IsolationForest

        from marketpulse.ml.anomaly import train_anomaly_detector

        iso = train_anomaly_detector(feature_df, "AAPL_TEST")

        assert isinstance(iso, IsolationForest)

    def test_predict_returns_minus_one_or_one(self, feature_df):
        from marketpulse.ml.anomaly import train_anomaly_detector

        iso = train_anomaly_detector(feature_df, "AAPL_TEST")
        preds = iso.predict(feature_df)

        assert set(preds).issubset({-1, 1})

    def test_anomaly_fraction_near_contamination(self, feature_df):
        from marketpulse.ml.anomaly import train_anomaly_detector

        # contamination=0.05 means ~5% of training data should be flagged
        iso = train_anomaly_detector(feature_df, "AAPL_TEST")
        preds = iso.predict(feature_df)
        anomaly_rate = (preds == -1).mean()

        # Allow ± 3% tolerance around the 5% contamination parameter
        assert 0.02 <= anomaly_rate <= 0.08


# ══════════════════════════════════════════════════════════════════════════════
# PredictionService tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionService:
    """Tests for the lazy-loading PredictionService."""

    def test_predict_returns_none_when_no_model(self, tmp_path, feature_df, monkeypatch):
        """If no .pkl file exists, predict() should return None (not raise)."""
        from marketpulse.ml.service import PredictionService

        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        service = PredictionService()

        result = service.predict("AAPL_NOMODEL", feature_df)

        assert result is None

    def test_predict_returns_valid_signal(self, tmp_path, feature_df, feature_labels, monkeypatch):
        """After training + saving, predict() should return a PredictionResult."""
        from unittest.mock import MagicMock  # noqa: F811

        from marketpulse.ml.classifier import save_classifier, train_classifier
        from marketpulse.ml.service import PredictionService

        monkeypatch.setenv("MODEL_DIR", str(tmp_path))

        # Train and save a classifier
        clf, acc = train_classifier(feature_df, feature_labels, "AAPL_TEST")
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.update.return_value = 0
        save_classifier(clf, "AAPL_TEST", acc, mock_session)

        # PredictionService should load and use it
        service = PredictionService()
        result = service.predict("AAPL_TEST", feature_df)

        assert result is not None
        assert result["signal"] in ("BUY", "HOLD", "SELL")
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["is_anomaly"], bool)

    def test_invalidate_cache_clears_specific_ticker(self, feature_df, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        from marketpulse.ml.service import PredictionService

        service = PredictionService()
        service._classifiers["AAPL"] = object()   # put something in cache
        service._classifiers["MSFT"] = object()

        service.invalidate_cache("AAPL")

        assert "AAPL" not in service._classifiers
        assert "MSFT" in service._classifiers   # MSFT unaffected

    def test_invalidate_all_clears_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        from marketpulse.ml.service import PredictionService

        service = PredictionService()
        service._classifiers["AAPL"] = object()
        service._anomaly_detectors["AAPL"] = object()

        service.invalidate_cache()  # no ticker → clear all

        assert not service._classifiers
        assert not service._anomaly_detectors
