# Run with: python -i   or  python smoke_test_ml.py
# Load .env file for local smoke testing
from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Setup: patch MODEL_DIR to a temp directory (no real file writes)
with tempfile.TemporaryDirectory() as tmp_dir:
    os.environ["DATABASE_URL"] = "postgresql://skip:skip@localhost/skip"
    os.environ["MODEL_DIR"] = tmp_dir
    os.environ["TICKERS"] = "AAPL"

    # ─────────────────────────────────────────────────────────────────────────────
    # TEST 1: _make_labels() — NaN pattern and label distribution
    # ─────────────────────────────────────────────────────────────────────────────
    print("=== TEST 1: _make_labels() ===")
    from marketpulse.ml.features import FEATURE_COLS, LOOKAHEAD, _make_labels

    np.random.seed(0)
    n = 100
    prices = 180.0 + np.cumsum(np.random.normal(0, 0.5, n))
    close = pd.Series(prices)

    labels = _make_labels(close)

    # Last LOOKAHEAD rows should be NaN
    assert labels.iloc[-LOOKAHEAD:].isna().all(), \
        f"Last {LOOKAHEAD} rows should be NaN"
    # All other rows should have a valid label
    assert labels.iloc[:-LOOKAHEAD].notna().all(), \
        "All non-trailing rows should have a label"
    # Labels should be in {-1.0, 0.0, 1.0}
    valid = labels.dropna()
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0}), \
        f"Unexpected label values: {valid.unique()}"

    buy_pct = (valid == 1.0).mean()
    sell_pct = (valid == -1.0).mean()
    hold_pct = (valid == 0.0).mean()
    print(f"  Labels: BUY={buy_pct:.1%} SELL={sell_pct:.1%} HOLD={hold_pct:.1%}")
    print(f"  ✓ Last {LOOKAHEAD} rows are NaN, rest have valid labels")

    # ─────────────────────────────────────────────────────────────────────────────
    # TEST 2: train_classifier() — training and prediction format
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n=== TEST 2: train_classifier() ===")
    from marketpulse.ml.classifier import SIGNAL_MAP, train_classifier
    from marketpulse.ml.features import FEATURE_COLS  # noqa: F811

    n_train = 200
    np.random.seed(42)
    X_synth = pd.DataFrame(
        np.random.randn(n_train, len(FEATURE_COLS)),
        columns=FEATURE_COLS,
    )
    # Scale bb_position to [0, 1] (its real range)
    X_synth["bb_position"] = X_synth["bb_position"].abs() % 1.0
    # Scale rsi_14 to [20, 80]
    X_synth["rsi_14"] = 50 + X_synth["rsi_14"] * 15

    y_synth = pd.Series(
        np.random.choice([-1, 0, 1], size=n_train, p=[0.3, 0.4, 0.3])
    )

    pipeline, accuracy = train_classifier(X_synth, y_synth, "AAPL_TEST")
    print(f"  Accuracy on random data: {accuracy:.4f}")
    print("  (Expected ~0.33–0.55 for random labels — higher means the model")
    print("   accidentally found a pattern in the synthetic noise)")

    # pipeline should have .predict() and .predict_proba()
    X_single = X_synth.iloc[[-1]]
    pred = pipeline.predict(X_single)
    proba = pipeline.predict_proba(X_single)
    assert int(pred[0]) in [-1, 0, 1], f"Unexpected class: {pred[0]}"
    assert abs(sum(proba[0]) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
    print(f"  Single prediction: {SIGNAL_MAP[int(pred[0])]} "
          f"(conf={max(proba[0]):.4f})")
    print("  ✓ Pipeline produces valid predictions and probabilities")

    # ─────────────────────────────────────────────────────────────────────────────
    # TEST 3: save_classifier() + load_classifier() roundtrip
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n=== TEST 3: save/load roundtrip ===")
    from pathlib import Path

    # Mock the DB session — save_classifier writes to ModelRegistry
    from unittest.mock import MagicMock

    import joblib  # noqa: F401

    from marketpulse.ml.classifier import load_classifier, save_classifier
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.update.return_value = 1

    file_path = save_classifier(pipeline, "AAPL_TEST", accuracy, mock_session)
    print(f"  Saved to: {file_path}")
    assert Path(file_path).exists(), "File should exist after save"

    loaded = load_classifier("AAPL_TEST")
    assert loaded is not None, "load_classifier should return a Pipeline"
    pred_loaded = loaded.predict(X_single)
    assert pred_loaded[0] == pred[0], "Loaded model should produce same prediction"
    print(f"  Loaded prediction matches: {SIGNAL_MAP[int(pred_loaded[0])]}")
    print("  ✓ Save/load roundtrip produces identical predictions")

    # ─────────────────────────────────────────────────────────────────────────────
    # TEST 4: train_anomaly_detector() + prediction
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n=== TEST 4: IsolationForest anomaly detector ===")
    from marketpulse.ml.anomaly import (
        load_anomaly_detector,
        save_anomaly_detector,
        train_anomaly_detector,
    )

    iso = train_anomaly_detector(X_synth, "AAPL_TEST")
    predictions_train = iso.predict(X_synth)
    n_anomalies = (predictions_train == -1).sum()
    print(f"  Training anomalies: {n_anomalies}/{len(X_synth)} "
          f"({n_anomalies/len(X_synth):.1%})")

    # Save and reload
    save_anomaly_detector(iso, "AAPL_TEST", mock_session)
    loaded_iso = load_anomaly_detector("AAPL_TEST")
    assert loaded_iso is not None
    pred_iso = loaded_iso.predict(X_single)[0]
    print(f"  Single bar prediction: {'ANOMALY' if pred_iso == -1 else 'NORMAL'}")
    print("  ✓ Anomaly detector trained, saved, and reloaded")

    # ─────────────────────────────────────────────────────────────────────────────
    # TEST 5: PredictionService — full predict flow
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n=== TEST 5: PredictionService ===")
    from marketpulse.ml.service import PredictionService

    service = PredictionService()

    # First call: loads from disk
    result = service.predict("AAPL_TEST", X_synth)
    assert result is not None, "Should return a PredictionResult"
    assert result["signal"] in ("BUY", "HOLD", "SELL")
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["is_anomaly"], bool)
    print(f"  First predict (cold): {result['signal']} "
          f"(conf={result['confidence']:.4f}, anomaly={result['is_anomaly']})")

    # Second call: from in-memory cache (much faster)
    result2 = service.predict("AAPL_TEST", X_synth)
    assert result2 == result, "Cached result should match"
    print("  Second predict (cached): same result ✓")

    # Test cache invalidation
    service.invalidate_cache("AAPL_TEST")
    assert "AAPL_TEST" not in service._classifiers, "Cache should be empty"
    print("  Cache invalidated ✓")

    # Test missing model returns None
    result_missing = service.predict("ZZZZ_NOTREAL", X_synth)
    assert result_missing is None, "Missing model should return None"
    print("  Missing model returns None ✓")

    print("\n=== All ML smoke tests passed ✓ ===")
