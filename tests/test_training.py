"""
Unit tests for training pipeline
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_val = pd.DataFrame(np.random.randn(20, 10), columns=[f"feature_{i}" for i in range(10)])
    y_val = pd.Series(np.random.randint(0, 2, 20))
    return X_train, y_train, X_val, y_val


def test_isolation_forest_training(sample_training_data):
    """Test Isolation Forest model training."""
    X_train, _, _, _ = sample_training_data

    model = IsolationForest(contamination=0.1, n_estimators=10, random_state=42)
    model.fit(X_train)

    # Test prediction
    predictions = model.predict(X_train)
    assert len(predictions) == len(X_train)
    assert set(predictions).issubset({-1, 1})


def test_model_artifacts_structure():
    """Test model artifact structure."""
    artifacts = {"model": IsolationForest(), "scaler": None, "feature_cols": ["feature_0", "feature_1"]}

    assert "model" in artifacts
    assert "scaler" in artifacts
    assert "feature_cols" in artifacts
    assert len(artifacts["feature_cols"]) == 2
