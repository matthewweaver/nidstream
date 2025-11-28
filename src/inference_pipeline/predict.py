"""
Inference Pipeline - Production Prediction Module

Loads trained model and makes predictions on new network flow data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntrusionDetector:
    """
    Production inference class for network intrusion detection.
    """

    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize detector with trained model.

        Args:
            model_path: Path to saved model (.pkl file)
            metadata_path: Path to model metadata (.json file)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.feature_names = None

        self.load_model()

        if metadata_path:
            self.load_metadata(metadata_path)

    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded successfully: {type(self.model).__name__}")

    def load_metadata(self, metadata_path: str):
        """Load model metadata."""
        metadata_path = Path(metadata_path)
        logger.info(f"Loading metadata from {metadata_path}")

        # Support both .pkl and .json formats
        if metadata_path.suffix == ".pkl":
            self.metadata = joblib.load(metadata_path)
        else:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        self.feature_names = self.metadata.get("features", [])
        logger.info(f"Loaded metadata for model: {self.metadata.get('model_name')}")
        logger.info(f"Expected features: {len(self.feature_names)}")

    def validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and align input features with expected features.

        Args:
            X: Input DataFrame

        Returns:
            Validated DataFrame with correct feature order
        """
        if self.feature_names is None:
            logger.warning("No feature names loaded - skipping validation")
            return X

        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0

        # Check for extra features
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features (will be dropped): {extra_features}")

        # Ensure correct order
        X = X[self.feature_names]

        return X

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> Union[np.ndarray, tuple]:
        """
        Make predictions on input data.

        Args:
            X: Input DataFrame with network flow features
            return_proba: If True, also return prediction probabilities

        Returns:
            predictions: Binary predictions (0=benign, 1=attack)
            probabilities: (optional) Attack probabilities
        """
        # Validate features
        X_validated = self.validate_features(X)

        # Make predictions
        predictions = self.model.predict(X_validated)

        if return_proba:
            probabilities = self.model.predict_proba(X_validated)[:, 1]
            return predictions, probabilities

        return predictions

    def predict_single(self, flow_features: Dict) -> Dict:
        """
        Make prediction on a single network flow.

        Args:
            flow_features: Dictionary of flow features

        Returns:
            Dictionary with prediction result and probability
        """
        # Convert to DataFrame
        X = pd.DataFrame([flow_features])

        # Make prediction
        prediction, probability = self.predict(X, return_proba=True)

        result = {
            "prediction": int(prediction[0]),
            "prediction_label": "attack" if prediction[0] == 1 else "benign",
            "attack_probability": float(probability[0]),
            "confidence": float(max(probability[0], 1 - probability[0])),
        }

        return result

    def predict_batch(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions on a batch of flows with detailed results.

        Args:
            X: DataFrame of network flows
            threshold: Probability threshold for classification

        Returns:
            DataFrame with original data plus predictions
        """
        # Make predictions
        predictions, probabilities = self.predict(X, return_proba=True)

        # Create results DataFrame
        results = X.copy()
        results["prediction"] = predictions
        results["attack_probability"] = probabilities
        results["prediction_label"] = results["prediction"].map({0: "benign", 1: "attack"})
        results["confidence"] = np.maximum(probabilities, 1 - probabilities)

        # Flag high-confidence attacks
        results["high_confidence_attack"] = (results["prediction"] == 1) & (results["attack_probability"] >= 0.8)

        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_type": type(self.model).__name__,
            "model_path": str(self.model_path),
        }

        if self.metadata:
            info.update(
                {
                    "model_name": self.metadata.get("model_name"),
                    "strategy": self.metadata.get("strategy"),
                    "training_date": self.metadata.get("training_date"),
                    "metrics": self.metadata.get("metrics"),
                    "n_features": self.metadata.get("n_features"),
                }
            )

        return info


def load_detector(model_dir: str = "models") -> IntrusionDetector:
    """
    Convenience function to load the best model.

    Args:
        model_dir: Directory containing model files

    Returns:
        IntrusionDetector instance
    """
    model_dir = Path(model_dir)

    # Load the best model
    model_path = model_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")

    logger.info(f"Loading model: {model_path.name}")

    # Find metadata file - try .pkl first, then .json
    metadata_path = model_dir / "model_metadata.pkl"
    if not metadata_path.exists():
        metadata_path = model_dir / "model_metadata.json"
        if not metadata_path.exists():
            metadata_path = None
            logger.warning("Metadata file not found")

    return IntrusionDetector(str(model_path), str(metadata_path) if metadata_path else None)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on network flow data")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing model files")
    parser.add_argument("--input-file", type=str, required=True, help="Input CSV file with network flows")
    parser.add_argument("--output-file", type=str, default="predictions.csv", help="Output file for predictions")

    args = parser.parse_args()

    # Load detector
    detector = load_detector(args.model_dir)

    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    data = pd.read_csv(args.input_file)

    # Make predictions
    logger.info("Making predictions...")
    results = detector.predict_batch(data)

    # Save results
    results.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to {args.output_file}")

    # Print summary
    n_attacks = (results["prediction"] == 1).sum()
    logger.info(f"Summary: {n_attacks}/{len(results)} flows flagged as attacks")
