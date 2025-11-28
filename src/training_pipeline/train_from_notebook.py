"""
Training Pipeline - Production Training Script

This script takes the learnings from your notebook experiments and
provides a production-ready training pipeline with:
- Command-line interface
- MLflow experiment tracking
- Model versioning and registration
- Comprehensive logging
- Both SMOTE and Class Weight strategies
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NetworkIntrusionTrainer:
    """
    Production training class for network intrusion detection models.
    Implements both SMOTE and Class Weight strategies based on notebook experiments.
    """

    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self, data_dir: str):
        """Load preprocessed training and test data."""
        logger.info(f"Loading data from {data_dir}")

        self.X_train = pd.read_csv(Path(data_dir) / "X_train.csv")
        self.X_test = pd.read_csv(Path(data_dir) / "X_test.csv")
        self.y_train = pd.read_csv(Path(data_dir) / "y_train.csv")["label"].values
        self.y_test = pd.read_csv(Path(data_dir) / "y_test.csv")["label"].values

        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Class distribution - Benign: {(self.y_train == 0).sum()}, Attack: {(self.y_train == 1).sum()}")

        # Calculate class imbalance ratio
        self.imbalance_ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        logger.info(f"Class imbalance ratio: {self.imbalance_ratio:.2f}:1")

    def prepare_strategies(self):
        """Prepare both SMOTE and Class Weight strategies."""
        logger.info("Preparing training strategies...")

        # Strategy 1: SMOTE
        if self.config.get("use_smote", True):
            logger.info("Applying SMOTE...")
            smote = SMOTE(random_state=self.config.get("random_state", 42))
            self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
            logger.info(f"After SMOTE: {self.X_train_smote.shape}")

        # Strategy 2: Class Weights
        self.class_weights = compute_class_weight("balanced", classes=np.unique(self.y_train), y=self.y_train)
        self.class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        self.scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        logger.info(f"Class weights - Benign: {self.class_weight_dict[0]:.4f}, Attack: {self.class_weight_dict[1]:.4f}")
        logger.info(f"XGBoost scale_pos_weight: {self.scale_pos_weight:.2f}")

    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Comprehensive evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
        }

    def train_model(self, model_type: str, strategy: str, mlflow_run_name: str = None):
        """Train a single model with specified strategy."""
        logger.info(f"Training {model_type} with {strategy} strategy...")

        # Select training data based on strategy
        if strategy == "SMOTE":
            X_train_use = self.X_train_smote
            y_train_use = self.y_train_smote
            model_params = {}
        else:  # Class Weight
            X_train_use = self.X_train
            y_train_use = self.y_train
            model_params = {"class_weight": "balanced"} if model_type != "XGBoost" else {}

        # Create model
        random_state = self.config.get("random_state", 42)

        if model_type == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1, **model_params)
        elif model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=self.config.get("rf_n_estimators", 100),
                max_depth=self.config.get("rf_max_depth", 10),
                random_state=random_state,
                n_jobs=-1,
                **model_params,
            )
        elif model_type == "XGBoost":
            xgb_params = {
                "n_estimators": self.config.get("xgb_n_estimators", 100),
                "max_depth": self.config.get("xgb_max_depth", 6),
                "learning_rate": self.config.get("xgb_learning_rate", 0.1),
                "random_state": random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            if strategy == "ClassWeight":
                xgb_params["scale_pos_weight"] = self.scale_pos_weight
            model = xgb.XGBClassifier(**xgb_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train with MLflow tracking
        with mlflow.start_run(run_name=mlflow_run_name or f"{model_type}_{strategy}"):
            # Log parameters
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "strategy": strategy,
                    "imbalance_ratio": self.imbalance_ratio,
                    "n_train_samples": len(X_train_use),
                    **{k: v for k, v in self.config.items() if k.startswith(model_type.lower())},
                }
            )

            # Train model
            import time

            start_time = time.time()
            model.fit(X_train_use, y_train_use)
            train_time = time.time() - start_time

            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Evaluate
            metrics = self.evaluate_model(self.y_test, y_pred, y_pred_proba)
            metrics["train_time"] = train_time

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            mlflow.log_metrics(
                {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                    "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                }
            )

            # Log model
            if model_type == "XGBoost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            logger.info(f"Model trained - PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}")

            # Store results
            model_name = f"{model_type}_{strategy}"
            self.models[model_name] = model
            self.results[model_name] = metrics

            return model, metrics

    def train_all_models(self):
        """Train all model combinations."""
        model_types = self.config.get("model_types", ["LogisticRegression", "RandomForest", "XGBoost"])
        strategies = self.config.get("strategies", ["SMOTE", "ClassWeight"])

        for model_type in model_types:
            for strategy in strategies:
                if strategy == "SMOTE" and not self.config.get("use_smote", True):
                    continue
                self.train_model(model_type, strategy)

    def select_best_model(self, metric="pr_auc"):
        """Select best model based on specified metric."""
        logger.info(f"Selecting best model based on {metric}...")

        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Metrics: {self.results[best_model_name]}")

        return self.best_model, best_model_name

    def save_best_model(self, output_dir: str):
        """Save the best model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_path / f"best_model_{self.best_model_name.lower()}.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            "model_name": self.best_model_name,
            "model_type": type(self.best_model).__name__,
            "strategy": "SMOTE" if "SMOTE" in self.best_model_name else "ClassWeight",
            "metrics": self.results[self.best_model_name],
            "training_date": datetime.now().isoformat(),
            "imbalance_ratio": float(self.imbalance_ratio),
            "features": list(self.X_train.columns),
            "n_features": len(self.X_train.columns),
            "config": self.config,
        }

        metadata_path = output_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Save all results
        results_path = output_path / "all_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Register model with MLflow
        if self.config.get("register_model", False):
            self.register_model_mlflow()

        return model_path, metadata_path

    def register_model_mlflow(self):
        """Register the best model in MLflow Model Registry."""
        logger.info("Registering model in MLflow Model Registry...")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_name = "nidstream-intrusion-detector"

        mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered as {model_name}")


def main():
    """Main training pipeline execution."""
    parser = argparse.ArgumentParser(description="Train network intrusion detection models")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing processed data")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--use-smote", action="store_true", default=True, help="Use SMOTE strategy")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="network-intrusion-detection", help="MLflow experiment name")
    parser.add_argument("--register-model", action="store_true", help="Register best model in MLflow Model Registry")

    args = parser.parse_args()

    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load config
    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "random_state": 42,
            "use_smote": args.use_smote,
            "model_types": ["LogisticRegression", "RandomForest", "XGBoost"],
            "strategies": ["SMOTE", "ClassWeight"],
            "rf_n_estimators": 100,
            "rf_max_depth": 10,
            "xgb_n_estimators": 100,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.1,
            "register_model": args.register_model,
        }

    # Initialize trainer
    trainer = NetworkIntrusionTrainer(config)

    # Execute pipeline
    try:
        logger.info("Starting training pipeline...")

        # Load data
        trainer.load_data(args.data_dir)

        # Prepare strategies
        trainer.prepare_strategies()

        # Train all models
        trainer.train_all_models()

        # Select best model
        trainer.select_best_model(metric="pr_auc")

        # Save best model
        trainer.save_best_model(args.output_dir)

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
