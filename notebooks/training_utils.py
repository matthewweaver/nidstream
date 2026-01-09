"""
Shared utilities for model training notebooks.
Eliminates code duplication across model training notebooks.
"""

import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def load_training_data(use_smote=False):
    """
    Load training and test data.

    Args:
        use_smote: If True, load SMOTE-balanced training data

    Returns:
        tuple: (X_train, X_test, y_train, y_test, project_root)
    """
    project_root = Path().resolve()
    processed_dir = project_root / "data" / "processed"

    print(f"Loading {'SMOTE' if use_smote else 'original'} training data...")

    if use_smote:
        X_train = pd.read_csv(processed_dir / "X_train_smote.csv", dtype=np.float32)
        y_train = pd.read_csv(processed_dir / "y_train_smote.csv")["label"].values
    else:
        X_train = pd.read_csv(processed_dir / "X_train.csv", dtype=np.float32)
        y_train = pd.read_csv(processed_dir / "y_train.csv")["label"].values

    X_test = pd.read_csv(processed_dir / "X_test.csv", dtype=np.float32)
    y_test = pd.read_csv(processed_dir / "y_test.csv")["label"].values

    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Train class distribution: Benign={np.sum(y_train == 0)}, Attack={np.sum(y_train == 1)}")

    return X_train, X_test, y_train, y_test, project_root


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train a model and evaluate on test set.

    Args:
        model: sklearn-compatible model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name for printing

    Returns:
        tuple: (trained_model, metrics_dict)
    """
    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"✅ Training completed in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
        "train_time": train_time,
    }

    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        if metric != "train_time":
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value:.2f}s")

    return model, metrics


def save_models(model_smote, model_weighted, metrics_smote, metrics_weighted, model_prefix, project_root):
    """
    Save trained models and metrics to disk.

    Args:
        model_smote: SMOTE-trained model
        model_weighted: Class weight-trained model
        metrics_smote: Metrics for SMOTE model
        metrics_weighted: Metrics for weighted model
        model_prefix: Prefix for filenames (e.g., 'lr', 'rf', 'xgb')
        project_root: Project root path
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    smote_path = models_dir / f"{model_prefix}_smote.pkl"
    weighted_path = models_dir / f"{model_prefix}_weighted.pkl"

    joblib.dump(model_smote, smote_path)
    joblib.dump(model_weighted, weighted_path)

    print(f"✅ Saved: {smote_path}")
    print(f"✅ Saved: {weighted_path}")

    # Save metrics to metrics subfolder
    metrics = {f"{model_prefix.upper()}_SMOTE": metrics_smote, f"{model_prefix.upper()}_Weighted": metrics_weighted}
    metrics_path = metrics_dir / f"{model_prefix}_metrics.pkl"
    joblib.dump(metrics, metrics_path)

    print(f"✅ Saved metrics: {metrics_path}")


def log_to_mlflow(model, metrics, run_name, model_type, strategy, hyperparams, X_train, X_test, y_train, mlflow_logger=None):
    """
    Log model and metrics to MLflow.

    Args:
        model: Trained model
        metrics: Metrics dictionary
        run_name: Name for the MLflow run
        model_type: Type of model (e.g., 'LogisticRegression')
        strategy: Training strategy ('SMOTE' or 'Class_Weight')
        hyperparams: Dictionary of model hyperparameters
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        mlflow_logger: MLflow logging function (mlflow.sklearn, mlflow.xgboost, etc.)
    """
    print(f"Logging {run_name} to MLflow...")

    with mlflow.start_run(run_name=run_name):
        # Log common parameters
        common_params = {
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "class_imbalance_ratio": float(np.sum(y_train == 0) / np.sum(y_train == 1)) if strategy != "SMOTE" else 1.0,
            "strategy": strategy,
            "model_type": model_type,
        }
        mlflow.log_params(common_params)

        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "train_time_seconds": metrics["train_time"],
            }
        )

        # Log model with descriptive name
        # Convert run_name to artifact path: "LR_SMOTE" -> "lr_smote"
        artifact_path = run_name.lower().replace(" ", "_").replace("-", "_")
        if mlflow_logger:
            mlflow_logger.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

        # Set tags
        model_family = run_name.split("_")[0]
        mlflow.set_tags({"model_family": model_family, "strategy": strategy})

        print(f"  ✅ Run ID: {mlflow.active_run().info.run_id}")


def print_summary(metrics_smote, metrics_weighted, model_name):
    """
    Print training summary comparing both strategies.

    Args:
        metrics_smote: Metrics for SMOTE model
        metrics_weighted: Metrics for weighted model
        model_name: Name of model (e.g., 'Logistic Regression')
    """
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} TRAINING COMPLETE")
    print("=" * 80)

    print("\nSMOTE Strategy:")
    print(f"  PR-AUC: {metrics_smote['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics_smote['f1']:.4f}")
    print(f"  Recall: {metrics_smote['recall']:.4f}")

    print("\nClass Weight Strategy:")
    print(f"  PR-AUC: {metrics_weighted['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics_weighted['f1']:.4f}")
    print(f"  Recall: {metrics_weighted['recall']:.4f}")

    better = "SMOTE" if metrics_smote["pr_auc"] > metrics_weighted["pr_auc"] else "Class Weight"
    print(f"\n✅ Better strategy for {model_name}: {better}")
    print("=" * 80)
