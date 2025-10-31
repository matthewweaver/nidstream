"""
Training Pipeline - Model Training Module

Trains anomaly detection models on CSE-CIC-IDS2018 network flow data.
Supports multiple algorithms: Isolation Forest, XGBoost, Autoencoder.
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost


def load_data_for_training(train_path: str, val_path: str) -> tuple:
    """
    Load training and validation data from Parquet.
    Converts Spark output to pandas for sklearn/xgboost.
    """
    # For large datasets, consider using Dask or sampling
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Separate features and labels
    feature_cols = [c for c in train_df.columns if c not in ['Label', 'attack_type', 'is_attack']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_attack']
    
    X_val = val_df[feature_cols]
    y_val = val_df['is_attack']
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    return X_train, y_train, X_val, y_val, feature_cols


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42
) -> IsolationForest:
    """
    Train Isolation Forest for unsupervised anomaly detection.
    
    Isolation Forest is effective for detecting anomalies by isolating outliers
    in feature space. Works well for high-dimensional network flow data.
    """
    print("\nTraining Isolation Forest...")
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train)
    
    print("✓ Isolation Forest trained")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier for supervised anomaly detection.
    
    XGBoost handles class imbalance well and provides feature importance.
    """
    print("\nTraining XGBoost...")
    
    if params is None:
        # Default params optimized for imbalanced data
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # AUC-PR better for imbalanced data
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    print("✓ XGBoost trained")
    return model


def save_model(model, model_path: str, scaler=None, feature_cols=None):
    """Save trained model and associated artifacts."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    
    print(f"✓ Model saved to {model_path}")


def train_with_mlflow(
    model_type: str = "xgboost",
    experiment_name: str = "nidstream-training"
):
    """
    Train model with MLflow tracking.
    
    Args:
        model_type: 'isolation_forest' or 'xgboost'
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_type}_baseline"):
        # Load data
        X_train, y_train, X_val, y_val, feature_cols = load_data_for_training(
            "data/processed/train_features.parquet",
            "data/processed/val_features.parquet"
        )
        
        # Scale features (important for many ML algorithms)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert back to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        
        # Train model
        if model_type == "isolation_forest":
            model = train_isolation_forest(X_train_scaled)
            mlflow.sklearn.log_model(model, "model")
        
        elif model_type == "xgboost":
            model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            mlflow.xgboost.log_model(model, "model")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Save model artifacts
        model_path = f"models/{model_type}_baseline.pkl"
        save_model(model, model_path, scaler, feature_cols)
        mlflow.log_artifact(model_path)
        
        print(f"\n✓ Training complete! MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import sys
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    train_with_mlflow(model_type=model_type)
