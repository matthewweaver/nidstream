"""
Training Pipeline - Model Evaluation Module

Evaluates trained models on test set with comprehensive metrics.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import mlflow


def load_model(model_path: str):
    """Load trained model and artifacts."""
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['scaler'], artifacts['feature_cols']


def evaluate_model(model_path: str, test_path: str = "data/processed/test_features.parquet"):
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to saved model
        test_path: Path to test data
    """
    print("Loading model and test data...")
    model, scaler, feature_cols = load_model(model_path)
    test_df = pd.read_parquet(test_path)
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_attack']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print("\nGenerating predictions...")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        # For Isolation Forest
        y_pred_scores = model.decision_function(X_test_scaled)
        y_pred = (y_pred_scores < 0).astype(int)  # -1 = outlier (attack), 1 = inlier (benign)
        y_pred_proba = -y_pred_scores  # Convert to anomaly scores
    
    # Calculate metrics
    print("\n=== Evaluation Results ===\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    print(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    
    # ROC-AUC and PR-AUC
    if hasattr(model, 'predict_proba'):
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
    
    # Performance by attack type
    print("\n=== Performance by Attack Type ===")
    test_df['predictions'] = y_pred
    test_df['pred_proba'] = y_pred_proba
    
    attack_performance = test_df.groupby('attack_type').agg({
        'is_attack': 'count',
        'predictions': lambda x: (x == test_df.loc[x.index, 'is_attack']).mean()
    }).rename(columns={'is_attack': 'count', 'predictions': 'accuracy'})
    
    print(attack_performance)
    
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'test_df': test_df
    }


def evaluate_with_mlflow(model_path: str, experiment_name: str = "nidstream-evaluation"):
    """Evaluate model and log results to MLflow."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="test_evaluation"):
        results = evaluate_model(model_path)
        
        # Log metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        mlflow.log_metric("accuracy", accuracy_score(results['y_test'], results['y_pred']))
        mlflow.log_metric("precision", precision_score(results['y_test'], results['y_pred']))
        mlflow.log_metric("recall", recall_score(results['y_test'], results['y_pred']))
        mlflow.log_metric("f1", f1_score(results['y_test'], results['y_pred']))
        
        if 'y_pred_proba' in results:
            mlflow.log_metric("roc_auc", roc_auc_score(results['y_test'], results['y_pred_proba']))
            mlflow.log_metric("pr_auc", average_precision_score(results['y_test'], results['y_pred_proba']))
        
        print(f"\n✓ Evaluation logged to MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/xgboost_baseline.pkl"
    evaluate_with_mlflow(model_path)
