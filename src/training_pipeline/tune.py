"""
Training Pipeline - Hyperparameter Tuning Module

Uses Optuna for hyperparameter optimization with MLflow tracking.
"""

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna
import mlflow
import mlflow.xgboost


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_pr = average_precision_score(y_val, y_pred_proba)
    
    return auc_pr


def tune_xgboost(
    n_trials: int = 50,
    experiment_name: str = "nidstream-tuning"
):
    """
    Run hyperparameter tuning with Optuna and MLflow.
    
    Args:
        n_trials: Number of Optuna trials
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    
    feature_cols = [c for c in train_df.columns if c not in ['Label', 'attack_type', 'is_attack']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_attack']
    X_val = val_df[feature_cols]
    y_val = val_df['is_attack']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
    
    print(f"\nStarting hyperparameter tuning with {n_trials} trials...")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Log best params to MLflow
    with mlflow.start_run(run_name="best_params"):
        best_params = study.best_params
        best_score = study.best_value
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_auc_pr", best_score)
        
        print(f"\n✓ Best AUC-PR: {best_score:.4f}")
        print(f"✓ Best params: {best_params}")
        
        # Train final model with best params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42,
            'n_jobs': -1
        })
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=True)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        print(f"\n✓ Tuning complete! MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    tune_xgboost(n_trials=n_trials)
