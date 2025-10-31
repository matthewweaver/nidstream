"""
FastAPI Service for Network Intrusion Detection

Provides REST API endpoints for:
- Health checks
- Real-time prediction on network flows
- Batch prediction processing
"""

import os
import pickle
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="NIDStream API",
    description="Network Intrusion Detection System - Anomaly Detection API",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
MODEL = None
SCALER = None
FEATURE_COLS = None


class NetworkFlow(BaseModel):
    """Single network flow for prediction."""
    features: Dict[str, float]


class BatchRequest(BaseModel):
    """Batch of network flows for prediction."""
    flows: List[Dict[str, float]]


def load_model(model_path: str = "models/xgboost_baseline.pkl"):
    """Load model artifacts on startup."""
    global MODEL, SCALER, FEATURE_COLS
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    MODEL = artifacts['model']
    SCALER = artifacts['scaler']
    FEATURE_COLS = artifacts['feature_cols']
    
    print(f"✓ Model loaded from {model_path}")


@app.on_event("startup")
async def startup_event():
    """Load model on API startup."""
    model_path = os.getenv("MODEL_PATH", "models/xgboost_baseline.pkl")
    load_model(model_path)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NIDStream API - Network Intrusion Detection System",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "features": len(FEATURE_COLS) if FEATURE_COLS else 0
    }


@app.post("/predict")
async def predict(flow: NetworkFlow):
    """
    Predict anomaly score for a single network flow.
    
    Returns:
        - is_attack: Binary prediction (0=Benign, 1=Attack)
        - anomaly_score: Continuous anomaly score [0,1]
        - confidence: Model confidence
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([flow.features])
        
        # Ensure all required features are present
        missing_features = set(FEATURE_COLS) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing_features)}"
            )
        
        # Select and order features
        df = df[FEATURE_COLS]
        
        # Scale features
        df_scaled = SCALER.transform(df)
        
        # Predict
        if hasattr(MODEL, 'predict_proba'):
            proba = MODEL.predict_proba(df_scaled)[0, 1]
            prediction = int(proba >= 0.5)
        else:
            # Isolation Forest
            score = MODEL.decision_function(df_scaled)[0]
            prediction = int(score < 0)
            proba = float(-score)  # Convert to anomaly score
        
        return {
            "is_attack": prediction,
            "anomaly_score": float(proba),
            "confidence": float(abs(proba - 0.5) * 2),  # Distance from decision boundary
            "prediction": "ATTACK" if prediction == 1 else "BENIGN"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """
    Predict anomaly scores for batch of network flows.
    
    Optimized for throughput on larger batches.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.flows)
        
        # Ensure all required features are present
        df = df[FEATURE_COLS]
        
        # Scale features
        df_scaled = SCALER.transform(df)
        
        # Predict
        if hasattr(MODEL, 'predict_proba'):
            probas = MODEL.predict_proba(df_scaled)[:, 1]
            predictions = (probas >= 0.5).astype(int)
        else:
            scores = MODEL.decision_function(df_scaled)
            predictions = (scores < 0).astype(int)
            probas = -scores
        
        return {
            "predictions": predictions.tolist(),
            "anomaly_scores": probas.tolist(),
            "summary": {
                "total": len(predictions),
                "attacks_detected": int(predictions.sum()),
                "benign": int((predictions == 0).sum())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
