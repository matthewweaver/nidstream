"""
FastAPI Service for Network Intrusion Detection

Provides REST API endpoints for:
- Health checks
- Single flow predictions
- Batch predictions
- Model information
"""

import io
import logging

# Import your inference module
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))

from inference_pipeline.predict import IntrusionDetector, load_detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NIDStream - Network Intrusion Detection API", description="ML-powered network intrusion detection service", version="1.0.0"
)

# Global detector instance
detector: Optional[IntrusionDetector] = None


# Pydantic models for request/response
class FlowFeatures(BaseModel):
    """Network flow features for prediction"""

    # Add your actual feature names here based on your model
    # Example features from CIC-IDS2018:
    flow_duration: float = Field(..., description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(..., description="Total packets in forward direction")
    total_backward_packets: int = Field(..., description="Total packets in backward direction")
    total_length_fwd_packets: float = Field(..., description="Total size of packet in forward direction")
    total_length_bwd_packets: float = Field(..., description="Total size of packet in backward direction")
    fwd_packet_length_max: float = Field(0, description="Maximum size of packet in forward direction")
    fwd_packet_length_min: float = Field(0, description="Minimum size of packet in forward direction")
    fwd_packet_length_mean: float = Field(0, description="Mean size of packet in forward direction")
    bwd_packet_length_max: float = Field(0, description="Maximum size of packet in backward direction")
    bwd_packet_length_min: float = Field(0, description="Minimum size of packet in backward direction")
    bwd_packet_length_mean: float = Field(0, description="Mean size of packet in backward direction")
    flow_bytes_s: float = Field(0, description="Flow bytes per second")
    flow_packets_s: float = Field(0, description="Flow packets per second")

    class Config:
        extra = "allow"  # Allow additional fields


class PredictionResponse(BaseModel):
    """Response for single prediction"""

    is_attack: bool = Field(..., description="True if attack detected, False if benign")
    confidence: float = Field(..., description="Model confidence (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp")

    class Config:
        extra = "forbid"  # Only return defined fields


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""

    total_flows: int
    attacks_detected: int
    attack_percentage: float
    predictions: List[Dict]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""

    model_type: str
    model_name: Optional[str] = None
    strategy: Optional[str] = None
    training_date: Optional[str] = None
    metrics: Optional[Dict] = None
    n_features: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global detector
    try:
        model_dir = Path("models")
        logger.info(f"Loading model from {model_dir}")
        detector = load_detector(str(model_dir))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "NIDStream - Network Intrusion Detection API",
        "service": "NIDStream - Network Intrusion Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {"health": "/health", "predict": "/predict", "predict_batch": "/predict/batch", "model_info": "/model/info"},
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_loaded = detector is not None

    return {"status": "healthy" if model_loaded else "degraded", "model_loaded": model_loaded, "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single_flow(features: Dict = Body(..., description="Network flow features as JSON")):
    """
    Predict whether a single network flow is an attack.

    Args:
        features: Network flow features as a dictionary

    Returns:
        Prediction result with is_attack and confidence
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # features is already a dict
        flow_dict = features

        # Make prediction
        result = detector.predict_single(flow_dict)

        # Return only the simplified response
        return PredictionResponse(
            is_attack=bool(result["prediction"] == 1), confidence=result["confidence"], timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(..., description="CSV file with network flows")):
    """
    Predict multiple network flows from CSV file.

    Args:
        file: CSV file containing network flow data

    Returns:
        Batch prediction results
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        logger.info(f"Processing batch of {len(df)} flows")

        # Make predictions
        results = detector.predict_batch(df)

        # Calculate statistics
        total_flows = len(results)
        attacks_detected = (results["prediction"] == 1).sum()
        attack_percentage = (attacks_detected / total_flows * 100) if total_flows > 0 else 0

        # Convert to list of dicts
        predictions = results[["prediction", "prediction_label", "attack_probability", "confidence"]].to_dict("records")

        return {
            "total_flows": total_flows,
            "attacks_detected": int(attacks_detected),
            "attack_percentage": round(attack_percentage, 2),
            "predictions": predictions[:100],  # Limit to first 100 for response size
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        info = detector.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model (useful after retraining)"""
    global detector

    try:
        model_dir = Path("models")
        logger.info(f"Reloading model from {model_dir}")
        detector = load_detector(str(model_dir))
        logger.info("Model reloaded successfully")

        return {"status": "success", "message": "Model reloaded successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
