<div align="center">
  <img src=".github/assets/nidstream_logo.png" alt="NIDStream Logo" width="200"/>
  <h1>NIDStream - Network Intrusion Detection ML Pipeline</h1>
  <p>
    <em>An end-to-end machine learning system for detecting network intrusions</em>
  </p>
</div>

## Project Overview

Traditional Intrusion Detection Systems (IDS) rely on known signatures. This project shifts to **predictive anomaly detection** - identifying subtle, multi-dimensional patterns in network flows that indicate malicious activity (Botnet, DoS, DDoS, Infiltration, etc.) before they match known signatures.

### Key Features

- 🌊 **Large-scale data processing** with PySpark on S3 data lake
- 🔬 **MLflow tracking** for experiment management and model versioning
- 🐳 **Containerized deployment** with Docker on AWS ECS/Fargate
- ⚖️ **Load balancing** with AWS Application Load Balancer
- 📊 **Real-time dashboard** with Streamlit for anomaly visualization
- 🚀 **CI/CD pipeline** with GitHub Actions
- 🧪 **Comprehensive testing** with pytest

## Architecture

The pipeline follows: `Load (S3) → Preprocess (Spark) → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Spark-based data loading, preprocessing, feature engineering from S3
  - `load.py`: Load BCCC-CSE-CIC-IDS2018 CSVs from S3 using PySpark
  - `preprocess.py`: Clean network flow data, handle missing values, normalize
  - `feature_engineering.py`: Extract temporal patterns, flow statistics, protocol features
  
- **`src/training_pipeline/`**: Model training with hyperparameter tuning
  - `train.py`: Train anomaly detection models (Isolation Forest, XGBoost, Autoencoder)
  - `tune.py`: Optuna-based hyperparameter optimization with MLflow
  - `eval.py`: Evaluate on test set with precision, recall, F1, AUC-ROC
  
- **`src/inference_pipeline/`**: Production inference
  - `inference.py`: Real-time anomaly scoring on new network flows
  
- **`src/batch/`**: Batch prediction processing
  - `run_batch.py`: Process network logs in batches
  
- **`src/api/`**: FastAPI REST service
  - `main.py`: API endpoints for health checks, predictions, batch processing

### Web Applications

- **`app.py`**: Streamlit dashboard for real-time anomaly feed visualization
  - Interactive time-series plots of anomaly scores
  - Filter by attack type, protocol, source/destination
  - Display model predictions vs ground truth labels

### Cloud Infrastructure

- **AWS S3**: Data lake for BCCC-CSE-CIC-IDS2018 CSVs, processed features, trained models
- **Amazon ECR**: Container registry for Docker images
- **Amazon ECS/Fargate**: Serverless container orchestration
- **Application Load Balancer**: Traffic distribution between API and dashboard
- **GitHub Actions**: Automated testing and deployment

## Dataset: BCCC-CSE-CIC-IDS2018

The [BCCC-CSE-CIC-IDS2018](https://www.kaggle.com/datasets/bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018) is an updated version of the Canadian Institute for Cybersecurity dataset containing network traffic captures with labeled attacks:
- **Benign traffic**: Normal network activity
- **Attack types**: Botnet, Brute Force, DoS, DDoS, Infiltration, Web Attacks, etc.
- **Features**: 300+ flow-based features (packet stats, protocol info, timing patterns)
- **Source**: Available on [Kaggle](https://www.kaggle.com/datasets/bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018?resource=download)

## Getting Started

### Prerequisites

- Python 3.11+
- UV for Python package management
- AWS account with S3 access
- Docker (for containerization)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd nidstream

# One-command setup (installs all dependencies from PyPI)
./setup.sh
```

**Note**: The `setup.sh` script ensures dependencies are installed from PyPI only, ignoring any corporate package indexes you may have configured globally.

### Environment Setup

Create `.env` file with AWS credentials:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-nids-data-bucket
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Common Commands

### Data Pipeline

```bash
# Load data from S3 using Spark
uv run python src/feature_pipeline/load.py

# Preprocess network flows
uv run python -m src.feature_pipeline.preprocess

# Feature engineering
uv run python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline

```bash
# Train baseline anomaly detection model
uv run python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow tracking
uv run python src/training_pipeline/tune.py

# Evaluate model performance
uv run python src/training_pipeline/eval.py
```

### MLflow Tracking

```bash
# Start MLflow UI
uv run mlflow ui --host 0.0.0.0 --port 5000
```

Visit `http://localhost:5000` to view experiments and model metrics.

### API Service

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Generate a sample flow from test data
python scripts/csv_to_json.py 0

# Make prediction with real data
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_flow.json

# Pretty-print the response
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_flow.json | python -m json.tool
```

### Streamlit Dashboard

```bash
# Start dashboard locally
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Visit `http://localhost:8501` to interact with the dashboard.

### Docker

```bash
# Build API container
docker build -t nidstream-api .

# Build Streamlit container
docker build -t nidstream-dashboard -f Dockerfile.streamlit .

# Run containers
docker run -p 8000:8000 --env-file .env nidstream-api
docker run -p 8501:8501 --env-file .env nidstream-dashboard
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_features.py
uv run pytest tests/test_training.py

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## Project Structure

```
nidstream/
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # GitHub Actions CI/CD
├── configs/
│   ├── model_config.yaml       # Model hyperparameters
│   └── pipeline_config.yaml    # Pipeline settings
├── data/
│   ├── raw/                    # Raw CSVs from S3 (gitignored)
│   ├── processed/              # Preprocessed features (gitignored)
│   └── predictions/            # Batch predictions (gitignored)
├── models/
│   └── *.pkl                   # Trained models (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── feature_pipeline/
│   ├── training_pipeline/
│   ├── inference_pipeline/
│   ├── batch/
│   └── api/
├── tests/
│   ├── test_features.py
│   ├── test_training.py
│   └── test_inference.py
├── app.py                      # Streamlit dashboard
├── Dockerfile                  # API container
├── Dockerfile.streamlit        # Dashboard container
├── nids-api-task-def.json     # ECS task definition (API)
├── nids-dashboard-task-def.json # ECS task definition (Dashboard)
├── pyproject.toml              # UV dependencies
└── README.md
```

## Key Design Patterns

### Spark Integration
- All data loading and preprocessing use PySpark for scalability
- Handles large BCCC-CSE-CIC-IDS2018 dataset distributed across S3
- Spark sessions configured for optimal memory usage

### Data Leakage Prevention
- Time-based train/validation/test splits (not random)
- Encoders and scalers fitted only on training data
- Strict feature selection to avoid target leakage

### Model Versioning
- MLflow tracks all experiments with metrics and parameters
- Models registered with version control
- Easy rollback to previous model versions

### Cloud-Native Design
- S3-first storage for data and models
- Containerized services for portability
- Auto-scaling with ECS Fargate
- Environment-based configuration

## Attack Types Detected

1. **Botnet**: Command & control traffic patterns
2. **DoS/DDoS**: Denial of service attacks
3. **Brute Force**: Authentication attacks
4. **Infiltration**: Network penetration attempts
5. **Web Attacks**: SQL injection, XSS, etc.

## Performance Metrics

- **Precision**: Minimize false positives (benign traffic flagged as attack)
- **Recall**: Maximize true positives (catch all attacks)
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Overall classifier performance
- **Anomaly Score Distribution**: Visualize separation between benign and malicious

## Contributing

This is a portfolio project showcasing end-to-end ML engineering skills. Feel free to fork and adapt for your own use cases.

## License

MIT License

## Acknowledgments

- Canadian Institute for Cybersecurity and BCCC for the BCCC-CSE-CIC-IDS2018 dataset
- Dataset available at: https://www.kaggle.com/datasets/bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018
