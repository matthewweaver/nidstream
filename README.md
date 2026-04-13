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

## Architecture

The pipeline follows: `Load (S3) → Preprocess (Spark) → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Spark-based data loading, preprocessing, feature engineering from S3
  - `load.py`: Load BCCC-CSE-CIC-IDS2018 CSVs from S3 using PySpark
  - `preprocess.py`: Clean network flow data, handle missing values, normalize
  - `feature_engineering.py`: Extract temporal patterns, flow statistics, protocol features
  - `temporal_features.py`: **NEW** - Sliding window features for time-series attack detection
  
- **`src/training_pipeline/`**: Model training with hyperparameter tuning
  - `train.py`: Train anomaly detection models (Isolation Forest, XGBoost, Autoencoder)
  - `train_temporal.py`: **NEW** - Train models with temporal/sequential features
  - `tune.py`: Optuna-based hyperparameter optimization with MLflow
  - `eval.py`: Evaluate on test set with precision, recall, F1, AUC-ROC
  
- **`src/inference_pipeline/`**: Production inference
  - `predict.py`: Real-time anomaly scoring on new network flows
  - `predict_temporal.py`: **NEW** - Stateful inference with temporal context
  
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

- Python 3.11+ and [UV](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- AWS account with an `nidstream` profile configured in `~/.aws/credentials`
- For AWS training: AWS CLI + [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)

### Installation

```bash
git clone <repo-url>
cd nidstream
uv sync
```

The `.env` file is pre-configured for local development. No changes needed to get started.

---

## Running Locally

All local services (MLflow, FastAPI, Streamlit) run via Docker Compose:

```bash
bash scripts/start.sh
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

```bash
# Stop everything
bash scripts/stop.sh --local
```

MLflow experiment data is stored in `./mlruns/` and persists across restarts.

---

## Running on AWS (EMR Training)

Training notebooks run on an EMR cluster for large-scale PySpark jobs. Local services still run via Docker Compose, with MLflow backed by S3 so experiment history is shared between EMR and local runs.

### One-time setup

```bash
# Create IAM roles needed for EMR + SSM access
bash scripts/emr/01_create_emr_roles.sh
```

### Start everything

```bash
bash scripts/start.sh --aws
```

This will:
1. Start local services (MLflow pointed at `s3://nidstream/mlflow`, FastAPI, Streamlit)
2. Sync `notebooks/training_utils.py` to S3
3. Find a running `nidstream-spark` cluster or launch a new one (~5-8 min first time)
4. Open SSM port-forwarding tunnels — press `Ctrl+C` to close tunnels (services keep running)

| Tunnel | URL |
|--------|-----|
| Livy / PySpark kernel | localhost:8998 |
| Spark UI | http://localhost:4040 |
| YARN Resource Manager | http://localhost:8088 |

### Connect VS Code notebooks to EMR

1. Open a notebook in VS Code
2. Click the kernel picker (top-right)
3. Select `pysparkkernel` — Spark Magics connects to Livy on `localhost:8998` automatically

### Stop everything (including EMR)

```bash
bash scripts/stop.sh
```

This stops Docker services, closes SSM tunnels, and terminates the EMR cluster to stop billing.

> The cluster also auto-terminates after 2 hours idle. Running cost: ~$0.67/hr (1 master + 2 core m5.xlarge).

---

## Training Notebooks

Notebooks live in `notebooks/BCCC-CSE-CIC-IDS2018/`. Each handles both local (local Spark) and EMR (existing session) automatically via environment detection.

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02a_feature_engineering.ipynb` | Feature engineering (pandas) |
| `02b_feature_engineering_pyspark.ipynb` | Feature engineering (PySpark) |
| `03a_train_logistic_regression_pyspark.ipynb` | Logistic Regression |
| `03b_train_random_forest.ipynb` | Random Forest |
| `03c_train_xgboost.ipynb` | XGBoost |
| `04_model_comparison.ipynb` | Compare all trained models |

All PySpark training notebooks log metrics and parameters to MLflow automatically.

---

## Temporal Streaming (Real-time Simulation)

### From the Streamlit Dashboard (recommended)

1. In the sidebar under **Temporal Streaming**, upload `X_test_temporal.csv`
2. Set stream speed (0.5x–10x) and max flows (0 = continuous loop)
3. Click **Start Stream** — predictions appear live with attack detection alerts

### From the command line

```bash
uv run python scripts/stream_temporal.py              # 1x speed
uv run python scripts/stream_temporal.py --speed 2.0  # 2x speed
uv run python scripts/stream_temporal.py --speed 5.0 --max-flows 100
```

---

## Testing

```bash
uv run pytest
uv run pytest --cov=src --cov-report=html
```

## Project Structure

```
nidstream/
├── .github/
│   └── workflows/ci.yml            # GitHub Actions CI
├── configs/
│   ├── model_config.yaml           # Model hyperparameters
│   └── pipeline_config.yaml        # Pipeline settings
├── data/
│   ├── raw/                        # Raw CSVs (gitignored)
│   └── processed/                  # Parquet features (gitignored)
├── models/                         # Trained models (gitignored)
│   └── pyspark/                    # PySpark ML models saved here
├── notebooks/
│   └── BCCC-CSE-CIC-IDS2018/       # Training notebooks (local + EMR)
│       ├── training_utils.py       # Shared utilities (synced to S3 for EMR)
│       └── 01_eda … 04_*.ipynb
├── scripts/
│   ├── start.sh                    # Start all services (local or --aws)
│   ├── stop.sh                     # Stop all services (--local skips EMR)
│   └── emr/
│       ├── 01_create_emr_roles.sh  # One-time IAM setup
│       ├── 02_launch_cluster.sh    # Launch EMR cluster
│       ├── 03_ssh_tunnel.sh        # Open SSM port-forwarding tunnels
│       └── 04_terminate_cluster.sh # Terminate cluster
├── src/
│   ├── api/main.py                 # FastAPI inference service
│   ├── inference_pipeline/         # Model loading and prediction
│   └── schemas/                    # Pydantic request/response models
├── app.py                          # Streamlit dashboard
├── Dockerfile                      # FastAPI container
├── Dockerfile.streamlit            # Streamlit container
├── Dockerfile.mlflow               # MLflow container (with S3 support)
├── docker-compose.yml              # Local services stack
├── pyproject.toml                  # Dependencies (UV)
└── .env                            # Local environment config
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
