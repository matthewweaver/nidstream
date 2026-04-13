#!/usr/bin/env bash
# Start all nidstream services.
#
# Usage:
#   bash scripts/start.sh          # local mode — docker-compose only
#   bash scripts/start.sh --aws    # AWS mode   — docker-compose + EMR tunnel
#
# Local mode starts:  MLflow (:5000), FastAPI (:8000), Streamlit (:8501)
# AWS mode adds:      SSM tunnels to running EMR cluster (or launches one)
#
# Prerequisites (one-time for AWS): bash scripts/emr/01_create_emr_roles.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EMR_DIR="$SCRIPT_DIR/emr"
REGION="eu-west-1"
BUCKET="nidstream"
CLUSTER_NAME="nidstream-spark"
CLUSTER_ID_FILE="$REPO_ROOT/.emr-cluster-id"

MODE="local"
if [[ "${1:-}" == "--aws" ]]; then
    MODE="aws"
fi

# ── URL opener ───────────────────────────────────────────────────────────────
# Cross-platform browser opener.  Backgrounded with a delay so services have
# time to come up first (docker healthchecks for local services; SSM tunnel
# handshake for AWS UIs).  Set NO_OPEN=1 to disable (e.g. on a CI box).
open_urls() {
    [[ "${NO_OPEN:-0}" == "1" ]] && return 0
    local opener
    if command -v open &>/dev/null;       then opener="open"
    elif command -v xdg-open &>/dev/null; then opener="xdg-open"
    else
        echo "    (skipping browser open — no 'open'/'xdg-open' on PATH; set NO_OPEN=1 to silence)"
        return 0
    fi
    for url in "$@"; do
        "$opener" "$url" >/dev/null 2>&1 || true
    done
}

# Schedule URL opens in the background after a delay (in seconds) so the
# user doesn't get a flood of browser tabs landing on connection-refused pages.
schedule_open_urls() {
    local delay="$1"; shift
    ( sleep "$delay" && open_urls "$@" ) &
    disown 2>/dev/null || true
}

# ── Docker services ───────────────────────────────────────────────────────────
echo "==> Starting docker-compose services (mode=$MODE)..."

if [[ "$MODE" == "aws" ]]; then
    # Backend store must be a DB/filesystem (S3 is unsupported for metadata/model registry).
    # Keep metadata local; point artifacts at S3 so EMR runs and local UI share the same store.
    export MLFLOW_BACKEND_STORE="sqlite:///mlruns/mlflow.db"
    export MLFLOW_ARTIFACT_ROOT="s3://${BUCKET}/mlflow/artifacts"
fi

cd "$REPO_ROOT"
docker compose up -d --build

echo ""
echo "    MLflow    : http://localhost:5000"
echo "    API       : http://localhost:8000"
echo "    Streamlit : http://localhost:8501"

# Open the docker-service UIs once their healthchecks have a chance to pass.
# (start_period in docker-compose.yml is 5-15s; 8s is a reasonable middle ground.)
schedule_open_urls 8 \
    "http://localhost:5000" \
    "http://localhost:8000/docs" \
    "http://localhost:8501"

if [[ "$MODE" == "local" ]]; then
    echo ""
    echo "All local services started. Run 'bash scripts/stop.sh' to stop."
    exit 0
fi

# ── AWS mode: EMR + tunnels ───────────────────────────────────────────────────
echo ""
echo "==> Syncing notebooks/training_utils.py to S3..."
aws --no-cli-pager s3 cp "$REPO_ROOT/notebooks/training_utils.py" "s3://${BUCKET}/emr/training_utils.py"

echo ""
CLUSTER_ID=""

# Check stored cluster ID first
if [[ -f "$CLUSTER_ID_FILE" ]]; then
    STORED_ID=$(cat "$CLUSTER_ID_FILE")
    STATE=$(aws --no-cli-pager emr describe-cluster \
        --cluster-id "$STORED_ID" \
        --region "$REGION" \
        --query 'Cluster.Status.State' \
        --output text 2>/dev/null || echo "UNKNOWN")
    if [[ "$STATE" == "WAITING" || "$STATE" == "RUNNING" ]]; then
        CLUSTER_ID="$STORED_ID"
        echo "==> Reusing cluster from .emr-cluster-id: $CLUSTER_ID (state=$STATE)"
    else
        echo "==> Stored cluster $STORED_ID is not active (state=$STATE), checking by name..."
    fi
fi

# Fall back: search for active cluster by name
if [[ -z "$CLUSTER_ID" ]]; then
    CLUSTER_ID=$(aws --no-cli-pager emr list-clusters \
        --region "$REGION" \
        --active \
        --query "Clusters[?Name=='$CLUSTER_NAME'] | [0].Id" \
        --output text 2>/dev/null || echo "")
    if [[ -n "$CLUSTER_ID" && "$CLUSTER_ID" != "None" ]]; then
        echo "==> Found running cluster '$CLUSTER_NAME': $CLUSTER_ID"
        echo "$CLUSTER_ID" > "$CLUSTER_ID_FILE"
    else
        CLUSTER_ID=""
    fi
fi

# Launch a new cluster if none found
if [[ -z "$CLUSTER_ID" ]]; then
    echo "==> No running cluster found — launching a new one (5-8 min)..."
    bash "$EMR_DIR/02_launch_cluster.sh"
else
    echo "==> Skipping cluster launch."
fi

# Open SSM tunnels — blocks until Ctrl+C; docker services keep running.
# Schedule the EMR UIs to open after the tunnels have had time to handshake.
# 03_ssh_tunnel.sh waits for the SSM agent (up to ~10 min on first launch),
# but on a reused cluster the tunnels are usually live within ~10s.
schedule_open_urls 25 \
    "http://localhost:8088" \
    "http://localhost:18080"

echo ""
echo "==> Opening SSM tunnels (Ctrl+C stops tunnels but leaves docker services up)..."
bash "$EMR_DIR/03_ssh_tunnel.sh"
