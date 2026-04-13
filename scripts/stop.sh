#!/usr/bin/env bash
# Stop all nidstream services: docker-compose, SSM tunnels, and EMR cluster.
#
# Usage:
#   bash scripts/stop.sh          # stop docker services + tunnels + terminate EMR
#   bash scripts/stop.sh --local  # stop docker services + tunnels only (no EMR)
#
# Run from repo root: bash scripts/stop.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REGION="eu-west-1"
CLUSTER_ID_FILE="$REPO_ROOT/.emr-cluster-id"

TERMINATE_EMR=true
if [[ "${1:-}" == "--local" ]]; then
    TERMINATE_EMR=false
fi

cd "$REPO_ROOT"

# ── 1. Docker services ────────────────────────────────────────────────────────
echo "==> Stopping docker-compose services..."
if docker compose ps -q 2>/dev/null | grep -q .; then
    docker compose down
    echo "    Done."
else
    echo "    No running containers found."
fi

# ── 2. SSM tunnels ────────────────────────────────────────────────────────────
echo ""
echo "==> Stopping SSM port-forwarding tunnels..."

KILLED=0
for PORT in 8998 4040 8088; do
    PIDS=$(lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    for PID in $PIDS; do
        CMD=$(ps -p "$PID" -o comm= 2>/dev/null || true)
        if [[ "$CMD" == *"session-manager"* ]] || [[ "$CMD" == *"aws"* ]]; then
            kill "$PID" 2>/dev/null \
                && echo "    Stopped tunnel on :$PORT (pid $PID)" \
                && KILLED=$((KILLED + 1))
        fi
    done
done

SSM_PIDS=$(pgrep -f "ssm start-session.*nidstream" 2>/dev/null || true)
for PID in $SSM_PIDS; do
    kill "$PID" 2>/dev/null \
        && echo "    Stopped SSM session (pid $PID)" \
        && KILLED=$((KILLED + 1))
done

[[ $KILLED -eq 0 ]] && echo "    No active tunnels found."

# ── 3. EMR cluster ────────────────────────────────────────────────────────────
if [[ "$TERMINATE_EMR" == false ]]; then
    echo ""
    echo "Skipping EMR termination (--local mode)."
    echo "Done."
    exit 0
fi

echo ""
echo "==> Terminating EMR cluster..."

if [[ ! -f "$CLUSTER_ID_FILE" ]]; then
    # No stored ID — check if anything is running by name
    CLUSTER_ID=$(aws --no-cli-pager emr list-clusters \
        --region "$REGION" \
        --active \
        --query "Clusters[?Name=='nidstream-spark'] | [0].Id" \
        --output text 2>/dev/null || echo "")
    if [[ -z "$CLUSTER_ID" || "$CLUSTER_ID" == "None" ]]; then
        echo "    No active EMR cluster found."
        echo ""
        echo "Done."
        exit 0
    fi
    echo "$CLUSTER_ID" > "$CLUSTER_ID_FILE"
fi

bash "$SCRIPT_DIR/emr/04_terminate_cluster.sh"

echo ""
echo "Done. All services stopped."
