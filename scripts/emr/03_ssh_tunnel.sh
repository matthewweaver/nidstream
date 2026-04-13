#!/usr/bin/env bash
# Forward ports from the EMR master to localhost using AWS SSM Session Manager.
# No SSH port (22) needs to be open in the security group — auth is via IAM.
#
# Prerequisites (one-time):
#   1. bash scripts/emr/01_create_emr_roles.sh   (attaches AmazonSSMManagedInstanceCore)
#   2. brew install --cask session-manager-plugin  (or the AWS installer)
#      https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
#
# After running this, in VS Code:
#   1. Open a notebook
#   2. Click the kernel name (top-right of notebook)
#   3. Select "Select Another Kernel..." → "Existing Jupyter Server..."
#   4. Enter: http://localhost:8888  (press Enter twice, no token needed)
#   5. Choose the "PySpark" kernel
#
# Run: bash scripts/emr/03_ssh_tunnel.sh

set -euo pipefail

REGION="eu-west-1"
CLUSTER_ID=$(cat .emr-cluster-id 2>/dev/null || echo "")

if [[ -z "$CLUSTER_ID" ]]; then
    echo "ERROR: .emr-cluster-id not found. Run 02_launch_cluster.sh first."
    exit 1
fi

# Verify session-manager-plugin is installed
if ! command -v session-manager-plugin &>/dev/null; then
    echo "ERROR: session-manager-plugin not found."
    echo "       Install it: brew install --cask session-manager-plugin"
    echo "       Or: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html"
    exit 1
fi

echo "==> Resolving EMR master instance for cluster $CLUSTER_ID..."
read INSTANCE_ID MASTER_DNS < <(aws emr list-instances \
    --cluster-id "$CLUSTER_ID" \
    --instance-group-types MASTER \
    --query 'Instances[0].[Ec2InstanceId,PrivateDnsName]' \
    --output text \
    --region "$REGION" \
    --no-cli-pager)

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
    echo "ERROR: Could not resolve master instance ID. Is the cluster running?"
    exit 1
fi

echo "    Instance : $INSTANCE_ID"
echo "    Master DNS: $MASTER_DNS"

# Wait for the SSM agent to be Online before opening tunnels.
# If the cluster was launched before 01_create_emr_roles.sh was run, the SSM
# agent will not have registered yet. It retries automatically every ~5 minutes,
# so allow up to 10 minutes here.
echo "==> Waiting for SSM agent on $INSTANCE_ID..."
SSM_TIMEOUT=600
SSM_ELAPSED=0
until aws --no-cli-pager ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text \
        --region "$REGION" 2>/dev/null | grep -q 'Online'; do
    if (( SSM_ELAPSED >= SSM_TIMEOUT )); then
        echo ""
        echo "ERROR: SSM agent not reachable after ${SSM_TIMEOUT}s."
        echo ""
        echo "  Most likely cause: the cluster was launched before IAM policies were set up."
        echo "  The SSM agent retries registration every ~5 minutes — if it has been less"
        echo "  than 10 minutes since running 01_create_emr_roles.sh, wait and retry:"
        echo ""
        echo "    bash scripts/emr/03_ssh_tunnel.sh"
        echo ""
        echo "  If it still fails, terminate and re-launch the cluster so it picks up the"
        echo "  correct IAM permissions from the start:"
        echo ""
        echo "    bash scripts/emr/04_terminate_cluster.sh"
        echo "    bash scripts/emr/02_launch_cluster.sh"
        exit 1
    fi
    sleep 15
    SSM_ELAPSED=$(( SSM_ELAPSED + 15 ))
    echo "    ... ${SSM_ELAPSED}s — waiting for SSM agent (retries every ~5 min on first launch)"
done
echo "    ✓ SSM agent online"

echo "==> Clearing any existing Livy sessions..."
curl -s http://127.0.0.1:8998/sessions 2>/dev/null | python3 -c "
import sys, json, urllib.request
try:
    sessions = json.load(sys.stdin).get('sessions', [])
    for s in sessions:
        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:8998/sessions/{s[\"id\"]}', method='DELETE'))
        print(f'    Deleted session {s[\"id\"]} (state={s[\"state\"]})')
    if not sessions:
        print('    No sessions to clear')
except Exception as e:
    print(f'    (skipped: {e})')
" 2>/dev/null || true

echo "    Forwarding localhost:8998  → master:8998             (Livy / PySpark kernel — loopback)"
echo "    Forwarding localhost:5001  → ${MASTER_DNS}:5001   (MLflow tracking server — remote-host)"
echo "    Forwarding localhost:8088  → ${MASTER_DNS}:8088   (YARN Resource Manager UI — remote-host)"
echo "    Forwarding localhost:20888 → ${MASTER_DNS}:20888  (YARN proxy → live Spark UI — remote-host)"
echo "    Forwarding localhost:18080 → ${MASTER_DNS}:18080  (Spark History Server — remote-host)"
echo ""
echo "    Note: YARN/Spark UIs bind to the master's internal hostname, not loopback,"
echo "          so they require AWS-StartPortForwardingSessionToRemoteHost"
echo "          (granted by scripts/emr/01_create_emr_roles.sh)."
echo ""
echo "    Press Ctrl+C to stop all tunnels (cluster keeps running)."
echo ""
echo "    In VS Code (once tunnels are open):"
echo "      1. Open a notebook"
echo "      2. Click the kernel name in the top-right corner of the notebook"
echo "      3. Select 'Select Another Kernel...'"
echo "      4. Choose the 'pysparkkernel' (local sparkmagic kernel)"
echo "         — sparkmagic connects to Livy on localhost:8998 automatically"
echo ""
echo "    YARN UI         : http://localhost:8088"
echo "    Live Spark UI   : http://localhost:8088 → click app → 'ApplicationMaster'"
echo "                      (or http://localhost:20888/proxy/<application_id>/ directly)"
echo "    Spark History   : http://localhost:18080  (after the app finishes)"
echo ""

# Background loop: reap dead/excess Livy sessions every 30s to prevent
# YARN AM memory exhaustion from failed kernel startups accumulating.
livy_cleanup_loop() {
    while true; do
        sleep 30
        python3 -c "
import urllib.request, json
try:
    sessions = json.loads(urllib.request.urlopen('http://127.0.0.1:8998/sessions').read()).get('sessions', [])
    # Delete any session in a terminal/failed state
    dead = [s for s in sessions if s['state'] in ('dead', 'killed', 'error', 'shutting_down')]
    for s in dead:
        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:8998/sessions/{s[\"id\"]}', method='DELETE'))
        print(f'[livy-cleanup] Removed {s[\"state\"]} session {s[\"id\"]}', flush=True)
    # If more than 1 live session, keep only the most recent (highest id)
    live = sorted([s for s in sessions if s['state'] not in ('dead', 'killed', 'error', 'shutting_down')], key=lambda s: s['id'])
    for s in live[:-1]:
        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:8998/sessions/{s[\"id\"]}', method='DELETE'))
        print(f'[livy-cleanup] Removed stale session {s[\"id\"]} (kept {live[-1][\"id\"]})', flush=True)
except Exception:
    pass
" 2>/dev/null
    done
}

# Kill all background SSM sessions on Ctrl+C / exit
PIDS=()
cleanup() {
    echo ""
    echo "==> Stopping SSM tunnels..."
    kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

livy_cleanup_loop &
PIDS+=($!)

# Livy and MLflow server both bind to 0.0.0.0 — loopback document works for both.
aws ssm start-session \
    --target "$INSTANCE_ID" \
    --document-name AWS-StartPortForwardingSession \
    --parameters '{"portNumber":["8998"],"localPortNumber":["8998"]}' \
    --region "$REGION" &
PIDS+=($!)

# MLflow tracking server (started from the notebook setup cell on the EMR master).
# Use RemoteHost document — more reliable than loopback when the service starts
# after the SSM session is established.
aws ssm start-session \
    --target "$INSTANCE_ID" \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters "{\"host\":[\"${MASTER_DNS}\"],\"portNumber\":[\"5001\"],\"localPortNumber\":[\"5001\"]}" \
    --region "$REGION" &
PIDS+=($!)

# YARN/Spark UIs bind to the master's internal hostname, not loopback.
# Use the remote-host document so the SSM agent on master proxies to ${MASTER_DNS}:port.
for port in 8088 20888 18080; do
    aws ssm start-session \
        --target "$INSTANCE_ID" \
        --document-name AWS-StartPortForwardingSessionToRemoteHost \
        --parameters "{\"host\":[\"${MASTER_DNS}\"],\"portNumber\":[\"${port}\"],\"localPortNumber\":[\"${port}\"]}" \
        --region "$REGION" &
    PIDS+=($!)
done

# Block until Ctrl+C
wait "${PIDS[@]}"
