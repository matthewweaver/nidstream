#!/usr/bin/env bash
# Terminate the EMR cluster. Stops all billing immediately.
# The cluster also auto-terminates after 2 hours of idle time.
#
# Run: bash scripts/emr/04_terminate_cluster.sh

set -euo pipefail

REGION="eu-west-1"
CLUSTER_ID=$(cat .emr-cluster-id 2>/dev/null || echo "")

if [[ -z "$CLUSTER_ID" ]]; then
    echo "ERROR: .emr-cluster-id not found."
    echo "Get the cluster ID from AWS console and run:"
    echo "  aws --profile nidstream emr terminate-clusters --cluster-ids <ID> --region $REGION"
    exit 1
fi

echo "==> Terminating EMR cluster: $CLUSTER_ID"
aws --no-cli-pager emr terminate-clusters \
    --cluster-ids "$CLUSTER_ID" \
    --region "$REGION"

echo "✓ Termination initiated. Billing stops within a few minutes."
echo ""

# Show final state
aws --no-cli-pager emr describe-cluster \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --query 'Cluster.{State:Status.State,Reason:Status.StateChangeReason.Message}' \
    --output table

rm -f .emr-cluster-id .emr-master-dns
