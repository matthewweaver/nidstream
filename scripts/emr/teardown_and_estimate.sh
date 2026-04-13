#!/usr/bin/env bash
# Shut down all running nidstream AWS resources and estimate remaining monthly costs.
# Safe to run at any time — read-only checks before any terminations.
#
# Usage: bash scripts/emr/teardown_and_estimate.sh

set -uo pipefail

REGION="eu-west-1"
BUCKET="nidstream"
PROFILE="nidstream"
AWS="aws --profile $PROFILE --no-cli-pager"

# Prices (eu-west-1, on-demand, USD, as of 2026)
PRICE_M5_XLARGE=0.222      # per hour
PRICE_S3_GB=0.023           # per GB per month
PRICE_EBS_GP3=0.088         # per GB per month (if any EBS volumes orphaned)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "========================================"
echo "  NIDStream AWS Teardown & Cost Report"
echo "========================================"
echo ""

# ── 1. EMR Clusters ──────────────────────────────────────────────────────────
echo -e "${CYAN}[1/4] EMR Clusters${NC}"

ACTIVE_CLUSTERS=$($AWS emr list-clusters \
    --region "$REGION" \
    --cluster-states STARTING BOOTSTRAPPING RUNNING WAITING \
    --query 'Clusters[*].[Id,Name,Status.State]' \
    --output text 2>/dev/null || echo "")

if [[ -z "$ACTIVE_CLUSTERS" ]]; then
    echo -e "  ${GREEN}✓ No active EMR clusters${NC}"
else
    echo -e "  ${RED}⚠  Active clusters found — terminating:${NC}"
    while IFS=$'\t' read -r cluster_id cluster_name state; do
        echo "     $cluster_id  ($cluster_name)  [$state]"
        $AWS emr terminate-clusters --cluster-ids "$cluster_id" --region "$REGION"
        echo -e "     ${GREEN}✓ Termination requested${NC}"
    done <<< "$ACTIVE_CLUSTERS"
fi

# ── 2. EC2 Instances (transfer + EMR residual) ───────────────────────────────
echo ""
echo -e "${CYAN}[2/4] EC2 Instances${NC}"

RUNNING_EC2=$($AWS ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running,pending" \
               "Name=tag:Project,Values=nidstream" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
    --output text 2>/dev/null || echo "")

if [[ -z "$RUNNING_EC2" ]]; then
    echo -e "  ${GREEN}✓ No running EC2 instances tagged Project=nidstream${NC}"
else
    echo -e "  ${RED}⚠  Running instances found — terminating:${NC}"
    while IFS=$'\t' read -r instance_id instance_type name; do
        echo "     $instance_id  $instance_type  ($name)"
        $AWS ec2 terminate-instances --instance-ids "$instance_id" --region "$REGION" > /dev/null
        echo -e "     ${GREEN}✓ Termination requested${NC}"
    done <<< "$RUNNING_EC2"
fi

# ── 3. Orphaned EBS Volumes ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3/4] Orphaned EBS Volumes${NC}"

ORPHAN_VOLS=$($AWS ec2 describe-volumes \
    --region "$REGION" \
    --filters "Name=status,Values=available" \
    --query 'Volumes[*].[VolumeId,Size,VolumeType]' \
    --output text 2>/dev/null || echo "")

if [[ -z "$ORPHAN_VOLS" ]]; then
    echo -e "  ${GREEN}✓ No orphaned EBS volumes${NC}"
else
    echo -e "  ${YELLOW}⚠  Unattached EBS volumes (not auto-deleted — review manually):${NC}"
    TOTAL_EBS_GB=0
    while IFS=$'\t' read -r vol_id size vol_type; do
        echo "     $vol_id  ${size}GB  $vol_type"
        TOTAL_EBS_GB=$((TOTAL_EBS_GB + size))
    done <<< "$ORPHAN_VOLS"
    EBS_COST=$(echo "$TOTAL_EBS_GB * $PRICE_EBS_GP3" | bc -l 2>/dev/null || echo "?")
    echo -e "     ${YELLOW}Monthly cost: ~\$$(printf '%.2f' $EBS_COST) (${TOTAL_EBS_GB}GB @ \$${PRICE_EBS_GP3}/GB)${NC}"
    echo "     To delete: aws --profile $PROFILE ec2 delete-volume --volume-id <id> --region $REGION"
fi

# ── 4. S3 Storage Cost Estimate ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/4] S3 Storage — s3://${BUCKET}${NC}"

# Get size of each top-level prefix (single pass — no associative arrays for bash 3 compat)
TOTAL_BYTES=0

echo ""
echo "  Folder breakdown:"
for prefix in "data/raw" "data/processed" "models" "mlflow" "emr"; do
    bytes=$($AWS s3 ls "s3://${BUCKET}/${prefix}/" --recursive 2>/dev/null \
        | awk '{s+=$3} END {print s+0}')
    bytes=${bytes:-0}
    TOTAL_BYTES=$((TOTAL_BYTES + bytes))
    if [[ $bytes -gt 1073741824 ]]; then
        human=$(echo "scale=2; $bytes/1073741824" | bc)
        unit="GB"
    elif [[ $bytes -gt 1048576 ]]; then
        human=$(echo "scale=2; $bytes/1048576" | bc)
        unit="MB"
    else
        human=$(echo "scale=0; $bytes/1024" | bc)
        unit="KB"
    fi
    printf "    %-22s %8s %s\n" "s3://$BUCKET/$prefix" "$human" "$unit"
done

TOTAL_GB=$(echo "scale=4; $TOTAL_BYTES/1073741824" | bc)
MONTHLY_S3=$(echo "scale=4; $TOTAL_GB * $PRICE_S3_GB" | bc)

echo ""
echo "  ──────────────────────────────────────"
printf "  Total S3 size : %8.2f GB\n"  "$TOTAL_GB"
printf "  Monthly cost  : \$%7.4f  (\$%.3f/GB × %.2f GB)\n" \
    "$MONTHLY_S3" "$PRICE_S3_GB" "$TOTAL_GB"
echo "  ──────────────────────────────────────"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Ongoing monthly cost estimate"
echo "========================================"
printf "  S3 storage         : \$%.4f\n" "$MONTHLY_S3"
if [[ -n "$ORPHAN_VOLS" ]]; then
    printf "  EBS volumes        : \$%.2f  ← delete these!\n" "${EBS_COST:-0}"
fi
echo "  EMR / EC2          : \$0.00  (pay only when running)"
echo ""
echo -e "  ${CYAN}Estimated total: ~\$$(printf '%.2f' $MONTHLY_S3)/month at current S3 usage${NC}"
echo ""
echo "  Note: S3 costs scale linearly — uploading the full"
echo "  100GB BCCC dataset will cost ~\$$(echo "100 * $PRICE_S3_GB" | bc)/month."
echo "  EMR cluster (1 master + 2 core m5.xlarge) costs"
echo "  ~\$$(echo "scale=2; ($PRICE_M5_XLARGE * 3)" | bc)/hr while running."
echo "========================================"
echo ""
