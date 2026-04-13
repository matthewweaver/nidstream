#!/usr/bin/env bash
# Launch an on-demand EMR cluster with Spark + Jupyter Enterprise Gateway.
# VS Code connects to the kernel gateway over an SSH tunnel (see 03_ssh_tunnel.sh).
#
# Costs (eu-west-1, on-demand):
#   1x m5.xlarge master  = ~$0.222/hr
#   4x m5.xlarge core    = ~$0.888/hr  (use spot to cut ~70%)
#   Total                = ~$1.11/hr
#
# Run: bash scripts/emr/02_launch_cluster.sh

set -euo pipefail

REGION="eu-west-1"
BUCKET="nidstream"
KEY_NAME="nidstream-emr"
CLUSTER_NAME="nidstream-spark"
EMR_RELEASE="emr-7.5.0"
LOG_URI="s3://${BUCKET}/emr/logs/"
# Instance sizing — adjust CORE_COUNT to scale up for bigger jobs.
# 4 core nodes give 16 vCPU / ~96 GB executor RAM, which is enough headroom for
# LR on the 5 GB BCCC processed dataset (333 features × ~1M rows, dense vectors).
MASTER_TYPE="m5.xlarge"
CORE_TYPE="m5.xlarge"
CORE_COUNT=4

echo "==> Uploading bootstrap files to S3..."
aws --no-cli-pager s3 cp "$(dirname "$0")/bootstrap.sh" "s3://${BUCKET}/emr/bootstrap.sh"
aws --no-cli-pager s3 cp "$(dirname "$0")/requirements.txt" "s3://${BUCKET}/emr/requirements.txt"

echo "==> Getting default VPC subnet..."
VPC_ID=$(aws --no-cli-pager ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws --no-cli-pager ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)

echo "==> Launching EMR cluster '$CLUSTER_NAME'..."
CLUSTER_ID=$(aws --no-cli-pager emr create-cluster \
    --region "$REGION" \
    --name "$CLUSTER_NAME" \
    --release-label "$EMR_RELEASE" \
    --applications Name=Spark Name=JupyterEnterpriseGateway \
    --ec2-attributes "{
        \"KeyName\": \"${KEY_NAME}\",
        \"SubnetId\": \"${SUBNET_ID}\",
        \"InstanceProfile\": \"EMR_EC2_DefaultRole\"
    }" \
    --instance-groups "[
        {
            \"Name\": \"Master\",
            \"InstanceGroupType\": \"MASTER\",
            \"InstanceType\": \"${MASTER_TYPE}\",
            \"InstanceCount\": 1
        },
        {
            \"Name\": \"Core\",
            \"InstanceGroupType\": \"CORE\",
            \"InstanceType\": \"${CORE_TYPE}\",
            \"InstanceCount\": ${CORE_COUNT}
        }
    ]" \
    --bootstrap-actions "Path=s3://${BUCKET}/emr/bootstrap.sh,Name=InstallPythonPackages" \
    --log-uri "$LOG_URI" \
    --service-role EMR_DefaultRole \
    --auto-termination-policy '{"IdleTimeout": 7200}' \
    --configurations "[
        {
            \"Classification\": \"spark-env\",
            \"Configurations\": [{
                \"Classification\": \"export\",
                \"Properties\": {
                    \"PYSPARK_PYTHON\": \"/usr/bin/python3\"
                }
            }]
        },
        {
            \"Classification\": \"spark\",
            \"Properties\": {
                \"maximizeResourceAllocation\": \"true\"
            }
        },
        {
            \"Classification\": \"spark-defaults\",
            \"Properties\": {
                \"spark.serializer\": \"org.apache.spark.serializer.KryoSerializer\",
                \"spark.kryoserializer.buffer.max\": \"512m\",
                \"spark.sql.execution.arrow.pyspark.enabled\": \"false\"
            }
        },
        {
            \"Classification\": \"livy-conf\",
            \"Properties\": {
                \"livy.server.session.timeout\": \"8h\",
                \"livy.server.session.timeout-check\": \"true\"
            }
        }
    ]" \
    --query 'ClusterId' \
    --output text)

echo "$CLUSTER_ID" > .emr-cluster-id
echo ""
echo "==> Cluster launching: $CLUSTER_ID"
echo "    This takes 5-8 minutes. Waiting for WAITING state..."

aws --no-cli-pager emr wait cluster-running \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION"

# Resolve the master EC2 instance ID
MASTER_INSTANCE_ID=$(aws --no-cli-pager emr list-instances \
    --cluster-id "$CLUSTER_ID" \
    --instance-group-types MASTER \
    --query 'Instances[0].Ec2InstanceId' \
    --output text \
    --region "$REGION")

echo "    Master instance : $MASTER_INSTANCE_ID"
echo "    Waiting for SSM agent to register (may take ~2 minutes)..."

# Poll until the SSM agent on the master shows as Online
SSM_TIMEOUT=180  # seconds
SSM_ELAPSED=0
until aws --no-cli-pager ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$MASTER_INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text \
        --region "$REGION" 2>/dev/null | grep -q 'Online'; do
    if (( SSM_ELAPSED >= SSM_TIMEOUT )); then
        echo ""
        echo "    ⚠ SSM agent did not register within ${SSM_TIMEOUT}s."
        echo "      The cluster is running — retry: bash scripts/emr/03_ssh_tunnel.sh"
        break
    fi
    sleep 10
    SSM_ELAPSED=$(( SSM_ELAPSED + 10 ))
    echo "    ... ${SSM_ELAPSED}s elapsed"
done

echo ""
echo "✓ Cluster ready."
echo "  Cluster ID : $CLUSTER_ID"
echo "  Instance   : $MASTER_INSTANCE_ID"
echo ""
echo "Next — open SSH tunnel and connect VS Code:"
echo "  bash scripts/emr/03_ssh_tunnel.sh"
