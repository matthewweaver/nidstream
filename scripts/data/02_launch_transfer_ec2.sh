#!/usr/bin/env bash
# Launch an EC2 instance sized for Kaggle → S3 transfer (~100GB dataset).
# Uses a t3.xlarge (4 vCPU, 16GB RAM) with 220 GB EBS gp3 — terminated manually when done.
# Run from project root with: bash scripts/data/02_launch_transfer_ec2.sh

set -euo pipefail

REGION="eu-west-1"
KEY_NAME="nidstream-emr"
PROFILE_NAME="nidstream-ec2-transfer"
SG_NAME="nidstream-transfer-ssh"
INSTANCE_TYPE="t3.xlarge"   # ~$0.20/hr — fast enough for download + upload
VOLUME_SIZE=220              # GB — dataset is ~100GB, leave headroom for extraction

# Latest Amazon Linux 2023 AMI (x86_64)
AMI_ID=$(aws --no-cli-pager ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=al2023-ami-2023*-x86_64" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

VPC_ID=$(aws --no-cli-pager ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

SUBNET_ID=$(aws --no-cli-pager ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)

SG_ID=$(aws --no-cli-pager ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text)

echo "==> Launching EC2 instance..."
echo "    AMI          : $AMI_ID (Amazon Linux 2023)"
echo "    Instance type: $INSTANCE_TYPE"
echo "    EBS volume   : ${VOLUME_SIZE} GB gp3"
echo "    Subnet       : $SUBNET_ID"

INSTANCE_ID=$(aws --no-cli-pager ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    --block-device-mappings "[{
        \"DeviceName\": \"/dev/xvda\",
        \"Ebs\": {
            \"VolumeSize\": ${VOLUME_SIZE},
            \"VolumeType\": \"gp3\",
            \"Throughput\": 250,
            \"DeleteOnTermination\": true
        }
    }]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nidstream-transfer},{Key=Project,Value=nidstream}]" \
    --associate-public-ip-address \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Waiting for instance to start (takes ~60s)..."
aws --no-cli-pager ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION"

PUBLIC_IP=$(aws --no-cli-pager ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# Save instance ID for the terminate script
echo "$INSTANCE_ID" > .transfer-instance-id

echo ""
echo "✓ Instance running."
echo "  Instance ID : $INSTANCE_ID"
echo "  Public IP   : $PUBLIC_IP"
echo "  Key         : ~/.ssh/${KEY_NAME}.pem"
echo ""
echo "SSH in (wait ~30s for sshd to start):"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "Then copy and run the transfer script:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem scripts/data/03_on_instance_transfer.sh ec2-user@${PUBLIC_IP}:~/"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo "  # On instance: bash 03_on_instance_transfer.sh <KAGGLE_USERNAME> <KAGGLE_KEY>"
echo ""
echo "When done, terminate:"
echo "  aws --profile nidstream ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
