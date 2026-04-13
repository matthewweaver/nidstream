#!/usr/bin/env bash
# One-time setup: IAM role + instance profile, key pair, security group
# Run from project root with: bash scripts/data/01_setup_transfer_infra.sh

set -euo pipefail

REGION="eu-west-1"
BUCKET="nidstream"
KEY_NAME="nidstream-emr"
ROLE_NAME="nidstream-ec2-transfer"
PROFILE_NAME="nidstream-ec2-transfer"
SG_NAME="nidstream-transfer-ssh"

echo "==> Creating EC2 key pair..."
mkdir -p "$HOME/.ssh"
if aws --no-cli-pager ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "    Key pair '$KEY_NAME' already exists, skipping."
else
    aws --no-cli-pager ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "$HOME/.ssh/${KEY_NAME}.pem"
    chmod 400 "$HOME/.ssh/${KEY_NAME}.pem"
    echo "    Saved to ~/.ssh/${KEY_NAME}.pem"
fi

echo "==> Creating IAM role for EC2..."
if aws --no-cli-pager iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "    Role '$ROLE_NAME' already exists, skipping."
else
    aws --no-cli-pager iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' > /dev/null

    aws --no-cli-pager iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "S3NidstreamWrite" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": [\"s3:PutObject\",\"s3:GetObject\",\"s3:ListBucket\",\"s3:DeleteObject\"],
                \"Resource\": [
                    \"arn:aws:s3:::${BUCKET}\",
                    \"arn:aws:s3:::${BUCKET}/*\"
                ]
            }]
        }"
    echo "    Role created with S3 write access to s3://$BUCKET"
fi

echo "==> Creating instance profile..."
if aws --no-cli-pager iam get-instance-profile --instance-profile-name "$PROFILE_NAME" &>/dev/null; then
    echo "    Instance profile '$PROFILE_NAME' already exists, skipping."
else
    aws --no-cli-pager iam create-instance-profile \
        --instance-profile-name "$PROFILE_NAME" > /dev/null
    aws --no-cli-pager iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME"
    echo "    Waiting 10s for IAM propagation..."
    sleep 10
fi

echo "==> Creating security group (SSH only from your IP)..."
MY_IP=$(curl -s https://checkip.amazonaws.com)
VPC_ID=$(aws --no-cli-pager ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

EXISTING_SG=$(aws --no-cli-pager ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "$EXISTING_SG" == "None" || "$EXISTING_SG" == "" ]]; then
    SG_ID=$(aws --no-cli-pager ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "SSH access for nidstream data transfer" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' --output text)

    aws --no-cli-pager ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 \
        --cidr "${MY_IP}/32" \
        --region "$REGION"
    echo "    Security group '$SG_NAME' ($SG_ID) created, SSH allowed from $MY_IP"
else
    SG_ID="$EXISTING_SG"
    echo "    Security group '$SG_NAME' ($SG_ID) already exists."
fi

echo ""
echo "✓ Infrastructure setup complete."
echo "  Key pair : ~/.ssh/${KEY_NAME}.pem"
echo "  IAM role : $ROLE_NAME"
echo "  Sec group: $SG_ID (VPC: $VPC_ID)"
echo ""
echo "Next: bash scripts/data/02_launch_transfer_ec2.sh"
