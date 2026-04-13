#!/usr/bin/env bash
# One-time setup: create the IAM roles EMR needs to launch clusters.
# Run once per AWS account: bash scripts/emr/01_create_emr_roles.sh

set -euo pipefail

echo "==> Creating EMR default service roles..."

# This creates: EMR_DefaultRole, EMR_EC2_DefaultRole, EMR_AutoScaling_DefaultRole
aws --no-cli-pager emr create-default-roles --region eu-west-1 2>&1 || true

# Attach SSM policy so the EC2 instances can register with Systems Manager.
# This allows port-forwarding via 'aws ssm start-session' without any open
# inbound security-group rules (no SSH port 22 exposure needed).
echo "==> Attaching AmazonSSMManagedInstanceCore to EMR_EC2_DefaultRole..."
aws --no-cli-pager iam attach-role-policy \
    --role-name EMR_EC2_DefaultRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>&1 || true

# Grant the calling IAM user permission to use SSM port-forwarding.
# This is required for 03_ssh_tunnel.sh — no SSH port (22) exposure needed.
CALLER_ARN=$(aws --no-cli-pager sts get-caller-identity --query 'Arn' --output text)
CALLER_TYPE=$(echo "$CALLER_ARN" | cut -d: -f6 | cut -d/ -f1)   # user or assumed-role
CALLER_NAME=$(echo "$CALLER_ARN" | cut -d/ -f2)

echo "==> Granting SSMPortForwarding policy to $CALLER_TYPE/$CALLER_NAME..."
if [[ "$CALLER_TYPE" == "user" ]]; then
    aws --no-cli-pager iam put-user-policy \
        --user-name "$CALLER_NAME" \
        --policy-name SSMPortForwarding \
        --policy-document '{
          "Version": "2012-10-17",
          "Statement": [
            {
              "Sid": "SSMPortForwardingSessions",
              "Effect": "Allow",
              "Action": [
                "ssm:StartSession",
                "ssm:TerminateSession",
                "ssm:DescribeSessions",
                "ssm:GetConnectionStatus"
              ],
              "Resource": [
                "arn:aws:ec2:*:*:instance/*",
                "arn:aws:ssm:*:*:document/AWS-StartPortForwardingSession",
                "arn:aws:ssm:*:*:document/AWS-StartPortForwardingSessionToRemoteHost",
                "arn:aws:ssm:*:*:session/*"
              ]
            },
            {
              "Sid": "SSMDescribeInstances",
              "Effect": "Allow",
              "Action": [
                "ssm:DescribeInstanceInformation"
              ],
              "Resource": "*"
            }
          ]
        }' 2>&1 || true
    echo "    ✓ SSMPortForwarding inline policy applied to user/$CALLER_NAME"
else
    echo "    ⚠ Caller is $CALLER_TYPE — attach the SSMPortForwarding policy manually to the role."
    echo "      Required actions: ssm:StartSession, ssm:TerminateSession, ssm:DescribeSessions, ssm:GetConnectionStatus"
fi

# Verify
echo ""
echo "==> Verifying roles..."
for role in EMR_DefaultRole EMR_EC2_DefaultRole; do
    if aws --no-cli-pager iam get-role --role-name "$role" &>/dev/null; then
        echo "    ✓ $role"
    else
        echo "    ✗ $role (missing — check IAM permissions)"
    fi
done

echo ""
echo "✓ EMR roles ready."
echo "Next: bash scripts/emr/02_launch_cluster.sh"
