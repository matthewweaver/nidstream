#!/usr/bin/env bash
# EMR bootstrap: install Python packages on every node.
# Runs on master + all core nodes before Spark/Livy start.
set -euo pipefail

BUCKET="nidstream"
REQUIREMENTS_S3="s3://${BUCKET}/emr/requirements.txt"

echo "==> Downloading requirements..."
aws s3 cp "$REQUIREMENTS_S3" /tmp/requirements.txt

echo "==> Installing packages..."
sudo pip3 install --upgrade --ignore-installed -r /tmp/requirements.txt

echo "==> Verifying..."
python3 -c "import numpy, pandas, sklearn, mlflow; print('OK - all packages installed')"
