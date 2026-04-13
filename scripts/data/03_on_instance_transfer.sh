#!/usr/bin/env bash
# Run this ON the EC2 transfer instance (not locally).
# Downloads the Kaggle dataset file-by-file and uploads each to S3.
# This avoids needing full 100GB free at once — each file is deleted after upload.
#
# Usage:
#   bash 03_on_instance_transfer.sh <KAGGLE_USERNAME> <KAGGLE_KEY>
#
# Find your Kaggle credentials at: https://www.kaggle.com/settings → API → Create New Token

set -euo pipefail

KAGGLE_USERNAME="${1:?Usage: $0 <KAGGLE_USERNAME> <KAGGLE_KEY>}"
KAGGLE_KEY="${2:?Usage: $0 <KAGGLE_USERNAME> <KAGGLE_KEY>}"
DATASET="bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018"
S3_BUCKET="nidstream"
S3_PREFIX="data/raw/BCCC-CSE-CIC-IDS2018"
REGION="eu-west-1"
DOWNLOAD_DIR="$HOME/kaggle_download"

echo "==> Setting up environment..."
sudo dnf install -y python3-pip unzip > /dev/null 2>&1
pip3 install --quiet kaggle

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

echo "==> Listing dataset files..."
FILE_LIST=$(kaggle datasets files "$DATASET" --csv | tail -n +2 | cut -d',' -f1)
TOTAL=$(echo "$FILE_LIST" | wc -l)
echo "    Found $TOTAL files to transfer"
echo ""

COUNT=0
while IFS= read -r filename; do
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] Downloading: $filename"

    kaggle datasets download "$DATASET" \
        --file "$filename" \
        --path "$DOWNLOAD_DIR" \
        --quiet

    # Handle zip files — extract then upload contents, or upload zip as-is
    DOWNLOADED_FILE="$DOWNLOAD_DIR/$filename"
    if [[ ! -f "$DOWNLOADED_FILE" ]]; then
        # kaggle CLI may have added .zip extension
        DOWNLOADED_FILE="${DOWNLOADED_FILE}.zip"
    fi

    echo "    Uploading to s3://${S3_BUCKET}/${S3_PREFIX}/${filename}..."
    aws s3 cp "$DOWNLOADED_FILE" \
        "s3://${S3_BUCKET}/${S3_PREFIX}/${filename}" \
        --region "$REGION" \
        --no-progress

    rm -f "$DOWNLOADED_FILE"
    echo "    ✓ Done (local copy deleted)"
    echo ""

done <<< "$FILE_LIST"

echo "================================================"
echo "✓ All $TOTAL files transferred to:"
echo "  s3://${S3_BUCKET}/${S3_PREFIX}/"
echo ""
echo "Verify with:"
echo "  aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --human-readable"
echo ""
echo "You can now terminate this instance."

# Wipe credentials from disk
rm -f "$HOME/.kaggle/kaggle.json"
