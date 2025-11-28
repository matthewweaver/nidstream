#!/bin/bash
# Start MLflow Tracking Server

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting MLflow Tracking Server...${NC}"

# Set tracking URI to local directory
export MLFLOW_TRACKING_URI="file:./mlruns"

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow UI
echo -e "${GREEN}MLflow UI will be available at: http://localhost:5000${NC}"
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
echo ""

mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns
