#!/bin/bash
# Setup script for nidstream project
# This ensures dependencies are installed from PyPI, not corporate registries

set -e

echo "🚀 Setting up nidstream project..."

# Temporarily override corporate PyPI index URLs for this installation only
# These unset commands only affect this script's execution
echo "📦 Installing dependencies from PyPI (ignoring corporate indexes)..."

# Run uv sync with environment overrides
PIP_EXTRA_INDEX_URL="" \
UV_EXTRA_INDEX_URL="" \
PIP_INDEX_URL="https://pypi.org/simple" \
UV_INDEX_URL="https://pypi.org/simple" \
uv sync --all-extras

echo "✅ Setup complete!"
echo ""
echo "To run the API:"
echo "  uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Note: Your global environment variables are unchanged."
