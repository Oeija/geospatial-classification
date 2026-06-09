#!/usr/bin/env bash
set -euo pipefail

# run-docker.sh — helper script to build and run the geospatial-classification
# pipeline with GPU support.
#
# Requires: Docker + Docker Compose + nvidia-docker runtime.
#
# Override PyTorch variant via environment variables:
#   ./run-docker.sh                         # default = stable PyTorch CUDA 12.8
#   PYTORCH_INDEX=https://download.pytorch.org/whl/cpu USE_PRE=0 ./run-docker.sh
#   PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 USE_PRE=1 ./run-docker.sh
#
# USE_PRE=1 is required for nightly indices so pip accepts pre-release wheels.
# Only use it with the nightly index (e.g. RTX 5060 / Blackwell).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# PyTorch variant selection
# ------------------------------------------------------------------------------
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
USE_PRE="${USE_PRE:-0}"

if [ "$USE_PRE" = "1" ]; then
    VARIANT_NAME="PyTorch PRE-RELEASE (nightly) CUDA 12.8"
else
    VARIANT_NAME="PyTorch STABLE CUDA 12.8"
fi

echo "============================================"
echo " Geospatial Classification — Docker Runner"
echo "============================================"
echo ""
echo "PyTorch variant: $VARIANT_NAME"
echo "  Index:  $PYTORCH_INDEX"
echo "  USE_PRE: $USE_PRE"

# ------------------------------------------------------------------------------
# Step 1: Build the image
# ------------------------------------------------------------------------------
echo ""
echo "[1/4] Building Docker image..."
docker-compose build \
    --build-arg PYTORCH_INDEX="$PYTORCH_INDEX" \
    --build-arg USE_PRE="$USE_PRE"

# ------------------------------------------------------------------------------
# Step 2: Train CNN backbone
# ------------------------------------------------------------------------------
echo ""
echo "[2/4] Training CNN backbone..."
docker-compose run --rm train-cnn

# ------------------------------------------------------------------------------
# Step 3: Train CNN-ViT hybrid
# ------------------------------------------------------------------------------
echo ""
echo "[3/4] Training CNN-ViT hybrid..."
docker-compose run --rm train-hybrid

# ------------------------------------------------------------------------------
# Step 4: Evaluate
# ------------------------------------------------------------------------------
echo ""
echo "[4/4] Evaluating model..."
docker-compose run --rm evaluate

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Checkpoints: ./checkpoints/"
echo " Logs:        ./logs/"
echo " Plots:       ./plots/"
echo "============================================"
