#!/usr/bin/env bash
set -euo pipefail

# docker-test.sh — run pytest inside the GPU-enabled Docker container.
#
# Override PyTorch variant via environment variables (same semantics as run-docker.sh):
#   PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 USE_PRE=1 ./docker-test.sh

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

echo "Running tests inside Docker container..."
echo "  Variant: $VARIANT_NAME"
echo "  Index:   $PYTORCH_INDEX"
echo "  USE_PRE: $USE_PRE"
echo ""

# Build if needed
docker-compose build \
    --build-arg PYTORCH_INDEX="$PYTORCH_INDEX" \
    --build-arg USE_PRE="$USE_PRE"

# Run tests
docker-compose run --rm test

echo ""
echo "Tests complete."
