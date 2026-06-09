# ------------------------------------------------------------------------------
# Multi-target Dockerfile for geospatial-classification
#
# Default build: GPU with stable PyTorch + CUDA 12.8
# Override PYTORCH_INDEX and USE_PRE for CPU-only or nightly (RTX 5060/Blackwell)
#
# Examples:
#   docker build -t gsc .
#   docker build -t gsc --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu .
#   docker build -t gsc \
#       --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 \
#       --build-arg USE_PRE=1 .
# ------------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for Pillow, matplotlib, and PyTorch
# libgl1 replaces the obsolete libgl1-mesa-glx in Debian Trixie+
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# PyTorch variant selection
#   PYTORCH_INDEX : pip index URL (default = stable CUDA 12.8)
#   USE_PRE       : set to 1 when using the nightly index so that pre-release
#                   wheels are accepted.  Only needed for RTX 5060/Blackwell
#                   or other architectures not yet covered by stable wheels.
# ------------------------------------------------------------------------------
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG USE_PRE=0

COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    if [ "$USE_PRE" = "1" ]; then \
        echo "Installing PyTorch PRE-RELEASE from ${PYTORCH_INDEX}"; \
        pip install --pre torch --index-url ${PYTORCH_INDEX}; \
        pip install --pre torchvision --index-url ${PYTORCH_INDEX} --no-deps; \
    else \
        echo "Installing PyTorch STABLE from ${PYTORCH_INDEX}"; \
        pip install torch torchvision --index-url ${PYTORCH_INDEX}; \
    fi

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the package in editable mode after source is present
RUN pip install -e /app

CMD ["python", "-m", "geospatial_classification.scripts.train_cnn", "--config", "configs/train_cnn.yaml"]
