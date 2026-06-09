"""Reproducibility and checkpointing utilities."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch.

    Also forces cuDNN into deterministic mode for reproducible convolutions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d", seed)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Re-seed each DataLoader worker so their RNGs don't collide."""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save model state dict along with optional metadata."""
    payload: dict = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: Optional[str] = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint into *model* and return any stored metadata."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=torch.device(device))
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        logger.info("Loaded checkpoint from %s", path)
        return checkpoint.get("metadata", {})
    else:
        model.load_state_dict(checkpoint, strict=strict)
        logger.info("Loaded state_dict from %s", path)
        return {}
