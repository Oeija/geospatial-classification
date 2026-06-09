"""Data setup helpers: auto-extract and reproducible train/val splits."""

import json
import logging
import tarfile
from pathlib import Path
from typing import Tuple

import httpx
import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets

logger = logging.getLogger(__name__)

DEFAULT_DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"
)


def _download_tar_sync(url: str, dest: Path) -> None:
    """Download a tar archive synchronously via httpx."""
    logger.info("Downloading dataset from %s ...", url)
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Saved archive to %s (%d bytes)", dest, dest.stat().st_size)


def ensure_extracted(
    dataset_path: str = "data/raw/images_dataSAT",
    tar_path: str = "images-dataSAT.tar",
    download_url: str = DEFAULT_DATASET_URL,
) -> Path:
    """Extract the dataset archive if it hasn't been extracted yet.

    If the extracted directory and the local tar are both missing, the
    archive is downloaded from *download_url* automatically.

    Args:
        dataset_path: Expected path to the extracted dataset.
        tar_path: Path to the ``.tar`` archive (local or to be created).
        download_url: Fallback URL to fetch the archive.

    Returns:
        Path to the extracted dataset directory.
    """
    dest = Path(dataset_path)
    if dest.exists() and any(dest.iterdir()):
        logger.info("Dataset already present at %s", dest)
        return dest

    tar_file = Path(tar_path)
    if not tar_file.exists():
        logger.info("Local archive not found at %s", tar_file)
        tar_file.parent.mkdir(parents=True, exist_ok=True)
        _download_tar_sync(download_url, tar_file)

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s ...", tar_file, dest.parent)
    with tarfile.open(tar_file, "r") as tf:
        tf.extractall(path=dest.parent)
    logger.info("Dataset extracted to %s", dest)
    return dest


def load_or_create_splits(
    full_dataset: datasets.ImageFolder,
    train_split: float,
    seed: int,
    processed_dir: str = "data/processed",
) -> Tuple[Subset, Subset]:
    """Create train/val splits and cache indices for reproducibility.

    If cached indices exist for the given split fraction and seed, they are
    reloaded so every run uses the exact same train/val samples.

    Args:
        full_dataset: The full ``ImageFolder`` dataset.
        train_split: Fraction of data to use for training (e.g. 0.8).
        seed: Random seed for the split.
        processed_dir: Directory where split indices are cached.

    Returns:
        (train_dataset, val_dataset)
    """
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    cache_file = processed / f"split_indices_train{train_split}_seed{seed}.json"

    if cache_file.exists():
        logger.info("Loading cached split indices from %s", cache_file)
        with open(cache_file, "r") as f:
            indices = json.load(f)
        train_dataset = Subset(full_dataset, indices["train"])
        val_dataset = Subset(full_dataset, indices["val"])
        return train_dataset, val_dataset

    logger.info("Creating new train/val split (train_split=%.2f, seed=%d)", train_split, seed)
    total = len(full_dataset)
    train_size = int(train_split * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    indices = {
        "train": train_dataset.indices,
        "val": val_dataset.indices,
    }
    with open(cache_file, "w") as f:
        json.dump(indices, f)
    logger.info("Cached split indices to %s", cache_file)

    return train_dataset, val_dataset
