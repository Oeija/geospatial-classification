"""Data loading and dataset utilities."""

import logging
import os
import tarfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomBinaryClassDataset(Dataset):
    """A custom dataset for agricultural land classification.

    Expects two directories on disk, one per class.
    """

    def __init__(
        self,
        non_agri_dir: str,
        agri_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for d, label in [(non_agri_dir, 0), (agri_dir, 1)]:
            if not os.path.isdir(d):
                logger.warning("Directory not found, skipping: %s", d)
                continue
            for img in sorted(os.listdir(d)):
                path = os.path.join(d, img)
                if os.path.isfile(path):
                    self.image_paths.append(path)
                    self.labels.append(label)

        if not self.image_paths:
            raise RuntimeError("No images found in the provided directories.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


async def download_dataset(
    url: str,
    extract_dir: str = ".",
    overwrite: bool = True,
) -> Path:
    """Download a ``.tar`` dataset and extract it.

    Falls back to ``httpx`` + ``tarfile`` if ``skillsnetwork`` is not available.

    Returns:
        Path to the extracted directory.
    """
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    try:
        import skillsnetwork  # type: ignore[import-not-found]

        await skillsnetwork.prepare(  # type: ignore[misc]
            url=url, path=extract_dir, overwrite=overwrite
        )
        logger.info("Downloaded via skillsnetwork: %s", url)
    except Exception:
        logger.warning("skillsnetwork unavailable; falling back to manual download.")
        file_name = Path(url).name
        tar_path = extract_path / file_name
        if not tar_path.exists() or overwrite:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                tar_path.write_bytes(response.content)
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_dir)
        logger.info("Downloaded and extracted: %s", tar_path)

    return extract_path


async def download_model_weights(url: str, model_path: str) -> Path:
    """Download a model checkpoint if it does not already exist."""
    path = Path(model_path)
    if path.exists():
        logger.info("Model already exists at %s", path)
        return path

    logger.info("Downloading model from %s ...", url)
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info("Saved model to %s", path)
    return path
