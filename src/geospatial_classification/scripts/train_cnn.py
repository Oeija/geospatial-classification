"""CLI script to train the CNN backbone."""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from geospatial_classification.config import get_device, load_config
from geospatial_classification.data import ensure_extracted, load_or_create_splits
from geospatial_classification.data.transforms import get_train_transform, get_val_transform
from geospatial_classification.evaluation.plots import plot_training_curves
from geospatial_classification.models import ConvNet
from geospatial_classification.training import fit, set_seed, worker_init_fn
from geospatial_classification.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN backbone.")
    parser.add_argument(
        "--config",
        default="configs/train_cnn.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
        log_file=cfg.get("logging", {}).get("log_file"),
    )
    logger.info("Loaded config from %s", args.config)

    # Reproducibility
    seed = cfg["training"]["seed"]
    set_seed(seed)

    # Device
    device_pref = cfg["training"].get("device", "auto")
    device = get_device() if device_pref == "auto" else device_pref
    logger.info("Using device: %s", device)

    # Data
    ds_cfg = cfg["data"]
    img_size = ds_cfg["img_size"]
    batch_size = ds_cfg["batch_size"]
    num_workers = ds_cfg.get("num_workers", 4)
    dataset_path = ds_cfg["dataset_path"]

    # Ensure dataset is extracted and load cached train/val splits
    ensure_extracted(dataset_path=dataset_path)

    train_transform = get_train_transform(img_size)
    val_transform = get_val_transform(img_size)

    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    train_dataset, val_dataset = load_or_create_splits(
        full_dataset,
        train_split=ds_cfg["train_split"],
        seed=seed,
    )
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
    )
    logger.info(
        "Train: %d samples | Val: %d samples", len(train_dataset), len(val_dataset)
    )

    # Model
    model_cfg = cfg["model"]
    model = ConvNet(num_classes=model_cfg["num_classes"]).to(device)
    logger.info(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    checkpoint_path = model_cfg["checkpoint_name"]

    patience = cfg["training"].get("early_stopping", {}).get("patience")
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg["training"]["epochs"],
        checkpoint_path=checkpoint_path,
        patience=patience,
    )

    logger.info("Training complete. Best checkpoint: %s", checkpoint_path)
    plot_training_curves(history, save_path="plots/cnn_training_curves.png")


if __name__ == "__main__":
    main()
