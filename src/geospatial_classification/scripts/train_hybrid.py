"""CLI script to train the CNN-ViT hybrid."""

import argparse
import logging
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from geospatial_classification.config import get_device, load_config
from geospatial_classification.data import ensure_extracted, load_or_create_splits
from geospatial_classification.data.transforms import get_train_transform, get_val_transform
from geospatial_classification.evaluation.plots import plot_training_curves
from geospatial_classification.models import CNN_ViT_Hybrid
from geospatial_classification.training import fit, load_checkpoint, set_seed, worker_init_fn
from geospatial_classification.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN-ViT hybrid.")
    parser.add_argument(
        "--config",
        default="configs/train_hybrid.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
        log_file=cfg.get("logging", {}).get("log_file"),
    )
    logger.info("Loaded config from %s", args.config)

    seed = cfg["training"]["seed"]
    set_seed(seed)

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
    logger.info("Train: %d samples | Val: %d samples", len(train_dataset), len(val_dataset))

    # Model
    m_cfg = cfg["model"]
    model = CNN_ViT_Hybrid(
        num_classes=m_cfg["num_classes"],
        embed_dim=m_cfg["embed_dim"],
        depth=m_cfg["depth"],
        heads=m_cfg["heads"],
    ).to(device)

    # Transfer learning: load CNN backbone
    cnn_path = m_cfg.get("pretrained_cnn_path")
    if cnn_path and Path(cnn_path).exists():
        load_checkpoint(
            model.cnn, cnn_path, device=device, strict=m_cfg.get("strict_loading", True)
        )
        logger.info("Loaded pre-trained CNN from %s", cnn_path)
    else:
        logger.warning("Pre-trained CNN path not found: %s", cnn_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    checkpoint_path = m_cfg["checkpoint_name"]

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
    plot_training_curves(history, save_path="plots/hybrid_training_curves.png")


if __name__ == "__main__":
    main()
