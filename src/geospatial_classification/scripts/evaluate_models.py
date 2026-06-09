"""CLI script to evaluate the PyTorch CNN-ViT hybrid model."""

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from geospatial_classification.config import get_device, load_config
from geospatial_classification.data import ensure_extracted
from geospatial_classification.data.transforms import get_inference_transform
from geospatial_classification.evaluation import plot_roc, print_metrics
from geospatial_classification.models import CNN_ViT_Hybrid
from geospatial_classification.training import set_seed
from geospatial_classification.utils import setup_logging

logger = logging.getLogger(__name__)


def evaluate_pytorch(
    model: CNN_ViT_Hybrid,
    loader: DataLoader,
    device: str,
) -> tuple:
    """Run PyTorch inference and return labels, preds, probs."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="PyTorch inference"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch CNN-ViT hybrid.")
    parser.add_argument(
        "--config",
        default="configs/evaluate.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
        log_file=cfg.get("logging", {}).get("log_file"),
    )
    logger.info("Loaded config from %s", args.config)

    seed = cfg["evaluation"]["seed"]
    set_seed(seed)

    device_pref = cfg["evaluation"].get("device", "auto")
    device = get_device() if device_pref == "auto" else device_pref
    logger.info("Using device: %s", device)

    ds_cfg = cfg["data"]
    img_size = ds_cfg["img_size"]
    batch_size = ds_cfg["batch_size"]
    dataset_path = ds_cfg["dataset_path"]

    class_labels = cfg["evaluation"]["class_labels"]
    plots_dir = Path(cfg["evaluation"].get("plots_dir", "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    pt_cfg = cfg["models"]["pytorch"]
    pt_model = CNN_ViT_Hybrid(
        num_classes=len(class_labels),
        embed_dim=pt_cfg["embed_dim"],
        depth=pt_cfg["depth"],
        heads=pt_cfg["heads"],
    ).to(device)

    pt_path = pt_cfg["state_dict_path"]
    if Path(pt_path).exists():
        state = torch.load(pt_path, map_location=torch.device(device))
        if "state_dict" in state:
            pt_model.load_state_dict(state["state_dict"], strict=False)
        else:
            pt_model.load_state_dict(state, strict=False)
        logger.info("Loaded PyTorch model from %s", pt_path)
    else:
        logger.error("PyTorch checkpoint not found: %s", pt_path)
        return

    ensure_extracted(dataset_path=dataset_path)

    transform = get_inference_transform((img_size, img_size))
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    pt_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    y_true_pt, y_pred_pt, y_prob_pt = evaluate_pytorch(pt_model, pt_loader, device)
    print_metrics(y_true_pt, y_pred_pt, y_prob_pt, class_labels, "PyTorch CNN-ViT Hybrid")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_roc(y_true_pt, y_prob_pt, "PyTorch Model", ax=ax)
    fig.savefig(plots_dir / "roc_pytorch.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true_pt, y_pred_pt, display_labels=class_labels, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix — PyTorch CNN-ViT Hybrid")
    fig.savefig(plots_dir / "confusion_matrix_pytorch.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
