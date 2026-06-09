"""Training and validation loops."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Run one training epoch.

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = loss_sum / total if total else 0.0
    accuracy = correct / total if total else 0.0
    logger.info("Train loss: %.4f | accuracy: %.4f", avg_loss, accuracy)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Run one validation epoch.

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / total if total else 0.0
    accuracy = correct / total if total else 0.0
    logger.info("Val   loss: %.4f | accuracy: %.4f", avg_loss, accuracy)
    return avg_loss, accuracy


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epochs: int,
    checkpoint_path: str,
    patience: Optional[int] = None,
) -> Dict[str, list]:
    """Train a model for multiple epochs with validation, checkpointing, and
    optional early stopping.

    Args:
        patience: Number of epochs to wait for validation loss improvement
            before stopping early. ``None`` disables early stopping.

    Returns:
        History dictionary with ``loss`` and ``accuracy`` keys for
        ``train`` and ``val`` splits.
    """
    best_loss = float("inf")
    epochs_no_improve = 0
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Saved new best checkpoint to %s", checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if patience is not None and epochs_no_improve >= patience:
            logger.info(
                "Early stopping triggered after %d epochs (no improvement for %d epochs).",
                epoch,
                patience,
            )
            break

    return history
