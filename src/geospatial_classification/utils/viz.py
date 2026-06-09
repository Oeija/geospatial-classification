"""Visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(img: torch.Tensor) -> np.ndarray:
    """Un-normalize a tensor image and convert to numpy HWC format.

    Assumes the tensor was normalized with mean=0.5 and std=0.5
    per channel.  For ImageNet normalization use
    :func:`imshow_imagenet`.
    """
    img = img / 2 + 0.5  # back to [0, 1]
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def imshow_imagenet(img: torch.Tensor) -> np.ndarray:
    """Un-normalize an ImageNet-normalized tensor image.

    Mean and std are hard-coded to the standard ImageNet values.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img.numpy()
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def show_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list,
    n: int = 8,
    title: str = "Batch",
) -> None:
    """Display a grid of images with their labels."""
    n = min(n, images.size(0))
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        img = imshow_imagenet(images[i])
        ax.imshow(img)
        ax.set_title(f"{class_names[labels[i].item()]}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
