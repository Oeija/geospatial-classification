"""Image transforms and augmentation pipelines."""

from typing import Tuple

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(
    img_size: int = 64,
    rotation: int = 40,
    shear: float = 0.2,
) -> transforms.Compose:
    """Heavy augmentation pipeline for training."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(rotation),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=shear),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transform(img_size: int = 64) -> transforms.Compose:
    """Lightweight pipeline for validation / testing (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_inference_transform(img_size: Tuple[int, int] = (64, 64)) -> transforms.Compose:
    """Inference pipeline used during model evaluation."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
