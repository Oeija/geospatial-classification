"""Unit tests for data utilities."""

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from geospatial_classification.data.download import CustomBinaryClassDataset
from geospatial_classification.data.transforms import (
    get_inference_transform,
    get_train_transform,
    get_val_transform,
)


class TestTransforms:
    def test_train_transform_output_shape(self) -> None:
        tfm = get_train_transform(img_size=64)
        from PIL import Image

        img = Image.new("RGB", (128, 128), color="red")
        tensor = tfm(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_val_transform_no_augmentation(self) -> None:
        tfm = get_val_transform(img_size=64)
        from PIL import Image

        img = Image.new("RGB", (128, 128), color="blue")
        tensor = tfm(img)
        assert tensor.shape == (3, 64, 64)

    def test_inference_transform(self) -> None:
        tfm = get_inference_transform((64, 64))
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        tensor = tfm(img)
        assert tensor.shape == (3, 64, 64)


class TestCustomDataset:
    def test_empty_dirs_raise(self, tmp_path: Path) -> None:
        non_agri = tmp_path / "non_agri"
        agri = tmp_path / "agri"
        non_agri.mkdir()
        agri.mkdir()
        with pytest.raises(RuntimeError):
            CustomBinaryClassDataset(str(non_agri), str(agri))

    def test_dataset_length(self, tmp_path: Path) -> None:
        from PIL import Image

        non_agri = tmp_path / "non_agri"
        agri = tmp_path / "agri"
        non_agri.mkdir()
        agri.mkdir()
        Image.new("RGB", (10, 10)).save(non_agri / "a.png")
        Image.new("RGB", (10, 10)).save(agri / "b.png")

        ds = CustomBinaryClassDataset(str(non_agri), str(agri))
        assert len(ds) == 2

        img, label = ds[0]
        assert label == 0
        img2, label2 = ds[1]
        assert label2 == 1
