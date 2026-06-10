"""Unit tests for training utilities."""

import numpy as np
import torch

from geospatial_classification.training import set_seed, worker_init_fn


class TestReproducibility:
    def test_set_seed_torch(self) -> None:
        set_seed(123)
        a = torch.rand(10)
        set_seed(123)
        b = torch.rand(10)
        assert torch.allclose(a, b)

    def test_set_seed_numpy(self) -> None:
        set_seed(456)
        a = np.random.rand(10)
        set_seed(456)
        b = np.random.rand(10)
        assert np.allclose(a, b)

    def test_worker_init_fn(self) -> None:
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        worker_init_fn(0, base_seed=42)
        b = np.random.rand(5)
        assert np.allclose(a, b)
