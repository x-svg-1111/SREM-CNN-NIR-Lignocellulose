from __future__ import annotations

import numpy as np
import torch

from srem_nir.data.synthetic import make_synthetic_srem_dataset
from srem_nir.explain.interactions import pairwise_region_interactions
from srem_nir.models import CNNConfig, SREMCNN
from srem_nir.paper_config import paper_baseline_params, paper_srem_params
from srem_nir.spectra import build_srem_tensor
from srem_nir.split import ks_train_random_val_test_split


def test_srem_tensor_shape_and_range() -> None:
    data = make_synthetic_srem_dataset(n_samples=20, n_wavenumbers=300)
    bundle = build_srem_tensor(data["X"], data["wavenumbers"], target_length=80)
    assert bundle.X_mnp.shape == (20, 80, 3)
    assert bundle.X_pytorch.shape == (20, 3, 80)
    assert float(bundle.X_mnp.min()) >= -1e-5
    assert float(bundle.X_mnp.max()) <= 1.00001


def test_ks_split_sizes_and_disjointness() -> None:
    data = make_synthetic_srem_dataset(n_samples=50, n_wavenumbers=120)
    split = ks_train_random_val_test_split(data["X"], seed=2025)
    assert len(split.train) == 30
    assert len(split.val) == 10
    assert len(split.test) == 10
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)


def test_manuscript_split_sizes_for_550_samples() -> None:
    data = make_synthetic_srem_dataset(n_samples=550, n_wavenumbers=120)
    split = ks_train_random_val_test_split(data["X"], seed=2025)
    assert len(split.train) == 330
    assert len(split.val) == 110
    assert len(split.test) == 110


def test_interaction_decomposition_shape() -> None:
    X = np.random.default_rng(0).normal(size=(8, 3, 20)).astype(np.float32)
    model = SREMCNN(
        CNNConfig(
            input_length=20,
            block_filters=((4, 4, 4), (4, 4, 4), (4, 4, 4)),
            block_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3)),
            dense=8,
        )
    )
    out = pairwise_region_interactions(model, X, device="cpu")
    assert out["coupling_matrix"].shape == (3, 3)
    assert np.allclose(np.diag(out["coupling_matrix"]), 0.0)
    assert torch.isfinite(torch.as_tensor(out["coupling_matrix"])).all()


def test_paper_table_s1_params_are_available() -> None:
    assert paper_baseline_params("cellulose", "plsr")["n_components"] == 14
    assert paper_baseline_params("hemicellulose", "svr")["kernel"] == "linear"
    assert paper_baseline_params("lignin", "cnn")["kernels"] == (7, 25, 17)
    assert paper_baseline_params("lignin", "cnn")["se_mode"] == ("SE3",)
    lignin_srem = paper_srem_params("lignin")
    assert lignin_srem["block_filters"] == ((128, 4, 32), (128, 8, 64), (32, 4, 8))
    assert lignin_srem["strides"] == (2, 1)
    cellulose_srem = paper_srem_params("cellulose")
    assert cellulose_srem["se_mode"] == ("SE2", "SE3")
    assert cellulose_srem["se_after_conv"] == (False, True, True)
