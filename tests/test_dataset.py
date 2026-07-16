"""
test_dataset.py — Tests du TransientDataset (sous-échantillonnage temporel t_stride).
Dataset synthétique npz, CPU uniquement.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch

ns, Nt, N = 4, 10, 8


@pytest.fixture
def toy_npz(tmp_path):
    rng   = np.random.default_rng(0)
    U     = rng.random((ns, Nt, N, N)).astype(np.float32)
    theta = rng.random((ns, 3)).astype(np.float32)
    path  = tmp_path / 'toy.npz'
    np.savez(path, U=U, theta=theta, dt=np.array([1.0]))
    return str(path), U


def test_t_stride_default_identity(toy_npz):
    from laplace_surrogate.data.dataset import TransientDataset
    path, U = toy_npz
    ds = TransientDataset(path, laplace=False)
    assert ds.Nt == Nt
    assert ds.dt == 1.0


def test_t_stride_subsamples_grid(toy_npz):
    from laplace_surrogate.data.dataset import TransientDataset
    path, U = toy_npz
    ds = TransientDataset(path, laplace=False, t_stride=2)
    assert ds.Nt == 5
    assert ds.dt == 2.0
    ds.fit([0, 1, 2])
    _, u = ds[0]
    assert u.shape == (5, N, N)
    # La dénormalisation doit redonner exactement U[0, ::2]
    u_denorm = u.numpy() * ds.U_std + ds.U_mean
    np.testing.assert_allclose(u_denorm, U[0, ::2], rtol=1e-5, atol=1e-5)


def test_t_stride_ceil(toy_npz):
    from laplace_surrogate.data.dataset import TransientDataset
    path, _ = toy_npz
    ds = TransientDataset(path, laplace=False, t_stride=3)
    assert ds.Nt == 4          # ceil(10/3)
    assert ds.dt == 3.0


def test_t_stride_rejected_with_laplace(toy_npz):
    from laplace_surrogate.data.dataset import TransientDataset
    path, _ = toy_npz
    with pytest.raises(NotImplementedError):
        TransientDataset(path, laplace=True, s_list=np.array([0.0 + 1.0j]), t_stride=2)
