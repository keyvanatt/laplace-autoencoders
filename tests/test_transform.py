"""
test_transform.py — Smoke tests pour les transformées de Laplace.
Tailles jouets, CPU uniquement, < 5 s.
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np

from laplace_surrogate.laplace_transform.learnable import LearnableLaplace
from laplace_surrogate.laplace_transform.forward import laplace_forward_tik
from laplace_surrogate.laplace_transform.inverse import laplace_inverse_tik


B, Nt, D, K = 2, 8, 4, 4
DT = 1.0


def _make_ll():
    ll = LearnableLaplace(K=K, dt=DT, Nt=Nt, gamma_init=1e-2, learnable=False)
    ll.eval()
    return ll


def test_learnable_laplace_forward_shape():
    ll    = _make_ll()
    z     = torch.randn(B, Nt, D)
    z_hat = ll.forward_transform(z)
    assert z_hat.shape == (B, K, D), f"Expected (B,K,D) got {z_hat.shape}"
    assert z_hat.is_complex()


def test_learnable_laplace_inverse_shape():
    ll    = _make_ll()
    z     = torch.randn(B, Nt, D)
    z_hat = ll.forward_transform(z)
    z_rec = ll.inverse_transform(z_hat, Nt)
    assert z_rec.shape == (B, Nt, D)
    assert z_rec.dtype == torch.float32


def test_learnable_laplace_roundtrip():
    """Aller-retour Laplace sur signal lisse — erreur < 50 % (tolérance large, K=4 petit)."""
    ll = _make_ll()
    t  = torch.linspace(0, 5, Nt)
    z  = torch.exp(-0.5 * (t - 2.5) ** 2)[None, :, None].expand(B, Nt, D)
    z_hat = ll.forward_transform(z)
    z_rec = ll.inverse_transform(z_hat, Nt)
    rel_err = (z_rec - z).norm() / z.norm()
    assert rel_err < 0.5, f"Roundtrip rel_err={rel_err:.3f}"


def test_forward_tik_shape():
    K_   = 6
    Nt_  = 8
    s_im = np.linspace(0, math.pi / DT, K_)
    s    = (0.01 + 1j * s_im).astype(np.complex128)
    U    = torch.randn(5, Nt_)
    U_hat = laplace_forward_tik(U, s, dt=DT, rule='trap')
    assert U_hat.shape == (5, K_)
    assert U_hat.is_complex()


def test_inverse_tik_shape():
    K_   = 6
    Nt_  = 8
    s_im = np.linspace(0, math.pi / DT, K_)
    s    = torch.tensor((0.01 + 1j * s_im), dtype=torch.complex128)
    U_hat = torch.randn(3, K_, dtype=torch.complex128)
    U_rec = laplace_inverse_tik(U_hat, s, dt=DT, Nt=Nt_,
                                alpha_t=0.0, lam=1e-6, rule='trap')
    assert U_rec.shape == (3, Nt_)
