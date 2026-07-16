"""
test_models.py — Smoke tests pour les modèles (shape + loss).
Tailles jouets, CPU uniquement, < 15 s.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import pytest
import torch
import numpy as np

# Tailles jouets
N, Nt, K, B = 16, 8, 4, 2
LD   = 8   # latent_dim
TDIM = 3   # theta_dim


# ---------------------------------------------------------------------------
# SLAE
# ---------------------------------------------------------------------------

def test_slae_forward_shape():
    from laplace_surrogate.models.slae import SLAE
    model = SLAE(N=N, latent_dim=LD)
    u     = torch.randn(B, 2, N, N)
    U_hat, z = model(u, freq_ratio=0.5)
    assert U_hat.shape == (B, 2, N, N)
    assert z.shape == (B, LD)


def test_slae_loss():
    from laplace_surrogate.models.slae import SLAE
    model = SLAE(N=N, latent_dim=LD)
    u     = torch.randn(B, 2, N, N)
    U_hat, z = model(u, freq_ratio=0.0)
    loss, metrics = model.loss(u, U_hat, z)
    assert loss.ndim == 0
    assert isinstance(metrics, dict)
    assert 'recon_loss' in metrics


# ---------------------------------------------------------------------------
# LLAE
# ---------------------------------------------------------------------------

def test_llae_forward_shape():
    from laplace_surrogate.models.llae import LLAE
    model = LLAE(N=N, Nt=Nt, latent_dim=LD, K=K, dt=1.0)
    u     = torch.randn(B, Nt, N, N)
    U_rec, z_hat, z, z_rec = model(u)
    assert U_rec.shape == (B, Nt, N, N)
    assert z.shape    == (B, Nt, LD)


def test_llae_loss():
    from laplace_surrogate.models.llae import LLAE
    model = LLAE(N=N, Nt=Nt, latent_dim=LD, K=K, dt=1.0)
    u     = torch.randn(B, Nt, N, N)
    U_rec, z_hat, z, z_rec = model(u)
    loss, metrics = model.loss(u, U_rec, z_hat, z, z_rec)
    assert loss.ndim == 0
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# SLAEModel
# ---------------------------------------------------------------------------

def test_slae_model_generate_shape():
    from laplace_surrogate.models.slae_surrogate import SLAEModel
    model = SLAEModel(
        K=K, Nt=Nt, N=N, theta_dim=TDIM,
        latent_dim=LD, hidden_dim=32, head_dim=16,
        n_trunk=2, n_head=2, freq_L=4, surr_freq_L=4, dt=1.0,
    )
    model.U_mean.zero_()
    model.U_std.fill_(1.0)
    model.laplace.s_re.zero_()
    model.laplace.s_im.copy_(torch.linspace(0, math.pi, K))

    theta_n = torch.zeros(B, TDIM)
    with torch.no_grad():
        U_pred = model.generate(theta_n)
    assert U_pred.shape == (B, Nt, N, N)


# ---------------------------------------------------------------------------
# DL-ROM (baseline sans Laplace)
# ---------------------------------------------------------------------------

def test_dlrom_ae_forward_shape():
    from dl_rom.dlrom_ae import DLROMAE
    model = DLROMAE(N=N, Nt=Nt, latent_dim=LD)
    u     = torch.randn(B, Nt, N, N)
    U_rec, z = model(u)
    assert U_rec.shape == (B, Nt, N, N)
    assert z.shape == (B, Nt, LD)


def test_dlrom_ae_loss():
    from dl_rom.dlrom_ae import DLROMAE
    model = DLROMAE(N=N, Nt=Nt, latent_dim=LD)
    u     = torch.randn(B, Nt, N, N)
    U_rec, z = model(u)
    loss, metrics = model.loss(u, U_rec, z)
    assert loss.ndim == 0
    assert isinstance(metrics, dict)
    assert 'recon' in metrics


def test_dlrom_model_forward_and_generate():
    from dl_rom.dlrom_ae import DLROMAE
    from dl_rom.dlrom_surrogate import DLROMModel
    ae    = DLROMAE(N=N, Nt=Nt, latent_dim=LD)
    model = DLROMModel(ae, theta_dim=TDIM, hidden_dim=32, head_dim=16,
                       n_trunk=2, n_head=2, freq_L=4)
    theta = torch.zeros(B, TDIM)
    u     = torch.randn(B, Nt, N, N)

    U_rec, z_pred, z_true = model(theta, u)
    assert U_rec.shape  == (B, Nt, N, N)
    assert z_pred.shape == (B, Nt, LD)
    assert z_true.shape == (B, Nt, LD)

    loss, metrics = model.loss(u, U_rec, z_pred, z_true)
    assert loss.ndim == 0
    assert 'lat' in metrics and 'spat' in metrics

    U_gen = model.generate(theta)
    assert U_gen.shape == (B, Nt, N, N)


def test_dlrom_model_encoder_frozen():
    from dl_rom.dlrom_ae import DLROMAE
    from dl_rom.dlrom_surrogate import DLROMModel
    ae    = DLROMAE(N=N, Nt=Nt, latent_dim=LD)
    model = DLROMModel(ae, theta_dim=TDIM, hidden_dim=32, head_dim=16,
                       n_trunk=2, n_head=2, freq_L=4)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    assert all(p.requires_grad for p in model.surrogate.parameters())


# ---------------------------------------------------------------------------
# CorrectionAE
# ---------------------------------------------------------------------------

def test_correction_ae_shape():
    from laplace_surrogate.models.corrector import CorrectionAE
    model = CorrectionAE(N=N, base_ch=4)
    frame = torch.randn(B, N, N)  # (B, N, N) — une frame par sample
    out   = model(frame)
    assert out.shape == (B, N, N)


def test_correction_ae_zero_init():
    """out_conv zero-init → correction résiduelle ≈ 0 → U_corrected ≈ U_pred."""
    from laplace_surrogate.models.corrector import CorrectionAE
    model = CorrectionAE(N=N, base_ch=4)
    model.eval()
    frame = torch.randn(B, N, N)
    with torch.no_grad():
        out = model(frame)
    assert torch.allclose(out, frame, atol=1e-3), \
        f"Résidu non nul à l'initialisation : {(out - frame).abs().max():.4f}"


