"""
llae.py — Latent Laplace Autoencoder (LLAE).

Pipeline : U(t) → ConvEncoder(t_ratio) → z(t) → LearnableLaplace → ẑ(s_k)
           → LearnableLaplace⁻¹ → z̃(t) → ConvDecoder(t_ratio) → Û(t)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseAutoEncoder
from laplace_surrogate.models.encoder_decoder import ConvEncoder, ConvDecoder
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


class LLAE(BaseAutoEncoder):
    """
    Latent Laplace Autoencoder (LLAE).

    Autoencoder dont la transformée de Laplace opère dans l'espace latent,
    avec conditionnement FiLM sur t_ratio = t / (Nt-1) dans l'encodeur et le décodeur.

    Paramètres
    ----------
    N           : résolution spatiale (multiple de 8)
    Nt          : nombre de pas de temps
    latent_dim  : dimension de l'espace latent z
    K           : nombre de fréquences de Laplace apprenables
    dt          : pas de temps (fixe)
    beta        : poids ridge sur z (régularisation latente)
    beta_latent : poids de la MSE de reconstruction dans l'espace latent
    gamma_init  : valeur initiale de l'amortissement γ
    time_L      : niveaux de fréquence pour l'encoding sinusoïdal du temps
    """

    def __init__(
        self,
        N                    : int   = 64,
        Nt                   : int   = 150,
        latent_dim           : int   = 64,
        K                    : int   = 64,
        dt                   : float = 1.0,
        beta                 : float = 1e-3,
        beta_latent          : float = 0.1,
        gamma_init           : float = 1e-2,
        time_L               : int   = 8,
        learnable_laplace    : bool  = False,
        alpha_t              : float = math.exp(-2.0),
        lam                  : float = math.exp(-2.0),
        decoder_norm         : str   = 'gn',
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.Nt          = Nt
        self.beta        = beta
        self.beta_latent = beta_latent

        self.encoder = ConvEncoder(in_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)
        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L,
                                   norm=decoder_norm)
        self.laplace  = LearnableLaplace(K, dt, Nt, gamma_init, learnable=learnable_laplace,
                                         alpha_t=alpha_t, lam=lam)

    def _make_t_ratios(self, Nt: int, B: int, dtype, device) -> torch.Tensor:
        """Retourne (B*Nt, 1) avec t_ratio = t / max(Nt-1, 1)."""
        t = torch.arange(Nt, dtype=dtype, device=device) / max(Nt - 1, 1)
        return t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)

    def _encode_seq(self, U: torch.Tensor) -> torch.Tensor:
        """U : (B, Nt, N, N) → z : (B, Nt, latent_dim)"""
        B, Nt, N, _ = U.shape
        frames   = U.reshape(B * Nt, 1, N, N)
        t_ratios = self._make_t_ratios(Nt, B, U.dtype, U.device)
        return self.encoder(frames, t_ratios).view(B, Nt, self.latent_dim)

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, Nt, latent_dim) → U : (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._make_t_ratios(Nt, B, z.dtype, z.device)
        U = self.decoder(flat, t_ratios)
        return U.view(B, Nt, U.shape[-1], U.shape[-1])

    def forward(
        self, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        U : (B, Nt, N, N)
        retourne : (U_rec, z_hat, z, z_rec)
        """
        Nt    = U.shape[1]
        z     = self._encode_seq(U)
        z_hat = self.laplace.forward_transform(z)
        z_rec = self.laplace.inverse_transform(z_hat, Nt)
        U_rec = self._decode_seq(z_rec)
        return U_rec, z_hat, z, z_rec

    def loss(
        self,
        U    : torch.Tensor,
        U_rec: torch.Tensor,
        z_hat: torch.Tensor,
        z    : torch.Tensor,
        z_rec: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        recon   = F.mse_loss(U_rec, U)
        lat_rec = F.mse_loss(z_rec, z)
        ridge   = z_hat.abs().pow(2).mean()
        total   = recon + self.beta_latent * lat_rec + self.beta * ridge
        return total, {
            'recon'  : recon.detach(),
            'lat_rec': lat_rec.detach(),
            'ridge'  : ridge.detach(),
        }

