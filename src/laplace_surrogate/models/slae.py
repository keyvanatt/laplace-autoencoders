"""
slae.py — Spatial Laplace Autoencoder (SLAE).

Pipeline : Û(s_k) → LaplaceEncoder → z → LaplaceDecoder → Û_rec(s_k)
Conditionné sur freq_ratio = k/K via FiLM sinusoïdal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseAutoEncoder
from laplace_surrogate.models.encoder_decoder import SinusoidalFreqEncoding, _to_f_vec


class LaplaceEncoder(nn.Module):
    """
    U → z, conditionné sur freq_ratio = k / K ∈ [0, 1].

    U : (B, 2, N, N)  champ spatial complexe (Re, Im), normalisé
    """

    def __init__(self, N: int, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N = N

        self.conv1 = nn.Sequential(nn.Conv2d(2,   16,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,  32,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))

        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)

        conv_out = 64 * (N // 8) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne z : (B, latent_dim)."""
        B = U.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, U.dtype, U.device))

        x = self._film(self.conv1(U), self.film1, f_emb)
        x = self._film(self.conv2(x), self.film2, f_emb)
        x = self._film(self.conv3(x), self.film3, f_emb)
        h = x.flatten(start_dim=1)
        return self.fc(h)


class LaplaceDecoder(nn.Module):
    """
    z → Û (complexe, normalisé), conditionné sur freq_ratio = k / K ∈ [0, 1].

    z : (B, latent_dim)
    retourne : (B, 2, N, N)  — canaux Re et Im

    Architecture :
    - base = N // 8 → 3 étapes d'upsampling.
    - FiLM conditioning (sinusoïdal) après chaque bloc deconv.
    - Deux blocs de raffinement séparés (Re / Im) pour les détails fins.
    """

    def __init__(self, N: int = 64, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N    = N
        self.base = N // 8

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.base ** 2),
            nn.ReLU(),
        )

        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())

        self.refine_re = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),
        )
        self.refine_im = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne Û : (B, 2, N, N)."""
        B = z.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, z.dtype, z.device))

        x = self.fc(z).view(B, 128, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, f_emb)
        x = self._film(self.deconv2(x), self.film2, f_emb)
        x = self._film(self.deconv3(x), self.film3, f_emb)

        re = self.refine_re(x)
        im = self.refine_im(x)
        return torch.cat([re, im], dim=1)


class SLAE(BaseAutoEncoder):
    """
    Spatial Laplace Autoencoder (SLAE).

    Autoencoder déterministe conditionné sur la fréquence de Laplace k/K.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 128)
    latent_dim : dimension de l'espace latent z
    beta       : poids de la régularisation ridge (L2 sur la sortie)
    freq_L     : nombre de niveaux de fréquence pour le sinusoidal encoding
    """

    def __init__(
        self,
        N          : int   = 64,
        latent_dim : int   = 32,
        beta       : float = 1e-3,
        freq_L     : int   = 8,
    ):
        super().__init__()

        self.beta       = beta
        self.latent_dim = latent_dim

        self.encoder = LaplaceEncoder(N, latent_dim, freq_L=freq_L)
        self.decoder = LaplaceDecoder(N, latent_dim, freq_L=freq_L)

    def forward(
        self, U: torch.Tensor, freq_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne : (Û, z)

        Û : (B, 2, N, N)  reconstruction normalisée
        z : (B, latent_dim)  code latent
        """
        z     = self.encoder(U, freq_ratio)
        U_hat = self.decoder(z, freq_ratio)
        return U_hat, z

    def loss(
        self,
        U     : torch.Tensor,
        U_hat : torch.Tensor,
        z     : torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(U_hat, U)
        ridge      = z.pow(2).mean()
        total      = recon_loss + self.beta * ridge
        return total, {
            'recon_loss': recon_loss.detach(),
            'ridge'     : ridge.detach(),
        }

