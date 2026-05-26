"""
encoder_decoder.py — Backbone Conv partagé (FiLM + sinusoidal encoding).

ConvEncoder  : (B, in_ch, N, N) → (B, latent_dim)
ConvDecoder  : (B, latent_dim)  → (B, out_ch, N, N)

Utilisé par LLAE et LSLAE (in_channels=1 pour frames temporelles,
in_channels=2 pour frames fréquentielles Re/Im).
"""
import math

import torch
import torch.nn as nn


def _to_f_vec(freq_ratio, B: int, dtype, device) -> torch.Tensor:
    """Convertit freq_ratio (float scalaire ou tenseur (B,)) en (B, 1)."""
    if isinstance(freq_ratio, (float, int)):
        return torch.full((B, 1), float(freq_ratio), dtype=dtype, device=device)
    return freq_ratio.to(dtype=dtype, device=device).view(B, 1)


class SinusoidalFreqEncoding(nn.Module):
    """
    Encode un scalaire f ∈ [0, 1] en un vecteur de dimension 2·L via un
    positional encoding sinusoïdal (style NeRF), avant de le passer dans
    un petit MLP pour obtenir l'embedding de conditionnement FiLM.

    Pour le ième niveau de fréquence (i = 0, …, L-1) :
        sin(2^i · π · f),  cos(2^i · π · f)
    """

    def __init__(self, L: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.L = L
        freqs = math.pi * (2.0 ** torch.arange(L).float())
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * L, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f : (B, 1) → (B, out_dim)"""
        angles = f * self.freqs
        enc = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.mlp(enc)


class ConvEncoder(nn.Module):
    """
    Encodeur générique : (B, in_channels, N, N) → (B, latent_dim)
    conditionné sur un ratio scalaire ∈ [0, 1] via FiLM sinusoïdal.

    in_channels=1 : frame temporelle réelle U(t)
    in_channels=2 : frame fréquentielle complexe Û(s_k) — canaux Re et Im
    N doit être multiple de 8.
    """

    def __init__(self, in_channels: int, N: int, latent_dim: int, cond_L: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 4, 2, 1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.cond_enc = SinusoidalFreqEncoding(L=cond_L, hidden_dim=64, out_dim=64)
        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)
        conv_out = 64 * (N // 8) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, cond_ratio=0.0) -> torch.Tensor:
        """U : (B, in_channels, N, N) → z : (B, latent_dim)"""
        B = U.shape[0]
        emb = self.cond_enc(_to_f_vec(cond_ratio, B, U.dtype, U.device))
        x = self._film(self.conv1(U), self.film1, emb)
        x = self._film(self.conv2(x), self.film2, emb)
        x = self._film(self.conv3(x), self.film3, emb)
        return self.fc(x.flatten(1))


class ConvDecoder(nn.Module):
    """
    Décodeur générique : (B, latent_dim) → (B, out_channels, N, N)
    conditionné sur un ratio scalaire ∈ [0, 1] via FiLM sinusoïdal.

    out_channels=1 : frame temporelle réelle
    out_channels=2 : frame fréquentielle complexe — canaux Re et Im

    Backbone : FC + 3 deconv stride-2 → 32 canaux.
    Tête     : out_channels blocs de raffinement séparés Conv(32→32→32→1).
    N doit être multiple de 8.
    """

    def __init__(self, out_channels: int, N: int, latent_dim: int, cond_L: int = 8):
        super().__init__()
        self.base = N // 8
        self.fc = nn.Sequential(nn.Linear(latent_dim, 64 * self.base ** 2), nn.ReLU())
        self.cond_enc = SinusoidalFreqEncoding(L=cond_L, hidden_dim=64, out_dim=64)
        self.film1 = nn.Linear(64, 2 * 64)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 16)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU())
        self.heads = nn.ModuleList([self._make_head() for _ in range(out_channels)])

    @staticmethod
    def _make_head() -> nn.Sequential:
        head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1,  1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def _film(self, x: torch.Tensor, proj: nn.Linear, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, cond_ratio=0.0) -> torch.Tensor:
        """z : (B, latent_dim) → (B, out_channels, N, N)"""
        B = z.shape[0]
        emb = self.cond_enc(_to_f_vec(cond_ratio, B, z.dtype, z.device))
        x = self.fc(z).view(B, 64, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, emb)
        x = self._film(self.deconv2(x), self.film2, emb)
        x = self._film(self.deconv3(x), self.film3, emb)
        return torch.cat([head(x) for head in self.heads], dim=1)
