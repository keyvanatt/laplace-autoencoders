"""
llae_surrogate.py — Surrogate θ → latents Laplace pour LLAE (LLAEModel).

LLAESurrogate : trunk MLP partagé + K heads fréquentiels, sortie (B, K, out_dim).
LLAEModel     : surrogate + décodeur pour LLAE (pipeline LLAE).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from laplace_surrogate.models.encoder_decoder import ConvDecoder, ConvEncoder
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers = []
    for i in range(n_layers):
        d_in  = in_dim    if i == 0            else hidden_dim
        d_out = out_dim   if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class LLAESurrogate(nn.Module):
    """
    Trunk MLP partagé (θ → h) + K FFN heads par fréquence (h → z_k).

    Conditionnement fréquentiel : encoding sinusoïdal de k/(K-1) concaténé à h
    avant chaque head.

    Paramètres
    ----------
    theta_dim  : dimension de θ (entrée)
    out_dim    : dimension de sortie par fréquence
    K          : nombre de fréquences de Laplace
    shared_dim : largeur du trunk
    head_dim   : largeur cachée de chaque head
    n_trunk    : nombre de couches Linear dans le trunk
    n_head     : nombre de couches Linear dans chaque head (≥ 2)
    freq_L     : niveaux sinusoïdaux pour l'encoding fréquentiel
    """

    def __init__(
        self,
        theta_dim  : int,
        out_dim    : int,
        K          : int,
        shared_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        self.K      = K
        self.freq_L = freq_L
        freq_cond   = 2 * freq_L

        trunk_layers = []
        for i in range(n_trunk):
            d_in = theta_dim if i == 0 else shared_dim
            trunk_layers += [nn.Linear(d_in, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()]
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            _make_mlp(shared_dim + freq_cond, head_dim, out_dim, n_head)
            for _ in range(K)
        ])

        freq_ratios = torch.arange(K).float() / max(K - 1, 1)
        self.register_buffer('_freq_ratios', freq_ratios)

    def _sinenc(self, ratios: torch.Tensor) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.freq_L, device=ratios.device, dtype=ratios.dtype)) * torch.pi
        x = ratios[:, None] * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """theta: (B, theta_dim) → (B, K, out_dim)"""
        h        = self.trunk(theta)
        freq_enc = self._sinenc(self._freq_ratios)
        B = h.shape[0]
        outs = []
        for k, head in enumerate(self.heads):
            e_k = freq_enc[k].unsqueeze(0).expand(B, -1)
            outs.append(head(torch.cat([h, e_k], dim=1)))
        return torch.stack(outs, dim=1)


class LLAEModel(nn.Module):
    """
    Surrogate θ → U_rec pour LLAE.

    Trainable : LLAESurrogate (θ → ẑ_pred) + ConvDecoder(1) (z̃ → U)
    Gelé      : ConvEncoder(1) + LearnableLaplace (cibles latentes pendant le train)
    """

    def __init__(
        self,
        ae,
        theta_dim  : int,
        shared_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        D  = ae.latent_dim
        K  = ae.laplace.K
        self.latent_dim = D
        self.K          = K
        self.Nt         = ae.Nt

        self.surrogate = LLAESurrogate(
            theta_dim, 2 * D, K, shared_dim, head_dim, n_trunk, n_head, freq_L,
        )
        self.decoder = copy.deepcopy(ae.decoder)

        self.encoder = _freeze(copy.deepcopy(ae.encoder))
        self.laplace  = _freeze(copy.deepcopy(ae.laplace))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.laplace.eval()
        return self

    @torch.no_grad()
    def _encode_targets(self, U: torch.Tensor) -> torch.Tensor:
        """U: (B, Nt, N, N) → ẑ_true: (B, K, D) complex"""
        B, Nt, N, _ = U.shape
        frames   = U.reshape(B * Nt, 1, N, N)
        t        = torch.arange(Nt, dtype=U.dtype, device=U.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        chunks_f = frames.split(256)
        chunks_t = t_ratios.split(256)
        z = torch.cat([self.encoder(f, t) for f, t in zip(chunks_f, chunks_t)])
        z = z.view(B, Nt, self.latent_dim)
        return self.laplace.forward_transform(z)

    def _predict_z_hat(self, theta_norm: torch.Tensor) -> torch.Tensor:
        D   = self.latent_dim
        out = self.surrogate(theta_norm)
        return torch.complex(out[..., :D], out[..., D:])

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Nt, D) → U_rec: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t        = torch.arange(Nt, dtype=z.dtype, device=z.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        if self.training:
            chunks_z = flat.split(256)
            chunks_t = t_ratios.split(256)
            U = torch.cat([
                grad_ckpt(self.decoder, zc, tc, use_reentrant=False)
                for zc, tc in zip(chunks_z, chunks_t)
            ])
        else:
            U = self.decoder(flat, t_ratios)
        N = U.shape[-1]
        return U.view(B, Nt, N, N)

    def forward(
        self, theta_norm: torch.Tensor, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_hat_true = self._encode_targets(U)
        z_hat_pred = self._predict_z_hat(theta_norm)
        z_tilde    = self.laplace.inverse_transform(z_hat_pred, self.Nt)
        U_rec      = self._decode_seq(z_tilde)
        return U_rec, z_hat_pred, z_hat_true

    def loss(
        self,
        U          : torch.Tensor,
        U_rec      : torch.Tensor,
        z_hat_pred : torch.Tensor,
        z_hat_true : torch.Tensor,
        alpha_lat  : float = 1.0,
        alpha_spat : float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        lat_loss  = (z_hat_pred - z_hat_true.detach()).abs().pow(2).mean()
        spat_loss = F.mse_loss(U_rec, U)
        total     = alpha_lat * lat_loss + alpha_spat * spat_loss
        return total, {'lat': lat_loss.detach(), 'spat': spat_loss.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        z_hat   = self._predict_z_hat(theta_norm)
        z_tilde = self.laplace.inverse_transform(z_hat, self.Nt)
        return self._decode_seq(z_tilde)
