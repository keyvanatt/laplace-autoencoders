"""
dlrom_surrogate.py — Surrogate DL-ROM : θ → z(t) → décodeur → U(t).

Même FreqSurrogate que les pipelines Laplace, mais avec K = Nt heads : chaque
head prédit z(t) pour un pas de temps, conditionnée par le ratio t/(Nt-1)
(le même que celui du FiLM du décodeur). Pas de transformée de Laplace.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from laplace_surrogate.models.surrogate_base import FreqSurrogate


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class DLROMModel(nn.Module):
    """
    Surrogate end-to-end DL-ROM.

    Trainable : FreqSurrogate (θ → z_pred(t)) + ConvDecoder (z → U)
    Gelé      : ConvEncoder (toujours)

    Constructeur : DLROMModel(ae, theta_dim, hidden_dim, head_dim, n_trunk, n_head, freq_L)
    Les paramètres de l'AE (latent_dim, Nt) sont lus depuis l'objet ae.
    """

    def __init__(
        self,
        ae,
        theta_dim  : int,
        hidden_dim : int = 512,
        head_dim   : int = 256,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        D  = ae.latent_dim
        Nt = ae.Nt
        self.latent_dim = D
        self.Nt         = Nt
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.n_trunk    = n_trunk
        self.n_head     = n_head
        self.freq_L     = freq_L

        # K = Nt : une head par pas de temps, conditionnée par t/(Nt-1)
        self.surrogate = FreqSurrogate(
            theta_dim=theta_dim, out_dim=D, K=Nt,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
        )
        # load_dlrom_from_ckpt gèle tout l'AE : il faut redégeler le décodeur,
        # entraînable ici (comme dans LLAEModel).
        self.decoder = copy.deepcopy(ae.decoder)
        for p in self.decoder.parameters():
            p.requires_grad_(True)
        self.encoder = _freeze(copy.deepcopy(ae.encoder))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _make_t_ratios(self, B: int, dtype, device) -> torch.Tensor:
        t = torch.arange(self.Nt, dtype=dtype, device=device) / max(self.Nt - 1, 1)
        return t.unsqueeze(0).expand(B, -1).reshape(B * self.Nt, 1)

    @torch.no_grad()
    def _encode_targets(self, U: torch.Tensor) -> torch.Tensor:
        """U: (B, Nt, N, N) → z_true: (B, Nt, D)"""
        B, Nt, N, _ = U.shape
        frames   = U.reshape(B * Nt, 1, N, N)
        t_ratios = self._make_t_ratios(B, U.dtype, U.device)
        chunks_f = frames.split(256)
        chunks_t = t_ratios.split(256)
        z = torch.cat([self.encoder(f, t) for f, t in zip(chunks_f, chunks_t)])
        return z.view(B, Nt, self.latent_dim)

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Nt, D) → U_rec: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._make_t_ratios(B, z.dtype, z.device)
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
        z_true = self._encode_targets(U)
        z_pred = self.surrogate(theta_norm)          # (B, Nt, D)
        U_rec  = self._decode_seq(z_pred)
        return U_rec, z_pred, z_true

    def loss(
        self,
        U          : torch.Tensor,
        U_rec      : torch.Tensor,
        z_pred     : torch.Tensor,
        z_true     : torch.Tensor,
        alpha_lat  : float = 1.0,
        alpha_spat : float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        lat_loss  = F.mse_loss(z_pred, z_true.detach())
        spat_loss = F.mse_loss(U_rec, U)
        total     = alpha_lat * lat_loss + alpha_spat * spat_loss
        return total, {'lat': lat_loss.detach(), 'spat': spat_loss.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        z_pred = self.surrogate(theta_norm)
        return self._decode_seq(z_pred)
