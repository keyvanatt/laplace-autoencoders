"""
llae_surrogate.py — Surrogate LLAE : θ → ẑ_k (complexe) → Laplace⁻¹ → z(t) → décodeur → U(t).
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


class LLAEModel(nn.Module):
    """
    Surrogate end-to-end LLAE.

    Trainable : FreqSurrogate (θ → ẑ_pred) + ConvDecoder (z̃ → U)
                + optionnellement LearnableLaplace si learnable_laplace=True
    Gelé      : ConvEncoder (toujours)

    Constructeur : LLAEModel(ae, theta_dim, hidden_dim, head_dim, n_trunk, n_head, freq_L,
                              learnable_laplace)
    Les paramètres de l'AE (latent_dim, K, Nt) sont lus depuis l'objet ae.
    """

    def __init__(
        self,
        ae,
        theta_dim         : int,
        hidden_dim        : int  = 512,
        head_dim          : int  = 256,
        n_trunk           : int  = 4,
        n_head            : int  = 2,
        freq_L            : int  = 6,
        learnable_laplace : bool = False,
    ):
        super().__init__()
        D  = ae.latent_dim
        K  = ae.laplace.K
        self.latent_dim       = D
        self.K                = K
        self.Nt               = ae.Nt
        self.hidden_dim       = hidden_dim
        self.head_dim         = head_dim
        self.n_trunk          = n_trunk
        self.n_head           = n_head
        self.freq_L           = freq_L
        self.learnable_laplace = learnable_laplace

        self.surrogate = FreqSurrogate(
            theta_dim=theta_dim, out_dim=2 * D, K=K,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
        )
        # load_llae_from_ckpt gèle tout l'AE : il faut redégeler le décodeur, qui est
        # entraînable ici (sinon lr_decoder est sans effet et il reste figé en phase 1).
        self.decoder = copy.deepcopy(ae.decoder)
        for p in self.decoder.parameters():
            p.requires_grad_(True)
        self.encoder = _freeze(copy.deepcopy(ae.encoder))
        self.laplace = copy.deepcopy(ae.laplace)
        if learnable_laplace:
            self.laplace.learnable = True   # garantit que les caches sont invalidés à chaque step
            for p in self.laplace.parameters():
                p.requires_grad_(True)
        else:
            _freeze(self.laplace)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if not self.learnable_laplace:
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
    def generate(self, theta_norm: torch.Tensor, Nt: int | None = None,
                 rescale_reg: bool = True) -> torch.Tensor:
        # Nt (optionnel) choisit le nombre de frames temporelles en sortie : la transformée
        # inverse rééchantillonne z(t) sur Nt points à horizon T constant, et le décodeur
        # (conditionné sur t/T) reconstruit chaque frame. rescale_reg recalibre la
        # régularisation Laplace sur la nouvelle grille (cf. inverse_transform).
        Nt      = self.Nt if Nt is None else Nt
        z_hat   = self._predict_z_hat(theta_norm)
        z_tilde = self.laplace.inverse_transform(z_hat, Nt, rescale_reg=rescale_reg)
        return self._decode_seq(z_tilde)
