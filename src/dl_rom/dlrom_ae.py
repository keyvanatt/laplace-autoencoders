"""
dlrom_ae.py — Autoencoder DL-ROM (phase 1 de la baseline).

Pipeline : U(t) → ConvEncoder(t_ratio) → z(t) → ConvDecoder(t_ratio) → Û(t)

Identique au LLAE (mêmes ConvEncoder/ConvDecoder, même conditionnement FiLM
sur t_ratio = t / (Nt-1)), mais sans transformée de Laplace : la reconstruction
est frame par frame dans le domaine temporel.
"""
import torch
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseAutoEncoder
from laplace_surrogate.models.encoder_decoder import ConvEncoder, ConvDecoder


class DLROMAE(BaseAutoEncoder):
    """
    Autoencoder DL-ROM.

    Paramètres
    ----------
    N            : résolution spatiale (multiple de 8)
    Nt           : nombre de pas de temps
    latent_dim   : dimension de l'espace latent z
    beta         : poids ridge sur z (régularisation latente)
    time_L       : niveaux de fréquence pour l'encoding sinusoïdal du temps
    decoder_norm : normalisation des blocs deconv ('gn' ou 'bn')
    """

    def __init__(
        self,
        N            : int   = 128,
        Nt           : int   = 150,
        latent_dim   : int   = 64,
        beta         : float = 1e-2,
        time_L       : int   = 8,
        decoder_norm : str   = 'gn',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.Nt         = Nt
        self.beta       = beta

        self.encoder = ConvEncoder(in_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)
        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L,
                                   norm=decoder_norm)

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

    def forward(self, U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        U : (B, Nt, N, N)
        retourne : (U_rec, z)
        """
        z     = self._encode_seq(U)
        U_rec = self._decode_seq(z)
        return U_rec, z

    def loss(
        self,
        U    : torch.Tensor,
        U_rec: torch.Tensor,
        z    : torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        recon = F.mse_loss(U_rec, U)
        ridge = z.pow(2).mean()
        total = recon + self.beta * ridge
        return total, {
            'recon': recon.detach(),
            'ridge': ridge.detach(),
        }
