"""
slae_surrogate.py — Surrogate SLAE : θ → z_k (réel) → décodeur spatial → Laplace⁻¹ → U(t).
"""
import torch

from laplace_surrogate.models.base import BaseDecoder
from laplace_surrogate.models.slae import LaplaceDecoder
from laplace_surrogate.models.surrogate_base import FreqSurrogate
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


class SLAEModel(BaseDecoder):
    """
    Surrogate end-to-end SLAE.

    Constructeur standard (compatible pipeline d'inférence) :
      SLAEModel(K, Nt, N, theta_dim, latent_dim, freq_L, hidden_dim, ...)

    Constructeur depuis un AE entraîné (training) :
      SLAEModel.from_ae(ae, latent_dim, freq_L, Nt, K, theta_dim, hidden_dim, ...)
    """

    def __init__(
        self,
        K           : int,
        Nt          : int,
        N           : int,
        theta_dim   : int,
        latent_dim  : int,
        freq_L      : int,
        hidden_dim  : int   = 512,
        head_dim    : int   = 256,
        n_trunk     : int   = 4,
        n_head      : int   = 2,
        surr_freq_L : int   = 6,
        dt          : float = 1.0,
        alpha_t     : float = 0.007,
        lam         : float = 3e-5,
    ):
        super().__init__()

        self.surrogate = FreqSurrogate(
            theta_dim=theta_dim, out_dim=latent_dim, K=K,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=surr_freq_L,
        )
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.shared_decoder.requires_grad_(False)

        self.K           = K
        self.Nt          = Nt
        self.N           = N
        self.theta_dim   = theta_dim
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.head_dim    = head_dim
        self.n_trunk     = n_trunk
        self.n_head      = n_head
        self.freq_L      = freq_L
        self.surr_freq_L = surr_freq_L

        self.register_buffer('U_mean', torch.zeros(N, N))
        self.register_buffer('U_std',  torch.ones(N, N))

        self.laplace = LearnableLaplace(
            K=K, dt=dt, Nt=Nt,
            learnable=False, alpha_t=alpha_t, lam=lam,
        )

    @classmethod
    def from_ae(cls, ae, latent_dim: int, freq_L: int, **kwargs) -> 'SLAEModel':
        """
        Construit depuis un AE entraîné et copie les poids du décodeur.
        latent_dim et freq_L sont lus depuis le checkpoint AE (ckpt_utils),
        N est lu depuis ae.decoder.N.
        """
        model = cls(latent_dim=latent_dim, freq_L=freq_L, N=ae.decoder.N, **kwargs)
        model.shared_decoder.load_state_dict(ae.decoder.state_dict())
        model.shared_decoder.requires_grad_(False)
        return model

    def _forward_k(self, theta_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """θ_norm → (M, z_pred)
        M      : (B, N², K) complex64
        z_pred : (B, K, latent_dim) float32
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N

        z_pred = self.surrogate(theta_norm, self.K)              # (B, K, latent_dim)
        z_flat = z_pred.permute(1, 0, 2).reshape(self.K * B, self.latent_dim)

        fr = torch.tensor(
            [k / max(self.K - 1, 1) for k in range(self.K)],
            device=theta_norm.device, dtype=torch.float32,
        ).repeat_interleave(B)

        preds = self.shared_decoder(z_flat.float(), fr)           # (K*B, 2, N, N)
        preds = preds.view(self.K, B, 2, self.N, self.N)
        M = torch.complex(
            preds[:, :, 0].reshape(self.K, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(self.K, B, NN).permute(1, 2, 0).float(),
        )
        return M, z_pred

    def _forward_train(self, theta_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward d'entraînement : retourne (U_norm, z_pred) sans dénormalisation.
        U_norm : (B, Nt, N, N)  — champ normalisé
        z_pred : (B, K, latent_dim) — codes latents prédits (pour la loss latente)
        """
        M, z_pred = self._forward_k(theta_norm)
        U_norm = self.laplace.inverse_transform(M.permute(0, 2, 1), self.Nt)
        return U_norm.reshape(theta_norm.shape[0], self.Nt, self.N, self.N), z_pred

    def _generate(self, theta_norm: torch.Tensor, **kwargs) -> torch.Tensor:
        device = theta_norm.device
        M, _   = self._forward_k(theta_norm)
        U_norm = self.laplace.inverse_transform(M.permute(0, 2, 1), self.Nt)
        U_norm = U_norm.reshape(theta_norm.shape[0], self.Nt, self.N, self.N)
        return U_norm * self.U_std.to(device) + self.U_mean.to(device)

    def _generate_diff(self, theta_norm: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._generate(theta_norm)

    def loss(self, *_, **__):
        raise NotImplementedError

    def __repr__(self) -> str:
        n_surr = sum(p.numel() for p in self.surrogate.parameters())
        n_dec  = sum(p.numel() for p in self.shared_decoder.parameters())
        return (
            f"SLAEModel(K={self.K}, Nt={self.Nt}, N={self.N}, "
            f"theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
            f"  surrogate : trunk({self.n_trunk}×{self.hidden_dim}) "
            f"+ {self.K} heads({self.n_head}×{self.head_dim}) "
            f"surr_freq_L={self.surr_freq_L}  [{n_surr:,} params]\n"
            f"  decoder   : LaplaceDecoder freq_L={self.freq_L}  [{n_dec:,} params]"
        )
