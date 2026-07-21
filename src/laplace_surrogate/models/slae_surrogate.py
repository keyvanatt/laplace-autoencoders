"""
slae_surrogate.py — Surrogate SLAE : θ → z_k (réel) → décodeur spatial → Laplace⁻¹ → U(t).
"""
import torch
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseDecoder
from laplace_surrogate.models.slae import LaplaceEncoder, LaplaceDecoder
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

        self.encoder = LaplaceEncoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.encoder.requires_grad_(False)

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

        self.register_buffer('U_mean',   torch.zeros(N, N))
        self.register_buffer('U_std',    torch.ones(N, N))
        # Normalisation Laplace par fréquence, pixel par pixel (K, 2, N, N)
        self.register_buffer('lap_mean', torch.zeros(K, 2, N, N))
        self.register_buffer('lap_std',  torch.ones(K, 2, N, N))

        self.laplace = LearnableLaplace(
            K=K, dt=dt, Nt=Nt,
            learnable=False, alpha_t=alpha_t, lam=lam,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.laplace.eval()
        return self

    @classmethod
    def from_ae(cls, ae, latent_dim: int, freq_L: int, **kwargs) -> 'SLAEModel':
        """Construit depuis un AE entraîné et copie les poids de l'encodeur et du décodeur."""
        model = cls(latent_dim=latent_dim, freq_L=freq_L, N=ae.decoder.N, **kwargs)
        model.shared_decoder.load_state_dict(ae.decoder.state_dict())
        model.shared_decoder.requires_grad_(False)
        model.encoder.load_state_dict(ae.encoder.state_dict())
        model.encoder.requires_grad_(False)
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

        preds = self.shared_decoder(z_flat.float(), fr)           # (K*B, 2, N, N) — espace Laplace normalisé
        preds = preds.view(self.K, B, 2, self.N, self.N)

        # Dénormalise dans le domaine Laplace → frames physiques
        preds = preds * self.lap_std.unsqueeze(1) + self.lap_mean.unsqueeze(1)  # (K, B, 2, N, N)

        M = torch.complex(
            preds[:, :, 0].reshape(self.K, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(self.K, B, NN).permute(1, 2, 0).float(),
        )
        return M, z_pred

    def _forward_train(self, theta_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward d'entraînement : retourne (U_norm, z_pred) normalisé réel.
        M (physique) → L⁻¹ → U physique → normalise réel → U_norm.
        """
        M, z_pred = self._forward_k(theta_norm)
        U_phys    = self.laplace.inverse_transform(M.permute(0, 2, 1), self.Nt)
        U_phys    = U_phys.reshape(theta_norm.shape[0], self.Nt, self.N, self.N)
        U_norm    = (U_phys - self.U_mean) / self.U_std
        return U_norm, z_pred

    @torch.no_grad()
    def _encode_targets(self, u_norm: torch.Tensor) -> torch.Tensor:
        """u_norm : (B, Nt, N, N) normalisé réel → z_true : (B, K, latent_dim)
        Récupère U physique, applique Laplace, normalise dans le domaine Laplace.
        """
        B, Nt, N, _ = u_norm.shape
        u_phys = u_norm * self.U_std + self.U_mean                        # (B, Nt, N, N)
        u_flat = u_phys.reshape(B, Nt, N * N).float()
        u_hat  = self.laplace.forward_transform(u_flat)                   # (B, K, N²) complex

        re     = u_hat.real.reshape(B, self.K, N, N)
        im     = u_hat.imag.reshape(B, self.K, N, N)
        frames = torch.stack([re, im], dim=2).reshape(B * self.K, 2, N, N)

        # Normalise dans le domaine Laplace — même espace que l'encodeur AE
        lap_mean = self.lap_mean.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.K, 2, N, N)
        lap_std  = self.lap_std.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.K, 2, N, N)
        frames   = (frames - lap_mean) / lap_std

        fr = torch.tensor(
            [k / max(self.K - 1, 1) for k in range(self.K)],
            device=u_norm.device, dtype=u_norm.dtype,
        ).unsqueeze(0).expand(B, -1).reshape(B * self.K)
        z = self.encoder(frames, fr)
        return z.view(B, self.K, self.latent_dim)

    def forward(
        self, theta_norm: torch.Tensor, U_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        theta_norm : (B, theta_dim)
        U_norm     : (B, Nt, N, N) normalisé réel
        retourne   : (u_pred_norm, z_pred, z_true)
        """
        u_pred_norm, z_pred = self._forward_train(theta_norm)
        z_true = self._encode_targets(U_norm)
        return u_pred_norm, z_pred, z_true

    def loss(
        self,
        u_true_norm : torch.Tensor,
        u_pred_norm : torch.Tensor,
        z_pred      : torch.Tensor,
        z_true      : torch.Tensor,
        alpha_lat   : float = 1.0,
        alpha_spat  : float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        spat_loss = F.mse_loss(u_pred_norm.float(), u_true_norm.float())
        lat_loss  = F.mse_loss(z_pred.float(), z_true.float())
        total     = alpha_spat * spat_loss + alpha_lat * lat_loss
        return total, {'spat': spat_loss.detach(), 'lat': lat_loss.detach()}

    def _generate(self, theta_norm: torch.Tensor, Nt: int | None = None,
                  rescale_reg: bool = True, **kwargs) -> torch.Tensor:
        # M est déjà en espace Laplace physique (dénormalisé dans _forward_k)
        # → inverse Laplace donne directement U physique.
        # Nt (optionnel) choisit le nombre de frames temporelles en sortie : la transformée
        # inverse rééchantillonne U(t) sur Nt points à horizon T constant. rescale_reg
        # recalibre la régularisation Laplace sur la nouvelle grille (cf. inverse_transform).
        Nt   = self.Nt if Nt is None else Nt
        M, _ = self._forward_k(theta_norm)
        U    = self.laplace.inverse_transform(M.permute(0, 2, 1), Nt, rescale_reg=rescale_reg)
        return U.reshape(theta_norm.shape[0], Nt, self.N, self.N)

    def _generate_diff(self, theta_norm: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._generate(theta_norm)

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
