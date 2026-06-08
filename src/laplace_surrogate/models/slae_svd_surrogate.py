"""
slae_svd_surrogate.py — SVD Surrogate SLAE : θ → G_k_SVD → z_k → Laplace⁻¹ → U(t).

Variante surrogate du pipeline SLAE avec compression SVD des latents Laplace.
Les latents z_k ∈ R^(K × D) sont déjà dans le domaine de Laplace (réels) ;
la compression SVD est donc appliquée directement sans transformée Laplace
supplémentaire sur les coefficients SVD.

Pipeline (phase 2 offline) :
  z_k = encoder_SLAE(Laplace(U(t)))  [ns, K, D]  latents Laplace réels
  SVD tronquée sur z_train → V       [D, k_svd]   (calculée sur toutes les fréquences)
  G_k = z_k @ V                      [ns, K, k_svd]
  stats(G_k) → G_mean, G_std         [K, k_svd]

Pipeline (entraînement) :
  θ → FreqSurrogate → G_k_norm [B, K, k_svd]
  → dénorm → G_k [B, K, k_svd]
  → @ V.T → z_k [B, K, D]
  → LaplaceDecoder → frames Laplace [B, K, 2, N, N]
  → Laplace⁻¹ → Û(t) [B, Nt, N, N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.slae import LaplaceEncoder, LaplaceDecoder
from laplace_surrogate.models.surrogate_base import FreqSurrogate
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


class SLAESVDModel(nn.Module):
    """
    Surrogate SLAE avec compression SVD des latents du domaine Laplace.

    Paramètres appris :
      V              [latent_dim, k_svd]  base SVD (initialisée offline, fine-tunée)
      surrogate      FreqSurrogate  θ → G_k_norm [B, K, k_svd]
      shared_decoder LaplaceDecoder (poids copiés depuis SLAE, optionnellement fine-tuné)
    """

    def __init__(
        self,
        K           : int,
        Nt          : int,
        N           : int,
        theta_dim   : int,
        latent_dim  : int,
        k_svd       : int,
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
        self.K           = K
        self.Nt          = Nt
        self.N           = N
        self.theta_dim   = theta_dim
        self.latent_dim  = latent_dim
        self.k_svd       = k_svd
        self.hidden_dim  = hidden_dim
        self.head_dim    = head_dim
        self.n_trunk     = n_trunk
        self.n_head      = n_head
        self.freq_L      = freq_L
        self.surr_freq_L = surr_freq_L

        self.V = nn.Parameter(torch.zeros(latent_dim, k_svd))

        self.register_buffer('G_mean',   torch.zeros(K, k_svd))
        self.register_buffer('G_std',    torch.ones( K, k_svd))
        self.register_buffer('U_mean',   torch.zeros(N, N))
        self.register_buffer('U_std',    torch.ones( N, N))
        self.register_buffer('lap_mean', torch.zeros(K, 2, N, N))
        self.register_buffer('lap_std',  torch.ones( K, 2, N, N))

        self.surrogate = FreqSurrogate(
            theta_dim=theta_dim, out_dim=k_svd, K=K,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=surr_freq_L,
        )
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.shared_decoder.requires_grad_(False)
        self.encoder = LaplaceEncoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.encoder.requires_grad_(False)

        self.laplace = LearnableLaplace(K=K, dt=dt, Nt=Nt, learnable=False,
                                        alpha_t=alpha_t, lam=lam)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.laplace.eval()
        return self

    @classmethod
    def from_ae(cls, ae, latent_dim: int, freq_L: int, k_svd: int, **kwargs) -> 'SLAESVDModel':
        """Construit depuis un AE SLAE entraîné et copie les poids encodeur/décodeur."""
        model = cls(latent_dim=latent_dim, freq_L=freq_L, k_svd=k_svd, N=ae.decoder.N, **kwargs)
        model.shared_decoder.load_state_dict(ae.decoder.state_dict())
        model.shared_decoder.requires_grad_(False)
        model.encoder.load_state_dict(ae.encoder.state_dict())
        model.encoder.requires_grad_(False)
        return model

    # ------------------------------------------------------------------

    def _decode_z_k(self, z_k: torch.Tensor) -> torch.Tensor:
        """
        z_k: (B, K, latent_dim) → U_phys: (B, Nt, N, N)
        Décode les latents Laplace en frames spatiales, dénormalise, puis inverse Laplace.
        """
        B  = z_k.shape[0]
        NN = self.N * self.N
        z_flat = z_k.permute(1, 0, 2).reshape(self.K * B, self.latent_dim)
        fr = torch.tensor(
            [k / max(self.K - 1, 1) for k in range(self.K)],
            device=z_k.device, dtype=z_k.dtype,
        ).repeat_interleave(B)
        preds = self.shared_decoder(z_flat.float(), fr)           # (K*B, 2, N, N) normalisé Laplace
        preds = preds.view(self.K, B, 2, self.N, self.N)
        preds = preds * self.lap_std.unsqueeze(1) + self.lap_mean.unsqueeze(1)  # dénorm
        M = torch.complex(
            preds[:, :, 0].reshape(self.K, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(self.K, B, NN).permute(1, 2, 0).float(),
        )
        U = self.laplace.inverse_transform(M.permute(0, 2, 1), self.Nt)
        return U.reshape(B, self.Nt, self.N, self.N)

    def _forward_train(self, theta_norm: torch.Tensor):
        """θ_norm → (U_pred_norm, G_k_norm)"""
        G_k_norm = self.surrogate(theta_norm)                     # (B, K, k_svd)
        G_k      = G_k_norm * self.G_std + self.G_mean
        z_k      = G_k @ self.V.T                                 # (B, K, latent_dim)
        U_phys   = self._decode_z_k(z_k)
        U_norm   = (U_phys - self.U_mean) / self.U_std
        return U_norm, G_k_norm

    @torch.no_grad()
    def _encode_targets(self, U_norm: torch.Tensor) -> torch.Tensor:
        """U_norm: (B, Nt, N, N) → z_true: (B, K, latent_dim)"""
        B, Nt, N, _ = U_norm.shape
        u_phys  = U_norm * self.U_std + self.U_mean
        u_flat  = u_phys.reshape(B, Nt, N * N).float()
        u_hat   = self.laplace.forward_transform(u_flat)          # (B, K, N²) complexe
        re      = u_hat.real.reshape(B, self.K, N, N)
        im      = u_hat.imag.reshape(B, self.K, N, N)
        frames  = torch.stack([re, im], dim=2).reshape(B * self.K, 2, N, N)
        lap_mean = self.lap_mean.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.K, 2, N, N)
        lap_std  = self.lap_std.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.K, 2, N, N)
        frames   = (frames - lap_mean) / lap_std
        fr = torch.tensor(
            [k / max(self.K - 1, 1) for k in range(self.K)],
            device=U_norm.device, dtype=U_norm.dtype,
        ).unsqueeze(0).expand(B, -1).reshape(B * self.K)
        z = self.encoder(frames, fr)
        return z.view(B, self.K, self.latent_dim)

    # ------------------------------------------------------------------

    def forward(self, theta_norm: torch.Tensor, U_norm: torch.Tensor):
        """
        theta_norm : (B, theta_dim)
        U_norm     : (B, Nt, N, N) normalisé temporel
        retourne   : (U_pred_norm, G_k_norm, G_true_norm)
        """
        U_pred_norm, G_k_norm = self._forward_train(theta_norm)
        z_true = self._encode_targets(U_norm)
        with torch.no_grad():
            G_true      = z_true.float() @ self.V.detach()
            G_true_norm = (G_true - self.G_mean) / self.G_std
        return U_pred_norm, G_k_norm, G_true_norm

    def loss(
        self,
        U_true_norm  : torch.Tensor,
        U_pred_norm  : torch.Tensor,
        G_k_norm     : torch.Tensor,
        G_true_norm  : torch.Tensor,
        alpha_lat    : float = 1.0,
        alpha_spat   : float = 1.0,
    ):
        spat  = F.mse_loss(U_pred_norm.float(), U_true_norm.float())
        lat   = F.mse_loss(G_k_norm.float(), G_true_norm.float())
        total = alpha_spat * spat + alpha_lat * lat
        return total, {'spat': spat.detach(), 'lat': lat.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """θ_norm → U physique (B, Nt, N, N)."""
        G_k_norm = self.surrogate(theta_norm)
        G_k      = G_k_norm * self.G_std + self.G_mean
        z_k      = G_k @ self.V.T
        return self._decode_z_k(z_k)
