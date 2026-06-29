"""
llae_tucker_surrogate.py — Tucker Surrogate LLAE : θ → Ĝ_core → ẑ(s_k) → z(t) → U(t).

Variante surrogate du pipeline LLAE avec compression de Tucker des latents Laplace.
Comme LLAE-SVD, ce n'est pas un AE indépendant : il réutilise l'encodeur et le
décodeur d'un LLAE pré-entraîné. Contrairement à SVD (une seule base latente V
appliquée AVANT Laplace), la décomposition de Tucker est appliquée APRÈS la
transformée de Laplace, sur le tenseur complexe Ẑ ∈ (ns, K, d_z), et factorise
conjointement le mode fréquence (K → r_s) et le mode latent (d_z → r_z).

Les facteurs U_s [K, r_s] et U_z [d_z, r_z] sont complexes, calculés offline par
HOOI et FIGÉS (buffers, non entraînés) — ce qui préserve leur orthonormalité.

Pipeline (phase 2 offline) :
  z(t) = encoder_LLAE(U(t))          [ns, Nt, d_z]
  Ẑ(s_k) = Laplace(z(t))            [ns, K, d_z] complexe
  HOOI(Ẑ_train) → U_s, U_z          (figés)
  G = U_sᴴ Ẑ U_z                    [ns, r_s, r_z] complexe (core)
  stats(G) → G_hat_mean, G_hat_std   [r_s, r_z, 2]

Pipeline (entraînement) :
  θ → FreqSurrogate → Ĝ_norm [B, r_s, r_z, 2]
  → dénorm → Ĝ [B, r_s, r_z] complexe
  → U_s Ĝ U_zᴴ → ẑ(s_k) [B, K, d_z] complexe
  → Laplace⁻¹ → z̃(t) [B, Nt, d_z]
  → ConvDecoder → Û(t) [B, Nt, N, N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.encoder_decoder import ConvDecoder
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace
from laplace_surrogate.models.surrogate_base import FreqSurrogate
from laplace_surrogate.models import tucker


class LLAETuckerModel(nn.Module):
    """
    Surrogate LLAE avec compression de Tucker des latents du domaine Laplace.

    Buffers (figés) :
      U_s [K, r_s]   facteur fréquence  (complexe)
      U_z [d_z, r_z] facteur latent     (complexe)
    Paramètres appris :
      surrogate  FreqSurrogate  θ → Ĝ_norm [B, r_s, r_z*2]
      decoder    ConvDecoder (poids copiés depuis LLAE, fine-tunés)
    """

    def __init__(
        self,
        N          : int   = 128,
        Nt         : int   = 150,
        theta_dim  : int   = 3,
        latent_dim : int   = 64,
        r_s        : int   = 8,
        r_z        : int   = 16,
        K          : int   = 16,
        dt         : float = 1.0,
        time_L     : int   = 8,
        hidden_dim : int   = 512,
        head_dim   : int   = 256,
        n_trunk    : int   = 4,
        n_head     : int   = 2,
        freq_L     : int   = 6,
    ):
        super().__init__()
        self.N          = N
        self.Nt         = Nt
        self.r_s        = r_s
        self.r_z        = r_z
        self.latent_dim = latent_dim
        self.K          = K
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.n_trunk    = n_trunk
        self.n_head     = n_head
        self.freq_L     = freq_L

        self.laplace = LearnableLaplace(K, dt, Nt, learnable=False)

        # Facteurs de Tucker figés (complexes)
        self.register_buffer('U_s', torch.zeros(K, r_s, dtype=torch.cfloat))
        self.register_buffer('U_z', torch.zeros(latent_dim, r_z, dtype=torch.cfloat))

        self.register_buffer('G_hat_mean', torch.zeros(r_s, r_z, 2))
        self.register_buffer('G_hat_std',  torch.ones( r_s, r_z, 2))
        self.register_buffer('U_mean',     torch.zeros(N, N))
        self.register_buffer('U_std',      torch.ones( N, N))
        self.register_buffer('theta_mean', torch.zeros(theta_dim))
        self.register_buffer('theta_std',  torch.ones( theta_dim))

        self.surrogate = FreqSurrogate(
            theta_dim=theta_dim, out_dim=r_z * 2, K=r_s,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
        )
        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)

    # ------------------------------------------------------------------

    def set_tucker_factors(self, U_s: torch.Tensor, U_z: torch.Tensor):
        with torch.no_grad():
            self.U_s.copy_(U_s.to(torch.cfloat))
            self.U_z.copy_(U_z.to(torch.cfloat))

    def set_normalization(self, G_hat_mean, G_hat_std, U_mean, U_std, theta_mean, theta_std):
        def _t(x):
            return x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.G_hat_mean.copy_(_t(G_hat_mean))
        self.G_hat_std.copy_( _t(G_hat_std))
        self.U_mean.copy_(    _t(U_mean))
        self.U_std.copy_(     _t(U_std))
        self.theta_mean.copy_(_t(theta_mean))
        self.theta_std.copy_( _t(theta_std))

    def load_ae_decoder(self, ae):
        """Copie les poids du décodeur depuis un LLAE entraîné."""
        self.decoder.load_state_dict(ae.decoder.state_dict())

    def load_laplace_from_ae(self, ae):
        """Copie les paramètres Laplace depuis un LLAE (pôles, alpha_t, lam)."""
        self.laplace.s_re.data.copy_(ae.laplace.s_re.data)
        self.laplace.s_im.data.copy_(ae.laplace.s_im.data)
        self.laplace.log_alpha_t.data.copy_(ae.laplace.log_alpha_t.data)
        self.laplace.log_lam.data.copy_(ae.laplace.log_lam.data)
        self.laplace._eval_cache  = None
        self.laplace._F_fwd_cache = None

    # ------------------------------------------------------------------

    def _t_ratios(self, Nt, B, dtype, device):
        t = torch.arange(Nt, dtype=dtype, device=device) / max(Nt - 1, 1)
        return t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Nt, latent_dim) → U_norm: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._t_ratios(Nt, B, z.dtype, z.device)
        return self.decoder(flat, t_ratios).view(B, Nt, self.N, self.N)

    def _predict_and_reconstruct(self, theta_norm: torch.Tensor):
        """θ_norm → (U_pred_norm, G_hat_norm)"""
        B          = theta_norm.shape[0]
        G_hat_norm = self.surrogate(theta_norm).view(B, self.r_s, self.r_z, 2)
        G_hat_ri   = G_hat_norm * self.G_hat_std + self.G_hat_mean
        G_hat_phys = torch.complex(G_hat_ri[..., 0], G_hat_ri[..., 1])
        z_hat      = tucker.reconstruct(G_hat_phys, self.U_s, self.U_z)  # (B, K, d_z) complexe
        z_tilde    = self.laplace.inverse_transform(z_hat, self.Nt)
        U_pred     = self._decode_seq(z_tilde)
        return U_pred, G_hat_norm

    # ------------------------------------------------------------------

    def forward(self, theta_norm: torch.Tensor, z_true: torch.Tensor):
        """
        theta_norm : (B, theta_dim)
        z_true     : (B, Nt, latent_dim)  latents LLAE encodés (encoder gelé)
        retourne   : (U_pred_norm, G_hat_norm, G_hat_true_norm)
        """
        U_pred, G_hat_norm = self._predict_and_reconstruct(theta_norm)

        with torch.no_grad():
            z_hat_true      = self.laplace.forward_transform(z_true.float())   # (B, K, d_z)
            G_true          = tucker.project(z_hat_true, self.U_s, self.U_z)   # (B, r_s, r_z)
            G_hat_true_ri   = torch.stack([G_true.real, G_true.imag], dim=-1)
            G_hat_true_norm = (G_hat_true_ri - self.G_hat_mean) / self.G_hat_std

        return U_pred, G_hat_norm, G_hat_true_norm

    def loss(self, U_norm, U_pred, G_hat_norm, G_hat_true_norm, alpha_lat=1.0):
        spat  = F.mse_loss(U_pred, U_norm)
        lat   = F.mse_loss(G_hat_norm, G_hat_true_norm)
        total = spat + alpha_lat * lat
        return total, {'spat': spat.detach(), 'lat': lat.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """θ_norm → U physique (B, Nt, N, N)."""
        U_pred, _ = self._predict_and_reconstruct(theta_norm)
        return U_pred * self.U_std[None, None] + self.U_mean[None, None]
