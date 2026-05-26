"""
lslae.py — Latent SVD Laplace AE (LSLAE).

Alias LSLAEModel = LSLAE.

Pipeline :
  Phase 1 — LLAE pré-entraîné (inchangé).
  Phase 2 — offline :
      z(t) = encoder(U(t))  →  SVD tronquée  →  V [D, k_svd]
      G(t) = z(t) @ V  →  [n, Nt, k_svd]
      Ĝ(k) = laplace.forward_transform(G)  →  [n, K, k_svd] complex
  Phase 3 — entraîne LSLAE :
      θ → LLAESurrogate → Ĝ_norm [B, K, k_svd, 2]
      → dénorm → Ĝ_phys [B, K, k_svd] complex
      → laplace.inverse_transform → G̃(t) [B, Nt, k_svd]
      → @ V.T → z̃(t) [B, Nt, D] → ConvDecoder(t_ratio) → Û_norm(t) [B, Nt, N, N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseDecoder
from laplace_surrogate.models.encoder_decoder import ConvDecoder
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace
from laplace_surrogate.models.llae_surrogate import LLAESurrogate


class LSLAE(BaseDecoder):
    """
    Latent SVD Laplace AE (LSLAE).

    Surrogate θ → Û(t) via base SVD espace latent + transformée de Laplace régularisée.

    Paramètre appris :
      V     [latent_dim, k_svd]   base SVD (initialisée offline, fine-tunée end-to-end)
      proj  LLAESurrogate  θ_norm → Ĝ_norm [B, K, k_svd, 2]
      decoder  ConvDecoder

    Paramètres
    ----------
    N, Nt, theta_dim, latent_dim : doivent correspondre au LLAE source
    k_svd  : nombre de modes SVD retenus
    K      : nombre de fréquences Laplace (doit correspondre à l'AE source)
    dt     : pas de temps
    gamma_init : amortissement initial (écrasé par load_laplace_from_ae)
    time_L : niveaux FiLM (doit correspondre au LLAE source)
    """

    def __init__(
        self,
        N          : int   = 128,
        Nt         : int   = 150,
        theta_dim  : int   = 3,
        latent_dim : int   = 64,
        k_svd      : int   = 16,
        K          : int   = 16,
        dt         : float = 1.0,
        gamma_init : float = 1e-2,
        time_L     : int   = 8,
        shared_dim : int   = 256,
        head_dim   : int   = 128,
        n_trunk    : int   = 4,
        n_head     : int   = 2,
        freq_L     : int   = 6,
    ):
        super().__init__()
        self.N          = N
        self.Nt         = Nt
        self.k_svd      = k_svd
        self.latent_dim = latent_dim
        self.K          = K
        self.dt         = dt

        self.laplace = LearnableLaplace(K, dt, Nt, gamma_init=gamma_init, learnable=False)

        self.V = nn.Parameter(torch.zeros(latent_dim, k_svd))

        self.register_buffer('G_hat_mean', torch.zeros(K, k_svd, 2))
        self.register_buffer('G_hat_std',  torch.ones( K, k_svd, 2))
        self.register_buffer('U_mean',     torch.zeros(N, N))
        self.register_buffer('U_std',      torch.ones( N, N))
        self.register_buffer('theta_mean', torch.zeros(theta_dim))
        self.register_buffer('theta_std',  torch.ones( theta_dim))

        self.proj = LLAESurrogate(
            theta_dim=theta_dim, out_dim=k_svd * 2, K=K,
            shared_dim=shared_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
        )

        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)

    def set_svd_basis(self, V):
        """V : Tensor ou ndarray (latent_dim, k_svd). Initialise le paramètre V."""
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=torch.float32)
        with torch.no_grad():
            self.V.copy_(V.float())

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
        """Copie les poids du décodeur depuis un LLAE (entraînable)."""
        self.decoder.load_state_dict(ae.decoder.state_dict())

    def load_laplace_from_ae(self, ae):
        """Copie les paramètres Laplace (s_k, α_t, λ) depuis un LLAE et gèle."""
        self.laplace.s_re.data.copy_(ae.laplace.s_re.data)
        self.laplace.s_im.data.copy_(ae.laplace.s_im.data)
        self.laplace.log_alpha_t.data.copy_(ae.laplace.log_alpha_t.data)
        self.laplace.log_lam.data.copy_(ae.laplace.log_lam.data)
        self.laplace._eval_cache  = None
        self.laplace._F_fwd_cache = None

    def _t_ratios(self, Nt, B, dtype, device):
        t = torch.arange(Nt, dtype=dtype, device=device) / max(Nt - 1, 1)
        return t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)

    def _decode_seq(self, z):
        """z: (B, Nt, latent_dim) → U_norm: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._t_ratios(Nt, B, z.dtype, z.device)
        return self.decoder(flat, t_ratios).view(B, Nt, self.N, self.N)

    def _forward_full(self, theta_norm):
        B = theta_norm.shape[0]
        G_hat_norm = self.proj(theta_norm).view(B, self.K, self.k_svd, 2)
        G_hat_ri   = G_hat_norm * self.G_hat_std + self.G_hat_mean
        G_hat_phys = torch.complex(G_hat_ri[..., 0], G_hat_ri[..., 1])
        G_tilde    = self.laplace.inverse_transform(G_hat_phys, self.Nt)
        z_tilde    = G_tilde @ self.V.T
        U_pred     = self._decode_seq(z_tilde)
        return U_pred, G_hat_norm

    def forward(self, theta_norm, z_true):
        """
        theta_norm : (B, D_θ)
        z_true     : (B, Nt, latent_dim)  latents encodés (frozen encoder, offline)
        """
        U_pred, G_hat_norm = self._forward_full(theta_norm)

        with torch.no_grad():
            G_true     = z_true.float() @ self.V.detach()
            G_hat_true = self.laplace.forward_transform(G_true)
            G_hat_true_ri = torch.stack([G_hat_true.real, G_hat_true.imag], dim=-1)
            G_hat_true_norm = (G_hat_true_ri - self.G_hat_mean) / self.G_hat_std

        return U_pred, G_hat_norm, G_hat_true_norm

    def loss(self, U_norm, U_pred, G_hat_norm, G_hat_true_norm, alpha_lat=1.0):
        spat  = F.mse_loss(U_pred, U_norm)
        lat   = F.mse_loss(G_hat_norm, G_hat_true_norm)
        total = spat + alpha_lat * lat
        return total, {'spat': spat.detach(), 'lat': lat.detach()}

    def _generate(self, theta_norm, **kwargs):
        """theta_norm : (B, D_θ) → Û : (B, Nt, N, N) valeurs physiques."""
        U_pred, _ = self._forward_full(theta_norm)
        return U_pred * self.U_std[None, None] + self.U_mean[None, None]


LSLAEModel = LSLAE
