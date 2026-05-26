"""
slae_surrogate.py — Surrogate θ → latents Laplace pour SLAE (SLAEModel).

SLAESurrogate : trunk MLP partagé + K heads fréquentiels, sortie réelle z_k ∈ R^latent_dim.
SLAEModel     : surrogate + shared LaplaceDecoder gelé + LearnableLaplace.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace_surrogate.models.base import BaseDecoder
from laplace_surrogate.models.slae import LaplaceDecoder
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """n_layers=2 → Linear+LN+GELU → Linear ; n_layers=4 → trois couches cachées."""
    layers = []
    for i in range(n_layers):
        d_in  = in_dim    if i == 0            else hidden_dim
        d_out = out_dim   if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class SLAESurrogate(nn.Module):
    """
    Trunk MLP partagé (θ→h) + K heads par fréquence (h ⊕ freq_enc → z_k).

    Sortie réelle : z_k ∈ R^latent_dim (le décodeur AE est conditionné sur
    freq_ratio via FiLM, pas via le latent lui-même).

    Paramètres
    ----------
    K          : nombre de fréquences de Laplace
    theta_dim  : dimension de θ
    latent_dim : dimension de z_k (sortie de chaque head)
    hidden_dim : largeur du trunk
    head_dim   : largeur cachée de chaque head
    n_trunk    : nombre de couches Linear dans le trunk
    n_head     : nombre de couches Linear dans chaque head
    freq_L     : niveaux sinusoïdaux pour l'encoding fréquentiel
    """

    def __init__(
        self,
        K          : int,
        theta_dim  : int,
        latent_dim : int,
        hidden_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        self.K         = K
        self.freq_L    = freq_L
        freq_cond      = 2 * freq_L

        trunk_layers: list[nn.Module] = []
        for i in range(n_trunk):
            d_in = theta_dim if i == 0 else hidden_dim
            trunk_layers += [nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            _make_mlp(hidden_dim + freq_cond, head_dim, latent_dim, n_head)
            for _ in range(K)
        ])

        freq_ratios = torch.arange(K).float() / max(K - 1, 1)
        self.register_buffer('_freq_ratios', freq_ratios)

    def _sinenc(self, ratios: torch.Tensor) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.freq_L, device=ratios.device, dtype=ratios.dtype)) * torch.pi
        x = ratios[:, None] * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=1)

    def forward(self, theta: torch.Tensor, n_active: int | None = None) -> torch.Tensor:
        """theta: (B, theta_dim) → (B, n_active, latent_dim)"""
        if n_active is None:
            n_active = self.K
        h        = self.trunk(theta)
        freq_enc = self._sinenc(self._freq_ratios[:n_active])
        B = h.shape[0]
        outs = []
        for k in range(n_active):
            e_k = freq_enc[k].unsqueeze(0).expand(B, -1)
            outs.append(self.heads[k](torch.cat([h, e_k], dim=1)))
        return torch.stack(outs, dim=1)


class SLAEModel(BaseDecoder):
    """
    Modèle complet SLAE-surrogate :
      - un SLAESurrogate (trunk partagé + K heads)
      - un unique shared_decoder (LaplaceDecoder gelé, initialisé depuis le SLAE)

    Correspond au surrogate end-to-end décrit dans l'article (section SLAE surrogate).
    """

    def __init__(
        self,
        K          : int,
        Nt         : int,
        N          : int,
        theta_dim  : int       = 4,
        latent_dim : int       = 64,
        hidden_dim : int       = 256,
        head_dim   : int       = 128,
        n_trunk    : int       = 4,
        n_head     : int       = 2,
        freq_L     : int       = 8,
        surr_freq_L: int       = 6,
        k_max      : int | None = None,
        dt         : float     = 1.0,
        gamma_init : float     = 1e-2,
        alpha_t    : float     = 0.007,
        lam        : float     = 3e-5,
        learnable_laplace: bool = False,
    ):
        super().__init__()

        self.surrogate = SLAESurrogate(
            K=K, theta_dim=theta_dim, latent_dim=latent_dim,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=surr_freq_L,
        )
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.shared_decoder.requires_grad_(False)

        self.K          = K
        self.Nt         = Nt
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.n_trunk    = n_trunk
        self.n_head     = n_head
        self.freq_L     = freq_L
        self.surr_freq_L = surr_freq_L
        self.k_max      = k_max

        self.register_buffer('U_mean', torch.zeros(N, N))
        self.register_buffer('U_std',  torch.ones(N, N))
        self.register_buffer('s_real', torch.zeros(K, dtype=torch.float64))
        self.register_buffer('s_imag', torch.zeros(K, dtype=torch.float64))

        self.laplace = LearnableLaplace(
            K=K, dt=dt, Nt=Nt, gamma_init=gamma_init,
            learnable=learnable_laplace,
            alpha_t=alpha_t, lam=lam,
        )

    def set_ae_decoder(self, ae):
        """Charge les poids du décodeur AE (SLAE) dans shared_decoder."""
        self.shared_decoder.load_state_dict(ae.decoder.state_dict())
        self.shared_decoder.requires_grad_(False)

    def _forward_k_diff(self, theta_norm: torch.Tensor, n_active: int) -> torch.Tensor:
        """
        Un seul forward trunk, puis heads + décodeur batchés.
        Retourne M_active : (B, N*N, n_active) complex64.
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N

        z_active = self.surrogate(theta_norm, n_active)
        z_flat   = z_active.permute(1, 0, 2).reshape(n_active * B, self.latent_dim)

        fr = torch.tensor(
            [k / max(self.K - 1, 1) for k in range(n_active)],
            device=theta_norm.device, dtype=torch.float32,
        ).repeat_interleave(B)

        preds = self.shared_decoder(z_flat.float(), fr)
        preds = preds.view(n_active, B, 2, self.N, self.N)
        return torch.complex(
            preds[:, :, 0].reshape(n_active, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(n_active, B, NN).permute(1, 2, 0).float(),
        )

    def _generate(self, theta_norm: torch.Tensor, **kwargs) -> torch.Tensor:
        B      = theta_norm.shape[0]
        NN     = self.N ** 2
        device = theta_norm.device

        limit    = self.k_max
        n_active = min(limit + 1, self.K) if limit is not None else self.K
        M_active = self._forward_k_diff(theta_norm, n_active)

        if n_active < self.K:
            pad = torch.zeros(B, NN, self.K - n_active, dtype=torch.complex64, device=device)
            M   = torch.cat([M_active, pad], dim=2)
        else:
            M = M_active

        U_norm = self.laplace.inverse_transform(M.permute(0, 2, 1), self.Nt)
        U_norm = U_norm.reshape(B, self.Nt, self.N, self.N)
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
            f"  surrogate : trunk({self.n_trunk}×{self.hidden_dim}, GELU) "
            f"+ {self.K} heads({self.n_head}×{self.head_dim}) "
            f"surr_freq_L={self.surr_freq_L}  [{n_surr:,} params]\n"
            f"  decoder   : LaplaceDecoder freq_L={self.freq_L}  [{n_dec:,} params, gelé]"
        )
