"""
surrogate_base.py — Bloc MLP θ → latents fréquentiels, commun à tous les surrogates.

FreqSurrogate : trunk MLP partagé + K heads conditionnés par l'encoding sinusoïdal
                de la fréquence de Laplace k/(K-1) ∈ [0, 1].
                Utilisé par SLAEModel, LLAEModel et LSLAE.
"""
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_layers):
        d_in  = in_dim    if i == 0            else hidden_dim
        d_out = out_dim   if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class FreqSurrogate(nn.Module):
    """
    Trunk MLP partagé (θ → h) + K heads fréquentiels (h ⊕ sinenc(k/(K-1)) → z_k).

    Paramètres
    ----------
    theta_dim  : dimension de θ (entrée)
    out_dim    : dimension de sortie par fréquence k
    K          : nombre de fréquences de Laplace
    hidden_dim : largeur du trunk
    head_dim   : largeur cachée de chaque head
    n_trunk    : couches Linear dans le trunk (avec LN+GELU entre chacune)
    n_head     : couches Linear dans chaque head
    freq_L     : niveaux sinusoïdaux pour l'encoding fréquentiel (sortie : 2*freq_L dims)
    """

    def __init__(
        self,
        theta_dim  : int,
        out_dim    : int,
        K          : int,
        hidden_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        self.K          = K
        self.out_dim    = out_dim
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.n_trunk    = n_trunk
        self.n_head     = n_head
        self.freq_L     = freq_L
        freq_cond       = 2 * freq_L

        trunk_layers: list[nn.Module] = []
        for i in range(n_trunk):
            d_in = theta_dim if i == 0 else hidden_dim
            trunk_layers += [nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            _mlp(hidden_dim + freq_cond, head_dim, out_dim, n_head)
            for _ in range(K)
        ])

        freq_ratios = torch.arange(K).float() / max(K - 1, 1)
        self.register_buffer('_freq_ratios', freq_ratios)

    def _sinenc(self, ratios: torch.Tensor) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.freq_L, device=ratios.device, dtype=ratios.dtype)) * torch.pi
        x = ratios[:, None] * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=1)

    def forward(self, theta: torch.Tensor, n_active: int | None = None) -> torch.Tensor:
        """theta: (B, theta_dim) → (B, n_active, out_dim)"""
        if n_active is None:
            n_active = self.K
        h        = self.trunk(theta)
        freq_enc = self._sinenc(self._freq_ratios[:n_active])
        B        = h.shape[0]
        outs     = [
            self.heads[k](torch.cat([h, freq_enc[k].unsqueeze(0).expand(B, -1)], dim=1))
            for k in range(n_active)
        ]
        return torch.stack(outs, dim=1)
