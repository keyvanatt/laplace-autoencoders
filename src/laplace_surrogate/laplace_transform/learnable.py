import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnableLaplace(nn.Module):
    """
    Transformée de Laplace discrète avec K points s_k libres.

    Paramètres apprenables
    ----------------------
    s_re        : (K,) parties réelles σ_k des pôles s_k = σ_k + iω_k
    s_im        : (K,) parties imaginaires ω_k
    log_alpha_t : ln(α_t), poids du lissage temporel dans l'inversion
    log_lam     : ln(λ), ridge dans l'inversion

    Initialisation : σ_k = gamma_init, ω_k uniformément sur [0, π/dt].

    La matrice DtTDt (terme de lissage) est constante → buffer pré-calculé.
    En mode eval, les matrices A et F_full sont mises en cache après le premier
    appel (A ne change plus dès que les paramètres sont gelés).
    """

    def __init__(self, K: int, dt: float, Nt: int, gamma_init: float = 1e-2,
                 learnable: bool = False,
                 alpha_t: float = math.exp(-2.0), lam: float = math.exp(-2.0)):
        super().__init__()
        self.K         = K
        self.dt        = dt
        self.Nt        = Nt
        self.learnable = learnable

        # Contour de Bromwich tronqué : K premières fréquences FFT (ω_k = 2π·k/(Nt·dt))
        s_im_init = torch.tensor(
            2 * math.pi * np.fft.rfftfreq(Nt, d=dt)[:K], dtype=torch.float32
        )

        self.s_re        = nn.Parameter(torch.full((K,), gamma_init),
                                        requires_grad=learnable)
        self.s_im        = nn.Parameter(s_im_init.clone(), requires_grad=learnable)
        self.log_alpha_t = nn.Parameter(torch.tensor(math.log(alpha_t)), requires_grad=learnable)
        self.log_lam     = nn.Parameter(torch.tensor(math.log(lam)),     requires_grad=learnable)

        self.register_buffer('_alpha_t_fixed', torch.tensor(alpha_t))
        self.register_buffer('_lam_fixed',     torch.tensor(lam))

        Dt = (torch.diag(torch.ones(Nt - 1), 1) - torch.eye(Nt))[:Nt - 1, :]
        self.register_buffer('_DtTDt', Dt.T @ Dt)   # (Nt, Nt) float32

        self.register_buffer('_s_init_re', torch.full((K,), gamma_init))
        self.register_buffer('_s_init_im', s_im_init)

        self._eval_cache: Optional[tuple] = None
        self._F_fwd_cache: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        if mode and self.learnable:
            self._eval_cache = None
            self._F_fwd_cache = None
        return super().train(mode)

    @property
    def s_list(self) -> torch.Tensor:
        """s_k = σ_k + iω_k,  (K,) complex64."""
        return torch.complex(self.s_re, self.s_im)

    def _build_inv_matrices(self, device: torch.device):
        s = self.s_list.to(dtype=torch.complex64, device=device)
        c_mask = s.imag > 0
        s_full = torch.cat([s, torch.conj(s[c_mask])])

        t = torch.arange(self.Nt, dtype=torch.float32, device=device) * self.dt
        w = torch.ones(self.Nt, dtype=torch.float32, device=device)
        w[0] = 0.5; w[-1] = 0.5

        F_full = self.dt * w[None, :] * torch.exp(-s_full[:, None] * t[None, :])

        # Assemblage et factorisation de A en float64. À K=16 les pôles se resserrent,
        # les colonnes de F_full deviennent quasi colinéaires et κ(A) dépasse le plafond
        # du float32 (~1e7) : Cholesky pose alors un pivot négatif sur le dernier mineur
        # ("not positive-definite"). Le float64 (~1e15) absorbe la marge. Coût négligeable
        # (A est Nt×Nt = 150×150). F_full reste complex64 pour le matmul aval avec z_hat.
        F64    = F_full.to(torch.complex128)
        FtF    = torch.real(torch.conj(F64).T @ F64)

        alpha_t = (self._alpha_t_fixed if not self.learnable else self.log_alpha_t.exp()).double()
        lam     = (self._lam_fixed     if not self.learnable else self.log_lam.exp()).double()
        A = (FtF
             + alpha_t * self._DtTDt.to(device=device, dtype=torch.float64)
             + lam * torch.eye(self.Nt, dtype=torch.float64, device=device))

        L = torch.linalg.cholesky(A)   # float64
        return s_full, F_full, L, c_mask

    def _get_inv_matrices(self, device: torch.device):
        # Pôles apprenables : A et F_full dépendent de paramètres qui bougent à chaque
        # step, donc jamais de cache. On ne peut pas s'appuyer sur train() pour invalider :
        # Lightning restaure le mode train sans repasser par Module.train(True), si bien
        # qu'un cache rempli sous no_grad (sanity-check) resterait utilisé à l'entraînement
        # et couperait le gradient vers s_re/s_im/log_alpha_t/log_lam.
        if self.learnable:
            return self._build_inv_matrices(device)

        if self._eval_cache is not None:
            cached = self._eval_cache
            if cached[0].device != device:
                self._eval_cache = tuple(x.to(device) for x in cached)
            return self._eval_cache
        matrices = self._build_inv_matrices(device)
        self._eval_cache = matrices
        return matrices

    def _get_F_fwd(self, Nt: int, device: torch.device) -> torch.Tensor:
        if self._F_fwd_cache is not None:
            if self._F_fwd_cache.device != device:
                self._F_fwd_cache = self._F_fwd_cache.to(device)
            return self._F_fwd_cache
        s = self.s_list.to(dtype=torch.complex64, device=device)
        t = torch.arange(Nt, dtype=torch.float32, device=device) * self.dt
        w = torch.ones(Nt, dtype=torch.float32, device=device)
        w[0] = 0.5; w[-1] = 0.5
        F = self.dt * w[None, :] * torch.exp(-s[:, None] * t[None, :])
        if not self.learnable:
            self._F_fwd_cache = F
        return F

    def forward_transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, Nt, latent_dim)
        → ẑ : (B, K, latent_dim) complex64
        """
        B, Nt, D = z.shape
        device = z.device
        z_flat = z.permute(0, 2, 1).reshape(B * D, Nt).float()
        F = self._get_F_fwd(Nt, device)
        z_hat_flat = z_flat.to(dtype=F.dtype) @ F.T
        return z_hat_flat.view(B, D, self.K).permute(0, 2, 1)

    def inverse_transform(self, z_hat: torch.Tensor, Nt: int) -> torch.Tensor:
        """
        ẑ : (B, K, latent_dim) complex64
        → z_rec : (B, Nt, latent_dim) float32
        """
        assert Nt == self.Nt, f"Nt mismatch: got {Nt}, expected {self.Nt}"
        B, K, D = z_hat.shape
        device  = self.s_re.device

        s_full, F_full, L, c_mask = self._get_inv_matrices(device)

        z_hat_flat  = z_hat.permute(0, 2, 1).reshape(B * D, K)
        U_hat_full  = torch.cat(
            [z_hat_flat, torch.conj(z_hat_flat[:, c_mask])], dim=1
        ).to(torch.complex64)

        RHS = torch.real(U_hat_full @ torch.conj(F_full))
        z_rec_flat = torch.cholesky_solve(RHS.T.double(), L).T   # L en float64
        return z_rec_flat.view(B, D, self.Nt).permute(0, 2, 1).float()

    def log_scatter(self, epoch: int):
        """Retourne un wandb.Image du scatter s_k (initial → courant)."""
        import wandb
        import numpy as np
        import matplotlib.pyplot as plt

        s_cur = self.s_list.detach().cpu().numpy()
        s_ini = np.vectorize(complex)(
            self._s_init_re.cpu().numpy(),
            self._s_init_im.cpu().numpy(),
        )
        K    = len(s_cur)
        cmap = plt.cm.plasma

        fig, ax = plt.subplots(figsize=(5, 4))
        for k, (a, b) in enumerate(zip(s_ini, s_cur)):
            col = cmap(k / max(K - 1, 1))
            ax.plot([a.real, b.real], [a.imag, b.imag], color=col, lw=0.6, alpha=0.5)
        ax.scatter(s_ini.real, s_ini.imag, c=np.arange(K), cmap='plasma',
                   s=50, marker='o', zorder=3, alpha=0.4, label='initial')
        ax.scatter(s_cur.real, s_cur.imag, c=np.arange(K), cmap='plasma',
                   s=80, marker='*', zorder=4, label='current')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Re(s)')
        ax.set_ylabel('Im(s)')
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_title(f's-points  epoch {epoch}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    def log_dict(self, epoch: int, scatter_every: int = 10) -> dict:
        """
        Payload wandb : α_t, λ, liste des s_k et scatter périodique.
        Mêmes clés que scripts/laplace_opti.py pour pouvoir superposer les courbes.
        """
        alpha_t = self.log_alpha_t.exp() if self.learnable else self._alpha_t_fixed
        lam     = self.log_lam.exp()     if self.learnable else self._lam_fixed
        payload = {
            'params/alpha_t': float(alpha_t.detach().cpu()),
            'params/lam':     float(lam.detach().cpu()),
            's_points/text':  self.log_text(),
        }
        if epoch % scatter_every == 0:
            payload['s_points/scatter'] = self.log_scatter(epoch)
        return payload

    def log_text(self):
        """Retourne un wandb.Html listant les s_k courants."""
        import wandb
        s = self.s_list.detach().cpu().numpy().tolist()
        body = ", ".join(f"{z.real:.4f}+{z.imag:.4f}j" for z in s)
        return wandb.Html(
            f"<pre style='font-family:monospace;white-space:pre-wrap;'>[{body}]</pre>"
        )
