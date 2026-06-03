"""
dataset.py — TransientDataset migré depuis transient/dataset.py.

Aucune modification de la logique — seul l'import de laplace_forward_tik est mis à jour.
"""
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

from laplace_surrogate.laplace_transform.forward import laplace_forward_tik


class TransientDataset(Dataset):
    """
    Dataset pour les champs transitoires (ns, Nt, N, N).

    Modes
    -----
    laplace=False (défaut) : __getitem__ retourne (theta_norm, U)
    laplace=True           : applique la transformée de Laplace au chargement,
                             __getitem__ retourne (theta_norm, U_laplace_norm)
                             où U_laplace_norm a la forme (K, 2, N, N).

    Usage
    -----
    dataset = TransientDataset('dataset/ch4_rotated.npy', dt=1.0, laplace=True, s_list=s)
    dataset.fit(train_indices)
    theta_n, target_n = dataset[i]
    """

    def __init__(self, data_path: str, laplace: bool = False,
                 s_list=None, rule: str = 'trap', dt: float = 1.0,
                 doe_path: str | None = None, interp_size: int | None = None,
                 cache_dir: str = '/Data/KAT', ns_max: int | None = None):
        if data_path.endswith('.npy'):
            U_raw = np.load(data_path, mmap_mode='r')
            if doe_path is None:
                doe_path = str(Path(data_path).parent / 'doe_rotated.npy')
            doe = np.load(doe_path)
            theta_np = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)
            if ns_max is not None:
                U_raw    = U_raw[:ns_max]
                theta_np = theta_np[:ns_max]
            self.theta = torch.tensor(theta_np, dtype=torch.float32)
            self.dt    = dt
            ns, Nt, H, W = U_raw.shape
            self.ns, self.Nt = ns, Nt
            self.N = interp_size if interp_size is not None else H
            self._U_raw     = U_raw
            self.U = U_raw
            self._data_path = data_path
            self._cache_dir = Path(cache_dir)
        else:
            data   = np.load(data_path)
            self.U = torch.tensor(data['U'],     dtype=torch.float32)
            self.theta = torch.tensor(data['theta'], dtype=torch.float32)
            dt_raw     = data['dt']
            self.dt    = float(dt_raw[0]) if hasattr(dt_raw, '__len__') else float(dt_raw)
            self.ns, self.Nt, self.N, _ = self.U.shape
            self._U_raw: np.ndarray | None = None

        self.theta_dim = self.theta.shape[1]
        self.interp_size = interp_size

        self.laplace = laplace
        if laplace:
            if s_list is None:
                raise ValueError("s_list est requis quand laplace=True")
            s_list = np.asarray(s_list, dtype=np.complex128)
            self._s_hash = hashlib.md5(
                np.round(s_list, 2).astype(np.complex64).tobytes()
            ).hexdigest()[:8]
            self._s_list = s_list
            self.s       = s_list  # alias public pour compat avec les scripts
            self._rule   = rule
            self.K       = len(s_list)
        else:
            self.K = 0

        # Stats de normalisation (remplies par fit())
        self.theta_mean: torch.Tensor | None = None
        self.theta_std:  torch.Tensor | None = None
        self.U_mean:     np.ndarray  | None = None
        self.U_std:      np.ndarray  | None = None
        self.U_laplace:  np.ndarray  | None = None
        # Normalisation Laplace par fréquence, pixel par pixel (K, 2, N, N)
        self.lap_mean:   np.ndarray  | None = None
        self.lap_std:    np.ndarray  | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_indices):
        """Calcule les stats de normalisation sur les indices d'entraînement."""
        idx = list(train_indices)

        # θ
        theta_train = self.theta[idx]
        self.theta_mean = theta_train.mean(0)
        self.theta_std  = theta_train.std(0).clamp(min=1e-8)

        # U pixel-wise
        if self._U_raw is not None:
            # Streaming sur les sims train (évite d'allouer tout en RAM)
            U_sum  = np.zeros((self.N, self.N), dtype=np.float64)
            U_sum2 = np.zeros((self.N, self.N), dtype=np.float64)
            n = len(idx)
            for i in tqdm(idx, desc='fit U_mean/std', leave=False):
                u = self._load_u(i)   # (Nt, N, N) float32
                U_sum  += u.mean(0).astype(np.float64)
                U_sum2 += (u ** 2).mean(0).astype(np.float64)
            self.U_mean = (U_sum  / n).astype(np.float32)
            self.U_std  = np.sqrt(np.maximum(U_sum2 / n - self.U_mean ** 2, 0)).astype(np.float32)
            self.U_std  = np.where(self.U_std < 1e-8, 1.0, self.U_std)
        else:
            U_train    = self.U[idx]
            self.U_mean = U_train.mean(dim=(0, 1)).numpy().astype(np.float32)
            self.U_std  = U_train.std(dim=(0, 1)).numpy().astype(np.float32)
            self.U_std  = np.where(self.U_std < 1e-8, 1.0, self.U_std)

        if self.laplace:
            self._compute_laplace(idx)

    def _load_u(self, i: int) -> np.ndarray:
        """Charge la simulation i, interpole si besoin, retourne (Nt, N, N) float32."""
        u = self._U_raw[i].astype(np.float32)   # (Nt, H, W)
        if self.interp_size is not None and u.shape[-1] != self.interp_size:
            u_t = torch.from_numpy(u).unsqueeze(1)  # (Nt, 1, H, W)
            u_t = F.interpolate(u_t, size=(self.interp_size, self.interp_size), mode='bilinear', align_corners=False)
            u = u_t.squeeze(1).numpy()
        return u   # (Nt, N, N)

    def _laplace_frame(self, i: int, s_t: torch.Tensor) -> np.ndarray:
        """Calcule la transformée de Laplace brute de la sim i → (K, 2, N, N) float32."""
        u = self._load_u(i) if self._U_raw is not None else self.U[i].numpy()
        u_flat = torch.tensor(u.reshape(self.Nt, self.N * self.N).T, dtype=torch.float64)
        uhat   = laplace_forward_tik(u_flat, s_t, self.dt, self._rule)  # (N², K) complex
        uhat_np = uhat.numpy().reshape(self.N, self.N, self.K)
        frame = np.empty((self.K, 2, self.N, self.N), dtype=np.float32)
        frame[:, 0] = uhat_np.real.transpose(2, 0, 1)
        frame[:, 1] = uhat_np.imag.transpose(2, 0, 1)
        return frame

    def _compute_laplace(self, train_idx):
        """
        Pré-calcule la transformée de Laplace (sans normalisation réel) et normalise
        dans le domaine de Laplace fréquence par fréquence, pixel par pixel.

        2 passes :
          1. Stats (mean, std) calculées sur les sims d'entraînement.
          2. Normalisation appliquée à toutes les sims → mmap.
        """
        cache_dir  = self._cache_dir
        stem       = Path(self._data_path).stem if self._U_raw is not None else 'npz'
        idx_hash   = hashlib.md5(np.array(sorted(train_idx)).tobytes()).hexdigest()[:8]
        lap_path   = cache_dir / f"{stem}_laplace_N{self.N}_s{self._s_hash}_trap_idx{idx_hash}_lapnorm.npy"
        stats_path = cache_dir / f"{stem}_laplace_N{self.N}_s{self._s_hash}_trap_idx{idx_hash}_lapstats.npz"

        if lap_path.exists() and stats_path.exists():
            self.U_laplace = np.load(str(lap_path), mmap_mode='r')
            d = np.load(str(stats_path))
            self.lap_mean = d['mean']   # (K, 2, N, N) float32
            self.lap_std  = d['std']    # (K, 2, N, N) float32
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        s_t     = torch.tensor(self._s_list, dtype=torch.complex128)
        n_train = len(train_idx)

        # --- Passe 1 : stats sur les sims d'entraînement ---
        lap_sum  = np.zeros((self.K, 2, self.N, self.N), dtype=np.float64)
        lap_sum2 = np.zeros((self.K, 2, self.N, self.N), dtype=np.float64)
        for i in tqdm(train_idx, desc='Laplace stats (1/2)', leave=False):
            f = self._laplace_frame(i, s_t).astype(np.float64)
            lap_sum  += f
            lap_sum2 += f * f
        lap_mean = (lap_sum / n_train).astype(np.float32)
        lap_var  = np.maximum(lap_sum2 / n_train - lap_mean.astype(np.float64) ** 2, 0.0)
        lap_std  = np.sqrt(lap_var).astype(np.float32)
        lap_std  = np.where(lap_std < 1e-8, 1.0, lap_std)

        # --- Passe 2 : normalise toutes les sims → mmap ---
        out = np.lib.format.open_memmap(
            str(lap_path), mode='w+', dtype=np.float32,
            shape=(self.ns, self.K, 2, self.N, self.N),
        )
        for i in tqdm(range(self.ns), desc='Laplace normalise (2/2)', leave=False):
            out[i] = (self._laplace_frame(i, s_t) - lap_mean) / lap_std
        out.flush()

        np.savez(str(stats_path), mean=lap_mean, std=lap_std)
        self.U_laplace = np.load(str(lap_path), mmap_mode='r')
        self.lap_mean  = lap_mean
        self.lap_std   = lap_std

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.ns

    def __getitem__(self, idx: int):
        theta_norm = (self.theta[idx] - self.theta_mean) / self.theta_std

        if self.laplace:
            u_lap = self.U_laplace[idx].copy()    # (K, 2, N, N)
            return theta_norm, torch.from_numpy(u_lap).float()
        else:
            if self._U_raw is not None:
                u = self._load_u(idx)
            else:
                u = self.U[idx].numpy()
            u_norm = (u - self.U_mean) / self.U_std  # (Nt, N, N)
            return theta_norm, torch.from_numpy(u_norm).float()
