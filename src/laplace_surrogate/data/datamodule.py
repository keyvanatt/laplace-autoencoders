"""
datamodule.py — LightningDataModule pour les trois phases d'entraînement.

mode='ae'         → _LaplaceFlatDataset  (paires sim×freq pour l'AE)
mode='surrogate'  → _SurrogateDataset    (une simulation complète par item)
mode='corrector'  → _FrameDataset        (depuis memmaps pré-calculés)
"""
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset interne — phase AE
# ---------------------------------------------------------------------------

class _LaplaceFlatDataset(Dataset):
    """Paires (sim, freq). Appeler reshuffle() avant chaque epoch."""

    def __init__(self, U_laplace: np.ndarray, indices, K: int):
        self.U_laplace   = U_laplace
        self._indices    = [int(i) for i in indices]
        self._n_freqs    = K
        self._freq_ratio = [k / max(K - 1, 1) for k in range(K)]
        self.pairs       = self._make_pairs()

    def _make_pairs(self):
        return [(i, k) for i in self._indices for k in range(self._n_freqs)]

    def reshuffle(self):
        random.shuffle(self._indices)
        self.pairs = self._make_pairs()
        random.shuffle(self.pairs)  # interleave sims et fréquences dans les batchs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sim_i, k = self.pairs[idx]
        u = torch.from_numpy(self.U_laplace[sim_i, k].copy()).float()  # (2, N, N)
        return u, torch.tensor(self._freq_ratio[k], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset interne — phase surrogate
# ---------------------------------------------------------------------------

class _SurrogateDataset(Dataset):
    """Une simulation complète (K fréquences) par item."""

    def __init__(self, U_laplace: np.ndarray, theta_norm: torch.Tensor, indices):
        self.U_laplace  = U_laplace
        self.theta_norm = theta_norm.cpu()
        self._indices   = [int(i) for i in indices]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sim_i = self._indices[idx]
        th = self.theta_norm[sim_i]
        u  = torch.from_numpy(self.U_laplace[sim_i].copy()).float()  # (K, 2, N, N)
        return th, u


# ---------------------------------------------------------------------------
# Dataset interne — phase AE pour LLAE (séquences temporelles complètes)
# ---------------------------------------------------------------------------

class _TimeSeqDataset(Dataset):
    """Séquences temporelles complètes (Nt, N, N) par simulation — pour LLAE."""

    def __init__(self, dataset, indices):
        self._ds  = dataset
        self._idx = [int(i) for i in indices]

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, idx):
        sim_i = self._idx[idx]
        if self._ds._U_raw is not None:
            u = self._ds._load_u(sim_i)
        else:
            u = self._ds.U[sim_i].numpy()
        u_norm = (u - self._ds.U_mean) / self._ds.U_std
        return torch.from_numpy(u_norm).float()  # (Nt, N, N)


# ---------------------------------------------------------------------------
# Dataset interne — phase corrector
# ---------------------------------------------------------------------------

class _FrameDataset(Dataset):
    """Paires (U_pred, U_true) de shape (kt, N, N) depuis les memmaps."""

    def __init__(self, U_pred: np.ndarray, U_true: np.ndarray, indices):
        self.U_pred  = U_pred
        self.U_true  = U_true
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.U_pred[i].copy()),
                torch.from_numpy(self.U_true[i].copy()))


# ---------------------------------------------------------------------------
# Pré-calcul memmaps pour le corrector
# ---------------------------------------------------------------------------

def precompute(dataset, surrogate_ckpt: str, kt: int, cache_dir: str,
               batch_size: int = 32, dt: float = 1.0,
               alpha_t: float = 0.0, lam: float = 1e-6,
               rule: str = 'trap', seed: int = 0):
    """Génère ou charge les memmaps U_pred/U_true de shape (ns, kt, N, N)."""
    from laplace_surrogate.inference.pipeline import InferencePipeline

    ns  = dataset.ns
    Nt  = dataset.Nt
    N   = dataset.N
    os.makedirs(cache_dir, exist_ok=True)

    surr_name = os.path.splitext(os.path.basename(surrogate_ckpt))[0]
    pred_path = os.path.join(cache_dir, f"correction_upred_{surr_name}_kt{kt}_N{N}_s{seed}.npy")
    true_path = os.path.join(cache_dir, f"correction_utrue_kt{kt}_N{N}_s{seed}.npy")

    if os.path.exists(pred_path) and os.path.exists(true_path):
        print(f"Cache trouvé :\n  {pred_path}\n  {true_path}")
        return (np.load(pred_path, mmap_mode='r'),
                np.load(true_path, mmap_mode='r'))

    print(f"Pré-calcul U_pred/U_true → {cache_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = InferencePipeline.from_checkpoint(surrogate_ckpt, device)
    pipe.model.eval()

    rng   = np.random.default_rng(seed)
    t_idx = rng.integers(0, Nt, size=(ns, kt))

    pred_mmap = np.lib.format.open_memmap(pred_path, mode='w+', dtype=np.float32, shape=(ns, kt, N, N))
    true_mmap = np.lib.format.open_memmap(true_path, mode='w+', dtype=np.float32, shape=(ns, kt, N, N))

    theta_raw = dataset.theta.numpy()
    for start in tqdm(range(0, ns, batch_size), desc='Pré-calcul'):
        end = min(start + batch_size, ns)
        B   = end - start

        u_raw = torch.from_numpy(dataset._U_raw[start:end].copy()).float()
        if u_raw.shape[-1] != N:
            u_raw = F.interpolate(
                u_raw.reshape(B * Nt, 1, u_raw.shape[-2], u_raw.shape[-1]),
                size=(N, N), mode='bilinear', align_corners=False,
            ).reshape(B, Nt, N, N)

        u_pred = torch.from_numpy(
            pipe.predict(theta_raw[start:end], dt=dt, alpha_t=alpha_t, lam=lam, rule=rule)
        )

        for i in range(B):
            ti = t_idx[start + i]
            pred_mmap[start + i] = u_pred[i, ti].numpy()
            true_mmap[start + i] = u_raw[i, ti].numpy()

    pred_mmap.flush()
    true_mmap.flush()
    print(f"Sauvegardé — {pred_mmap.nbytes / 1e9:.1f} Go × 2")

    del pipe
    torch.cuda.empty_cache()

    return (np.load(pred_path, mmap_mode='r'),
            np.load(true_path, mmap_mode='r'))


# ---------------------------------------------------------------------------
# DataModule principal
# ---------------------------------------------------------------------------

class TransientDataModule(pl.LightningDataModule):
    """
    DataModule unifié pour les trois phases d'entraînement.

    Args:
        cfg:  DictConfig Hydra (doit avoir cfg.data, cfg.training, cfg.model, cfg.seed).
        mode: 'ae' | 'surrogate' | 'corrector'
    """

    def __init__(self, cfg, mode: str = 'ae'):
        super().__init__()
        self.cfg  = cfg
        self.mode = mode

        self.dataset       = None
        self.train_dataset = None
        self.val_dataset   = None
        self.train_idx     = None
        self.val_idx       = None
        self.test_idx      = None

    # ------------------------------------------------------------------

    def setup(self, stage=None):
        from laplace_surrogate.data.dataset import TransientDataset

        cfg_d = self.cfg.data
        cfg_t = self.cfg.training

        model_name = self.cfg.model.get('name', 'slae')
        # LLAE encode dans le domaine temporel : pas besoin de la transformée de Laplace
        laplace = self.mode == 'surrogate' or (self.mode == 'ae' and model_name != 'llae')

        # s_list pour la transformée de Laplace
        s_list = None
        if laplace:
            K      = self.cfg.model.get('K', 16)
            gamma  = self.cfg.model.get('gamma_init', 0.0)
            s_im   = np.linspace(0.0, math.pi / cfg_d.dt, K)
            s_list = (gamma + 1j * s_im).astype(np.complex128)

        self.dataset = TransientDataset(
            cfg_d.data_path,
            laplace=laplace,
            s_list=s_list,
            rule=cfg_d.rule,
            interp_size=cfg_d.interp_size,
            dt=cfg_d.dt,
        )

        # Splits (test fixe + train/val aléatoire reproductible)
        split         = np.load(cfg_d.split_path)
        self.test_idx = split['test_idx'].tolist()
        test_set      = set(self.test_idx)
        non_test      = [i for i in range(self.dataset.ns) if i not in test_set]

        gen = torch.Generator()
        gen.manual_seed(self.cfg.seed)
        perm    = torch.randperm(len(non_test), generator=gen).tolist()
        n_train = int(cfg_d.train_val_split * len(non_test))
        self.train_idx = [non_test[i] for i in perm[:n_train]]
        self.val_idx   = [non_test[i] for i in perm[n_train:]]

        self.dataset.fit(self.train_idx)

        if self.mode == 'ae' and model_name == 'llae':
            self.train_dataset = _TimeSeqDataset(self.dataset, self.train_idx)
            self.val_dataset   = _TimeSeqDataset(self.dataset, self.val_idx)

        elif self.mode == 'ae':
            print("Chargement Laplace en RAM...", end=' ', flush=True)
            t0 = time.perf_counter()
            U = np.ascontiguousarray(self.dataset.U_laplace)
            self.dataset.U_laplace = U
            print(f"OK — {U.nbytes / 1e9:.1f} Go, {time.perf_counter() - t0:.1f}s")

            K    = self.dataset.K
            self.train_dataset = _LaplaceFlatDataset(U, self.train_idx, K)
            self.val_dataset   = _LaplaceFlatDataset(U, self.val_idx,   K)

        elif self.mode == 'surrogate':
            print("Chargement Laplace en RAM...", end=' ', flush=True)
            t0 = time.perf_counter()
            U = np.ascontiguousarray(self.dataset.U_laplace)
            self.dataset.U_laplace = U
            print(f"OK — {U.nbytes / 1e9:.1f} Go, {time.perf_counter() - t0:.1f}s")

            theta_norm = torch.tensor(
                (self.dataset.theta.numpy() - self.dataset.theta_mean.numpy())
                / self.dataset.theta_std.numpy(),
                dtype=torch.float32,
            )
            self.train_dataset = _SurrogateDataset(U, theta_norm, self.train_idx)
            self.val_dataset   = _SurrogateDataset(U, theta_norm, self.val_idx)

        elif self.mode == 'corrector':
            U_pred, U_true = precompute(
                self.dataset,
                surrogate_ckpt=cfg_t.surrogate_ckpt,
                kt=cfg_t.kt,
                cache_dir=cfg_d.cache_dir,
                dt=cfg_d.dt,
                alpha_t=float(cfg_t.get('alpha_t', 0.0)),
                lam=float(cfg_t.get('lam', 1e-6)),
                rule=cfg_d.rule,
            )
            self.train_dataset = _FrameDataset(U_pred, U_true, self.train_idx)
            self.val_dataset   = _FrameDataset(U_pred, U_true, self.val_idx)

        else:
            raise ValueError(f"mode inconnu : {self.mode!r}. Attendu : ae | surrogate | corrector")

    # ------------------------------------------------------------------

    def train_dataloader(self):
        nw = 8 if self.mode == 'ae' else 4
        # SLAE AE : reshuffle() manuel → pas de shuffle DataLoader
        # LLAE AE et autres modes : shuffle DataLoader standard
        use_dl_shuffle = not (self.mode == 'ae' and hasattr(self.train_dataset, 'reshuffle'))
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=use_dl_shuffle,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        nw = 8 if self.mode == 'ae' else 4
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
        )
