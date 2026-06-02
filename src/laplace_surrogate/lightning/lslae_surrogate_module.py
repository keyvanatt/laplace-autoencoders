"""
lslae_surrogate_module.py — LightningModule pour l'entraînement du surrogate LSLAE θ→Ĝ→U(t).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.

Offline (dans _build_model) :
  1. Charge LLAE pré-entraîné (encoder gelé)
  2. Encode toutes les simulations train+val → z_all [ns, Nt, D]
  3. SVD tronquée sur z_train → V [D, k_svd]
  4. Calcule stats de Ĝ = laplace.forward_transform(z @ V)

Training step :
  - Batch : (theta_norm, U_laplace_norm)  [même format que surrogate SLAE/LLAE]
  - Inverse Laplace → U_true (physique)
  - Encode U_norm → z_true via encoder gelé (on-the-fly)
  - forward(theta_norm, z_true) → (U_pred_norm, G_hat_norm, G_hat_true_norm)
  - Loss = spat + alpha_lat * lat
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers offline (phase 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_latents(encoder, U_raw, indices, N, Nt,
                     U_mean, U_std, device, batch_size=32):
    """Encode les simulations aux indices donnés → (ns, Nt, D) sur CPU."""
    import torch.nn.functional as F
    zs = []
    for start in tqdm(range(0, len(indices), batch_size), desc='  Encoding latents', leave=False):
        idx_batch = indices[start:start + batch_size]
        U_batch = torch.from_numpy(U_raw[idx_batch].copy()).float()
        B, Nt_raw = U_batch.shape[:2]
        if U_batch.shape[-1] != N:
            U_batch = F.interpolate(
                U_batch.reshape(B * Nt_raw, 1, U_batch.shape[-2], U_batch.shape[-1]),
                size=(N, N), mode='bilinear', align_corners=False,
            ).reshape(B, Nt_raw, N, N)
        U_norm = (U_batch.to(device) - U_mean) / U_std
        frames = U_norm.reshape(B * Nt_raw, 1, N, N)
        t = torch.arange(Nt_raw, dtype=U_norm.dtype, device=device) / max(Nt_raw - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt_raw, 1)
        z = encoder(frames, t_ratios).view(B, Nt_raw, -1)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def _compute_svd_and_stats(laplace, z_all, train_local, k_svd, device, batch_size=64):
    """SVD tronquée sur z_train → V [D, k_svd], puis stats (K, k_svd, 2)."""
    _, _, D = z_all.shape
    Z_train_flat = z_all[train_local].reshape(-1, D)
    tqdm.write(f"  SVD tronquée sur {tuple(Z_train_flat.shape)} → k_svd={k_svd}")
    _, _, Vh = torch.svd_lowrank(Z_train_flat, q=k_svd)
    V = Vh.contiguous()
    del Z_train_flat

    G_hat_list = []
    for start in tqdm(range(0, len(train_local), batch_size), desc='  Laplace fwd (stats)', leave=False):
        pos_batch = train_local[start:start + batch_size]
        G_batch = (z_all[pos_batch] @ V).to(device)
        G_hat = laplace.forward_transform(G_batch)
        G_hat_list.append(G_hat.cpu())

    G_hat_train = torch.cat(G_hat_list, dim=0)
    G_hat_ri   = torch.stack([G_hat_train.real, G_hat_train.imag], dim=-1).float()
    G_hat_mean = G_hat_ri.mean(0)
    G_hat_std  = G_hat_ri.std(0).clamp(min=1e-8)
    return V, G_hat_mean, G_hat_std


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class LSLAESurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate LSLAE θ→Ĝ→z→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model, self._encoder = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.lslae import LSLAE
        from laplace_surrogate.laplace_transform.learnable import LearnableLaplace

        cfg_m = self.cfg.model
        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N, Nt, K  = ds.N, ds.Nt, ds.K
        theta_dim = ds.theta_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Chargement LLAE (encoder gelé) ───────────────────────────────────
        from laplace_surrogate.lightning.ckpt_utils import load_llae_from_ckpt
        ae, latent_dim, dt, time_L = load_llae_from_ckpt(cfg_t.ae_ckpt, N, Nt, K)

        # Laplace avec les pôles du dataset (ds.s) — pour inverser U_laplace_norm du batch.
        data_laplace = LearnableLaplace(K=K, dt=dt, Nt=Nt, learnable=False,
                                        alpha_t=cfg_t.alpha_t, lam=cfg_t.lam)
        data_laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        data_laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        data_laplace.requires_grad_(False)
        self._data_laplace = data_laplace
        self._ae_latent_dim = latent_dim
        self._ae_dt         = dt
        self._ae_time_L     = time_L
        ae.to(device)

        # ── Phase 2 : encodage offline ───────────────────────────────────────
        all_idx     = dm.train_idx + dm.val_idx
        train_local = list(range(len(dm.train_idx)))

        print("LSLAE phase 2 : encodage des latents...")
        z_all = _compute_latents(
            ae.encoder, ds._U_raw, all_idx, N, Nt,
            ds.U_mean.to(device), ds.U_std.to(device), device,
            batch_size=cfg_t.get('encode_batch_size', 32),
        )

        print("LSLAE phase 2 : SVD + stats Ĝ...")
        V, G_hat_mean, G_hat_std = _compute_svd_and_stats(
            ae.laplace, z_all, train_local, cfg_t.k_svd, device,
        )

        # ── Modèle LSLAE ─────────────────────────────────────────────────────
        model = LSLAE(
            N=N, Nt=Nt, theta_dim=theta_dim,
            latent_dim=latent_dim,
            k_svd=cfg_t.k_svd,
            K=K, dt=dt,
            gamma_init=cfg_m.get('gamma_init', 0.0),
            time_L=time_L,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_t.freq_L,
        )
        model.load_ae_decoder(ae)
        model.load_laplace_from_ae(ae)
        model.set_svd_basis(V)
        model.set_normalization(G_hat_mean, G_hat_std, ds.U_mean, ds.U_std,
                                ds.theta_mean, ds.theta_std)

        # Garde l'encoder gelé pour z_true on-the-fly en training
        encoder = ae.encoder.cpu()
        del ae

        return model, encoder

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _laplace_to_u(self, U_laplace_norm: torch.Tensor) -> torch.Tensor:
        """(B, K, 2, N, N) → U_true physique (B, Nt, N, N)."""
        B, K, _, N, _ = U_laplace_norm.shape
        NN = N * N
        re  = U_laplace_norm[:, :, 0].reshape(B, K, NN).float()
        im  = U_laplace_norm[:, :, 1].reshape(B, K, NN).float()
        z_hat = torch.complex(re, im)
        U_norm = self.model.laplace.inverse_transform(z_hat, self.model.Nt)
        U_std  = self.model.U_std.to(U_norm.device)
        U_mean = self.model.U_mean.to(U_norm.device)
        return U_norm.reshape(B, self.model.Nt, N, N) * U_std + U_mean

    @torch.no_grad()
    def _encode_seq(self, U_norm: torch.Tensor) -> torch.Tensor:
        """(B, Nt, N, N) → z (B, Nt, latent_dim) via encoder gelé."""
        B, Nt, N, _ = U_norm.shape
        frames = U_norm.reshape(B * Nt, 1, N, N)
        t = torch.arange(Nt, dtype=U_norm.dtype, device=U_norm.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        z = self._encoder(frames, t_ratios)
        return z.view(B, Nt, -1)

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        theta_norm, U_laplace_norm = batch
        U_true = self._laplace_to_u(U_laplace_norm)
        U_std  = self.model.U_std.to(U_true.device)
        U_mean = self.model.U_mean.to(U_true.device)
        U_norm = (U_true - U_mean) / U_std
        z_true = self._encode_seq(U_norm)

        U_pred, G_hat_norm, G_hat_true_norm = self.model(theta_norm, z_true)
        alpha_lat = float(self.cfg.training.get('alpha_lat', 1.0))
        loss, metrics = self.model.loss(U_norm, U_pred, G_hat_norm, G_hat_true_norm,
                                        alpha_lat=alpha_lat)

        with torch.no_grad():
            l2rel = ((U_pred - U_norm).flatten(1).norm(dim=1)
                     / (U_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta_norm, U_laplace_norm = batch
        U_true = self._laplace_to_u(U_laplace_norm)
        U_std  = self.model.U_std.to(U_true.device)
        U_mean = self.model.U_mean.to(U_true.device)
        U_norm = (U_true - U_mean) / U_std
        z_true = self._encode_seq(U_norm)

        U_pred, G_hat_norm, G_hat_true_norm = self.model(theta_norm, z_true)
        alpha_lat = float(self.cfg.training.get('alpha_lat', 1.0))
        loss, metrics = self.model.loss(U_norm, U_pred, G_hat_norm, G_hat_true_norm,
                                        alpha_lat=alpha_lat)

        l2rel = ((U_pred - U_norm).flatten(1).norm(dim=1)
                 / (U_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,            on_epoch=True, prog_bar=True)
        self.log('val/spat',  metrics['spat'], on_epoch=True)
        self.log('val/lat',   metrics['lat'],  on_epoch=True)
        self.log('val/l2rel', l2rel,           on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        param_groups = [
            {'params': self.model.surrogate.parameters(), 'lr': cfg_t.lr_surrogate},
            {'params': self.model.decoder.parameters(),   'lr': cfg_t.lr_decoder},
            {'params': [self.model.V],                    'lr': cfg_t.lr_V},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'val/l2rel',
                'interval':  'epoch',
            },
        }

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds    = dm.dataset
        cfg_m = self.cfg.model
        cfg_t = self.cfg.training
        m     = self.model
        checkpoint.update({
            'model_type':  'LSLAEModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  self._ae_latent_dim,
            'k_svd':       m.k_svd,
            'dt':          self._ae_dt,
            'time_L':      self._ae_time_L,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'U_mean':      ds.U_mean,
            'U_std':       ds.U_std,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
