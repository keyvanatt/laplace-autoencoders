"""
slae_svd_surrogate_module.py — LightningModule pour le surrogate SVD-SLAE θ→G_k_SVD→U(t).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.

Offline (dans _build_model) :
  1. Charge SLAE pré-entraîné (encoder + décodeur gelés)
  2. Encode toutes les simulations train+val via l'encodeur SLAE → z_all [ns, K, D]
     (latents dans le domaine de Laplace, réels)
  3. SVD tronquée sur z_train (reshape ns*K × D) → V [D, k_svd]
  4. Calcule stats de G_k = z_all @ V sur les indices train

Batch d'entraînement : (theta_norm, U_norm) — domaine temporel.
Loss = spatial MSE + alpha_lat * latent MSE (sur G_k normalisé).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers offline
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_slae_latents(slae_model, U_raw, indices, N, Nt, device, batch_size=32):
    """
    Encode les simulations aux indices donnés via le modèle SLAE intermédiaire.
    Retourne z_all (ns, K, latent_dim) sur CPU.
    """
    import torch.nn.functional as F
    zs = []
    for start in tqdm(range(0, len(indices), batch_size), desc='  Encoding SLAE latents', leave=False):
        idx_batch = indices[start:start + batch_size]
        U_batch   = torch.from_numpy(U_raw[idx_batch].copy()).float()
        B, Nt_raw = U_batch.shape[:2]
        if U_batch.shape[-1] != N:
            U_batch = F.interpolate(
                U_batch.reshape(B * Nt_raw, 1, U_batch.shape[-2], U_batch.shape[-1]),
                size=(N, N), mode='bilinear', align_corners=False,
            ).reshape(B, Nt_raw, N, N)
        U_norm = (U_batch.to(device) - slae_model.U_mean) / slae_model.U_std
        z      = slae_model._encode_targets(U_norm)   # (B, K, latent_dim)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def _compute_slae_svd_and_stats(z_all, train_local, k_svd):
    """
    SVD tronquée sur z_train (reshape ns_train*K × D) → V [D, k_svd].
    Calcule G_mean, G_std sur z_train[train_local] @ V.
    """
    ns, K, D = z_all.shape
    Z_flat   = z_all[train_local].reshape(-1, D)
    tqdm.write(f"  SVD tronquée sur {tuple(Z_flat.shape)} → k_svd={k_svd}")
    _, _, Vh = torch.svd_lowrank(Z_flat, q=k_svd)
    V        = Vh.contiguous()                        # [D, k_svd]
    del Z_flat

    G_train = (z_all[train_local] @ V)               # [ns_train, K, k_svd]
    G_mean  = G_train.mean(0)                         # [K, k_svd]
    G_std   = G_train.std(0).clamp(min=1e-8)          # [K, k_svd]
    return V, G_mean, G_std


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SLAESVDSurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate SVD-SLAE θ→G_k_SVD→z_k→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.slae_svd_surrogate import SLAESVDModel
        from laplace_surrogate.models.slae_surrogate import SLAEModel
        from laplace_surrogate.lightning.ckpt_utils import load_slae_from_ckpt, laplace_reg

        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N, Nt, K  = ds.N, ds.Nt, ds.K
        theta_dim = ds.theta_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Chargement SLAE (encodeur + décodeur gelés) ───────────────────────
        ae, latent_dim, freq_L = load_slae_from_ckpt(cfg_t.ae_ckpt, N)
        self._alpha_t, self._lam = laplace_reg(self.cfg)

        # Modèle intermédiaire pour l'encodage offline (normalisations dataset incluses)
        slae_tmp = SLAEModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=self._alpha_t,
            lam=self._lam,
        )
        slae_tmp.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        slae_tmp.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        slae_tmp.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        slae_tmp.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        slae_tmp.laplace.requires_grad_(False)
        if ds.lap_mean is not None:
            slae_tmp.lap_mean.copy_(torch.tensor(ds.lap_mean, dtype=torch.float32))
            slae_tmp.lap_std.copy_( torch.tensor(ds.lap_std,  dtype=torch.float32))
        slae_tmp.to(device)

        # ── Phase 2 offline : encodage + SVD ─────────────────────────────────
        all_idx     = dm.train_idx + dm.val_idx
        train_local = list(range(len(dm.train_idx)))

        print("SVD-SLAE surrogate : encodage des latents Laplace...")
        z_all = _compute_slae_latents(
            slae_tmp, ds._U_raw, all_idx, N, Nt, device,
            batch_size=cfg_t.get('encode_batch_size', 32),
        )

        print("SVD-SLAE surrogate : SVD + stats G_k...")
        V, G_mean, G_std = _compute_slae_svd_and_stats(z_all, train_local, cfg_t.k_svd)

        # ── Modèle SLAESVDModel ───────────────────────────────────────────────
        model = SLAESVDModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            k_svd=cfg_t.k_svd,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=self._alpha_t,
            lam=self._lam,
        )
        # Normalisation dataset
        model.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        model.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        model.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        model.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        model.laplace.requires_grad_(False)
        if ds.lap_mean is not None:
            model.lap_mean.copy_(torch.tensor(ds.lap_mean, dtype=torch.float32))
            model.lap_std.copy_( torch.tensor(ds.lap_std,  dtype=torch.float32))
        # Base SVD + stats
        with torch.no_grad():
            model.V.copy_(V.float())
            model.G_mean.copy_(G_mean.float())
            model.G_std.copy_( G_std.float())

        if cfg_t.lr_decoder > 0.0:
            model.shared_decoder.requires_grad_(True)

        del slae_tmp
        return model

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        U_pred_norm, G_k_norm, G_true_norm = self.model(theta_norm, U_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(U_norm, U_pred_norm, G_k_norm, G_true_norm,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        with torch.no_grad():
            l2rel = ((U_pred_norm.float() - U_norm.float()).flatten(1).norm(dim=1)
                     / (U_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        U_pred_norm, G_k_norm, G_true_norm = self.model(theta_norm, U_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(U_norm, U_pred_norm, G_k_norm, G_true_norm,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        l2rel = ((U_pred_norm.float() - U_norm.float()).flatten(1).norm(dim=1)
                 / (U_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

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
            {'params': [self.model.V],                    'lr': cfg_t.lr_V},
        ]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': self.model.shared_decoder.parameters(),
                                 'lr': cfg_t.lr_decoder})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/l2rel', 'interval': 'epoch'},
        }

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds = dm.dataset
        m  = self.model
        checkpoint.update({
            'model_type':  'SLAESVDModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           m.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  m.latent_dim,
            'k_svd':       m.k_svd,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'surr_freq_L': m.surr_freq_L,
            'dt':          self.cfg.data.dt,
            'alpha_t':     self._alpha_t,
            'lam':         self._lam,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
