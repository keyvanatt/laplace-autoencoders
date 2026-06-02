"""
llae_surrogate_module.py — LightningModule pour l'entraînement end-to-end du surrogate LLAE θ→z (phase 2).

Batch : (theta_norm, U_norm) avec U_norm = (U - U_mean) / U_std  [domaine temporel].
Loss spatiale sur U_norm.
Supervision latente : encoder(U_norm) → Laplace → ẑ_true (dans LLAEModel.forward).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig


class LLAESurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate LLAE θ→ẑ→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    def _build_model(self, dm):
        from laplace_surrogate.lightning.ckpt_utils import load_llae_from_ckpt
        from laplace_surrogate.models.llae_surrogate import LLAEModel

        cfg_t = self.cfg.training

        ds        = dm.dataset
        Nt        = ds.Nt
        K         = self.cfg.model.K   # lu depuis le checkpoint AE par peek_ae_hparams (ds.K=0 car laplace=False)
        theta_dim = ds.theta_dim

        ae, latent_dim, dt, time_L = load_llae_from_ckpt(cfg_t.ae_ckpt, ds.N, Nt, K)
        self._ae_latent_dim = latent_dim
        self._ae_dt         = dt
        self._ae_time_L     = time_L

        return LLAEModel(
            ae=ae, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_t.freq_L,
            learnable_laplace=bool(self.cfg.model.get('learnable_laplace', False)),
        )

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        th, u_true_norm = batch                   # (B, 3), (B, Nt, N, N)
        U_rec, z_hat_pred, z_hat_true = self.model(th, u_true_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(u_true_norm, U_rec, z_hat_pred, z_hat_true,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        with torch.no_grad():
            l2rel = ((U_rec.float() - u_true_norm).flatten(1).norm(dim=1)
                     / (u_true_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        th, u_true_norm = batch
        U_rec, z_hat_pred, z_hat_true = self.model(th, u_true_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(u_true_norm, U_rec, z_hat_pred, z_hat_true,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)
        l2rel = ((U_rec.float() - u_true_norm).flatten(1).norm(dim=1)
                 / (u_true_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,            on_epoch=True, prog_bar=True)
        self.log('val/spat',  metrics['spat'], on_epoch=True)
        self.log('val/lat',   metrics['lat'],  on_epoch=True)
        self.log('val/l2rel', l2rel,           on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        param_groups = [{'params': self.model.surrogate.parameters(), 'lr': cfg_t.lr_surrogate}]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': self.model.decoder.parameters(), 'lr': cfg_t.lr_decoder})
        if self.model.learnable_laplace:
            lr_laplace = float(cfg_t.get('lr_laplace', 1e-5))
            param_groups.append({'params': self.model.laplace.parameters(), 'lr': lr_laplace})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/l2rel', 'interval': 'epoch'},
        }

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds = dm.dataset
        m  = self.model
        checkpoint.update({
            'model_type':  'LLAEModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  self._ae_latent_dim,
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
