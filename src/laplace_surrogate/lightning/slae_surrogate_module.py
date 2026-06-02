"""
slae_surrogate_module.py — LightningModule pour l'entraînement end-to-end du surrogate SLAE θ→z (phase 2).

Batch : (theta_norm, U_norm) avec U_norm = (U - U_mean) / U_std  [domaine temporel].
Délègue entièrement le forward et la loss à SLAEModel (encodeur, décodeur, inversion Laplace).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig


class SLAESurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate SLAE θ→z→U(t) (décodeur gelé ou fine-tuné)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    def _build_model(self, dm):
        from laplace_surrogate.lightning.ckpt_utils import load_slae_from_ckpt
        from laplace_surrogate.models.slae_surrogate import SLAEModel

        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N         = ds.N
        Nt        = ds.Nt
        K         = ds.K
        theta_dim = ds.theta_dim

        ae, latent_dim, freq_L = load_slae_from_ckpt(cfg_t.ae_ckpt, N)

        model = SLAEModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=cfg_t.alpha_t,
            lam=cfg_t.lam,
        )
        model.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        model.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        # Pôles identiques à ceux utilisés pour calculer U_laplace dans le dataset
        model.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        model.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        model.laplace.requires_grad_(False)

        if cfg_t.lr_decoder > 0.0:
            model.shared_decoder.requires_grad_(True)

        return model

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        th, u_true_norm  = batch                   # (B, 3), (B, Nt, N, N)
        u_pred, z_pred, z_true = self.model(th, u_true_norm)
        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(u_true_norm, u_pred, z_pred, z_true,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        with torch.no_grad():
            l2rel = ((u_pred.float() - u_true_norm.float()).flatten(1).norm(dim=1)
                     / (u_true_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        th, u_true_norm  = batch
        u_pred, z_pred, z_true = self.model(th, u_true_norm)
        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(u_true_norm, u_pred, z_pred, z_true,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        l2rel = ((u_pred.float() - u_true_norm.float()).flatten(1).norm(dim=1)
                 / (u_true_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,            on_epoch=True, prog_bar=True)
        self.log('val/spat',  metrics['spat'], on_epoch=True)
        self.log('val/lat',   metrics['lat'],  on_epoch=True)
        self.log('val/l2rel', l2rel,           on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        param_groups = [{'params': self.model.surrogate.parameters(), 'lr': cfg_t.lr_surrogate}]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': self.model.shared_decoder.parameters(), 'lr': cfg_t.lr_decoder})

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
            'model_type':  'SLAEModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           m.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  m.latent_dim,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'surr_freq_L': m.surr_freq_L,
            'dt':          self.cfg.data.dt,
            'alpha_t':     self.cfg.training.alpha_t,
            'lam':         self.cfg.training.lam,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
