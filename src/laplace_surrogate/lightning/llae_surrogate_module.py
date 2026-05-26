"""
llae_surrogate_module.py — LightningModule pour l'entraînement end-to-end du surrogate LLAE θ→z (phase 2).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig


class LLAESurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate LLAE θ→ẑ→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.llae import LLAE
        from laplace_surrogate.models.llae_surrogate import LLAEModel

        cfg_m = self.cfg.model
        cfg_t = self.cfg.training

        ds        = dm.dataset
        N         = ds.N
        Nt        = ds.Nt
        K         = ds.K
        theta_dim = ds.theta_dim

        ae_ck = torch.load(cfg_t.ae_ckpt, map_location='cpu', weights_only=False)
        ae = LLAE(
            N=N, Nt=Nt,
            latent_dim=cfg_m.latent_dim,
            K=K, dt=cfg_m.dt,
            time_L=cfg_m.time_L,
        )
        ae.load_state_dict(ae_ck['model_state'])
        ae.eval()

        return LLAEModel(
            ae=ae, theta_dim=theta_dim,
            shared_dim=cfg_t.shared_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_t.freq_L,
        )

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        th, u_laplace_norm = batch
        B, K, _, N, _ = u_laplace_norm.shape
        u_true = self._laplace_to_u(u_laplace_norm)
        U_rec, z_hat_pred, z_hat_true = self.model(th, u_true)
        loss, metrics = self.model.loss(u_true, U_rec, z_hat_pred, z_hat_true)

        with torch.no_grad():
            l2rel = ((U_rec.float() - u_true).flatten(1).norm(dim=1)
                     / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,  on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/l2rel', l2rel, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        th, u_laplace_norm = batch
        u_true = self._laplace_to_u(u_laplace_norm)
        U_rec, z_hat_pred, z_hat_true = self.model(th, u_true)
        loss, metrics = self.model.loss(u_true, U_rec, z_hat_pred, z_hat_true)
        l2rel = ((U_rec.float() - u_true).flatten(1).norm(dim=1)
                 / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,  on_epoch=True, prog_bar=True)
        self.log('val/l2rel', l2rel, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _laplace_to_u(self, u_laplace_norm: torch.Tensor) -> torch.Tensor:
        """Reconstruit U(t) non-normalisé depuis les spectres (B, K, 2, N, N)."""
        B, K, _, N, _ = u_laplace_norm.shape
        laplace = self.model.laplace
        NN = N * N
        re  = u_laplace_norm[:, :, 0].reshape(B, K, NN).float()
        im  = u_laplace_norm[:, :, 1].reshape(B, K, NN).float()
        z_hat = torch.complex(re, im)
        return laplace.inverse_transform(z_hat, self.model.Nt).reshape(B, self.model.Nt, N, N)

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=cfg_t.lr_surrogate, weight_decay=1e-4,
        )
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
        ds    = dm.dataset
        cfg_m = self.cfg.model
        cfg_t = self.cfg.training
        checkpoint.update({
            'model_type':  'LLAEModel',
            'K':           self.model.K,
            'Nt':          self.model.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  cfg_m.latent_dim,
            'dt':          cfg_m.dt,
            'time_L':      cfg_m.time_L,
            'shared_dim':  cfg_t.shared_dim,
            'head_dim':    cfg_t.head_dim,
            'n_trunk':     cfg_t.n_trunk,
            'n_head':      cfg_t.n_head,
            'freq_L':      cfg_t.freq_L,
            'U_mean':      ds.U_mean,
            'U_std':       ds.U_std,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
        })
