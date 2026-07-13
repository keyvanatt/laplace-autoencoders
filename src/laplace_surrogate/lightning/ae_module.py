"""
ae_module.py — LightningModule pour l'entraînement de l'autoencoder (phase 1).

Supporte SLAE (cfg.model.name='slae') et LLAE (cfg.model.name='llae').
"""
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


class AELightningModule(pl.LightningModule):
    """Phase 1 : entraînement de l'autoencoder SLAE ou LLAE."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg   = cfg
        self.model = self._build_model()

    # ------------------------------------------------------------------

    def _build_model(self):
        name = self.cfg.model.name
        if name == 'slae':
            from laplace_surrogate.models.slae import SLAE
            return SLAE(
                N=self.cfg.model.N,
                latent_dim=self.cfg.model.latent_dim,
                beta=self.cfg.model.beta,
                freq_L=self.cfg.model.freq_L,
            )
        elif name == 'llae':
            from laplace_surrogate.models.llae import LLAE
            return LLAE(
                N=self.cfg.model.N,
                Nt=self.cfg.model.Nt,
                latent_dim=self.cfg.model.latent_dim,
                K=self.cfg.model.K,
                dt=self.cfg.model.dt,
                beta=self.cfg.model.beta,
                beta_latent=self.cfg.model.beta_latent,
                gamma_init=self.cfg.model.gamma_init,
                time_L=self.cfg.model.time_L,
                learnable_laplace=self.cfg.model.learnable_laplace,
                alpha_t=float(self.cfg.model.alpha_t),
                lam=float(self.cfg.model.lam),
            )
        else:
            raise ValueError(f"Modèle AE inconnu : {name!r}. Attendu : slae | llae")

    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        if hasattr(dm, 'train_dataset') and hasattr(dm.train_dataset, 'reshuffle'):
            dm.train_dataset.reshuffle()

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.cfg.model.name == 'llae':
            U                       = batch                           # (B, Nt, N, N)
            U_rec, z_hat, z, z_rec  = self.model(U)
            loss, metrics           = self.model.loss(U, U_rec, z_hat, z, z_rec)
            with torch.no_grad():
                l2rel = ((U_rec - U).flatten(1).norm(dim=1)
                         / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
            self.log('train/loss',    loss,                 on_step=True,  on_epoch=True, prog_bar=True)
            self.log('train/recon',   metrics['recon'],     on_step=False, on_epoch=True)
            self.log('train/lat_rec', metrics['lat_rec'],   on_step=False, on_epoch=True)
            self.log('train/ridge',   metrics['ridge'],     on_step=False, on_epoch=True)
            self.log('train/l2rel',   l2rel,                on_step=False, on_epoch=True, prog_bar=True)
        else:  # slae
            u, freq_ratio = batch
            u_hat, z      = self.model(u, freq_ratio)
            loss, metrics = self.model.loss(u, u_hat, z)
            with torch.no_grad():
                l2rel = ((u_hat - u).flatten(1).norm(dim=1)
                         / (u.flatten(1).norm(dim=1) + 1e-8)).mean()
            self.log('train/loss',  loss,                  on_step=True,  on_epoch=True, prog_bar=True)
            self.log('train/recon', metrics['recon_loss'], on_step=False, on_epoch=True)
            self.log('train/ridge', metrics['ridge'],      on_step=False, on_epoch=True)
            self.log('train/l2rel', l2rel,                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.model.name == 'llae':
            U                       = batch
            U_rec, z_hat, z, z_rec  = self.model(U)
            loss, metrics           = self.model.loss(U, U_rec, z_hat, z, z_rec)
            l2rel = ((U_rec - U).flatten(1).norm(dim=1)
                     / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
            self.log('val/loss',    loss,               on_epoch=True, prog_bar=True)
            self.log('val/recon',   metrics['recon'],   on_epoch=True)
            self.log('val/lat_rec', metrics['lat_rec'], on_epoch=True)
            self.log('val/ridge',   metrics['ridge'],   on_epoch=True)
            self.log('val/l2rel',   l2rel,              on_epoch=True, prog_bar=True)
        else:  # slae
            u, freq_ratio = batch
            u_hat, z      = self.model(u, freq_ratio)
            loss, metrics = self.model.loss(u, u_hat, z)
            l2rel = ((u_hat - u).flatten(1).norm(dim=1)
                     / (u.flatten(1).norm(dim=1) + 1e-8)).mean()
            self.log('val/loss',  loss,                 on_epoch=True, prog_bar=True)
            self.log('val/recon', metrics['recon_loss'], on_epoch=True)
            self.log('val/ridge', metrics['ridge'],     on_epoch=True)
            self.log('val/l2rel', l2rel,                on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        """Suit les pôles de Laplace appris (s_k, α_t, λ) — LLAE learnable_laplace uniquement."""
        if self.cfg.model.name != 'llae' or not self.cfg.model.get('learnable_laplace', False):
            return
        if self.trainer.sanity_checking:
            return
        exp = getattr(self.logger, 'experiment', None)
        if exp is None or not hasattr(exp, 'log'):
            return
        exp.log(self.model.laplace.log_dict(self.current_epoch), step=self.global_step)

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg_t.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'val/loss',
                'interval':  'epoch',
            },
        }

    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):
        saved_cfg = checkpoint.get('hyper_parameters', {}).get('cfg')
        if saved_cfg is None:
            return
        saved = OmegaConf.to_container(OmegaConf.create(saved_cfg), resolve=True)
        current = OmegaConf.to_container(self.cfg, resolve=True)
        diffs = []
        for section in ('model', 'training'):
            s_sec = saved.get(section, {})
            c_sec = current.get(section, {})
            for key in set(s_sec) | set(c_sec):
                s_val, c_val = s_sec.get(key), c_sec.get(key)
                if s_val != c_val:
                    diffs.append(f"  {section}.{key}: {s_val} → {c_val}")
        if diffs:
            import warnings
            warnings.warn(
                "Resuming from checkpoint with different hyperparameters:\n"
                + "\n".join(diffs),
                UserWarning,
                stacklevel=2,
            )

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None or dm.dataset is None:
            return
        checkpoint['N']          = dm.dataset.N
        checkpoint['latent_dim'] = self.cfg.model.latent_dim
        checkpoint['val_loss']   = self.trainer.callback_metrics.get('val/loss', float('inf'))
        if hasattr(dm.dataset, 's'):
            checkpoint['s_list'] = dm.dataset.s
        if hasattr(dm.dataset, 'dt'):
            checkpoint['dt'] = dm.dataset.dt
        checkpoint['rule'] = self.cfg.data.get('rule', 'trap')
        if getattr(dm.dataset, 'U_mean', None) is not None:
            checkpoint['U_mean'] = dm.dataset.U_mean
            checkpoint['U_std']  = dm.dataset.U_std
        if getattr(dm.dataset, 'lap_mean', None) is not None:
            checkpoint['lap_mean'] = dm.dataset.lap_mean  # (K, 2, N, N) float32
            checkpoint['lap_std']  = dm.dataset.lap_std
