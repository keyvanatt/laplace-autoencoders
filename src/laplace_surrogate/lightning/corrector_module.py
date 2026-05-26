"""
corrector_module.py — LightningModule pour l'entraînement du CorrectionAE (phase 3).

Pré-requis : les memmaps U_pred/U_true ont été calculés par le DataModule (mode='corrector').
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig


class CorrectorLightningModule(pl.LightningModule):
    """Phase 3 : UNet résiduel de correction frame-par-frame."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg   = cfg
        self.model = self._build_model()

    # ------------------------------------------------------------------

    def _build_model(self):
        from laplace_surrogate.models.corrector import CorrectionAE
        return CorrectionAE(
            N=self.cfg.model.N,
            base_ch=self.cfg.training.base_ch,
        )

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        pred_frames, true_frames = batch
        B, kt, N, _ = pred_frames.shape
        pred = pred_frames.reshape(B * kt, N, N)
        true = true_frames.reshape(B * kt, N, N)

        lambda_grad = self.cfg.training.lambda_grad
        U_corr = self.model(pred)
        loss, metrics = self.model.loss(U_corr, true, pred, lambda_grad=lambda_grad)

        with torch.no_grad():
            denom      = true.flatten(1).norm(dim=1) + 1e-8          # (B*kt,)
            l2rel      = ((U_corr - true).flatten(1).norm(dim=1) / denom).mean()
            l2rel_surr = ((pred   - true).flatten(1).norm(dim=1) / denom).mean()

        self.log('train/loss',            loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/mse',             metrics['mse'],  on_step=False, on_epoch=True)
        self.log('train/grad_loss',       metrics['grad'], on_step=False, on_epoch=True)
        self.log('train/l2rel',           l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/l2rel_surrogate', l2rel_surr,      on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_frames, true_frames = batch
        B, kt, N, _ = pred_frames.shape
        pred = pred_frames.reshape(B * kt, N, N)
        true = true_frames.reshape(B * kt, N, N)

        lambda_grad = self.cfg.training.lambda_grad
        U_corr = self.model(pred)
        loss, metrics = self.model.loss(U_corr, true, pred, lambda_grad=lambda_grad)
        denom  = true.flatten(1).norm(dim=1) + 1e-8                  # (B*kt,)
        l2rel  = ((U_corr - true).flatten(1).norm(dim=1) / denom).mean()

        self.log('val/loss',  loss,   on_epoch=True, prog_bar=True)
        self.log('val/mse',   metrics['mse'], on_epoch=True)
        self.log('val/l2rel', l2rel,  on_epoch=True, prog_bar=True)

        # Images wandb toutes les 5 epochs, premier batch seulement
        if batch_idx == 0 and (self.current_epoch + 1) % 5 == 0:
            self._log_images(pred, U_corr, true)

        return loss

    def _log_images(self, pred, U_corr, true):
        try:
            import wandb

            def _to_img(t):
                t = t.float().cpu()
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
                return wandb.Image(t.numpy())

            self.logger.experiment.log({
                'images/u_pred':      _to_img(pred[0]),
                'images/u_corrected': _to_img(U_corr[0]),
                'images/u_true':      _to_img(true[0]),
                'images/residual':    _to_img((U_corr[0] - pred[0]).abs()),
            })
        except Exception:
            pass

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg_t.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=8, min_lr=1e-7,
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
        cfg_t = self.cfg.training
        checkpoint.update({
            'model_type':     'CorrectionAE',
            'N':              self.cfg.model.N,
            'base_ch':        cfg_t.base_ch,
            'surrogate_ckpt': cfg_t.surrogate_ckpt,
            'test_idx':       np.asarray(dm.test_idx) if dm is not None else None,
        })
