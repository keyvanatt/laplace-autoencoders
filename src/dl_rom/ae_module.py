"""
ae_module.py — LightningModule pour l'entraînement de l'AE DL-ROM (phase 1).

Hérite d'AELightningModule (optimiseur, checkpointing, stats dataset) et
remplace la construction du modèle et les steps par la variante sans Laplace.
Batch : U_norm de shape (B, Nt, N, N) — même _TimeSeqDataset que le LLAE.
"""
import torch
from omegaconf import DictConfig

from laplace_surrogate.lightning.ae_module import AELightningModule


class DLROMAELightningModule(AELightningModule):
    """Phase 1 : entraînement de l'autoencoder DL-ROM."""

    def _build_model(self):
        from dl_rom.dlrom_ae import DLROMAE
        return DLROMAE(
            N=self.cfg.model.N,
            Nt=self.cfg.model.Nt,
            latent_dim=self.cfg.model.latent_dim,
            beta=self.cfg.model.beta,
            time_L=self.cfg.model.time_L,
        )

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        U             = batch                       # (B, Nt, N, N)
        U_rec, z      = self.model(U)
        loss, metrics = self.model.loss(U, U_rec, z)
        with torch.no_grad():
            l2rel = ((U_rec - U).flatten(1).norm(dim=1)
                     / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
        self.log('train/loss',  loss,             on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/recon', metrics['recon'], on_step=False, on_epoch=True)
        self.log('train/ridge', metrics['ridge'], on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,            on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        U             = batch
        U_rec, z      = self.model(U)
        loss, metrics = self.model.loss(U, U_rec, z)
        l2rel = ((U_rec - U).flatten(1).norm(dim=1)
                 / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
        self.log('val/loss',  loss,             on_epoch=True, prog_bar=True)
        self.log('val/recon', metrics['recon'], on_epoch=True)
        self.log('val/ridge', metrics['ridge'], on_epoch=True)
        self.log('val/l2rel', l2rel,            on_epoch=True, prog_bar=True)
        return loss
