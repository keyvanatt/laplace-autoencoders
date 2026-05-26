"""
train_corrector.py — Phase 3 : entraînement du CorrectionAE (UNet résiduel).

Le DataModule appelle precompute() pour générer les memmaps U_pred/U_true
(ou les réutilise depuis le cache). Le surrogate_ckpt est lu depuis
cfg.training.surrogate_ckpt (configs/training/corrector.yaml).

Usage :
    PYTHONPATH=src python scripts/train_corrector.py training=corrector
    PYTHONPATH=src python scripts/train_corrector.py training=corrector training.epochs=200
    PYTHONPATH=src python scripts/train_corrector.py training=corrector training.surrogate_ckpt=checkpoints/SLAEModel__slae_ld64_K16_g0.0__t4h2.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    from laplace_surrogate.lightning.corrector_module import CorrectorLightningModule
    from laplace_surrogate.data.datamodule import TransientDataModule

    pl.seed_everything(cfg.seed, workers=True)

    from pathlib import Path

    dm     = TransientDataModule(cfg, mode='corrector')
    module = CorrectorLightningModule(cfg)

    surr_stem = Path(cfg.training.surrogate_ckpt).stem
    tag       = f"CorrectionAE__{surr_stem}__ch{cfg.training.base_ch}"
    logger    = WandbLogger(project=cfg.project, name=tag, group='corrector')

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename=tag,
        monitor='val/l2rel',
        save_top_k=1,
        mode='min',
    )
    early_stop = EarlyStopping(
        monitor='val/l2rel',
        patience=cfg.training.patience,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb, early_stop],
        log_every_n_steps=1,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
