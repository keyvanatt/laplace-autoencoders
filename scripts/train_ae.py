"""
train_ae.py — Phase 1 : entraînement de l'autoencoder (SLAE ou LLAE).

Usage :
    PYTHONPATH=src python scripts/train_ae.py                    # SLAE par défaut
    PYTHONPATH=src python scripts/train_ae.py model=llae         # LLAE
    PYTHONPATH=src python scripts/train_ae.py model.latent_dim=128
    PYTHONPATH=src python scripts/train_ae.py model.K=8
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig


def _ae_tag(cfg_m) -> str:
    """Construit le tag unique encodant les hps clés du modèle AE."""
    name = cfg_m.name
    ld   = cfg_m.latent_dim
    g    = cfg_m.gamma_init
    ll   = '_ll' if cfg_m.get('learnable_laplace', False) else ''
    ol   = '_ol' if cfg_m.get('optimal_laplace',   False) else ''
    if name == 'slae':
        K = cfg_m.get('K', 'all')
        return f"{name}_ld{ld}_K{K}_g{g}{ll}{ol}"
    elif name == 'llae':
        return f"{name}_ld{ld}_K{cfg_m.K}_g{g}{ll}"
    else:
        return name


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    from laplace_surrogate.lightning.ae_module import AELightningModule
    from laplace_surrogate.data.datamodule import TransientDataModule

    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    dm     = TransientDataModule(cfg, mode='ae')
    module = AELightningModule(cfg)

    from omegaconf import OmegaConf
    tag      = _ae_tag(cfg.model)
    run_name = f"{tag}_ae"
    logger   = WandbLogger(
        project=cfg.project,
        name=run_name,
        group=f"{cfg.model.name}_ae",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename=tag,
        monitor='val/loss',
        save_top_k=1,
        mode='min',
    )
    early_stop = EarlyStopping(
        monitor='val/loss',
        patience=cfg.training.patience,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[ckpt_cb, early_stop],
        log_every_n_steps=50,
    )
    ckpt_path = cfg.training.get('ckpt_path', None)
    trainer.fit(module, dm, ckpt_path=ckpt_path)

    from pathlib import Path
    actual_ckpt = Path(ckpt_cb.best_model_path).name
    logger.experiment.summary["ckpt_filename"] = actual_ckpt


if __name__ == "__main__":
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, weights_only=False, **kw: _orig_load(*a, weights_only=False, **kw)
    main()
