"""
train_surrogate.py — Phase 2 : entraînement end-to-end du surrogate θ→z→U(t).

Le DataModule doit être initialisé (setup) avant la construction du module,
car l'initialisation du SLAEModel/LLAEModel dépend des stats du dataset.

Usage :
    PYTHONPATH=src python scripts/train_surrogate.py model=slae training=surrogate_slae
    PYTHONPATH=src python scripts/train_surrogate.py model=llae training=surrogate_llae
    PYTHONPATH=src python scripts/train_surrogate.py model=slae training=surrogate_slae training.ae_ckpt=checkpoints/slae_ld64_K16_g0.0.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig


_SURROGATE_MODULES = {
    'slae':  ('laplace_surrogate.lightning.slae_surrogate_module',  'SLAESurrogateLightningModule',  'SLAEModel'),
    'llae':  ('laplace_surrogate.lightning.llae_surrogate_module',  'LLAESurrogateLightningModule',  'LLAEModel'),
    'lslae': ('laplace_surrogate.lightning.lslae_surrogate_module', 'LSLAESurrogateLightningModule', 'LSLAEModel'),
}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import importlib
    from pathlib import Path
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    from laplace_surrogate.data.datamodule import TransientDataModule

    import torch
    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    model_name = cfg.model.name
    if model_name not in _SURROGATE_MODULES:
        raise ValueError(f"Pas de surrogate pour model.name='{model_name}'. Choix : {list(_SURROGATE_MODULES)}")
    mod_path, cls_name, model_cls = _SURROGATE_MODULES[model_name]
    LightningModule = getattr(importlib.import_module(mod_path), cls_name)

    # Lit K depuis le checkpoint AE, avant dm.setup() qui en a besoin
    from laplace_surrogate.lightning.ckpt_utils import peek_ae_hparams
    from omegaconf import open_dict
    ae_hparams = peek_ae_hparams(cfg.training.ae_ckpt)
    with open_dict(cfg):
        cfg.model.K = ae_hparams['K']

    dm = TransientDataModule(cfg, mode='surrogate')
    dm.setup()

    module = LightningModule(cfg, dm)

    ae_stem  = Path(cfg.training.ae_ckpt).stem
    cfg_t    = cfg.training
    surr_tag = f"t{cfg_t.n_trunk}h{cfg_t.n_head}"
    tag      = f"{model_cls}__{ae_stem}__{surr_tag}"
    run_name = f"{tag}_surr"

    logger = WandbLogger(project=cfg.project, name=run_name, group=f"{model_name}_surr")

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
        precision='16-mixed',
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[ckpt_cb, early_stop],
        log_every_n_steps=20,
    )
    ckpt_path = cfg.training.get('ckpt_path', None)
    trainer.fit(module, dm, ckpt_path=ckpt_path)

    actual_ckpt = Path(ckpt_cb.best_model_path).name
    logger.experiment.summary["ckpt_filename"] = actual_ckpt


if __name__ == "__main__":
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, weights_only=False, **kw: _orig_load(*a, weights_only=False, **kw)
    main()
