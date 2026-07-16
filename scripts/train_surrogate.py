"""
train_surrogate.py — Phase 2 : entraînement end-to-end du surrogate θ→z→U(t).

Le DataModule doit être initialisé (setup) avant la construction du module,
car l'initialisation du SLAEModel/LLAEModel dépend des stats du dataset.

Usage :
    PYTHONPATH=src python scripts/train_surrogate.py model=slae training=surrogate_slae
    PYTHONPATH=src python scripts/train_surrogate.py model=llae training=surrogate_llae
    # Variantes SVD (compression des latents par SVD avant le surrogate) :
    PYTHONPATH=src python scripts/train_surrogate.py model=slae training=surrogate_slae_svd
    PYTHONPATH=src python scripts/train_surrogate.py model=llae training=surrogate_llae_svd
    # Variantes Tucker (compression conjointe fréquence+latent, facteurs figés HOOI) :
    PYTHONPATH=src python scripts/train_surrogate.py model=slae training=surrogate_slae_tucker
    PYTHONPATH=src python scripts/train_surrogate.py model=llae training=surrogate_llae_tucker
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig


_SURROGATE_MODULES = {
    'slae':  ('laplace_surrogate.lightning.slae_surrogate_module',     'SLAESurrogateLightningModule',    'SLAEModel'),
    'llae':  ('laplace_surrogate.lightning.llae_surrogate_module',     'LLAESurrogateLightningModule',    'LLAEModel'),
    'dlrom': ('dl_rom.surrogate_module',                               'DLROMSurrogateLightningModule',   'DLROMModel'),
}

_SVD_SURROGATE_MODULES = {
    'slae':  ('laplace_surrogate.lightning.slae_svd_surrogate_module', 'SLAESVDSurrogateLightningModule', 'SLAESVDModel'),
    'llae':  ('laplace_surrogate.lightning.llae_svd_surrogate_module', 'LLAESVDSurrogateLightningModule', 'LLAESVDModel'),
}

_TUCKER_SURROGATE_MODULES = {
    'slae':  ('laplace_surrogate.lightning.slae_tucker_surrogate_module', 'SLAETuckerSurrogateLightningModule', 'SLAETuckerModel'),
    'llae':  ('laplace_surrogate.lightning.llae_tucker_surrogate_module', 'LLAETuckerSurrogateLightningModule', 'LLAETuckerModel'),
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
    use_tucker = bool(cfg.training.get('r_s', None) is not None
                      and cfg.training.get('r_z', None) is not None)
    use_svd    = bool(cfg.training.get('k_svd', None) is not None
                      and cfg.training.get('lr_V', None) is not None)
    if use_tucker:
        module_map, variant = _TUCKER_SURROGATE_MODULES, '_tucker'
    elif use_svd:
        module_map, variant = _SVD_SURROGATE_MODULES, '_svd'
    else:
        module_map, variant = _SURROGATE_MODULES, ''
    if model_name not in module_map:
        raise ValueError(
            f"Pas de surrogate{variant} pour model.name='{model_name}'. "
            f"Choix : {list(module_map)}"
        )
    mod_path, cls_name, model_cls = module_map[model_name]
    LightningModule = getattr(importlib.import_module(mod_path), cls_name)

    # Lit K, gamma_init et pôles optimaux depuis le checkpoint AE, avant dm.setup()
    # (le DL-ROM n'a pas de transformée de Laplace : rien à lire)
    from laplace_surrogate.lightning.ckpt_utils import peek_ae_hparams
    from omegaconf import open_dict
    if model_name == 'dlrom':
        # Hérite le sous-échantillonnage temporel de l'AE : le nombre de heads du
        # surrogate (K = Nt) doit correspondre à la grille sur laquelle l'AE a été entraîné.
        from dl_rom.ckpt_utils import peek_ae_t_stride
        ae_ts = peek_ae_t_stride(cfg.training.ae_ckpt)
        if int(cfg.data.get('t_stride', 1)) != ae_ts:
            with open_dict(cfg):
                cfg.data.t_stride = ae_ts
            print(f"[train_surrogate] data.t_stride={ae_ts} hérité du checkpoint AE")
    else:
        ae_hparams = peek_ae_hparams(cfg.training.ae_ckpt)
        with open_dict(cfg):
            cfg.model.K          = ae_hparams['K']
            cfg.model.gamma_init = ae_hparams['gamma_init']
            # Propager les pôles optimaux uniquement si l'AE a été entraîné avec
            # (les anciens checkpoints n'ont pas cette info → on ne touche pas le cfg)
            if ae_hparams['optimal_laplace']:
                cfg.model.optimal_laplace      = True
                cfg.model.optimal_laplace_path = ae_hparams['optimal_laplace_path']
            # Idem pour les pôles apprenables : si l'AE est un _ll, le surrogate
            # continue par défaut à raffiner les pôles (hérités et déjà optimisés).
            if ae_hparams['learnable_laplace']:
                cfg.model.learnable_laplace = True

    dm = TransientDataModule(cfg, mode='surrogate')
    dm.setup()

    module = LightningModule(cfg, dm)

    ae_stem  = Path(cfg.training.ae_ckpt).stem
    cfg_t    = cfg.training
    surr_tag = f"t{cfg_t.n_trunk}h{cfg_t.n_head}"
    if use_tucker:
        surr_tag = f"{surr_tag}_rs{cfg_t.r_s}rz{cfg_t.r_z}"
    elif use_svd:
        surr_tag = f"{surr_tag}_ksvd{cfg_t.k_svd}"
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
