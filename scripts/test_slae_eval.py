import sys, os
sys.path.insert(0, 'src')

import torch
import numpy as np
from omegaconf import OmegaConf
from laplace_surrogate.lightning.ae_module import AELightningModule
from laplace_surrogate.data.datamodule import TransientDataModule

ckpt_path = 'checkpoints/slae_ld64_K16_g0.0_ol.ckpt'
module = AELightningModule.load_from_checkpoint(ckpt_path, map_location='cpu', weights_only=False)

cfg = module.cfg
dm = TransientDataModule(cfg, mode='ae')
dm.setup()
val_loader = dm.val_dataloader()

module.eval()
with torch.no_grad():
    for batch in val_loader:
        u, freq_ratio = batch
        u_hat, z = module.model(u, freq_ratio)
        l2rel = ((u_hat - u).flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-8)).mean()
        print("SLAE Val batch L2rel (eval mode):", l2rel.item() * 100)
        break
