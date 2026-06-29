import sys, os
sys.path.insert(0, 'src')

import torch
import numpy as np
from omegaconf import OmegaConf
from laplace_surrogate.lightning.ae_module import AELightningModule
from laplace_surrogate.data.datamodule import TransientDataModule

ckpt_path = 'checkpoints/llae_ld64_K16_g0.0_ll.ckpt'
module = AELightningModule.load_from_checkpoint(ckpt_path, map_location='cuda', weights_only=False)

cfg = module.cfg
dm = TransientDataModule(cfg, mode='ae')
dm.setup()
val_loader = dm.val_dataloader()

module.eval()
with torch.no_grad():
    for batch in val_loader:
        U = batch.cuda()
        U_rec, z_hat, z, z_rec = module.model(U)
        l2rel = ((U_rec - U).flatten(1).norm(dim=1) / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
        print("Val batch L2rel (eval mode):", l2rel.item() * 100)
        break

module.train()
with torch.no_grad():
    for batch in val_loader:
        U = batch.cuda()
        U_rec, z_hat, z, z_rec = module.model(U)
        l2rel = ((U_rec - U).flatten(1).norm(dim=1) / (U.flatten(1).norm(dim=1) + 1e-8)).mean()
        print("Val batch L2rel (train mode):", l2rel.item() * 100)
        break
