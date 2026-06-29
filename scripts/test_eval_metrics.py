import sys, os
sys.path.insert(0, 'src')

import torch
import numpy as np
from laplace_surrogate.inference.pipeline_ae import InferencePipelineAE

ckpt_path = 'checkpoints/llae_ld64_K16_g0.0_ll.ckpt'
pipe = InferencePipelineAE.from_checkpoint(ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu')

ds = pipe.dataset
train_idx = [i for i in range(ds.ns) if i not in set(pipe.test_idx)] if hasattr(ds, 'ns') else []
# Actually, the pipeline has test_idx.
split = np.load('dataset/split.npz')
test_idx = split['test_idx'].tolist()
train_idx = [i for i in range(ds.ns) if i not in set(test_idx)]

idx = train_idx[0:4]
U_raw = np.stack([ds._load_u(i) for i in idx])

print("Evaluating with model.eval()...")
pipe.model.eval()
U_rec_eval_phys = pipe.reconstruct(U_raw)

# Calculate physical error
l2rel_eval_phys = [np.linalg.norm(U_rec_eval_phys[i] - U_raw[i]) / np.linalg.norm(U_raw[i]) * 100.0 for i in range(len(idx))]
print("L2rel physical:", l2rel_eval_phys)

# Calculate normalized error
with torch.no_grad():
    u_norm = (U_raw - ds.U_mean) / ds.U_std
    u_t    = torch.from_numpy(u_norm).float().to(pipe.device)
    U_rec_norm, _, _, _ = pipe.model(u_t)
    U_rec_norm = U_rec_norm.cpu().numpy()
    
    l2rel_eval_norm = [np.linalg.norm(U_rec_norm[i] - u_norm[i]) / np.linalg.norm(u_norm[i]) for i in range(len(idx))]
    print("L2rel normalized:", l2rel_eval_norm)

print("Evaluating with model.train()...")
pipe.model.train()
U_rec_train_phys = pipe.reconstruct(U_raw)
l2rel_train_phys = [np.linalg.norm(U_rec_train_phys[i] - U_raw[i]) / np.linalg.norm(U_raw[i]) * 100.0 for i in range(len(idx))]
print("L2rel physical train mode:", l2rel_train_phys)

with torch.no_grad():
    U_rec_norm_train, _, _, _ = pipe.model(u_t)
    U_rec_norm_train = U_rec_norm_train.cpu().numpy()
    l2rel_train_norm = [np.linalg.norm(U_rec_norm_train[i] - u_norm[i]) / np.linalg.norm(u_norm[i]) for i in range(len(idx))]
    print("L2rel normalized train mode:", l2rel_train_norm)
