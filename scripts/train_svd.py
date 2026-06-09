"""
train_svd.py — Train FreqSurrogate for every (K, K_SVD, gamma) config in svd_bases.pt.

Loads bases and coefficients from checkpoints/svd_bases.pt (produced by learn_svd.py),
trains one surrogate per config, and saves all model states in a single file:
  checkpoints/svd_surrogates.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laplace_surrogate.models.surrogate_base import FreqSurrogate


def train_one(K, K_SVD, gamma, bases, coeffs, theta_norm,
              hidden_dim, head_dim, n_trunk, n_head, freq_L,
              epochs, lr, batch_size, device):
    key = f'K{K}_g{gamma:.3f}'
    if key not in bases:
        raise KeyError(f'Config {key} not found in svd_bases.pt.')

    K_SVD_MAX = bases[key]['k_svd_max']
    if K_SVD > K_SVD_MAX:
        raise ValueError(f'K_SVD={K_SVD} > k_svd_max={K_SVD_MAX} for config {key}.')

    c = coeffs[key]
    coeffs_train = c['train'][:, :, :, :K_SVD]   # (n_train, K, 2, K_SVD)
    coeffs_test  = c['test'][:, :, :, :K_SVD]
    sorted_train = c['sorted_train']
    sorted_test  = c['sorted_test']
    coeff_mean   = c['coeff_mean'][:, :, :K_SVD]  # (K, 2, K_SVD)
    coeff_std    = c['coeff_std'][:, :, :K_SVD]

    n_train = len(sorted_train)
    n_test  = len(sorted_test)

    coeffs_norm_train = (coeffs_train - coeff_mean) / coeff_std
    coeffs_norm_test  = (coeffs_test  - coeff_mean) / coeff_std

    theta_train    = theta_norm[sorted_train].to(device)
    theta_test     = theta_norm[sorted_test].to(device)
    coeffs_train_t = coeffs_norm_train.reshape(n_train, K, -1).to(device)
    coeffs_test_t  = coeffs_norm_test.reshape(n_test,  K, -1).to(device)

    surrogate = FreqSurrogate(
        theta_dim=3, out_dim=2 * K_SVD, K=K,
        hidden_dim=hidden_dim, head_dim=head_dim,
        n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
    ).to(device)

    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6)

    train_dl = DataLoader(TensorDataset(theta_train, coeffs_train_t),
                          batch_size=batch_size, shuffle=True)

    best_test_loss = float('inf')
    best_state     = None

    for epoch in range(1, epochs + 1):
        surrogate.train()
        train_loss = 0.0
        for theta_b, coeff_b in train_dl:
            pred = surrogate(theta_b)
            loss = nn.functional.mse_loss(pred, coeff_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * theta_b.shape[0]
        train_loss /= n_train

        surrogate.eval()
        with torch.no_grad():
            test_loss = nn.functional.mse_loss(
                surrogate(theta_test), coeffs_test_t).item()
        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in surrogate.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f'  epoch {epoch:4d} | train {train_loss:.4e} | test {test_loss:.4e}')

    return best_state, best_test_loss, coeff_mean, coeff_std


if __name__ == '__main__':
    SVD_CKPT  = 'checkpoints/svd_bases.pt'
    SAVE_PATH = Path('checkpoints/svd_surrogates.pt')

    HIDDEN_DIM = 512
    HEAD_DIM   = 256
    N_TRUNK    = 4
    N_HEAD     = 2
    FREQ_L     = 6
    EPOCHS     = 300
    LR         = 3e-4
    BATCH_SIZE = 64
    SEED       = 42
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    print(f'Device: {device}')

    print(f'Loading {SVD_CKPT} ...')
    ckpt       = torch.load(SVD_CKPT, map_location='cpu')
    configs    = ckpt['configs']      # list of (K, k_svd, gamma)
    bases      = ckpt['bases']
    coeffs     = ckpt['coeffs']
    theta_norm = ckpt['theta_norm']

    print(f'{len(configs)} configs to train:')
    for K, k_svd, g in configs:
        print(f'  K={K}  K_SVD={k_svd}  gamma={g}')

    all_models = {}

    for K, K_SVD, gamma in tqdm(configs, desc='Configs'):
        tag = f'K{K}_ksvd{K_SVD}_g{gamma:.3f}'
        print(f'\n=== {tag} ===')
        torch.manual_seed(SEED)

        best_state, best_loss, coeff_mean, coeff_std = train_one(
            K, K_SVD, gamma, bases, coeffs, theta_norm,
            HIDDEN_DIM, HEAD_DIM, N_TRUNK, N_HEAD, FREQ_L,
            EPOCHS, LR, BATCH_SIZE, device,
        )
        print(f'  best test loss: {best_loss:.4e}')

        all_models[tag] = {
            'model_state': best_state,
            'best_test_loss': best_loss,
            'coeff_mean': coeff_mean,
            'coeff_std':  coeff_std,
            'K': K, 'K_SVD': K_SVD, 'gamma': gamma,
        }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'models': all_models,
        'configs': configs,
        'NT': ckpt['NT'], 'N': ckpt['N'], 'DT': ckpt['DT'],
        'ALPHA_T': ckpt['ALPHA_T'], 'LAM': ckpt['LAM'],
        'theta_mean': ckpt['theta_mean'], 'theta_std': ckpt['theta_std'],
        'bases': {k: {'V_r': v['V_r'], 'V_i': v['V_i'],
                      's_im': v['s_im'], 'k_svd_max': v['k_svd_max']}
                  for k, v in bases.items()},
    }, SAVE_PATH)
    print(f'\nSaved to {SAVE_PATH}')
