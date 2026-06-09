"""
train_svd.py — Train one FreqSurrogate per (K, K_SVD, gamma) config in svd_bases.pt.

Loss mirrors the AE surrogate:
  alpha_lat  * MSE(c_pred_norm, c_true_norm)          [latent — SVD coefficients]
  alpha_spat * MSE(U_rec_pred,  U_rec_true)           [spatial — inverse-Laplace reconstruction]

All configs are trained sequentially and saved in a single file: svd_surrogates.pt.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laplace_surrogate.models.surrogate_base import FreqSurrogate
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_from_coeffs(coeffs, V_r, V_i, laplace, Nt):
    """
    coeffs : (B, K, 2, K_SVD) — denormalized SVD coefficients
    returns  (B, Nt, N²)      — reconstructed physical field
    """
    U_hat = (
        torch.einsum('bkj,knj->bkn', coeffs[:, :, 0, :], V_r).to(torch.complex64)
        + 1j * torch.einsum('bkj,knj->bkn', coeffs[:, :, 1, :], V_i).to(torch.complex64)
    )
    return laplace.inverse_transform(U_hat, Nt)   # (B, Nt, N²)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class SVDSurrogateLightningModule(pl.LightningModule):

    def __init__(self, K, K_SVD, gamma, V_r, V_i, NT, DT, ALPHA_T, LAM,
                 coeff_mean, coeff_std,
                 hidden_dim, head_dim, n_trunk, n_head, freq_L,
                 lr, alpha_lat, alpha_spat):
        super().__init__()
        self.save_hyperparameters(ignore=['V_r', 'V_i', 'coeff_mean', 'coeff_std'])

        self.surrogate = FreqSurrogate(
            theta_dim=3, out_dim=2 * K_SVD, K=K,
            hidden_dim=hidden_dim, head_dim=head_dim,
            n_trunk=n_trunk, n_head=n_head, freq_L=freq_L,
        )

        # Fixed SVD bases and normalization stats as buffers
        self.register_buffer('V_r',        V_r)           # (K, N², K_SVD)
        self.register_buffer('V_i',        V_i)
        self.register_buffer('coeff_mean', coeff_mean)    # (K, 2, K_SVD)
        self.register_buffer('coeff_std',  coeff_std)

        self.laplace = LearnableLaplace(
            K=K, dt=DT, Nt=NT, gamma_init=gamma,
            learnable=False, alpha_t=ALPHA_T, lam=LAM,
        )

        self.K = K; self.K_SVD = K_SVD; self.NT = NT
        self.lr = lr
        self.alpha_lat  = alpha_lat
        self.alpha_spat = alpha_spat

    def _loss(self, batch):
        theta_b, coeffs_b = batch              # (B, 3), (B, K, 2*K_SVD)
        pred = self.surrogate(theta_b)         # (B, K, 2*K_SVD)

        lat_loss = F.mse_loss(pred, coeffs_b)

        # Denormalize → reconstruct U
        B  = theta_b.shape[0]
        cm = self.coeff_mean   # (K, 2, K_SVD)
        cs = self.coeff_std
        pred_raw = pred.view(B, self.K, 2, self.K_SVD) * cs + cm
        true_raw = coeffs_b.view(B, self.K, 2, self.K_SVD) * cs + cm

        self.laplace.to(self.device).eval()
        U_pred = reconstruct_from_coeffs(pred_raw, self.V_r, self.V_i, self.laplace, self.NT)
        U_true = reconstruct_from_coeffs(true_raw, self.V_r, self.V_i, self.laplace, self.NT)
        spat_loss = F.mse_loss(U_pred, U_true)

        loss = self.alpha_lat * lat_loss + self.alpha_spat * spat_loss
        return loss, lat_loss, spat_loss

    def training_step(self, batch, batch_idx):
        loss, lat, spat = self._loss(batch)
        self.log('train/loss', loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/lat',  lat,  on_step=False, on_epoch=True)
        self.log('train/spat', spat, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, lat, spat = self._loss(batch)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/lat',  lat,  on_epoch=True)
        self.log('val/spat', spat, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.surrogate.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler,
                             'monitor': 'val/loss', 'interval': 'epoch'},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    SVD_CKPT   = 'checkpoints/svd_bases.pt'
    SAVE_PATH  = Path('checkpoints/svd_surrogates.pt')
    WANDB_PROJECT = 'laplace-svd-surrogate'

    HIDDEN_DIM  = 512
    HEAD_DIM    = 256
    N_TRUNK     = 4
    N_HEAD      = 2
    FREQ_L      = 6
    ALPHA_LAT   = 1.0
    ALPHA_SPAT  = 1.0
    EPOCHS      = 300
    LR          = 3e-4
    BATCH_SIZE  = 64
    PATIENCE    = 40
    SEED        = 42
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f'Device: {DEVICE}')

    print(f'Loading {SVD_CKPT} ...')
    ckpt       = torch.load(SVD_CKPT, map_location='cpu')
    configs    = ckpt['configs']
    bases      = ckpt['bases']
    coeffs_all = ckpt['coeffs']
    theta_norm = ckpt['theta_norm']
    NT         = ckpt['NT']
    DT         = ckpt['DT']
    ALPHA_T    = ckpt['ALPHA_T']
    LAM        = ckpt['LAM']

    print(f'{len(configs)} configs to train:')
    for K, k_svd, g in configs:
        print(f'  K={K}  K_SVD={k_svd}  gamma={g}')

    all_models = {}

    for K, K_SVD, gamma in configs:
        tag     = f'K{K}_ksvd{K_SVD}_g{gamma:.3f}'
        key_base = f'K{K}_g{gamma:.3f}'

        print(f'\n=== {tag} ===')
        torch.manual_seed(SEED)

        # ---- Extract data for this config ----
        c = coeffs_all[key_base]
        V_r = bases[key_base]['V_r'][:, :, :K_SVD]   # (K, N², K_SVD)
        V_i = bases[key_base]['V_i'][:, :, :K_SVD]

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

        theta_train    = theta_norm[sorted_train]
        theta_test     = theta_norm[sorted_test]
        coeffs_train_t = coeffs_norm_train.reshape(n_train, K, -1)
        coeffs_test_t  = coeffs_norm_test.reshape(n_test,  K, -1)

        train_dl = DataLoader(TensorDataset(theta_train, coeffs_train_t),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(TensorDataset(theta_test,  coeffs_test_t),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # ---- Lightning module ----
        module = SVDSurrogateLightningModule(
            K=K, K_SVD=K_SVD, gamma=gamma,
            V_r=V_r, V_i=V_i,
            NT=NT, DT=DT, ALPHA_T=ALPHA_T, LAM=LAM,
            coeff_mean=coeff_mean, coeff_std=coeff_std,
            hidden_dim=HIDDEN_DIM, head_dim=HEAD_DIM,
            n_trunk=N_TRUNK, n_head=N_HEAD, freq_L=FREQ_L,
            lr=LR, alpha_lat=ALPHA_LAT, alpha_spat=ALPHA_SPAT,
        )

        # ---- Logger + Trainer ----
        logger = WandbLogger(project=WANDB_PROJECT, name=tag, reinit=True)

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            callbacks=[EarlyStopping(monitor='val/loss', patience=PATIENCE, mode='min')],
            log_every_n_steps=10,
        )

        trainer.fit(module, train_dl, val_dl)

        best_state = {k: v.cpu().clone() for k, v in module.surrogate.state_dict().items()}
        best_loss  = float(trainer.callback_metrics.get('val/loss', torch.tensor(float('nan'))))

        all_models[tag] = {
            'model_state':     best_state,
            'best_val_loss':   best_loss,
            'coeff_mean':      coeff_mean,
            'coeff_std':       coeff_std,
            'K': K, 'K_SVD': K_SVD, 'gamma': gamma,
        }

        logger.experiment.finish()

    # ---- Save all ----
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'models':  all_models,
        'configs': configs,
        'NT': NT, 'N': ckpt['N'], 'DT': DT,
        'ALPHA_T': ALPHA_T, 'LAM': LAM,
        'theta_mean': ckpt['theta_mean'], 'theta_std': ckpt['theta_std'],
        'bases': {k: {'V_r': v['V_r'], 'V_i': v['V_i'],
                      's_im': v['s_im'], 'k_svd_max': v['k_svd_max']}
                  for k, v in bases.items()},
    }, SAVE_PATH)
    print(f'\nSaved to {SAVE_PATH}')
