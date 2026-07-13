"""
slae_tucker_surrogate_module.py — LightningModule pour le surrogate Tucker-SLAE θ→G_core→U(t).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.

Offline (dans _build_model) :
  1. Charge SLAE pré-entraîné (encoder + décodeur gelés)
  2. Encode toutes les simulations train+val → z_all [ns, K, d_z] (latents Laplace réels)
  3. HOOI(z_train) → U_s [K, r_s], U_z [d_z, r_z]  (figés)
  4. Calcule stats de la core G = U_sᵀ z_k U_z sur les indices train

Batch d'entraînement : (theta_norm, U_norm) — domaine temporel.
Loss = spatial MSE + alpha_lat * latent MSE (sur la core normalisée).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm

from laplace_surrogate.lightning.slae_svd_surrogate_module import _compute_slae_latents
from laplace_surrogate.models import tucker


# ---------------------------------------------------------------------------
# Helper offline
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_slae_tucker_and_stats(z_all, train_local, r_s, r_z, device, n_iter=5):
    """
    HOOI sur z_train (ns_train, K, d_z) réel → (U_s, U_z) figés, puis stats core.
    Retourne (U_s [K, r_s], U_z [d_z, r_z], G_mean [r_s, r_z], G_std [r_s, r_z]).
    """
    Z = z_all[train_local].to(device)
    tqdm.write(f"  HOOI sur {tuple(Z.shape)} → (r_s={r_s}, r_z={r_z})")
    U_s, U_z = tucker.hooi(Z, r_s, r_z, n_iter=n_iter)

    G       = tucker.project(Z, U_s, U_z)            # (ns_train, r_s, r_z)
    G_mean  = G.mean(0)                              # (r_s, r_z)
    G_std   = G.std(0).clamp(min=1e-8)              # (r_s, r_z)
    return U_s.cpu(), U_z.cpu(), G_mean.cpu(), G_std.cpu()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SLAETuckerSurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate Tucker-SLAE θ→G_core→z_k→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.slae_tucker_surrogate import SLAETuckerModel
        from laplace_surrogate.models.slae_surrogate import SLAEModel
        from laplace_surrogate.lightning.ckpt_utils import load_slae_from_ckpt, laplace_reg

        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N, Nt, K  = ds.N, ds.Nt, ds.K
        theta_dim = ds.theta_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Chargement SLAE (encodeur + décodeur gelés) ───────────────────────
        ae, latent_dim, freq_L = load_slae_from_ckpt(cfg_t.ae_ckpt, N)
        self._alpha_t, self._lam = laplace_reg(self.cfg)

        # Modèle intermédiaire pour l'encodage offline (normalisations dataset incluses)
        slae_tmp = SLAEModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=self._alpha_t,
            lam=self._lam,
        )
        slae_tmp.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        slae_tmp.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        slae_tmp.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        slae_tmp.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        slae_tmp.laplace.requires_grad_(False)
        if ds.lap_mean is not None:
            slae_tmp.lap_mean.copy_(torch.tensor(ds.lap_mean, dtype=torch.float32))
            slae_tmp.lap_std.copy_( torch.tensor(ds.lap_std,  dtype=torch.float32))
        slae_tmp.to(device)

        # ── Phase 2 offline : encodage + Tucker ──────────────────────────────
        all_idx     = dm.train_idx + dm.val_idx
        train_local = list(range(len(dm.train_idx)))

        print("Tucker-SLAE surrogate : encodage des latents Laplace...")
        z_all = _compute_slae_latents(
            slae_tmp, ds._U_raw, all_idx, N, Nt, device,
            batch_size=cfg_t.get('encode_batch_size', 32),
        )

        print("Tucker-SLAE surrogate : HOOI + stats core...")
        U_s, U_z, G_mean, G_std = _compute_slae_tucker_and_stats(
            z_all, train_local, cfg_t.r_s, cfg_t.r_z, device,
            n_iter=cfg_t.get('hooi_iter', 5),
        )

        # ── Modèle SLAETuckerModel ────────────────────────────────────────────
        model = SLAETuckerModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            r_s=cfg_t.r_s, r_z=cfg_t.r_z,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=self._alpha_t,
            lam=self._lam,
        )
        # Normalisation dataset
        model.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        model.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        model.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        model.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        model.laplace.requires_grad_(False)
        if ds.lap_mean is not None:
            model.lap_mean.copy_(torch.tensor(ds.lap_mean, dtype=torch.float32))
            model.lap_std.copy_( torch.tensor(ds.lap_std,  dtype=torch.float32))
        # Facteurs de Tucker + stats
        model.set_tucker_factors(U_s, U_z)
        with torch.no_grad():
            model.G_mean.copy_(G_mean.float())
            model.G_std.copy_( G_std.float())

        if cfg_t.lr_decoder > 0.0:
            model.shared_decoder.requires_grad_(True)

        del slae_tmp
        return model

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        U_pred_norm, G_norm, G_true_norm = self.model(theta_norm, U_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(U_norm, U_pred_norm, G_norm, G_true_norm,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        with torch.no_grad():
            l2rel = ((U_pred_norm.float() - U_norm.float()).flatten(1).norm(dim=1)
                     / (U_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        U_pred_norm, G_norm, G_true_norm = self.model(theta_norm, U_norm)

        alpha_lat  = float(self.cfg.training.alpha_lat)
        alpha_spat = float(self.cfg.training.alpha_spat)
        loss, metrics = self.model.loss(U_norm, U_pred_norm, G_norm, G_true_norm,
                                        alpha_lat=alpha_lat, alpha_spat=alpha_spat)

        l2rel = ((U_pred_norm.float() - U_norm.float()).flatten(1).norm(dim=1)
                 / (U_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,            on_epoch=True, prog_bar=True)
        self.log('val/spat',  metrics['spat'], on_epoch=True)
        self.log('val/lat',   metrics['lat'],  on_epoch=True)
        self.log('val/l2rel', l2rel,           on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        param_groups = [
            {'params': self.model.surrogate.parameters(), 'lr': cfg_t.lr_surrogate},
        ]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': self.model.shared_decoder.parameters(),
                                 'lr': cfg_t.lr_decoder})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/l2rel', 'interval': 'epoch'},
        }

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds = dm.dataset
        m  = self.model
        checkpoint.update({
            'model_type':  'SLAETuckerModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           m.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  m.latent_dim,
            'r_s':         m.r_s,
            'r_z':         m.r_z,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'surr_freq_L': m.surr_freq_L,
            'dt':          self.cfg.data.dt,
            'alpha_t':     self._alpha_t,
            'lam':         self._lam,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
