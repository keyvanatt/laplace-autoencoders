"""
llae_tucker_surrogate_module.py — LightningModule pour le surrogate Tucker-LLAE θ→Ĝ_core→U(t).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.

Offline (dans _build_model) :
  1. Charge LLAE pré-entraîné (encoder gelé)
  2. Encode toutes les simulations train+val → z_all [ns, Nt, d_z]
  3. Ẑ_train = Laplace(z_train) [ns_train, K, d_z] complexe
  4. HOOI(Ẑ_train) → U_s [K, r_s], U_z [d_z, r_z]  (figés)
  5. Calcule stats de la core G = U_sᴴ Ẑ U_z

Batch d'entraînement : (theta_norm, U_norm) — domaine temporel.
Loss = spatial MSE + alpha_lat * latent MSE (sur la core normalisée).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm

from laplace_surrogate.lightning.llae_svd_surrogate_module import _compute_llae_latents
from laplace_surrogate.models import tucker


# ---------------------------------------------------------------------------
# Helper offline
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_tucker_and_stats(laplace, z_all, train_local, r_s, r_z, device,
                              n_iter=5, batch_size=64):
    """
    Ẑ_train = Laplace(z_train), HOOI → (U_s, U_z) complexes figés, puis stats core.
    Retourne (U_s [K, r_s], U_z [d_z, r_z], G_mean [r_s, r_z, 2], G_std [r_s, r_z, 2]).
    """
    # Ẑ_train (ns_train, K, d_z) complexe
    Z_hat_list = []
    for start in tqdm(range(0, len(train_local), batch_size),
                      desc='  Laplace fwd (Ẑ)', leave=False):
        pos_batch = train_local[start:start + batch_size]
        z_batch   = z_all[pos_batch].to(device)
        Z_hat_list.append(laplace.forward_transform(z_batch).cpu())
    Z_hat = torch.cat(Z_hat_list, dim=0).to(device)

    tqdm.write(f"  HOOI sur {tuple(Z_hat.shape)} → (r_s={r_s}, r_z={r_z})")
    U_s, U_z = tucker.hooi(Z_hat, r_s, r_z, n_iter=n_iter)

    G        = tucker.project(Z_hat, U_s, U_z)                 # (ns_train, r_s, r_z) complexe
    G_ri     = torch.stack([G.real, G.imag], dim=-1).float()
    G_mean   = G_ri.mean(0)
    G_std    = G_ri.std(0).clamp(min=1e-8)
    return U_s.cpu(), U_z.cpu(), G_mean.cpu(), G_std.cpu()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class LLAETuckerSurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate Tucker-LLAE θ→Ĝ_core→z→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model, self._encoder = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.llae_tucker_surrogate import LLAETuckerModel
        from laplace_surrogate.lightning.ckpt_utils import load_llae_from_ckpt

        cfg_t = self.cfg.training

        ds        = dm.dataset
        N, Nt     = ds.N, ds.Nt
        K         = self.cfg.model.K   # ds.K=0 car laplace=False en mode surrogate
        theta_dim = ds.theta_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Chargement LLAE (encoder gelé) ───────────────────────────────────
        ae, latent_dim, dt, time_L = load_llae_from_ckpt(cfg_t.ae_ckpt, N, Nt, K)
        self._ae_latent_dim = latent_dim
        self._ae_dt         = dt
        self._ae_time_L     = time_L
        ae.to(device)

        # ── Phase 2 offline : encodage + Tucker ──────────────────────────────
        all_idx     = dm.train_idx + dm.val_idx
        train_local = list(range(len(dm.train_idx)))

        print("Tucker-LLAE surrogate : encodage des latents...")
        z_all = _compute_llae_latents(
            ae.encoder, ds._U_raw, all_idx, N, Nt,
            torch.as_tensor(ds.U_mean).to(device), torch.as_tensor(ds.U_std).to(device), device,
            batch_size=cfg_t.get('encode_batch_size', 32),
        )

        print("Tucker-LLAE surrogate : HOOI + stats core...")
        U_s, U_z, G_hat_mean, G_hat_std = _compute_tucker_and_stats(
            ae.laplace, z_all, train_local, cfg_t.r_s, cfg_t.r_z, device,
            n_iter=cfg_t.get('hooi_iter', 5),
        )

        # ── Modèle LLAETuckerModel ────────────────────────────────────────────
        model = LLAETuckerModel(
            N=N, Nt=Nt, theta_dim=theta_dim,
            latent_dim=latent_dim, r_s=cfg_t.r_s, r_z=cfg_t.r_z,
            K=K, dt=dt, time_L=time_L,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_t.freq_L,
        )
        model.load_ae_decoder(ae)
        model.load_laplace_from_ae(ae)
        model.set_tucker_factors(U_s, U_z)
        model.set_normalization(G_hat_mean, G_hat_std, ds.U_mean, ds.U_std,
                                ds.theta_mean, ds.theta_std)

        encoder = ae.encoder.cpu()
        del ae

        return model, encoder

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_seq(self, U_norm: torch.Tensor) -> torch.Tensor:
        """(B, Nt, N, N) → z (B, Nt, latent_dim) via encoder gelé."""
        B, Nt, N, _ = U_norm.shape
        frames   = U_norm.reshape(B * Nt, 1, N, N)
        t        = torch.arange(Nt, dtype=U_norm.dtype, device=U_norm.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        z        = self._encoder(frames, t_ratios)
        return z.view(B, Nt, -1)

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        z_true = self._encode_seq(U_norm)

        U_pred, G_hat_norm, G_hat_true_norm = self.model(theta_norm, z_true)
        alpha_lat = float(self.cfg.training.get('alpha_lat', 1.0))
        loss, metrics = self.model.loss(U_norm, U_pred, G_hat_norm, G_hat_true_norm,
                                        alpha_lat=alpha_lat)

        with torch.no_grad():
            l2rel = ((U_pred - U_norm).flatten(1).norm(dim=1)
                     / (U_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  metrics['spat'], on_step=False, on_epoch=True)
        self.log('train/lat',   metrics['lat'],  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,           on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta_norm, U_norm = batch
        z_true = self._encode_seq(U_norm)

        U_pred, G_hat_norm, G_hat_true_norm = self.model(theta_norm, z_true)
        alpha_lat = float(self.cfg.training.get('alpha_lat', 1.0))
        loss, metrics = self.model.loss(U_norm, U_pred, G_hat_norm, G_hat_true_norm,
                                        alpha_lat=alpha_lat)

        l2rel = ((U_pred - U_norm).flatten(1).norm(dim=1)
                 / (U_norm.flatten(1).norm(dim=1) + 1e-8)).mean()

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
            {'params': self.model.decoder.parameters(),   'lr': cfg_t.lr_decoder},
        ]
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
            'model_type':  'LLAETuckerModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  self._ae_latent_dim,
            'r_s':         m.r_s,
            'r_z':         m.r_z,
            'dt':          self._ae_dt,
            'time_L':      self._ae_time_L,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'U_mean':      ds.U_mean,
            'U_std':       ds.U_std,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
