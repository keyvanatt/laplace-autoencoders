"""
llae_svd_surrogate_module.py — LightningModule pour le surrogate SVD-LLAE θ→Ĝ_SVD→U(t).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module.

Offline (dans _build_model) :
  1. Charge LLAE pré-entraîné (encoder gelé)
  2. Encode toutes les simulations train+val → z_all [ns, Nt, D]
  3. Transformée de Laplace → ẑ_train [ns, K, D], puis SVD complexe tronquée du
     dépliage mode-latent → V [D, k_svd]  (base FIGÉE, cas r_s = K de Tucker)
  4. Calcule stats de Ĝ = ẑ @ V

Batch d'entraînement : (theta_norm, U_norm) — domaine temporel.
Loss = spatial MSE + alpha_lat * latent MSE (sur Ĝ normalisé).
"""
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm

from laplace_surrogate.models.tucker import _truncated_left


# ---------------------------------------------------------------------------
# Helpers offline
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_llae_latents(encoder, U_raw, indices, N, Nt,
                           U_mean, U_std, device, batch_size=32):
    """Encode les simulations aux indices donnés → (ns, Nt, D) sur CPU."""
    import torch.nn.functional as F
    zs = []
    for start in tqdm(range(0, len(indices), batch_size), desc='  Encoding latents', leave=False):
        idx_batch = indices[start:start + batch_size]
        U_batch   = torch.from_numpy(U_raw[idx_batch].copy()).float()
        B, Nt_raw = U_batch.shape[:2]
        if U_batch.shape[-1] != N:
            U_batch = F.interpolate(
                U_batch.reshape(B * Nt_raw, 1, U_batch.shape[-2], U_batch.shape[-1]),
                size=(N, N), mode='bilinear', align_corners=False,
            ).reshape(B, Nt_raw, N, N)
        U_norm   = (U_batch.to(device) - U_mean) / U_std
        frames   = U_norm.reshape(B * Nt_raw, 1, N, N)
        t        = torch.arange(Nt_raw, dtype=U_norm.dtype, device=device) / max(Nt_raw - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt_raw, 1)
        z        = encoder(frames, t_ratios).view(B, Nt_raw, -1)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def _compute_svd_and_stats(laplace, z_all, train_local, k_svd, device, batch_size=64):
    """SVD complexe tronquée sur les latents de Laplace ẑ(s_k) → V [D, k_svd], puis stats Ĝ.

    V est la base du **mode latent** du tenseur de Laplace Ẑ ∈ (ns, K, D) : les
    k_svd premiers vecteurs singuliers gauches de son dépliage mode-2 (D, ns*K).
    C'est exactement l'initialisation HOSVD du facteur U_z de Tucker, et à
    r_s = K (facteur fréquence unitaire, donc mode fréquence non compressé) HOOI
    converge vers cette même base : LLAE-SVD est le cas r_s = K de LLAE-Tucker.

    La base est complexe et FIGÉE, comme les facteurs Tucker.
    """
    _, _, D = z_all.shape

    z_hat_list = []
    for start in tqdm(range(0, len(train_local), batch_size),
                      desc='  Laplace fwd (fit V)', leave=False):
        pos_batch = train_local[start:start + batch_size]
        z_hat     = laplace.forward_transform(z_all[pos_batch].to(device))
        z_hat_list.append(z_hat.cpu())
    z_hat_train = torch.cat(z_hat_list, dim=0)                      # (ns, K, D) complexe

    # Dépliage mode-latent (D, ns*K), puis vecteurs singuliers gauches — cf. tucker._truncated_left
    unfold = z_hat_train.permute(2, 0, 1).reshape(D, -1)
    tqdm.write(f"  SVD complexe (domaine de Laplace) sur {tuple(unfold.shape)} -> k_svd={k_svd}")
    V = _truncated_left(unfold, k_svd)                              # (D, k_svd) complexe
    del unfold

    G_hat_train = z_hat_train @ V                                   # (ns, K, k_svd)
    G_hat_ri    = torch.stack([G_hat_train.real, G_hat_train.imag], dim=-1).float()
    G_hat_mean  = G_hat_ri.mean(0)
    G_hat_std   = G_hat_ri.std(0).clamp(min=1e-8)
    return V, G_hat_mean, G_hat_std


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class LLAESVDSurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate SVD-LLAE θ→Ĝ_SVD→z→U(t)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model, self._encoder = self._build_model(datamodule)

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.llae_svd_surrogate import LLAESVDModel
        from laplace_surrogate.laplace_transform.learnable import LearnableLaplace
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

        # ── Laplace avec pôles du dataset (pour decode U_laplace_norm si besoin) ─
        # Ici le batch est en domaine temporel (U_norm), pas besoin d'inversion dataset.

        # ── Phase 2 offline : encodage + SVD ─────────────────────────────────
        all_idx     = dm.train_idx + dm.val_idx
        train_local = list(range(len(dm.train_idx)))

        print("SVD-LLAE surrogate : encodage des latents...")
        z_all = _compute_llae_latents(
            ae.encoder, ds._U_raw, all_idx, N, Nt,
            torch.as_tensor(ds.U_mean).to(device), torch.as_tensor(ds.U_std).to(device), device,
            batch_size=cfg_t.get('encode_batch_size', 32),
        )

        print("SVD-LLAE surrogate : SVD + stats Ĝ...")
        V, G_hat_mean, G_hat_std = _compute_svd_and_stats(
            ae.laplace, z_all, train_local, cfg_t.k_svd, device,
        )

        # ── Modèle LLAESVDModel ───────────────────────────────────────────────
        model = LLAESVDModel(
            N=N, Nt=Nt, theta_dim=theta_dim,
            latent_dim=latent_dim, k_svd=cfg_t.k_svd,
            K=K, dt=dt, time_L=time_L,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_t.freq_L,
        )
        model.load_ae_decoder(ae)
        model.load_laplace_from_ae(ae)
        model.set_svd_basis(V)
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
            'model_type':  'LLAESVDModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  self._ae_latent_dim,
            'k_svd':       m.k_svd,
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
