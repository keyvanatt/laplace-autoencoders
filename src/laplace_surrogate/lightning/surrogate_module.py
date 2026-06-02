"""
surrogate_module.py — LightningModule pour l'entraînement end-to-end du surrogate SLAE θ→z (phase 2).

Batch : (theta_norm, U_norm) avec U_norm = (U - U_mean) / U_std  [domaine temporel].
Loss spatiale sur U_norm.
Supervision latente : forward_transform(U_norm) → normalise avec lap_mean/lap_std → encoder.
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig


class SLAESurrogateLightningModule(pl.LightningModule):
    """Phase 2 : entraînement end-to-end surrogate SLAE θ→z→U(t) (décodeur gelé ou fine-tuné)."""

    def __init__(self, cfg: DictConfig, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        self.cfg   = cfg
        self.model = self._build_model(datamodule)

    def _build_model(self, dm):
        from laplace_surrogate.lightning.ckpt_utils import load_slae_from_ckpt
        from laplace_surrogate.models.slae_surrogate import SLAEModel

        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N         = ds.N
        Nt        = ds.Nt
        K         = ds.K
        theta_dim = ds.theta_dim

        ae, latent_dim, freq_L = load_slae_from_ckpt(cfg_t.ae_ckpt, N)

        model = SLAEModel.from_ae(
            ae, latent_dim=latent_dim, freq_L=freq_L,
            K=K, Nt=Nt, theta_dim=theta_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            surr_freq_L=cfg_t.freq_L,
            dt=cfg_d.dt,
            alpha_t=cfg_t.alpha_t,
            lam=cfg_t.lam,
        )
        model.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        model.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        # Pôles ds.s — mêmes que ceux utilisés pour calculer _lap_mean/_lap_std
        model.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        model.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        model.laplace.requires_grad_(False)

        if cfg_t.lr_decoder > 0.0:
            model.shared_decoder.requires_grad_(True)

        # Stats de normalisation Laplace (K, 2, 1, 1) : pour undo dans _pred_u_norm et _encode_latents
        self._lap_mean = torch.tensor(ds._lap_mean[0], dtype=torch.float32)
        self._lap_std  = torch.tensor(ds._lap_std[0],  dtype=torch.float32)

        # Encodeur gelé pour supervision latente
        self._encoder = ae.encoder
        return model

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _pred_u_norm(self, theta_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        θ_norm → (U_pred_norm, z_pred)
        Decoder output (u_laplace_norm) → undo lap norm → L⁻¹(ds.s) → U_pred_norm.
        """
        B      = theta_norm.shape[0]
        device = theta_norm.device

        M, z_pred = self.model._forward_k(theta_norm)
        # M : (B, N², K) complex, en espace u_laplace_norm = (L(U_norm) - lap_mean) / lap_std
        lap_mean_re = self._lap_mean[:, 0, 0, 0].to(device)  # (K,)
        lap_mean_im = self._lap_mean[:, 1, 0, 0].to(device)
        lap_std_re  = self._lap_std[:, 0, 0, 0].to(device)
        lap_std_im  = self._lap_std[:, 1, 0, 0].to(device)

        M_re = M.real * lap_std_re + lap_mean_re  # (B, N², K) = L_re(U_norm)
        M_im = M.imag * lap_std_im + lap_mean_im
        M_unnorm = torch.complex(M_re, M_im)

        u_pred_norm = self.model.laplace.inverse_transform(
            M_unnorm.permute(0, 2, 1), self.model.Nt
        ).reshape(B, self.model.Nt, self.model.N, self.model.N)

        return u_pred_norm, z_pred

    @torch.no_grad()
    def _encode_latents(self, u_norm: torch.Tensor) -> torch.Tensor:
        """
        u_norm : (B, Nt, N, N) → z_true : (B, K, latent_dim)
        Calcule L(U_norm) via forward_transform, normalise avec lap_mean/lap_std,
        puis encode avec l'encodeur gelé.
        """
        B, Nt, N, _ = u_norm.shape
        K      = self.model.K
        device = u_norm.device

        # Laplace forward : (B, Nt, N²) → (B, K, N²) complex
        u_flat = u_norm.reshape(B, Nt, N * N).float()
        u_hat  = self.model.laplace.forward_transform(u_flat)  # (B, K, N²)

        # Normalise comme u_laplace_norm (espace d'entraînement de l'encodeur)
        re = u_hat.real.reshape(B, K, N, N)
        im = u_hat.imag.reshape(B, K, N, N)
        u_lap = torch.stack([re, im], dim=2)                   # (B, K, 2, N, N)
        lap_mean = self._lap_mean.to(device).unsqueeze(0)      # (1, K, 2, 1, 1)
        lap_std  = self._lap_std.to(device).unsqueeze(0)
        u_lap_norm = (u_lap - lap_mean) / lap_std

        encoder = self._encoder.to(device)
        frames  = u_lap_norm.reshape(B * K, 2, N, N)
        fr = torch.tensor(
            [k / max(K - 1, 1) for k in range(K)],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(B, -1).reshape(B * K)
        z = encoder(frames, fr)
        return z.view(B, K, -1)

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        th, u_true_norm = batch                          # (B, 3), (B, Nt, N, N)
        u_pred_norm, z_pred = self._pred_u_norm(th)

        spat_loss = F.mse_loss(u_pred_norm.float(), u_true_norm.float())
        z_true    = self._encode_latents(u_true_norm)
        lat_loss  = F.mse_loss(z_pred.float(), z_true.float())
        alpha_lat = float(self.cfg.training.alpha_lat)
        loss      = spat_loss + alpha_lat * lat_loss

        with torch.no_grad():
            l2rel = ((u_pred_norm.float() - u_true_norm.float()).flatten(1).norm(dim=1)
                     / (u_true_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,      on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/spat',  spat_loss, on_step=False, on_epoch=True)
        self.log('train/lat',   lat_loss,  on_step=False, on_epoch=True)
        self.log('train/l2rel', l2rel,     on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        th, u_true_norm = batch
        u_pred_norm, z_pred = self._pred_u_norm(th)

        spat_loss = F.mse_loss(u_pred_norm.float(), u_true_norm.float())
        z_true    = self._encode_latents(u_true_norm)
        lat_loss  = F.mse_loss(z_pred.float(), z_true.float())
        alpha_lat = float(self.cfg.training.alpha_lat)
        loss      = spat_loss + alpha_lat * lat_loss

        l2rel = ((u_pred_norm.float() - u_true_norm.float()).flatten(1).norm(dim=1)
                 / (u_true_norm.float().flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,      on_epoch=True, prog_bar=True)
        self.log('val/spat',  spat_loss, on_epoch=True)
        self.log('val/lat',   lat_loss,  on_epoch=True)
        self.log('val/l2rel', l2rel,     on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        param_groups = [{'params': self.model.surrogate.parameters(), 'lr': cfg_t.lr_surrogate}]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': self.model.shared_decoder.parameters(), 'lr': cfg_t.lr_decoder})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/l2rel', 'interval': 'epoch'},
        }

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds = dm.dataset
        m  = self.model
        checkpoint.update({
            'model_type':  'SLAEModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           m.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  m.latent_dim,
            'hidden_dim':  m.hidden_dim,
            'head_dim':    m.head_dim,
            'n_trunk':     m.n_trunk,
            'n_head':      m.n_head,
            'freq_L':      m.freq_L,
            'surr_freq_L': m.surr_freq_L,
            'dt':          self.cfg.data.dt,
            'alpha_t':     self.cfg.training.alpha_t,
            'lam':         self.cfg.training.lam,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
            'model_state': {k[len('model.'):]: v
                            for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')},
        })
