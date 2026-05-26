"""
surrogate_module.py — LightningModule pour l'entraînement end-to-end du surrogate SLAE θ→z (phase 2).

Pré-requis : le DataModule doit avoir appelé setup() avant la construction de ce module,
car l'initialisation du modèle dépend des stats du dataset (U_mean, U_std, s_list).
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

    # ------------------------------------------------------------------

    def _build_model(self, dm):
        from laplace_surrogate.models.slae import SLAE
        from laplace_surrogate.models.slae_surrogate import SLAEModel

        cfg_m = self.cfg.model
        cfg_t = self.cfg.training
        cfg_d = self.cfg.data

        ds        = dm.dataset
        N         = ds.N
        Nt        = ds.Nt
        K         = ds.K
        theta_dim = ds.theta_dim

        # Charger l'AE gelé pour initialiser le décodeur partagé
        ae_ck = torch.load(cfg_t.ae_ckpt, map_location='cpu', weights_only=False)
        ae = SLAE(N=N, latent_dim=cfg_t.latent_dim, beta=cfg_m.beta, freq_L=cfg_m.freq_L)
        # Lightning sauvegarde l'état sous 'state_dict' avec le préfixe 'model.'
        raw_sd = ae_ck['state_dict']
        ae_sd  = {k[len('model.'):]: v for k, v in raw_sd.items() if k.startswith('model.')}
        ae.load_state_dict(ae_sd)
        ae.eval()

        model = SLAEModel(
            K=K, Nt=Nt, N=N, theta_dim=theta_dim,
            latent_dim=cfg_t.latent_dim,
            hidden_dim=cfg_t.hidden_dim,
            head_dim=cfg_t.head_dim,
            n_trunk=cfg_t.n_trunk,
            n_head=cfg_t.n_head,
            freq_L=cfg_m.freq_L,
            surr_freq_L=cfg_t.surr_freq_L,
            k_max=cfg_m.get('k_max', None),
            dt=cfg_d.dt,
            alpha_t=cfg_t.alpha_t,
            lam=cfg_t.lam,
        )
        model.set_ae_decoder(ae)
        model.U_mean.copy_(torch.tensor(ds.U_mean, dtype=torch.float32))
        model.U_std.copy_( torch.tensor(ds.U_std,  dtype=torch.float32))
        model.laplace.s_re.data.copy_(torch.tensor(ds.s.real, dtype=torch.float32))
        model.laplace.s_im.data.copy_(torch.tensor(ds.s.imag, dtype=torch.float32))
        model.laplace.requires_grad_(False)

        if cfg_t.lr_decoder > 0.0:
            model.shared_decoder.requires_grad_(True)

        return model

    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _laplace_to_u(U_laplace_norm: torch.Tensor, model) -> torch.Tensor:
        """Reconstruit U(t) physique depuis les spectres normalisés (B, K, 2, N, N)."""
        B, K, _, N, _ = U_laplace_norm.shape
        NN  = N * N
        dev = model.laplace.s_re.device
        re    = U_laplace_norm[:, :, 0].reshape(B, K, NN).float().to(dev)
        im    = U_laplace_norm[:, :, 1].reshape(B, K, NN).float().to(dev)
        z_hat = torch.complex(re, im)                          # (B, K, NN) complex64
        U_norm = model.laplace.inverse_transform(z_hat, model.Nt)  # (B, Nt, NN)
        U_norm = U_norm.reshape(B, model.Nt, N, N)
        return U_norm * model.U_std.to(dev) + model.U_mean.to(dev)

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        th, u_laplace_norm = batch
        u_true = self._laplace_to_u(u_laplace_norm, self.model)
        u_pred = self.model._generate_diff(th)
        loss   = F.mse_loss(u_pred.float(), u_true)

        with torch.no_grad():
            l2rel = ((u_pred.float() - u_true).flatten(1).norm(dim=1)
                     / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('train/loss',  loss,  on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/l2rel', l2rel, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        th, u_laplace_norm = batch
        u_true = self._laplace_to_u(u_laplace_norm, self.model)
        u_pred = self.model._generate_diff(th)
        loss   = F.mse_loss(u_pred.float(), u_true)
        l2rel  = ((u_pred.float() - u_true).flatten(1).norm(dim=1)
                  / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean()

        self.log('val/loss',  loss,  on_epoch=True, prog_bar=True)
        self.log('val/l2rel', l2rel, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg_t = self.cfg.training
        surrogate_params = list(self.model.surrogate.parameters())
        decoder_params   = list(self.model.shared_decoder.parameters())

        param_groups = [{'params': surrogate_params, 'lr': cfg_t.lr_surrogate}]
        if cfg_t.lr_decoder > 0.0:
            param_groups.append({'params': decoder_params, 'lr': cfg_t.lr_decoder})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'val/l2rel',
                'interval':  'epoch',
            },
        }

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        ds    = dm.dataset
        cfg_t = self.cfg.training
        cfg_d = self.cfg.data
        m     = self.model
        checkpoint.update({
            'model_type':  'SLAEModel',
            'K':           m.K,
            'Nt':          m.Nt,
            'N':           ds.N,
            'theta_dim':   ds.theta_dim,
            'latent_dim':  cfg_t.latent_dim,
            'hidden_dim':  cfg_t.hidden_dim,
            'head_dim':    cfg_t.head_dim,
            'n_trunk':     cfg_t.n_trunk,
            'n_head':      cfg_t.n_head,
            'k_max':       self.cfg.model.get('k_max', None),
            'freq_L':      self.cfg.model.freq_L,
            'surr_freq_L': cfg_t.surr_freq_L,
            'dt':          cfg_d.dt,
            'alpha_t':     cfg_t.alpha_t,
            'lam':         cfg_t.lam,
            'theta_mean':  ds.theta_mean,
            'theta_std':   ds.theta_std,
            'test_idx':    np.asarray(dm.test_idx),
        })
