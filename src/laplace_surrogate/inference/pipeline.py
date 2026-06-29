"""
pipeline.py — Interface d'inférence unifiée pour tous les surrogates transitoires.

Usage
-----
    from laplace_surrogate.inference.pipeline import InferencePipeline

    pipe = InferencePipeline.from_checkpoint('checkpoints/SLAEModel_best.pt')
    U_pred = pipe.predict([[k, A, C]])   # (B, Nt, N, N) float32

Supporte les backends (détection automatique depuis model_type) :
  - SLAEModel    (pipeline SLAE — AE spatial + surrogate)
  - LLAEModel    (pipeline LLAE — AE latent + surrogate)
  - SLAESVDModel (SLAE + compression SVD des latents Laplace)
  - LLAESVDModel (LLAE + compression SVD des latents temporels)
  - SLAETuckerModel (SLAE + compression Tucker des latents Laplace)
  - LLAETuckerModel (LLAE + compression Tucker des latents Laplace)
  - CorrectionAE (post-traitement UNet, enchaîné avec SLAEModel)
"""
from __future__ import annotations

import numpy as np
import torch


class InferencePipeline:
    """
    Encapsule un surrogate chargé depuis un checkpoint .pt.

    Attributs publics
    -----------------
    model      : module PyTorch (eval, sur device)
    ckpt       : dict complet du checkpoint
    device     : torch.device
    model_type : str  (ex. 'SLAEModel')
    """

    def __init__(self, model: torch.nn.Module, ckpt: dict, device: torch.device):
        self.model      = model
        self.ckpt       = ckpt
        self.device     = device
        self.model_type = ckpt.get('model_type', 'SLAEModel')

    # ------------------------------------------------------------------
    # Constructeur principal
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device=None) -> 'InferencePipeline':
        """
        Charge un surrogate depuis un fichier .pt.

        Parameters
        ----------
        ckpt_path : chemin vers le fichier .pt
        device    : torch.device, str ou None (auto-détect GPU/CPU)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_type = ckpt.get('model_type', 'SLAEModel')
        model      = _build_model(model_type, ckpt, device)

        # CorrectionAE : charger le surrogate associé et exposer ses stats
        if model_type == 'CorrectionAE':
            from laplace_surrogate.models.corrector import CorrectedSLAEModel
            surr_pipe = cls.from_checkpoint(ckpt['surrogate_ckpt'], device)
            model     = CorrectedSLAEModel(surr_pipe.model, model).to(device)
            ckpt['theta_mean'] = surr_pipe.ckpt['theta_mean']
            ckpt['theta_std']  = surr_pipe.ckpt['theta_std']
            ckpt['dt']         = surr_pipe.ckpt.get('dt', 1.0)
            ckpt['alpha_t']    = surr_pipe.ckpt.get('alpha_t', 0.0)
            ckpt['lam']        = surr_pipe.ckpt.get('lam', 1e-6)
            model.eval()
            return cls(model, ckpt, device)

        # Supporte les deux formats : direct (.pt custom) et Lightning (.ckpt)
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            # Lightning sauvegarde sous 'state_dict' avec le préfixe 'model.'
            state = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items()
                     if k.startswith('model.')}
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        return cls(model, ckpt, device)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        theta_raw,
        dt: float | None = None,
        alpha_t: float = 0.0,
        lam: float = 1e-6,
        rule: str = 'trap',
        k_max: int | None = None,
    ) -> np.ndarray:
        """
        Prédit U(t) pour un batch de theta.

        Parameters
        ----------
        theta_raw : array-like (B, theta_dim) ou (theta_dim,) — valeurs physiques
        dt        : pas de temps (si None, lu depuis le checkpoint)

        Returns
        -------
        U_pred : np.ndarray (B, Nt, N, N) float32 — valeurs physiques
        """
        ckpt   = self.ckpt
        device = self.device

        theta_mean = torch.tensor(ckpt['theta_mean'], dtype=torch.float32, device=device)
        theta_std  = torch.tensor(ckpt['theta_std'],  dtype=torch.float32, device=device)

        theta_t = torch.tensor(np.asarray(theta_raw, dtype=np.float32), device=device)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        theta_norm = (theta_t - theta_mean) / theta_std

        mtype = self.model_type

        if mtype in ('SLAEModel', 'CorrectionAE'):
            dt_eff  = dt if dt is not None else float(ckpt.get('dt', 1.0))
            alpha_t = float(ckpt.get('alpha_t', alpha_t))
            lam     = float(ckpt.get('lam', lam))
            U_pred  = self.model.generate(
                theta_norm, dt=dt_eff, alpha_t=alpha_t, lam=lam, rule=rule, k_max=k_max
            )
            return U_pred.cpu().numpy()

        elif mtype == 'LLAEModel':
            U_pred = self.model.generate(theta_norm)
            U_mean = torch.as_tensor(ckpt['U_mean'], dtype=torch.float32, device=device)
            U_std  = torch.as_tensor(ckpt['U_std'],  dtype=torch.float32, device=device)
            U_pred = U_pred * U_std + U_mean
            return U_pred.cpu().numpy()

        else:  # SLAE/LLAE SVD & Tucker — generate() retourne déjà U physique
            U_pred = self.model.generate(theta_norm)
            return U_pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Construction du modèle depuis le checkpoint
# ---------------------------------------------------------------------------

def _build_model(model_type: str, ckpt: dict, device: torch.device) -> torch.nn.Module:
    if model_type == 'SLAEModel':
        from laplace_surrogate.models.slae_surrogate import SLAEModel
        return SLAEModel(
            K           = ckpt['K'],
            Nt          = ckpt['Nt'],
            N           = ckpt['N'],
            theta_dim   = ckpt['theta_dim'],
            latent_dim  = ckpt['latent_dim'],
            hidden_dim  = ckpt['hidden_dim'],
            head_dim    = ckpt.get('head_dim', 128),
            n_trunk     = ckpt.get('n_trunk', 4),
            n_head      = ckpt.get('n_head', 2),
            freq_L      = ckpt.get('freq_L', 8),
            surr_freq_L = ckpt.get('surr_freq_L', 6),
        ).to(device)

    elif model_type == 'LLAEModel':
        import math
        from laplace_surrogate.models.llae import LLAE
        from laplace_surrogate.models.llae_surrogate import LLAEModel
        learnable_laplace = bool(ckpt.get('learnable_laplace', False))
        alpha_t           = float(ckpt.get('alpha_t', math.exp(-2.0)))
        lam               = float(ckpt.get('lam',     math.exp(-2.0)))
        ae_dummy = LLAE(
            N=ckpt['N'], Nt=ckpt['Nt'], latent_dim=ckpt['latent_dim'], K=ckpt['K'],
            dt=ckpt['dt'], time_L=ckpt.get('time_L', 8),
            learnable_laplace=learnable_laplace, alpha_t=alpha_t, lam=lam,
        ).to(device)
        hidden_dim = ckpt.get('hidden_dim') or ckpt.get('shared_dim', 256)
        return LLAEModel(
            ae=ae_dummy, theta_dim=ckpt['theta_dim'],
            hidden_dim=hidden_dim, head_dim=ckpt['head_dim'],
            n_trunk=ckpt['n_trunk'], n_head=ckpt['n_head'], freq_L=ckpt['freq_L'],
            learnable_laplace=learnable_laplace,
        ).to(device)

    elif model_type == 'LLAESVDModel':
        from laplace_surrogate.models.llae_svd_surrogate import LLAESVDModel
        hidden_dim = ckpt.get('hidden_dim') or ckpt.get('shared_dim', 256)
        return LLAESVDModel(
            N          = ckpt['N'],
            Nt         = ckpt['Nt'],
            theta_dim  = ckpt['theta_dim'],
            latent_dim = ckpt['latent_dim'],
            k_svd      = ckpt['k_svd'],
            K          = ckpt['K'],
            dt         = ckpt['dt'],
            time_L     = ckpt.get('time_L', 8),
            hidden_dim = hidden_dim,
            head_dim   = ckpt['head_dim'],
            n_trunk    = ckpt['n_trunk'],
            n_head     = ckpt['n_head'],
            freq_L     = ckpt['freq_L'],
        ).to(device)

    elif model_type == 'SLAESVDModel':
        from laplace_surrogate.models.slae_svd_surrogate import SLAESVDModel
        hidden_dim = ckpt.get('hidden_dim') or ckpt.get('shared_dim', 256)
        return SLAESVDModel(
            K           = ckpt['K'],
            Nt          = ckpt['Nt'],
            N           = ckpt['N'],
            theta_dim   = ckpt['theta_dim'],
            latent_dim  = ckpt['latent_dim'],
            k_svd       = ckpt['k_svd'],
            freq_L      = ckpt['freq_L'],
            hidden_dim  = hidden_dim,
            head_dim    = ckpt['head_dim'],
            n_trunk     = ckpt['n_trunk'],
            n_head      = ckpt['n_head'],
            surr_freq_L = ckpt.get('surr_freq_L', 6),
            dt          = ckpt.get('dt', 1.0),
            alpha_t     = ckpt.get('alpha_t', 0.007),
            lam         = ckpt.get('lam', 3e-5),
        ).to(device)

    elif model_type == 'LLAETuckerModel':
        from laplace_surrogate.models.llae_tucker_surrogate import LLAETuckerModel
        hidden_dim = ckpt.get('hidden_dim') or ckpt.get('shared_dim', 256)
        return LLAETuckerModel(
            N          = ckpt['N'],
            Nt         = ckpt['Nt'],
            theta_dim  = ckpt['theta_dim'],
            latent_dim = ckpt['latent_dim'],
            r_s        = ckpt['r_s'],
            r_z        = ckpt['r_z'],
            K          = ckpt['K'],
            dt         = ckpt['dt'],
            time_L     = ckpt.get('time_L', 8),
            hidden_dim = hidden_dim,
            head_dim   = ckpt['head_dim'],
            n_trunk    = ckpt['n_trunk'],
            n_head     = ckpt['n_head'],
            freq_L     = ckpt['freq_L'],
        ).to(device)

    elif model_type == 'SLAETuckerModel':
        from laplace_surrogate.models.slae_tucker_surrogate import SLAETuckerModel
        hidden_dim = ckpt.get('hidden_dim') or ckpt.get('shared_dim', 256)
        return SLAETuckerModel(
            K           = ckpt['K'],
            Nt          = ckpt['Nt'],
            N           = ckpt['N'],
            theta_dim   = ckpt['theta_dim'],
            latent_dim  = ckpt['latent_dim'],
            r_s         = ckpt['r_s'],
            r_z         = ckpt['r_z'],
            freq_L      = ckpt['freq_L'],
            hidden_dim  = hidden_dim,
            head_dim    = ckpt['head_dim'],
            n_trunk     = ckpt['n_trunk'],
            n_head      = ckpt['n_head'],
            surr_freq_L = ckpt.get('surr_freq_L', 6),
            dt          = ckpt.get('dt', 1.0),
            alpha_t     = ckpt.get('alpha_t', 0.007),
            lam         = ckpt.get('lam', 3e-5),
        ).to(device)

    elif model_type == 'CorrectionAE':
        from laplace_surrogate.models.corrector import CorrectionAE
        ae = CorrectionAE(N=ckpt['N'], base_ch=ckpt['base_ch']).to(device)
        ae.load_state_dict(ckpt['model_state'])
        ae.eval()
        return ae  # CorrectedSLAEModel assemblé dans from_checkpoint

    else:
        raise ValueError(f"model_type inconnu : {model_type!r}")
