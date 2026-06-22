"""Utilitaires partagés pour charger les checkpoints AE dans les surrogates."""
import torch


def _ae_model_cfg(ae_ck: dict) -> dict:
    """Extrait le sous-dict model depuis les hyper_parameters d'un checkpoint Lightning."""
    return (ae_ck.get('hyper_parameters', {}).get('cfg') or {}).get('model', {})


def _strip_model_prefix(state_dict: dict) -> dict:
    """Supprime le préfixe 'model.' ajouté par Lightning lors de la sauvegarde."""
    return {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}


def peek_ae_hparams(ae_ckpt_path: str) -> dict:
    """
    Lit les hyperparamètres structurels d'un checkpoint AE sans instancier le modèle.
    Retourne {'K', 'gamma_init', 'optimal_laplace', 'optimal_laplace_path'}
    — utilisé par le DataModule avant dm.setup().
    """
    ae_ck = torch.load(ae_ckpt_path, map_location='cpu', weights_only=False)
    m_cfg = _ae_model_cfg(ae_ck)
    return {
        'K':                   int(m_cfg['K']),
        'gamma_init':          float(m_cfg.get('gamma_init', 0.0)),
        'optimal_laplace':     bool(m_cfg.get('optimal_laplace', False)),
        'optimal_laplace_path': m_cfg.get('optimal_laplace_path', None),
    }


def load_slae_from_ckpt(ae_ckpt_path: str, N: int):
    """
    Charge un SLAE depuis un checkpoint Lightning AE.
    Les hyperparamètres d'architecture (latent_dim, freq_L) sont lus depuis
    le checkpoint, pas depuis la config courante.
    Retourne (ae, latent_dim, freq_L).
    """
    from laplace_surrogate.models.slae import SLAE
    ae_ck      = torch.load(ae_ckpt_path, map_location='cpu', weights_only=False)
    m_cfg      = _ae_model_cfg(ae_ck)
    latent_dim = int(ae_ck.get('latent_dim') or m_cfg['latent_dim'])
    freq_L     = int(m_cfg['freq_L'])
    beta       = float(m_cfg.get('beta', 1e-2))
    ae = SLAE(N=N, latent_dim=latent_dim, beta=beta, freq_L=freq_L)
    ae.load_state_dict(_strip_model_prefix(ae_ck['state_dict']))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, latent_dim, freq_L


def load_llae_from_ckpt(ae_ckpt_path: str, N: int, Nt: int, K: int):
    """
    Charge un LLAE depuis un checkpoint Lightning AE.
    Les hyperparamètres d'architecture (latent_dim, dt, time_L, learnable_laplace,
    alpha_t, lam) sont lus depuis le checkpoint, pas depuis la config courante.
    Retourne (ae, latent_dim, dt, time_L).
    """
    from laplace_surrogate.models.llae import LLAE
    ae_ck             = torch.load(ae_ckpt_path, map_location='cpu', weights_only=False)
    m_cfg             = _ae_model_cfg(ae_ck)
    latent_dim        = int(ae_ck.get('latent_dim') or m_cfg['latent_dim'])
    dt                = float(ae_ck.get('dt') or m_cfg.get('dt', 1.0))
    time_L            = int(m_cfg.get('time_L', 8))
    learnable_laplace = bool(m_cfg.get('learnable_laplace', False))
    import math
    alpha_t           = float(m_cfg.get('alpha_t', math.exp(-2.0)))
    lam               = float(m_cfg.get('lam',     math.exp(-2.0)))
    ae = LLAE(N=N, Nt=Nt, latent_dim=latent_dim, K=K, dt=dt, time_L=time_L,
              learnable_laplace=learnable_laplace, alpha_t=alpha_t, lam=lam)
    ae.load_state_dict(_strip_model_prefix(ae_ck['state_dict']))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, latent_dim, dt, time_L
