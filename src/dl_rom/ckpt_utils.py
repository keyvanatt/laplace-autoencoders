"""Chargement d'un checkpoint AE DL-ROM pour la phase surrogate."""
import torch

from laplace_surrogate.lightning.ckpt_utils import _ae_model_cfg, _strip_model_prefix


def peek_ae_t_stride(ae_ckpt_path: str) -> int:
    """Lit data.t_stride depuis les hyper_parameters d'un checkpoint AE (1 si absent)."""
    ae_ck = torch.load(ae_ckpt_path, map_location='cpu', weights_only=False)
    d_cfg = (ae_ck.get('hyper_parameters', {}).get('cfg') or {}).get('data', {})
    return int(d_cfg.get('t_stride', 1) or 1)


def load_dlrom_from_ckpt(ae_ckpt_path: str, N: int, Nt: int):
    """
    Charge un DLROMAE depuis un checkpoint Lightning AE.
    Les hyperparamètres d'architecture (latent_dim, time_L) sont lus depuis
    le checkpoint, pas depuis la config courante.
    Retourne (ae, latent_dim, dt, time_L).
    """
    from dl_rom.dlrom_ae import DLROMAE
    ae_ck      = torch.load(ae_ckpt_path, map_location='cpu', weights_only=False)
    m_cfg      = _ae_model_cfg(ae_ck)
    latent_dim = int(ae_ck.get('latent_dim') or m_cfg['latent_dim'])
    dt         = float(ae_ck.get('dt') or m_cfg.get('dt', 1.0))
    time_L     = int(m_cfg.get('time_L', 8))
    beta       = float(m_cfg.get('beta', 1e-2))
    sd = _strip_model_prefix(ae_ck['state_dict'])
    decoder_norm = 'bn' if any('decoder.deconv' in k and 'running_mean' in k for k in sd) else 'gn'
    ae = DLROMAE(N=N, Nt=Nt, latent_dim=latent_dim, beta=beta, time_L=time_L,
                 decoder_norm=decoder_norm)
    ae.load_state_dict(sd)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, latent_dim, dt, time_L
