"""
pipeline_ae.py — Interface d'inférence pour les AE phase 1 (checkpoints Lightning .ckpt).

Usage
-----
    from laplace_surrogate.inference.pipeline_ae import InferencePipelineAE

    pipe   = InferencePipelineAE.from_checkpoint('checkpoints/llae_ld64_K16_g0.01.ckpt')
    U_rec  = pipe.reconstruct(U_raw)   # (B, Nt, N, N) float32, valeurs physiques

Supporte LLAE et SLAE.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import torch


class InferencePipelineAE:
    """
    Encapsule un AE chargé depuis un checkpoint Lightning .ckpt.

    Attributs publics
    -----------------
    model      : module PyTorch (eval, sur device)
    model_name : str  — 'llae' ou 'slae'
    dataset    : TransientDataset fitté sur le train set
    test_idx   : list[int]  — indices du test set
    device     : torch.device
    """

    def __init__(self, model, model_name, dataset, cfg, test_idx, device, s_list=None):
        self.model      = model
        self.model_name = model_name
        self.dataset    = dataset
        self.cfg        = cfg
        self.test_idx   = test_idx
        self.device     = device
        self._s_list    = s_list   # None pour LLAE, np.ndarray complex128 pour SLAE

    # ------------------------------------------------------------------
    # Constructeur principal
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device=None) -> 'InferencePipelineAE':
        """
        Charge un AE depuis un checkpoint Lightning .ckpt.

        Pour SLAE : lit s_list, lap_mean, lap_std directement depuis le checkpoint
        quand ils sont présents (sauvegardés par on_save_checkpoint). Sinon fallback
        sur dataset.fit() pour rétrocompatibilité avec les anciens checkpoints.
        """
        from laplace_surrogate.lightning.ae_module import AELightningModule
        from laplace_surrogate.data.dataset import TransientDataset

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        # La normalisation du décodeur n'est pas dans la config des anciens
        # checkpoints : on la déduit des poids, comme le fait déjà le pipeline
        # surrogate (_decoder_norm_from_ckpt). Sans cela, un LLAE entraîné avec
        # BatchNorm est reconstruit en GroupNorm et le chargement échoue.
        from laplace_surrogate.inference.pipeline import _decoder_norm_from_ckpt
        from omegaconf import OmegaConf, open_dict

        _probe = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        _cfg = OmegaConf.create(_probe['hyper_parameters']['cfg'])
        with open_dict(_cfg):
            _cfg.model.decoder_norm = _decoder_norm_from_ckpt(_probe)
        del _probe

        module = AELightningModule.load_from_checkpoint(
            ckpt_path, map_location=device, weights_only=False, cfg=_cfg
        )
        module.eval()
        cfg        = module.cfg
        model      = module.model
        model_name = cfg.model.name

        # Lire le checkpoint brut pour les stats éventuellement pré-calculées
        raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        cfg_d   = cfg.data
        laplace = (model_name == 'slae')

        # --- s_list ---
        s_list = None
        if laplace:
            if 's_list' in raw_ckpt:
                s_list = raw_ckpt['s_list'].astype(np.complex128)
            else:
                # Fallback : reconstruire depuis config + données
                K     = cfg.model.K
                _tmp  = np.load(cfg_d.data_path, mmap_mode='r')
                Nt    = _tmp.shape[1]
                del _tmp
                gamma = cfg.model.gamma_init
                s_im  = 2 * np.pi * np.fft.rfftfreq(Nt, d=cfg_d.dt)[:K]
                s_list = (gamma + 1j * s_im).astype(np.complex128)

        dataset = TransientDataset(
            cfg_d.data_path,
            laplace=laplace,
            s_list=s_list,
            rule=cfg_d.rule,
            dt=cfg_d.dt,
            interp_size=cfg_d.interp_size,
        )

        # --- stats de normalisation ---
        _need_fit = True
        if 'U_mean' in raw_ckpt and 'U_std' in raw_ckpt:
            dataset.U_mean = raw_ckpt['U_mean']
            dataset.U_std  = raw_ckpt['U_std']
            _need_fit = False
            
        if laplace and 'lap_mean' in raw_ckpt and 'lap_std' in raw_ckpt:
            # Chemin rapide : stats déjà dans le checkpoint
            dataset.lap_mean = raw_ckpt['lap_mean']
            dataset.lap_std  = raw_ckpt['lap_std']
            if _need_fit:
                dataset.U_mean = None
                dataset.U_std  = None
                _need_fit = False

        if _need_fit:
            split    = np.load(cfg_d.split_path)
            test_idx = split['test_idx'].tolist()
            test_set = set(test_idx)
            non_test = [i for i in range(dataset.ns) if i not in test_set]
            gen      = torch.Generator()
            gen.manual_seed(cfg.seed)
            perm      = torch.randperm(len(non_test), generator=gen).tolist()
            n_train   = int(cfg_d.train_val_split * len(non_test))
            train_idx = [non_test[i] for i in perm[:n_train]]
            print("Calcul des stats de normalisation sur le train set…")
            dataset.fit(train_idx)

        split    = np.load(cfg_d.split_path)
        test_idx = split['test_idx'].tolist()

        return cls(model, model_name, dataset, cfg, test_idx, device, s_list=s_list)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, U_raw: np.ndarray) -> np.ndarray:
        """
        Reconstruit U(t) depuis les valeurs physiques brutes via l'AE.

        Parameters
        ----------
        U_raw : (Nt, N, N) ou (B, Nt, N, N) float32 ndarray — valeurs physiques

        Returns
        -------
        U_rec : (B, Nt, N, N) float32 ndarray — valeurs physiques reconstruites
        """
        if U_raw.ndim == 3:
            U_raw = U_raw[np.newaxis]

        if self.model_name == 'llae':
            return self._reconstruct_llae(U_raw)
        elif self.model_name == 'slae':
            return self._reconstruct_slae(U_raw)
        else:
            raise NotImplementedError(
                f"Modèle non supporté par InferencePipelineAE : {self.model_name!r}. "
                "Attendu : 'llae' | 'slae'"
            )

    # ------------------------------------------------------------------

    def _reconstruct_llae(self, U_raw: np.ndarray) -> np.ndarray:
        """U_raw : (B, Nt, N, N) → (B, Nt, N, N) physique."""
        ds     = self.dataset
        u_norm = (U_raw - ds.U_mean) / ds.U_std
        u_t    = torch.from_numpy(u_norm).float().to(self.device)
        U_rec_norm = self.model(u_t)[0].cpu().numpy()
        return (U_rec_norm * ds.U_std + ds.U_mean).astype(np.float32)

    def _reconstruct_slae(self, U_raw: np.ndarray) -> np.ndarray:
        """U_raw : (B, Nt, N, N) → (B, Nt, N, N) physique."""
        from laplace_surrogate.laplace_transform.forward import laplace_forward_tik
        from laplace_surrogate.laplace_transform.inverse import laplace_inverse_tik

        ds   = self.dataset
        cfg  = self.cfg
        B, Nt, N, _ = U_raw.shape
        K       = cfg.model.K
        dt      = cfg.data.dt
        rule    = cfg.data.rule
        # alpha_t / lam pour l'inversion Tikhonov — DOIVENT correspondre au contour.
        # Pour un contour optimisé offline (_ol), les pôles ont été optimisés
        # CONJOINTEMENT avec (alpha_t, lam), stockés dans optimal_laplace_path.
        # Les réutiliser : sinon on inverse un contour taillé pour un ridge lourd
        # avec un ridge quasi nul → système mal conditionné → erreur explosée.
        # La phase 1 de SLAE est entierement dans le domaine de Laplace et n'inverse
        # jamais : le checkpoint AE ne porte aucun (alpha_t, lam). L'inversion
        # n'apparait qu'ici, a l'evaluation, et le choix appartient donc a l'appelant —
        # les notebooks d'evaluation le posent explicitement en ecrivant cfg.model.alpha_t
        # et cfg.model.lam sur le pipeline. Les valeurs ci-dessous ne sont qu'un repli
        # pour un appel qui n'aurait rien precise ; elles laissent l'inversion quasi nue
        # et ne correspondent a aucun regime utilise dans l'etude.
        alpha_t = float(getattr(cfg.model, 'alpha_t', 0.0))
        lam     = float(getattr(cfg.model, 'lam',     1e-6))
        if getattr(cfg.model, 'optimal_laplace', False):
            opt_path = getattr(cfg.model, 'optimal_laplace_path', None)
            if opt_path and os.path.exists(opt_path):
                _opt    = torch.load(opt_path, map_location='cpu', weights_only=False)
                alpha_t = float(_opt['alpha_t'])
                lam     = float(_opt['lam'])
            else:
                warnings.warn(
                    f"optimal_laplace=True mais optimal_laplace_path introuvable "
                    f"({opt_path!r}) — inversion avec alpha_t={alpha_t}, lam={lam}."
                )
        # Tout le chemin en complex64 sur GPU, comme le surrogate (pipeline.py) et
        # comme LLAE, dont la transformee est interne au modele. Deux choses le
        # permettent : laplace_inverse_tik assemble et resout desormais les equations
        # normales en float64 quelle que soit la precision demandee — c'est la seule
        # etape mal conditionnee, et elle porte sur une matrice Nt×Nt — et l'inversion
        # utilise le ridge reel (7e-3, 3e-5) au lieu du repli quasi nul qui la mettait
        # au bord de la singularite. Les produits couteux, sur la dimension des noeuds,
        # restent en simple precision.
        s_t = torch.tensor(self._s_list, dtype=torch.complex64, device=self.device)
        fr  = torch.tensor([k / max(K - 1, 1) for k in range(K)],
                           dtype=torch.float32, device=self.device)
        lap_mean = torch.as_tensor(ds.lap_mean, dtype=torch.float32, device=self.device)
        lap_std  = torch.as_tensor(ds.lap_std,  dtype=torch.float32, device=self.device)

        U_out = np.empty((B, Nt, N, N), dtype=np.float32)

        for b in range(B):
            # Forward Laplace : (N², Nt) → (N², K) complex
            u_flat = torch.as_tensor(U_raw[b].reshape(Nt, N * N).T,
                                     dtype=torch.float32, device=self.device)
            uhat   = laplace_forward_tik(u_flat, s_t, dt, rule)     # (N², K)

            u3     = uhat.reshape(N, N, K)
            frames = torch.stack([u3.real.permute(2, 0, 1),
                                  u3.imag.permute(2, 0, 1)], dim=1)  # (K, 2, N, N)

            # Encoder / decoder les K frequences en un seul batch. Le decodeur est en
            # eval(), sa BatchNorm lit ses statistiques courantes : le resultat ne
            # depend pas de la taille du batch.
            with torch.no_grad():
                y, _ = self.model((frames - lap_mean) / lap_std, fr)
            rec = y * lap_std + lap_mean

            # (K, 2, N, N) → (N², K) complexe
            uhat_rec_t = torch.complex(rec[:, 0], rec[:, 1]) \
                              .permute(1, 2, 0).reshape(N * N, K).to(torch.complex64)

            # Inverse Laplace : (N², K) → (N², Nt)
            u_rec    = laplace_inverse_tik(uhat_rec_t, s_t, dt, Nt, alpha_t, lam, rule)
            U_out[b] = u_rec.float().cpu().numpy().T.reshape(Nt, N, N)

        return U_out
