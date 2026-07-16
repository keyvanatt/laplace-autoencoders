"""
evaluate.py — Évaluation d'un surrogate sur le test set.

Usage :
    PYTHONPATH=src python scripts/evaluate.py eval.ckpt_path=checkpoints/SLAEModel__slae_ld64_K16_g0.0__t4h2.pt
    PYTHONPATH=src python scripts/evaluate.py eval.ckpt_path=checkpoints/CorrectionAE__SLAEModel__slae_ld64_K16_g0.0__t4h2__ch16.pt eval.n_gifs=5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wandb

    from laplace_surrogate.data.dataset import TransientDataset
    from laplace_surrogate.inference.pipeline import InferencePipeline
    from laplace_surrogate.utils.visualization import animate_comparaison

    cfg_d  = cfg.data
    cfg_ev = cfg.eval

    ckpt_path = cfg_ev.ckpt_path
    n_gifs    = int(cfg_ev.get('n_gifs', 3))
    out_dir   = cfg_ev.get('out_dir', 'plots/')
    k_max_ov  = cfg_ev.get('k_max', None)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Chargement dataset (sans Laplace pour l'évaluation)
    # -----------------------------------------------------------------------
    dataset = TransientDataset(
        cfg_d.data_path,
        laplace=False,
        dt=cfg_d.dt,
        interp_size=cfg_d.interp_size,
    )

    split     = np.load(cfg_d.split_path)
    test_idx  = split['test_idx'].tolist()
    train_idx = [i for i in range(dataset.ns) if i not in set(test_idx)]
    dataset.fit(train_idx)

    # -----------------------------------------------------------------------
    # Pipeline d'inférence
    # -----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe   = InferencePipeline.from_checkpoint(ckpt_path, device)
    # Modèles entraînés sur une grille temporelle sous-échantillonnée (dlrom t_stride>1) :
    # la vérité terrain est comparée sur la même grille.
    t_stride = int(pipe.ckpt.get('t_stride', 1))

    run_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    wandb.init(project=cfg.project, name=f"eval_{run_name}", config={
        'ckpt_path': ckpt_path,
        'n_test':    len(test_idx),
    })

    # -----------------------------------------------------------------------
    # Prédiction + métriques
    # -----------------------------------------------------------------------
    l2rel_list = []
    U_true_list, U_pred_list = [], []

    theta_raw = dataset.theta.numpy()
    U_raw     = dataset._U_raw  # (ns, Nt, H, W) mmap

    print(f"Évaluation sur {len(test_idx)} samples...")
    for idx in test_idx:
        theta_i = theta_raw[idx:idx+1]
        kw = {} if k_max_ov is None else {'k_max': k_max_ov}
        U_p = pipe.predict(theta_i, **kw)[0]           # (Nt, N, N)
        U_t = np.asarray(U_raw[idx][::t_stride], dtype=np.float32) # (Nt, H, W)

        if U_t.shape[-1] != U_p.shape[-1]:
            import torch.nn.functional as F_nn
            Nt_t, H, W = U_t.shape
            N = U_p.shape[-1]
            U_t = F_nn.interpolate(
                torch.from_numpy(U_t).unsqueeze(1),
                size=(N, N), mode='bilinear', align_corners=False,
            ).squeeze(1).numpy()

        denom = np.linalg.norm(U_t)
        if denom < 1e-12:
            continue
        l2rel = np.linalg.norm(U_p - U_t) / denom * 100.0
        l2rel_list.append(float(l2rel))
        U_true_list.append(U_t)
        U_pred_list.append(U_p)

    l2rel_arr = np.array(l2rel_list)
    median_l2 = float(np.median(l2rel_arr))
    mean_l2   = float(np.mean(l2rel_arr))
    p90_l2    = float(np.percentile(l2rel_arr, 90))

    print(f"L2rel  median={median_l2:.2f}%  mean={mean_l2:.2f}%  p90={p90_l2:.2f}%")

    # -----------------------------------------------------------------------
    # Sauvegarde JSON
    # -----------------------------------------------------------------------
    metrics = {
        'ckpt':   ckpt_path,
        'n_test': len(l2rel_list),
        'median': median_l2,
        'mean':   mean_l2,
        'p90':    p90_l2,
        'l2rel':  l2rel_list,
    }
    json_path = os.path.join(out_dir, f"{run_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques → {json_path}")

    wandb.summary['l2rel_median'] = median_l2
    wandb.summary['l2rel_mean']   = mean_l2
    wandb.summary['l2rel_p90']    = p90_l2

    # -----------------------------------------------------------------------
    # Histogramme
    # -----------------------------------------------------------------------
    hist_path = os.path.join(out_dir, f"{run_name}_l2rel_hist.png")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(l2rel_arr, bins=40, color='steelblue', edgecolor='white')
    ax.axvline(median_l2, color='orange', linestyle='--', label=f'median {median_l2:.1f}%')
    ax.set_xlabel('L2rel (%)')
    ax.set_ylabel('count')
    ax.set_title(run_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"Histogramme → {hist_path}")
    wandb.log({'l2rel_hist': wandb.Image(hist_path)})

    # -----------------------------------------------------------------------
    # GIFs (best / median / worst)
    # -----------------------------------------------------------------------
    if n_gifs > 0 and len(l2rel_list) > 0:
        order = np.argsort(l2rel_arr)
        picks = {}
        picks['best']   = int(order[0])
        picks['worst']  = int(order[-1])
        picks['median'] = int(order[len(order) // 2])
        for rank in range(1, max(0, n_gifs - 3)):
            picks[f'rank{rank+1}'] = int(order[rank])

        for tag, pos in picks.items():
            gif_path = os.path.join(out_dir, f"{run_name}_{tag}.gif")
            animate_comparaison(
                U_true_list[pos], U_pred_list[pos],
                gif_path,
                fps=10,
                title_a='U_true', title_b='U_pred',
                title_fn=lambda t, l=l2rel_list[pos]: f't={t}  L2rel={l:.1f}%',
            )
            print(f"GIF {tag} → {gif_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
