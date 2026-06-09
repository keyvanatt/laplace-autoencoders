"""
learn_svd.py — Laplace SVD baseline: bases & reconstruction sweep.

Three 1-D sweeps around a base config (K=16, K_SVD=64, gamma=0.1):
  - gamma  ∈ {0.0, 0.01, 0.1}     (K=16,  K_SVD=64)
  - K      ∈ {8, 16, 32}          (K_SVD=64,  gamma=0.1)
  - K_SVD  ∈ {32, 64, 128}        (K=16,  gamma=0.1)

Output: checkpoints/svd_bases.pt  (single bundled file)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laplace_surrogate.data.dataset import TransientDataset
from laplace_surrogate.laplace_transform.learnable import LearnableLaplace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s_list(K, NT, DT, gamma=0.0):
    s_im = 2 * math.pi * np.fft.fftfreq(NT, d=DT)[:K]
    return [complex(gamma, float(w)) for w in s_im]


def _make_F(s_list, NT, dt, device):
    t = torch.arange(NT, dtype=torch.float64) * dt
    w = torch.ones(NT, dtype=torch.float64); w[0] = 0.5; w[-1] = 0.5
    rows = [dt * w * torch.exp(-torch.tensor(s, dtype=torch.complex128) * t)
            for s in s_list]
    return torch.stack(rows).to(torch.complex64).to(device)


def _laplace_batch(U_np, F, device):
    """U_np: (B, Nt, H, W) float32 → (B, K, N²) complex64 on device."""
    B, Nt, H, W = U_np.shape
    N2 = H * W
    U  = torch.from_numpy(U_np.reshape(B, Nt, N2)).permute(0, 2, 1)
    Uc = U.to(torch.complex64).to(device)
    out = (Uc.reshape(B * N2, Nt) @ F.T).reshape(B, N2, F.shape[0])
    return out.permute(0, 2, 1)   # (B, K, N²)


# ---------------------------------------------------------------------------
# 2-pass randomized SVD
# ---------------------------------------------------------------------------

def compute_svd_onepass(U_raw, s_list, train_idx, k_svd_max, NT, dt,
                        batch_size=64, device='cpu', n_oversample=10):
    """Returns V_r, V_i: (K, N², k_svd_max) float32 on CPU."""
    K       = len(s_list)
    N2      = U_raw.shape[2] * U_raw.shape[3]
    n_train = len(train_idx)
    q       = k_svd_max + n_oversample

    F         = _make_F(s_list, NT, dt, device)
    train_arr = np.sort(train_idx)

    Omega_r = torch.randn(K, N2, q, device=device)
    Omega_i = torch.randn(K, N2, q, device=device)
    Y_r = torch.zeros(K, n_train, q, device=device)
    Y_i = torch.zeros(K, n_train, q, device=device)

    for i in tqdm(range(0, n_train, batch_size), desc=f'SVD K={K} [1/2]', leave=False):
        j   = min(i + batch_size, n_train)
        U_b = U_raw[train_arr[i:j]].astype(np.float32)
        Uh  = _laplace_batch(U_b, F, device)
        Y_r[:, i:j, :] = torch.einsum('bkn,knq->kbq', Uh.real, Omega_r)
        Y_i[:, i:j, :] = torch.einsum('bkn,knq->kbq', Uh.imag, Omega_i)

    del Omega_r, Omega_i

    # Batched QR: (K, n_train, q) in one call
    Q_r, _ = torch.linalg.qr(Y_r)
    Q_i, _ = torch.linalg.qr(Y_i)
    del Y_r, Y_i

    B_r = torch.zeros(K, q, N2, device=device)
    B_i = torch.zeros(K, q, N2, device=device)

    for i in tqdm(range(0, n_train, batch_size), desc=f'SVD K={K} [2/2]', leave=False):
        j   = min(i + batch_size, n_train)
        U_b = U_raw[train_arr[i:j]].astype(np.float32)
        Uh  = _laplace_batch(U_b, F, device)
        B_r += torch.einsum('kbq,bkn->kqn', Q_r[:, i:j, :], Uh.real)
        B_i += torch.einsum('kbq,bkn->kqn', Q_i[:, i:j, :], Uh.imag)

    del Q_r, Q_i

    # Batched SVD on GPU: (K, q, N2) → Vh (K, q, N2)
    _, _, Vt_r = torch.linalg.svd(B_r, full_matrices=False)
    _, _, Vt_i = torch.linalg.svd(B_i, full_matrices=False)
    del B_r, B_i

    V_r = Vt_r[:, :k_svd_max, :].permute(0, 2, 1).contiguous().cpu()  # (K, N², k_svd_max)
    V_i = Vt_i[:, :k_svd_max, :].permute(0, 2, 1).contiguous().cpu()

    return V_r, V_i


def compute_coefficients_onepass(U_raw, idx_list, s_list, V_r, V_i, NT, dt,
                                 batch_size=64, device='cpu'):
    """Returns coeffs (n, K, 2, k_svd) float32, sorted_idx list."""
    K     = len(s_list)
    k_svd = V_r.shape[2]
    n     = len(idx_list)

    F    = _make_F(s_list, NT, dt, device)
    Vr_d = V_r.to(device)
    Vi_d = V_i.to(device)

    sorted_arr = np.sort(idx_list)
    coeffs     = torch.empty(n, K, 2, k_svd)

    for i in tqdm(range(0, n, batch_size), desc='Coefficients', leave=False):
        j   = min(i + batch_size, n)
        U_b = U_raw[sorted_arr[i:j]].astype(np.float32)
        Uh  = _laplace_batch(U_b, F, device)
        coeffs[i:j, :, 0, :] = torch.einsum('bkn,knj->bkj', Uh.real, Vr_d).cpu()
        coeffs[i:j, :, 1, :] = torch.einsum('bkn,knj->bkj', Uh.imag, Vi_d).cpu()

    return coeffs, sorted_arr.tolist()


# ---------------------------------------------------------------------------
# SVD reconstruction evaluation
# ---------------------------------------------------------------------------

def eval_svd_reconstruction(U_raw, test_idx, V_r, V_i, s_list, dt, NT, N,
                            alpha_t, lam, gamma, device, batch_size=32):
    """Returns errors_phys (n_test,) and freq_errors (n_test, K) in %."""
    K     = len(s_list)
    k_svd = V_r.shape[2]
    N2    = N * N

    F     = _make_F(s_list, NT, dt, device)
    V_r_d = V_r.to(device)
    V_i_d = V_i.to(device)

    laplace = LearnableLaplace(K=K, dt=dt, Nt=NT, gamma_init=gamma,
                               learnable=False, alpha_t=alpha_t, lam=lam).to(device)
    laplace.eval()

    errors_phys = []
    freq_errors = []
    test_arr    = np.sort(test_idx)

    with torch.no_grad():
        for start in tqdm(range(0, len(test_idx), batch_size),
                          desc=f'Eval K={K} ksvd={k_svd} g={gamma:.3f}', leave=False):
            end = min(start + batch_size, len(test_idx))
            B   = end - start

            U_batch = torch.from_numpy(U_raw[test_arr[start:end]].astype(np.float32))
            U_flat  = U_batch.reshape(B, NT, N2).permute(0, 2, 1)

            U_hat = (U_flat.reshape(B * N2, NT).to(torch.complex64).to(device)
                     @ F.T).reshape(B, N2, K).permute(0, 2, 1)

            c_r  = torch.einsum('bkn,knj->bkj', U_hat.real, V_r_d)
            c_i  = torch.einsum('bkn,knj->bkj', U_hat.imag, V_i_d)
            U_ap = (torch.einsum('bkj,knj->bkn', c_r, V_r_d).to(torch.complex64)
                    + 1j * torch.einsum('bkj,knj->bkn', c_i, V_i_d).to(torch.complex64))

            diff = (U_ap - U_hat).norm(dim=2)
            ref  = U_hat.norm(dim=2).clamp(1e-12)
            freq_errors.append((diff / ref * 100).cpu().numpy())

            U_rec  = laplace.inverse_transform(U_ap, NT)
            U_true = U_batch.reshape(B, NT, N2).to(device)
            d = (U_rec - U_true).reshape(B, -1).norm(dim=1)
            r = U_true.reshape(B, -1).norm(dim=1).clamp(1e-12)
            errors_phys.extend((d / r * 100).cpu().numpy().tolist())

    return np.array(errors_phys), np.vstack(freq_errors)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_violin_sweep(ax, sweep_dict, xlabel, title):
    xs_vals   = sorted(sweep_dict.keys())
    datasets  = [list(sweep_dict[v]) for v in xs_vals]
    positions = list(range(len(xs_vals)))
    parts = ax.violinplot(datasets, positions=positions, widths=0.65,
                          showmedians=True, showmeans=True, showextrema=True)
    for pc in parts['bodies']: pc.set_alpha(0.55)
    parts['cmedians'].set_color('black');  parts['cmedians'].set_linewidth(2.2)
    parts['cmeans'].set_color('orange');   parts['cmeans'].set_linewidth(1.8)
    parts['cmeans'].set_linestyle('--')
    for key in ('cmaxes', 'cmins', 'cbars'): parts[key].set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f'{v:.3f}' if isinstance(v, float) else str(v) for v in xs_vals], fontsize=9)
    ax.set_xlim(-0.7, len(xs_vals) - 0.3)
    ax.legend(handles=[
        Line2D([0], [0], color='black',  linewidth=2.2,                 label='median'),
        Line2D([0], [0], color='orange', linewidth=1.8, linestyle='--', label='mean'),
    ], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel('L2rel (%)')
    ax.grid(alpha=0.3, axis='y')


def plot_freq_errors(ax, sweep_freq_dict, var_label, title):
    cmap   = plt.cm.viridis
    values = sorted(sweep_freq_dict.keys())
    norm_c = plt.Normalize(vmin=0, vmax=max(len(values) - 1, 1))
    for i, val in enumerate(values):
        mat     = sweep_freq_dict[val]
        ks      = np.arange(mat.shape[1])
        medians = np.median(mat, axis=0)
        p25     = np.percentile(mat, 25, axis=0)
        p75     = np.percentile(mat, 75, axis=0)
        color   = cmap(norm_c(i))
        label   = f'{var_label}={val:.3f}' if isinstance(val, float) else f'{var_label}={val}'
        ax.fill_between(ks, p25, p75, alpha=0.18, color=color)
        ax.plot(ks, medians, marker='o', markersize=4, linewidth=1.8, color=color, label=label)
    ax.set_xticks(np.arange(list(sweep_freq_dict.values())[0].shape[1]))
    ax.set_xlabel('Frequency index k')
    ax.set_ylabel('Median L2rel in Laplace domain (%)')
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.35)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    DATA_PATH  = 'dataset/ch4_rotated.npy'
    SPLIT_PATH = 'dataset/split.npz'

    # Base config
    K_BASE     = 16
    K_SVD_BASE = 64
    GAMMA_BASE = 1e-2

    # 1-D sweeps (other two axes fixed at base)
    GAMMA_VALUES = [0.0, 1e-3, 1e-2]
    K_VALUES     = [8, 16, 32]
    K_SVD_VALUES = [32, 64, 128]

    DT      = 1.0
    NT      = 150
    N       = 128
    ALPHA_T = 7e-3
    LAM     = 3e-5
    SEED    = 42

    CACHE_DIR = Path('checkpoints/svd_cache')
    PLOT_DIR  = Path('plots')
    SAVE_PATH = Path('checkpoints/svd_bases.pt')
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

    SVD_BATCH  = 128 if DEVICE == 'cuda' else 32
    EVAL_BATCH = 64  if DEVICE == 'cuda' else 16

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    print(f'Device: {device}')

    # All (K, k_svd, gamma) configs across the three sweeps (deduplicated)
    configs = set()
    for g in GAMMA_VALUES:
        configs.add((K_BASE, K_SVD_BASE, g))
    for K in K_VALUES:
        configs.add((K, K_SVD_BASE, GAMMA_BASE))
    for k_svd in K_SVD_VALUES:
        configs.add((K_BASE, k_svd, GAMMA_BASE))
    configs = sorted(configs)

    # k_svd_max per (K, gamma) — compute bases once at the largest k_svd needed
    kg_ksvd_max: dict[tuple, int] = {}
    for K, k_svd, g in configs:
        pair = (K, g)
        kg_ksvd_max[pair] = max(kg_ksvd_max.get(pair, 0), k_svd)

    print(f'Unique (K, gamma) base pairs: {len(kg_ksvd_max)}')
    print(f'Total eval configs:           {len(configs)}')

    # ---- Dataset & split ----
    dataset = TransientDataset(DATA_PATH, laplace=False, dt=DT)
    ns      = dataset.ns
    U_raw   = dataset._U_raw

    if Path(SPLIT_PATH).exists():
        split     = np.load(SPLIT_PATH)
        test_idx  = split['test_idx'].tolist()
        train_idx = [i for i in range(ns) if i not in set(test_idx)]
        print(f'Using split.npz — train: {len(train_idx)}  test: {len(test_idx)}')
    else:
        rng       = np.random.default_rng(SEED)
        perm      = rng.permutation(ns)
        n_train_  = int(0.85 * ns)
        train_idx = perm[:n_train_].tolist()
        test_idx  = perm[n_train_:].tolist()
        print(f'Random split — train: {len(train_idx)}  test: {len(test_idx)}')

    dataset.fit(train_idx)
    theta_norm = torch.as_tensor(
        (dataset.theta - dataset.theta_mean) / dataset.theta_std, dtype=torch.float32)

    # ---- Phase 1: bases + coefficients for each unique (K, gamma) ----
    print('\n=== Phase 1: SVD bases & coefficients ===')
    all_bases  = {}
    all_coeffs = {}

    for (K, g), k_svd_max in kg_ksvd_max.items():
        key        = f'K{K}_g{g:.3f}'
        base_cache = CACHE_DIR / f'bases_{key}_ksvdmax{k_svd_max}.pt'

        if base_cache.exists():
            print(f'  [cache]   bases {key}  (k_svd_max={k_svd_max})')
            bc     = torch.load(base_cache, map_location='cpu')
            V_r, V_i, s_list = bc['V_r'], bc['V_i'], bc['s_list']
        else:
            print(f'  [compute] bases {key}  (k_svd_max={k_svd_max})')
            s_list = _make_s_list(K, NT, DT, gamma=g)
            V_r, V_i = compute_svd_onepass(
                U_raw, s_list, train_idx, k_svd_max, NT, DT, SVD_BATCH, device)
            torch.save({'V_r': V_r, 'V_i': V_i, 's_list': s_list}, base_cache)

        coeff_cache = CACHE_DIR / f'coeffs_{key}_ksvdmax{k_svd_max}.pt'
        if coeff_cache.exists():
            print(f'  [cache]   coeffs {key}')
            cc           = torch.load(coeff_cache, map_location='cpu')
            coeffs_train = cc['train']
            coeffs_test  = cc['test']
            sorted_train = cc['sorted_train']
            sorted_test  = cc['sorted_test']
        else:
            print(f'  [compute] coeffs {key}')
            coeffs_train, sorted_train = compute_coefficients_onepass(
                U_raw, train_idx, s_list, V_r, V_i, NT, DT, SVD_BATCH, device)
            coeffs_test, sorted_test = compute_coefficients_onepass(
                U_raw, test_idx, s_list, V_r, V_i, NT, DT, SVD_BATCH, device)
            torch.save({
                'train': coeffs_train, 'test': coeffs_test,
                'sorted_train': sorted_train, 'sorted_test': sorted_test,
            }, coeff_cache)

        coeff_mean = coeffs_train.mean(dim=0)            # (K, 2, k_svd_max)
        coeff_std  = coeffs_train.std(dim=0).clamp(1e-8)

        all_bases[key] = {
            'V_r': V_r, 'V_i': V_i,
            's_im': np.array([s.imag for s in s_list]),
            'k_svd_max': k_svd_max,
        }
        all_coeffs[key] = {
            'train': coeffs_train, 'test': coeffs_test,
            'sorted_train': sorted_train, 'sorted_test': sorted_test,
            'coeff_mean': coeff_mean, 'coeff_std': coeff_std,
        }

    # ---- Phase 2: reconstruction eval for each (K, k_svd, gamma) ----
    print('\n=== Phase 2: reconstruction evaluation ===')
    all_eval = {}

    for K, k_svd, g in configs:
        key_base   = f'K{K}_g{g:.3f}'
        key        = f'K{K}_ksvd{k_svd}_g{g:.3f}'
        eval_cache = CACHE_DIR / f'eval_{key}.npz'

        if eval_cache.exists():
            print(f'  [cache]   eval {key}')
            d = np.load(eval_cache)
            all_eval[key] = {'errors_phys': d['errors_phys'],
                             'freq_errors': d['freq_errors']}
        else:
            print(f'  [compute] eval {key}')
            s_list = _make_s_list(K, NT, DT, gamma=g)
            V_r    = all_bases[key_base]['V_r'][:, :, :k_svd]
            V_i    = all_bases[key_base]['V_i'][:, :, :k_svd]
            ep, ef = eval_svd_reconstruction(
                U_raw, test_idx, V_r, V_i, s_list,
                DT, NT, N, ALPHA_T, LAM, g, device, EVAL_BATCH)
            np.savez(eval_cache, errors_phys=ep, freq_errors=ef)
            all_eval[key] = {'errors_phys': ep, 'freq_errors': ef}

        print(f'    median={np.median(all_eval[key]["errors_phys"]):.1f}%')

    # ---- Phase 3: plots ----
    print('\n=== Phase 3: plots ===')

    # Sweep gamma (K=K_BASE, K_SVD=K_SVD_BASE)
    sweep_g_phys = {g: all_eval[f'K{K_BASE}_ksvd{K_SVD_BASE}_g{g:.3f}']['errors_phys']
                    for g in GAMMA_VALUES}
    sweep_g_freq = {g: all_eval[f'K{K_BASE}_ksvd{K_SVD_BASE}_g{g:.3f}']['freq_errors']
                    for g in GAMMA_VALUES}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_violin_sweep(axes[0], sweep_g_phys, 'gamma',
                      f'SVD: L2rel vs gamma  (K={K_BASE}, K_SVD={K_SVD_BASE})')
    plot_freq_errors(axes[1], sweep_g_freq, 'gamma',
                     f'SVD: per-freq L2rel  (K={K_BASE}, vary gamma)')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'svd_sweep_gamma.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved svd_sweep_gamma.png')

    # Sweep K (gamma=GAMMA_BASE, K_SVD=K_SVD_BASE) — violin only (freq dims differ)
    sweep_K_phys = {K: all_eval[f'K{K}_ksvd{K_SVD_BASE}_g{GAMMA_BASE:.3f}']['errors_phys']
                    for K in K_VALUES}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_violin_sweep(ax, sweep_K_phys, 'K',
                      f'SVD: L2rel vs K  (gamma={GAMMA_BASE}, K_SVD={K_SVD_BASE})')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'svd_sweep_K.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved svd_sweep_K.png')

    # Sweep K_SVD (K=K_BASE, gamma=GAMMA_BASE)
    sweep_ksvd_phys = {k: all_eval[f'K{K_BASE}_ksvd{k}_g{GAMMA_BASE:.3f}']['errors_phys']
                       for k in K_SVD_VALUES}
    sweep_ksvd_freq = {k: all_eval[f'K{K_BASE}_ksvd{k}_g{GAMMA_BASE:.3f}']['freq_errors']
                       for k in K_SVD_VALUES}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_violin_sweep(axes[0], sweep_ksvd_phys, 'K_SVD',
                      f'SVD: L2rel vs K_SVD  (K={K_BASE}, gamma={GAMMA_BASE})')
    plot_freq_errors(axes[1], sweep_ksvd_freq, 'K_SVD',
                     f'SVD: per-freq L2rel  (K={K_BASE}, vary K_SVD)')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'svd_sweep_ksvd.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved svd_sweep_ksvd.png')

    # ---- Phase 4: single checkpoint ----
    print(f'\n=== Saving {SAVE_PATH} ===')
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'NT': NT, 'N': N, 'DT': DT, 'ALPHA_T': ALPHA_T, 'LAM': LAM,
        'configs': configs,
        'K_BASE': K_BASE, 'K_SVD_BASE': K_SVD_BASE, 'GAMMA_BASE': GAMMA_BASE,
        'train_idx': train_idx, 'test_idx': test_idx,
        'theta_norm': theta_norm,
        'theta_mean': dataset.theta_mean,
        'theta_std':  dataset.theta_std,
        'bases':  all_bases,
        'coeffs': all_coeffs,
        'eval':   all_eval,
    }, SAVE_PATH)
    print(f'Saved to {SAVE_PATH}')
