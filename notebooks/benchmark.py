"""benchmark — inference-cost measurement, identical across the evaluation notebooks.

Each notebook used to measure cost its own way: some with ``time.perf_counter()``
and a single metric, ``eval_dlrom_cost`` with CUDA Events and four. The numbers were
therefore comparable neither across sections of the article nor with the table
caption, which announced CUDA Events where the notebook timed on the host.

This module holds the reference version and every notebook calls it: one
implementation, hence one methodology.

Usage ::

    import benchmark
    bench[name] = benchmark.measure(lambda: pipe.model.generate(theta), device)
    ...
    benchmark.figures(bench, label_of=_pretty_name, color_of=_color, device=device)

``figures`` writes one figure **per metric** (latency, FLOPs, memory, energy) via
``figstyle.save`` rather than a multi-panel figure, so each can be placed — or
dropped — independently in the article.

Unavailable metrics (no GPU, no ``pynvml``, a FLOP counter tripping on an operator)
are ``nan`` and their figure is omitted without failing the rest.
"""

from __future__ import annotations

import time

import numpy as np
import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
except Exception:  # torch trop ancien
    FlopCounterMode = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_H = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as _e:  # noqa: BLE001
    pynvml = None
    _NVML_H = None
    _NVML_ERR = str(_e)

__all__ = ["measure", "figures", "table", "METRICS"]

N_WARMUP = 10
N_RUNS = 50
E_MIN_TIME = 1.5      # s   — durée minimale d'intégration de la puissance
E_MIN_ITERS = 60      # itérations minimales, pour lisser le bruit du compteur

# clé → (titre d'axe, facteur d'échelle, format, suffixe du nom de fichier)
METRICS = {
    "lat_med":  ("Latency (ms / sample, bs=1)", 1.0,   "{:.1f}", "latency"),
    "flops":    ("FLOPs per sample (G)",        1e-9,  "{:.1f}", "flops"),
    "mem_mb":   ("Peak GPU memory (MB)",        1.0,   "{:.0f}", "memory"),
    "energy_j": ("Energy per sample (mJ)",      1e3,   "{:.1f}", "energy"),
}


# ── Primitives ────────────────────────────────────────────────────────────────

def _time_gpu(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Latency from CUDA Events: measured device-side, insensitive to Python overhead."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(n_runs):
        ev_s.record(); fn(); ev_e.record(); torch.cuda.synchronize()
        ms.append(ev_s.elapsed_time(ev_e))
    return np.array(ms)


def _time_cpu(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    for _ in range(n_warmup):
        fn()
    ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter(); fn(); ms.append((time.perf_counter() - t0) * 1e3)
    return np.array(ms)


def _peak_mem_mb(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _flops(fn):
    if FlopCounterMode is None:
        raise RuntimeError("FlopCounterMode indisponible")
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return float(fc.get_total_flops())


def _energy(fn, handle, min_time=E_MIN_TIME, min_iters=E_MIN_ITERS):
    """Energy per sample: mean power integrated over a burst.

    A single reading is meaningless — the NVML counter is sampled slowly relative to
    an inference of a few milliseconds — so we loop until both a minimum duration and
    a minimum iteration count are covered.
    """
    torch.cuda.synchronize()
    powers = []
    t0 = time.perf_counter()
    it = 0
    while True:
        fn(); it += 1
        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle))
        if (time.perf_counter() - t0) >= min_time and it >= min_iters:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p = float(np.mean(powers)) / 1000.0          # mW → W
    return p * dt / it, p


# ── API ───────────────────────────────────────────────────────────────────────

def measure(fn, device="cuda", name="", verbose=True) -> dict:
    """Measure the cost of one inference call at batch size 1.

    Args:
        fn:     zero-argument callable performing a complete inference.
        device: ``'cuda'`` or ``'cpu'``; off CUDA only latency is measured.
        name:   label for the log line.

    Returns:
        dict with ``lat_med``, ``lat_std``, ``flops``, ``mem_mb``, ``energy_j`` and
        ``power_w``. Unavailable metrics are ``nan``.
    """
    cuda = (device == "cuda")
    b = {}

    lat = (_time_gpu if cuda else _time_cpu)(fn)
    b["lat_med"] = float(np.median(lat))
    b["lat_std"] = float(np.std(lat))

    try:
        b["flops"] = _flops(fn)
    except Exception as e:  # noqa: BLE001
        b["flops"] = np.nan
        if verbose:
            print(f'  {name} FLOPs KO: {e}')

    if cuda:
        try:
            b["mem_mb"] = _peak_mem_mb(fn)
        except Exception as e:  # noqa: BLE001
            b["mem_mb"] = np.nan
            if verbose:
                print(f'  {name} memory failed: {e}')
        if pynvml is not None:
            try:
                b["energy_j"], b["power_w"] = _energy(fn, _NVML_H)
            except Exception as e:  # noqa: BLE001
                b["energy_j"] = b["power_w"] = np.nan
                if verbose:
                    print(f'  {name} energy failed: {e}')
        else:
            b["energy_j"] = b["power_w"] = np.nan
    else:
        b["mem_mb"] = b["energy_j"] = b["power_w"] = np.nan

    if verbose:
        print(f'{name:<45} lat={b["lat_med"]:7.1f}±{b["lat_std"]:5.1f} ms  '
              f'flops={b["flops"]/1e9:6.1f} G  mem={b["mem_mb"]:6.0f} MB  '
              f'E={b["energy_j"]*1e3:6.1f} mJ')
    return b


def figures(bench: dict, label_of=None, color_of=None, device="cuda",
            metrics=None, prefix="bench"):
    """One figure per metric, as sorted horizontal bars.

    Args:
        bench:    ``{name: dict from measure()}``.
        label_of: ``name -> readable label`` (default: the raw name).
        color_of: ``name -> colour`` (default: grey).
        metrics:  subset of ``METRICS``; default: all.
        prefix:   filename prefix, completed by the metric suffix.
    """
    import matplotlib.pyplot as plt
    import figstyle

    if not bench:
        print("benchmark: nothing to plot.")
        return []

    label_of = label_of or (lambda n: n)
    color_of = color_of or (lambda n: "0.5")
    written = []

    for key in (metrics or METRICS):
        axis_label, scale, fmt, suffix = METRICS[key]
        names = [n for n in bench if np.isfinite(bench[n].get(key, np.nan))]
        if not names:
            print(f"benchmark: '{key}' not measured — figure omitted.")
            continue
        names = sorted(names, key=lambda n: bench[n][key])

        vals = [bench[n][key] * scale for n in names]
        errs = ([bench[n].get("lat_std", 0.0) * scale for n in names]
                if key == "lat_med" else None)

        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.6))
        y = np.arange(len(names))
        ax.barh(y, vals, color=[color_of(n) for n in names], alpha=0.9, zorder=2)
        if errs is not None:
            ax.errorbar(vals, y, xerr=errs, fmt="none", color="black",
                        capsize=3, lw=1.2, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(n) for n in names], fontsize=9)
        for yi, v in enumerate(vals):
            ax.annotate(fmt.format(v), (v, yi), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8, color="0.3")
        ax.set_xlabel(axis_label)
        ax.grid(axis="x", alpha=0.4, zorder=0)
        ax.margins(x=0.12)
        fig.tight_layout()
        plt.show()
        figstyle.save(f"{prefix}_{suffix}")
        written.append(f"{prefix}_{suffix}")

    return written


def scatter(bench: dict, accuracy: dict, label_of=None, color_of=None,
            device="cuda", metrics=None, prefix="cost", ylabel="Median $L^2$ error (%)"):
    """Accuracy against cost: one figure per metric.

    Args:
        bench:    ``{name: dict from measure()}``.
        accuracy: ``{name: error}`` — already aggregated (median), not the distribution.

    This is the view carrying the article's argument: at comparable error, which
    family costs least. Keeping the metrics separate avoids having to choose here
    between latency, FLOPs and energy, which do not rank identically — latency
    depends on effective parallelism where FLOPs do not.
    """
    import matplotlib.pyplot as plt
    import figstyle

    if not bench or not accuracy:
        print("benchmark: no accuracy/cost scatter — missing data.")
        return []

    label_of = label_of or (lambda n: n)
    color_of = color_of or (lambda n: "0.5")
    written = []

    for key in (metrics or METRICS):
        axis_label, scale, fmt, suffix = METRICS[key]
        names = [n for n in bench
                 if n in accuracy and np.isfinite(bench[n].get(key, np.nan))]
        if not names:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for n in names:
            x, y = bench[n][key] * scale, accuracy[n]
            ax.scatter(x, y, color=color_of(n), s=80, zorder=3)
            ax.annotate(label_of(n), (x, y), fontsize=7,
                        xytext=(5, 3), textcoords="offset points")
        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
        figstyle.save(f"{prefix}_{suffix}")
        written.append(f"{prefix}_{suffix}")

    return written


def table(bench: dict, label_of=None):
    """Text summary, sorted by increasing latency."""
    if not bench:
        print("benchmark: no measurements.")
        return
    label_of = label_of or (lambda n: n)
    print(f"{'checkpoint':<45}{'lat (ms)':>12}{'±':>8}{'FLOPs (G)':>12}"
          f"{'mem (MB)':>11}{'E (mJ)':>10}")
    print("-" * 98)
    for n in sorted(bench, key=lambda n: bench[n]["lat_med"]):
        b = bench[n]
        print(f"{label_of(n):<45}{b['lat_med']:>12.1f}{b['lat_std']:>8.1f}"
              f"{b['flops']/1e9:>12.1f}{b['mem_mb']:>11.0f}{b['energy_j']*1e3:>10.1f}")
