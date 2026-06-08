import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def _draw_rect_patch(ax, rect, use_contour, shape=None):
    if rect is None:
        return
    rx0, ry0, rx1, ry1 = rect
    if use_contour:
        xy, w, h = (rx0, ry0), rx1 - rx0, ry1 - ry0
    else:
        H, W = shape
        xy = (rx0 * W, ry0 * H)
        w, h = (rx1 - rx0) * W, (ry1 - ry0) * H
    ax.add_patch(Rectangle(xy, w, h, linewidth=1.5,
                            edgecolor='white', facecolor='gray', alpha=0.7))


def animate(
    frames: np.ndarray,
    output_path: str,
    fps: int = 10,
    cmap: str = "RdBu_r",
    label: str = "value",
    X: np.ndarray = None,
    Y: np.ndarray = None,
    title_fn=None,
    rect=None,
) -> None:
    """Save a 3D [time, x, y] numpy array as a GIF."""
    vmin, vmax = frames.min(), frames.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]

    use_contour = X is not None and Y is not None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    if use_contour:
        ax.contourf(X, Y, frames[0], levels=50, cmap=cmap_obj, norm=norm)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, label=label)
    else:
        im = ax.imshow(frames[0], cmap=cmap_obj, norm=norm, origin="lower")
        plt.colorbar(im, ax=ax, label=label)

    _draw_rect_patch(ax, rect, use_contour, shape=frames.shape[1:])
    title = ax.set_title(title_fn(0) if title_fn else "t = 0")

    def update(t):
        if use_contour:
            for c in ax.collections:
                c.remove()
            ax.contourf(X, Y, frames[t], levels=50, cmap=cmap_obj, norm=norm)
        else:
            im.set_data(frames[t])
        title.set_text(title_fn(t) if title_fn else f"t = {t}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")


def animate_comparaison(
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    output_path: str,
    fps: int = 10,
    cmap: str = "RdBu_r",
    label: str = "value",
    X: np.ndarray = None,
    Y: np.ndarray = None,
    title_a: str = "Original",
    title_b: str = "Reconstruit",
    title_err: str = "|Erreur|",
    title_fn=None,
    rect=None,
) -> None:
    """Save side-by-side comparison GIF: frames_a | frames_b | |frames_a - frames_b|."""
    err = np.abs(frames_a - frames_b)

    vmin = min(frames_a.min(), frames_b.min())
    vmax = max(frames_a.max(), frames_b.max())
    norm_ab  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norm_err = mcolors.Normalize(vmin=0, vmax=err.max())
    cmap_ab  = matplotlib.colormaps[cmap]
    cmap_err = matplotlib.colormaps["hot_r"]

    use_contour = X is not None and Y is not None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_a, ax_b, ax_e = axes
    for ax in axes:
        ax.set_aspect("equal")

    if use_contour:
        ax_a.contourf(X, Y, frames_a[0], levels=50, cmap=cmap_ab, norm=norm_ab)
        ax_b.contourf(X, Y, frames_b[0], levels=50, cmap=cmap_ab, norm=norm_ab)
        ax_e.contourf(X, Y, err[0],      levels=50, cmap=cmap_err, norm=norm_err)
    else:
        im_a = ax_a.imshow(frames_a[0], cmap=cmap_ab,  norm=norm_ab,  origin="lower")
        im_b = ax_b.imshow(frames_b[0], cmap=cmap_ab,  norm=norm_ab,  origin="lower")
        im_e = ax_e.imshow(err[0],      cmap=cmap_err, norm=norm_err, origin="lower")

    plt.colorbar(cm.ScalarMappable(norm=norm_ab,  cmap=cmap_ab),  ax=ax_a, label=label)
    plt.colorbar(cm.ScalarMappable(norm=norm_ab,  cmap=cmap_ab),  ax=ax_b, label=label)
    plt.colorbar(cm.ScalarMappable(norm=norm_err, cmap=cmap_err), ax=ax_e, label="|err|")

    shape = frames_a.shape[1:]
    for ax in axes:
        _draw_rect_patch(ax, rect, use_contour, shape=shape)

    ax_a.set_title(title_a)
    ax_b.set_title(title_b)
    ax_e.set_title(title_err)

    suptitle = fig.suptitle(title_fn(0) if title_fn else "t = 0")

    def update(t):
        if use_contour:
            for ax in axes:
                for c in ax.collections:
                    c.remove()
            ax_a.contourf(X, Y, frames_a[t], levels=50, cmap=cmap_ab,  norm=norm_ab)
            ax_b.contourf(X, Y, frames_b[t], levels=50, cmap=cmap_ab,  norm=norm_ab)
            ax_e.contourf(X, Y, err[t],      levels=50, cmap=cmap_err, norm=norm_err)
        else:
            im_a.set_data(frames_a[t])
            im_b.set_data(frames_b[t])
            im_e.set_data(err[t])
        suptitle.set_text(title_fn(t) if title_fn else f"t = {t}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_a), interval=1000 // fps, blit=False
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    import random
    import numpy as np
    from laplace_surrogate.inference.pipeline_ae import InferencePipelineAE

    # ─────────────────────────────────────────────────────────────────────────
    CKPT_PATH = 'checkpoints/slae_ld64_K16_g0.01.ckpt'
    OUT_DIR   = 'plots'
    N_SAMPLES = 3
    FPS       = 10
    SEED      = 0
    # ─────────────────────────────────────────────────────────────────────────

    random.seed(SEED)
    np.random.seed(SEED)

    pipe   = InferencePipelineAE.from_checkpoint(CKPT_PATH)
    chosen = random.sample(pipe.test_idx, N_SAMPLES)

    for rank, sim_i in enumerate(chosen):
        print(f"\n[{rank+1}/{N_SAMPLES}] simulation {sim_i}")
        u_raw = pipe.dataset._load_u(sim_i)        # (Nt, N, N)
        U_rec = pipe.reconstruct(u_raw)[0]         # (Nt, N, N)

        out = os.path.join(OUT_DIR, f'sim{sim_i:04d}_comparaison.gif')
        animate_comparaison(
            u_raw, U_rec,
            output_path=out,
            fps=FPS,
            label='CH4',
            title_a='Référence',
            title_b=f'Reconstruction {pipe.model_name.upper()}',
            title_fn=lambda t, s=sim_i: f'sim {s} — t = {t}',
        )
