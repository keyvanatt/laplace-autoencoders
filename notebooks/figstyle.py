"""figstyle — single style and save path for the article figures.

The figures came from five notebooks written over six weeks: fonts, sizes and axis
labels had drifted, files were 100 dpi PNGs named after the cell index
(``optae_cell12_00.png``), and every figure carried a title duplicating the LaTeX
caption. This module settles the three.

Usage, in a single cell after the imports ::

    import figstyle
    figstyle.setup(run="optae")        # run=None for the default set

then, at the end of every cell producing a figure ::

    figstyle.save("contour_poles")     # -> article/images/optae_contour_poles.pdf

``save`` takes the current figure, so the variable need not be named and the file
name no longer depends on the cell position.

The ``run`` prefix exists because the same notebook serves twice — once on the
prescribed-contour checkpoints, once on the optimised ones — and must write two
distinct sets of files.

In article mode (``titles=False``, the default) ``suptitle`` calls and figure-level
titles are suppressed: the text stays in the notebook source but does not reach the
PDF, where the LaTeX caption already carries it. Sub-panel titles
(``ax.set_title``) are kept, as they carry the hyperparameters.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = ["setup", "save", "PALETTE", "MODEL_COLORS", "MARKERS", "LABELS", "outdir", "run_tag"]

# ── Locations ──────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
STYLE_FILE = _HERE / "paper.mplstyle"
DEFAULT_OUTDIR = _REPO / "article" / "images"

# ── Shared visual identity ────────────────────────────────────────────────
# One family = one colour + one marker, identical across all figures. The markers
# carry the information on their own, so the figure stays readable in greyscale and
# for a deuteranopic reader, for whom the red/green pair of the earlier cost axes
# was the worst possible choice.

PALETTE = {
    "llae":   "#C0392B",  # brick red
    "slae":   "#2471A3",  # blue
    "dlrom":  "#7D3C98",  # purple (was green: red/green confusion)
    "pod":    "#566573",  # slate grey
    "tucker": "#B9770E",  # amber
    "svd":    "#117A65",  # teal
}

# Colours by model_type, as the notebooks index them. The SVD and Tucker variants
# are lighter and darker shades of their family colour, so a figure mixing all
# seven stays readable.
MODEL_COLORS = {
    "SLAEModel":       PALETTE["slae"],
    "SLAESVDModel":    "#6FA8DC",
    "SLAETuckerModel": "#16405F",
    "LLAEModel":       PALETTE["llae"],
    "LLAESVDModel":    "#E8836F",
    "LLAETuckerModel": "#7B1E12",
    "DLROMModel":      PALETTE["dlrom"],
    "CorrectionAE":    PALETTE["svd"],
}

MARKERS = {
    "llae": "*", "slae": "D", "dlrom": "o", "pod": "s", "tucker": "^", "svd": "v",
}

# Vocabulary: must match the article word for word, otherwise the reader has to
# map figure labels to paper terms by hand.
LABELS = {
    "prescribed": "prescribed contour",
    "learnable":  "learnable contour",
    "offline":    "offline contour",
    "l2rel":      r"Relative $L^2$ error (\%)" if mpl.rcParams["text.usetex"] else "Relative $L^2$ error (%)",
}

_state = {"run": None, "outdir": DEFAULT_OUTDIR, "titles": False, "saved": [], "by_id": {},
          "shown": []}
_orig_suptitle = Figure.suptitle
_orig_show = plt.show


def _show_capturing(*a, **k):
    """``plt.show`` that keeps a reference to the figures being displayed.

    Cells are written ``plt.show()`` then ``figstyle.save(...)``. Under the inline
    backend (Jupyter, nbclient) ``show`` *closes* the figures after rendering them,
    so the ``gcf()`` in ``save`` returns a fresh empty figure and the "no figure
    produced" guard silently drops every save. Interactively the error is invisible
    — the plots appear — but nothing reaches ``article/images``.

    We therefore stack the live figures just before display. A closed figure remains
    perfectly saveable: its canvas is intact, only interactive management is detached.
    """
    _state["shown"] = [plt.figure(n) for n in plt.get_fignums()]
    return _orig_show(*a, **k)


# ── API ───────────────────────────────────────────────────────────────────────

def setup(run: str | None = None, titles: bool = False, outdir: str | os.PathLike | None = None) -> None:
    """Apply the article style and arm the save path.

    Args:
        run:     file prefix for this set of figures (``"optae"``, ``"optsurr"``…).
                 ``None`` means no prefix.
        titles:  ``True`` keeps ``suptitle`` (convenient interactively), ``False``
                 for article output.
        outdir:  destination; defaults to ``article/images`` in the repository.
    """
    if not STYLE_FILE.exists():
        raise FileNotFoundError(f"{STYLE_FILE} not found — figstyle.py must sit next to it.")
    plt.style.use(str(STYLE_FILE))

    _state["run"] = run
    _state["titles"] = titles
    _state["outdir"] = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    _state["outdir"].mkdir(parents=True, exist_ok=True)
    _state["saved"] = []
    _state["by_id"] = {}
    _state["shown"] = []

    plt.show = _show_capturing

    # Suppress suptitle without removing the call from the notebook source.
    if titles:
        Figure.suptitle = _orig_suptitle
    else:
        def _muted(self, *a, **k):  # noqa: ANN001, ANN202
            return None
        Figure.suptitle = _muted

    print(f"figstyle: style={STYLE_FILE.name}  run={run or '(none)'}  "
          f"titles={'kept' if titles else 'suppressed'}  ->  {_state['outdir']}")


def save(name: str, fig=None, also_png: bool = False, **kw):
    """Save the current figure under a semantic name, as a vector PDF.

    Args:
        name:     name without extension or prefix (``"contour_poles"``).
        fig:      figure; defaults to the current one.
        also_png: additionally write a 300 dpi PNG (preview only, never for the article).

    Two guards, because the call sits at the end of a cell and several cells plot
    only conditionally:

    * a figure without axes is not written (the cell produced nothing);
    * if the same figure is saved twice under different names, the cell did not plot
      and ``gcf()`` returned the previous one — we still write, but say so loudly,
      as the error would otherwise survive into the article PDF.
    """
    if fig is None:
        fig = plt.gcf()
        # Empty gcf(): the preceding plt.show() closed the figure (inline backend).
        # Fall back on those captured by _show_capturing, in display order, one per
        # call — a cell showing two figures saves them in two calls.
        if not fig.get_axes() and _state["shown"]:
            plt.close(fig)
            fig = _state["shown"].pop(0)

    if not fig.get_axes():
        print(f"  figstyle: '{name}' skipped — the cell produced no figure.")
        return None

    stem = f"{_state['run']}_{name}" if _state["run"] else name
    prev = _state["by_id"].get(id(fig))
    if prev is not None and prev != stem:
        print(f"  !! figstyle: '{stem}' reuses the figure already saved as '{prev}'. "
              f"The cell most likely plotted nothing (untaken branch?) — check it.")

    out = _state["outdir"] / f"{stem}.pdf"
    fig.savefig(out, **kw)
    if also_png:
        fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    _state["by_id"][id(fig)] = stem
    _state["saved"].append(stem)
    print(f"  ↳ {_short(out)}")
    return out


def _short(path: Path) -> str:
    """Path relative to the repository when possible, absolute otherwise.
    """
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)

def outdir() -> Path:
    """Current figure destination."""
    return _state["outdir"]


def run_tag() -> str | None:
    """Current prefix, or ``None``."""
    return _state["run"]


def summary() -> None:
    """List what has been written since the last ``setup`` — call at the end of a
    notebook to check that no expected figure is missing.
    """
    if not _state["saved"]:
        print("figstyle: no figure saved.")
        return
    print(f"figstyle: {len(_state['saved'])} figure(s) in {_state['outdir']}")
    for s in _state["saved"]:
        print(f"  {s}.pdf")
