"""Shared matplotlib setup for the demo gallery."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def set_style():
    """Paper-ready matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": True,
    })


def save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES_DIR.parent)}")
