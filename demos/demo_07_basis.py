"""Visualise what the ExpMsFEM basis functions on an interior coarse cell
actually look like on a rough coefficient. Six panels, all on the same
`(N_f+1)×(N_f+1)` fine-scale grid of a central coarse cell:

  * two of the four Q1-harmonic nodal-basis functions (corner hats that adapt
    to the local coefficient rather than being piecewise-linear)
  * the top three edge eigen-modes on the cell's top edge (ranked by the
    generalised eigenvalue of `R^T N R v = λ P v`)
  * the per-cell bubble (Dirichlet-0 solve with constant RHS)

Writes figures/basis.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem import element_basis, local_ops
from expmsfem.coefficients import afun_periodic


def main():
    set_style()
    N_c, N_f, N_e = 4, 32, 3
    m, n = 1, 1           # an interior coarse cell (0-indexed)

    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=4)
    element_basis.prefactor_edges(ws, N_e, n_workers=4)
    value, _, _, _ = element_basis.element_basis_cached(ws, m, n, N_e)
    bubble = local_ops.bubble_cached(ws, m, n)

    # Column layout inside `value`:
    #   0..3                       nodal (corners 0,1,2,3)
    #   4..4+N_e                   bottom edge block (N_e modes + 1 bubble col)
    #   4+(N_e+1)..4+2(N_e+1)      top edge block
    #   ...
    top_block_start = 4 + (N_e + 1)

    # Real coordinates of the coarse cell (cell-local plot on [m/N_c, (m+1)/N_c]²)
    H = 1.0 / N_c
    xs = np.linspace(m * H, (m + 1) * H, N_f + 1)
    ys = np.linspace(n * H, (n + 1) * H, N_f + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    panels = [
        ("nodal basis — corner $(m, n)$",
         value[:, 0].reshape(N_f + 1, N_f + 1), "RdBu_r"),
        ("nodal basis — corner $(m+1, n+1)$",
         value[:, 2].reshape(N_f + 1, N_f + 1), "RdBu_r"),
        ("top-edge eigenmode $\\phi_1$",
         value[:, top_block_start].reshape(N_f + 1, N_f + 1), "RdBu_r"),
        ("top-edge eigenmode $\\phi_2$",
         value[:, top_block_start + 1].reshape(N_f + 1, N_f + 1), "RdBu_r"),
        ("top-edge eigenmode $\\phi_3$",
         value[:, top_block_start + 2].reshape(N_f + 1, N_f + 1), "RdBu_r"),
        ("cell bubble",
         bubble.reshape(N_f + 1, N_f + 1), "magma"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (title, Z, cmap) in zip(axes.flat, panels):
        if cmap == "RdBu_r":
            vmax = np.max(np.abs(Z)) + 1e-12
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto",
                               vmin=-vmax, vmax=vmax)
        else:
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle(
        "ExpMsFEM basis on an interior cell of the periodic coefficient "
        f"(cell $(m, n)={(m, n)}$; $N_f={N_f}$, $N_e={N_e}$)",
        y=1.02, fontsize=13,
    )
    save(fig, "basis.png")


if __name__ == "__main__":
    main()
