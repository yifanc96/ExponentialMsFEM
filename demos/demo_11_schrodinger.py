"""ExpMsFEM for the 2D Schrödinger eigenvalue problem.

Five panels: the potential `V(x)` (a deep periodic Bloch lattice of 8×8
wells) plus four of the lowest eigenfunctions of `H = -½Δ + V`. The
eigenfunctions localise on individual lattice sites — a classical
tight-binding picture that is sensitive to the fine structure of `V`,
and which the ExpMsFEM edge + cell-interior basis reproduces on a coarse
`N_c = 8` mesh.

Writes figures/schrodinger.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.schrodinger.driver import run_schrodinger_expmsfem
from expmsfem.schrodinger.fem import solve_fine_eigenpairs
from expmsfem.schrodinger.potentials import V_periodic_lattice


def main():
    set_style()
    depth = 400.0
    n_periods = 4
    V = lambda x, y: V_periodic_lattice(x, y, depth=depth, n_periods=n_periods)

    N_c, N_f, N_e, K_int = 8, 16, 3, 6
    print(f"[demo_11] Schrödinger on {n_periods}×{n_periods} lattice "
          f"(V_min = -{depth * 2}), N_c={N_c}, N_f={N_f}, "
          f"N_e={N_e}, K_int={K_int}")
    E_ref, psi_ref, H_ref, M_ref = solve_fine_eigenpairs(
        V, N_c * N_f, k=6, sigma=-2 * depth,
    )
    out = run_schrodinger_expmsfem(
        V, N_c, N_f, N_e, k=6, K_int=K_int,
        E_ref=E_ref, H_ref=H_ref, M_ref=M_ref,
        n_workers=4,
    )
    E_ms = out["E"]
    psi_ms = out["psi_ms_fine"]
    rel = np.abs(E_ms - E_ref) / np.abs(E_ref)
    print(f"  E_ref = {E_ref}")
    print(f"  E_ms  = {E_ms}")
    print(f"  rel err = {rel}")

    N = N_c * N_f + 1
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    V_grid = V(X, Y)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.3))

    # V(x)
    im0 = axes[0].pcolormesh(X, Y, V_grid, cmap="RdYlBu_r", shading="auto")
    axes[0].set_title(f"$V(x)$  (lattice depth $={depth}$)")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)

    # 4 eigenfunctions
    for k in range(4):
        psi = psi_ms[:, k].reshape(N, N)
        vmax = np.max(np.abs(psi))
        im = axes[k + 1].pcolormesh(X, Y, psi, cmap="RdBu_r", shading="auto",
                                    vmin=-vmax, vmax=vmax)
        axes[k + 1].set_title(
            f"$\\psi_{k}$  ($E_{k} = {E_ms[k]:.2f}$, "
            f"ref ${E_ref[k]:.2f}$, rel err ${rel[k]:.1e}$)"
        )
        plt.colorbar(im, ax=axes[k + 1], fraction=0.045, pad=0.03)

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    fig.suptitle(
        "Schrödinger ExpMsFEM: lowest 4 eigenstates on a 4×4 Bloch lattice "
        f"($N_c={N_c}$, $N_f={N_f}$, $N_e={N_e}$, $K_{{\\mathrm{{int}}}}"
        f"={K_int}$)",
        y=1.03, fontsize=13,
    )
    save(fig, "schrodinger.png")


if __name__ == "__main__":
    main()
