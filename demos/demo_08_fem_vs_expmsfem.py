"""Why multiscale beats plain FEM on rough coefficients.

Setup: the 5-scale periodic coefficient `afun_periodic` has finest
oscillation period `1/65`. A plain Q1 FEM on a coarse `N_c × N_c` grid
cannot see these oscillations when `1/N_c ≫ 1/65`, and its H¹ error
plateaus at an O(1) "resonance" level until the coarse mesh is refined
past the finest microscale. ExpMsFEM on the same coarse grid builds a
problem-adapted basis that encodes the fine-scale information and drops
the error by several orders of magnitude at every `N_c`.

We fix `N_c · N_f = 128` (a 128×128 fine reference grid) and sweep
`N_c ∈ {2, 4, 8, 16, 32}`. For each `N_c`:

  * plain FEM solves `solve_fine(a, N_c)` at resolution `N_c`; we
    bilinearly upsample to 128×128 to compute the H¹ error against the
    fine reference.
  * ExpMsFEM solves `run_expmsfem(a, N_c, N_f=128/N_c, N_e=4)` whose
    reconstruction is already on the 128×128 grid.

Writes figures/fem_vs_expmsfem.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from _util import set_style, save
from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def upsample(u_coarse, N_coarse, N_fine):
    """Bilinear interpolation of a (N_coarse+1)² Q1 solution onto a
    (N_fine+1)² grid."""
    xs = np.linspace(0, 1, N_coarse + 1)
    U = u_coarse.reshape(N_coarse + 1, N_coarse + 1)
    interp = RegularGridInterpolator((xs, xs), U, bounds_error=False,
                                     fill_value=0.0)
    X, Y = np.meshgrid(np.linspace(0, 1, N_fine + 1),
                       np.linspace(0, 1, N_fine + 1), indexing="xy")
    # RegularGridInterpolator expects (j, i) for row-major — careful about axes.
    pts = np.stack([Y.ravel(), X.ravel()], axis=-1)
    # Wait: U is reshaped (N+1, N+1) where the outer axis is y (row) and
    # inner is x (col) given the j-major ordering used by local_to_global_nodes.
    # Easier: map to (x, y) coordinates consistently.
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    vals = interp(pts).reshape(N_fine + 1, N_fine + 1)
    return vals.ravel()


def main():
    set_style()
    N_ref = 128
    N_e = 3

    # Plain FEM: sweep further up to show where it eventually starts to converge.
    plain_Ns = [4, 8, 16, 32, 64, 128]
    # ExpMsFEM: keep N_c · N_f = N_ref so the reconstruction is already on
    # the reference grid, and stop at N_c = 16 so N_f stays ≥ 8 and N_e = 3
    # is well inside the per-edge eigenspace (rank N_f-1).
    ms_Nc_list = [2, 4, 8, 16]

    print("[demo_08] computing reference solution on 128×128 grid")
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_ref)

    err_plain = []
    dofs_plain = []
    for N in plain_Ns:
        u_plain, _, _, _ = solve_fine(afun_periodic, N)
        u_plain_fine = upsample(u_plain, N, N_ref)
        e = u_plain_fine - u_ref
        h1 = float(np.sqrt(e @ (K_ref @ e) / (u_ref @ (K_ref @ u_ref))))
        err_plain.append(h1)
        dofs_plain.append((N + 1) ** 2)
        print(f"  plain FEM N={N:3d}:  H1 = {h1:.2e}")

    err_ms = []
    dofs_ms = []
    for N_c in ms_Nc_list:
        N_f = N_ref // N_c
        out = run_expmsfem(
            afun_periodic, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=4,
        )
        err_ms.append(out["e_H1"])
        dofs_ms.append((N_c + 1) ** 2 + 2 * N_c * (N_c + 1) * (N_e + 1))
        print(f"  ExpMsFEM  N_c={N_c:3d}, N_f={N_f:3d}, N_e={N_e}:  "
              f"H1 = {out['e_H1']:.2e}")

    fig, (ax_nc, ax_dof) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_nc.loglog(plain_Ns, err_plain, marker="s", linewidth=2,
                 label="plain Q1 FEM on $N \\times N$ mesh")
    ax_nc.loglog(ms_Nc_list, err_ms, marker="o", linewidth=2,
                 label=f"ExpMsFEM ($N_f={N_ref}/N_c$, $N_e={N_e}$)")
    ax_nc.axvline(65, color="grey", linestyle=":", alpha=0.6,
                  label="finest microscale $1/\\epsilon_5 = 65$")
    ax_nc.set_xlabel("coarse mesh size $N$ or $N_c$")
    ax_nc.set_ylabel("relative $H^1$ error")
    ax_nc.legend(loc="lower left")
    ax_nc.set_title("Error vs mesh size")

    ax_dof.loglog(dofs_plain, err_plain, marker="s", linewidth=2,
                  label="plain Q1 FEM")
    ax_dof.loglog(dofs_ms, err_ms, marker="o", linewidth=2,
                  label=f"ExpMsFEM coarse system ($N_e={N_e}$)")
    ax_dof.set_xlabel("degrees of freedom")
    ax_dof.set_ylabel("relative $H^1$ error")
    ax_dof.legend(loc="lower left")
    ax_dof.set_title("Error vs solver DOFs")

    fig.suptitle(
        "Plain FEM resonance error vs problem-adapted ExpMsFEM basis "
        "(5-scale periodic $a(x)$, finest period $1/65$)",
        y=1.03, fontsize=13,
    )
    save(fig, "fem_vs_expmsfem.png")


if __name__ == "__main__":
    main()
