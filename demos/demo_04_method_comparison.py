"""Head-to-head comparison of three multiscale methods on the periodic
coefficient:

  * Exp        — full ExpMsFEM (N_e edge eigenmodes + 1 edge-bubble per edge
                 + cell bubble; exponential convergence in N_e)
  * H+bubble   — classical MsFEM nodal basis + N_e edge eigenmodes
                 (no edge-bubble) + cell bubble
  * O(H)       — classical MsFEM nodal basis + N_e edge eigenmodes only

The plot highlights the characteristic ordering Exp < H+bubble < O(H) in
H¹ error, plus O(H)'s early stall without the cell bubble.

Writes figures/method_comparison.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem import local_ops
from expmsfem.baselines import run_hbubble, run_OH
from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def main():
    set_style()
    N_c, N_f, N_e_max = 8, 16, 5

    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=4)

    methods = {
        "Exp": run_expmsfem,
        "H+bubble": run_hbubble,
        "O(H)": run_OH,
    }
    h1 = {m: [] for m in methods}
    for N_e in range(1, N_e_max + 1):
        for name, fn in methods.items():
            out = fn(
                afun_periodic, N_c, N_f, N_e,
                u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
                workspace=ws, n_workers=4,
            )
            h1[name].append(out["e_H1"])
            print(f"  N_e={N_e}  {name:>9}: H1={out['e_H1']:.2e}")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    markers = {"Exp": "o", "H+bubble": "s", "O(H)": "^"}
    for name, vals in h1.items():
        ax.semilogy(range(1, N_e_max + 1), vals, marker=markers[name],
                    label=name)
    ax.set_xlabel("edge eigenmodes per edge, $N_e$")
    ax.set_ylabel("relative $H^1$ error")
    ax.set_xticks(range(1, N_e_max + 1))
    ax.legend(title="method")
    ax.set_title(
        "Method comparison on periodic coefficient "
        f"($N_c={N_c}$, $N_f={N_f}$)"
    )
    save(fig, "method_comparison.png")


if __name__ == "__main__":
    main()
