"""Exponential-convergence plot for ExpMsFEM across three coefficient fields.

Sweeps N_e = 1..5 on a moderate grid (N_c=8, N_f=16) and plots relative
H¹ error on a log-linear axis — the signature of exponential decay is a
straight line.

Writes figures/convergence.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem import local_ops
from expmsfem.coefficients import (
    afun_periodic, afun_random, afun_highcontrast, generate_random_field,
)
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def sweep(a_fun, N_c, N_f, N_e_max, label, workers=4):
    u_ref, K_ref, M_ref, _ = solve_fine(a_fun, N_c * N_f)
    ws = local_ops.Workspace(a_fun, N_c, N_f)
    ws.prefactor_all(n_workers=workers)
    h1 = []
    l2 = []
    for N_e in range(1, N_e_max + 1):
        out = run_expmsfem(
            a_fun, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            workspace=ws, n_workers=workers,
        )
        h1.append(out["e_H1"])
        l2.append(out["e_L2"])
        print(f"  {label:>13}  N_e={N_e}  H1={out['e_H1']:.2e}")
    return np.array(h1), np.array(l2)


def main():
    set_style()
    N_c, N_f, N_e_max = 8, 16, 5

    goo = generate_random_field(M=128, seed=0)
    problems = [
        ("Periodic 5-scale", afun_periodic),
        ("Random rough", afun_random(goo, M=128)),
        ("High-contrast 64×", afun_highcontrast),
    ]

    h1 = {}
    l2 = {}
    for label, a_fun in problems:
        print(f"[demo_02] solving: {label}")
        h1[label], l2[label] = sweep(a_fun, N_c, N_f, N_e_max, label)

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(12, 4.5))
    N_e_axis = np.arange(1, N_e_max + 1)
    markers = ["o", "s", "^"]
    for (label, _), m in zip(problems, markers):
        ax_h.semilogy(N_e_axis, h1[label], marker=m, label=label)
        ax_l.semilogy(N_e_axis, l2[label], marker=m, label=label)
    for ax, title in [
        (ax_h, "relative $H^1$ error"),
        (ax_l, "relative $L^2$ error"),
    ]:
        ax.set_xlabel("edge eigenmodes per edge, $N_e$")
        ax.set_ylabel(title)
        ax.set_xticks(N_e_axis)
        ax.legend(loc="upper right")
    fig.suptitle(
        f"ExpMsFEM exponential convergence ($N_c={N_c}$, $N_f={N_f}$)",
        y=1.02, fontsize=13,
    )
    save(fig, "convergence.png")


if __name__ == "__main__":
    main()
