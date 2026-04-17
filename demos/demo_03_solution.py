"""Side-by-side: fine-FEM reference, ExpMsFEM reconstruction, and the pointwise
error for the periodic-coefficient problem.

Writes figures/solution.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem


def main():
    set_style()
    N_c, N_f, N_e = 8, 32, 5
    print(f"[demo_03] running N_c={N_c} N_f={N_f} N_e={N_e}")
    out = run_expmsfem(afun_periodic, N_c, N_f, N_e, n_workers=4)
    N = N_c * N_f + 1
    u_ref = out["u_ref_fine"].reshape(N, N)
    u_ms = out["u_ms_fine"].reshape(N, N)
    err = u_ms - u_ref

    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, Z, title, cmap in [
        (axes[0], u_ref, "fine-FEM reference $u_{\\mathrm{ref}}$", "RdBu_r"),
        (axes[1], u_ms, "ExpMsFEM reconstruction $u_{\\mathrm{ms}}$", "RdBu_r"),
        (axes[2], err, "pointwise error $u_{\\mathrm{ms}} - u_{\\mathrm{ref}}$",
         "RdBu_r"),
    ]:
        vmax = np.max(np.abs(Z))
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto",
                           vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    rel = f"rel $H^1$ = {out['e_H1']:.2e}"
    fig.suptitle(
        "ExpMsFEM on periodic coefficient "
        f"($N_c={N_c}$, $N_f={N_f}$, $N_e={N_e}$) — {rel}",
        y=1.03, fontsize=13,
    )
    save(fig, "solution.png")


if __name__ == "__main__":
    main()
