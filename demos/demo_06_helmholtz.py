"""Helmholtz ExpMsFEM demo: real and imaginary parts of the multiscale
solution, plus the pointwise error vs the fine-scale reference.

Writes figures/helmholtz_field.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.helmholtz.driver import run_expmsfem_helm
from expmsfem.helmholtz.fem import solve_fine


def main():
    set_style()
    N_c, N_f, N_e = 8, 8, 5
    k0 = 2.0
    print(f"[demo_06] running Helmholtz N_c={N_c} N_f={N_f} N_e={N_e} k0={k0}")
    u_ref, B_ref, C_ref, _ = solve_fine(N_c * N_f, k0)
    out = run_expmsfem_helm(
        N_c, N_f, N_e, k0,
        u_ref=u_ref, B_ref=B_ref, C_ref=C_ref, n_workers=4,
    )
    N = N_c * N_f + 1
    u_ms = out["u_ms_fine"].reshape(N, N)
    u_ref_2d = u_ref.reshape(N, N)
    err = u_ms - u_ref_2d

    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("Re $u_{\\mathrm{ms}}$", u_ms.real, "RdBu_r"),
        ("Im $u_{\\mathrm{ms}}$", u_ms.imag, "RdBu_r"),
        ("$|u_{\\mathrm{ms}} - u_{\\mathrm{ref}}|$", np.abs(err), "magma"),
    ]
    for ax, (title, Z, cmap) in zip(axes, panels):
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
    rel = f"rel $H^1$ = {out['e_H1']:.2e}"
    fig.suptitle(
        f"Helmholtz impedance ExpMsFEM ($k_0={k0}$, $N_c=N_f={N_c}$, "
        f"$N_e={N_e}$) — {rel}",
        y=1.03, fontsize=13,
    )
    save(fig, "helmholtz_field.png")


if __name__ == "__main__":
    main()
