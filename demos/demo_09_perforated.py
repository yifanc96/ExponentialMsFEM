"""Perforated-domain (fictitious-domain) demonstration.

We realise a perforated-domain-style problem on the rectangular background
mesh by using the high-contrast coefficient

    a(x) = 1       outside the holes
    a(x) = 1e-6   inside each hole

so the "holes" become near-insulating regions that the solution cannot
diffuse through. As `a_in / a_out → 0`, the configuration converges to the
true perforated-domain problem (Dirichlet-zero trace on each hole
boundary). ExpMsFEM handles this without any mesh modification — the edge
eigen-basis adapts automatically to every jump in `a`, so the coarse
system solves a well-conditioned low-DOF problem even when the geometry
gets complicated.

Shows three panels:
  1. coefficient `a(x)` (bulk vs holes).
  2. ExpMsFEM reconstruction `u_ms`.
  3. pointwise error `u_ms - u_ref` vs a fine-FEM reference.

Writes figures/perforated.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.coefficients import afun_perforated, default_hole_lattice
from expmsfem.driver import run_expmsfem


def main():
    set_style()
    # 16 holes on a 4×4 lattice, radius 0.07 (~14% of coarse-cell width at N_c=8).
    hole_centers = default_hole_lattice(n_per_side=4, margin=0.15)
    hole_radius = 0.07
    a_fun = afun_perforated(hole_centers, hole_radius,
                            a_out=1.0, a_in=1e-6)

    N_c, N_f, N_e = 8, 16, 4
    print(f"[demo_09] running ExpMsFEM on perforated "
          f"{len(hole_centers)}-hole domain "
          f"(N_c={N_c}, N_f={N_f}, N_e={N_e})")
    out = run_expmsfem(a_fun, N_c, N_f, N_e, n_workers=4)

    N = N_c * N_f + 1
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    a_grid = a_fun(X, Y)
    u_ms = out["u_ms_fine"].reshape(N, N)
    u_ref = out["u_ref_fine"].reshape(N, N)
    err = u_ms - u_ref

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # a(x)
    im0 = axes[0].pcolormesh(X, Y, np.log10(a_grid), cmap="viridis",
                             shading="auto")
    axes[0].set_title("$\\log_{10} a(x)$  (bulk $=1$, holes $=10^{-6}$)")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)
    # Overlay hole outlines
    theta = np.linspace(0, 2 * np.pi, 80)
    for cx, cy in hole_centers:
        axes[0].plot(cx + hole_radius * np.cos(theta),
                     cy + hole_radius * np.sin(theta),
                     color="white", linewidth=0.8, alpha=0.8)

    # u_ms
    vmax = np.max(np.abs(u_ms))
    im1 = axes[1].pcolormesh(X, Y, u_ms, cmap="RdBu_r", shading="auto",
                             vmin=-vmax, vmax=vmax)
    axes[1].set_title("ExpMsFEM solution $u_{\\mathrm{ms}}$")
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03)
    for cx, cy in hole_centers:
        axes[1].plot(cx + hole_radius * np.cos(theta),
                     cy + hole_radius * np.sin(theta),
                     color="black", linewidth=0.6, alpha=0.6)

    # pointwise error
    vmax = np.max(np.abs(err)) + 1e-20
    im2 = axes[2].pcolormesh(X, Y, err, cmap="RdBu_r", shading="auto",
                             vmin=-vmax, vmax=vmax)
    axes[2].set_title("pointwise error $u_{\\mathrm{ms}} - u_{\\mathrm{ref}}$")
    plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.03)

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    rel = f"rel $H^1$ = {out['e_H1']:.2e}"
    fig.suptitle(
        f"Perforated domain via fictitious-domain coefficient "
        f"(16 holes, contrast $10^6$, $N_c={N_c}$, $N_f={N_f}$, "
        f"$N_e={N_e}$) — {rel}",
        y=1.03, fontsize=13,
    )
    save(fig, "perforated.png")


if __name__ == "__main__":
    main()
