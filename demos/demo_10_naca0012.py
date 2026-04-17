"""CFD-adjacent demo: a NACA 0012 airfoil embedded in the [0, 1]² box
far-field, realised via the fictitious-domain trick

    a(x, y) = 1         outside the airfoil (fluid / bulk)
    a(x, y) = 1e-6     inside the airfoil (solid obstacle)

This is not Navier-Stokes — the PDE is the scalar elliptic
`-div(a grad u) = f` with homogeneous Dirichlet on the box boundary. The
point is that a geometrically non-trivial solid (an airfoil with a sharp
trailing edge and a sqrt-singular leading edge) drops into the ExpMsFEM
pipeline with zero mesh modifications. The edge eigen-basis adapts to
the coefficient jump at the airfoil surface automatically.

Three panels:

  1. `log10 a(x)` with the airfoil outline overlaid.
  2. ExpMsFEM solution `u_ms` — the solution is pushed around the
     obstacle.
  3. Pointwise error `u_ms - u_ref` vs fine-FEM reference.

Writes figures/naca0012.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.coefficients import afun_naca0012, _naca_thickness
from expmsfem.driver import run_expmsfem


def _airfoil_outline(chord_start, chord_end, y_center, thickness=0.12,
                     alpha_deg=0.0, n=200):
    """Return (xs, ys) of the closed NACA outline (upper + lower surface)
    in the domain frame, including angle-of-attack rotation."""
    chord = chord_end - chord_start
    xr = np.linspace(0, 1, n)
    yt = _naca_thickness(xr, thickness) * chord
    # body frame
    xb_upper = xr * chord
    yb_upper = yt
    xb_lower = xb_upper[::-1]
    yb_lower = -yt[::-1]
    xb = np.concatenate([xb_upper, xb_lower])
    yb = np.concatenate([yb_upper, yb_lower])
    # rotate by alpha about leading edge, then translate
    alpha = np.deg2rad(alpha_deg)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    x = chord_start + cos_a * xb - sin_a * yb
    y = y_center + sin_a * xb + cos_a * yb
    return x, y


def main():
    set_style()
    chord_start, chord_end, y_center = 0.2, 0.8, 0.5
    alpha_deg = 6.0       # 6° angle of attack
    a_out, a_in = 1.0, 1e-6
    a_fun = afun_naca0012(chord_start, chord_end, y_center,
                          thickness=0.12, alpha_deg=alpha_deg,
                          a_out=a_out, a_in=a_in)

    N_c, N_f, N_e = 8, 32, 4
    print(f"[demo_10] running ExpMsFEM on NACA 0012 at "
          f"alpha={alpha_deg}° (N_c={N_c}, N_f={N_f}, N_e={N_e})")
    out = run_expmsfem(a_fun, N_c, N_f, N_e, n_workers=4)

    N = N_c * N_f + 1
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    a_grid = a_fun(X, Y)
    u_ms = out["u_ms_fine"].reshape(N, N)
    u_ref = out["u_ref_fine"].reshape(N, N)
    err = u_ms - u_ref

    # Airfoil outline for overlays
    ox, oy = _airfoil_outline(chord_start, chord_end, y_center,
                              thickness=0.12, alpha_deg=alpha_deg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Coefficient
    im0 = axes[0].pcolormesh(X, Y, np.log10(a_grid), cmap="viridis",
                             shading="auto")
    axes[0].set_title(
        f"$\\log_{{10}} a(x)$  (bulk $=1$, airfoil $=10^{{-6}}$)"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)
    axes[0].plot(ox, oy, color="white", linewidth=1.0)

    # Solution — mask the airfoil interior so the external flow is readable
    # (with a_in=1e-6 the fictitious-domain interior solution scales as 1/a_in
    # and would saturate the colormap).
    outside = a_grid > 0.5 * (a_out + a_in)
    u_ms_masked = np.where(outside, u_ms, np.nan)
    u_ref_masked = np.where(outside, u_ref, np.nan)
    err_masked = np.where(outside, err, np.nan)

    vmax = np.nanmax(np.abs(u_ms_masked))
    im1 = axes[1].pcolormesh(X, Y, u_ms_masked, cmap="RdBu_r", shading="auto",
                             vmin=-vmax, vmax=vmax)
    axes[1].set_title("ExpMsFEM solution $u_{\\mathrm{ms}}$ (outside airfoil)")
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03)
    axes[1].fill(ox, oy, color="lightgrey", alpha=0.85, edgecolor="black",
                 linewidth=0.8)

    # Error
    vmax = np.nanmax(np.abs(err_masked)) + 1e-20
    im2 = axes[2].pcolormesh(X, Y, err_masked, cmap="RdBu_r", shading="auto",
                             vmin=-vmax, vmax=vmax)
    axes[2].set_title("pointwise error $u_{\\mathrm{ms}} - u_{\\mathrm{ref}}$ (outside)")
    plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.03)
    axes[2].fill(ox, oy, color="lightgrey", alpha=0.85, edgecolor="black",
                 linewidth=0.6)

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    rel = f"rel $H^1$ = {out['e_H1']:.2e}"
    fig.suptitle(
        f"NACA 0012 in box far-field (chord $\\in [{chord_start}, {chord_end}]$, "
        f"$\\alpha={alpha_deg}^\\circ$, $N_c={N_c}$, $N_f={N_f}$, "
        f"$N_e={N_e}$) — {rel}",
        y=1.03, fontsize=13,
    )
    save(fig, "naca0012.png")


if __name__ == "__main__":
    main()
