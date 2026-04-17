"""Time-dependent Schrödinger with *periodic* BC.

Same physics as `demo_11_wavepacket` but on a torus: when the wavepacket
crosses the right edge of `[0, 1]²` it reappears on the left. The
multiscale potential is period-1 in both axes:

    V(x, y) = V₀ · ( sin²(π · 16 · x) + sin²(π · 16 · y) )

so V has period `1/16`, below the coarse mesh `H = 1/16` — a genuine
sub-H oscillation that still tiles the torus exactly.

The packet is launched near the right boundary (`x₀ = 0.8`) with momentum
`+x` so it wraps within a few tens of steps. Top row: fine-FEM reference
`|ψ(t, x)|²`. Bottom row: ExpMsFEM reconstruction with periodic BC.

Writes figures/wavepacket_periodic.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.schrodinger.time_dep import (
    SemiclassicalParam, solve_fine_backward_euler, run_expmsfem_schrodinger,
)


def main():
    set_style()
    eps = 0.3
    dt = 1e-3

    V0 = 2.0
    k_V = 16.0  # period 1/16 — fits the torus exactly and is sub-H = 1/16
    V = lambda x, y: V0 * (np.sin(np.pi * k_V * x) ** 2
                           + np.sin(np.pi * k_V * y) ** 2)

    N_c, N_f, N_e = 16, 8, 3
    N_fine = N_c * N_f
    xs = np.linspace(0, 1, N_fine + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    sigma = 0.07
    # Start near the right edge so the packet wraps around quickly.
    x0, y0, kx = 0.8, 0.5, 4.0
    psi0 = (np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
            * np.exp(1j * kx * X / eps)).ravel().astype(np.complex128)

    param = SemiclassicalParam(eps=eps, V_fun=V, dt=dt, boundary="periodic")
    n_steps = 150
    save_stride = 30

    print(f"[demo_12] periodic wavepacket through multiscale V "
          f"(ε={eps}, Δt={dt}, N_c={N_c}, N_f={N_f}, N_e={N_e})")
    print("[demo_12] fine reference (periodic) ...")
    ts_ref, frames_ref, B, M = solve_fine_backward_euler(
        param, psi0, N_fine, n_steps, save_stride=save_stride,
    )
    print("[demo_12] ExpMsFEM (periodic) ...")
    ts_ms, frames_ms, prop = run_expmsfem_schrodinger(
        param, psi0, N_c, N_f, N_e, n_steps, save_stride=save_stride,
        n_workers=4,
    )

    # Relative L² error on the full (N_fine+1)² grid
    h = 1.0 / N_fine
    for i, t in enumerate(ts_ms):
        err = frames_ref[:, i] - frames_ms[:, i]
        rel = float(np.sqrt(
            np.real(np.sum(err.conj() * err)) * h ** 2
            / max(np.real(np.sum(frames_ref[:, i].conj() * frames_ref[:, i])) * h ** 2,
                  1e-30)
        ))
        print(f"  t = {t:.3f}  rel L2 = {rel:.2e}")

    # Six-column figure: V(x) + 5 snapshots
    snap_indices = [0, 1, 2, 3, 4, 5]
    V_grid = V(X, Y)
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))

    for row in range(2):
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, V_grid, cmap="viridis", shading="auto")
        ax.set_title(f"V(x)  (V₀={V0}, period 1/{int(k_V)})")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    for col_offset, idx in enumerate(snap_indices[1:]):
        col = col_offset + 1
        t = ts_ms[idx]
        ref_sq = np.abs(frames_ref[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        ms_sq = np.abs(frames_ms[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        vmax = max(ref_sq.max(), ms_sq.max())

        for row, Z, label in [(0, ref_sq, "fine FEM"),
                              (1, ms_sq, "ExpMsFEM")]:
            ax = axes[row, col]
            im = ax.pcolormesh(X, Y, Z, cmap="magma", shading="auto",
                               vmin=0, vmax=vmax)
            ax.set_title(f"{label}  |ψ|², t = {t:.3f}")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    fig.suptitle(
        "Time-dependent Schrödinger with periodic BC — wavepacket wrap-around "
        f"(ε={eps}, Δt={dt}, multiscale V period 1/16 < H = 1/{N_c})",
        y=1.01, fontsize=13,
    )
    save(fig, "wavepacket_periodic.png")


if __name__ == "__main__":
    main()
