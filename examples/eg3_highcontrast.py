"""Reproduce Matlab elliptic/Highcontrast/Exp/main.m: exponential-convergence
sweep on a 64× high-contrast coefficient with 49 circular inclusions.

Run:
    .venv/bin/python examples/eg3_highcontrast.py [--N_c 32 --N_f 32 --N_e_max 7]
"""

import argparse
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from expmsfem import element_basis, local_ops
from expmsfem.coefficients import afun_highcontrast
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_c", type=int, default=32)
    ap.add_argument("--N_f", type=int, default=32)
    ap.add_argument("--N_e_max", type=int, default=7)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--save", type=str, default="eg3_highcontrast.npz")
    args = ap.parse_args()

    N_c, N_f = args.N_c, args.N_f
    print(f"N_c = {N_c}, N_f = {N_f}, N fine = {N_c * N_f}")

    t0 = time.time()
    u_ref, K_ref, M_ref, _ = solve_fine(afun_highcontrast, N_c * N_f)
    print(f"fine reference: {time.time() - t0:.2f}s")

    ws = local_ops.Workspace(afun_highcontrast, N_c, N_f)
    t0 = time.time()
    ws.prefactor_all(n_workers=args.n_workers)
    print(f"prefactor LUs: {time.time() - t0:.2f}s")

    # Pre-populate edge cache at N_e_max so every sweep iteration just slices.
    t0 = time.time()
    element_basis.prefactor_edges(ws, args.N_e_max, n_workers=args.n_workers)
    print(f"prefactor edges @ N_e={args.N_e_max}: {time.time() - t0:.2f}s")

    L2 = np.zeros(args.N_e_max)
    H1 = np.zeros(args.N_e_max)
    times = np.zeros(args.N_e_max)

    for N_e in range(1, args.N_e_max + 1):
        t_start = time.time()
        out = run_expmsfem(
            afun_highcontrast, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=args.n_workers,
            workspace=ws,
            verbose=True,
        )
        dt = time.time() - t_start
        L2[N_e - 1] = out["e_L2"]
        H1[N_e - 1] = out["e_H1"]
        times[N_e - 1] = dt
        print(f"N_e={N_e}: L2={out['e_L2']:.3e}  H1={out['e_H1']:.3e}  "
              f"time={dt:.1f}s")

    np.savez(args.save, N_c=N_c, N_f=N_f, N_e=np.arange(1, args.N_e_max + 1),
             L=L2, H=H1, times=times)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
