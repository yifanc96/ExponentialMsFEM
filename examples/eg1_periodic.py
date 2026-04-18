"""Reproduce Matlab elliptic/periodic/Exp/main.m: exponential-convergence
sweep over N_e = 1..7 at N_c = N_f = 32 on the periodic multi-scale coefficient.

Run:
    .venv/bin/python examples/eg1_periodic.py [--N_c 32 --N_f 32 --N_e_max 7]
"""

import argparse
import os
import time

import numpy as np

# Serial BLAS inside threaded pool avoids oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine
from expmsfem import local_ops, element_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_c", type=int, default=32)
    ap.add_argument("--N_f", type=int, default=32)
    ap.add_argument("--N_e_max", type=int, default=7)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--save", type=str, default="eg1_periodic.npz")
    args = ap.parse_args()

    N_c, N_f = args.N_c, args.N_f
    print(f"N_c = {N_c}, N_f = {N_f}, N fine = {N_c * N_f}")

    t0 = time.time()
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    print(f"fine reference: {time.time() - t0:.2f}s")

    # Pre-factor the Workspace ONCE and pre-populate the per-edge eigen-basis
    # cache at the *largest* N_e we'll sweep. Each subsequent per-N_e run then
    # slices cached columns and skips the heavy eigendecomposition entirely.
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    t0 = time.time()
    ws.prefactor_all(n_workers=args.n_workers)
    print(f"prefactor LUs (shared): {time.time() - t0:.2f}s "
          f"({len(ws._cell)} cells, {len(ws._patch)} patches)")

    t0 = time.time()
    element_basis.prefactor_edges(ws, args.N_e_max, n_workers=args.n_workers)
    print(f"prefactor edges @ N_e={args.N_e_max}: {time.time() - t0:.2f}s "
          f"({len(ws._edge_cache)} edges, reused across the N_e sweep)")

    L2 = np.zeros(args.N_e_max)
    H1 = np.zeros(args.N_e_max)
    times = np.zeros(args.N_e_max)

    for N_e in range(1, args.N_e_max + 1):
        t_start = time.time()
        out = run_expmsfem(
            afun_periodic, N_c, N_f, N_e,
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
