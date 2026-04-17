"""Reproduce Matlab helmholtz/case1/Exp/main.m: N_c=N_f=8, k₀=2, N_e=1..7.

The Matlab script iterates N_e=1..7 and stores L² / H¹ errors. Our port runs
the same sweep. Note: at N_e = N_f - 1 the generalized eigenproblem sits right
at the rank boundary (R'NR has rank ≤ N_f - 1); the resulting mode is
numerically unreliable and Matlab's eigs() can produce similar artifacts.
For stable use keep N_e ≤ N_f - 2.

Run:
    .venv/bin/python examples/eg_helm.py [--N_c 8 --N_f 8 --k0 2 --N_e_max 6]
"""

import argparse
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from expmsfem.helmholtz import local_ops as hlo
from expmsfem.helmholtz.driver import run_expmsfem_helm
from expmsfem.helmholtz.fem import solve_fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_c", type=int, default=8)
    ap.add_argument("--N_f", type=int, default=8)
    ap.add_argument("--k0", type=float, default=2.0)
    ap.add_argument("--N_e_max", type=int, default=6)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--save", type=str, default="eg_helm.npz")
    args = ap.parse_args()

    N_c, N_f, k0 = args.N_c, args.N_f, args.k0
    print(f"N_c = {N_c}, N_f = {N_f}, k0 = {k0}, N fine = {N_c * N_f}")

    t0 = time.time()
    u_ref, B_ref, C_ref, _ = solve_fine(N_c * N_f, k0)
    print(f"fine reference: {time.time() - t0:.2f}s")

    ws = hlo.HelmWorkspace(N_c, N_f, k0)
    t0 = time.time()
    ws.prefactor_all(n_workers=args.n_workers)
    print(f"prefactor LUs: {time.time() - t0:.2f}s "
          f"({len(ws._cell)} cells, {len(ws._patch)} patches)")

    L2 = np.zeros(args.N_e_max)
    H1 = np.zeros(args.N_e_max)
    H1m = np.zeros(args.N_e_max)
    times = np.zeros(args.N_e_max)

    for N_e in range(1, args.N_e_max + 1):
        t_start = time.time()
        out = run_expmsfem_helm(
            N_c, N_f, N_e, k0,
            u_ref=u_ref, B_ref=B_ref, C_ref=C_ref,
            workspace=ws, n_workers=args.n_workers, verbose=True,
        )
        dt = time.time() - t_start
        L2[N_e - 1] = out["e_L2"]
        H1[N_e - 1] = out["e_H1"]
        H1m[N_e - 1] = out["e_H1_matlab"]
        times[N_e - 1] = dt
        print(f"N_e={N_e}: L2={out['e_L2']:.3e}  H1={out['e_H1']:.3e}  "
              f"time={dt:.1f}s")

    np.savez(args.save, N_c=N_c, N_f=N_f, k0=k0,
             N_e=np.arange(1, args.N_e_max + 1),
             L=L2, H=H1, H_matlab=H1m, times=times)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
