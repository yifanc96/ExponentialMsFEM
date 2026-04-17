"""Compare Exp / H+bubble / O(H) on the periodic coefficient.

Reproduces the side-by-side convergence plot from the paper, mirroring
Matlab `elliptic/periodic/{Exp,H+bubble,O(H)}/main.m`. All three methods
share the same Workspace (so the LU prefactor cost is paid once across
methods).

Run:
    .venv/bin/python examples/eg1_periodic_baselines.py [--N_c 32 --N_f 32 --N_e_max 7]
"""

import argparse
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from expmsfem import local_ops
from expmsfem.baselines import run_hbubble, run_OH
from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_c", type=int, default=32)
    ap.add_argument("--N_f", type=int, default=32)
    ap.add_argument("--N_e_max", type=int, default=7)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--save", type=str, default="eg1_baselines.npz")
    args = ap.parse_args()

    N_c, N_f = args.N_c, args.N_f
    print(f"N_c = {N_c}, N_f = {N_f}, N fine = {N_c * N_f}")

    t0 = time.time()
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    print(f"fine reference: {time.time() - t0:.2f}s")

    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    t0 = time.time()
    ws.prefactor_all(n_workers=args.n_workers)
    print(f"prefactor LUs (shared): {time.time() - t0:.2f}s")

    methods = {
        "Exp": run_expmsfem,
        "H+bubble": run_hbubble,
        "O(H)": run_OH,
    }
    results = {name: {"L": [], "H": [], "times": []} for name in methods}

    for N_e in range(1, args.N_e_max + 1):
        for name, fn in methods.items():
            t0 = time.time()
            out = fn(
                afun_periodic, N_c, N_f, N_e,
                u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
                workspace=ws, n_workers=args.n_workers,
            )
            dt = time.time() - t0
            results[name]["L"].append(out["e_L2"])
            results[name]["H"].append(out["e_H1"])
            results[name]["times"].append(dt)
            print(f"  N_e={N_e} {name:>9}: "
                  f"L2={out['e_L2']:.3e}  H1={out['e_H1']:.3e}  {dt:.1f}s")

    save_dict = {"N_c": N_c, "N_f": N_f, "N_e": np.arange(1, args.N_e_max + 1)}
    for name, r in results.items():
        key = name.replace("+", "p").replace("(", "_").replace(")", "")
        save_dict[f"L_{key}"] = np.array(r["L"])
        save_dict[f"H_{key}"] = np.array(r["H"])
        save_dict[f"times_{key}"] = np.array(r["times"])
    np.savez(args.save, **save_dict)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
