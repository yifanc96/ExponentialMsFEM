"""Diagnostic plot: the generalised eigenvalues λ_k of the edge-basis
eigenproblem  R' N R v = λ P v  decay geometrically in k. This is the
theoretical source of ExpMsFEM's exponential convergence — keeping the top
N_e eigenvectors captures all but an exp(−αN_e) fraction of the patch
harmonic space's energy.

Plots the spectrum on a single representative interior edge for each of the
three elliptic coefficients.

Writes figures/eigenvalue_decay.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla

from _util import set_style, save
from expmsfem import local_ops
from expmsfem.coefficients import (
    afun_periodic, afun_random, afun_highcontrast, generate_random_field,
)


def spectrum(a_fun, N_c, N_f):
    L1, L2, N = local_ops.harmext(a_fun, m=N_c // 2, n=N_c // 2,
                                  N_c=N_c, N_f=N_f, i=1)
    R, P, _ = local_ops.restrict(a_fun, m=N_c // 2, n=N_c // 2,
                                 N_c=N_c, N_f=N_f, t=1)
    P_reg = P + 1e-14 * np.eye(P.shape[0])
    eigvals = sla.eigvalsh(R.T @ N @ R, P_reg)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    return eigvals


def main():
    set_style()
    N_c, N_f = 8, 16

    goo = generate_random_field(M=128, seed=0)
    problems = [
        ("Periodic", afun_periodic),
        ("Random rough", afun_random(goo, M=128)),
        ("High-contrast", afun_highcontrast),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = ["o", "s", "^"]
    for (label, a), m in zip(problems, markers):
        print(f"[demo_05] computing spectrum for: {label}")
        eigs = spectrum(a, N_c, N_f)
        # Normalise so the top eigenvalue is 1 — emphasises the decay rate
        eigs = eigs / eigs.max()
        top = min(len(eigs), 12)
        ax.semilogy(range(1, top + 1), eigs[:top], marker=m, label=label)
    ax.set_xlabel("mode index $k$")
    ax.set_ylabel("normalised $|\\lambda_k|$")
    ax.set_xticks(range(1, 13))
    ax.legend(title="coefficient")
    ax.set_title(
        "Generalised eigenvalues of $R^{\\!T} N R$ vs $P$ on a central edge\n"
        f"($N_c={N_c}$, $N_f={N_f}$) — geometric decay drives exp. convergence"
    )
    save(fig, "eigenvalue_decay.png")


if __name__ == "__main__":
    main()
