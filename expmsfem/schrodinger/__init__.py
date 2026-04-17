"""ExpMsFEM for the time-independent Schrödinger eigenvalue problem on
Ω = (0, 1)², in atomic units (ℏ = m = 1):

    H ψ = E ψ,    H = -½ Δ + V(x)

with Dirichlet-zero boundary conditions. The Hamiltonian is self-adjoint
and the eigenvalues are real.

The fine-scale reference uses `scipy.sparse.linalg.eigsh` with shift-invert
to find the lowest `k` eigenpairs. ExpMsFEM produces dense coarse
`H_coarse` and `M_coarse` matrices (size ≈ (N_c+1)² + 4·N_c·(N_c+1)·N_e);
`scipy.linalg.eigh` gives all their eigenpairs in one shot.

Entry points:

    from expmsfem.schrodinger.potentials import (V_harmonic_oscillator,
                                                 V_double_well, V_random)
    from expmsfem.schrodinger.driver import run_schrodinger_expmsfem
    from expmsfem.schrodinger.fem import solve_fine_eigenpairs

The edge-eigenbasis construction uses the Hamiltonian H (kinetic + potential)
as the local operator, so the multiscale basis adapts automatically to V(x)
— a rough or disordered potential exhibits the same kind of exponential
convergence in `N_e` as the rough coefficient does in the elliptic case.
"""
