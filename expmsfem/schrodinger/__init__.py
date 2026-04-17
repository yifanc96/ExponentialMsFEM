"""ExpMsFEM for the 2D time-dependent Schrödinger equation on Ω = (0, 1)²
in the semi-classical regime, in atomic units (ℏ = m = 1):

    i ε ∂_t ψ = -½ ε² Δ ψ + V(x) ψ,   ψ(0, x) = ψ₀(x),   ψ|_∂Ω = 0.

We use a **backward-Euler** time discretisation; each step solves the
linear, complex, indefinite elliptic problem

    ( -½ ε² Δ + V - i ε / Δt ) ψ_{n+1} = -i ε / Δt · ψ_n

which fits the classical ExpMsFEM framework: harmonic-bubble
decomposition, edge-eigenbasis (`N_e` per edge), edge bubble, and a
ψ_n-driven **cell bubble** recomputed per step. Offline cost (per-cell
LU factors of the shifted complex operator, edge eigenbasis) is paid
once when Δt is held fixed — every time step is a cheap coarse solve
plus cell-bubble correction.

Entry points:

    from expmsfem.schrodinger.potentials import (V_harmonic_oscillator,
                                                 V_double_well,
                                                 V_periodic_lattice)
    from expmsfem.schrodinger.time_dep import (SemiclassicalParam,
                                                solve_fine_backward_euler)
    from expmsfem.schrodinger.propagator import run_expmsfem_schrodinger
"""
