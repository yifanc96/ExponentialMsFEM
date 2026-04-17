# Driver / example for the Helmholtz ExpMsFEM, mirroring Matlab
# helmholtz/case1/Exp/main.m. Runs a sweep over N_e = 1..N_e_max at k0 = 2
# on a plane-wave-impedance problem with a = v = 1 and reports relative L²
# and energy errors versus a fine-scale Q1 reference.
#
# Usage (threaded):
#   julia -t 4 Helmholtz/ExponentialMsFEMsolver.jl
#
# Environment variables:
#   EXPMSFEM_NCE     (default 8)    coarse cells per dimension
#   EXPMSFEM_NFE     (default 8)    fine cells per coarse cell
#   EXPMSFEM_NE_MAX  (default 6)    max edge eigenmodes per edge
#   EXPMSFEM_K0      (default 2.0)  wavenumber
#
# Note: at N_e = N_f − 1 the generalised eigenproblem sits at the rank
# boundary of R'NR (R has N_f − 1 rows) and can pick up a spurious mode.
# Keep N_e ≤ N_f − 2 for stable convergence — default N_e_max = 6 stays
# safely inside that band for N_f = 8.

include("PDEstruct.jl")
include("ExponentialMsFEMutility.jl")
using Logging
using Printf


# --- Coefficients (matches Matlab helmholtz/case1/Exp/*.m) ---
afun(t, s) = 1.0
vfun(t, s) = 1.0
betafun(t, s) = ((1 - t) * (1 - s) * t * s == 0.0) ? 1.0 : 0.0
ffun(t, s) = 0.0

# Per-side plane-wave impedance data: g(x, y, k0) = -i k0 a_side · exp(-0.6i k0 x - 0.8i k0 y)
# with a_side indexed by which boundary edge the point lies on.
function gfun(t, s, k0)
    if t == 0.0
        a_side = 0.4
    elseif t == 1.0
        a_side = 1.6
    elseif s == 0.0
        a_side = 0.2
    elseif s == 1.0
        a_side = 1.8
    else
        a_side = 0.0
    end
    return -1im * k0 * a_side * exp(-0.6im * k0 * t - 0.8im * k0 * s)
end


## Parameters ---------------------------------------------------------------
Nce    = parse(Int, get(ENV, "EXPMSFEM_NCE", "8"))
Nfe    = parse(Int, get(ENV, "EXPMSFEM_NFE", "8"))
N_e_max = parse(Int, get(ENV, "EXPMSFEM_NE_MAX", "6"))
k0     = parse(Float64, get(ENV, "EXPMSFEM_K0", "2.0"))

PDE = Helmholtz2D(k0, afun, vfun, betafun, ffun, gfun)
MsFEMparam = ExpHelmholtz_2d2ScaleUnifQuadMesh(Nce, Nfe)


## Fine-scale reference ----------------------------------------------------
println("[Helmholtz ref] fine-scale solve on ", Nce * Nfe, "×", Nce * Nfe, " grid")
@time u_ref, B_ref, C_ref, F_ref = Helmholtz_FineSolve(PDE, Nce * Nfe)
@info "|u_ref|.max = $(maximum(abs.(u_ref)))"


## Pre-factor workspace ----------------------------------------------------
ws = ExpHelmholtz_Workspace(MsFEMparam, PDE)
@time helm_prefactor_all!(ws)


## Sweep over N_e ----------------------------------------------------------
L2_err = zeros(N_e_max)
H1_err = zeros(N_e_max)
for N_e in 1:N_e_max
    println("\n---- Helmholtz ExpMsFEM with N_e = $N_e ----")
    @time fine_sol, _ = Helmholtz_ExpMsFEM_Solver(
        MsFEMparam, PDE, N_e; ws = ws, prefactor = false
    )
    e = u_ref .- fine_sol
    # Complex sesquilinear error forms (take real part of the quadratic form).
    num_L2 = real(e' * (C_ref * e))
    den_L2 = real(u_ref' * (C_ref * u_ref))
    num_H1 = real(e' * (B_ref * e))
    den_H1 = real(u_ref' * (B_ref * u_ref))
    L2_err[N_e] = sqrt(num_L2 / den_L2)
    H1_err[N_e] = sqrt(num_H1 / den_H1)
    @info @sprintf("[Helmholtz ExpMsFEM] N_e=%d  L2_rel=%.3e  H1_rel=%.3e",
                   N_e, L2_err[N_e], H1_err[N_e])
end

println("\n==== Helmholtz ExpMsFEM error summary (k0=$(k0), Nce=$(Nce), Nfe=$(Nfe)) ====")
for N_e in 1:N_e_max
    @printf("  N_e=%d  L2=%.3e  H1=%.3e\n", N_e, L2_err[N_e], H1_err[N_e])
end
