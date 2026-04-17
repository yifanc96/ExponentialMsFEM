# Driver / example for the Exponentially convergent Multiscale Finite Element
# Method (ExpMsFEM), mirroring Matlab elliptic/periodic/Exp/main.m.
#
# Usage (single-thread):
#   julia --project=. Elliptic/ExponentialMsFEMsolver.jl
# With threads:
#   julia -t 4 --project=. Elliptic/ExponentialMsFEMsolver.jl

include("PDEstruct.jl")
include("FEMutility.jl")
include("ExponentialMsFEMutility.jl")
using Logging
using ForwardDiff
using Printf


## PDE parameters (matches Matlab elliptic/periodic/Exp/afun.m exactly) ------
function afun(t, s)
    eps1 = 1/5; eps2 = 1/13; eps3 = 1/17; eps4 = 1/31; eps5 = 1/65
    return (1/6) * (
        (1.1 + sin(2π * t / eps1)) / (1.1 + sin(2π * s / eps1)) +
        (1.1 + sin(2π * s / eps2)) / (1.1 + cos(2π * t / eps2)) +
        (1.1 + cos(2π * t / eps3)) / (1.1 + sin(2π * s / eps3)) +
        (1.1 + sin(2π * s / eps4)) / (1.1 + cos(2π * t / eps4)) +
        (1.1 + cos(2π * t / eps5)) / (1.1 + sin(2π * s / eps5)) +
        sin(4 * s^2 * t^2) + 1.0
    )
end

function u(x)
    return sin(2π * x[1]) * sin(π * x[2]) * exp(x[1] + 2 * x[2])
end

function rhs(t, s)
    x = [t, s]
    return -sum(
        ForwardDiff.gradient(x -> afun(x[1], x[2]), x) .*
        ForwardDiff.gradient(u, x)
    ) - afun(x[1], x[2]) * tr(ForwardDiff.hessian(u, x))
end

bdy_Diri(t, s) = u([t, s])
bdy_Neum(t, s) = ForwardDiff.derivative(t -> u([t, s]), t)
function bdy_Robin(t, s)
    return 1.0, -ForwardDiff.derivative(t -> u([t, s]), t) + u([t, s])
end
bdy_type(t, s) = 1  # Dirichlet everywhere

PDEparam = VarElliptic(afun, rhs, bdy_type, bdy_Diri, bdy_Neum, bdy_Robin)


## Mesh parameters ---------------------------------------------------------
# For reproducing Matlab's main.m use Nce = Nfe = 32. For a quick smoke test
# keep them small.
Nce = get(ENV, "EXPMSFEM_NCE", "8") |> x -> parse(Int, x)
Nfe = get(ENV, "EXPMSFEM_NFE", "8") |> x -> parse(Int, x)
N_e_max = get(ENV, "EXPMSFEM_NE_MAX", "5") |> x -> parse(Int, x)

MsFEMparam = ExpMsFEM_2d2ScaleUnifQuadMesh(Nce, Nfe)


## Fine FEM reference ------------------------------------------------------
println("[ExpMsFEM] computing fine-scale reference on ", Nce * Nfe, "x", Nce * Nfe, " grid")
FEMparam = FEM_2dUnifQuadMesh(Nce * Nfe)
FEMstore = FEM_StiffnMassAssembly(FEMparam, PDEparam)
A_fine, F_fine = FEM_BdyRhsAssembly(FEMparam, PDEparam, FEMstore)
u_ref = A_fine \ F_fine
x = FEMparam.Grid_x
y = FEMparam.Grid_y
truth = [u([x[i], y[j]]) for j in 1:(Nce * Nfe + 1) for i in 1:(Nce * Nfe + 1)]
L2_fine = sqrt((truth - u_ref)' * FEMstore.M * (truth - u_ref) /
               (truth' * FEMstore.M * truth))
energy_fine = sqrt((truth - u_ref)' * FEMstore.A * (truth - u_ref) /
                   (truth' * FEMstore.A * truth))
@info "[Fine FEM ref] Relative energy err $(energy_fine), Relative L2 $(L2_fine)"


## ExpMsFEM sweep over N_e = 1..N_e_max ------------------------------------
ws = ExpMsFEM_Workspace(MsFEMparam, PDEparam)
@time prefactor_all!(ws)

L2_err = zeros(N_e_max)
H1_err = zeros(N_e_max)
for N_e in 1:N_e_max
    println("\n---- ExpMsFEM with N_e = $N_e ----")
    @time fine_sol, _ = ExpMsFEM_Solver(MsFEMparam, PDEparam, N_e;
                                        ws = ws, prefactor = false)
    L2 = sqrt((u_ref - fine_sol)' * FEMstore.M * (u_ref - fine_sol) /
              (u_ref' * FEMstore.M * u_ref))
    H1 = sqrt((u_ref - fine_sol)' * FEMstore.A * (u_ref - fine_sol) /
              (u_ref' * FEMstore.A * u_ref))
    L2_err[N_e] = L2
    H1_err[N_e] = H1
    @info @sprintf("[ExpMsFEM] N_e=%d  L2_rel=%.3e  H1_rel=%.3e",
                   N_e, L2, H1)
end

println("\n==== ExpMsFEM error summary ====")
for N_e in 1:N_e_max
    @printf("  N_e=%d  L2=%.3e  H1=%.3e\n", N_e, L2_err[N_e], H1_err[N_e])
end
