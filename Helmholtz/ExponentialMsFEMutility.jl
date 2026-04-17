# Exponentially Convergent Multiscale FEM for the 2D impedance Helmholtz
# problem on Ω = (0, 1)². Mirrors the Matlab `helmholtz/case1/Exp/*.m`
# reference.
#
# Key algorithmic differences from the elliptic case (see Elliptic/
# ExponentialMsFEMutility.jl):
#   * Complex-valued solution and stiffness throughout.
#   * Per-fine-cell element stiffness depends on position: cells touching
#     ∂Ω acquire an impedance term (−i k₀ β · edge-mass) on their boundary
#     edge. K is complex-symmetric (K[i,j] == K[j,i], not Hermitian).
#   * Sesquilinear forms: Matlab uses `K = value.' * B * conj(value)` and
#     solves `A.' \ F` (non-conjugate transpose).
#   * No Dirichlet elimination on ∂Ω — the impedance BC stays in the system.
#     A per-cell `dirichlet_indices` helper (corresponding to Matlab `bc.m`)
#     excludes ∂Ω sides from the local Dirichlet set.
#   * Generalised eigenproblem (R' N R) v = λ P v is complex-general — use
#     `eigen(M, P)` not Hermitian `eigh`, and sort by |λ| descending.

using SparseArrays
using LinearAlgebra
using Logging
import Base.Threads: @threads, nthreads


# Reference 4×4 Q1 matrices (scale-free on a square).
const _Q1_STIFF_REF = [
     2/3  -1/6  -1/3  -1/6
    -1/6   2/3  -1/6  -1/3
    -1/3  -1/6   2/3  -1/6
    -1/6  -1/3  -1/6   2/3
]

const _Q1_MASS_REF = [
    1/9   1/18  1/36  1/18
    1/18  1/9   1/18  1/36
    1/36  1/18  1/9   1/18
    1/18  1/36  1/18  1/9
]


# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------

struct ExpHelmholtz_2d2ScaleUnifQuadMesh{Ti,Tf}
    Nce::Ti
    Nfe::Ti
    Ne::Ti
    CGrid_x::Vector{Tf}
    CGrid_y::Vector{Tf}
    ElemNode_loc2glo::Function
end

function ExpHelmholtz_2d2ScaleUnifQuadMesh(Nce::Int, Nfe::Int)
    x = collect(LinRange(0, 1, Nce + 1))
    y = copy(x)
    function ElemNode_loc2glo(N, i, j, ind_node)
        if ind_node <= 2
            return (N + 1) * (j - 1) + i + ind_node - 1
        else
            return (N + 1) * j + i + 4 - ind_node
        end
    end
    @info "[Helmholtz mesh] $(Nce+1) coarse nodes per dim, $(Nce*Nfe+1) fine nodes per dim"
    return ExpHelmholtz_2d2ScaleUnifQuadMesh(Nce, Nfe, Nce * Nfe,
                                             x, y, ElemNode_loc2glo)
end


# -----------------------------------------------------------------------------
# Per-fine-cell element stiffness (complex K and real K_norm)
# -----------------------------------------------------------------------------

# For a Q1 cell of side h with corners
#   1 = (xlow, ylow), 2 = (xhigh, ylow), 3 = (xhigh, yhigh), 4 = (xlow, yhigh)
# we evaluate a, v at the cell centre and β at the 4 edge midpoints. The
# complex entries follow Matlab elementstiff1.m / elementstiff2.m exactly:
#
#   diagonal (i=i):
#       K_norm = 2/3 a + k0² v² h²/9
#       K      = 2/3 a − k0² v² h²/9 − i k0 β_corner[i] h/3
#
#   opposite corners (|i-j|=2):
#       K_norm = −1/3 a + k0² v² h²/36
#       K      = −1/3 a − k0² v² h²/36
#
#   adjacent corners (1,2),(2,3),(3,4)  (and transposes):
#       K_norm = −1/6 a + k0² v² h²/18
#       K      = −1/6 a − k0² v² h²/18 − i k0 β₁[min(i,j)] h/6
#     where β₁ = [β_bot, β_right, β_top] indexed by the shared edge
#
#   wrap (1,4) / (4,1):
#       K      = −1/6 a − k0² v² h²/18 − i k0 β_left h/6
#
# where β_corner[i] = β_edge_1_of_corner + β_edge_2_of_corner.
function _helm_element_matrices(PDE::Helmholtz2D,
                                xlow, xhigh, ylow, yhigh)
    xmid = 0.5 * (xlow + xhigh)
    ymid = 0.5 * (ylow + yhigh)
    h = xhigh - xlow
    a_c = PDE.afun(xmid, ymid)
    v_c = PDE.vfun(xmid, ymid)
    k0 = PDE.k0

    b_bot   = PDE.betafun(xmid, ylow)
    b_top   = PDE.betafun(xmid, yhigh)
    b_left  = PDE.betafun(xlow, ymid)
    b_right = PDE.betafun(xhigh, ymid)

    β_corner = (b_bot + b_left, b_bot + b_right, b_top + b_right, b_top + b_left)
    β1 = (b_bot, b_right, b_top)

    lap = a_c .* _Q1_STIFF_REF
    mass = (v_c ^ 2) * (h ^ 2) .* _Q1_MASS_REF

    K_norm = lap .+ (k0 ^ 2) .* mass
    K = Complex{Float64}.(lap .- (k0 ^ 2) .* mass)

    # Impedance contributions
    for i = 1:4
        K[i, i] -= 1im * k0 * β_corner[i] * h / 3
    end
    for (i, j, bi) in ((1, 2, 1), (2, 1, 1),
                       (2, 3, 2), (3, 2, 2),
                       (3, 4, 3), (4, 3, 3))
        K[i, j] -= 1im * k0 * β1[bi] * h / 6
    end
    K[1, 4] -= 1im * k0 * b_left * h / 6
    K[4, 1] -= 1im * k0 * b_left * h / 6

    return K, K_norm
end


# Per-fine-cell Q1 load combining interior source (ffun) + impedance boundary
# data (gfun at edge midpoints, picking up the h/2 edge-mass factor).
function _helm_element_load(PDE::Helmholtz2D, xlow, xhigh, ylow, yhigh)
    xmid = 0.5 * (xlow + xhigh)
    ymid = 0.5 * (ylow + yhigh)
    h = xhigh - xlow
    fi = PDE.ffun(xmid, ymid)
    g_bot   = PDE.gfun(xmid, ylow, PDE.k0)
    g_top   = PDE.gfun(xmid, yhigh, PDE.k0)
    g_left  = PDE.gfun(xlow, ymid, PDE.k0)
    g_right = PDE.gfun(xhigh, ymid, PDE.k0)
    base = fi * (h ^ 2) / 4
    return Complex{Float64}[
        base + (g_bot + g_left) * h / 2,
        base + (g_bot + g_right) * h / 2,
        base + (g_top + g_right) * h / 2,
        base + (g_top + g_left) * h / 2,
    ]
end


# -----------------------------------------------------------------------------
# Q1 sparse assembly on a rectangular fine grid
# -----------------------------------------------------------------------------

# Local-to-global node index on an (Nx+1) × (Ny+1) grid, counterclockwise
# 1 = (i,j), 2 = (i+1,j), 3 = (i+1,j+1), 4 = (i,j+1), row-major x-first.
function _loc2glo_rect(Nx, i, j, ind_node)
    if ind_node <= 2
        return (Nx + 1) * (j - 1) + i + ind_node - 1
    else
        return (Nx + 1) * j + i + 4 - ind_node
    end
end


function _assemble_helm_stiffness(PDE::Helmholtz2D, xs::AbstractVector, ys::AbstractVector)
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    nNodes = (Nx + 1) * (Ny + 1)
    nnz_per_cell = 16
    total = nnz_per_cell * Nx * Ny
    Irow = zeros(Int, total)
    Jcol = zeros(Int, total)
    Aval = zeros(Complex{Float64}, total)
    for j = 1:Ny
        ylow = ys[j]; yhigh = ys[j+1]
        for i = 1:Nx
            xlow = xs[i]; xhigh = xs[i+1]
            K, _ = _helm_element_matrices(PDE, xlow, xhigh, ylow, yhigh)
            for p = 1:4
                gp = _loc2glo_rect(Nx, i, j, p)
                for q = 1:4
                    gq = _loc2glo_rect(Nx, i, j, q)
                    idx = nnz_per_cell * Ny * (i - 1) + nnz_per_cell * (j - 1) + 4 * (p - 1) + q
                    Irow[idx] = gp
                    Jcol[idx] = gq
                    Aval[idx] = K[p, q]
                end
            end
        end
    end
    return sparse(Irow, Jcol, Aval, nNodes, nNodes)
end


# Assemble the real norm-reference matrix B (for energy norm) on the same grid.
function _assemble_helm_stiffness_norm(PDE::Helmholtz2D, xs::AbstractVector, ys::AbstractVector)
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    nNodes = (Nx + 1) * (Ny + 1)
    total = 16 * Nx * Ny
    Irow = zeros(Int, total); Jcol = zeros(Int, total); Aval = zeros(total)
    for j = 1:Ny, i = 1:Nx
        _, Kn = _helm_element_matrices(PDE, xs[i], xs[i+1], ys[j], ys[j+1])
        for p = 1:4, q = 1:4
            gp = _loc2glo_rect(Nx, i, j, p)
            gq = _loc2glo_rect(Nx, i, j, q)
            idx = 16 * Ny * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
            Irow[idx] = gp; Jcol[idx] = gq; Aval[idx] = Kn[p, q]
        end
    end
    return sparse(Irow, Jcol, Aval, nNodes, nNodes)
end


# Q1 mass assembly (real, no coefficient) — used for L² errors.
function _assemble_q1_mass(Nx, Ny, h)
    nNodes = (Nx + 1) * (Ny + 1)
    total = 16 * Nx * Ny
    Irow = zeros(Int, total); Jcol = zeros(Int, total); Vv = zeros(total)
    for j = 1:Ny, i = 1:Nx
        for p = 1:4, q = 1:4
            gp = _loc2glo_rect(Nx, i, j, p)
            gq = _loc2glo_rect(Nx, i, j, q)
            idx = 16 * Ny * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
            Irow[idx] = gp; Jcol[idx] = gq
            Vv[idx] = (h ^ 2) * _Q1_MASS_REF[p, q]
        end
    end
    return sparse(Irow, Jcol, Vv, nNodes, nNodes)
end


function _assemble_helm_load(PDE::Helmholtz2D, xs::AbstractVector, ys::AbstractVector)
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    nNodes = (Nx + 1) * (Ny + 1)
    F = zeros(Complex{Float64}, nNodes)
    for j = 1:Ny, i = 1:Nx
        floc = _helm_element_load(PDE, xs[i], xs[i+1], ys[j], ys[j+1])
        for p = 1:4
            F[_loc2glo_rect(Nx, i, j, p)] += floc[p]
        end
    end
    return F
end


# -----------------------------------------------------------------------------
# Fine-scale reference solver (mirrors Matlab helmholtz/case1/Exp/FEM.m)
# -----------------------------------------------------------------------------

# Returns the fine-scale solution, the real norm-matrix B, the Q1 mass C,
# and the complex load F. No Dirichlet elimination — the impedance term on
# ∂Ω cells is fully baked into the stiffness K.
function Helmholtz_FineSolve(PDE::Helmholtz2D, N_f::Int)
    xs = collect(LinRange(0, 1, N_f + 1))
    ys = copy(xs)
    K = _assemble_helm_stiffness(PDE, xs, ys)
    B = _assemble_helm_stiffness_norm(PDE, xs, ys)
    C = _assemble_q1_mass(N_f, N_f, 1.0 / N_f)
    F = _assemble_helm_load(PDE, xs, ys)
    # Matlab uses `A.' \ F` (non-conjugate transpose).
    u = transpose(K) \ F
    return u, B, C, F
end


# -----------------------------------------------------------------------------
# Dirichlet-set helper (bc.m equivalent)
# -----------------------------------------------------------------------------

# Returns the 1-indexed node indices on a (Nx+1)x(Ny+1) grid that should be
# Dirichlet-pinned for a patch at coarse position (m, n) with coarse-cell-
# count arguments (N_c_x, N_c_y). A side is "Dirichlet" only if it is
# interior to the coarse grid — domain-boundary sides keep the impedance
# Robin BC baked into the stiffness matrix.
function helm_dirichlet_indices(N_c_x::Int, N_c_y::Int,
                                Nx::Int, Ny::Int, m::Int, n::Int)
    perim = reduce(vcat, collect.([
        1:Nx+1,
        Nx+2:Nx+1:(Nx+1)*(Ny+1),
        2*(Nx+1):Nx+1:(Nx+1)*(Ny+1),
        Nx*Ny+Ny+2:(Nx+1)*(Ny+1)-1,
    ]))
    a = Set(perim)
    b = Set(perim)
    if n > 1
        setdiff!(b, Set(collect(1:Nx+1)))
    end
    if n < N_c_y
        setdiff!(b, Set(collect(Nx*Ny+Ny+1:(Nx+1)*(Ny+1))))
    end
    if m > 1
        setdiff!(b, Set(collect(1:Nx+1:(Nx+1)*(Ny+1))))
    end
    if m < N_c_x
        setdiff!(b, Set(collect(Nx+1:Nx+1:(Nx+1)*(Ny+1))))
    end
    setdiff!(a, b)
    return sort!(collect(a))
end


# -----------------------------------------------------------------------------
# Complex Dirichlet-solve factor
# -----------------------------------------------------------------------------

struct ComplexInteriorFactor
    A::SparseMatrixCSC{Complex{Float64}, Int}
    interior::Vector{Int}
    bdy::Vector{Int}
    A_ib::SparseMatrixCSC{Complex{Float64}, Int}
    lu::Any  # UmfpackLU{Complex{Float64},Int} factorisation
    lu_T::Any  # factorisation of A_ii transposed (for `A.' \ F` solves)
end

function ComplexInteriorFactor(A::SparseMatrixCSC{Complex{Float64}, Int},
                               bdy::Vector{Int})
    n = size(A, 1)
    mask = trues(n)
    mask[bdy] .= false
    interior = findall(mask)
    A_ii = A[interior, interior]
    A_ib = A[interior, bdy]
    return ComplexInteriorFactor(A, interior, bdy, A_ib,
                                 lu(A_ii), lu(sparse(transpose(A_ii))))
end


# Solve A u = F with u[bdy] = bdy_vals. `transpose_solve` picks whether to
# use the factorisation of A_ii or A_ii.'  (the Matlab `\.'` convention).
function interior_solve(fac::ComplexInteriorFactor, F, bdy_vals;
                        transpose_solve::Bool = false)
    if ndims(F) == 1
        rhs = F[fac.interior] .- vec(fac.A_ib * bdy_vals)
        u_i = transpose_solve ? (fac.lu_T \ rhs) : (fac.lu \ rhs)
        u = zeros(Complex{Float64}, size(fac.A, 1))
        u[fac.interior] .= u_i
        u[fac.bdy] .= bdy_vals
        return u
    else
        k = size(F, 2)
        rhs = F[fac.interior, :] .- fac.A_ib * bdy_vals
        u_i = transpose_solve ? (fac.lu_T \ rhs) : (fac.lu \ rhs)
        u = zeros(Complex{Float64}, size(fac.A, 1), k)
        u[fac.interior, :] .= u_i
        u[fac.bdy, :] .= bdy_vals
        return u
    end
end


# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

mutable struct ExpHelmholtz_Workspace
    MsFEMparam::ExpHelmholtz_2d2ScaleUnifQuadMesh
    PDE::Helmholtz2D
    cell_factors::Dict{Tuple{Int,Int}, ComplexInteriorFactor}
    # (ci, cj, t) → (factor, Nx, Ny, m0, n0)
    patch_factors::Dict{Tuple{Int,Int,Int},
                        Tuple{ComplexInteriorFactor, Int, Int, Int, Int}}
    edge_cache::Dict{Tuple{Int,Int,Int},
                     Tuple{Matrix{Complex{Float64}}, Matrix{Complex{Float64}},
                           Vector{Complex{Float64}}, Vector{Complex{Float64}}}}
end

ExpHelmholtz_Workspace(mp, PDE) = ExpHelmholtz_Workspace(mp, PDE,
    Dict{Tuple{Int,Int}, ComplexInteriorFactor}(),
    Dict{Tuple{Int,Int,Int}, Tuple{ComplexInteriorFactor, Int, Int, Int, Int}}(),
    Dict{Tuple{Int,Int,Int}, Tuple{Matrix{Complex{Float64}}, Matrix{Complex{Float64}},
                                    Vector{Complex{Float64}}, Vector{Complex{Float64}}}}())


# -----------------------------------------------------------------------------
# basefun / basefun1 (complex)
# -----------------------------------------------------------------------------

function helm_basefun(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    xs = LinRange(mp.CGrid_x[ci], mp.CGrid_x[ci+1], mp.Nfe + 1)
    ys = LinRange(mp.CGrid_y[cj], mp.CGrid_y[cj+1], mp.Nfe + 1)
    return _assemble_helm_stiffness(ws.PDE, xs, ys)
end


function helm_basefun1(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int, t::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe
    if t == 1
        if ci == 1
            m0, Nx_cells = 1, 2
        elseif ci == Nce
            m0, Nx_cells = Nce - 1, 2
        else
            m0, Nx_cells = ci - 1, 3
        end
        n0, Ny_cells = cj, 2
    else
        if cj == 1
            n0, Ny_cells = 1, 2
        elseif cj == Nce
            n0, Ny_cells = Nce - 1, 2
        else
            n0, Ny_cells = cj - 1, 3
        end
        m0, Nx_cells = ci, 2
    end
    xs = LinRange(mp.CGrid_x[m0], mp.CGrid_x[m0 + Nx_cells], Nx_cells * Nfe + 1)
    ys = LinRange(mp.CGrid_y[n0], mp.CGrid_y[n0 + Ny_cells], Ny_cells * Nfe + 1)
    A = _assemble_helm_stiffness(ws.PDE, xs, ys)
    return A, length(xs) - 1, length(ys) - 1, m0, n0
end


# -----------------------------------------------------------------------------
# Prefactor caches
# -----------------------------------------------------------------------------

function build_cell_factor!(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int)
    A = helm_basefun(ws, ci, cj)
    mp = ws.MsFEMparam
    bdy = helm_dirichlet_indices(mp.Nce, mp.Nce, mp.Nfe, mp.Nfe, ci, cj)
    fac = ComplexInteriorFactor(A, bdy)
    ws.cell_factors[(ci, cj)] = fac
    return fac
end


function build_patch_factor!(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int, t::Int)
    A, Nx, Ny, m0, n0 = helm_basefun1(ws, ci, cj, t)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    # Matlab restrict.m feeds `bc(N_c-2, N_c-1, N_x, N_y, m-1, n)` for t=1
    # and `bc(N_c-1, N_c-2, N_x, N_y, m, n-1)` for t=2.
    if t == 1
        bdy = helm_dirichlet_indices(Nce - 2, Nce - 1, Nx, Ny, ci - 1, cj)
    else
        bdy = helm_dirichlet_indices(Nce - 1, Nce - 2, Nx, Ny, ci, cj - 1)
    end
    fac = ComplexInteriorFactor(A, bdy)
    ws.patch_factors[(ci, cj, t)] = (fac, Nx, Ny, m0, n0)
    return fac
end


function helm_prefactor_all!(ws::ExpHelmholtz_Workspace)
    Nce = ws.MsFEMparam.Nce
    cell_keys = [(ci, cj) for cj = 1:Nce for ci = 1:Nce]
    patch_keys = vcat(
        [(ci, cj, 1) for cj = 1:(Nce - 1) for ci = 1:Nce],
        [(ci, cj, 2) for cj = 1:Nce for ci = 1:(Nce - 1)],
    )

    println("[Helmholtz ExpMsFEM] prefactoring $(length(cell_keys)) cells and $(length(patch_keys)) patches with $(nthreads()) threads")

    cr = Vector{ComplexInteriorFactor}(undef, length(cell_keys))
    @threads for k = 1:length(cell_keys)
        ci, cj = cell_keys[k]
        A = helm_basefun(ws, ci, cj)
        mp = ws.MsFEMparam
        bdy = helm_dirichlet_indices(mp.Nce, mp.Nce, mp.Nfe, mp.Nfe, ci, cj)
        cr[k] = ComplexInteriorFactor(A, bdy)
    end
    for (k, key) in enumerate(cell_keys)
        ws.cell_factors[key] = cr[k]
    end

    pr = Vector{Tuple{ComplexInteriorFactor, Int, Int, Int, Int}}(undef, length(patch_keys))
    @threads for k = 1:length(patch_keys)
        ci, cj, t = patch_keys[k]
        A, Nx, Ny, m0, n0 = helm_basefun1(ws, ci, cj, t)
        mp = ws.MsFEMparam
        if t == 1
            bdy = helm_dirichlet_indices(mp.Nce - 2, mp.Nce - 1, Nx, Ny, ci - 1, cj)
        else
            bdy = helm_dirichlet_indices(mp.Nce - 1, mp.Nce - 2, Nx, Ny, ci, cj - 1)
        end
        pr[k] = (ComplexInteriorFactor(A, bdy), Nx, Ny, m0, n0)
    end
    for (k, key) in enumerate(patch_keys)
        ws.patch_factors[key] = pr[k]
    end

    @info "[Helmholtz ExpMsFEM] prefactor complete: $(length(ws.cell_factors)) cells, $(length(ws.patch_factors)) patches"
    return nothing
end


# -----------------------------------------------------------------------------
# bubble (complex, with ffun + gfun RHS)
# -----------------------------------------------------------------------------

function helm_bubble(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    fac = get(ws.cell_factors, (ci, cj), nothing)
    if fac === nothing
        fac = build_cell_factor!(ws, ci, cj)
    end
    xs = collect(LinRange(mp.CGrid_x[ci], mp.CGrid_x[ci+1], mp.Nfe + 1))
    ys = collect(LinRange(mp.CGrid_y[cj], mp.CGrid_y[cj+1], mp.Nfe + 1))
    F = _assemble_helm_load(ws.PDE, xs, ys)
    bdy_vals = zeros(Complex{Float64}, length(fac.bdy))
    return interior_solve(fac, F, bdy_vals; transpose_solve = true)
end


# -----------------------------------------------------------------------------
# harmext (complex)
# -----------------------------------------------------------------------------

# Mirrors Matlab harmext.m: for each shared edge, solve `K1 u = 0` on cell
# (ci, cj) with identity-hat Dirichlet data on the shared-edge interior, same
# for (ci, cj+1) (if t=1) or (ci+1, cj) (if t=2). Non-transpose solve matches
# the Matlab `K1 \ F1` used there.
function helm_harmext(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int, t::Int)
    mp = ws.MsFEMparam
    Nfe = mp.Nfe
    fac1 = get(ws.cell_factors, (ci, cj), nothing)
    if fac1 === nothing
        fac1 = build_cell_factor!(ws, ci, cj)
    end
    other = t == 1 ? (ci, cj + 1) : (ci + 1, cj)
    fac2 = get(ws.cell_factors, other, nothing)
    if fac2 === nothing
        fac2 = build_cell_factor!(ws, other...)
    end

    # Build Dirichlet data of shape (|bdy|, Nfe-1): identity placed on the
    # shared-edge-interior rows of each cell's Dirichlet set.
    function _identity_on_shared_edge(bdy::Vector{Int}, side::Symbol)
        target = if side == :top
            [Nfe * (Nfe + 1) + k for k = 2:Nfe]
        elseif side == :bottom
            collect(2:Nfe)
        elseif side == :right
            [k * (Nfe + 1) + Nfe + 1 for k = 1:(Nfe - 1)]
        elseif side == :left
            [k * (Nfe + 1) + 1 for k = 1:(Nfe - 1)]
        else
            error("bad side")
        end
        f = zeros(Complex{Float64}, length(bdy), Nfe - 1)
        pos = Dict(b => k for (k, b) in enumerate(bdy))
        for (col, idx) in enumerate(target)
            if haskey(pos, idx)
                f[pos[idx], col] = 1.0 + 0im
            end
        end
        return f
    end

    f1 = _identity_on_shared_edge(fac1.bdy, t == 1 ? :top : :right)
    f2 = _identity_on_shared_edge(fac2.bdy, t == 1 ? :bottom : :left)

    nNodes = (Nfe + 1) ^ 2
    F_zero = zeros(Complex{Float64}, nNodes, Nfe - 1)
    L1 = interior_solve(fac1, F_zero, f1; transpose_solve = false)
    L2 = interior_solve(fac2, F_zero, f2; transpose_solve = false)

    # N = L1' K1 L1 + L2' K2 L2 (Matlab uses ' = conjugate transpose)
    N = L1' * (fac1.A * L1) .+ L2' * (fac2.A * L2)
    return L1, L2, N
end


# -----------------------------------------------------------------------------
# restrict (complex)
# -----------------------------------------------------------------------------

# Returns (R, P, bub). P is complex-symmetric (Matlab uses `harm.' * A *
# conj(harm)`). R and bub are the shared-edge traces after linear-endpoint
# subtraction and endpoint-row drop.
function helm_restrict(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int, t::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe

    pfac = get(ws.patch_factors, (ci, cj, t), nothing)
    if pfac === nothing
        build_patch_factor!(ws, ci, cj, t)
        pfac = ws.patch_factors[(ci, cj, t)]
    end
    fac, Nx, Ny, m0, n0 = pfac

    # Active Dirichlet DOFs — Matlab restrict.m ladder:
    # Interior patches use `speye(2(Nx+Ny)-1)` over the full perimeter
    # (except last DOF). Boundary patches use `speye(length(b))` directly
    # (all entries of the Dirichlet set b).
    interior_patch = (t == 1 && 2 < ci < Nce - 1 && 1 < cj < Nce - 1) ||
                     (t == 2 && 2 < cj < Nce - 1 && 1 < ci < Nce - 1)

    if interior_patch
        P_size = 2 * (Nx + Ny)
        n_active = P_size - 1
        f = zeros(Complex{Float64}, length(fac.bdy), n_active)
        # place identity on the first n_active rows of fac.bdy (= the
        # perimeter set minus the very last DOF)
        for k = 1:n_active
            f[k, k] = 1.0 + 0im
        end
    else
        n_active = length(fac.bdy)
        f = Matrix{Complex{Float64}}(I, n_active, n_active)
    end

    n_patch = (Nx + 1) * (Ny + 1)
    F_zero = zeros(Complex{Float64}, n_patch, n_active)
    harm = interior_solve(fac, F_zero, f; transpose_solve = false)

    # Patch bubble with the true ffun/gfun load
    xs = collect(LinRange(mp.CGrid_x[m0], mp.CGrid_x[m0 + div(Nx, Nfe)], Nx + 1))
    ys = collect(LinRange(mp.CGrid_y[n0], mp.CGrid_y[n0 + div(Ny, Nfe)], Ny + 1))
    G = _assemble_helm_load(ws.PDE, xs, ys)
    bub_full = interior_solve(fac, G, zeros(Complex{Float64}, length(fac.bdy));
                              transpose_solve = false)

    # P = harm.' * A * conj(harm) (Matlab convention)
    P = transpose(harm) * (fac.A * conj.(harm))
    P = 0.5 .* (P .+ transpose(P))  # complex-symmetrise

    # Identify edge nodes in the patch coordinate system
    leng = Nfe + 1
    if t == 1
        col_start, col_end = ci == 1 ? (1, Nfe + 1) : (Nfe + 1, 2 * Nfe + 1)
        row_idx = Nfe + 1
        edge_nodes = [(row_idx - 1) * (Nx + 1) + k for k = col_start:col_end]
    else
        row_start, row_end = cj == 1 ? (1, Nfe + 1) : (Nfe + 1, 2 * Nfe + 1)
        col_idx = Nfe + 1
        edge_nodes = [(k - 1) * (Nx + 1) + col_idx for k = row_start:row_end]
    end

    R_full = harm[edge_nodes, :]
    bub_edge = bub_full[edge_nodes]
    lin_left = collect(LinRange(1, 0, leng))
    lin_right = collect(LinRange(0, 1, leng))
    R_full = R_full .- lin_left * R_full[1:1, :] .- lin_right * R_full[leng:leng, :]
    bub_edge = bub_edge .- lin_left * bub_edge[1] .- lin_right * bub_edge[leng]
    R = R_full[2:(leng - 1), :]
    bub_edge = bub_edge[2:(leng - 1)]

    # Matlab fallback: if bubble trace is identically zero (interior patches
    # with f ≡ 0 whose boundary data doesn't reach the shared edge), use a
    # parabolic stand-in so the element stiffness stays non-singular.
    if norm(bub_edge) == 0
        for i = 1:(leng - 2)
            bub_edge[i] = i * (leng - 1 - i) / (leng - 1) ^ 2
        end
    end
    return R, P, bub_edge
end


# -----------------------------------------------------------------------------
# Eigen modes (complex general)
# -----------------------------------------------------------------------------

# Top-N_e modes of (R' N R, P) by |λ|. Both M and P are complex but neither
# need be Hermitian here (P is complex-symmetric due to the Matlab transpose
# convention). Rank(M) ≤ R.shape[1] (the shared-edge interior dimension), so
# we clamp N_e accordingly.
function _helm_eigen_modes(R::AbstractMatrix, N::AbstractMatrix,
                           P::AbstractMatrix, N_e::Int)
    M = R' * N * R
    Preg = P + 1e-12 * I
    vals, vecs = eigen(M, Preg)
    order = sortperm(abs.(vals), rev = true)
    k_eff = min(N_e, length(order))
    V = vecs[:, order[1:k_eff]]
    if k_eff < N_e
        V = hcat(V, zeros(Complex{Float64}, size(V, 1), N_e - k_eff))
    end
    return V
end


# -----------------------------------------------------------------------------
# Per-edge build + cache
# -----------------------------------------------------------------------------

function _helm_build_edge_data(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int,
                               t::Int, N_e::Int)
    L1, L2, N = helm_harmext(ws, ci, cj, t)
    R, P, bub = helm_restrict(ws, ci, cj, t)
    V = _helm_eigen_modes(R, N, P, N_e)
    RV = R * V
    return (L1 * RV, L2 * RV, vec(L1 * bub), vec(L2 * bub))
end


function helm_prefactor_edges!(ws::ExpHelmholtz_Workspace, N_e::Int)
    empty!(ws.edge_cache)
    Nce = ws.MsFEMparam.Nce
    keys = vcat(
        [(ci, cj, 1) for cj = 1:(Nce - 1) for ci = 1:Nce],
        [(ci, cj, 2) for cj = 1:Nce for ci = 1:(Nce - 1)],
    )
    results = Vector{Tuple{Matrix{Complex{Float64}}, Matrix{Complex{Float64}},
                           Vector{Complex{Float64}}, Vector{Complex{Float64}}}}(
                          undef, length(keys))
    @threads for k = 1:length(keys)
        ci, cj, t = keys[k]
        results[k] = _helm_build_edge_data(ws, ci, cj, t, N_e)
    end
    for (k, key) in enumerate(keys)
        ws.edge_cache[key] = results[k]
    end
    @info "[Helmholtz ExpMsFEM] prefactored $(length(keys)) interior edges"
    return nothing
end


# -----------------------------------------------------------------------------
# Per-cell basis (nodal + edge contributions)
# -----------------------------------------------------------------------------

# Analytical Q1 hat values at the Dirichlet-pinned nodes of a cell. For a cell
# with fine grid of (Nfe+1)x(Nfe+1) nodes, the 4 corner hats evaluated at
# node (i, j) are:
#   φ_1 = (1 - u)(1 - v)     φ_2 = u(1 - v)
#   φ_3 = u v                 φ_4 = (1 - u) v
# with u = (i - 1) / Nfe, v = (j - 1) / Nfe (1-indexed i, j).
function _nodal_boundary_hats(Nfe::Int, bdy::Vector{Int})
    f = zeros(Complex{Float64}, length(bdy), 4)
    for (k, idx) in enumerate(bdy)
        # idx is 1-indexed; recover (i, j) on a (Nfe+1)x(Nfe+1) grid
        j = div(idx - 1, Nfe + 1) + 1
        i = idx - (j - 1) * (Nfe + 1)
        u = (i - 1) / Nfe
        v = (j - 1) / Nfe
        f[k, 1] = (1 - u) * (1 - v)
        f[k, 2] = u * (1 - v)
        f[k, 3] = u * v
        f[k, 4] = (1 - u) * v
    end
    return f
end


function _helm_nodal_basis(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    fac = ws.cell_factors[(ci, cj)]
    nNodes = (mp.Nfe + 1) ^ 2
    F_zero = zeros(Complex{Float64}, nNodes, 4)
    hats = _nodal_boundary_hats(mp.Nfe, fac.bdy)
    return interior_solve(fac, F_zero, hats; transpose_solve = false)
end


function helm_ElementBasis(ws::ExpHelmholtz_Workspace, ci::Int, cj::Int, N_e::Int)
    mp = ws.MsFEMparam
    Nfe = mp.Nfe
    Nce = mp.Nce
    count = 4 + 4 * (N_e + 1)
    nNodes = (Nfe + 1) ^ 2
    value = zeros(Complex{Float64}, nNodes, count)

    value[:, 1:4] = _helm_nodal_basis(ws, ci, cj)

    block = N_e + 1
    off = 4
    # Bottom (shared with (ci, cj-1), lower cell = (ci, cj-1), use L2 side)
    if cj > 1
        L1RV, L2RV, L1b, L2b = ws.edge_cache[(ci, cj - 1, 1)]
        value[:, (off + 1):(off + N_e)] = L2RV
        value[:, off + N_e + 1] = L2b
    end
    off += block
    # Top (shared with (ci, cj+1), lower cell = (ci, cj), L1)
    if cj < Nce
        L1RV, L2RV, L1b, L2b = ws.edge_cache[(ci, cj, 1)]
        value[:, (off + 1):(off + N_e)] = L1RV
        value[:, off + N_e + 1] = L1b
    end
    off += block
    # Left (shared with (ci-1, cj), lower = (ci-1, cj), L2)
    if ci > 1
        L1RV, L2RV, L1b, L2b = ws.edge_cache[(ci - 1, cj, 2)]
        value[:, (off + 1):(off + N_e)] = L2RV
        value[:, off + N_e + 1] = L2b
    end
    off += block
    # Right (shared with (ci+1, cj), lower = (ci, cj), L1)
    if ci < Nce
        L1RV, L2RV, L1b, L2b = ws.edge_cache[(ci, cj, 2)]
        value[:, (off + 1):(off + N_e)] = L1RV
        value[:, off + N_e + 1] = L1b
    end

    # Sesquilinear element stiffness: K = value.' * A * conj(value)
    A = ws.cell_factors[(ci, cj)].A
    K_elem = transpose(value) * (A * conj.(value))

    # Element load: Matlab writes F[row,:] += ffun·h²/4·conj(value[row,:])
    # then f = sum(F, dims=1). So f[k] = Σ_row F_fine[row]·conj(value[row,k])
    # = (conj(value)^T · F_fine)[k] = (value' · F_fine)[k] in Julia, where
    # `value'` is the conjugate transpose.
    xs = collect(LinRange(mp.CGrid_x[ci], mp.CGrid_x[ci+1], Nfe + 1))
    ys = collect(LinRange(mp.CGrid_y[cj], mp.CGrid_y[cj+1], Nfe + 1))
    F_fine = _assemble_helm_load(ws.PDE, xs, ys)
    f_elem = value' * F_fine

    # Matlab elementstiff.m subtracts the bubble contribution:
    #   f <- f − bubble.' * B * conj(value)
    bub_full = helm_bubble(ws, ci, cj)
    f_elem = f_elem .- vec(transpose(bub_full) * (A * conj.(value)))

    return value, A, K_elem, f_elem
end


# -----------------------------------------------------------------------------
# Global assembly
# -----------------------------------------------------------------------------

function _helm_global_dof(Nce, ci, cj, p, N_e, loc2glo)
    N_p = N_e + 1
    if p <= 4
        return loc2glo(Nce, ci, cj, p)
    elseif p <= 4 + N_p
        return (ci - 1 + (cj - 1) * Nce) * N_p + (Nce + 1) ^ 2 + p - 4
    elseif p <= 4 + 2 * N_p
        return (ci - 1 + cj * Nce) * N_p + (Nce + 1) ^ 2 + p - 4 - N_p
    elseif p <= 4 + 3 * N_p
        return (ci - 1 + (cj - 1) * (Nce + 1) + Nce * (Nce + 1)) * N_p +
               (Nce + 1) ^ 2 + p - 4 - 2 * N_p
    else
        return (ci + (cj - 1) * (Nce + 1) + Nce * (Nce + 1)) * N_p +
               (Nce + 1) ^ 2 + p - 4 - 3 * N_p
    end
end


# Domain-boundary DOFs for the Helmholtz global system: only the 4 corners
# of Ω are pinned on the nodal side (non-corner ∂Ω nodal DOFs stay free —
# the impedance BC is carried in the element stiffness). Edge-aux DOFs for
# horizontal edges at k=0 or k=Nce and vertical edges at m=0 or m=Nce are
# pinned to zero (the corresponding basis columns were set to zero).
function _helm_global_bdy(Nce::Int, N_e::Int)
    N_p = N_e + 1
    bdy = Int[]
    # 4 corners of Ω
    push!(bdy, 1)
    push!(bdy, Nce + 1)
    push!(bdy, (Nce + 1) * Nce + 1)
    push!(bdy, (Nce + 1) ^ 2)
    base_h = (Nce + 1) ^ 2
    # horizontal at k=0: cells (ci=1..Nce, cj=1 via their "bottom" slot)
    # That DOF has global index (ci-1 + 0*Nce)*N_p + base_h + 1..N_p
    for ci = 1:Nce
        for slot = 1:N_p
            push!(bdy, base_h + (ci - 1) * N_p + slot)
        end
    end
    # horizontal at k=Nce: ci=1..Nce, cj=Nce, top slot of that cell
    for ci = 1:Nce
        for slot = 1:N_p
            push!(bdy, base_h + (ci - 1 + Nce * Nce) * N_p + slot)
        end
    end
    base_v = base_h + Nce * (Nce + 1) * N_p
    # vertical at m=0: x-col 0, n=0..Nce-1
    for n = 0:(Nce - 1)
        for slot = 1:N_p
            push!(bdy, base_v + (0 + n * (Nce + 1)) * N_p + slot)
        end
    end
    # vertical at m=Nce
    for n = 0:(Nce - 1)
        for slot = 1:N_p
            push!(bdy, base_v + (Nce + n * (Nce + 1)) * N_p + slot)
        end
    end
    return sort!(unique!(bdy))
end


function Helmholtz_ExpMsFEM_GlobalAssembly(ws::ExpHelmholtz_Workspace, N_e::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe
    N_p = N_e + 1
    count = 4 + 4 * N_p
    nNodes = (Nce + 1) ^ 2 + 2 * Nce * (Nce + 1) * N_p

    val_all = zeros(Complex{Float64}, (Nfe + 1) ^ 2, count, Nce, Nce)
    K_all   = zeros(Complex{Float64}, count, count, Nce, Nce)
    f_all   = zeros(Complex{Float64}, count, Nce, Nce)
    bub_all = zeros(Complex{Float64}, (Nfe + 1) ^ 2, Nce, Nce)

    println("[Helmholtz ExpMsFEM] assembling per-cell bases with $(nthreads()) threads")
    cell_keys = [(ci, cj) for cj = 1:Nce for ci = 1:Nce]
    @threads for k = 1:length(cell_keys)
        ci, cj = cell_keys[k]
        value, _, K_elem, f_elem = helm_ElementBasis(ws, ci, cj, N_e)
        val_all[:, :, ci, cj] = value
        K_all[:, :, ci, cj] = K_elem
        f_all[:, ci, cj] = f_elem
        bub_all[:, ci, cj] = helm_bubble(ws, ci, cj)
    end
    @info "[Helmholtz ExpMsFEM] bases assembled"

    # Global COO assembly
    nnz_total = count ^ 2 * Nce ^ 2
    Ir = zeros(Int, nnz_total); Jc = zeros(Int, nnz_total)
    Vv = zeros(Complex{Float64}, nnz_total)
    F = zeros(Complex{Float64}, nNodes)
    for cj = 1:Nce, ci = 1:Nce
        gidx = [_helm_global_dof(Nce, ci, cj, p, N_e, mp.ElemNode_loc2glo)
                for p in 1:count]
        for p = 1:count
            for q = 1:count
                idx = count ^ 2 * Nce * (ci - 1) + count ^ 2 * (cj - 1) +
                      count * (p - 1) + q
                Ir[idx] = gidx[p]; Jc[idx] = gidx[q]
                Vv[idx] = K_all[p, q, ci, cj]
            end
            F[gidx[p]] += f_all[p, ci, cj]
        end
    end
    A = sparse(Ir, Jc, Vv, nNodes, nNodes)

    bdy = _helm_global_bdy(Nce, N_e)
    A[bdy, :] .= 0
    A[:, bdy] .= 0
    F[bdy] .= 0
    A[bdy, bdy] .= sparse(LinearAlgebra.I, length(bdy), length(bdy))

    # Matlab solve: u = A.' \ F (non-conjugate transpose)
    u_global = transpose(A) \ F
    @info "[Helmholtz ExpMsFEM] global linear system solved"

    return u_global, val_all, bub_all, N_e
end


function Helmholtz_ExpMsFEM_FineConstruct(ws::ExpHelmholtz_Workspace,
                                          u_global::Vector{Complex{Float64}},
                                          val_all::Array{Complex{Float64}, 4},
                                          bub_all::Array{Complex{Float64}, 3},
                                          N_e::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe
    count = 4 + 4 * (N_e + 1)
    fine = zeros(Complex{Float64}, Nce * Nfe + 1, Nce * Nfe + 1)

    @threads for k = 1:(Nce * Nce)
        cj = div(k - 1, Nce) + 1
        ci = mod(k - 1, Nce) + 1
        zs = zeros(Complex{Float64}, (Nfe + 1) ^ 2)
        for p = 1:count
            gp = _helm_global_dof(Nce, ci, cj, p, N_e, mp.ElemNode_loc2glo)
            zs .+= u_global[gp] .* val_all[:, p, ci, cj]
        end
        zs .+= bub_all[:, ci, cj]
        fine[(ci - 1) * Nfe + 1 : ci * Nfe + 1,
             (cj - 1) * Nfe + 1 : cj * Nfe + 1] = reshape(zs, Nfe + 1, Nfe + 1)
    end
    return reshape(fine, (Nce * Nfe + 1) ^ 2)
end


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

function Helmholtz_ExpMsFEM_Solver(MsFEMparam::ExpHelmholtz_2d2ScaleUnifQuadMesh,
                                   PDE::Helmholtz2D, N_e::Int;
                                   ws::Union{ExpHelmholtz_Workspace, Nothing} = nothing,
                                   prefactor::Bool = true)
    if ws === nothing
        ws = ExpHelmholtz_Workspace(MsFEMparam, PDE)
    end
    if prefactor && isempty(ws.cell_factors)
        helm_prefactor_all!(ws)
    end
    helm_prefactor_edges!(ws, N_e)
    u_global, val_all, bub_all, _ =
        Helmholtz_ExpMsFEM_GlobalAssembly(ws, N_e)
    fine_sol = Helmholtz_ExpMsFEM_FineConstruct(ws, u_global, val_all,
                                                bub_all, N_e)
    return fine_sol, ws
end
