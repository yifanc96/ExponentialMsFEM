# Exponentially convergent Multiscale Finite Element Method (ExpMsFEM) on a
# 2D uniform rectangular grid. Reference:
#   Chen, Hou, Wang — "Exponentially Convergent Multiscale Finite Element
#   Method" (arXiv 2212.00823).
#
# Algorithm overview (per coarse cell (ci, cj)):
#   * 4 classical MsFEM nodal basis functions (Q1 fine-scale harmonic lifts
#     of piecewise-linear hat data on the perimeter).
#   * For each of the 4 coarse edges of the cell, (N_e + 1) edge basis
#     functions:
#       – N_e spectral modes obtained as the top-|λ| eigenvectors of the
#         generalised eigenproblem (R' N R) v = λ P v, where
#           N = energy-sum of discrete-harmonic extensions on the two cells
#               adjacent to the edge,
#           P = energy of discrete-harmonic lifts on an oversampled (3x2 or
#               2x2) patch around the edge,
#           R = restriction of patch-harmonic lifts to the shared edge
#               (minus the linear-endpoint part, interior nodes only).
#       – 1 "edge bubble" obtained by extending the patch bubble's edge trace
#         back into the cell via the same harmonic lift.
#   * 1 interior cell bubble added on top during fine-scale reconstruction.
#
# Implementation notes:
#   * Every cell and every edge patch is factorised once via `lu` and the
#     factors are cached on a `ExpMsFEM_Workspace` so that the harmext /
#     restrict / element_basis routines never refactorise.
#   * The generalised eigenproblem is computed densely once per edge and the
#     result (L1·R·V, L2·R·V, L1·bub, L2·bub) is cached — each interior edge
#     is visited twice (from both adjacent cells) but work is done once.
#   * `@threads` is used on the embarrassingly-parallel loops (prefactor over
#     cells / patches; per-cell basis assembly).

using SparseArrays
using LinearAlgebra
using Logging
import Base.Threads: @threads, nthreads


# Reference 4x4 Q1 stiffness (Laplacian part, scale-free on a square).
const _Q1_STIFF_REF = [
     2/3  -1/6  -1/3  -1/6
    -1/6   2/3  -1/6  -1/3
    -1/3  -1/6   2/3  -1/6
    -1/6  -1/3  -1/6   2/3
]

# Reference 4x4 Q1 mass without h² factor: entries {1/9, 1/18, 1/36}.
const _Q1_MASS_REF = [
    1/9   1/18  1/36  1/18
    1/18  1/9   1/18  1/36
    1/36  1/18  1/9   1/18
    1/18  1/36  1/18  1/9
]


# -----------------------------------------------------------------------------
# Mesh struct (mirrors MsFEM_2d2ScaleUnifQuadMesh in the existing MsFEM code)
# -----------------------------------------------------------------------------

struct ExpMsFEM_2d2ScaleUnifQuadMesh{Ti,Tf}
    Nce::Ti                            # coarse elements per dimension
    Nfe::Ti                            # fine elements per coarse element
    Ne::Ti                             # total fine elements per dimension
    CGrid_x::Vector{Tf}                # coarse grid nodes in x
    CGrid_y::Vector{Tf}                # coarse grid nodes in y
    ElemNode_loc2glo::Function         # local-to-global node indexing
    LocalBdyIndice::Vector{Ti}         # cell-perimeter node indices (1-idx)
    LocalBdyCondition::Matrix{Tf}      # 4xN_fbdy hat data for nodal basis
end

function ExpMsFEM_2d2ScaleUnifQuadMesh(Nce::Int, Nfe::Int)
    x = collect(LinRange(0, 1, Nce + 1))
    y = copy(x)

    function ElemNode_loc2glo(N, i, j, ind_node)
        if ind_node <= 2
            return (N + 1) * (j - 1) + i + ind_node - 1
        else
            return (N + 1) * j + i + 4 - ind_node
        end
    end

    LocalBdyIndice = reduce(vcat, collect.([
        1:Nfe+1,
        Nfe+2:Nfe+1:(Nfe+1)*(Nfe+1),
        2*(Nfe+1):Nfe+1:(Nfe+1)*(Nfe+1),
        Nfe*Nfe+Nfe+2:(Nfe+1)*(Nfe+1)-1,
    ]))

    LocalBdyCondition = reduce(vcat, collect.([
        LinRange(1, 0, Nfe+1), LinRange(1-1/Nfe, 0, Nfe),
        zeros(Nfe), zeros(Nfe-1),
        LinRange(0, 1, Nfe+1), zeros(Nfe),
        LinRange(1-1/Nfe, 0, Nfe), zeros(Nfe-1),
        zeros(Nfe+1), zeros(Nfe),
        LinRange(1/Nfe, 1, Nfe), LinRange(1/Nfe, 1-1/Nfe, Nfe-1),
        zeros(Nfe+1), LinRange(1/Nfe, 1, Nfe),
        zeros(Nfe), LinRange(1-1/Nfe, 1/Nfe, Nfe-1),
    ]))
    LocalBdyCondition = reshape(LocalBdyCondition, 4 * Nfe, 4)

    @info "[Mesh generation] ExpMsFEM mesh: $(Nce+1) coarse nodes per dim, $(Nce*Nfe+1) fine nodes per dim"
    return ExpMsFEM_2d2ScaleUnifQuadMesh(Nce, Nfe, Nce * Nfe, x, y,
                                         ElemNode_loc2glo,
                                         LocalBdyIndice, LocalBdyCondition)
end


# -----------------------------------------------------------------------------
# Cached LU factors for Dirichlet solves
# -----------------------------------------------------------------------------

# `InteriorFactor` carries the sparse system, the precomputed UMFPACK LU for
# the interior submatrix, and the interior-to-boundary coupling needed to
# handle non-zero Dirichlet data: A_ii u_i = F_i − A_ib u_b.
struct InteriorFactor{Tf}
    A::SparseMatrixCSC{Tf,Int}
    interior::Vector{Int}
    bdy::Vector{Int}
    A_ib::SparseMatrixCSC{Tf,Int}
    lu::Any  # SuiteSparse UmfpackLU factorisation of A[interior, interior]
end

function InteriorFactor(A::SparseMatrixCSC{Tf,Int}, bdy::Vector{Int}) where Tf
    n = size(A, 1)
    mask = trues(n)
    mask[bdy] .= false
    interior = findall(mask)
    A_ii = A[interior, interior]
    A_ib = A[interior, bdy]
    lu_fac = lu(A_ii)
    return InteriorFactor(A, interior, bdy, A_ib, lu_fac)
end

# Solve A u = F with u[bdy] = bdy_vals.  F, bdy_vals can be vector or matrix.
function interior_solve(fac::InteriorFactor{Tf}, F, bdy_vals) where Tf
    rhs = F[fac.interior, :] .- fac.A_ib * bdy_vals
    u_i = fac.lu \ rhs
    n = size(fac.A, 1)
    if ndims(F) == 1
        u = zeros(Tf, n)
        u[fac.interior] .= vec(u_i)
        u[fac.bdy] .= vec(bdy_vals)
    else
        k = size(F, 2)
        u = zeros(Tf, n, k)
        u[fac.interior, :] .= u_i
        u[fac.bdy, :] .= bdy_vals
    end
    return u
end


# -----------------------------------------------------------------------------
# Q1 assembly helpers
# -----------------------------------------------------------------------------

# Assemble the cell-local Q1 stiffness for the coefficient-weighted Laplacian
# on a rectangular fine grid with corners at (xs, ys). `xs, ys` are LinRanges
# of fine nodes (length Nx+1, Ny+1).
function _assemble_q1_stiffness(PDEparam, xs, ys)
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    nNodes = (Nx + 1) * (Ny + 1)

    Irow = zeros(Int, 16 * Nx * Ny)
    Jcol = zeros(Int, 16 * Nx * Ny)
    Aval = zeros(16 * Nx * Ny)

    # Local-to-global node mapping for this rectangular grid (Nx+1 nodes per row)
    function loc2glo(i, j, ind_node)
        if ind_node <= 2
            return (Nx + 1) * (j - 1) + i + ind_node - 1
        else
            return (Nx + 1) * j + i + 4 - ind_node
        end
    end

    for j = 1:Ny
        ylow = ys[j]; yhigh = ys[j+1]; ymid = 0.5 * (ylow + yhigh)
        for i = 1:Nx
            xlow = xs[i]; xhigh = xs[i+1]; xmid = 0.5 * (xlow + xhigh)
            a_center = PDEparam.a(xmid, ymid)
            for p = 1:4
                gp = loc2glo(i, j, p)
                for q = 1:4
                    gq = loc2glo(i, j, q)
                    idx = 16 * Ny * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
                    Irow[idx] = gp
                    Jcol[idx] = gq
                    Aval[idx] = a_center * _Q1_STIFF_REF[p, q]
                end
            end
        end
    end
    return sparse(Irow, Jcol, Aval, nNodes, nNodes)
end

# Boundary-node indices (1-indexed) on a rectangular grid of (Nx+1)x(Ny+1)
# nodes, in Matlab order: [bottom row, left-col interior, right-col interior,
# top-row interior].
function _rect_boundary_indices(Nx, Ny)
    return reduce(vcat, collect.([
        1:Nx+1,
        Nx+2:Nx+1:(Nx+1)*(Ny+1),
        2*(Nx+1):Nx+1:(Nx+1)*(Ny+1),
        Nx*Ny+Ny+2:(Nx+1)*(Ny+1)-1,
    ]))
end


# -----------------------------------------------------------------------------
# Workspace with per-cell / per-patch / per-edge caches
# -----------------------------------------------------------------------------

mutable struct ExpMsFEM_Workspace
    MsFEMparam::ExpMsFEM_2d2ScaleUnifQuadMesh
    PDEparam::Any
    # (ci, cj)   → InteriorFactor for cell stiffness
    cell_factors::Dict{Tuple{Int,Int}, InteriorFactor{Float64}}
    # (ci, cj, t) → (InteriorFactor, Nx, Ny, m0, n0)
    patch_factors::Dict{Tuple{Int,Int,Int},
                        Tuple{InteriorFactor{Float64}, Int, Int, Int, Int}}
    # (ci, cj, t) → (L1·R·V, L2·R·V, L1·bub, L2·bub)
    edge_cache::Dict{Tuple{Int,Int,Int},
                     Tuple{Matrix{Float64}, Matrix{Float64},
                           Vector{Float64}, Vector{Float64}}}
end

function ExpMsFEM_Workspace(MsFEMparam, PDEparam)
    return ExpMsFEM_Workspace(MsFEMparam, PDEparam,
                              Dict{Tuple{Int,Int}, InteriorFactor{Float64}}(),
                              Dict{Tuple{Int,Int,Int},
                                   Tuple{InteriorFactor{Float64}, Int, Int, Int, Int}}(),
                              Dict{Tuple{Int,Int,Int},
                                   Tuple{Matrix{Float64}, Matrix{Float64},
                                         Vector{Float64}, Vector{Float64}}}())
end


# -----------------------------------------------------------------------------
# basefun — cell-local Q1 stiffness
# -----------------------------------------------------------------------------

function basefun(ws::ExpMsFEM_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    xlow = mp.CGrid_x[ci]; xhigh = mp.CGrid_x[ci + 1]
    ylow = mp.CGrid_y[cj]; yhigh = mp.CGrid_y[cj + 1]
    xs = LinRange(xlow, xhigh, mp.Nfe + 1)
    ys = LinRange(ylow, yhigh, mp.Nfe + 1)
    return _assemble_q1_stiffness(ws.PDEparam, xs, ys)
end


# -----------------------------------------------------------------------------
# basefun1 — oversampled patch Q1 stiffness
# -----------------------------------------------------------------------------

# Returns the oversampled-patch stiffness A, its fine-grid dimensions
# (Nx_fine, Ny_fine), and the lower-left coarse-cell index (m0, n0) of the
# patch. For a horizontal edge (t=1) between cells (ci, cj) and (ci, cj+1),
# the patch spans 3 coarse cells in x (or 2 at the boundary) and 2 in y.
# For a vertical edge (t=2), vice versa.
function basefun1(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int)
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
    A = _assemble_q1_stiffness(ws.PDEparam, xs, ys)
    return A, Nx_cells * Nfe, Ny_cells * Nfe, m0, n0
end


# -----------------------------------------------------------------------------
# Workspace prefactor helpers
# -----------------------------------------------------------------------------

function build_cell_factor!(ws::ExpMsFEM_Workspace, ci::Int, cj::Int)
    A = basefun(ws, ci, cj)
    bdy = ws.MsFEMparam.LocalBdyIndice
    fac = InteriorFactor(A, bdy)
    ws.cell_factors[(ci, cj)] = fac
    return fac
end

function build_patch_factor!(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int)
    A, Nx, Ny, m0, n0 = basefun1(ws, ci, cj, t)
    bdy = _rect_boundary_indices(Nx, Ny)
    fac = InteriorFactor(A, bdy)
    ws.patch_factors[(ci, cj, t)] = (fac, Nx, Ny, m0, n0)
    return fac
end

# Prefactor every cell and every interior edge patch. Serial loops; the LU
# inside is thread-safe per call but dict writes must be serial.
function prefactor_all!(ws::ExpMsFEM_Workspace)
    Nce = ws.MsFEMparam.Nce
    println("[ExpMsFEM] prefactoring $(Nce^2) cells and $(2 * Nce * (Nce - 1)) patches using $(nthreads()) threads")

    # --- cells ---
    cell_keys = [(ci, cj) for cj = 1:Nce for ci = 1:Nce]
    cell_results = Vector{InteriorFactor{Float64}}(undef, length(cell_keys))
    @threads for k = 1:length(cell_keys)
        ci, cj = cell_keys[k]
        A = basefun(ws, ci, cj)
        bdy = ws.MsFEMparam.LocalBdyIndice
        cell_results[k] = InteriorFactor(A, bdy)
    end
    for (k, key) in enumerate(cell_keys)
        ws.cell_factors[key] = cell_results[k]
    end

    # --- horizontal patches (t=1): cj ∈ 1..Nce-1, ci ∈ 1..Nce ---
    # --- vertical patches (t=2):   ci ∈ 1..Nce-1, cj ∈ 1..Nce ---
    patch_keys = vcat(
        [(ci, cj, 1) for cj = 1:(Nce-1) for ci = 1:Nce],
        [(ci, cj, 2) for cj = 1:Nce for ci = 1:(Nce-1)],
    )
    patch_results = Vector{Tuple{InteriorFactor{Float64}, Int, Int, Int, Int}}(undef,
                                                                     length(patch_keys))
    @threads for k = 1:length(patch_keys)
        ci, cj, t = patch_keys[k]
        A, Nx, Ny, m0, n0 = basefun1(ws, ci, cj, t)
        bdy = _rect_boundary_indices(Nx, Ny)
        patch_results[k] = (InteriorFactor(A, bdy), Nx, Ny, m0, n0)
    end
    for (k, key) in enumerate(patch_keys)
        ws.patch_factors[key] = patch_results[k]
    end

    @info "[ExpMsFEM] prefactor complete: $(length(ws.cell_factors)) cells, $(length(ws.patch_factors)) patches"
    return nothing
end


# -----------------------------------------------------------------------------
# Cell bubble: Q1 Dirichlet solve with RHS from PDEparam.rhs
# -----------------------------------------------------------------------------

# Solve −∇·(a∇u) = f on the coarse cell with u = 0 on its perimeter, using the
# fine Q1 projected RHS via the cell mass matrix. Returns a length-(Nfe+1)^2
# vector.
function bubble(ws::ExpMsFEM_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    Nfe = mp.Nfe
    xlow = mp.CGrid_x[ci]; xhigh = mp.CGrid_x[ci + 1]
    ylow = mp.CGrid_y[cj]; yhigh = mp.CGrid_y[cj + 1]
    xs = collect(LinRange(xlow, xhigh, Nfe + 1))
    ys = collect(LinRange(ylow, yhigh, Nfe + 1))

    # Fine RHS vector: ∫ f φ_i ≈ M · f_nodal where f_nodal is f evaluated at
    # fine nodes (exact Q1 quadrature).
    Ne_total = mp.Ne
    h = 1.0 / Ne_total
    f_nodal = [ws.PDEparam.rhs(xs[i], ys[j]) for j in 1:(Nfe+1) for i in 1:(Nfe+1)]
    # Cell mass: M_ij = h² · _Q1_MASS_REF (scale-free identity per cell × h²)
    # Easier: build M sparse and multiply. But the scalar h² applies uniformly,
    # so use the reference mass and assemble with scatter.
    M = spzeros(length(f_nodal), length(f_nodal))
    # Assemble cell mass
    function loc2glo_fine(i, j, ind_node)
        if ind_node <= 2
            return (Nfe + 1) * (j - 1) + i + ind_node - 1
        else
            return (Nfe + 1) * j + i + 4 - ind_node
        end
    end
    Ir = zeros(Int, 16 * Nfe^2); Jc = zeros(Int, 16 * Nfe^2); Vv = zeros(16 * Nfe^2)
    for j = 1:Nfe, i = 1:Nfe
        for p = 1:4, q = 1:4
            idx = 16 * Nfe * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
            Ir[idx] = loc2glo_fine(i, j, p)
            Jc[idx] = loc2glo_fine(i, j, q)
            Vv[idx] = h^2 * _Q1_MASS_REF[p, q]
        end
    end
    M = sparse(Ir, Jc, Vv, (Nfe + 1)^2, (Nfe + 1)^2)
    F = M * f_nodal

    fac = get(ws.cell_factors, (ci, cj), nothing)
    if fac === nothing
        fac = build_cell_factor!(ws, ci, cj)
    end
    bdy_vals = zeros(length(fac.bdy))
    u = interior_solve(fac, F, bdy_vals)
    return u
end


# -----------------------------------------------------------------------------
# harmext — two-cell harmonic extension of identity edge-hat data
# -----------------------------------------------------------------------------

# For t=1: edge between cells (ci, cj) and (ci, cj+1); L1 lives on (ci, cj) with
# Dirichlet data on its TOP edge interior, L2 on (ci, cj+1) with data on its
# BOTTOM edge interior.
# For t=2: edge between cells (ci, cj) and (ci+1, cj); L1 on (ci, cj) RIGHT,
# L2 on (ci+1, cj) LEFT.
function harmext(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int)
    mp = ws.MsFEMparam
    Nfe = mp.Nfe

    # Acquire two cell factors
    fac1 = get(ws.cell_factors, (ci, cj), nothing)
    if fac1 === nothing
        fac1 = build_cell_factor!(ws, ci, cj)
    end
    if t == 1
        other = (ci, cj + 1)
    else
        other = (ci + 1, cj)
    end
    fac2 = get(ws.cell_factors, other, nothing)
    if fac2 === nothing
        fac2 = build_cell_factor!(ws, other...)
    end

    # Build Dirichlet data matrices f1, f2 of shape (4*Nfe, Nfe-1): identity
    # placed on the shared-edge interior rows of `b`.
    # b ordering: [bottom (Nfe+1), left_int (Nfe), right_int (Nfe), top_int (Nfe-1)]
    f1 = zeros(4 * Nfe, Nfe - 1)
    f2 = zeros(4 * Nfe, Nfe - 1)
    if t == 1
        # L1 top-int: rows (3*Nfe + 2) : (4*Nfe)  (1-indexed)
        for k = 1:(Nfe - 1)
            f1[3 * Nfe + 1 + k, k] = 1.0
        end
        # L2 bottom-int: rows 2 : Nfe
        for k = 1:(Nfe - 1)
            f2[1 + k, k] = 1.0
        end
    else
        # L1 right-col interior: rows (2*Nfe + 2) : (3*Nfe)
        for k = 1:(Nfe - 1)
            f1[2 * Nfe + 1 + k, k] = 1.0
        end
        # L2 left-col interior: rows (Nfe + 2) : (2*Nfe)
        for k = 1:(Nfe - 1)
            f2[Nfe + 1 + k, k] = 1.0
        end
    end

    n_nodes = (Nfe + 1)^2
    F_zero = zeros(n_nodes, Nfe - 1)
    L1 = interior_solve(fac1, F_zero, f1)
    L2 = interior_solve(fac2, F_zero, f2)

    # N = L1' · K1 · L1 + L2' · K2 · L2 — the shared-edge energy Gram matrix
    N = L1' * (fac1.A * L1) + L2' * (fac2.A * L2)
    return L1, L2, N
end


# -----------------------------------------------------------------------------
# restrict — oversampled-patch edge-restriction
# -----------------------------------------------------------------------------

# Returns
#   R   : (Nfe-1, n_active)  patch harmonic lift restricted to the shared edge
#         (with linear-endpoint part subtracted; two endpoint nodes dropped)
#   P   : (n_active, n_active) energy Gram on the 3x2 (or 2x2) patch
#   bub : (Nfe-1,) patch bubble trace on the shared edge
# where n_active = perimeter_of_patch − (1 + count_of_boundary_sides_stripped).
# Matches Matlab elliptic/periodic/Exp/restrict.m exactly.
function restrict(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe

    # Fetch patch factor
    pfac = get(ws.patch_factors, (ci, cj, t), nothing)
    if pfac === nothing
        build_patch_factor!(ws, ci, cj, t)
        pfac = ws.patch_factors[(ci, cj, t)]
    end
    fac, Nx, Ny, m0, n0 = pfac

    # Which perimeter DOFs are active Dirichlet sources (per Matlab
    # restrict.m's ladder; translated to 1-indexed). The generic interior
    # branch activates all but the last perimeter DOF. Patches touching ∂Ω
    # additionally strip the perimeter rows that lie on ∂Ω.
    P_size = 2 * (Nx + Ny)
    mask = trues(P_size)

    # Matlab 1-indexed M = ci, Nn = cj
    M = ci
    Nn = cj

    function zero_rng!(a, b)
        mask[a:b] .= false
    end

    if t == 1
        if Nn == 1 && 2 < M < Nce - 1
            zero_rng!(1, Nx + 1)
        elseif Nn == Nce - 1 && 2 < M < Nce - 1
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        elseif M < 3 && 1 < Nn < Nce - 1
            zero_rng!(Nx + 2, Nx + Ny + 1)
        elseif M > Nce - 2 && 1 < Nn < Nce - 1
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
        elseif Nn == 1 && M < 3
            zero_rng!(1, Nx + 1)
            zero_rng!(Nx + 2, Nx + Ny + 1)
        elseif Nn == 1 && M > Nce - 2
            zero_rng!(1, Nx + 1)
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
        elseif Nn == Nce - 1 && M < 3
            zero_rng!(Nx + 2, Nx + Ny + 1)
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        elseif Nn == Nce - 1 && M > Nce - 2
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        end
    else
        if M == 1 && 2 < Nn < Nce - 1
            zero_rng!(Nx + 2, Nx + Ny + 1)
        elseif M == Nce - 1 && 2 < Nn < Nce - 1
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
        elseif Nn < 3 && 1 < M < Nce - 1
            zero_rng!(1, Nx + 1)
        elseif Nn > Nce - 2 && 1 < M < Nce - 1
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        elseif M == 1 && Nn < 3
            zero_rng!(Nx + 2, Nx + Ny + 1)
            zero_rng!(1, Nx + 1)
        elseif M == Nce - 1 && Nn < 3
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
            zero_rng!(1, Nx + 1)
        elseif M == 1 && Nn > Nce - 2
            zero_rng!(Nx + 2, Nx + Ny + 1)
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        elseif M == Nce - 1 && Nn > Nce - 2
            zero_rng!(Nx + Ny + 2, Nx + 2 * Ny + 1)
            zero_rng!(Nx + 2 * Ny + 2, 2 * Nx + 2 * Ny)
        end
    end
    mask[P_size] = false  # always drop the very last perimeter DOF

    active_idx = findall(mask)
    n_active = length(active_idx)

    # Boundary data matrix of shape (|b_patch|, n_active): identity on active rows
    bdy_vals = zeros(length(fac.bdy), n_active)
    for k = 1:n_active
        bdy_vals[active_idx[k], k] = 1.0
    end

    n_patch = (Nx + 1) * (Ny + 1)
    F_zero = zeros(n_patch, n_active)
    harm = interior_solve(fac, F_zero, bdy_vals)

    # Patch bubble: RHS from PDEparam.rhs projected via cell mass matrix
    mp = ws.MsFEMparam
    h = 1.0 / mp.Ne
    # Build fine-scale xs, ys for the patch
    xs = collect(LinRange(mp.CGrid_x[m0], mp.CGrid_x[m0 + div(Nx, Nfe)], Nx + 1))
    ys = collect(LinRange(mp.CGrid_y[n0], mp.CGrid_y[n0 + div(Ny, Nfe)], Ny + 1))
    f_nodal = [ws.PDEparam.rhs(xs[i], ys[j]) for j in 1:(Ny+1) for i in 1:(Nx+1)]
    # Patch mass
    Ir = zeros(Int, 16 * Nx * Ny); Jc = zeros(Int, 16 * Nx * Ny); Vv = zeros(16 * Nx * Ny)
    function loc2glo_patch(i, j, ind_node)
        if ind_node <= 2
            return (Nx + 1) * (j - 1) + i + ind_node - 1
        else
            return (Nx + 1) * j + i + 4 - ind_node
        end
    end
    for j = 1:Ny, i = 1:Nx
        for p = 1:4, q = 1:4
            idx = 16 * Ny * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
            Ir[idx] = loc2glo_patch(i, j, p)
            Jc[idx] = loc2glo_patch(i, j, q)
            Vv[idx] = h^2 * _Q1_MASS_REF[p, q]
        end
    end
    M_patch = sparse(Ir, Jc, Vv, n_patch, n_patch)
    G = M_patch * f_nodal
    bub = interior_solve(fac, G, zeros(length(fac.bdy)))

    # P = harm' · A · harm
    P = harm' * (fac.A * harm)

    # Edge trace: pick the row/col of the patch containing the shared edge
    leng = Nfe + 1
    if t == 1
        if ci == 1
            col_start, col_end = 1, Nfe + 1
        else
            col_start, col_end = Nfe + 1, 2 * Nfe + 1
        end
        row_idx = Nfe + 1
        edge_nodes = [(row_idx - 1) * (Nx + 1) + k for k in col_start:col_end]
    else
        if cj == 1
            row_start, row_end = 1, Nfe + 1
        else
            row_start, row_end = Nfe + 1, 2 * Nfe + 1
        end
        col_idx = Nfe + 1
        edge_nodes = [(k - 1) * (Nx + 1) + col_idx for k in row_start:row_end]
    end

    R_full = harm[edge_nodes, :]
    bub_edge = bub[edge_nodes]

    # Subtract linear endpoint part, drop endpoints
    lin_left = collect(LinRange(1, 0, leng))
    lin_right = collect(LinRange(0, 1, leng))
    R_full = R_full .- lin_left * R_full[1:1, :] .- lin_right * R_full[leng:leng, :]
    bub_edge = bub_edge .- lin_left * bub_edge[1] .- lin_right * bub_edge[leng]
    R = R_full[2:(leng - 1), :]
    bub_edge = bub_edge[2:(leng - 1)]

    return R, P, bub_edge
end


# -----------------------------------------------------------------------------
# Element basis (4 nodal + per-edge eigen + edge bubble)
# -----------------------------------------------------------------------------

# Top-N_e generalised eigenvectors of (R'NR, P). Uses dense Hermitian eigh and
# sorts by |λ| descending (to match Matlab's `eigs(..., N_e)` default 'LM').
function _eigen_modes(R::Matrix{Float64}, N::Matrix{Float64},
                      P::Matrix{Float64}, N_e::Int)
    M = R' * N * R
    M = 0.5 * (M + M')
    Preg = P + 1e-14 * I
    Preg = 0.5 * (Preg + Preg')
    vals, vecs = eigen(Symmetric(M), Symmetric(Preg))
    order = sortperm(abs.(vals), rev = true)
    return vecs[:, order[1:N_e]]
end


function _nodal_basis(ws::ExpMsFEM_Workspace, ci::Int, cj::Int)
    mp = ws.MsFEMparam
    fac = ws.cell_factors[(ci, cj)]
    n_nodes = (mp.Nfe + 1)^2
    F_zero = zeros(n_nodes, 4)
    return interior_solve(fac, F_zero, mp.LocalBdyCondition)
end


# Build per-edge data (L1·R·V, L2·R·V, L1·bub, L2·bub). Pure function of
# inputs — used inside `prefactor_edges!` for parallel pre-population.
function _build_edge_data(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int, N_e::Int)
    L1, L2, N = harmext(ws, ci, cj, t)
    R, P, bub = restrict(ws, ci, cj, t)
    V = _eigen_modes(R, N, P, N_e)
    RV = R * V
    return (L1 * RV, L2 * RV, vec(L1 * bub), vec(L2 * bub))
end


# Look up pre-populated per-edge data. Raises if not present.
function _edge_data(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, t::Int, N_e::Int)
    cached = get(ws.edge_cache, (ci, cj, t), nothing)
    if cached === nothing
        # Lazy build (serial path; `prefactor_edges!` should be called ahead
        # of the threaded per-cell loop).
        cached = _build_edge_data(ws, ci, cj, t, N_e)
        ws.edge_cache[(ci, cj, t)] = cached
    end
    return cached
end


# Parallel pre-population of the edge cache. Each interior edge is computed
# exactly once (twice in Matlab — once per adjacent cell) and the result
# becomes read-only during the per-cell assembly loop.
function prefactor_edges!(ws::ExpMsFEM_Workspace, N_e::Int)
    empty!(ws.edge_cache)
    Nce = ws.MsFEMparam.Nce
    keys = vcat(
        [(ci, cj, 1) for cj = 1:(Nce - 1) for ci = 1:Nce],
        [(ci, cj, 2) for cj = 1:Nce for ci = 1:(Nce - 1)],
    )
    results = Vector{Tuple{Matrix{Float64}, Matrix{Float64},
                           Vector{Float64}, Vector{Float64}}}(undef, length(keys))
    @threads for k = 1:length(keys)
        ci, cj, t = keys[k]
        results[k] = _build_edge_data(ws, ci, cj, t, N_e)
    end
    for (k, key) in enumerate(keys)
        ws.edge_cache[key] = results[k]
    end
    @info "[ExpMsFEM] prefactored $(length(keys)) interior edges"
    return nothing
end


# For cell (ci, cj), build the full fine-scale basis matrix `value` of shape
# ((Nfe+1)², 4 + 4*(N_e+1)), the element stiffness K = value' B value, and
# element load f.
function ExpMsFEM_ElementBasis(ws::ExpMsFEM_Workspace, ci::Int, cj::Int, N_e::Int)
    mp = ws.MsFEMparam
    Nfe = mp.Nfe
    Nce = mp.Nce
    count = 4 + 4 * (N_e + 1)
    n_nodes = (Nfe + 1)^2
    value = zeros(n_nodes, count)

    # Cell stiffness B
    fac = ws.cell_factors[(ci, cj)]
    B = fac.A

    # Nodal basis (4)
    value[:, 1:4] = _nodal_basis(ws, ci, cj)

    # Per-edge blocks (N_e eigenmodes + 1 edge bubble)
    block = N_e + 1

    # Bottom edge: shared with (ci, cj-1). Lower cell of pair = (ci, cj-1),
    # current = upper, so take L2 side.
    off = 4
    if cj > 1
        L1_RV, L2_RV, L1_bub, L2_bub = _edge_data(ws, ci, cj - 1, 1, N_e)
        value[:, (off + 1):(off + N_e)] = L2_RV
        value[:, off + N_e + 1] = L2_bub
    end
    off += block

    # Top edge: shared with (ci, cj+1). Lower cell = (ci, cj) = current, so L1.
    if cj < Nce
        L1_RV, L2_RV, L1_bub, L2_bub = _edge_data(ws, ci, cj, 1, N_e)
        value[:, (off + 1):(off + N_e)] = L1_RV
        value[:, off + N_e + 1] = L1_bub
    end
    off += block

    # Left edge: shared with (ci-1, cj). Lower cell = (ci-1, cj), current = upper, L2.
    if ci > 1
        L1_RV, L2_RV, L1_bub, L2_bub = _edge_data(ws, ci - 1, cj, 2, N_e)
        value[:, (off + 1):(off + N_e)] = L2_RV
        value[:, off + N_e + 1] = L2_bub
    end
    off += block

    # Right edge: shared with (ci+1, cj). Lower cell = (ci, cj) = current, L1.
    if ci < Nce
        L1_RV, L2_RV, L1_bub, L2_bub = _edge_data(ws, ci, cj, 2, N_e)
        value[:, (off + 1):(off + N_e)] = L1_RV
        value[:, off + N_e + 1] = L1_bub
    end

    # Element stiffness K = value' B value
    K_elem = value' * (B * value)

    # Element load: f_elem[p] = ∫ rhs * value[:, p] dx. Use cell mass to form
    # ∫ rhs * φ_i per fine node then project onto basis.
    xlow = mp.CGrid_x[ci]; xhigh = mp.CGrid_x[ci + 1]
    ylow = mp.CGrid_y[cj]; yhigh = mp.CGrid_y[cj + 1]
    xs = collect(LinRange(xlow, xhigh, Nfe + 1))
    ys = collect(LinRange(ylow, yhigh, Nfe + 1))
    f_nodal = [ws.PDEparam.rhs(xs[i], ys[j]) for j in 1:(Nfe + 1) for i in 1:(Nfe + 1)]
    h = 1.0 / mp.Ne
    # Assemble cell mass M_cell (same pattern as bubble)
    Ir = zeros(Int, 16 * Nfe^2); Jc = zeros(Int, 16 * Nfe^2); Vv = zeros(16 * Nfe^2)
    function loc2glo_fine(i, j, ind_node)
        if ind_node <= 2
            return (Nfe + 1) * (j - 1) + i + ind_node - 1
        else
            return (Nfe + 1) * j + i + 4 - ind_node
        end
    end
    for j = 1:Nfe, i = 1:Nfe
        for p = 1:4, q = 1:4
            idx = 16 * Nfe * (i - 1) + 16 * (j - 1) + 4 * (p - 1) + q
            Ir[idx] = loc2glo_fine(i, j, p)
            Jc[idx] = loc2glo_fine(i, j, q)
            Vv[idx] = h^2 * _Q1_MASS_REF[p, q]
        end
    end
    M_cell = sparse(Ir, Jc, Vv, n_nodes, n_nodes)
    F_node = M_cell * f_nodal
    f_elem = value' * F_node

    return value, B, K_elem, f_elem
end


# -----------------------------------------------------------------------------
# Global assembly, solve, and fine reconstruction
# -----------------------------------------------------------------------------

# Global DOF index for slot `p` (1..count) in cell (ci, cj).
# count = 4 + 4*N_p where N_p = N_e + 1.
function _global_dof(Nce, ci, cj, p, N_e, loc2glo)
    N_p = N_e + 1
    if p <= 4
        return loc2glo(Nce, ci, cj, p)
    elseif p <= 4 + N_p
        return (ci - 1 + (cj - 1) * Nce) * N_p + (Nce + 1)^2 + p - 4
    elseif p <= 4 + 2 * N_p
        return (ci - 1 + cj * Nce) * N_p + (Nce + 1)^2 + p - 4 - N_p
    elseif p <= 4 + 3 * N_p
        return (ci - 1 + (cj - 1) * (Nce + 1) + Nce * (Nce + 1)) * N_p + (Nce + 1)^2 + p - 4 - 2 * N_p
    else
        return (ci + (cj - 1) * (Nce + 1) + Nce * (Nce + 1)) * N_p + (Nce + 1)^2 + p - 4 - 3 * N_p
    end
end


# Assemble global stiffness A, load F, and store per-cell basis values for
# reconstruction.
function ExpMsFEM_GlobalAssembly(ws::ExpMsFEM_Workspace, N_e::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe
    N_p = N_e + 1
    count = 4 + 4 * N_p
    nNodes = (Nce + 1)^2 + 2 * Nce * (Nce + 1) * N_p

    val_all = zeros((Nfe + 1)^2, count, Nce, Nce)
    K_all = zeros(count, count, Nce, Nce)
    f_all = zeros(count, Nce, Nce)
    bub_all = zeros((Nfe + 1)^2, Nce, Nce)

    println("[ExpMsFEM] assembling per-cell bases using $(nthreads()) threads")
    cell_keys = [(ci, cj) for cj = 1:Nce for ci = 1:Nce]
    @threads for k = 1:length(cell_keys)
        ci, cj = cell_keys[k]
        value, _, K_elem, f_elem = ExpMsFEM_ElementBasis(ws, ci, cj, N_e)
        val_all[:, :, ci, cj] = value
        K_all[:, :, ci, cj] = K_elem
        f_all[:, ci, cj] = f_elem
        bub_all[:, ci, cj] = bubble(ws, ci, cj)
    end
    @info "[ExpMsFEM] bases assembled"

    # Global sparse assembly via COO
    nnz = count^2 * Nce^2
    Ir = zeros(Int, nnz); Jc = zeros(Int, nnz); Vv = zeros(nnz)
    F = zeros(nNodes)
    for cj = 1:Nce, ci = 1:Nce
        gidx = [_global_dof(Nce, ci, cj, p, N_e, mp.ElemNode_loc2glo) for p in 1:count]
        for p = 1:count
            for q = 1:count
                idx = count^2 * Nce * (ci - 1) + count^2 * (cj - 1) + count * (p - 1) + q
                Ir[idx] = gidx[p]; Jc[idx] = gidx[q]
                Vv[idx] = K_all[p, q, ci, cj]
            end
            F[gidx[p]] += f_all[p, ci, cj]
        end
    end
    A = sparse(Ir, Jc, Vv, nNodes, nNodes)
    @info "[ExpMsFEM] global assembly complete"

    # Domain-boundary DOFs: corner nodes on ∂Ω + all horizontal edges at
    # k=0 or k=Nce + all vertical edges at m=0 or m=Nce.
    corner_bdy = reduce(vcat, collect.([
        1:Nce+1,
        Nce+2:Nce+1:(Nce+1)*(Nce+1),
        2*(Nce+1):Nce+1:(Nce+1)*(Nce+1),
        Nce*Nce+Nce+2:(Nce+1)*(Nce+1)-1,
    ]))
    base_h = (Nce + 1)^2
    h_bot = base_h .+ (1:(N_p * Nce))                                   # k = 0
    h_top = base_h + Nce^2 * N_p .+ (1:(N_p * Nce))                      # k = Nce
    edge_h_bdy = vcat(h_bot, h_top)

    base_v = base_h + 2 * Nce * (Nce + 1) * N_p - Nce * (Nce + 1) * N_p
    # Vertical DOFs live at offset (Nce+1)^2 + Nce*(Nce+1)*N_p .. +(Nce+1)*Nce*N_p
    base_v = (Nce + 1)^2 + Nce * (Nce + 1) * N_p
    edge_v_bdy = Int[]
    for slot = 1:N_p
        # m = 0 column (x-col index 1 in Matlab 1-idx): offset = (0 + n*(Nc+1))*N_p + slot
        for n = 0:(Nce - 1)
            push!(edge_v_bdy, base_v + (0 + n * (Nce + 1)) * N_p + slot)
        end
        # m = Nce column
        for n = 0:(Nce - 1)
            push!(edge_v_bdy, base_v + (Nce + n * (Nce + 1)) * N_p + slot)
        end
    end

    bdy = sort!(unique!(vcat(corner_bdy, edge_h_bdy, edge_v_bdy)))

    # Dirichlet-0 via direct row/col elimination
    A[bdy, :] .= 0
    A[:, bdy] .= 0
    F[bdy] .= 0
    A[bdy, bdy] .= sparse(LinearAlgebra.I, length(bdy), length(bdy))

    u_global = A \ F
    @info "[ExpMsFEM] global linear system solved"

    return u_global, val_all, bub_all, N_e
end


# Reconstruct the fine-scale solution from global coefficients and per-cell
# basis matrices; add the per-cell bubble on top.
function ExpMsFEM_FineConstruct(ws::ExpMsFEM_Workspace, u_global::Vector{Float64},
                                val_all::Array{Float64, 4},
                                bub_all::Array{Float64, 3}, N_e::Int)
    mp = ws.MsFEMparam
    Nce = mp.Nce
    Nfe = mp.Nfe
    count = 4 + 4 * (N_e + 1)
    fine = zeros(Nce * Nfe + 1, Nce * Nfe + 1)

    @threads for k = 1:(Nce * Nce)
        cj = div(k - 1, Nce) + 1
        ci = mod(k - 1, Nce) + 1
        zs = zeros((Nfe + 1)^2)
        for p = 1:count
            gp = _global_dof(Nce, ci, cj, p, N_e, mp.ElemNode_loc2glo)
            zs .+= u_global[gp] .* val_all[:, p, ci, cj]
        end
        zs .+= bub_all[:, ci, cj]
        fine[(ci - 1) * Nfe + 1:ci * Nfe + 1,
             (cj - 1) * Nfe + 1:cj * Nfe + 1] = reshape(zs, Nfe + 1, Nfe + 1)
    end

    @info "[ExpMsFEM] fine-scale solution reconstructed"
    return reshape(fine, (Nce * Nfe + 1)^2)
end


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

function ExpMsFEM_Solver(MsFEMparam::ExpMsFEM_2d2ScaleUnifQuadMesh,
                         PDEparam, N_e::Int;
                         ws::Union{ExpMsFEM_Workspace, Nothing} = nothing,
                         prefactor::Bool = true)
    if ws === nothing
        ws = ExpMsFEM_Workspace(MsFEMparam, PDEparam)
    end
    if prefactor && isempty(ws.cell_factors)
        prefactor_all!(ws)
    end
    # Edge cache depends on N_e (number of eigenmodes kept). Rebuild it.
    prefactor_edges!(ws, N_e)
    u_global, val_all, bub_all, _ = ExpMsFEM_GlobalAssembly(ws, N_e)
    fine_sol = ExpMsFEM_FineConstruct(ws, u_global, val_all, bub_all, N_e)
    return fine_sol, ws
end
