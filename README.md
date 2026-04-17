# ExponentialMsFEM

Exponentially convergent multiscale finite element methods in Julia.

## Status

- FEM and MsFEM on `Elliptic/`: finished.
- ExpMsFEM on `Elliptic/` (elliptic elliptic variable-coefficient problem): finished on this `julia-code` branch.
- ExpMsFEM on `Helmholtz/` (2D impedance Helmholtz problem): finished on this `julia-code` branch.

## Layout

### `Elliptic/`

- `PDEstruct.jl` — `VarElliptic` PDE data container (`a`, `rhs`, boundary handlers).
- `FEMutility.jl` / `FEMsolver.jl` — fine-scale Q1 reference solver on a uniform rectangular grid.
- `MsFEMutility.jl` / `MsFEMsolver.jl` — classical multiscale FEM with harmonic-extension basis + cell bubble.
- `ExponentialMsFEMutility.jl` — edge-enriched exponentially convergent multiscale method:
  - `basefun` / `basefun1` — Q1 cell and oversampled-patch stiffness.
  - `harmext` — two-cell harmonic lift of identity edge-hat data.
  - `restrict` — patch → edge restriction `(R, P, bub)` with linear-endpoint subtraction; handles the full 8-branch boundary-patch ladder.
  - `bubble` — per-cell Dirichlet bubble for the interior load.
  - `ExpMsFEM_Workspace` — cache for per-cell UMFPACK LU factors, per-edge patch factors, and per-edge `(L1·R·V, L2·R·V, L1·bub, L2·bub)`.
  - `prefactor_all!` / `prefactor_edges!` — parallel offline phases (`@threads`), dict writes serialised.
  - `ExpMsFEM_ElementBasis` / `ExpMsFEM_GlobalAssembly` / `ExpMsFEM_FineConstruct` / `ExpMsFEM_Solver` — full pipeline.
- `ExponentialMsFEMsolver.jl` — driver reproducing the Matlab `main.m` workflow on the 5-scale periodic coefficient with a manufactured solution `u = sin(2πx)·sin(πy)·exp(x + 2y)`.

### `Helmholtz/`

- `PDEstruct.jl` — `Helmholtz2D` PDE container (`k0`, `afun`, `vfun`, `betafun`, `ffun`, `gfun`) for `-∇·(a∇u) - k₀²v²u = f` on `(0,1)²` with impedance BC `∂u/∂n + ik₀βu = g`.
- `ExponentialMsFEMutility.jl` — complex-valued port of the elliptic utilities:
  - Per-fine-cell complex Q1 stiffness with position-dependent impedance term; real norm matrix `B_ref` for the energy error.
  - `helm_dirichlet_indices` (corresponds to Matlab `bc.m`) — excludes ∂Ω sides from local Dirichlet sets so the Robin BC stays in the system.
  - `ComplexInteriorFactor` carries both the LU and the transpose-LU since Matlab solves `A.' \ F` (non-conjugate transpose).
  - `helm_basefun` / `helm_basefun1` / `helm_bubble` / `helm_harmext` / `helm_restrict` — complex analogues of the elliptic routines.
  - `helm_ElementBasis` uses analytical Q1 hat values to build Dirichlet data for the nodal basis (replaces Matlab's 9-branch corner/edge/interior case split with one formula).
  - Sesquilinear forms: `K = value.' * B * conj(value)`, `f = value' * F_fine`, global `u = A.' \ F`.
- `ExponentialMsFEMsolver.jl` — driver reproducing Matlab `helmholtz/case1/Exp/main.m` at `N_c = N_f = 8`, `k₀ = 2` with a per-side plane-wave impedance load.

## Usage

```bash
julia -t 4 Elliptic/ExponentialMsFEMsolver.jl
julia -t 4 Helmholtz/ExponentialMsFEMsolver.jl
```

Controlled by environment variables (defaults in parentheses):

- `EXPMSFEM_NCE`     — coarse cells per dimension (8)
- `EXPMSFEM_NFE`     — fine cells per coarse cell (8)
- `EXPMSFEM_NE_MAX`  — max number of edge eigenmodes per edge (5 elliptic / 6 Helmholtz)
- `EXPMSFEM_K0`      — wavenumber, Helmholtz only (2.0)

Example:

```bash
EXPMSFEM_NCE=8 EXPMSFEM_NFE=16 EXPMSFEM_NE_MAX=5 julia -t 4 Elliptic/ExponentialMsFEMsolver.jl
```

Dependencies (`Project.toml`): `ForwardDiff` (elliptic driver only). Use `-t N` for `N`-way threaded offline.

## Validation

**Elliptic** at `Nce=8`, `Nfe=16`, exponential H¹ decay in the number of edge modes per edge:

| N_e | relative L²  | relative H¹ |
|-----|--------------|-------------|
| 1   | 4.26e-3      | 3.00e-2     |
| 2   | 1.39e-3      | 1.34e-2     |
| 3   | 4.52e-4      | 5.18e-3     |
| 4   | 7.37e-5      | 1.14e-3     |
| 5   | 2.24e-5      | 3.74e-4     |

**Helmholtz** at `Nce=Nfe=8`, `k₀=2`, impedance BC with per-side plane-wave data:

| N_e | relative L²  | relative H¹ |
|-----|--------------|-------------|
| 1   | 1.70e-5      | 6.10e-4     |
| 2   | 6.78e-7      | 3.43e-5     |
| 3   | 1.60e-7      | 8.75e-6     |
| 4   | 6.57e-9      | 4.78e-7     |
| 5   | 7.40e-10     | 6.16e-8     |

At `N_e = N_f − 1` the generalised eigenproblem sits at the numerical rank boundary of `R'NR` (`R` has `N_f − 1` rows) and can pick up a spurious mode. Keep `N_e ≤ N_f − 2` for stable convergence.

## Relevant papers

1. Yifan Chen, Thomas Y. Hou, Yixuan Wang. "[Exponential Convergence for Multiscale Linear Elliptic PDEs via Adaptive Edge Basis Functions](https://arxiv.org/abs/2007.07418)", SIAM Multiscale Modeling and Simulation, 2021.
```
@article{chen2021exponential,
  title={Exponential convergence for multiscale linear elliptic PDEs via adaptive edge basis functions},
  author={Chen, Yifan and Hou, Thomas Y and Wang, Yixuan},
  journal={Multiscale Modeling \& Simulation},
  volume={19},
  number={2},
  pages={980--1010},
  year={2021},
  publisher={SIAM}
}
```


2. Yifan Chen, Thomas Y. Hou, Yixuan Wang. "[Exponentially convergent multiscale methods for high frequency heterogeneous Helmholtz equations](https://arxiv.org/abs/2105.04080)", SIAM Multiscale Modeling and Simulation, 2023.
```
@article{chen2023exponentially,
  title={Exponentially convergent multiscale methods for 2d high frequency heterogeneous Helmholtz equations},
  author={Chen, Yifan and Hou, Thomas Y and Wang, Yixuan},
  journal={Multiscale Modeling \& Simulation},
  volume={21},
  number={3},
  pages={849--883},
  year={2023},
  publisher={SIAM}
}
```
3. Yifan Chen, Thomas Y. Hou, and Yixuan Wang. "[Exponentially convergent multiscale finite element method](https://link.springer.com/article/10.1007/s42967-023-00260-2)". Communications on Applied Mathematics and Computation, 2024.
```
@article{chen2024exponentially,
  title={Exponentially convergent multiscale finite element method},
  author={Chen, Yifan and Hou, Thomas Y and Wang, Yixuan},
  journal={Communications on Applied Mathematics and Computation},
  volume={6},
  number={2},
  pages={862--878},
  year={2024},
  publisher={Springer}
}
```
