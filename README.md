# ExponentialMsFEM

Exponentially convergent multiscale finite element methods in Julia.

## Status

- FEM (`FEMsolver.jl`, `FEMutility.jl`): finished.
- MsFEM (`MsFEMsolver.jl`, `MsFEMutility.jl`): finished.
- ExpMsFEM (`ExponentialMsFEMsolver.jl`, `ExponentialMsFEMutility.jl`): finished on this `julia-code` branch.

## Layout (`Elliptic/`)

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

## Usage

```bash
julia -t 4 Elliptic/ExponentialMsFEMsolver.jl
```

Controlled by environment variables (defaults in parentheses):

- `EXPMSFEM_NCE`  — coarse cells per dimension (8)
- `EXPMSFEM_NFE`  — fine cells per coarse cell (8)
- `EXPMSFEM_NE_MAX` — max number of edge eigenmodes per edge (5)

Example:

```bash
EXPMSFEM_NCE=8 EXPMSFEM_NFE=16 EXPMSFEM_NE_MAX=5 julia -t 4 Elliptic/ExponentialMsFEMsolver.jl
```

Dependencies (`Project.toml`): `ForwardDiff`. The first run auto-precompiles; use `-t N` for `N`-way threaded offline.

## Validation

At `Nce=8`, `Nfe=16`, exponential H¹ decay in the number of edge modes per edge:

| N_e | relative L²  | relative H¹ |
|-----|--------------|-------------|
| 1   | 4.26e-3      | 3.00e-2     |
| 2   | 1.39e-3      | 1.34e-2     |
| 3   | 4.52e-4      | 5.18e-3     |
| 4   | 7.37e-5      | 1.14e-3     |
| 5   | 2.24e-5      | 3.74e-4     |

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
