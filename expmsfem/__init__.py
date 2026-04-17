"""Exponentially convergent multiscale finite element methods (ExpMsFEM).

Python/JAX implementation of the method of Chen, Hou & Wang (SIAM MMS 2021,
2023; CAM 2024). Supports:

* Elliptic problems  `-div(a grad u) = f`  on `(0,1)^2` with
  Dirichlet-0 BC and periodic / random / high-contrast coefficients.
* Helmholtz impedance problems  `-div(a grad u) - k0^2 v^2 u = f`  with
  the impedance boundary `du/dn + i k0 beta u = g` (see `expmsfem.helmholtz`).

Entry points:

    from expmsfem.driver import run_expmsfem        # elliptic
    from expmsfem.helmholtz.driver import run_expmsfem_helm
    from expmsfem.baselines import run_hbubble, run_OH

`fem.solve_fine` computes the fine-scale Q1 reference solution used for
error measurement. `local_ops.Workspace` caches sparse LU factors so the
offline phase can be reused across `N_e` sweeps.
"""

import jax

jax.config.update("jax_enable_x64", True)
