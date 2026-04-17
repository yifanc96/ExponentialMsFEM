"""Helmholtz ExpMsFEM port of the Matlab `helmholtz/case1/Exp` code.

Solves the impedance-boundary Helmholtz problem on Ω = (0, 1)²:

    -∇·(a ∇u) - k₀² v² u = f      in Ω
          ∂u/∂n + i k₀ β u = g    on ∂Ω

with the ExpMsFEM edge-enriched multiscale basis (same construction as the
elliptic case, but with a complex, indefinite local operator that includes
the impedance boundary term on cells touching ∂Ω). For `case1`:

    a(x, y) = 1           (constant diffusion)
    v(x, y) = 1           (constant wave speed)
    β(x, y) = 1 on ∂Ω, 0 inside
    f(x, y) = 0
    g is a plane-wave trace from direction (0.6, 0.8):
        g(x, y) = -i k₀ · a_side · exp(-0.6 i k₀ x - 0.8 i k₀ y)
    with per-side coefficients a_side = [0.2, 1.6, 1.8, 0.4] for
    (y=0, x=1, y=1, x=0).
"""
