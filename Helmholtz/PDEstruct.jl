# PDE container for the 2D impedance-boundary Helmholtz problem
#   -∇·(a ∇u) - k0² v² u = f   in Ω = (0, 1)²
#   ∂u/∂n + i k0 β u = g       on ∂Ω
#
# `afun`, `vfun`, `betafun`, `ffun` are real-valued scalar functions
# (t, s) → ℝ. `gfun` is complex-valued and additionally takes k0 as its
# third argument: (t, s, k0) → ℂ.

abstract type AbstractPDEs end

struct Helmholtz2D <: AbstractPDEs
    k0::Float64
    afun::Function    # diffusion coefficient a(t, s)
    vfun::Function    # wave speed v(t, s)
    betafun::Function # impedance indicator β(t, s), 1 on ∂Ω, 0 else
    ffun::Function    # interior source f(t, s)
    gfun::Function    # boundary data g(t, s, k0), complex
end
