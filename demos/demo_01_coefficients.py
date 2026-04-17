"""Visualise the three elliptic coefficient fields that ship with the package:

  1. Multiscale periodic `afun_periodic` — sum of five incommensurate ratios,
     continuous and order-1 in magnitude.
  2. Random rough `afun_random` — bilinear interpolation of a 128×128 field
     of |randn| + 0.5 (contrast ~10).
  3. High-contrast channels `afun_highcontrast` — 49 circular inclusions of
     contrast 64×.

Writes figures/coefficients.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.coefficients import (
    afun_periodic, afun_highcontrast, afun_random, generate_random_field,
)


def main():
    set_style()
    n = 400
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    A_per = afun_periodic(X, Y)
    goo = generate_random_field(M=128, seed=0)
    A_rnd = afun_random(goo, M=128)(X, Y)
    A_hc = afun_highcontrast(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, A, title in zip(
        axes,
        [A_per, A_rnd, A_hc],
        ["Periodic 5-scale", "Random rough (seed 0)", "High-contrast 64×"],
    ):
        im = ax.pcolormesh(X, Y, A, cmap="viridis", shading="auto")
        ax.set_title(f"$a(x)$: {title}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Elliptic coefficient fields", y=1.02, fontsize=13)
    save(fig, "coefficients.png")


if __name__ == "__main__":
    main()
