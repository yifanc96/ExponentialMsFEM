"""Q1 bilinear reference stiffness and mass on a square element.

On a square, the 4x4 stiffness ∫ ∇φ_i·∇φ_j scales freely with element size,
so for constant coefficient a the local stiffness is just a · C below.
Mass matrix entries are h^2 × {1/9 on diagonal, 1/18 on edge-adjacent,
1/36 on opposite-corner}. Node ordering is counterclockwise:
1=(m,n), 2=(m+1,n), 3=(m+1,n+1), 4=(m,n+1).
"""

import numpy as np

# Reference 4x4 stiffness matrix for bilinear Q1 on a unit square with a=1.
# Matches Matlab elementstiff1/elementstiff2 exactly.
C_STIFF = np.array(
    [
        [2 / 3, -1 / 6, -1 / 3, -1 / 6],
        [-1 / 6, 2 / 3, -1 / 6, -1 / 3],
        [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
        [-1 / 6, -1 / 3, -1 / 6, 2 / 3],
    ],
    dtype=np.float64,
)

# Reference 4x4 mass matrix (without h^2 factor). M_ij/h^2 =
# {1/9 if i==j, 1/36 if opposite corner, 1/18 otherwise}.
C_MASS = np.array(
    [
        [1 / 9, 1 / 18, 1 / 36, 1 / 18],
        [1 / 18, 1 / 9, 1 / 18, 1 / 36],
        [1 / 36, 1 / 18, 1 / 9, 1 / 18],
        [1 / 18, 1 / 36, 1 / 18, 1 / 9],
    ],
    dtype=np.float64,
)
