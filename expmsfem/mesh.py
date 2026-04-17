"""Index utilities for rectangular Q1 grids.

Global node numbering on an (Nx+1) x (Ny+1) grid: row-major with x varying
first, so node (i, j) with 0 <= i <= Nx, 0 <= j <= Ny has global index
    idx = j * (Nx + 1) + i.

Matlab's loc2glo(N,m,n,i) is 1-indexed; we use 0-indexed (i,j) for the cell
with bottom-left node (i,j) where 0 <= i < Nx, 0 <= j < Ny. Local node order
is counterclockwise {(i,j), (i+1,j), (i+1,j+1), (i,j+1)}.
"""

import numpy as np


def node_index(Nx: int, i, j):
    """Global node index for grid coordinate (i, j). Broadcasts."""
    return j * (Nx + 1) + i


def local_to_global_nodes(Nx: int, Ny: int):
    """Return int array of shape (Nx*Ny, 4) mapping each cell (row-major by
    j then i) to its 4 global node indices in counterclockwise order."""
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="xy")
    ii = ii.ravel()  # (Nx*Ny,)
    jj = jj.ravel()
    g = np.empty((ii.size, 4), dtype=np.int64)
    g[:, 0] = node_index(Nx, ii, jj)
    g[:, 1] = node_index(Nx, ii + 1, jj)
    g[:, 2] = node_index(Nx, ii + 1, jj + 1)
    g[:, 3] = node_index(Nx, ii, jj + 1)
    return g


def cell_boundary_nodes(Nx: int, Ny: int):
    """Perimeter node indices on an (Nx+1)×(Ny+1) grid, 0-indexed, in the
    order `[bottom row, left-col interior, right-col interior, top-row
    interior]` (matches Matlab basefun.m / harmext.m / restrict.m).
    """
    bottom = np.arange(Nx + 1, dtype=np.int64)
    left = np.arange(1, Ny + 1, dtype=np.int64) * (Nx + 1)
    right = np.arange(1, Ny + 1, dtype=np.int64) * (Nx + 1) + Nx
    top_interior = Ny * (Nx + 1) + np.arange(1, Nx, dtype=np.int64)
    return np.concatenate([bottom, left, right, top_interior])


def interior_mask(Nx: int, Ny: int):
    """Return boolean array of shape ((Nx+1)*(Ny+1),) that is True for nodes
    NOT on the patch boundary."""
    n = (Nx + 1) * (Ny + 1)
    mask = np.ones(n, dtype=bool)
    mask[cell_boundary_nodes(Nx, Ny)] = False
    return mask


def domain_boundary_nodes(Nx: int, Ny: int):
    """Global boundary nodes of the whole domain (same formula as
    cell_boundary_nodes but called on the entire fine grid)."""
    return cell_boundary_nodes(Nx, Ny)
