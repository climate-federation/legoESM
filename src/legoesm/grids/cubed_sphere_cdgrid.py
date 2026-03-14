"""FV3-style C-D grid on the cubed-sphere.

Extends the A-grid ``CubedSphereGrid`` with staggered metric fields
needed for the C-D grid discretisation:

* **D-grid** (cell corners): prognostic wind components ``u_d, v_d``
  and absolute vorticity live at shape ``(6, n+1, n+1)``.
* **C-grid** (cell edges): normal velocity used for mass flux.
  ``u_c`` at x-interfaces: ``(6, n+1, n)``; ``v_c`` at y-interfaces:
  ``(6, n, n+1)``.

The key advantage is that vorticity on the D-grid is computed directly
from corner wind values without spatial averaging, eliminating the
Hollingsworth-Kallberg instability that plagues A-grid schemes.

References
----------
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import (
    CubedSphereGrid,
    _face_to_cartesian,
)
from legoesm.grids.halo import _face_gnomonic_to_lonlat


class CubedSphereCDGrid(NamedTuple):
    """C-D grid metrics for a cubed-sphere grid.

    Wraps a base ``CubedSphereGrid`` (A-grid) and adds the staggered
    metric arrays needed by the C-D grid discretisation.

    Attributes
    ----------
    base : CubedSphereGrid
        The underlying A-grid with cell-center metrics.
    lon_corner : jax.Array, shape (6, n+1, n+1)
        Longitude at cell corners (D-grid positions).
    lat_corner : jax.Array, shape (6, n+1, n+1)
        Latitude at cell corners.
    f_corner : jax.Array, shape (6, n+1, n+1)
        Coriolis parameter at cell corners.
    angle_corner : jax.Array, shape (6, n+1, n+1)
        Grid rotation angle at cell corners.
    cos_angle_corner : jax.Array, shape (6, n+1, n+1)
    sin_angle_corner : jax.Array, shape (6, n+1, n+1)
    dx_edge_y : jax.Array, shape (6, n, n+1)
        Length of cell edges in y-direction (v_c positions).
    dy_edge_x : jax.Array, shape (6, n+1, n)
        Length of cell edges in x-direction (u_c positions).
    area_corner : jax.Array, shape (6, n+1, n+1)
        Dual-cell area at corners (for vorticity equation).
    """
    base: CubedSphereGrid
    lon_corner: jax.Array
    lat_corner: jax.Array
    f_corner: jax.Array
    angle_corner: jax.Array
    cos_angle_corner: jax.Array
    sin_angle_corner: jax.Array
    dx_edge_y: jax.Array
    dy_edge_x: jax.Array
    area_corner: jax.Array

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def radius(self) -> float:
        return self.base.radius


def create_cubed_sphere_cdgrid(
    base: CubedSphereGrid,
    omega: float = 7.292e-5,
) -> CubedSphereCDGrid:
    """Create a C-D grid from an existing A-grid.

    Parameters
    ----------
    base : CubedSphereGrid
        A-grid cubed-sphere with cell-center metrics.
    omega : float
        Planetary rotation rate [rad/s].

    Returns
    -------
    CubedSphereCDGrid
    """
    n = base.n
    radius = base.radius

    # Cell corner positions (gnomonic grid edges: n+1 per side)
    alpha_edges = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n + 1)
    ax_e, ay_e = jnp.meshgrid(alpha_edges, alpha_edges, indexing='ij')

    all_lon_c, all_lat_c = [], []
    all_angle_c = []
    all_dx_ey, all_dy_ex = [], []
    all_area_c = []

    for face in range(6):
        lon_c, lat_c = _face_gnomonic_to_lonlat(face, ax_e, ay_e)
        all_lon_c.append(lon_c)
        all_lat_c.append(lat_c)

        # Cartesian coordinates at corners for metric computation
        cos_lat_c = jnp.cos(lat_c)
        x_c = cos_lat_c * jnp.cos(lon_c)
        y_c = cos_lat_c * jnp.sin(lon_c)
        z_c = jnp.sin(lat_c)

        # Edge lengths in x-direction (dy_edge_x): distance between
        # consecutive corners in the x-direction → shape (n+1, n)
        # (length of the cell edge crossed by u_c)
        chord_x = jnp.sqrt(
            (x_c[1:, :] - x_c[:-1, :]) ** 2
            + (y_c[1:, :] - y_c[:-1, :]) ** 2
            + (z_c[1:, :] - z_c[:-1, :]) ** 2
        )  # (n, n+1) — between consecutive x-corners, all y-corners
        # Actually: dy_edge_x has shape (n+1, n) — the edge between
        # corner (i,j) and (i,j+1), which is an x-oriented edge
        # u_c lives on x-oriented edges: (n+1, n)
        # These edges connect (i,j) to (i,j+1) — y-direction span
        # Wait, let me reconsider the staggering:
        # u_c at x-interfaces: between cell (i,j) and cell (i+1,j)
        #   The interface has corners at (i+1,j) and (i+1,j+1)
        #   Its length is the great-circle distance between those corners
        # v_c at y-interfaces: between cell (i,j) and cell (i,j+1)
        #   The interface has corners at (i,j+1) and (i+1,j+1)
        #   Its length is the great-circle distance between those corners

        # dy_edge_x: length of x-interface (u_c position), shape (n+1, n)
        # = distance between corner (i, j) and corner (i, j+1)
        chord_dy = jnp.sqrt(
            (x_c[:, 1:] - x_c[:, :-1]) ** 2
            + (y_c[:, 1:] - y_c[:, :-1]) ** 2
            + (z_c[:, 1:] - z_c[:, :-1]) ** 2
        )  # (n+1, n)
        dy_ex = radius * 2.0 * jnp.arcsin(jnp.clip(chord_dy / 2.0, 0.0, 1.0))
        all_dy_ex.append(dy_ex)

        # dx_edge_y: length of y-interface (v_c position), shape (n, n+1)
        # = distance between corner (i, j) and corner (i+1, j)
        chord_dx = jnp.sqrt(
            (x_c[1:, :] - x_c[:-1, :]) ** 2
            + (y_c[1:, :] - y_c[:-1, :]) ** 2
            + (z_c[1:, :] - z_c[:-1, :]) ** 2
        )  # (n, n+1)
        dx_ey = radius * 2.0 * jnp.arcsin(jnp.clip(chord_dx / 2.0, 0.0, 1.0))
        all_dx_ey.append(dx_ey)

        # Grid angle at corners (angle between gnomonic x-axis and east)
        dalpha = jnp.pi / (2 * n)
        # Extend by one cell in each direction for centred differences
        n_ext = n + 3  # n+1 + 2 for centred diffs
        alpha_ext = jnp.linspace(
            -jnp.pi / 4 - dalpha,
            jnp.pi / 4 + dalpha,
            n_ext,
        )
        ax_ext, ay_ext = jnp.meshgrid(alpha_ext, alpha_ext, indexing='ij')
        lon_ext, lat_ext = _face_gnomonic_to_lonlat(face, ax_ext, ay_ext)
        dlon = lon_ext[2:, 1:-1] - lon_ext[:-2, 1:-1]
        dlat = lat_ext[2:, 1:-1] - lat_ext[:-2, 1:-1]
        dlon = jnp.where(dlon > jnp.pi, dlon - 2 * jnp.pi, dlon)
        dlon = jnp.where(dlon < -jnp.pi, dlon + 2 * jnp.pi, dlon)
        cos_lat_ext = jnp.cos(lat_ext[1:-1, 1:-1])
        face_angle = jnp.arctan2(dlat, dlon * cos_lat_ext)
        all_angle_c.append(face_angle)

        # Dual-cell area at corners: average of the 4 surrounding cell areas
        # For interior corners (1..n-1, 1..n-1), average 4 cells.
        # For edge/corner of the face, use available cells (1-3).
        # Approximate: 0.25 * sum of up-to-4 surrounding cell areas
        area = base.area[face]  # (n, n)
        area_padded = jnp.pad(area, 1, mode='edge')  # (n+2, n+2)
        area_c_face = 0.25 * (
            area_padded[:-1, :-1]
            + area_padded[1:, :-1]
            + area_padded[:-1, 1:]
            + area_padded[1:, 1:]
        )  # (n+1, n+1)
        all_area_c.append(area_c_face)

    lon_corner = jnp.stack(all_lon_c, axis=0)
    lat_corner = jnp.stack(all_lat_c, axis=0)
    angle_corner = jnp.stack(all_angle_c, axis=0)
    dx_edge_y = jnp.stack(all_dx_ey, axis=0)
    dy_edge_x = jnp.stack(all_dy_ex, axis=0)
    area_corner = jnp.stack(all_area_c, axis=0)

    f_corner = 2.0 * omega * jnp.sin(lat_corner)
    cos_angle_corner = jnp.cos(angle_corner)
    sin_angle_corner = jnp.sin(angle_corner)

    _f32 = jnp.float32
    return CubedSphereCDGrid(
        base=base,
        lon_corner=lon_corner.astype(_f32),
        lat_corner=lat_corner.astype(_f32),
        f_corner=f_corner.astype(_f32),
        angle_corner=angle_corner.astype(_f32),
        cos_angle_corner=cos_angle_corner.astype(_f32),
        sin_angle_corner=sin_angle_corner.astype(_f32),
        dx_edge_y=dx_edge_y.astype(_f32),
        dy_edge_x=dy_edge_x.astype(_f32),
        area_corner=area_corner.astype(_f32),
    )
