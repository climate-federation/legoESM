"""FV operators for cubed-sphere with divergence damping utilities.

This module provides utilities for finite-volume discretizations on the
cubed-sphere, including default divergence damping coefficient calculation.
"""

from __future__ import annotations

from legoesm.grids.cubed_sphere import CubedSphereGrid


def default_div_damp_coeffs(
    grid: CubedSphereGrid,
    dt: float = 300.0,
    cfl_factor: float = 0.1,
) -> tuple[float, float]:
    """Compute default divergence damping coefficients for cubed-sphere FV.

    Divergence damping is used in finite-volume schemes to suppress spurious
    divergence oscillations that can arise from the discretization. The
    coefficients scale with the grid spacing and timestep for numerical stability.

    Parameters
    ----------
    grid : CubedSphereGrid
        The cubed-sphere horizontal grid.
    dt : float, optional
        Timestep in seconds. Default 300.0.
    cfl_factor : float, optional
        CFL scaling factor for stability margin (0 < cfl_factor < 1).
        Default 0.1 gives conservative (small) damping.

    Returns
    -------
    nu2 : float
        2nd-order divergence damping coefficient [m²/s].
    nu4 : float
        4th-order divergence damping coefficient [m⁴/s].

    Notes
    -----
    The coefficients are scaled as:
    - nu2 ~ (grid_spacing)^2 / dt
    - nu4 ~ (grid_spacing)^4 / dt

    This scaling ensures that divergence damping becomes stronger on
    finer grids (higher resolution), which is necessary for stability
    of finite-volume schemes.
    """
    # Approximate cell size in meters on the cubed-sphere
    # For a cubed-sphere with n points per edge and Earth radius ~6.371e6 m
    # each cell has characteristic size ~ (2*pi*R) / (6*n) meters
    earth_radius = 6.371e6  # meters
    face_perimeter = 2 * 3.141592653589793 * earth_radius / 3.0
    dx = face_perimeter / grid.n

    # Default damping coefficients scaled by grid spacing and timestep
    # cfl_factor provides a tunable stability margin
    nu2 = cfl_factor * (dx ** 2) / dt  # [m²/s]
    nu4 = cfl_factor * (dx ** 4) / dt  # [m⁴/s]

    return nu2, nu4
