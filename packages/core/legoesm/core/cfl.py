"""CFL (Courant-Friedrichs-Lewy) condition utilities.

Provides automatic time step estimation for explicit time integrators
on cubed-sphere, Gaussian, and lat-lon grids.
"""
from __future__ import annotations

import logging
import numpy as np

from legoesm import constants

logger = logging.getLogger("legoesm.cfl")


def estimate_min_dx_cubed_sphere(n: int, radius: float = constants.R_earth) -> float:
    """Estimate minimum grid spacing on a cubed-sphere grid.

    Parameters
    ----------
    n : int
        Cubed-sphere resolution (cells per face edge).
    radius : float
        Sphere radius [m].

    Returns
    -------
    dx_min : float
        Approximate minimum grid spacing [m].
    """
    # Each face spans pi/2 radians. Minimum dx occurs at cube corners
    # where the great-circle distance between adjacent points is smallest.
    # The corner-to-corner ratio is approximately 1/sqrt(3) of the face-center dx.
    dx_nominal = (np.pi / 2) * radius / n
    dx_min = dx_nominal / np.sqrt(3)
    return float(dx_min)


def estimate_min_dx_latlon(n_lat: int, radius: float = constants.R_earth) -> float:
    """Estimate minimum grid spacing on a latitude-longitude grid.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    radius : float
        Sphere radius [m].

    Returns
    -------
    dx_min : float
        Approximate minimum single-cell grid spacing [m].
        Occurs at the polar-most row where cos(lat) is smallest.
    """
    dlat = np.pi / n_lat
    dlon = 2.0 * np.pi / (2 * n_lat)  # n_lon = 2 * n_lat
    # Polar-most cell center is at lat = pi/2 - dlat/2
    cos_lat_pole = np.cos(np.pi / 2.0 - dlat / 2.0)
    # Single-cell dx at the polar-most row
    dx_pole = radius * dlon * cos_lat_pole
    # dy is constant: single-cell dy = radius * dlat
    dy = radius * dlat
    return float(min(dx_pole, dy))


def estimate_min_dx_gaussian(n_max: int, radius: float = constants.R_earth) -> float:
    """Estimate minimum grid spacing on a Gaussian grid.

    Parameters
    ----------
    n_max : int
        Spectral truncation.
    radius : float
        Sphere radius [m].

    Returns
    -------
    dx_min : float
        Approximate minimum grid spacing [m] (at the equator).
    """
    n_lon = 2 * (n_max + 1)  # standard quadratic grid
    dx_equator = 2 * np.pi * radius / n_lon
    return float(dx_equator)


def estimate_min_dx_icosahedral(level: int, radius: float = constants.R_earth) -> float:
    """Estimate minimum grid spacing on an icosahedral (Voronoi/MPAS) grid.

    Parameters
    ----------
    level : int
        Icosahedral subdivision level (e.g., 5 → ~10242 cells).
    radius : float
        Sphere radius [m].

    Returns
    -------
    dx_min : float
        Approximate minimum edge-to-edge spacing [m].
    """
    # Number of cells: 10 * 4^level + 2 (icosahedral subdivision)
    n_cells = 10 * 4**level + 2
    # Average cell spacing from sphere area / n_cells
    dx_avg = np.sqrt(4 * np.pi * radius**2 / n_cells)
    # Minimum spacing is ~0.85 of the average for quasi-uniform meshes
    return float(0.85 * dx_avg)


def cfl_max_dt(
    dx_min: float,
    wave_speed: float,
    cfl_number: float = 0.8,
    ndim: int = 2,
) -> float:
    """Compute maximum stable time step from CFL condition.

    Parameters
    ----------
    dx_min : float
        Minimum grid spacing [m].
    wave_speed : float
        Maximum wave propagation speed [m/s].
    cfl_number : float
        Target CFL number (< 1 for stability, default 0.8).
    ndim : int
        Number of spatial dimensions (2 for horizontal, 3 for 3D).

    Returns
    -------
    dt_max : float
        Maximum stable time step [s].
    """
    return cfl_number * dx_min / (wave_speed * np.sqrt(ndim))


def adaptive_hyperdiff_coeff(
    dx_min: float,
    dt: float,
    order: int = 4,
    safety: float = 0.5,
) -> float:
    """Compute a stable hyperdiffusion coefficient for a given grid and timestep.

    For an n-th order hyperdiffusion ν_n · (-1)^{n/2+1} ∇^n, the maximum
    stable coefficient under forward Euler is:

        ν_n · dt / dx^n < C(n)

    where C(n) is an order-dependent stability constant:
        - order 2: C = 0.5
        - order 4: C = 1/8
        - order 6: C = 1/48

    This function returns ν_n = safety * C(n) * dx^n / dt.

    Parameters
    ----------
    dx_min : float
        Minimum grid spacing [m].
    dt : float
        Time step [s].
    order : int
        Diffusion order (2 = Laplacian, 4 = biharmonic, 6 = triharmonic).
    safety : float
        Safety factor (0 < safety <= 1). Default 0.5 gives half the
        theoretical maximum, providing a margin for multi-stage RK schemes.

    Returns
    -------
    float
        Hyperdiffusion coefficient ν_n [m^n/s].
    """
    stability_constants = {2: 0.5, 4: 0.125, 6: 1.0 / 48.0}
    C_n = stability_constants.get(order, 1.0 / (2.0 ** order))
    return safety * C_n * dx_min ** order / dt


def pole_cell_dx(grid) -> float:
    """Return the zonal grid spacing at the polar-most cell [m].

    Parameters
    ----------
    grid : LatLonGrid
        Must have ``radius``, ``dlon``, ``dlat`` attributes.

    Returns
    -------
    float
        dx_pole = R * dlon * cos(π/2 − dlat/2).
    """
    R = float(grid.radius)
    dlon = float(grid.dlon)
    dlat = float(grid.dlat)
    return R * dlon * np.cos(np.pi / 2.0 - dlat / 2.0)


def max_laplacian_viscosity(dx_min: float, dt: float) -> float:
    """Maximum stable Laplacian viscosity coefficient.

    For forward-Euler Laplacian diffusion the stability constraint is
    A_h * dt / dx² < 0.5.  With a safety margin the practical limit is
    A_h_max = 0.4 * dx² / dt.

    Parameters
    ----------
    dx_min : float
        Minimum grid spacing [m].
    dt : float
        Time step [s].

    Returns
    -------
    float
        A_h_max [m²/s].
    """
    return 0.4 * dx_min ** 2 / dt


def cfl_number_from_state(
    u, v, dx_min: float, dt: float, runtime=None,
):
    """Compute the advective CFL number from wind fields (JAX-traceable).

    CFL = max(|u|, |v|) * dt * sqrt(2) / dx_min

    Parameters
    ----------
    u, v : jax.Array
        Wind components, any shape.
    dx_min : float
        Minimum grid spacing [m].
    dt : float
        Time step [s].
    runtime : ParallelRuntime or None, optional
        Parallel runtime for cross-rank global max in MPI mode.
        In multi-device SPMD mode (no MPI), ``jnp.max`` on a sharded
        array already produces the correct global result via XLA's
        automatic all-reduce, so this parameter is not needed.
        In MPI mode, pass the runtime so that the local max is
        combined across ranks.  If None, only the local max is used.

    Returns
    -------
    jax.Array (scalar)
        Maximum CFL number across all grid points and levels.
    """
    import jax.numpy as jnp
    speed = jnp.sqrt(u ** 2 + v ** 2)
    local_max_speed = jnp.max(speed)

    # In MPI mode, each rank has genuinely separate data and jnp.max
    # only sees the local shard.  Use the runtime's global_max to
    # combine across ranks.  In single-process multi-device SPMD,
    # jnp.max on a sharded array already inserts an XLA all-reduce.
    if runtime is not None:
        local_max_speed = runtime.global_max(local_max_speed)

    return local_max_speed * dt * jnp.sqrt(2.0) / dx_min


def cfl_check_and_adjust(
    dt: float,
    n: int,
    model_type: str = "shallow_water",
    max_wind: float = 0.0,
    gravity_wave_speed: float = 0.0,
    cfl_number: float = 0.8,
    radius: float = constants.R_earth,
    verbose: bool = True,
    grid_type: str = "cubed_sphere",
    use_polar_filter: bool = False,
) -> float:
    """Check CFL condition and reduce dt if needed.

    Parameters
    ----------
    dt : float
        Requested time step [s].
    n : int
        Grid resolution (cubed-sphere N, spectral truncation, or n_lat).
    model_type : str
        One of 'shallow_water', 'primitive_eq', 'compressible'.
    max_wind : float
        Maximum expected wind speed [m/s].
    gravity_wave_speed : float
        Gravity wave phase speed [m/s]. For shallow water, sqrt(g*H).
        For PE, ~sqrt(g*H_scale) or sound speed for compressible.
    cfl_number : float
        Target CFL number.
    radius : float
        Sphere radius [m].
    verbose : bool
        Print CFL diagnostics.
    grid_type : str
        Grid type: 'cubed_sphere', 'gaussian', 'latlon', or 'voronoi'.

    Returns
    -------
    dt_safe : float
        Adjusted time step (≤ dt) that satisfies CFL.
    """
    if grid_type == "latlon":
        if use_polar_filter:
            # Stage 3-E: Fourier polar filter truncates the
            # high-wavenumber modes that would violate CFL near the
            # poles, so the actual stability limit is the equatorial
            # CFL ``R * dlon``.  Without this branch the legacy
            # ``estimate_min_dx_latlon`` returns the pole-cell dx
            # (a ~60x smaller value at n_lat=180) and the driver
            # clamps ``dt`` to ~5 s — undoing the polar filter's
            # whole purpose.  See ``component_factory.py`` for the
            # mirror logic at the model-builder level.
            n_lon = 2 * n
            dx_min = float(2.0 * np.pi * radius / n_lon)
        else:
            dx_min = estimate_min_dx_latlon(n, radius)
    elif grid_type == "gaussian":
        dx_min = estimate_min_dx_gaussian(n, radius)
    elif grid_type in ("mpas", "voronoi"):
        # "voronoi" is the parallel-namespace alias for the icosahedral/MPAS
        # mesh; both dispatch to the icosahedral dx estimator.
        dx_min = estimate_min_dx_icosahedral(n, radius)
    elif grid_type == "cubed_sphere":
        dx_min = estimate_min_dx_cubed_sphere(n, radius)
    else:
        # Dispatch hardening: a typo'd grid_type must not silently take the
        # cubed-sphere dx (a wrong dt clamp on a lat-lon/mpas run).
        raise ValueError(
            f"cfl_check_and_adjust: unknown grid_type {grid_type!r}; expected "
            "one of 'latlon', 'gaussian', 'mpas', 'voronoi', 'cubed_sphere'."
        )

    # Total wave speed = max(wind) + gravity_wave_speed
    c_total = max_wind + gravity_wave_speed
    if c_total <= 0:
        # Estimate default wave speeds
        if model_type == "shallow_water":
            c_total = 20.0 + 250.0  # typical wind + sqrt(g*6000)
        elif model_type == "primitive_eq":
            c_total = 50.0 + 340.0  # jet + external gravity wave
        elif model_type == "compressible":
            c_total = 50.0 + 340.0  # jet + sound speed
        else:
            c_total = 50.0 + 300.0

    dt_max = cfl_max_dt(dx_min, c_total, cfl_number, ndim=2)

    if verbose:
        cfl_actual = c_total * dt / dx_min * np.sqrt(2)
        logger.info(f"  CFL check: dx_min={dx_min/1000:.0f} km, "
              f"c_max={c_total:.0f} m/s, "
              f"CFL(dt={dt:.0f}s)={cfl_actual:.2f}, "
              f"dt_max={dt_max:.0f}s")

    if dt > dt_max:
        # Round down to nearest "nice" value
        nice_values = [600, 450, 300, 240, 200, 180, 150, 120, 100, 90, 60, 45, 30, 20, 15, 10]
        dt_safe = dt_max
        for nv in nice_values:
            if nv <= dt_max:
                dt_safe = float(nv)
                break
        if verbose:
            logger.warning(f"  WARNING: dt={dt:.0f}s exceeds CFL limit. "
                  f"Reducing to dt={dt_safe:.0f}s")
        return dt_safe
    else:
        if verbose:
            logger.info(f"  CFL OK: dt={dt:.0f}s is within stability limit")
            if grid_type == "mpas":
                # NECESSARY, NOT SUFFICIENT for the hydrostatic TRiSK PE.
                # This advective/gravity-wave CFL has been observed to pass
                # ("OK") at time steps that then blow up: e.g. L4/nlev40 with
                # the gray AMIP deck NaNs at dt=450 s (CFL~0.65 here) within a
                # day, while dt=240/300 s run cleanly.  The extra constraints
                # this estimate does NOT capture are (a) the cold-start
                # radiative transient (a T=300 K isothermal IC carries a large
                # day-0 radiative tendency that a big dt cannot absorb) and
                # (b) the explicit ∇⁴ hyperdiffusion eigenvalues vs the
                # integrator's stability region.  Treat this as an upper bound
                # and keep a margin (CFL <~ 0.5) on a cold start.
                logger.info(
                    "  NOTE (mpas): advective/GW CFL is necessary but not "
                    "sufficient — hydrostatic TRiSK PE can still blow up "
                    "inside this limit (cold-start radiative transient, "
                    "hyperdiffusion). Keep a margin on a cold start."
                )
        return dt
