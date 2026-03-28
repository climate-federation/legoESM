"""CFL (Courant-Friedrichs-Lewy) condition utilities.

Provides automatic time step estimation for explicit time integrators
on cubed-sphere and Gaussian grids.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("legoesm.cfl")


def estimate_min_dx_cubed_sphere(n: int, radius: float = 6.371229e6) -> float:
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


def estimate_min_dx_gaussian(n_max: int, radius: float = 6.371229e6) -> float:
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


def cfl_number_from_state(
    u, v, dx_min: float, dt: float,
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

    Returns
    -------
    jax.Array (scalar)
        Maximum CFL number across all grid points and levels.
    """
    import jax.numpy as jnp
    speed = jnp.sqrt(u ** 2 + v ** 2)
    return jnp.max(speed) * dt * jnp.sqrt(2.0) / dx_min


def cfl_check_and_adjust(
    dt: float,
    n: int,
    model_type: str = "shallow_water",
    max_wind: float = 0.0,
    gravity_wave_speed: float = 0.0,
    cfl_number: float = 0.8,
    radius: float = 6.371229e6,
    verbose: bool = True,
) -> float:
    """Check CFL condition and reduce dt if needed.

    Parameters
    ----------
    dt : float
        Requested time step [s].
    n : int
        Grid resolution (cubed-sphere N or spectral truncation).
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

    Returns
    -------
    dt_safe : float
        Adjusted time step (≤ dt) that satisfies CFL.
    """
    dx_min = estimate_min_dx_cubed_sphere(n, radius)

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
        return dt
