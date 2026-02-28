"""State containers for legoESM components.

All states are registered as JAX pytrees so they work seamlessly with
jit, grad, vmap, scan, and checkpoint.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field


# ==============================================================================
# Shallow Water State (Milestone 1)
# ==============================================================================

class ShallowWaterState(NamedTuple):
    """State for the shallow water equations on the sphere.

    All fields are defined on cell centers of the cubed-sphere grid
    with shape (6, n, n) where 6 is the number of faces.

    Fields
    ------
    h : Field
        Fluid depth [m]. Prognostic.
    u : Field
        Zonal velocity [m/s]. Prognostic.
    v : Field
        Meridional velocity [m/s]. Prognostic.
    h_s : Field
        Surface topography height [m]. Static (not time-stepped).
    """
    h: Field
    u: Field
    v: Field
    h_s: Field


class ShallowWaterTendencies(NamedTuple):
    """Tendencies (time derivatives) for the shallow water equations.

    Same shape as ShallowWaterState prognostic fields.
    """
    dh_dt: Field
    du_dt: Field
    dv_dt: Field


# ==============================================================================
# Hydrostatic Primitive Equations (Milestone 2)
# ==============================================================================

class HydrostaticState(NamedTuple):
    """State for the hydrostatic primitive equations in sigma coordinates.

    Prognostic variables for the hydrostatic atmosphere on the cubed-sphere.
    3D fields have shape (6, n, n, n_levels) with the level axis last.
    2D fields (surface) have shape (6, n, n).

    Fields
    ------
    u : Field
        Zonal wind [m/s]. Prognostic. Shape (6, n, n, nlev).
    v : Field
        Meridional wind [m/s]. Prognostic. Shape (6, n, n, nlev).
    T : Field
        Temperature [K]. Prognostic. Shape (6, n, n, nlev).
    p_s : Field
        Surface pressure [Pa]. Prognostic. Shape (6, n, n).
    phis : Field
        Surface geopotential [m^2/s^2]. Static (not time-stepped).
        Shape (6, n, n).
    """
    u: Field
    v: Field
    T: Field
    p_s: Field
    phis: Field


class HydrostaticTendencies(NamedTuple):
    """Tendencies (time derivatives) for the hydrostatic primitive equations.

    Same structure as HydrostaticState. The phis tendency is always zero
    since surface geopotential is static.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dp_s_dt: Field
    dphis_dt: Field


# ==============================================================================
# Tracer Transport State (Prescribed-wind transport)
# ==============================================================================

class TracerState(NamedTuple):
    """State for prescribed-wind tracer transport.

    Used for DCMIP-2012 transport test cases where the wind field is
    prescribed analytically and only tracers are prognostic.

    Time is included as a prognostic variable so that SSP-RK3 evaluates
    the prescribed wind at the correct intermediate times for each
    Runge-Kutta stage (dtime_dt = 1.0).

    Fields
    ------
    tracers : Field
        Tracer mixing ratios. Shape (6, n, n, nlev, n_tracers).
        The last axis indexes individual tracers.
    time : Field
        Current simulation time [s]. Scalar (shape ()).
    """
    tracers: Field
    time: Field


class TracerTendencies(NamedTuple):
    """Tendencies for tracer transport.

    Same pytree structure as TracerState so that SSP-RK3 tree_map
    works correctly.
    """
    dtracers_dt: Field
    dtime_dt: Field


# ==============================================================================
# Full Atmosphere State (Future milestones)
# ==============================================================================

class AtmosphereState(NamedTuple):
    """State for the full 3D atmosphere.

    Prognostic variables for the fully compressible non-hydrostatic equations.
    Shape: (6, n, n, n_levels) for 3D fields.
    """
    rho: Field               # Dry air density [kg/m^3]
    theta: Field             # Potential temperature [K]
    u: Field                 # Zonal wind [m/s]
    v: Field                 # Meridional wind [m/s]
    w: Field                 # Vertical velocity [m/s]
    q_vapor: Field           # Specific humidity [kg/kg]
    q_cloud: Field           # Cloud water [kg/kg]
    q_ice: Field             # Cloud ice [kg/kg]
    p_surface: Field         # Surface pressure [Pa]


class AtmosphereTendencies(NamedTuple):
    """Tendencies for atmosphere prognostic variables."""
    drho_dt: Field
    dtheta_dt: Field
    du_dt: Field
    dv_dt: Field
    dw_dt: Field
    dq_vapor_dt: Field
    dq_cloud_dt: Field
    dq_ice_dt: Field
    dp_surface_dt: Field
