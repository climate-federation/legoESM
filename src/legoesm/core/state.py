"""State containers for legoESM components.

All states are registered as JAX pytrees so they work seamlessly with
jit, grad, vmap, scan, and checkpoint.
"""

from __future__ import annotations

from typing import NamedTuple

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
# Non-Hydrostatic Compressible Euler Equations
# ==============================================================================

class NonHydrostaticState(NamedTuple):
    """State for the fully compressible non-hydrostatic Euler equations.

    Uses reference-state subtraction: prognostic variables are perturbations
    from a 1D hydrostatically balanced reference state rho_0(z), theta_0(z).

    3D fields at full levels: shape (6, n, n, nlev).
    Vertical velocity at interfaces: shape (6, n, n, nlev+1).
    2D surface fields: shape (6, n, n).
    Tracers: shape (6, n, n, nlev, n_tracers).

    Fields
    ------
    u : Field
        Zonal wind [m/s]. Shape (6, n, n, nlev).
    v : Field
        Meridional wind [m/s]. Shape (6, n, n, nlev).
    w : Field
        Vertical velocity [m/s] at half (interface) levels.
        Shape (6, n, n, nlev+1). Lorenz staggering.
        Boundary conditions: w=0 at model top, w=v_h.grad(z_s) at surface.
    theta_prime : Field
        Potential temperature perturbation [K]. theta' = theta - theta_0(z).
        Shape (6, n, n, nlev).
    rho_prime : Field
        Dry density perturbation [kg/m^3]. rho' = rho - rho_0(z).
        Shape (6, n, n, nlev).
    phis : Field
        Surface geopotential [m^2/s^2]. Static (not time-stepped).
        Shape (6, n, n).
    tracers : Field
        Tracer mixing ratios [kg/kg]. Shape (6, n, n, nlev, n_tracers).
        For dry runs: n_tracers=0 (empty last axis).
        For moist runs: tracers[...,0]=q_vapor, [..1]=q_cloud, [..2]=q_rain.
    """
    u: Field
    v: Field
    w: Field
    theta_prime: Field
    rho_prime: Field
    phis: Field
    tracers: Field


class NonHydrostaticTendencies(NamedTuple):
    """Tendencies (time derivatives) for the non-hydrostatic equations.

    Same pytree structure as NonHydrostaticState so that SSP-RK3
    tree_map works correctly. The phis tendency is always zero.
    """
    du_dt: Field
    dv_dt: Field
    dw_dt: Field
    dtheta_prime_dt: Field
    drho_prime_dt: Field
    dphis_dt: Field
    dtracers_dt: Field


# ==============================================================================
# MPAS Voronoi Mesh States
# ==============================================================================

class MPASShallowWaterState(NamedTuple):
    """State for shallow water equations on an MPAS Voronoi mesh.

    Uses the TRiSK C-grid staggering: thickness on cells, normal
    velocity on edges, topography on cells.

    Fields
    ------
    h : Field
        Fluid depth [m]. Shape (nCells,). Prognostic.
    u : Field
        Normal velocity [m/s]. Shape (nEdges,). Prognostic.
        Positive in the direction from cellsOnEdge[0] to cellsOnEdge[1].
    h_s : Field
        Surface topography height [m]. Shape (nCells,). Static.
    """
    h: Field
    u: Field
    h_s: Field


class MPASShallowWaterTendencies(NamedTuple):
    """Tendencies for the MPAS shallow water equations.

    Same pytree structure as MPASShallowWaterState prognostic fields.
    """
    dh_dt: Field
    du_dt: Field
