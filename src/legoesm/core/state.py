"""State containers for legoESM components.

All states are registered as JAX pytrees so they work seamlessly with
jit, grad, vmap, scan, and checkpoint.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import jax

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
    tracers : dict[str, Field] | None
        Tracer mixing ratios as a dictionary. Keys are tracer names
        (e.g., "q_v", "q_c", "q_r"). Values are Fields with shape
        (6, n, n, nlev). None when tracers are not in use.
    """
    u: Field
    v: Field
    T: Field
    p_s: Field
    phis: Field
    tracers: dict[str, Field] | None = None


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


class FV3HydrostaticState(NamedTuple):
    """State for FV3 hydrostatic PE with D-grid prognostic winds.

    Winds are stored at D-grid cell corners (6, n+1, n+1, nlev).
    Scalars at cell centres (6, n, n[, nlev]).
    """
    u_d: Field    # D-grid x-velocity (6, n+1, n+1, nlev)
    v_d: Field    # D-grid y-velocity (6, n+1, n+1, nlev)
    T: Field      # Temperature (6, n, n, nlev)
    p_s: Field    # Surface pressure (6, n, n)
    phis: Field   # Surface geopotential (6, n, n)
    tracers: dict[str, Field] | None = None


class FV3HydrostaticTendencies(NamedTuple):
    """Tendencies for FV3 hydrostatic PE (D-grid winds)."""
    du_d_dt: Field   # (6, n+1, n+1, nlev)
    dv_d_dt: Field   # (6, n+1, n+1, nlev)
    dT_dt: Field     # (6, n, n, nlev)
    dp_s_dt: Field   # (6, n, n)
    dphis_dt: Field  # (6, n, n) — always zero


class PhysicsState(NamedTuple):
    """Prognostic physics state carried across timesteps.

    All fields are optional (None when the corresponding scheme is inactive).
    Registered as a JAX pytree so it can be checkpointed, vmapped, and
    passed through jax.lax.scan.
    """
    tke: jax.Array | None = None              # Turbulent kinetic energy (ncol, nlev)
    conv_mass_flux: jax.Array | None = None   # Convective mass flux (ncol,)
    conv_prog: jax.Array | None = None        # Convective prognostic state
    gwd_wave_action: jax.Array | None = None  # Gravity wave drag spectrum (ncol, nlev, n_wave)
    clubb_moments: jax.Array | None = None    # CLUBB higher-order moments (ncol, nlev, 5)
    radiation_tend: jax.Array | None = None   # Held radiation tendencies for sub-cycling


# ==============================================================================
# Physics Module Protocol
# ==============================================================================

@runtime_checkable
class PhysicsModuleProtocol(Protocol):
    """Interface contract for all physics parameterization modules.

    Every physics factory (make_radiation_physics, make_turbulence_physics, etc.)
    must return a callable matching this protocol.
    """
    def __call__(
        self,
        state: HydrostaticState,
        grid,
        coord,
        phys_state: PhysicsState | None = None,
    ) -> HydrostaticTendencies: ...

    def set_time(self, day_of_year: float, seconds_of_day: float) -> None: ...
    def reset_state(self) -> None: ...


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


# ==============================================================================
# MPAS Voronoi Mesh Ocean States
# ==============================================================================

# ==============================================================================
# MPAS Voronoi Mesh Atmosphere States
# ==============================================================================

class MPASHydrostaticState(NamedTuple):
    """State for the hydrostatic primitive equations on an MPAS Voronoi mesh.

    Uses TRiSK C-grid staggering: normal velocity on edges, scalars on cells.

    Fields
    ------
    u : Field
        Normal velocity [m/s]. Shape (nEdges, nlev). Prognostic.
    T : Field
        Temperature [K]. Shape (nCells, nlev). Prognostic.
    p_s : Field
        Surface pressure [Pa]. Shape (nCells,). Prognostic.
    phis : Field
        Surface geopotential [m^2/s^2]. Shape (nCells,). Static.
    """
    u: Field
    T: Field
    p_s: Field
    phis: Field


class MPASHydrostaticTendencies(NamedTuple):
    """Tendencies for the MPAS hydrostatic primitive equations."""
    du_dt: Field
    dT_dt: Field
    dp_s_dt: Field
    dphis_dt: Field


class MPASNonHydrostaticState(NamedTuple):
    """State for the non-hydrostatic compressible Euler equations on MPAS.

    Uses TRiSK C-grid staggering: normal velocity on edges, all other
    prognostic variables on cells. Vertical velocity w uses Lorenz
    staggering at interface levels.

    Fields
    ------
    u : Field
        Normal velocity [m/s]. Shape (nEdges, nlev). Prognostic.
    w : Field
        Vertical velocity [m/s] at interface levels.
        Shape (nCells, nlev+1). Lorenz staggering.
    theta_prime : Field
        Potential temperature perturbation [K]. Shape (nCells, nlev).
    rho_prime : Field
        Dry density perturbation [kg/m^3]. Shape (nCells, nlev).
    phis : Field
        Surface geopotential [m^2/s^2]. Shape (nCells,). Static.
    tracers : Field
        Tracer mixing ratios [kg/kg]. Shape (nCells, nlev, n_tracers).
    """
    u: Field
    w: Field
    theta_prime: Field
    rho_prime: Field
    phis: Field
    tracers: Field


class MPASNonHydrostaticTendencies(NamedTuple):
    """Tendencies for the MPAS non-hydrostatic equations."""
    du_dt: Field
    dw_dt: Field
    dtheta_prime_dt: Field
    drho_prime_dt: Field
    dphis_dt: Field
    dtracers_dt: Field


# ==============================================================================
# MPAS Voronoi Mesh Ocean States
# ==============================================================================

class MPASOceanState(NamedTuple):
    """State for ocean primitive equations on an MPAS Voronoi mesh.

    Uses TRiSK C-grid staggering: tracers/SSH on cells, normal
    velocity on edges.

    Fields
    ------
    u : Field
        Normal velocity [m/s]. Shape (nEdges, nlev). Prognostic.
    T : Field
        Potential temperature [degC]. Shape (nCells, nlev). Prognostic.
    S : Field
        Salinity [PSU]. Shape (nCells, nlev). Prognostic.
    eta : Field
        Sea surface height [m]. Shape (nCells,). Prognostic.
    H_bathy : Field
        Bathymetry depth [m]. Shape (nCells,). Positive downward. Static.
    land_mask : Field
        Ocean mask. Shape (nCells,). 1=ocean, 0=land. Static.
    """
    u: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class MPASOceanTendencies(NamedTuple):
    """Tendencies for MPAS ocean primitive equations.

    Only prognostic fields have tendencies.
    """
    du_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
