"""State containers for legoESM components.

All states are registered as JAX pytrees so they work seamlessly with
jit, grad, vmap, scan, and checkpoint.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable


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


# ==============================================================================
# Hydrostatic Primitive Equations (Milestone 2)
# ==============================================================================

class HydrostaticState(NamedTuple):
    """State for the hydrostatic primitive equations in sigma coordinates.

    Unified state type for all hydrostatic grids: cubed-sphere, lat-lon,
    and MPAS Voronoi mesh.  Field shapes depend on the grid type:

    - Cubed-sphere: 3D = (6, n, n, nlev), 2D = (6, n, n)
    - Lat-lon: 3D = (n_lat, n_lon, nlev), 2D = (n_lat, n_lon)
    - MPAS: u = (nEdges, nlev), T/p_s/phis = (nCells, nlev)/(nCells,)

    Fields
    ------
    u : Field
        Wind component [m/s]. Prognostic.
        Cubed-sphere/lat-lon: zonal wind on cell centres.
        MPAS: normal velocity on edges.
    T : Field
        Temperature [K]. Prognostic.
    p_s : Field
        Surface pressure [Pa]. Prognostic.
    phis : Field
        Surface geopotential [m^2/s^2]. Static (not time-stepped).
    v : Field | None
        Meridional wind [m/s]. Prognostic.
        None for MPAS (edge-based normal velocity has no separate v).
    tracers : dict[str, Field] | None
        Tracer mixing ratios as a dictionary. Keys are tracer names
        (e.g., "q_v", "q_c", "q_r"). None when tracers are not in use.
    """
    u: Field
    T: Field
    p_s: Field
    phis: Field
    v: Field | None = None
    tracers: dict[str, Field] | None = None


class HydrostaticTendencies(NamedTuple):
    """Tendencies (time derivatives) for the hydrostatic primitive equations.

    Unified tendency type for all hydrostatic grids. The phis tendency is
    always zero since surface geopotential is static.

    Fields
    ------
    du_dt : Field
        Wind tendency [m/s^2]. Cubed-sphere/lat-lon: zonal wind. MPAS: normal velocity on edges.
    dT_dt : Field
        Temperature tendency [K/s].
    dp_s_dt : Field
        Surface pressure tendency [Pa/s].
    dphis_dt : Field
        Geopotential tendency [m^2/s^3] — always zero.
    dv_dt : Field | None
        Meridional wind tendency [m/s^2]. None for MPAS.
    tracer_tendencies : dict[str, Field] | None
        Optional tracer tendencies from physics (convection, microphysics).
        Keys match HydrostaticState.tracers (e.g. "q_v", "q_c", "q_r").
        None when no physics scheme produces tracer tendencies.
    """
    du_dt: Field
    dT_dt: Field
    dp_s_dt: Field
    dphis_dt: Field
    dv_dt: Field | None = None
    tracer_tendencies: dict[str, Field] | None = None
    # Surface radiative net fluxes [W/m^2, +into surface], carried on the
    # radiation tendency so the lean MPAS/spectral loops can export them to the
    # coupler (the compiled cube/latlon path exports the equivalent via
    # PhysicsOutput; the lean HydrostaticTendencies had no such channel, which
    # left the coupled-voronoi ocean/land tiles forced with zero shortwave).
    # None on non-radiation tendencies and when radiation is inactive; added at
    # the end with None defaults so every existing constructor is unaffected.
    sw_net_sfc: Field | None = None
    lw_net_sfc: Field | None = None
    # Surface precipitation [kg/m^2/s, +into surface], carried on the
    # microphysics tendency so the lean MPAS/spectral loops can export it to the
    # coupler's ocean P-E / land forcing (the compiled path exports the
    # equivalent via PhysicsOutput.precip). None when microphysics is inactive;
    # this is the SURFACE precip (micro sedimentation), NOT the column vapour
    # sink (which is P-E and would double-count the separately-applied evap).
    precip: Field | None = None
    # TOA radiative fluxes (CMOR sign conventions: *_up positive UPWARD /
    # outgoing, sw_down_toa positive DOWNWARD / incoming) [W/m^2], carried on
    # the radiation tendency so the lean MPAS loop can feed the CMOR
    # rlut/rsut/rsdt accumulators (PhysicsOutput carries the equivalent on the
    # compiled path). None on non-radiation tendencies / radiation off.
    sw_up_toa: Field | None = None
    lw_up_toa: Field | None = None
    sw_down_toa: Field | None = None
    # Surface turbulent fluxes [W/m^2, positive UPWARD out of the surface —
    # the CMOR hfss/hfls convention, matching the surface-layer helpers'
    # shflx/lhflx sign], carried on the turbulence tendency for the same CMOR
    # feed (evspsbl is derived downstream as lhflx / L_v). None when
    # turbulence is off or a scheme computes no surface fluxes.
    shflx_sfc: Field | None = None
    lhflx_sfc: Field | None = None


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
    tracer_tendencies: dict[str, Field] | None = None
    # Optional advective-form tracer tendencies (6, n, n, nlev) per key, keyed
    # like FV3HydrostaticState.tracers (e.g. "q_v"/"q_c"/"q_r").  None when the
    # state carries no tracers (dry dycore).  Transported consistently with T
    # (the FV3 hydrostatic core advects scalars in advective form).


# PhysicsState is defined in legoesm.atmosphere.physics.physics_state
# (authoritative, concrete-array version).  Import from there directly:
#   from legoesm.atmosphere.physics.physics_state import PhysicsState


# ==============================================================================
# Physics Module Protocol
# ==============================================================================

@runtime_checkable
class PhysicsModuleProtocol(Protocol):
    """Interface contract for all physics parameterization modules.

    Every physics factory (make_radiation_physics, make_turbulence_physics, etc.)
    must return a callable matching this protocol.  The return is a 2-tuple
    of (tendencies, updated_phys_state).
    """
    def __call__(
        self,
        state: HydrostaticState,
        grid,
        coord,
        phys_state=None,
    ) -> tuple: ...

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

# Backward-compatibility aliases — MPAS uses the unified types with v=None.
MPASHydrostaticState = HydrostaticState
MPASHydrostaticTendencies = HydrostaticTendencies


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
# Doubly-periodic Cartesian plane non-hydrostatic state (CRM rollout)
# ==============================================================================
#
# Staged-not-integrated: this state pytree is the prognostic container for
# the future plane non-hydrostatic dycore (CRM rollout, PR2b). PR2a ships
# the state class plus mass-integral helpers consumed by the column-kernel
# refactor; the dycore body that produces and consumes the state lands in
# PR2b. Until then no public factory dispatches into it.
#
# Convention
# ----------
# Vertical-LAST axis layout, matching ``NonHydrostaticState`` (cubed-sphere)
# and ``MPASNonHydrostaticState``. All ``Field`` wrappers carry ``.data``
# JAX arrays of the listed shapes; tendency objects share the same pytree
# shape so ``jax.tree_util.tree_map`` works through SSP-RK3 averaging.
#
# Arakawa-C staggering (matches PR1 ``PlaneGrid`` + ``plane_operators``):
#   - scalar / cell centre: ``(ny, nx, nlev)``
#   - u at x-faces: ``(ny, nx, nlev)`` — no duplicated periodic endpoint
#   - v at y-faces: ``(ny, nx, nlev)`` — no duplicated periodic endpoint
#   - w at z-interfaces (Lorenz): ``(ny, nx, nlev+1)``
#   - phis: ``(ny, nx)``
#   - tracers: ``(ny, nx, nlev, n_tracers)``

class PlaneNonHydrostaticState(NamedTuple):
    """State for the non-hydrostatic compressible Euler equations on a
    doubly-periodic Cartesian plane.

    Uses reference-state subtraction: prognostic variables are
    perturbations from a 1D hydrostatically balanced reference state
    ``rho_0(z)``, ``theta_0(z)``.

    All horizontal arrays have shape ``(ny, nx, ...)`` with no
    duplicated periodic endpoint — periodic neighbours come from
    ``jnp.roll`` / ``jnp.pad(..., mode='wrap')`` in
    ``plane_operators``.

    Fields
    ------
    u : Field
        Zonal wind [m/s] at x-faces (Arakawa-C).
        Shape ``(ny, nx, nlev)``.
    v : Field
        Meridional wind [m/s] at y-faces (Arakawa-C).
        Shape ``(ny, nx, nlev)``.
    w : Field
        Vertical velocity [m/s] at half (interface) levels.
        Shape ``(ny, nx, nlev+1)``. Lorenz staggering.
        Rigid boundary conditions: ``w = 0`` at model top and bottom
        (the plane is flat — surface geopotential ``phis`` is zero).
    theta_prime : Field
        Potential temperature perturbation [K]. ``theta' = theta - theta_0(z)``.
        Shape ``(ny, nx, nlev)``.
    rho_prime : Field
        Dry density perturbation [kg/m^3]. ``rho' = rho - rho_0(z)``.
        Shape ``(ny, nx, nlev)``.
    phis : Field
        Surface geopotential [m^2/s^2]. Static (not time-stepped).
        Shape ``(ny, nx)``. Zero on a flat plane.
    tracers : Field
        Tracer mixing ratios [kg/kg]. Shape ``(ny, nx, nlev, n_tracers)``.
        PR2a accepts ``n_tracers == 0`` only (empty last axis). Tracer
        transport on the plane lands in PR3 together with microphysics.
    """
    u: Field
    v: Field
    w: Field
    theta_prime: Field
    rho_prime: Field
    phis: Field
    tracers: Field


# iter-241: cherry-picked from feature/crm-plane-spectral commit
# edbae138 ("Spectral plane CRM: state pytree + filter wrapper around
# FD dycore", 2026-05-24). That branch was never merged into main,
# leaving src/legoesm/atmosphere/dynamics/les/spectral_plane.py with
# broken ``from legoesm.core.state import SpectralPlanePhysicsState,
# SpectralPlanePhysicsTendencies`` imports. The two pytree classes
# below are the minimum required to unblock the spectral_plane
# import + restore plane_spectral coverage in
# tests/atmosphere/nonhydrostatic/integration/test_run_rcemip_long_cross_grid_smoke.py
# (which still has an xfail-strict marker that will fire as XPASS
# once the runtime ``float(jnp.log(100.0))`` Metal bug in
# spectral_pe.py is also resolved).
class SpectralPlanePhysicsState(NamedTuple):
    """Spectral (2D-Fourier xy + physical z) plane non-hydrostatic state.

    Pseudo-spectral counterpart of :class:`PlaneNonHydrostaticState`.
    Horizontal axes (y, x) are stored as ``rfft2`` complex coefficients
    of shape ``(ny, nx_r)`` with ``nx_r = nx // 2 + 1``; the vertical
    axis is unchanged (physical-space full levels at cell centres,
    half levels for ``w``).

    Fields
    ------
    u_hat, v_hat : Field
        Horizontal-Fourier coefficients of zonal/meridional wind
        components at full levels, shape ``(ny, nx_r, nlev)`` complex.
        Stored at the Arakawa-C ``u``-face / ``v``-face location in
        physical space — but the rfft2 of a face-staggered field is
        the same as the rfft2 of the cell-centred field (the
        face/cell distinction in physical space disappears in the
        spectral representation because the FFT basis functions are
        already located at every position). The factor-of-``i·kx`` /
        ``i·ky`` operators applied below ARE the C-grid PG/divergence
        adjoint pair.
    w_hat : Field
        Vertical velocity coefficients at half levels, shape
        ``(ny, nx_r, nlev+1)`` complex.
    theta_prime_hat, rho_prime_hat : Field
        Perturbation potential temperature / density at full levels,
        shape ``(ny, nx_r, nlev)`` complex.
    phis : Field
        Surface geopotential — kept PHYSICAL (real) and static, shape
        ``(ny, nx)``. Zero on a flat plane; included for parity with
        the FD plane state.
    tracers_hat : Field
        Tracer mixing-ratio coefficients, shape
        ``(ny, nx_r, nlev, n_tracers)`` complex.
    """
    u_hat: Field
    v_hat: Field
    w_hat: Field
    theta_prime_hat: Field
    rho_prime_hat: Field
    phis: Field
    tracers_hat: Field


class SpectralPlanePhysicsTendencies(NamedTuple):
    """Tendencies for the spectral plane non-hydrostatic equations.

    Same pytree shape as :class:`SpectralPlanePhysicsState` so
    ``jax.tree_util.tree_map`` works for SSP-RK3 averaging.
    """
    du_hat_dt: Field
    dv_hat_dt: Field
    dw_hat_dt: Field
    dtheta_prime_hat_dt: Field
    drho_prime_hat_dt: Field
    dphis_dt: Field
    dtracers_hat_dt: Field


class PlaneNonHydrostaticTendencies(NamedTuple):
    """Tendencies (time derivatives) for the plane non-hydrostatic equations.

    Same pytree structure as :class:`PlaneNonHydrostaticState` so
    ``jax.tree_util.tree_map`` works for SSP-RK3 averaging and physics
    coupling. ``dphis_dt`` is always zero because the plane surface is
    static; it is kept for pytree shape compatibility with the existing
    NH tendency interface.
    """
    du_dt: Field
    dv_dt: Field
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
    w : Field
        Vertical velocity [m/s]. Shape (nCells, nlev+1). Diagnostic field
        computed from flux divergence on half levels (surface first,
        bottom = 0).
    H_bathy : Field
        Bathymetry depth [m]. Shape (nCells,). Positive downward. Static.
    land_mask : Field
        Ocean mask. Shape (nCells,). 1=ocean, 0=land. Static.
    rho_ref_z : Field or None
        Optional horizontally-uniform reference density profile [kg/m³].
        Shape (nlev,). Frozen at init time from ``EOS(T_init, S_init,
        p_hydro)`` averaged over wet cells.  When present, the
        baroclinic PGF uses ``ρ' = ρ − ρ_ref(z)`` instead of
        ``ρ' = ρ − ρ_0``, attacking the partial-cell PGF residual at
        the seed (project_mpas_etopo_instability.md §"Option B").
        Static — never updated during integration.  None disables.
    """
    u: Field
    T: Field
    S: Field
    eta: Field
    w: Field
    H_bathy: Field
    land_mask: Field
    rho_ref_z: Field | None = None


class MPASOceanTendencies(NamedTuple):
    """Tendencies for MPAS ocean primitive equations.

    F_slow_u is the depth-mean of the full nonlinear momentum tendency
    (Coriolis via PV flux + PGF + KE).  Passed to the barotropic solver
    as slow forcing (MOM6 pattern, Hallberg & Adcroft 2009).  Issue #160.
    """
    du_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    F_slow_u: Field | None = None
