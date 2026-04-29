"""Shared helpers for physics parameterization integration bridges.

Functions here are used by multiple physics packages (GWD, microphysics,
turbulence) to convert between prognostic model variables and the
column-physics inputs each scheme expects.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


# ---------------------------------------------------------------------------
# Height / thickness from hydrostatic balance
# ---------------------------------------------------------------------------

def compute_heights_from_sigma(T, p_half):
    """Approximate full- and half-level heights from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.

    Returns
    -------
    z_full : array (ncol, nlev)
        Height at full levels [m].
    z_half : array (ncol, nlev+1)
        Height at half levels [m] (surface = 0).
    """
    ncol, nlev = T.shape

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(
        constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    )

    # Integrate from surface upward.  Use ``jnp.pad`` to append the
    # surface (z=0) boundary instead of allocating a fresh
    # ``jnp.zeros((ncol, 1))`` and concatenating — single Pad HLO op
    # vs alloc + concat (this helper is invoked by GWD / microphysics
    # / turbulence integrations every physics step).
    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)
    z_half_inner = z_half_cumsum[:, ::-1]
    z_half = jnp.pad(z_half_inner, ((0, 0), (0, 1)))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return z_full, z_half


def compute_layer_dz(T, p_half):
    """Approximate layer thicknesses from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.

    Returns
    -------
    dz : array (ncol, nlev)
        Layer thickness [m].
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    return jnp.abs(
        constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    )


# ---------------------------------------------------------------------------
# Density from ideal-gas law
# ---------------------------------------------------------------------------

def compute_rho(T, p_full):
    """Compute air density from the ideal gas law: rho = p / (R_d * T).

    Parameters
    ----------
    T : array
        Temperature [K].
    p_full : array
        Pressure at full levels [Pa].

    Returns
    -------
    array : Density [kg/m^3].
    """
    return p_full / (constants.R_d * jnp.clip(T, 1.0, None))


# ---------------------------------------------------------------------------
# Hydrostatic column extraction helpers (shared across physics integration bridges)
# ---------------------------------------------------------------------------

def extract_hydrostatic_columns(state, sigma_coord):
    """Extract T, p_s, pressures, and q_v from a HydrostaticState as columns.

    Parameters
    ----------
    state : HydrostaticState
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    dict with keys: T_col, p_full_col, p_half_col, q_v_col, ncol, nlev,
        shape_3d, shape_2d
    """
    from legoesm.grids.vertical import pressure_from_sigma

    T = state.T.data
    p_s = state.p_s.data
    nlev = sigma_coord.n_levels
    shape_3d = T.shape
    shape_2d = p_s.shape

    p_full = sigma_coord.pressure_at_full(p_s)
    p_half = sigma_coord.pressure_at_half(p_s)

    ncol = 1
    for s in shape_2d:
        ncol *= s

    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)

    # Extract water vapor
    q_v_col = jnp.zeros((ncol, nlev))
    if state.tracers is not None and "q_v" in state.tracers:
        _qv_raw = state.tracers["q_v"]
        _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        q_v_col = _qv_data.reshape(ncol, nlev)

    return dict(
        T_col=T_col, p_full_col=p_full_col, p_half_col=p_half_col,
        q_v_col=q_v_col, ncol=ncol, nlev=nlev,
        shape_3d=shape_3d, shape_2d=shape_2d,
    )


def extract_nonhydrostatic_columns(state, height_coord, terrain_metric):
    """Extract T, pressures, q_v from a NonHydrostaticState as columns.

    Parameters
    ----------
    state : NonHydrostaticState
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric

    Returns
    -------
    dict with keys: T_col, T, p_full_col, p_half_col, q_v_col, rho_col,
        ncol, nlev, shape_3d, shape_2d, shape_w, tracers, n_tracers,
        exner, theta_total, rho_total, z_full, z_half
    """
    from legoesm.atmosphere.physics.thermodynamics import (
        pressure_from_eos, reconstruct_half_level_pressure_hydrostatic,
        sanitize_theta_rho,
    )

    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data

    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    theta_total, rho_total = sanitize_theta_rho(theta_0 + theta_p, rho_0 + rho_p)

    p = pressure_from_eos(rho_total, theta_total)
    exner = (p / constants.p_ref) ** constants.kappa
    T = theta_total * exner

    nlev = height_coord.n_levels
    shape_3d = theta_p.shape
    shape_w = state.w.data.shape
    shape_2d = state.phis.data.shape
    n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

    ncol = 1
    for s in shape_2d:
        ncol *= s

    z_full = terrain_metric.z_full_3d.reshape(ncol, nlev)
    z_half = terrain_metric.z_half_3d.reshape(ncol, nlev + 1)
    p_half = reconstruct_half_level_pressure_hydrostatic(
        p_full=p, rho_full=rho_total, z_half=terrain_metric.z_half_3d,
    ).reshape(ncol, nlev + 1)

    T_col = T.reshape(ncol, nlev)
    p_full_col = p.reshape(ncol, nlev)
    rho_col = rho_total.reshape(ncol, nlev)

    q_v_col = jnp.zeros((ncol, nlev))
    if n_tracers > 0:
        q_v_col = tracers[..., 0].reshape(ncol, nlev)

    return dict(
        T_col=T_col, T=T, p_full_col=p_full_col, p_half_col=p_half,
        q_v_col=q_v_col, rho_col=rho_col,
        ncol=ncol, nlev=nlev,
        shape_3d=shape_3d, shape_2d=shape_2d, shape_w=shape_w,
        tracers=tracers, n_tracers=n_tracers,
        exner=exner, theta_total=theta_total, rho_total=rho_total,
        z_full=z_full, z_half=z_half,
    )


def extract_spectral_pe_columns(fields, sigma_coord, state=None):
    """Extract T, pressures, q_v from spectral PE grid fields as columns.

    Parameters
    ----------
    fields : dict
        Grid-space fields from spectral_pe_to_grid (keys: T, p_s, u, v, ...).
    sigma_coord : SigmaCoordinate
    state : SpectralHydrostaticState, optional
        If provided, extracts q_v from state.tracers.

    Returns
    -------
    dict with keys: T_col, p_full_col, p_half_col, q_v_col, ncol, nlev,
        n_lat, n_lon, T, p_s
    """
    T = fields['T']
    p_s = fields['p_s']
    nlev = sigma_coord.n_levels
    n_lat, n_lon = p_s.shape

    sigma_full = sigma_coord.sigma_full
    sigma_half = sigma_coord.sigma_half
    p_full = p_s[..., None] * sigma_full
    p_half = p_s[..., None] * sigma_half

    ncol = n_lat * n_lon
    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)

    q_v_col = jnp.zeros((ncol, nlev))
    if state is not None and hasattr(state, "tracers") and state.tracers is not None and "q_v" in state.tracers:
        _qv_raw = state.tracers["q_v"]
        _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        q_v_col = _qv_data.reshape(ncol, nlev)

    return dict(
        T_col=T_col, p_full_col=p_full_col, p_half_col=p_half_col,
        q_v_col=q_v_col, ncol=ncol, nlev=nlev,
        n_lat=n_lat, n_lon=n_lon, T=T, p_s=p_s,
    )


def make_zero_hydrostatic_tendencies(shape_3d, shape_2d, prefix=""):
    """Create zero HydrostaticTendencies with a naming prefix.

    Parameters
    ----------
    shape_3d : tuple
        Shape for 3D fields.
    shape_2d : tuple
        Shape for 2D fields.
    prefix : str
        Name prefix for fields (e.g. "conv", "turb").

    Returns
    -------
    HydrostaticTendencies
    """
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticTendencies

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    sfx = f"_{prefix}" if prefix else ""
    return HydrostaticTendencies(
        du_dt=Field(data=jnp.zeros(shape_3d), name=f"du_dt{sfx}", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=jnp.zeros(shape_3d), name=f"dv_dt{sfx}", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=jnp.zeros(shape_3d), name=f"dT_dt{sfx}", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=jnp.zeros(shape_2d), name=f"dp_s_dt{sfx}", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(data=jnp.zeros(shape_2d), name=f"dphis_dt{sfx}", dims=dims_2d, units="m^2/s^3"),
    )


def make_zero_nonhydrostatic_tendencies(shape_3d, shape_2d, shape_w, tracers, prefix=""):
    """Create zero NonHydrostaticTendencies with a naming prefix.

    Parameters
    ----------
    shape_3d : tuple
        Shape for 3D fields.
    shape_2d : tuple
        Shape for 2D fields.
    shape_w : tuple
        Shape for w field (half levels).
    tracers : jax.Array
        Tracer array (for shape of dtracers_dt).
    prefix : str
        Name prefix for fields.

    Returns
    -------
    NonHydrostaticTendencies
    """
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticTendencies

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")
    sfx = f"_{prefix}" if prefix else ""
    return NonHydrostaticTendencies(
        du_dt=Field(data=jnp.zeros(shape_3d), name=f"du_dt{sfx}", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=jnp.zeros(shape_3d), name=f"dv_dt{sfx}", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=jnp.zeros(shape_w), name=f"dw_dt{sfx}", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(data=jnp.zeros(shape_3d), name=f"dtheta_prime_dt{sfx}", dims=dims_3d, units="K/s"),
        drho_prime_dt=Field(data=jnp.zeros(shape_3d), name=f"drho_prime_dt{sfx}", dims=dims_3d, units="kg/m^3/s"),
        dphis_dt=Field(data=jnp.zeros(shape_2d), name=f"dphis_dt{sfx}", dims=dims_2d, units="m^2/s^3"),
        dtracers_dt=Field(data=jnp.zeros_like(tracers), name=f"dtracers_dt{sfx}", dims=dims_tr, units="1/s"),
    )


# ---------------------------------------------------------------------------
# Moisture-convergence diagnostic for Tiedtke / Bechtold closures
# ---------------------------------------------------------------------------

def compute_moisture_convergence(
    q_v_grid: jnp.ndarray,
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """Per-level horizontal moisture convergence ``MC = -∇·(q_v * u)``.

    Used by the Tiedtke / Bechtold convection schemes as the deep
    closure's mass-flux driver.  Implementation reuses the dycore's
    finite-volume tracer-flux divergence operator
    (:func:`legoesm.core.operators_3d.fv_flux_divergence_3d` for cubed
    sphere; :func:`legoesm.core.operators_fv_latlon_3d.fv_flux_divergence_latlon_3d`
    for lat-lon) and negates the result.

    Parameters
    ----------
    q_v_grid : jax.Array
        Water-vapor specific humidity in grid-shape:

        * cubed sphere — ``(face, n, n, nlev)``
        * lat-lon C-grid — ``(n_lat, n_lon, nlev)``
    u_grid, v_grid : jax.Array
        Horizontal wind components, same grid-shape as ``q_v_grid``.
    grid : CubedSphereGrid or LatLonGrid
        Discretized grid metadata used by the underlying divergence
        operator.

    Returns
    -------
    jax.Array, shape (ncol, nlev)
        Column-flattened moisture convergence [kg/kg/s].  Positive
        values indicate net moisture inflow to the column.

    Notes
    -----
    Uses the FV-flux-divergence operator with the slope limiter
    *disabled* — we want a smooth, fully-differentiable diagnostic, and
    the limiter introduces non-smooth ``where``-style branching that
    would break ``jax.grad`` through the convection trigger.  Tracer
    advection in the dycore proper still uses the limiter; this is a
    closure diagnostic, not an advected quantity.
    """
    from legoesm.grids.cubed_sphere import CubedSphereGrid

    if isinstance(grid, CubedSphereGrid):
        from legoesm.core.operators_3d import fv_flux_divergence_3d
        # q_v_grid shape (6, n, n, nlev)
        flux_div = fv_flux_divergence_3d(
            q_v_grid, u_grid, v_grid, grid, limiter=False,
        )
        # Reshape to (ncol, nlev)
        face, n, _, nlev = q_v_grid.shape
        return -flux_div.reshape(face * n * n, nlev)

    # Try lat-lon — duck-typed by attribute presence so we don't
    # introduce an import dependency for users who never touch lat-lon.
    if hasattr(grid, "dlat") and hasattr(grid, "dlon"):
        from legoesm.core.operators_fv_latlon_3d import (
            fv_flux_divergence_latlon_3d,
        )
        flux_div = fv_flux_divergence_latlon_3d(
            q_v_grid, u_grid, v_grid, grid, limiter=False,
        )
        n_lat, n_lon, nlev = q_v_grid.shape
        return -flux_div.reshape(n_lat * n_lon, nlev)

    raise TypeError(
        f"compute_moisture_convergence: unsupported grid type "
        f"{type(grid).__name__!r}.  Supported: CubedSphereGrid, LatLonGrid."
    )
