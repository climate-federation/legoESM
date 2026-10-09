"""Initialization for the lat-lon C-grid FV ocean model."""

from __future__ import annotations

import logging
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks


def partial_periodic_seam_wall_latlon(
    grid: LatLonGrid,
    open_lat_south_deg: float,
    open_lat_north_deg: float,
    seam_column_index: int = 0,
    base_mask: jnp.ndarray | None = None,
):
    """Build a per-cell land mask that creates a wall at one longitude
    column EVERYWHERE EXCEPT in a specified latitude band.

    The lat-lon C-grid operators wrap longitude periodically via
    ``jnp.roll``. To represent a closed basin with a re-entrant
    channel band (Drake passage analog), mark a single longitude
    column as land outside the open band — the periodic identification
    sees a wall there, except inside the band where the cells stay
    ocean.

    Parameters
    ----------
    grid : LatLonGrid
    open_lat_south_deg, open_lat_north_deg : float
        Latitude band [°N] in which the seam stays open (no wall).
    seam_column_index : int, default 0
        Longitude column index that hosts the wall. Defaults to the
        westernmost column (index 0); pick the easternmost (n_lon-1)
        for the same effect with periodic identification.
    base_mask : array, optional
        Pre-existing land mask (n_lat, n_lon). The seam wall is
        intersected with it.

    Returns
    -------
    land_mask : jax array, shape (n_lat, n_lon)
        1.0 = ocean, 0.0 = land.
    """
    lat_1d_deg = jnp.degrees(grid.lat)  # (n_lat,)
    in_open_band = (
        (lat_1d_deg >= open_lat_south_deg)
        & (lat_1d_deg <= open_lat_north_deg)
    )
    is_seam_col = jnp.zeros(grid.n_lon).at[seam_column_index].set(1.0)
    is_outside_band = jnp.where(in_open_band, 0.0, 1.0)
    seam_wall_2d = is_outside_band[:, None] * is_seam_col[None, :]
    if base_mask is None:
        land_mask = 1.0 - seam_wall_2d
    else:
        land_mask = jnp.where(base_mask > 0.5, 1.0 - seam_wall_2d, 0.0)
    return land_mask.astype(jnp.float32)


def idealized_bathymetry_latlon_cgrid(
    grid: LatLonGrid,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate idealized bathymetry and land mask on a lat-lon grid.

    Land is placed at high latitudes (|lat| > threshold).

    Parameters
    ----------
    grid : LatLonGrid
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude [degrees] above which cells are land.

    Returns
    -------
    H_bathy : array, shape (n_lat, n_lon)
    land_mask : array, shape (n_lat, n_lon)
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            f"land_lat_threshold must be in [0, 90], got {land_lat_threshold!r}",
        )

    dtype = get_policy().storage
    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

    return H_bathy, land_mask


def rest_state_latlon_cgrid_ocean(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
    land_mask_override: jnp.ndarray | None = None,
    H_bathy_override: jnp.ndarray | None = None,
    stratification: str = "exponential",
    nemo_prognostic_barotropic_velocity: bool = False,
    meridionally_periodic: bool = False,
) -> LatLonCGridOceanState:
    """Create a rest-state initial condition on a C-grid lat-lon grid.

    Temperature: exponential or linear profile (see ``stratification``).
    Salinity: uniform.
    Velocity: zero.
    Eta: zero.

    Parameters
    ----------
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    T_water_init_C, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land [degrees].  Ignored when
        *land_mask_override* is provided.
    land_mask_override : array (n_lat, n_lon), optional
        If provided, use this as the land mask (1=ocean, 0=land) instead
        of deriving one from *land_lat_threshold*.  Face masks (u_mask,
        v_mask) are computed from it automatically.
    H_bathy_override : array (n_lat, n_lon), optional
        If provided, use this as the per-cell bathymetry depth [m].
        When supplied together with *land_mask_override*, both are used
        as-is (caller is responsible for consistency between them).
        When supplied without *land_mask_override*, the land mask is
        derived from ``H_bathy_override > 0``.  When neither is given,
        a flat-bottom idealized bathymetry is constructed.
    stratification : str, default ``"exponential"``
        Initial vertical temperature profile:

        - ``"exponential"`` (default, BIT-IDENTICAL legacy):
          ``T(z) = T_deep + (T_water_init_C - T_deep) * exp(z / scale_depth)``.
        - ``"linear"``: ``T(z) = T_deep + (T_water_init_C - T_deep) *
          (1 - z_bottom_frac)`` where ``z_bottom_frac = z / z_bottom``,
          i.e. T varies linearly from ``T_water_init_C`` at the surface to
          ``T_deep`` at the deepest interface ``z = z_bottom``. This matches
          Veros ACC's ``T = (1 - z / z_bottom) * 15`` initial condition
          (``veros/setups/acc/acc.py:117``; ``T_deep=0`` there), which the
          exponential profile leaves ~1 °C too warm at the deepest cell.
    nemo_prognostic_barotropic_velocity : bool, default False
        Allocate NEMO key_RK3's separately carried U/V depth-mean velocity
        pair.  At rest, ``istate.F90:143-167`` initializes both Kbb fields to
        the depth mean of zero 3-D velocity and copies Kbb to Kmm, hence exact
        zero.  Non-NEMO recipes leave the pair as ``None`` and gain no array
        pytree leaves.
    meridionally_periodic : bool, default False
        Build the face masks under the y-wrap (the j-seam v-faces wet), as
        the model's ``config.meridionally_periodic`` steps them.  Default =
        the closed N/S walls.

    Returns
    -------
    LatLonCGridOceanState
    """
    from legoesm.grids.halo_latlon import meridional_periodicity
    with meridional_periodicity(meridionally_periodic):
        return _rest_state_body(
            grid, z_coord, T_water_init_C, T_deep, S_uniform, H_max,
            land_lat_threshold, land_mask_override, H_bathy_override,
            stratification, nemo_prognostic_barotropic_velocity)


def _rest_state_body(grid, z_coord, T_water_init_C, T_deep, S_uniform, H_max,
                     land_lat_threshold, land_mask_override, H_bathy_override,
                     stratification, nemo_prognostic_barotropic_velocity):
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    # Cast bathymetry/mask inputs to the active precision policy so that a
    # caller running under x32 does not silently get x64 fields (codex
    # adversarial review iter-1, bug #6).
    dtype = get_policy().storage
    if H_bathy_override is not None:
        H_bathy = jnp.asarray(H_bathy_override).astype(dtype)
        if land_mask_override is not None:
            land_mask = jnp.asarray(land_mask_override).astype(dtype)
        else:
            land_mask = (H_bathy > 0.0).astype(dtype)
    elif land_mask_override is not None:
        land_mask = jnp.asarray(land_mask_override).astype(dtype)
        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)
    else:
        H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
            grid, H_max, land_lat_threshold,
        )

    # Vertical T stratification: exponential (default, legacy) or linear.
    if stratification == "exponential":
        T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(
            z_coord.z_full_ref / _SCALE_DEPTH,
        )
    elif stratification == "linear":
        # T linear from T_water_init_C at the surface (z=0) to T_deep at the
        # deepest interface z_bottom (= z_half_ref[-1] = -H_max).
        # NOTE: this is NOT Veros's ACC initial condition — see
        # "veros_acc_linear" below (the earlier comment here claimed it was;
        # the transcription misread Veros's bottom-first ``zw[0]``).
        z_bottom = z_coord.z_half_ref[-1]
        frac = z_coord.z_full_ref / z_bottom          # 0 (surface) -> ~1 (bottom)
        T_profile = T_water_init_C + (T_deep - T_water_init_C) * frac
    elif stratification == "veros_acc_linear":
        # The LITERAL Veros ACC initial condition (veros/setups/acc/acc.py:117):
        #     temp = (1 - zt / zw[0]) * 15
        # Veros's arrays are BOTTOM-FIRST, so ``zw[0]`` is the TOP FACE of the
        # BOTTOM cell (~ -1724 m on the 15-level ACC grid), NOT the bottom
        # interface -H_max. The profile therefore goes slightly NEGATIVE in
        # the deepest cell (bottom-cell T ~ -1.15 C on the ACC grid) — a cold
        # abyss at t=0. The "linear" option above normalises by -H_max
        # instead, leaving the t=0 abyss ~2.15 K WARMER than Veros's; with the
        # ~3 Sv deep ventilation (century-scale fill time) that offset
        # persists essentially unreduced through any practical spin-up
        # (measured: it was the whole 30-yr ACC abyss bias, 2.33 vs 0.98 C;
        # with this profile the 5-yr abyss trajectories match to ~0.01 K).
        # ``T_deep`` is unused — Veros's formula has no independent deep
        # temperature. legoESM is surface-first: Veros zt = z_full_ref and
        # zw[0] = z_half_ref[-2] (the second-deepest interface).
        zw0 = z_coord.z_half_ref[-2]
        T_profile = T_water_init_C * (1.0 - z_coord.z_full_ref / zw0)
    else:
        raise ValueError(
            f"Unknown stratification={stratification!r}; expected "
            f"'exponential', 'linear' or 'veros_acc_linear'."
        )
    dtype = get_policy().storage
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    # C-grid velocity shapes
    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    # Face masks — consult ``grid.seam_wall_rows`` for a partial-periodic
    # seam wall (NEMO DINO); None on ordinary grids → fully periodic.
    u_mask, v_mask = compute_face_masks(land_mask, grid)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
        uu_b=(Field(data=jnp.zeros((n_lat, n_lon + 1), dtype=dtype),
                    name="uu_b", dims=dims_u2d, units="m/s",
                    staggering="edge")
              if nemo_prognostic_barotropic_velocity else None),
        vv_b=(Field(data=jnp.zeros((n_lat + 1, n_lon), dtype=dtype),
                    name="vv_b", dims=dims_v2d, units="m/s",
                    staggering="edge")
              if nemo_prognostic_barotropic_velocity else None),
    )


def wind_driven_gyre_latlon_cgrid(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> LatLonCGridOceanState:
    """Create initial condition for a wind-driven barotropic gyre on C-grid lat-lon.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    lon_deg = grid.lon2d * (180.0 / jnp.pi)
    lat_deg = grid.lat2d * (180.0 / jnp.pi)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

    T_3d = jnp.full((n_lat, n_lon, nlev), T_uniform, dtype=dtype)
    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def replace_land_mask(
    state: LatLonCGridOceanState,
    new_land_mask: jnp.ndarray,
    seam_wall_rows: jnp.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Replace land_mask and recompute u_mask/v_mask atomically.

    Use this instead of ``state._replace(land_mask=...)`` to ensure
    face masks stay consistent with the cell mask.  ``seam_wall_rows``
    (optional, ``(n_lat,)``, 1 = walled) closes the periodic-seam u-face
    on those rows for a partial-periodic geometry; ``None`` (default) =
    fully periodic (byte-identical).
    """
    new_land_mask = jnp.asarray(new_land_mask)
    u_mask, v_mask = compute_face_masks(new_land_mask, seam_wall_rows=seam_wall_rows)
    return state._replace(
        land_mask=Field(data=new_land_mask, name="land_mask",
                        dims=state.land_mask.dims, units=""),
        u_mask=Field(data=u_mask, name="u_mask",
                     dims=state.u_mask.dims, units=""),
        v_mask=Field(data=v_mask, name="v_mask",
                     dims=state.v_mask.dims, units=""),
    )


def apply_balanced_init(state, grid, z_coord, config,
                        taper_lat_deg=8.0, ref_depth_m=1500.0,
                        max_speed=2.5, with_ssh=True):
    """Initialise the cold-start in geostrophic / thermal-wind balance.

    The WOA cold-start blows up because it starts from REST (u=0) with a
    flat free surface (eta=0): the full baroclinic pressure-gradient force from
    WOA's density fronts is then UNBALANCED, and the violent geostrophic
    adjustment goes nonlinear (conclusively diagnosed -- the partial-cell PGF
    itself is NEMO-class, ~1e-6 m/s2 on a uniform-stratification rest test).

    This puts the flow in balance at t=0 so there is no adjustment shock:

    1. Baroclinic pressure anomaly ``p'`` from WOA T,S (surface-referenced),
       via the SAME ``iterate_eos_and_pressure_anomaly`` the dycore uses.
    2. LEVEL-OF-NO-MOTION reference: subtract the deepest-active ``p'_bottom``
       so the total pressure is flat at the seafloor (deep flow -> 0,
       surface-intensified ~1 m/s -- physical).  ``p_ref = p' - p'_bottom``.
    3. Geostrophic velocity from ``p_ref`` at cell centres,
       ``u_g = -(1/rho_0 f) dp_ref/dy``, ``v_g = +(1/rho_0 f) dp_ref/dx``,
       with the Coriolis singularity regularised near the equator
       ``1/f -> f/(f^2 + f_eps^2)`` (``f_eps = 2 Omega sin(taper_lat)`` -> the
       geostrophic velocity tapers smoothly to zero within ~|lat|<taper_lat).
       Mapped to the C-grid faces with ``cell_to_cgrid_winds`` (fold-aware).
    4. (with_ssh) Balanced free surface ``eta = -p'_bottom/(rho_0 g)`` (area-
       demeaned), so the dycore's total PGF ``-(1/rho_0) grad(p' + rho_0 g eta)
       = -(1/rho_0) grad(p_ref)`` exactly balances the geostrophic velocity.

    Pure IC change -- no dynamics-core modification.  Promoted from
    ``scripts/run/run_omip_core2.py`` so the coupled 3D-ocean driver can reuse
    the OMIP-validated cold-start balance (see omip_smag_cap_stabilizer /
    omip_rk3_coldstart_solve).
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        gradient_x_cgrid, gradient_y_cgrid, cell_to_cgrid_winds,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    from legoesm import constants

    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data
    rho_0 = float(config.rho_0)
    g_val = float(config.g)
    is_pc = isinstance(z_coord, OceanPartialCellCoordinate)
    eos_fn = make_eos_fn(config.eos, getattr(config, "eos_linear", None),
                         eos_nemo_seos=getattr(config, "eos_nemo_seos", None))
    h_actual = z_coord.h_partial if is_pc else None

    _, _, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask,
        lambda fld: neumann_fill_cgrid(fld, mask, grid=grid),
        eos_fn, z_coord.dz_ref, rho_0, g_val,
        n_iter=2, hi_precision_pressure=True, h_actual=h_actual,
    )                                                    # (n_lat, n_lon, nlev)

    # Reference level for the level-of-no-motion: a FIXED depth (~ref_depth_m)
    # common to all sufficiently-deep columns -- NOT the per-column seafloor
    # (whose depth varies with bathymetry, so a seafloor reference makes p_ref
    # and eta scale with DEPTH instead of the dynamic steric signal -> O(100 m)
    # spurious SSH).  Shallow columns reference their own bottom level.
    z_full = np.asarray(z_coord.z_full_ref)                 # (nlev,), negative
    k_ref = int(np.argmin(np.abs(z_full + ref_depth_m)))    # level nearest ref_depth
    if is_pc:
        bl = jnp.clip(z_coord.bottom_level, 0, z_coord.n_levels - 1)
        k_use = jnp.minimum(bl, k_ref)                      # (n_lat, n_lon)
    else:
        k_use = jnp.full(p_prime.shape[:-1], k_ref, dtype=jnp.int32)
    p_at_ref = jnp.take_along_axis(p_prime, k_use[..., None], axis=-1)  # (...,1)
    p_ref = p_prime - p_at_ref                              # 0 at the reference

    p_ref_filled = neumann_fill_cgrid(p_ref, mask, grid=grid)
    gx_u = gradient_x_cgrid(p_ref_filled, grid)            # (n_lat, n_lon+1, nlev)
    gy_v = gradient_y_cgrid(p_ref_filled, grid)            # (n_lat+1, n_lon, nlev)
    gx_T = 0.5 * (gx_u[:, :-1] + gx_u[:, 1:])              # (n_lat, n_lon, nlev)
    gy_T = 0.5 * (gy_v[:-1] + gy_v[1:])

    # grid-agnostic Coriolis: tripole LatLonCGridGeometry -> f_T, plain
    # LatLonGrid -> f; both expose the grid_coriolis @property -> (n_lat, n_lon).
    f_T = grid.grid_coriolis                                # (n_lat, n_lon)
    f_eps = 2.0 * constants.Omega * float(np.sin(np.deg2rad(taper_lat_deg)))
    inv_f = (f_T / (f_T ** 2 + f_eps ** 2))[..., None]     # -> 0 at the equator

    u_g = -(1.0 / rho_0) * inv_f * gy_T
    v_g = +(1.0 / rho_0) * inv_f * gx_T
    # Safety clip: geostrophy is invalid in the (tapered) equatorial band and
    # at any residual sharp IC front (e.g. flood-fill seams); bound the speed
    # to a physical maximum so those cells start bounded rather than at
    # tens of m/s.  Mid-latitude balanced flow is well below this.
    u_g = jnp.clip(u_g, -max_speed, max_speed)
    v_g = jnp.clip(v_g, -max_speed, max_speed)
    m3 = mask[..., None]
    if is_pc:
        m3 = m3 * z_coord.is_active.astype(m3.dtype)
    u_g = u_g * m3
    v_g = v_g * m3

    u_face, v_face = cell_to_cgrid_winds(u_g, v_g, grid)
    u_face = u_face * state.u_mask.data[..., None]
    u_face = u_face.at[:, -1].set(u_face[:, 0])            # periodic wrap column
    v_face = v_face * state.v_mask.data[..., None]

    repl = dict(
        u=state.u.replace(data=u_face),
        v=state.v.replace(data=v_face),
    )
    if with_ssh:
        eta = -p_at_ref[..., 0] / (rho_0 * g_val)          # (n_lat, n_lon)
        area = grid.area * mask   # grid-agnostic (both grids expose .area)
        eta_mean = jnp.sum(eta * area) / jnp.maximum(jnp.sum(area), 1.0)
        eta = jnp.clip(eta - eta_mean, -5.0, 5.0) * mask   # physical SSH bound
        repl["eta"] = state.eta.replace(data=eta)
    return state._replace(**repl)


def make_partial_cell_latlon(z_coord, H_bathy, land_mask, thin_threshold=0.3,
                             smoothing_passes=4, min_levels=2):
    """Build an ``OceanPartialCellCoordinate`` from a z* coord + bathymetry,
    with bathymetry smoothing + thin-cell snap + shallow-column masking.

    Promoted from ``scripts/run/run_omip_core2.py:make_partial_cell`` so the
    coupled 3D-ocean driver can reuse the OMIP-validated WOA-cold-start geometry.

    Why each step matters for the WOA cold start (all empirically required — the
    flat-bottom z-star coord blows up; this geometry survives, max|u|~1 m/s):

    * ``smoothing_passes`` Laplacian passes on the OCEAN ``H_bathy`` (land held
      fixed) reduce the bathymetric slope (r-factor ``|H_i-H_j|/(H_i+H_j)``);
      the spurious partial-cell PGF seed that blows up the cold start scales with
      the slope (NEMO/ROMS smooth bathymetry for exactly this reason).
    * the thin-cell snap snaps ``H_bathy`` DOWN to the interface above whenever
      the bottom partial cell would be thinner than ``thin_threshold*dz_ref``
      (MOM6/MITgcm thin-cell fix) — a tiny bottom cell is a ``1/h`` instability
      seed.
    * ``min_levels`` masks ocean columns with fewer than that many active
      reference levels as land (single-thin-level coastal cells amplify a tiny
      smc03 PGF residual into a blow-up).

    The returned ``OceanPartialCellCoordinate`` (vs a plain ``OceanZStarCoordinate``)
    also ACTIVATES the smc03 density-Jacobian PGF in ``ocean_pe_latlon_cgrid`` —
    a plain z-star coord silently falls back to the centered-difference PGF
    whose truncation error on the sharp WOA pycnocline seeds the blow-up.

    Returns ``(z_coord_partial, H_snapped, land_mask_out)``.
    """
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    from legoesm.ocean.bathymetry import laplacian_smooth_2d, compute_max_r_factor

    H_np = np.asarray(H_bathy, dtype=np.float64)
    lm0 = np.asarray(land_mask, dtype=np.float64)
    ocean = lm0 > 0.5

    if smoothing_passes and smoothing_passes > 0:
        r_before = float(compute_max_r_factor(H_np, lm0))
        H_s = H_np.copy()
        for _ in range(int(smoothing_passes)):
            H_sm = np.asarray(laplacian_smooth_2d(H_s, 1, is_cubed=False))
            H_s = np.where(ocean, H_sm, H_np)
        H_np = np.where(ocean, H_s, H_np)
        logging.getLogger("legoesm.ocean").info(
            "  partial-cell bathy smoothing: %d passes, max r-factor %.3f -> %.3f",
            smoothing_passes, r_before, float(compute_max_r_factor(H_np, lm0)))

    abs_z_half = np.abs(np.asarray(z_coord.z_half_ref))
    dz_ref_np = np.asarray(z_coord.dz_ref)
    H_snapped = H_np.copy()
    for k in range(z_coord.n_levels):
        top, bot = abs_z_half[k], abs_z_half[k + 1]
        in_layer = (H_np > top) & (H_np <= bot)
        too_thin = in_layer & ((H_np - top) < thin_threshold * dz_ref_np[k])
        H_snapped = np.where(too_thin, top, H_snapped)

    new_land = (H_snapped <= 0.0) & ocean
    if min_levels and int(min_levels) > 1:
        n_active = (abs_z_half[None, None, :z_coord.n_levels]
                    < H_snapped[..., None]).sum(axis=2)
        too_shallow = (n_active < int(min_levels)) & ocean & (H_snapped > 0.0)
        H_snapped = np.where(too_shallow, 0.0, H_snapped)
        new_land = new_land | too_shallow
    lm_out = np.where(new_land, 0.0, lm0)

    zc = create_partial_cell_coordinate(
        z_coord, jnp.asarray(H_snapped, dtype=jnp.float64))
    return zc, H_snapped.astype(np.float64), lm_out.astype(np.float64)
