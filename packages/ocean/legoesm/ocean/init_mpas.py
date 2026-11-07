"""Initialization routines for MPAS ocean on Voronoi meshes.

Provides idealized bathymetry, rest-state initial conditions,
and velocity reconstruction utilities.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.core.state import MPASOceanState
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
)


def partial_periodic_seam_wall_mpas(
    mesh: VoronoiMesh,
    open_lat_south_deg: float,
    open_lat_north_deg: float,
    seam_lon_deg: float | None = None,
    seam_strip_width_deg: float | None = None,
    base_mask: jnp.ndarray | None = None,
):
    """Build a per-cell land mask that creates a wall along a periodic
    seam everywhere EXCEPT in a specified latitude band.

    Use case: a regional Voronoi mesh constructed with
    ``periodic_x=True`` represents a basin whose east-west boundaries
    should be CLOSED everywhere except in a re-entrant channel band
    (Drake passage, Bering strait, ACC channel, etc.). Without
    intervention the periodicity makes flow wrap around at every
    latitude, including the subtropical / subpolar gyre region — which
    is non-physical for a closed basin. This helper marks cells within
    one cell-width of the periodic seam as LAND outside the open band,
    creating a wall there while leaving the band open.

    Parameters
    ----------
    mesh : VoronoiMesh
        Created by ``create_regional_voronoi_mesh(..., periodic_x=True)``.
    open_lat_south_deg, open_lat_north_deg : float
        Latitude band [°N] in which the seam stays open (no wall).
    seam_lon_deg : float, optional
        Longitude [°E] of the periodic seam (= ``lon_west`` of the
        regional mesh). Defaults to the minimum lon of the mesh.
    seam_strip_width_deg : float, optional
        Width of the wall strip in degrees of longitude. Defaults to
        one nominal cell width estimated from the mesh's median
        ``dcEdge``. Set explicitly for reproducibility.
    base_mask : array, optional
        Optional pre-existing land mask (e.g., to also mask buffer
        cells outside a lat-domain). The seam wall is intersected
        with this mask: cells already land stay land; cells that
        were ocean become land where the seam strip + outside-band
        condition holds.

    Returns
    -------
    land_mask : jax array, shape (nCells,)
        1.0 = ocean, 0.0 = land.

    Examples
    --------
    Drake-passage-like channel between -65°S and -45°N::

        mask = partial_periodic_seam_wall_mpas(
            mesh,
            open_lat_south_deg=-65.0,
            open_lat_north_deg=-45.0,
        )
    """
    lon_deg = (jnp.degrees(mesh.lonCell) + 180.0) % 360.0 - 180.0
    lat_deg = jnp.degrees(mesh.latCell)

    if seam_lon_deg is None:
        seam_lon_deg = float(jnp.min(lon_deg))

    if seam_strip_width_deg is None:
        median_dc_m = float(jnp.median(mesh.dcEdge))
        seam_strip_width_deg = (
            (median_dc_m / mesh.radius) * float(constants.RAD_TO_DEG)
        )

    # Modulo wrap so cells near the antipodal side of the periodic seam
    # don't get false-positive matches via negative differences.
    seam_dist = (lon_deg - seam_lon_deg) % 360.0
    near_seam = seam_dist < seam_strip_width_deg
    in_open_band = (
        (lat_deg >= open_lat_south_deg) & (lat_deg <= open_lat_north_deg)
    )
    seam_wall = near_seam & ~in_open_band

    if base_mask is None:
        is_ocean = ~seam_wall
    else:
        is_ocean = (base_mask > 0.5) & ~seam_wall
    return is_ocean.astype(jnp.float32)


def idealized_bathymetry_mpas(
    mesh: VoronoiMesh,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
):
    """Generate idealized bathymetry on Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    H_max : float
        Uniform ocean depth [m].
    land_lat_threshold : float
        Latitude [deg] above which is land.

    Returns
    -------
    H_bathy : jnp.ndarray, shape (nCells,)
        Bathymetry depth, positive downward.
    land_mask : jnp.ndarray, shape (nCells,)
        1=ocean, 0=land.
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            "land_lat_threshold must be in [0, 90] degrees, "
            f"got {land_lat_threshold!r}",
        )

    dtype = get_policy().storage
    lat_deg = jnp.abs(jnp.degrees(mesh.latCell))
    land_mask = (lat_deg < land_lat_threshold).astype(dtype)
    # H_bathy = H_max everywhere (including land) so that the z-star
    # Jacobian (eta + H_bathy) / H_max is smooth across coastlines.
    # The land_mask prevents actual flow on land cells.
    H_bathy = jnp.full_like(land_mask, H_max)
    return H_bathy, land_mask


def rest_state_mpas_ocean(
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
    bathymetry=None,
) -> MPASOceanState:
    """Create a rest-state initial condition on Voronoi mesh.

    Temperature follows an exponential profile from surface to depth.
    Salinity is uniform. Velocity and SSH are zero.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    T_water_init_C : float
        Surface temperature [degC].
    T_deep : float
        Deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Ocean depth [m]. Used by the idealized bathymetry path.
    land_lat_threshold : float
        Land above this latitude [deg]. Used by the idealized
        bathymetry path.
    bathymetry : BathymetryConfig or None, optional
        Realistic bathymetry config. When provided with
        ``source="file"``, ``H_bathy`` and ``land_mask`` are loaded
        from a NetCDF file (ETOPO/GEBCO style) via
        :func:`load_bathymetry_mpas` instead of the idealized path.
        ``H_max``/``land_lat_threshold`` are ignored in that case.
        Default ``None`` keeps the idealized behavior for backwards
        compatibility.

    Returns
    -------
    MPASOceanState
    """
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels

    use_realistic = (
        bathymetry is not None and getattr(bathymetry, "source", None) == "file"
    )
    if use_realistic:
        # Lazy import to avoid pulling xarray into idealized-only callers.
        from legoesm.ocean.bathymetry import load_bathymetry_mpas
        H_bathy, land_mask = load_bathymetry_mpas(mesh, bathymetry)
    else:
        H_bathy, land_mask = idealized_bathymetry_mpas(
            mesh, H_max, land_lat_threshold,
        )

    dtype = get_policy().storage

    # Temperature: exponential profile
    z_full = z_coord.z_full_ref  # (nlev,), negative values
    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(z_full / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_profile[jnp.newaxis, :], (nCells, nlev)).astype(dtype)

    # Salinity: uniform
    S_data = jnp.full((nCells, nlev), S_uniform, dtype=dtype)

    # Velocity: zero
    u_data = jnp.zeros((nEdges, nlev), dtype=dtype)

    # SSH: zero
    eta_data = jnp.zeros(nCells, dtype=dtype)

    w_data = jnp.zeros((nCells, nlev + 1), dtype=dtype)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        w=Field(data=w_data, name="w", dims=("nCells", "nlev+1"), units="m/s"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def wind_driven_gyre_mpas(
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> MPASOceanState:
    """Create initial condition for a wind-driven barotropic gyre on MPAS.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    Wind forcing is applied during integration via the MPAS physics pipeline.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    H_max : float
        Maximum ocean depth [m].
    lon_west, lon_east : float
        Basin longitude bounds [degrees].
    lat_south, lat_north : float
        Basin latitude bounds [degrees].
    T_uniform : float
        Uniform temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].

    Returns
    -------
    MPASOceanState
    """
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    # Basin land mask: ocean inside rectangle, land outside
    lon_deg = jnp.degrees(mesh.lonCell)  # (nCells,)
    lat_deg = jnp.degrees(mesh.latCell)  # (nCells,)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)

    # Uniform depth everywhere (smooth Jacobian at coastlines)
    H_bathy = jnp.full(nCells, H_max, dtype=dtype)

    # Uniform T and S - apply land masking to set land cells to 0.0°C
    T_data = jnp.full((nCells, nlev), T_uniform, dtype=dtype)
    S_data = jnp.full((nCells, nlev), S_uniform, dtype=dtype)

    # Apply land masking: land cells (mask <= 0.5) set to 0.0°C
    mask_3d = land_mask[:, jnp.newaxis]  # Broadcast to 3D
    T_data = jnp.where(mask_3d > 0.5, T_data, 0.0)
    S_data = jnp.where(mask_3d > 0.5, S_data, 0.0)

    # Zero velocity and SSH
    u_data = jnp.zeros((nEdges, nlev), dtype=dtype)
    eta_data = jnp.zeros(nCells, dtype=dtype)

    w_data = jnp.zeros((nCells, nlev + 1), dtype=dtype)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        w=Field(data=w_data, name="w", dims=("nCells", "nlev+1"), units="m/s"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def attach_static_rho_ref_z(
    state: MPASOceanState,
    mesh: VoronoiMesh,
    z_coord,
    config,
) -> MPASOceanState:
    """Compute and freeze a per-level reference density profile on the state.

    Runs the same 2-pass EOS+hydrostatic-pressure iteration that the
    runtime baroclinic tendency uses (via
    :func:`iterate_eos_and_pressure_anomaly`) on the *initial* T and S,
    averages the resulting in-situ density over wet cells per level,
    and writes the resulting ``(nlev,)`` profile to ``state.rho_ref_z``.

    Once attached, the runtime baroclinic PGF iteration uses
    ``ρ' = ρ − ρ_ref(z)`` instead of ``ρ' = ρ − ρ_0``.  The profile is
    NEVER recomputed during integration — that was the
    positive-feedback failure mode of the dynamic recomputed-mean
    version (``use_baroclinic_rho_ref=True``).  See
    project_mpas_etopo_instability.md §"Option B" for context.

    Parameters
    ----------
    state : MPASOceanState
        Must already carry the initial T, S, ``H_bathy``, ``land_mask``.
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        On a partial-cell coordinate, ``is_active`` and ``h_actual`` are
        used so that step-edge cells (zero-thickness layers below the
        seafloor) do not pollute the per-level wet-cell mean.
    config : MPASOceanConfig
        Used for ``eos``, ``eos_linear``, ``rho_0``, ``g``,
        ``min_water_column_m`` (for the partial-cell h_actual), and the
        switches ``use_h_actual_pgf`` / ``use_static_baroclinic_rho_ref``.
        When ``use_static_baroclinic_rho_ref`` is False, returns
        ``state`` unchanged (no-op).

    Returns
    -------
    MPASOceanState
        A new state with ``rho_ref_z`` populated when the config switch
        is on; otherwise the input state unchanged.
    """
    if not getattr(config, "use_static_baroclinic_rho_ref", False):
        return state
    if getattr(config, "use_baroclinic_rho_ref", False):
        raise ValueError(
            "use_static_baroclinic_rho_ref and use_baroclinic_rho_ref "
            "are mutually exclusive: the static frozen profile and the "
            "dynamic recomputed-mean profile cannot both be active. "
            "Set only one of them in MPASOceanConfig."
        )

    # Lazy imports to avoid a hard dependency from the init module on
    # the dynamics tendency stack.
    from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        compute_static_rho_ref_z,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.vertical import compute_layer_thickness

    T_3d = state.T.data
    S_3d = state.S.data
    eta_2d = state.eta.data
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    def _fill(field_cell):
        return fill_land_cells_mpas(field_cell, mask, c1, c2,
                                    mesh.edgesOnCell, mesh.nEdgesOnCell)

    eos_fn = make_eos_fn(
        config.eos, getattr(config, "eos_linear", None),
    )

    if isinstance(z_coord, OceanPartialCellCoordinate):
        is_active_3d = z_coord.is_active.astype(T_3d.dtype)
        # Mirror the runtime PGF guard (ocean_pe_mpas.py): h_actual
        # integration of the hydrostatic pressure pairs with the Adcroft
        # scheme only.  Building the static rho_ref(z) on the SAME grid
        # the runtime p'/p_hydro uses keeps ρ' = ρ − ρ_ref(z) consistent
        # between init and run for every scheme (centered stays on
        # dz_ref, so its static profile does too).
        if (
            getattr(config, "use_h_actual_pgf", False)
            and getattr(config, "pgf_scheme", "centered") == "adcroft"
        ):
            h_actual = compute_layer_thickness(
                eta_2d, H_bathy, z_coord,
                min_water_column_m=config.min_water_column_m,
            )
        else:
            h_actual = None
    else:
        is_active_3d = None
        h_actual = None

    rho_ref_z = compute_static_rho_ref_z(
        T_3d, S_3d, mask, _fill, eos_fn,
        z_coord.dz_ref, config.rho_0, config.g,
        n_iter=2,
        h_actual=h_actual,
        is_active_3d=is_active_3d,
    )

    rho_ref_field = Field(
        data=rho_ref_z,
        name="rho_ref_z",
        dims=("nlev",),
        units="kg/m^3",
    )
    return state._replace(rho_ref_z=rho_ref_field)


def reconstruct_cell_velocity(u_edge, mesh):
    """Reconstruct (u_east, v_north) at cell centers from edge normals.

    Uses the Perot reconstruction: project edge-normal velocities onto
    zonal/meridional directions, weighted by ``dvEdge * dcEdge / (2 * areaCell)``.
    This formula is exact for uniform flow on any Voronoi mesh.

    Note: ``edgeSignOnCell`` is NOT used here — it is needed for the
    divergence operator (flux balance) but not for velocity reconstruction.
    The edge-normal velocity ``u_edge`` already follows the edge's own
    normal direction (defined by ``angleEdge``), so projecting with
    ``cos(angleEdge)`` / ``sin(angleEdge)`` directly gives the correct
    eastward/northward components.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,) or (nEdges, nlev)
        Normal velocity at edges.
    mesh : VoronoiMesh

    Returns
    -------
    u_east : jax.Array, shape (nCells,) or (nCells, nlev)
        Zonal velocity at cell centers.
    v_north : jax.Array, shape (nCells,) or (nCells, nlev)
        Meridional velocity at cell centers.
    """
    is_3d = u_edge.ndim == 2

    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge.dtype)  # (maxEdges, nCells)
    eoc_safe = jnp.maximum(eoc, 0)

    # Reconstruction weight: dvEdge * dcEdge / (2 * areaCell)
    dv = mesh.dvEdge[eoc_safe] * mask  # (maxEdges, nCells)
    dc = mesh.dcEdge[eoc_safe] * mask  # (maxEdges, nCells)
    angle = mesh.angleEdge[eoc_safe]   # (maxEdges, nCells)
    weight = dv * dc / (2.0 * mesh.areaCell[jnp.newaxis, :])  # (maxEdges, nCells)

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    if is_3d:
        u_gathered = u_edge[eoc_safe] * mask[..., jnp.newaxis]
        w3d = weight[..., jnp.newaxis]
        # Both reductions share ``u_gathered * w3d`` weighting along the
        # ``maxEdges`` axis — fuse into one stacked sum.
        _stack = (u_gathered * w3d)[..., jnp.newaxis] * jnp.stack(
            [cos_a, sin_a], axis=-1,
        )[..., jnp.newaxis, :]
        _uv = jnp.sum(_stack, axis=0)
        u_east = _uv[..., 0]
        v_north = _uv[..., 1]
    else:
        u_gathered = u_edge[eoc_safe] * mask
        _stack = (u_gathered * weight)[..., jnp.newaxis] * jnp.stack(
            [cos_a, sin_a], axis=-1,
        )
        _uv = jnp.sum(_stack, axis=0)
        u_east = _uv[..., 0]
        v_north = _uv[..., 1]

    return u_east, v_north
