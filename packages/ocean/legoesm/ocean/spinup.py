"""Multi-decade ocean spin-up workflow + convergence diagnostics.

Provides primitives for centennial OMIP-2 / CMIP-class spin-up runs:

* :func:`compute_amoc_timeseries` — append-only AMOC@26.5°N tracker.
* :class:`SpinupHealth` — per-year health snapshot (AMOC, RPE drift,
  volume / heat / salt drift fractions).
* :func:`evaluate_health` — emit a SpinupHealth from current + initial
  diagnostics.
* :class:`ConvergenceCriteria` — declarative thresholds for
  equilibration detection (AMOC stability window, RPE drift bound,
  volume drift bound).
* :func:`is_converged` — apply criteria to the recent history.
* :func:`find_latest_restart` — locate the most-recent restart file in
  a run directory so the driver auto-resumes.
* :func:`bryan_accelerated_dt` — return the per-phase timestep used by
  the Bryan-Lewis (1984) distorted-physics accelerated spin-up
  protocol (long tracer dt during phase 1, gradual ramp-down to
  physical dt by phase 3).

All helpers are pure-Python / NumPy where possible so the workflow is
JIT-independent and easy to drive from a top-level run script.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np


SV = 1.0e6  # 1 Sverdrup [m³/s]


# ==============================================================================
# AMOC computation from C-grid ocean state
# ==============================================================================

def _grid_lat_v_deg(grid, n_lat_v: int) -> np.ndarray:
    """v-face latitudes in degrees, with sensible fallbacks.

    Prefers ``grid.lat_v`` (the canonical attribute on lat-lon C-grid
    geometries), falls back to ``grid.lat + grid.dlat`` shifts, and
    finally to a uniform ``linspace`` over the pole range.  The chosen
    fallback matches ``moc_streamfunction``'s lat_v derivation so the
    AMOC index resolves consistently with the streamfunction.
    """
    lat_v = getattr(grid, "lat_v", None)
    if lat_v is not None:
        return np.degrees(np.asarray(lat_v))
    lat = getattr(grid, "lat", None)
    dlat = getattr(grid, "dlat", None)
    if lat is not None and dlat is not None:
        lat_arr = np.asarray(lat)
        lat_v_rad = np.concatenate([
            [lat_arr[0] - 0.5 * float(dlat)],
            lat_arr + 0.5 * float(dlat),
        ])
        return np.degrees(lat_v_rad)
    return np.degrees(np.linspace(-np.pi / 2, np.pi / 2, n_lat_v))


def atlantic_basin_mask(
    grid,
    *,
    lon_min_deg: float = -75.0,
    lon_max_deg: float = 15.0,
) -> np.ndarray:
    """Crude Atlantic basin mask via longitude band.

    Produces a ``(n_lon,)`` boolean mask broadcast-compatible with
    ``land_mask`` of shape ``(n_lat, n_lon)``.  Default longitude
    band 75 W → 15 E covers the Atlantic from the American to the
    African / European coasts.  For ORCA-class production the
    caller should supply a proper basin mask file; this geometric
    proxy is adequate for spin-up monitoring at 26.5 °N where the
    Atlantic is well-isolated.

    Parameters
    ----------
    grid : LatLonGrid-like
        Must expose ``lon`` in radians.
    lon_min_deg, lon_max_deg : float
        Longitude band in degrees, ``[-180, 180]``-wrapped.  ``min``
        and ``max`` may straddle the prime meridian.

    Returns
    -------
    mask : ndarray ``(n_lon,)`` of bool
    """
    lon_rad = np.asarray(grid.lon)
    lon_deg = np.degrees(lon_rad)
    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    if lon_min_deg <= lon_max_deg:
        return (lon_wrapped >= lon_min_deg) & (lon_wrapped <= lon_max_deg)
    # Wrap-around band (e.g. [170°E, -170°E]).
    return (lon_wrapped >= lon_min_deg) | (lon_wrapped <= lon_max_deg)


def compute_amoc_from_state(
    v_face: np.ndarray,
    h_partial: np.ndarray,
    land_mask: np.ndarray,
    grid,
    *,
    target_lat_deg: float = 26.5,
    basin: str = "atlantic",
    basin_lon_min_deg: float = -75.0,
    basin_lon_max_deg: float = 15.0,
    lat_tol_deg: float = 5.0,
) -> float:
    """AMOC maximum at ``target_lat_deg`` from a C-grid ocean state.

    Wraps :func:`legoesm.ocean.diagnostics_streamfunction.moc_streamfunction`
    with basin-mask preprocessing and latitude-index resolution.
    Returns the maximum overturning streamfunction value [Sv] at the
    requested latitude, or NaN when the latitude is not on the grid
    within ``lat_tol_deg``.

    Parameters
    ----------
    v_face : array ``(n_lat+1, n_lon, nlev)``
        Meridional velocity at v-faces [m/s].
    h_partial : array ``(n_lat, n_lon, nlev)``
        Layer thickness [m] at cell centres (e.g. from
        :func:`legoesm.ocean.vertical.compute_layer_thickness`).
    land_mask : array ``(n_lat, n_lon)``
        Ocean mask (1 = ocean, 0 = land).
    grid : LatLonGrid-like
        Lat-lon C-grid metadata; must expose ``radius``, ``lon`` and
        either ``lat_v`` or (``lat`` + ``dlat``).
    target_lat_deg : float
        Target latitude in degrees (default 26.5 °N — RAPID array).
    basin : {"atlantic", "global"}
        Basin filter applied via ``land_mask × atlantic_lon_band``.
        ``"global"`` skips the longitude filter.
    basin_lon_min_deg, basin_lon_max_deg : float
        Longitude band (degrees) used when ``basin="atlantic"``.
    lat_tol_deg : float
        Maximum allowed distance between ``target_lat_deg`` and the
        nearest v-face latitude [°].  Beyond this the function
        returns NaN (target latitude not represented on the grid).

    Returns
    -------
    amoc_Sv : float
        Maximum overturning streamfunction at the target latitude
        in Sverdrups, or NaN when the latitude is off-grid or no
        finite streamfunction exists.
    """
    # Local import to avoid circular legoesm.ocean ← legoesm.ocean.spinup
    # at module import time.
    from legoesm.ocean.diagnostics_streamfunction import moc_streamfunction

    v_np = np.asarray(v_face)
    h_np = np.asarray(h_partial)
    mask_np = np.asarray(land_mask).astype(np.float64)

    if basin == "atlantic":
        atl_mask = atlantic_basin_mask(
            grid,
            lon_min_deg=basin_lon_min_deg,
            lon_max_deg=basin_lon_max_deg,
        ).astype(np.float64)
        # Broadcast (n_lon,) → (1, n_lon) so we get a per-cell mask.
        mask_eff = mask_np * atl_mask[None, :]
    elif basin == "global":
        mask_eff = mask_np
    else:
        raise ValueError(
            f"compute_amoc_from_state: unknown basin {basin!r}; "
            "expected one of 'atlantic', 'global'."
        )

    psi_Sv = moc_streamfunction(v_np, h_np, None, None, mask_eff, grid)
    # shape (n_lat+1, nlev) in Sv (moc_streamfunction already
    # divides by 1e6).

    n_lat_v = psi_Sv.shape[0]
    lat_v_deg = _grid_lat_v_deg(grid, n_lat_v)
    j = int(np.argmin(np.abs(lat_v_deg - target_lat_deg)))
    if abs(float(lat_v_deg[j]) - target_lat_deg) > lat_tol_deg:
        return float("nan")

    profile = psi_Sv[j, :]
    if not np.any(np.isfinite(profile)):
        return float("nan")
    # ``moc_streamfunction`` uses ``psi = -cumsum(V_zonal)`` so for the
    # canonical Atlantic overturning cell — northward upper transport,
    # NADW return flow below — ψ has its EXTREMUM as a NEGATIVE peak
    # at the interface between the two branches (typically 1000–2000 m
    # for Atlantic AMOC).  The RAPID convention reports AMOC as a
    # positive number, so we negate the minimum.  An anomalous run
    # with REVERSED circulation (collapsed AMOC, southward at surface)
    # would yield a negative reported value, which is the desired
    # signed-strength behaviour for spin-up monitoring.
    return float(-np.nanmin(profile))


def compute_acc_from_state(
    u_face: np.ndarray,
    h_partial: np.ndarray,
    land_mask: np.ndarray,
    grid,
    *,
    drake_lon_deg: float = -68.0,
    drake_lat_south_deg: float = -65.0,
    drake_lat_north_deg: float = -45.0,
    lon2d_deg: np.ndarray | None = None,
) -> float:
    """ACC (Drake throughflow) [Sv] from a C-grid state — SINGLE-MERIDIAN section.

    Reuses the tested
    :func:`legoesm.ocean.diagnostics_streamfunction.barotropic_streamfunction`
    for the depth-integrated transport streamfunction ψ_bt, then takes its
    max−min ALONG THE DRAKE MERIDIAN over the passage latitude band.  Along a
    single meridian ψ_bt runs monotonically from the South-America value to the
    Antarctica value, so the max−min equals the net zonal throughflow there
    (= ∫ u·h·dy across that meridian).  This is apples-to-apples with the NEMO
    reader (``scripts/validate/nemo_transports.acc_drake_core``) and the MPAS
    section (:func:`compute_acc_from_state_mpas`).  It deliberately does NOT use
    ``diagnostics_climate.acc_transport`` (whole-Drake-band max−min over ALL
    longitudes), which inflates the number by picking up Southern-Ocean gyre
    extrema (latlon probe gave 212 Sv that way vs ~146-159 by section).

    Sibling of :func:`compute_amoc_from_state` but for the ZONAL u-faces.  ACC is
    wind-driven and spins up in MONTHS (vs AMOC's decades), so it is meaningful
    on much shorter integrations.

    Parameters
    ----------
    u_face : array ``(n_lat, n_lon+1, nlev)``
        Zonal velocity at u-faces [m/s].
    h_partial : array ``(n_lat, n_lon, nlev)``
        Layer thickness [m] at cell centres.
    land_mask : array ``(n_lat, n_lon)``
        Ocean mask (1 = ocean, 0 = land).
    grid : LatLonGrid-like
        Must expose ``lat``/``lon`` (radians; 1-D regular or 2-D curvilinear) and
        the metadata ``barotropic_streamfunction`` needs (``radius``,
        ``dy``/``dlat``).
    drake_lon_deg : float
        Drake-Passage section longitude [°] (default -68).
    drake_lat_south_deg, drake_lat_north_deg : float
        Drake latitude band [°].
    lon2d_deg : array ``(n_lat, n_lon)`` or None
        Optional caller-supplied per-cell longitude [°].  On a regular lat-lon
        grid ``grid.lon`` (1-D) is correct; on the Drake meridian a tripole is
        regular too, so the 1-D fallback is fine there, but a 2-D override lets a
        folded-north section be placed exactly.

    Returns
    -------
    acc_Sv : float
        Drake throughflow magnitude in Sverdrups, or NaN if the band is empty.
    """
    # Local import to avoid circular legoesm.ocean ← legoesm.ocean.spinup.
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction

    u_np = np.asarray(u_face, dtype=np.float64)
    h_np = np.asarray(h_partial, dtype=np.float64)
    mask_np = np.asarray(land_mask).astype(np.float64)

    psi_bt_Sv = np.asarray(barotropic_streamfunction(u_np, h_np, mask_np, grid))
    n_lat, n_lon = psi_bt_Sv.shape

    # Passage latitude band (reduce a 2-D curvilinear lat to a per-row mean — the
    # Drake band sits in the regular Southern-Ocean part of any ORCA-like grid).
    lat_arr = np.degrees(np.asarray(grid.lat))
    lat_t_deg = lat_arr.mean(axis=1) if lat_arr.ndim == 2 else lat_arr
    band_rows = np.where((lat_t_deg >= drake_lat_south_deg)
                         & (lat_t_deg <= drake_lat_north_deg))[0]
    if band_rows.size == 0:
        return float("nan")

    # Drake meridian COLUMN.  ``psi_bt[:, i]`` is built from ``u[:, i]`` = the
    # WEST u-face of cell i (barotropic_streamfunction min-rules h_W=roll(h,1)),
    # i.e. the meridian at ``lon[i] - dlon/2``.  Select the column whose FACE
    # longitude is nearest ``drake_lon`` (circular); on a curvilinear 2-D lon the
    # Southern Ocean is ~regular so the half-cell shift is sub-cell.
    lon_src = lon2d_deg if lon2d_deg is not None else np.degrees(np.asarray(grid.lon))
    lon_arr = np.asarray(lon_src, dtype=np.float64)
    lon_row = (lon_arr[int(band_rows[band_rows.size // 2])]
               if lon_arr.ndim == 2 else lon_arr)
    dlon_deg = float(np.degrees(getattr(grid, "dlon", 0.0) or 0.0))
    if dlon_deg == 0.0 and np.asarray(lon_row).size > 1:
        # Tripole sets dlon=0 as a sentinel (non-uniform spacing); estimate the
        # LOCAL Drake-row cell spacing from lon_row so the half-cell U-face shift
        # still applies on the curvilinear grid (regular in the S.Ocean).
        _dl = np.diff(((np.asarray(lon_row) + 180.0) % 360.0) - 180.0)
        _dl = ((_dl + 180.0) % 360.0) - 180.0          # circular diff
        dlon_deg = float(np.median(np.abs(_dl)))
    lon_face = lon_row - 0.5 * dlon_deg
    drake_w = ((float(drake_lon_deg) + 180.0) % 360.0) - 180.0
    lon_w = ((lon_face + 180.0) % 360.0) - 180.0
    i_drake = int(np.argmin(np.abs(((lon_w - drake_w + 180.0) % 360.0) - 180.0)))

    # Section throughflow (EASTWARD-positive, SIGNED) across the band at the Drake
    # meridian.  ``psi_bt[j] = -cumsum_{0..j}(U_dz·dy)``, so the transport through
    # the band rows [a..b] = Σ_{a..b} U_dz·dy = psi_bt[a-1] - psi_bt[b] (the
    # streamfunction just SOUTH of the first band row minus just at the last; ψ
    # south of row 0 is 0).  A ``max-min`` over the band would drop row a
    # (~25% on a 4-row band) and lose the sign, so use the explicit endpoints —
    # matching the signed eastward-positive convention of the NEMO reader and the
    # MPAS section.
    a = int(band_rows[0])
    b = int(band_rows[-1])
    south = float(psi_bt_Sv[a - 1, i_drake]) if a > 0 else 0.0
    north = float(psi_bt_Sv[b, i_drake])
    if not (np.isfinite(south) and np.isfinite(north)):
        return float("nan")
    return south - north


# ==============================================================================
# MPAS Voronoi-mesh AMOC
# ==============================================================================

def atlantic_basin_mask_mpas(
    mesh,
    *,
    lon_min_deg: float = -75.0,
    lon_max_deg: float = 15.0,
) -> np.ndarray:
    """Per-cell Atlantic basin mask on a Voronoi mesh.

    Returns a ``(nCells,)`` boolean mask of cells whose centres fall
    inside the longitude band ``[lon_min_deg, lon_max_deg]``
    (degrees, ``-180..180`` after wrap).  Wrap-around bands
    (``lon_min > lon_max``) span the dateline.

    Parameters
    ----------
    mesh : VoronoiMesh
        Must expose ``lonCell`` in radians.
    """
    lon_deg = np.degrees(np.asarray(mesh.lonCell))
    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    if lon_min_deg <= lon_max_deg:
        return (lon_wrapped >= lon_min_deg) & (lon_wrapped <= lon_max_deg)
    return (lon_wrapped >= lon_min_deg) | (lon_wrapped <= lon_max_deg)


def compute_amoc_from_state_mpas(
    u_edge: np.ndarray,
    h_cell: np.ndarray,
    mesh,
    *,
    target_lat_deg: float = 26.5,
    lat_band_width_deg: float = 1.0,
    basin: str = "atlantic",
    basin_lon_min_deg: float = -75.0,
    basin_lon_max_deg: float = 15.0,
    lat_tol_deg: float = 5.0,
) -> float:
    """AMOC at ``target_lat_deg`` from a Voronoi-mesh ocean state.

    Latitude-bin meridional volume flux across cell edges, then take
    the cumulative depth integral to get the overturning
    streamfunction in Sv.  Returns the RAPID-style positive value
    ``-min(ψ)`` over the depth profile at the target band; NaN when
    the target latitude has no edges within ``lat_tol_deg``.

    Algorithm
    ---------
    For each edge ``e`` with normal angle ``α_e`` measured from
    east, the meridional component of the edge-normal velocity is
    ``v_north = u_edge · sin(α_e)``.  Edge volume flux per level is

        F_e(k) = u_edge(k) · sin(α_e) · dvEdge(e) · h_edge(e, k)

    where ``h_edge = 0.5 · (h_{c1} + h_{c2})`` is the centred edge
    thickness (one-sided at boundary edges).  Edges are binned by
    their cell-centre latitude ``latEdge`` into bands centred on the
    target latitude; the sum across all edges in a band gives the
    band's per-level meridional volume flux.  The streamfunction is
    the cumulative integral from the surface downward (matching the
    lat-lon ``moc_streamfunction`` sign convention).

    Atlantic basin filter: builds a per-cell Atlantic mask, projects
    to edges as the AND of both adjacent cells' Atlantic flags, then
    zeros out non-Atlantic edges before binning.

    Parameters
    ----------
    u_edge : array ``(nEdges, nlev)``
        Edge-normal velocity [m/s].  Shape may also be ``(nEdges,)``
        for a single-layer test field.
    h_cell : array ``(nCells, nlev)``
        Cell-centre layer thickness [m].
    mesh : VoronoiMesh
    target_lat_deg : float
        Latitude (°N) at which to evaluate AMOC.  Default 26.5°N.
    lat_band_width_deg : float
        Half-width of the latitude band around ``target_lat_deg`` [°].
        All edges with ``|latEdge - target| <= width`` participate.
    basin : {"atlantic", "global"}
        Basin filter.
    basin_lon_min_deg, basin_lon_max_deg : float
        Atlantic longitude band [°].
    lat_tol_deg : float
        If no edges fall within ``lat_tol_deg`` of the target
        latitude, return NaN.

    Returns
    -------
    amoc_Sv : float
        RAPID-style positive AMOC magnitude at ``target_lat_deg``
        in Sverdrups, or NaN if the target latitude is empty.
    """
    u = np.asarray(u_edge, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    h = np.asarray(h_cell, dtype=np.float64)
    if h.ndim == 1:
        h = h[:, None]
    if u.shape[-1] != h.shape[-1]:
        raise ValueError(
            f"compute_amoc_from_state_mpas: u_edge has {u.shape[-1]} "
            f"levels but h_cell has {h.shape[-1]}"
        )
    n_edges, nlev = u.shape

    angle = np.asarray(mesh.angleEdge, dtype=np.float64)
    # Canonical min-rule edge thickness (shared with the MPAS dynamics) so the
    # meridional-flux cross-section uses the SAME partial-cell flux closure as
    # the model — never re-derived.  Local imports keep spinup import-light.
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge

    dv = np.asarray(mesh.dvEdge, dtype=np.float64)
    sin_a = np.sin(angle)                                   # (nEdges,)

    # Min-rule edge thickness from cellsOnEdge (MITgcm hFacZ flux closure): the
    # shallower cell limits the flow-through area, so a partial-cell bottom step
    # or a coastline edge (one cell dry, h=0) carries no phantom meridional flux
    # (centred averaging would; the AMOC binning sums these edges).  Boundary
    # edges (c2<0) keep the one-sided thickness — their u_edge≈0 (no-normal-flow
    # BC) so the choice is immaterial there, but min_cell_to_edge's c2=-1 index
    # would be garbage, hence the explicit interior/boundary ``where``.
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    interior = c2 >= 0
    c1_safe = np.where(c1 >= 0, c1, 0)
    c2_safe = np.where(c2 >= 0, c2, 0)
    h_min = np.asarray(min_cell_to_edge(jnp.asarray(h), mesh))  # (nEdges, nlev)
    h_e = np.where(
        interior[:, None],
        h_min,
        h[c1_safe],
    )                                                       # (nEdges, nlev)

    # Basin filter applied via per-edge mask.
    if basin == "atlantic":
        cell_mask = atlantic_basin_mask_mpas(
            mesh,
            lon_min_deg=basin_lon_min_deg,
            lon_max_deg=basin_lon_max_deg,
        ).astype(np.float64)
        edge_mask = np.where(
            interior,
            cell_mask[c1_safe] * cell_mask[c2_safe],
            cell_mask[c1_safe],
        )
    elif basin == "global":
        edge_mask = np.ones(n_edges, dtype=np.float64)
    else:
        raise ValueError(
            f"compute_amoc_from_state_mpas: unknown basin {basin!r}; "
            "expected one of 'atlantic', 'global'."
        )

    # Per-edge per-level meridional volume flux.
    F_edge = (u * sin_a[:, None] * dv[:, None] * edge_mask[:, None]) * h_e
    # shape (nEdges, nlev) — [m/s · m · m] = m³/s.

    # Latitude binning.
    lat_edge_deg = np.degrees(np.asarray(mesh.latEdge))
    band = np.abs(lat_edge_deg - target_lat_deg) <= lat_band_width_deg
    if not np.any(band):
        # Try a wider tolerance fallback.
        band = np.abs(lat_edge_deg - target_lat_deg) <= lat_tol_deg
        if not np.any(band):
            return float("nan")

    F_band = F_edge[band, :].sum(axis=0)                    # (nlev,)
    if not np.any(np.isfinite(F_band)):
        return float("nan")

    # Cumulative depth integral (surface → bottom).  Matching the
    # lat-lon ``moc_streamfunction`` convention: ``ψ = -cumsum``.
    # A genuinely-zero band (e.g. Atlantic-filter excludes all
    # transport at this latitude) produces a flat zero ψ; the
    # ``-nanmin`` then returns 0.0 which is the physically correct
    # answer (no overturning).
    psi_m3s = -np.cumsum(F_band)
    psi_Sv = psi_m3s / 1.0e6
    return float(-np.nanmin(psi_Sv))


def compute_acc_from_state_mpas(
    u_edge: np.ndarray,
    h_cell: np.ndarray,
    mesh,
    *,
    drake_lon_deg: float = -68.0,
    drake_lat_south_deg: float = -65.0,
    drake_lat_north_deg: float = -45.0,
) -> float:
    """ACC (Drake throughflow) [Sv] from a Voronoi-mesh ocean state.

    SECTION-TRANSPORT method.  The Drake meridian (constant longitude
    ``drake_lon_deg``) is crossed by exactly those INTERIOR edges whose two
    cells lie on OPPOSITE sides of the meridian and whose edge latitude is in
    the passage band.  Each such edge is crossed once (no double counting); the
    eastward volume transport is the orientation-corrected edge-normal flux
    summed over those edges and depth:

        ACC = Σ_section Σ_k  s_e · u_edge(k) · dvEdge · h_edge(k)

    The MPAS edge normal points from ``cellsOnEdge[0]`` (c1) to ``cellsOnEdge[1]``
    (c2), so positive ``u_edge`` is c1→c2 flow.  The orientation
    ``s_e = sign(d2 − d1)`` (with ``d = signed circular longitude − drake_lon``)
    is +1 when c2 is EAST of c1, making the contribution eastward-positive.
    SIGNED (eastward +) so a reversed convention surfaces as a sign flip rather
    than being hidden.  Edges near the ANTIPODAL meridian (where opposite
    circular-longitude signs are a wrap artefact, not a true crossing) are
    excluded by the ``|d| < 90°`` guard.  NaN if no section edges.  Equivalent
    to the offline NEMO ``acc_drake_core`` section integral; wind-driven ACC
    spins up in months -> meaningful on multi-year runs.

    The edge cross-section uses the canonical MIN-RULE edge thickness
    (``mpas_partial_cell_helpers.min_cell_to_edge`` — MITgcm hFacZ flux closure),
    NOT a centred average: under partial cells the shallower cell limits the
    flow-through area, so a Drake edge across a bottom step or a coastline (one
    cell dry, h=0) carries no phantom sub-seafloor / into-land transport.

    Parameters
    ----------
    u_edge : array ``(nEdges, nlev)`` (or ``(nEdges,)`` single-level)
        Edge-normal velocity [m/s].
    h_cell : array ``(nCells, nlev)`` (or ``(nCells,)``)
        Cell-centre layer thickness [m].
    mesh : VoronoiMesh
        Must expose ``dvEdge``, ``cellsOnEdge`` (``[2, nEdges]``, −1 at
        boundaries), ``lonCell`` and ``latEdge`` (radians).
    drake_lon_deg : float
        Drake-Passage section longitude [°] (default −68).
    drake_lat_south_deg, drake_lat_north_deg : float
        Passage latitude band [°].

    Returns
    -------
    acc_Sv : float
        Eastward Drake throughflow in Sverdrups (signed), or NaN if no edges
        cross the section.
    """
    u = np.asarray(u_edge, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    h = np.asarray(h_cell, dtype=np.float64)
    if h.ndim == 1:
        h = h[:, None]
    if u.shape[-1] != h.shape[-1]:
        raise ValueError(
            f"compute_acc_from_state_mpas: u_edge has {u.shape[-1]} levels "
            f"but h_cell has {h.shape[-1]}"
        )

    # Canonical min-rule edge thickness (shared with the MPAS dynamics) so the
    # section flux uses the SAME partial-cell flux closure as the model — never
    # re-derived here.  Local imports keep spinup import-light and JAX-optional.
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge

    dv = np.asarray(mesh.dvEdge, dtype=np.float64)
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    interior = c2 >= 0
    c1_safe = np.where(c1 >= 0, c1, 0)
    c2_safe = np.where(c2 >= 0, c2, 0)
    # Min-rule edge thickness min(h[c1],h[c2]); boundary edges (c2<0) index a
    # wrong cell but are excluded from the section by ``interior`` below.
    h_e = np.asarray(min_cell_to_edge(jnp.asarray(h), mesh))   # (nEdges, nlev)

    # Signed circular longitude of each cell relative to the Drake meridian.
    lon_cell_deg = np.degrees(np.asarray(mesh.lonCell))
    drake_w = ((float(drake_lon_deg) + 180.0) % 360.0) - 180.0
    d1 = ((lon_cell_deg[c1_safe] - drake_w + 180.0) % 360.0) - 180.0
    d2 = ((lon_cell_deg[c2_safe] - drake_w + 180.0) % 360.0) - 180.0
    # Straddle = cells on opposite sides of the meridian, NEAR it (the |d|<90
    # guard rejects the antipodal meridian where opposite signs are a wrap
    # artefact rather than a real crossing).
    near = (np.abs(d1) < 90.0) & (np.abs(d2) < 90.0)
    straddle = interior & near & ((d1 * d2) < 0.0)

    lat_edge_deg = np.degrees(np.asarray(mesh.latEdge))
    band = ((lat_edge_deg >= drake_lat_south_deg)
            & (lat_edge_deg <= drake_lat_north_deg))
    section = straddle & band
    if not np.any(section):
        return float("nan")

    s_e = np.sign(d2 - d1)                                 # +1 if c2 east of c1
    # Oriented eastward volume transport per edge per level [m³/s].
    F = (s_e[:, None] * u * dv[:, None]) * h_e
    transport = float(F[section, :].sum())
    return transport / 1.0e6


# ==============================================================================
# AMOC timeseries
# ==============================================================================

def compute_amoc_timeseries(
    amoc_yearly_Sv: list[float] | np.ndarray,
    *,
    window_years: int = 30,
) -> dict:
    """Summarise the AMOC time series for spin-up health.

    Computes:
        * Latest value.
        * Trailing-window mean.
        * Trailing-window standard deviation (variability measure).
        * Trailing-window slope from a linear fit [Sv/yr] (drift rate).
        * Drift fraction over the window: ``|slope · window| / mean``.

    Parameters
    ----------
    amoc_yearly_Sv : sequence of float
        AMOC max at the monitoring latitude (typically 26.5°N) in Sv,
        one value per simulated year (oldest first).
    window_years : int
        Trailing window for stability metrics.  Default 30 years —
        long enough to average out interannual variability, short
        enough to detect persistent drift.
    """
    a = np.asarray(amoc_yearly_Sv, dtype=np.float64)
    if a.size == 0:
        return {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": window_years,
        }
    latest = float(a[-1])
    window = a[-window_years:] if a.size >= window_years else a
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    if window.size >= 2:
        years = np.arange(window.size, dtype=np.float64)
        slope, _ = np.polyfit(years, window, 1)
        slope = float(slope)
    else:
        slope = 0.0
    drift_fraction = (
        abs(slope * window.size) / max(abs(mean), 1e-12)
        if mean != 0.0 else float("inf")
    )
    return {
        "latest_Sv": latest,
        "window_mean_Sv": mean,
        "window_std_Sv": std,
        "window_slope_Sv_per_yr": slope,
        "window_drift_fraction": drift_fraction,
        "n_years": int(a.size),
        "window_years": int(min(window_years, a.size)),
    }


# ==============================================================================
# Spin-up health snapshot
# ==============================================================================

class SpinupHealth(NamedTuple):
    """Per-year spin-up health snapshot.

    ``rpe_drift_W_per_m2`` is the reference-potential-energy drift
    relative to the initial state, area-averaged — a standard
    Griffies (2015) mixing diagnostic.  ``volume_drift_frac`` and
    ``heat_drift_frac`` are the fractional drifts of the global
    volume + heat-content integrals (signed; positive = gain).
    ``salt_drift_frac`` follows the same convention for the salt
    integral.
    """
    year: int
    amoc_Sv: float
    rpe_drift_W_per_m2: float
    volume_drift_frac: float
    heat_drift_frac: float
    salt_drift_frac: float


def evaluate_health(
    *,
    year: int,
    amoc_Sv: float,
    rpe_drift_W_per_m2: float,
    volume_now: float,
    volume_init: float,
    heat_now: float,
    heat_init: float,
    salt_now: float,
    salt_init: float,
) -> SpinupHealth:
    """Build a :class:`SpinupHealth` from raw diagnostics."""
    def _drift(now: float, init: float) -> float:
        if abs(init) < 1e-30:
            return float("nan")
        return float((now - init) / init)

    return SpinupHealth(
        year=int(year),
        amoc_Sv=float(amoc_Sv),
        rpe_drift_W_per_m2=float(rpe_drift_W_per_m2),
        volume_drift_frac=_drift(volume_now, volume_init),
        heat_drift_frac=_drift(heat_now, heat_init),
        salt_drift_frac=_drift(salt_now, salt_init),
    )


# ==============================================================================
# Convergence criteria
# ==============================================================================

class ConvergenceCriteria(NamedTuple):
    """Thresholds for declaring spin-up equilibrium.

    Defaults are conservative OMIP-2 protocol values:
        * AMOC drift fraction < 5 % over a 30-year window.
        * RPE drift < 0.05 W/m² (Griffies 2015 mixing-budget bound).
        * Volume drift fraction < 1e-3.
        * Heat-content drift < 1e-2 (1 % over the spin-up).
        * Salt-mass drift < 1e-3 (sharp because salt is exactly
          conserved by transport).
    """
    window_years: int = 30
    amoc_drift_fraction_max: float = 0.05
    rpe_drift_W_per_m2_max: float = 0.05
    volume_drift_fraction_max: float = 1.0e-3
    heat_drift_fraction_max: float = 1.0e-2
    salt_drift_fraction_max: float = 1.0e-3


def is_converged(
    history: list[SpinupHealth],
    criteria: ConvergenceCriteria = ConvergenceCriteria(),
) -> dict:
    """Apply convergence criteria to the spin-up history.

    Returns a dict with the boolean ``"converged"`` plus per-criterion
    pass/fail flags + the underlying metric values so callers can log
    a detailed status.

    AMOC handling: when every AMOC value in the history is NaN
    (e.g. the driver has not yet wired the MOC streamfunction
    diagnostic), the AMOC criterion is skipped — the returned dict
    sets ``amoc_skipped=True`` and ``amoc_ok=True`` so the
    remaining criteria can still drive a convergence decision.  A
    partial-NaN history is still evaluated against the finite tail.
    """
    if len(history) == 0:
        return {"converged": False, "reason": "empty history"}

    amoc_series = [h.amoc_Sv for h in history]
    # Build the longest contiguous FINITE suffix that ends at the
    # last finite value in the series:
    #   1. Strip TRAILING NaNs (transient diagnostic gaps) — they
    #      don't invalidate the convergence check.
    #   2. From the resulting tail, walk backward over the
    #      contiguous finite values to obtain the evaluation window.
    # Guard: if the most recent ``max_trailing_nan_years`` block is
    # all NaN, treat as a data regression (``amoc_has_recent_nan``)
    # and fail the AMOC criterion — we will not pass convergence on
    # stale finite values that are far in the past.
    max_trailing_nan = max(int(criteria.window_years // 3), 5)

    # Identify the position of the last finite value.
    last_finite_idx = -1
    for i in range(len(amoc_series) - 1, -1, -1):
        if np.isfinite(amoc_series[i]):
            last_finite_idx = i
            break

    all_nan = last_finite_idx < 0
    trailing_nan_count = len(amoc_series) - 1 - last_finite_idx
    has_recent_nan = (not all_nan) and trailing_nan_count > max_trailing_nan

    if last_finite_idx >= 0:
        finite_suffix: list[float] = []
        for i in range(last_finite_idx, -1, -1):
            v = amoc_series[i]
            if np.isfinite(v):
                finite_suffix.append(v)
            else:
                break
        finite_suffix.reverse()
    else:
        finite_suffix = []

    amoc_skipped = all_nan

    if amoc_skipped:
        amoc_ok = True
        summary = {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": criteria.window_years,
        }
    elif has_recent_nan:
        # AMOC diagnostic has been missing for more than
        # ``max_trailing_nan`` years — too long to trust the finite
        # past as representative of current state.  Fail until the
        # diagnostic returns.
        amoc_ok = False
        summary = {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": criteria.window_years,
        }
    else:
        summary = compute_amoc_timeseries(
            finite_suffix, window_years=criteria.window_years,
        )
        amoc_ok = (
            np.isfinite(summary["window_drift_fraction"])
            and summary["window_drift_fraction"] < criteria.amoc_drift_fraction_max
        )

    latest = history[-1]
    rpe_ok = abs(latest.rpe_drift_W_per_m2) < criteria.rpe_drift_W_per_m2_max
    vol_ok = abs(latest.volume_drift_frac) < criteria.volume_drift_fraction_max
    heat_ok = abs(latest.heat_drift_frac) < criteria.heat_drift_fraction_max
    salt_ok = abs(latest.salt_drift_frac) < criteria.salt_drift_fraction_max

    converged = bool(amoc_ok and rpe_ok and vol_ok and heat_ok and salt_ok)
    return {
        "converged": converged,
        "amoc_ok": amoc_ok,
        "amoc_skipped": amoc_skipped,
        "amoc_has_recent_nan": has_recent_nan,
        "rpe_ok": rpe_ok,
        "volume_ok": vol_ok,
        "heat_ok": heat_ok,
        "salt_ok": salt_ok,
        "amoc_summary": summary,
        "latest": latest._asdict(),
        "criteria": criteria._asdict(),
    }


# ==============================================================================
# Restart-chain helpers
# ==============================================================================

_RESTART_YEAR_RE = re.compile(r"restart_year_(\d{4,})\.npz$")


def find_latest_restart(out_dir: str | Path) -> Path | None:
    """Return the path of the most-recent ``restart_year_YYYY.npz`` file.

    Looks in ``out_dir`` for files matching the
    ``restart_year_<NNNN>.npz`` naming convention used by
    :func:`legoesm.ocean.restart.save_restart`.  Returns the file
    with the maximum year number, or ``None`` if no restart is found.

    Parameters
    ----------
    out_dir : str or Path

    Returns
    -------
    Path or None
    """
    p = Path(out_dir)
    if not p.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for f in p.iterdir():
        m = _RESTART_YEAR_RE.search(f.name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# ==============================================================================
# Bryan-Lewis accelerated spin-up timestep
# ==============================================================================

def bryan_accelerated_dt(
    year: int,
    *,
    dt_physical_s: float = 1800.0,
    dt_tracer_ratio: float = 10.0,
    phase1_years: int = 200,
    phase2_years: int = 100,
) -> tuple[float, float]:
    """Return ``(dt_momentum, dt_tracer)`` for the Bryan-Lewis spin-up.

    Bryan & Lewis (1984) introduced a 3-phase distorted-physics
    accelerated spin-up that uses a long tracer timestep to advance
    the slow tracer field while the momentum equation is integrated
    on a physical timestep.  Phase 3 (post-acceleration) uses the
    physical timestep for both equations so the model returns to
    the standard regime before the production phase begins.

    Phase 1 (year < phase1_years): ``dt_tracer = ratio · dt_phys``.
    Phase 2 (phase1 ≤ year < phase1 + phase2): linear ramp from
    ``ratio · dt_phys`` down to ``dt_phys``.
    Phase 3 (year ≥ phase1 + phase2): ``dt_tracer = dt_phys``.

    Parameters
    ----------
    year : int
        0-indexed year inside the spin-up.
    dt_physical_s : float
        Physical timestep [s] for momentum.
    dt_tracer_ratio : float
        Tracer-to-momentum dt ratio during phase 1.
    phase1_years : int
        Length of accelerated phase 1.
    phase2_years : int
        Length of phase-2 ramp-down.

    Returns
    -------
    dt_momentum_s : float
    dt_tracer_s : float
    """
    if year < phase1_years:
        return dt_physical_s, dt_physical_s * dt_tracer_ratio
    if year < phase1_years + phase2_years:
        # Linear ramp from ratio → 1 over phase2.
        progress = (year - phase1_years) / max(phase2_years, 1)
        ratio = dt_tracer_ratio + (1.0 - dt_tracer_ratio) * progress
        return dt_physical_s, dt_physical_s * ratio
    return dt_physical_s, dt_physical_s
