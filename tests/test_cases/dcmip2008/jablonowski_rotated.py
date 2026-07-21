"""DCMIP 2008 4-x: rotated Jablonowski-Williamson tests.

Rotates the standard Jablonowski-Williamson (2006) baroclinic-wave test
about the y-axis by an angle ``alpha``, exposing dycore biases tied to
the polar lat-lon singularities or to cubed-sphere face/edge alignments.

Two flavours are exposed:

- ``rotated_steady_init*``  — α-rotated steady-state (no perturbation).
  Should remain steady to within discretisation error indefinitely.
- ``rotated_baroclinic_init*`` — α-rotated baroclinic wave (with the
  exponential perturbation). Wave growth begins around day 4 in any
  rotated frame.

The default ``alpha = π/4 = 45°`` is the canonical DCMIP-2008 4-1 / 4-2
choice: it pushes the jet stream over the cubed-sphere corners and
across the lat-lon poles where most discretisations are weakest.

Implementation strategy
-----------------------
The analytic JW06 solution is zonally symmetric in the unrotated frame.
We therefore:

1. Rotate every (lon, lat) grid point about the y-axis by -α to obtain
   coordinates (lon_r, lat_r) in the *rotated* (test-defining) frame.
2. Evaluate the existing analytic solution at lat_r — this gives
   pressure, temperature and a zonal wind ``u_r`` aligned with the
   rotated-frame east direction.
3. Transform ``u_r`` (with v_r = 0) back to geographic (east, north)
   components at (lon, lat) using the local rotation between the two
   frames.

All thermodynamic helpers from :mod:`tests.test_cases.baroclinic_wave`
are reused — no analytic formulas are re-derived here.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import SigmaCoordinate

from tests.test_cases.baroclinic_wave import (
    P0,
    compute_zonal_wind,
    evaluate_pressure_temperature,
    exponential_perturbation,
    find_z_for_pressure,
)


# ---------------------------------------------------------------------------
# Coordinate rotation: y-axis rotation by angle alpha
# ---------------------------------------------------------------------------


def _rotate_to_rotated_frame(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate geographic (lon, lat) into the rotated test-frame.

    Rotation is about the y-axis (i.e. the axis through (lon=π/2, lat=0)
    and (lon=3π/2, lat=0)). The rotation matrix maps the original North
    Pole (0,0,1) to (sin(α), 0, cos(α)).

    Returns
    -------
    lon_r, lat_r : jnp.ndarray
        Coordinates in the rotated frame.
    """
    # Cartesian unit vector in original frame
    x = jnp.cos(lat) * jnp.cos(lon)
    y = jnp.cos(lat) * jnp.sin(lon)
    z = jnp.sin(lat)

    # Rotation about y-axis by -alpha (we rotate the FRAME by +alpha,
    # which is equivalent to rotating points by -alpha)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    xr = cos_a * x - sin_a * z
    yr = y
    zr = sin_a * x + cos_a * z

    lat_r = jnp.arcsin(jnp.clip(zr, -1.0, 1.0))
    lon_r = jnp.arctan2(yr, xr)
    # Normalize lon_r to [0, 2π) to match grid convention if needed
    lon_r = jnp.mod(lon_r, 2.0 * jnp.pi)
    return lon_r, lat_r


def _wind_rotation_factors(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    lon_r: jnp.ndarray,
    lat_r: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project rotated-frame zonal wind onto geographic (east, north).

    Given a vector u_r * e_hat_rotated_east at the rotated-frame point
    (lon_r, lat_r), return scalar projection coefficients (cos_gamma,
    -sin_gamma) such that:

        u_east_geo  = u_r * cos_gamma
        v_north_geo =-u_r * sin_gamma

    The local angle ``gamma`` satisfies::

        cos(gamma) = e_hat_rotated_east(P_r) · e_hat_geo_east(P)
        sin(gamma) = e_hat_rotated_east(P_r) · n_hat_geo_north(P)

    Both unit vectors are evaluated in the original (unrotated) frame
    via the inverse rotation matrix (a rotation by +α about y-axis).
    """
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    # Rotated-frame east unit vector at (lon_r, lat_r), expressed in the
    # rotated coordinate basis: (-sin(lon_r), cos(lon_r), 0).
    er_x = -jnp.sin(lon_r)
    er_y = jnp.cos(lon_r)
    er_z = jnp.zeros_like(lon_r)

    # Inverse rotation (rotated → original): rotation by +alpha about y.
    e_x = cos_a * er_x + sin_a * er_z
    e_y = er_y
    e_z = -sin_a * er_x + cos_a * er_z

    # Geographic local east at (lon, lat) in original frame
    geo_e_x = -jnp.sin(lon)
    geo_e_y = jnp.cos(lon)
    geo_e_z = jnp.zeros_like(lon)

    # Geographic local north at (lon, lat) in original frame
    geo_n_x = -jnp.sin(lat) * jnp.cos(lon)
    geo_n_y = -jnp.sin(lat) * jnp.sin(lon)
    geo_n_z = jnp.cos(lat)

    cos_gamma = e_x * geo_e_x + e_y * geo_e_y + e_z * geo_e_z
    sin_gamma_neg_v = -(e_x * geo_n_x + e_y * geo_n_y + e_z * geo_n_z)
    return cos_gamma, sin_gamma_neg_v


# ---------------------------------------------------------------------------
# Per-grid initialisations
# ---------------------------------------------------------------------------


def _build_rotated_state_dense_grid(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    sigma_coord: SigmaCoordinate,
    perturbed: bool,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (u_east, v_north, T, p_s) on dense (..., nlev) lat/lon arrays.

    Used by both the cubed-sphere and lat-lon flavours; the cubed-sphere
    caller is responsible for the additional grid-rotation step.
    """
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full

    lon_r, lat_r = _rotate_to_rotated_frame(lon, lat, alpha)
    cos_gamma, sin_gamma_neg_v = _wind_rotation_factors(
        lon, lat, lon_r, lat_r, alpha,
    )

    p_s = jnp.full(lon.shape, P0)

    u_east = jnp.zeros(lon.shape + (nlev,))
    v_north = jnp.zeros(lon.shape + (nlev,))
    T_3d = jnp.zeros(lon.shape + (nlev,))

    for k in range(nlev):
        p_target = jnp.full(lon.shape, float(sigma_full[k]) * P0)

        # Evaluate analytic IC at rotated-frame latitude
        z_k_r = find_z_for_pressure(p_target, lat_r)
        _, T_k = evaluate_pressure_temperature(z_k_r, lat_r)
        u_r = compute_zonal_wind(z_k_r, lat_r, T_k)

        if perturbed:
            # Perturbation centred at (PERT_LON, PERT_LAT) in the rotated
            # frame — that is the canonical DCMIP definition.
            u_r = u_r + exponential_perturbation(lat_r, lon_r, z_k_r)

        # Project rotated-frame zonal wind onto geographic axes
        u_e_k = u_r * cos_gamma
        v_n_k = u_r * sin_gamma_neg_v

        u_east = u_east.at[..., k].set(u_e_k)
        v_north = v_north.at[..., k].set(v_n_k)
        T_3d = T_3d.at[..., k].set(T_k)

    return u_east, v_north, T_3d, p_s


def rotated_baroclinic_init(
    grid,
    sigma_coord: SigmaCoordinate,
    *,
    perturbed: bool = True,
    alpha: float = jnp.pi / 4.0,
) -> HydrostaticState:
    """Rotated J-W on a cubed-sphere grid (DCMIP 2008 §4-1 / §4-2).

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate
    perturbed : bool
        If False, returns the steady-state (DCMIP §4-1 rotated). If True
        (default), adds the exponential perturbation that triggers the
        baroclinic instability (DCMIP §4-2 rotated).
    alpha : float
        Rotation angle [rad]. Default π/4 is the canonical DCMIP value.
    """
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid

    lat = grid.lat   # (6, n, n)
    lon = grid.lon   # (6, n, n)

    u_east, v_north, T_3d, p_s = _build_rotated_state_dense_grid(
        lon, lat, sigma_coord, perturbed, float(alpha),
    )

    # Geographic (east, north) → grid-aligned (x, y)
    nlev = sigma_coord.n_levels
    u_grid = jnp.zeros_like(u_east)
    v_grid = jnp.zeros_like(u_east)
    for k in range(nlev):
        u_k, v_k = rotate_winds_geo_to_grid(
            u_east[..., k], v_north[..., k], grid.angle,
        )
        u_grid = u_grid.at[..., k].set(u_k)
        v_grid = v_grid.at[..., k].set(v_k)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    n = grid.n
    return HydrostaticState(
        u=Field(data=u_grid, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(
            data=jnp.zeros((6, n, n)),
            name="phis", dims=dims_2d, units="m^2/s^2",
        ),
    )


def rotated_baroclinic_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate,
    *,
    perturbed: bool = True,
    alpha: float = jnp.pi / 4.0,
) -> HydrostaticState:
    """Rotated J-W on a lat-lon grid."""
    lat_1d = jnp.asarray(grid.lat)
    lon_1d = jnp.asarray(grid.lon)
    lat = lat_1d[:, None]  # (n_lat, 1)
    lon = lon_1d[None, :]  # (1, n_lon)
    # Broadcast both to (n_lat, n_lon) explicitly so downstream
    # broadcasting matches the cubed-sphere flavour
    n_lat = lat_1d.size
    n_lon = lon_1d.size
    lat_2d = jnp.broadcast_to(lat, (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(lon, (n_lat, n_lon))

    u_east, v_north, T_3d, p_s = _build_rotated_state_dense_grid(
        lon_2d, lat_2d, sigma_coord, perturbed, float(alpha),
    )

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    return HydrostaticState(
        u=Field(data=u_east, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_north, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(
            data=jnp.zeros((n_lat, n_lon)),
            name="phis", dims=dims_2d, units="m^2/s^2",
        ),
    )


def rotated_baroclinic_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate,
    *,
    perturbed: bool = True,
    alpha: float = jnp.pi / 4.0,
):
    """Rotated J-W on an MPAS Voronoi mesh."""
    from legoesm.core.state import MPASHydrostaticState

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full

    # Cell-centre coordinates → (T, p_s) on cells
    lat_c = mesh.latCell
    lon_c = mesh.lonCell
    lon_c_r, lat_c_r = _rotate_to_rotated_frame(lon_c, lat_c, float(alpha))

    p_s_data = jnp.full((nCells,), P0)
    T_data = jnp.zeros((nCells, nlev))
    for k in range(nlev):
        sig_k = float(sigma_full[k])
        p_target = jnp.full((nCells,), sig_k * P0)
        z_c = find_z_for_pressure(p_target, lat_c_r)
        _, T_k = evaluate_pressure_temperature(z_c, lat_c_r)
        T_data = T_data.at[:, k].set(T_k)

    # Edge-midpoint coordinates → u_normal at edges
    lat_e = mesh.latEdge
    lon_e = mesh.lonEdge
    lon_e_r, lat_e_r = _rotate_to_rotated_frame(lon_e, lat_e, float(alpha))
    cos_gamma, sin_gamma_neg_v = _wind_rotation_factors(
        lon_e, lat_e, lon_e_r, lat_e_r, float(alpha),
    )
    cos_angle = jnp.cos(mesh.angleEdge)
    sin_angle = jnp.sin(mesh.angleEdge)

    u_data = jnp.zeros((nEdges, nlev))
    for k in range(nlev):
        sig_k = float(sigma_full[k])
        p_target = jnp.full((nEdges,), sig_k * P0)
        z_e = find_z_for_pressure(p_target, lat_e_r)
        _, T_e = evaluate_pressure_temperature(z_e, lat_e_r)
        u_r = compute_zonal_wind(z_e, lat_e_r, T_e)
        if perturbed:
            u_r = u_r + exponential_perturbation(lat_e_r, lon_e_r, z_e)

        # Geographic (east, north) at the edge midpoint
        u_e_geo = u_r * cos_gamma
        v_n_geo = u_r * sin_gamma_neg_v
        # Project onto edge normal (TRiSK convention: angleEdge is the
        # angle of the normal away from local east)
        u_normal = u_e_geo * cos_angle + v_n_geo * sin_angle
        u_data = u_data.at[:, k].set(u_normal)

    phis_data = jnp.zeros((nCells,))
    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(
            data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2",
        ),
    )


def rotated_baroclinic_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate,
    *,
    perturbed: bool = True,
    alpha: float = jnp.pi / 4.0,
):
    """Rotated J-W in spectral (Gaussian) space."""
    import numpy as np

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import (
        sh_analysis,
        sh_analysis_3d,
        sh_analysis_dmu_3d,
        sh_analysis_oc2_3d,
    )

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_sh = grid.n_sh

    lat_2d = np.broadcast_to(np.array(grid.lat)[:, None], (n_lat, n_lon))
    lon_2d = np.array(grid.lon2d)

    u_east_np, v_north_np, T_np, _ = _build_rotated_state_dense_grid(
        jnp.asarray(lon_2d), jnp.asarray(lat_2d),
        sigma_coord, perturbed, float(alpha),
    )

    u_jax = jnp.asarray(u_east_np, dtype=jnp.float64)
    v_jax = jnp.asarray(v_north_np, dtype=jnp.float64)
    T_jax = jnp.asarray(T_np, dtype=jnp.float64)

    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_jax * cos_lat_3d
    v_cos = v_jax * cos_lat_3d

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    T_hat = sh_analysis_3d(grid, T_jax)
    lnps_grid = jnp.full((n_lat, n_lon), jnp.log(P0), dtype=jnp.float64)
    lnps_hat = sh_analysis(grid, lnps_grid)
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(
            data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2",
        ),
    )


# ---------------------------------------------------------------------------
# Steady-state (no perturbation) convenience wrappers — DCMIP §4-1 rotated
# ---------------------------------------------------------------------------


def rotated_steady_init(grid, sigma_coord, *, alpha=jnp.pi / 4.0):
    return rotated_baroclinic_init(
        grid, sigma_coord, perturbed=False, alpha=alpha,
    )


def rotated_steady_init_latlon(grid, sigma_coord, *, alpha=jnp.pi / 4.0):
    return rotated_baroclinic_init_latlon(
        grid, sigma_coord, perturbed=False, alpha=alpha,
    )


def rotated_steady_init_mpas(mesh, sigma_coord, *, alpha=jnp.pi / 4.0):
    return rotated_baroclinic_init_mpas(
        mesh, sigma_coord, perturbed=False, alpha=alpha,
    )


def rotated_steady_init_spectral(grid, sigma_coord, *, alpha=jnp.pi / 4.0):
    return rotated_baroclinic_init_spectral(
        grid, sigma_coord, perturbed=False, alpha=alpha,
    )
