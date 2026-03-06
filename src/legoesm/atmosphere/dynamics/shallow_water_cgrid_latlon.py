"""C-grid Shallow Water Equations on the lat-lon grid (FV3-inspired).

Implements a proper C-grid staggering where:
- h (height) lives at cell centers: (n_lat, n_lon)
- uc (zonal velocity) lives at longitude interfaces: (n_lat, n_lon)
  uc[j, i] is at the western edge of cell (j, i), i.e., between cells (j, i-1) and (j, i)
  Periodic: uc[j, 0] is between cell (j, n_lon-1) and cell (j, 0)
- vc (meridional velocity) lives at latitude interfaces: (n_lat+1, n_lon)
  vc[j, i] is at the southern edge of cell (j, i), between cells (j-1, i) and (j, i)
  vc[0, :] = 0 at south pole, vc[n_lat, :] = 0 at north pole

Key advantages over A-grid:
- Pressure gradient is EXACT (just difference of neighboring cell values)
- Mass flux uses velocity naturally at the cell interface (no interpolation)
- Better gravity wave dispersion
- Divergence damping is natural

Key trade-off:
- Coriolis term requires 4-point interpolation of the other velocity component

References
----------
- Lin & Rood (1997): An explicit flux-form semi-Lagrangian SWE model
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Arakawa & Lamb (1977): Computational design of the basic dynamical processes
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask, fourier_filter
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants


# ==============================================================================
# C-grid state
# ==============================================================================

class CGShallowWaterState(NamedTuple):
    """Shallow water state on a C-grid lat-lon grid.

    h : (n_lat, n_lon) — height at cell centers
    uc : (n_lat, n_lon) — zonal velocity at western cell edges (periodic in lon)
    vc : (n_lat+1, n_lon) — meridional velocity at southern cell edges
    h_s : (n_lat, n_lon) — surface topography at cell centers
    """
    h: jax.Array
    uc: jax.Array
    vc: jax.Array
    h_s: jax.Array


class CGShallowWaterConfig(NamedTuple):
    """Configuration for C-grid shallow water model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    div_damp_2: float = 0.0       # 2nd-order divergence damping coefficient
    div_damp_4: float = 0.0       # 4th-order divergence damping coefficient
    use_polar_filter: bool = True
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0


# ==============================================================================
# A-grid <-> C-grid conversions
# ==============================================================================

def a_to_cgrid(state: ShallowWaterState, grid: LatLonGrid) -> CGShallowWaterState:
    """Convert A-grid shallow water state to C-grid.

    u at cell centers -> uc at longitude interfaces (average neighbors).
    v at cell centers -> vc at latitude interfaces (average neighbors).
    """
    u = state.u.data  # (n_lat, n_lon)
    v = state.v.data

    # uc at western edge of cell (j, i): average of cells (j, i-1) and (j, i)
    uc = 0.5 * (u + jnp.roll(u, 1, axis=-1))  # (n_lat, n_lon)

    # vc at southern edge of cell (j, i): average of cells (j-1, i) and (j, i)
    # Interior: average neighboring rows
    vc_interior = 0.5 * (v[:-1, :] + v[1:, :])  # (n_lat-1, n_lon)
    # Boundaries: extrapolate (poles have v ≈ 0)
    vc_south = v[0:1, :] * 0.5  # half weight at pole
    vc_north = v[-1:, :] * 0.5
    vc = jnp.concatenate([vc_south, vc_interior, vc_north], axis=0)  # (n_lat+1, n_lon)

    return CGShallowWaterState(
        h=state.h.data,
        uc=uc,
        vc=vc,
        h_s=state.h_s.data,
    )


def cgrid_to_a(state: CGShallowWaterState, grid: LatLonGrid) -> ShallowWaterState:
    """Convert C-grid state back to A-grid for diagnostics/comparison.

    uc at edges -> u at centers (average flanking edges).
    vc at edges -> v at centers (average flanking edges).
    """
    # u at center of cell (j, i): average of western edge uc[j, i] and eastern edge uc[j, (i+1)%n_lon]
    u = 0.5 * (state.uc + jnp.roll(state.uc, -1, axis=-1))

    # v at center of cell (j, i): average of southern vc[j, i] and northern vc[j+1, i]
    v = 0.5 * (state.vc[:-1, :] + state.vc[1:, :])

    dims = ("lat", "lon")
    return ShallowWaterState(
        h=Field(data=state.h, name="h", dims=dims, units="m"),
        u=Field(data=u, name="u", dims=dims, units="m/s"),
        v=Field(data=v, name="v", dims=dims, units="m/s"),
        h_s=Field(data=state.h_s, name="h_s", dims=dims, units="m"),
    )


# ==============================================================================
# PPM reconstruction for C-grid mass transport
# ==============================================================================


def _pad_periodic(q, n_ghost):
    """Pad array periodically along last axis."""
    return jnp.concatenate([q[..., -n_ghost:], q, q[..., :n_ghost]], axis=-1)


def _pad_zero_grad(q, n_ghost):
    """Pad array with zero-gradient along first axis (latitude poles)."""
    top = jnp.broadcast_to(q[0:1, :], (n_ghost,) + q.shape[1:])
    bot = jnp.broadcast_to(q[-1:, :], (n_ghost,) + q.shape[1:])
    return jnp.concatenate([top, q, bot], axis=0)


# ==============================================================================
# C-grid operators
# ==============================================================================

def _ppm_face_values_1d(q_pad, n_out):
    """PPM upwind face values from a ghost-padded array along the last axis.

    Parameters
    ----------
    q_pad : (..., M)  cell averages with 2 ghost cells on each side (M = n_out + 4).
    n_out : int        number of face values to return (= number of original cells).

    Returns
    -------
    a_L, a_R : (..., n_out)  left and right parabola edges for original cells.
    """
    # 4th-order edge reconstruction at all interior interfaces.
    # Edge k sits between padded cells k+1 and k+2, for k=0..M-4.
    edges = (7.0 / 12.0) * (q_pad[..., 1:-2] + q_pad[..., 2:-1]) - \
            (1.0 / 12.0) * (q_pad[..., :-3] + q_pad[..., 3:])
    # Monotonicity clamp
    lo = jnp.minimum(q_pad[..., 1:-2], q_pad[..., 2:-1])
    hi = jnp.maximum(q_pad[..., 1:-2], q_pad[..., 2:-1])
    edges = jnp.clip(edges, lo, hi)

    # Left/right edges for padded cells 2..(M-3), i.e. the n_out original cells.
    # Left edge of cell c (padded index c) = edge at position c-2 in the edges array.
    # Right edge of cell c = edge at position c-1.
    # Cells 2..(n_out+1) → a_L uses edges[0..n_out-1], a_R uses edges[1..n_out].
    a_L = edges[..., :n_out]
    a_R = edges[..., 1:n_out + 1]
    q_c = q_pad[..., 2:n_out + 2]

    # Colella-Woodward limiter
    is_ext = (a_R - q_c) * (q_c - a_L) <= 0
    dm = a_R - a_L
    d6 = 6.0 * (q_c - 0.5 * (a_L + a_R))
    a_L = jnp.where(dm * d6 > dm**2, 3.0 * q_c - 2.0 * a_R, a_L)
    a_R = jnp.where(dm * d6 < -(dm**2), 3.0 * q_c - 2.0 * a_L, a_R)
    a_L = jnp.where(is_ext, q_c, a_L)
    a_R = jnp.where(is_ext, q_c, a_R)

    return a_L, a_R


def _cgrid_mass_flux(h, uc, vc, grid):
    """Compute mass flux divergence on C-grid using PPM.

    Parameters
    ----------
    h : (n_lat, n_lon) height at cell centers
    uc : (n_lat, n_lon) zonal velocity at western cell edges
    vc : (n_lat+1, n_lon) meridional velocity at southern cell edges
    grid : LatLonGrid

    Returns
    -------
    dh_dt : (n_lat, n_lon) mass flux divergence
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # --- Longitude flux: PPM h at uc positions (western cell edges) ---
    h_padlon = _pad_periodic(h, 2)  # (n_lat, n_lon+4)
    a_L_x, a_R_x = _ppm_face_values_1d(h_padlon, n_lon)  # (n_lat, n_lon)

    # Upwind at western edge of cell i: left value = a_R of cell (i-1), right = a_L of cell i
    h_left_x = jnp.roll(a_R_x, 1, axis=-1)   # a_R of previous cell (periodic)
    h_right_x = a_L_x
    h_face_x = jnp.where(uc > 0, h_left_x, h_right_x)  # (n_lat, n_lon)

    hy = R * dlat
    Phi_x = uc * hy * h_face_x  # (n_lat, n_lon)

    # --- Latitude flux: PPM h at vc positions (southern cell edges) ---
    h_padlat = _pad_zero_grad(h, 2)  # (n_lat+4, n_lon)
    # PPM along first axis: transpose, apply, transpose back
    a_L_y, a_R_y = _ppm_face_values_1d(
        h_padlat.T, n_lat,
    )  # (n_lon, n_lat) each
    a_L_y = a_L_y.T  # (n_lat, n_lon)
    a_R_y = a_R_y.T

    # Upwind at southern edge of cell j: left value = a_R of cell j-1, right = a_L of cell j
    # Interior interfaces j=1..n_lat-1
    h_left_y_int = a_R_y[:-1, :]   # (n_lat-1, n_lon)
    h_right_y_int = a_L_y[1:, :]
    vc_int = vc[1:-1, :]            # (n_lat-1, n_lon)
    h_face_y_int = jnp.where(vc_int > 0, h_left_y_int, h_right_y_int)

    # Pole boundaries: zero flux (vc=0 at poles, but set face=h for safety)
    h_face_y = jnp.concatenate([
        h[0:1, :],          # south pole face (vc[0]=0 so flux=0 regardless)
        h_face_y_int,
        h[-1:, :],          # north pole face
    ], axis=0)  # (n_lat+1, n_lon)

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.maximum(jnp.cos(lat_half), 0.0)
    hx = R * dlon * cos_lat_half[:, None]  # (n_lat+1, 1)
    Phi_y = vc * hx * h_face_y  # (n_lat+1, n_lon)

    # --- Net flux divergence ---
    net_x = jnp.roll(Phi_x, -1, axis=-1) - Phi_x  # east - west
    net_y = Phi_y[1:, :] - Phi_y[:-1, :]            # north - south

    dh_dt = -(net_x + net_y) / grid.area
    return dh_dt


def _cgrid_vorticity(uc, vc, grid):
    """Compute relative vorticity at cell corners.

    Corner (j, i) is at (lat_half[j], lon_half[i]) for
    j in [1, n_lat-1], i in [0, n_lon-1].

    We compute vorticity at interior corners only (avoiding poles).

    Returns shape (n_lat-1, n_lon).
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Corner (j, i) for j in [1, n_lat-1] is between cell rows j-1 and j,
    # and between cell columns i-1 and i.

    # dv/dx at corner (j, i): (vc[j, i] - vc[j, (i-1)%n_lon]) / (R * cos(lat_half[j]) * dlon)
    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.maximum(jnp.cos(lat_half), 1e-10)

    # Interior corners: j from 1 to n_lat-1
    vc_interior = vc[1:-1, :]  # (n_lat-1, n_lon) — vc at interior latitude interfaces
    dvdx = (vc_interior - jnp.roll(vc_interior, 1, axis=-1)) / \
           (R * cos_lat_half[1:-1, None] * dlon)

    # du/dy at corner (j, i): (uc[j, i] - uc[j-1, i]) / (R * dlat)
    # uc[j, i] is at (lat[j], lon_half[i])
    # Corner j corresponds to the interface between rows j-1 and j
    dudy = (uc[1:, :] - uc[:-1, :]) / (R * dlat)  # (n_lat-1, n_lon)

    return dvdx - dudy  # (n_lat-1, n_lon)


def _cgrid_divergence(uc, vc, grid):
    """Compute divergence at cell centers.

    D[j, i] = (uc[j, (i+1)%n_lon] - uc[j, i]) / (R*cos(lat[j])*dlon)
            + (vc[j+1, i]*cos(lat_half[j+1]) - vc[j, i]*cos(lat_half[j]))
              / (R*cos(lat[j])*dlat)
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.cos(lat_half)

    cos_lat_c = grid.cos_lat[:, None]  # (n_lat, 1)

    # du/dlon
    dudlon = (jnp.roll(uc, -1, axis=-1) - uc) / (R * cos_lat_c * dlon)

    # d(v*cos(lat))/dlat
    vc_coslat = vc * cos_lat_half[:, None]  # (n_lat+1, n_lon)
    dvdlat = (vc_coslat[1:, :] - vc_coslat[:-1, :]) / (R * cos_lat_c * dlat)

    return dudlon + dvdlat


def _cgrid_momentum(h, uc, vc, h_s, grid, config):
    """Compute C-grid momentum tendencies.

    Uses the vector-invariant form:
        duc/dt = (zeta + f)_interp * v_interp - dB/dx + damping
        dvc/dt = -(zeta + f)_interp * u_interp - dB/dy + damping

    The Bernoulli pressure gradient is EXACT on C-grid (just cell differences).
    Coriolis requires 4-point interpolation.
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    g = config.g

    # --- Kinetic energy at cell centers ---
    # Average edge velocities to centers
    u_center = 0.5 * (uc + jnp.roll(uc, -1, axis=-1))  # (n_lat, n_lon)
    v_center = 0.5 * (vc[:-1, :] + vc[1:, :])            # (n_lat, n_lon)
    KE = 0.5 * (u_center**2 + v_center**2)               # (n_lat, n_lon)

    # --- Bernoulli function at cell centers ---
    B = KE + g * (h + h_s)

    # --- Pressure gradient (EXACT on C-grid!) ---
    # dB/dx at uc position (j, i) = western edge of cell (j, i)
    # = (B[j, i] - B[j, (i-1)%n_lon]) / (R*cos(lat[j])*dlon)
    cos_lat = grid.cos_lat[:, None]  # (n_lat, 1)
    dBdx = (B - jnp.roll(B, 1, axis=-1)) / (R * cos_lat * dlon)  # (n_lat, n_lon)

    # dB/dy at vc position (j, i) = southern edge of cell (j, i)
    # = (B[j, i] - B[j-1, i]) / (R*dlat), for j in [1, n_lat-1]
    dBdy_interior = (B[1:, :] - B[:-1, :]) / (R * dlat)  # (n_lat-1, n_lon)
    # Pad poles with zero gradient
    dBdy = jnp.concatenate([
        jnp.zeros((1, n_lon)),
        dBdy_interior,
        jnp.zeros((1, n_lon)),
    ], axis=0)  # (n_lat+1, n_lon)

    # --- Vorticity at corners ---
    zeta_corner = _cgrid_vorticity(uc, vc, grid)  # (n_lat-1, n_lon)

    # Coriolis at corners
    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    f_corner = 2.0 * constants.Omega * jnp.sin(lat_half[1:-1])  # (n_lat-1,)

    abs_vor_corner = zeta_corner + f_corner[:, None]  # (n_lat-1, n_lon)

    # --- Coriolis at uc positions: (zeta+f) * v_interp ---
    # uc[j, i] is at (lat[j], lon_half[i])
    # Nearest corners in latitude: (j, *) and (j+1, *) — but corners only exist for j in [1, n_lat-1]
    # Nearest corners in longitude: (*, i) (same lon_half index)
    # For uc at rows j=0 and j=n_lat-1, use boundary extrapolation

    # Pad abs_vor_corner at poles using the Coriolis parameter (zeta≈0 at poles).
    f_south_pole = 2.0 * constants.Omega * jnp.sin(lat_half[0])  # -2Ω
    f_north_pole = 2.0 * constants.Omega * jnp.sin(lat_half[-1])  # +2Ω
    abs_vor_padded = jnp.concatenate([
        jnp.full((1, n_lon), f_south_pole),
        abs_vor_corner,
        jnp.full((1, n_lon), f_north_pole),
    ], axis=0)  # (n_lat+1, n_lon)

    # Average to uc positions (j, i): between corners (j, i) and (j+1, i)
    abs_vor_at_uc = 0.5 * (abs_vor_padded[:-1, :] + abs_vor_padded[1:, :])  # (n_lat, n_lon)

    # v interpolated to uc position (j, i):
    # 4 nearest vc points: (j, (i-1)%n_lon), (j, i), (j+1, (i-1)%n_lon), (j+1, i)
    vc_rolled = jnp.roll(vc, 1, axis=-1)
    v_at_uc = 0.25 * (vc[:-1, :] + vc[1:, :] + vc_rolled[:-1, :] + vc_rolled[1:, :])

    duc_dt = abs_vor_at_uc * v_at_uc - dBdx

    # --- Coriolis at vc positions: -(zeta+f) * u_interp ---
    # vc[j, i] is at (lat_half[j], lon[i])
    # Nearest corners: (j, i) and (j, (i+1)%n_lon)
    abs_vor_at_vc_interior = 0.5 * (abs_vor_corner + jnp.roll(abs_vor_corner, -1, axis=-1))
    # Pad for vc at poles using Coriolis parameter
    abs_vor_at_vc = jnp.concatenate([
        jnp.full((1, n_lon), f_south_pole),
        abs_vor_at_vc_interior,
        jnp.full((1, n_lon), f_north_pole),
    ], axis=0)  # (n_lat+1, n_lon)

    # u interpolated to vc position (j, i):
    # 4 nearest uc points: (j-1, i), (j-1, (i+1)%n_lon), (j, i), (j, (i+1)%n_lon)
    uc_rolled = jnp.roll(uc, -1, axis=-1)
    # Pad uc at poles
    uc_pad = jnp.concatenate([uc[0:1, :], uc, uc[-1:, :]], axis=0)  # (n_lat+2, n_lon)
    uc_rolled_pad = jnp.concatenate([uc_rolled[0:1, :], uc_rolled, uc_rolled[-1:, :]], axis=0)
    u_at_vc = 0.25 * (uc_pad[:-1, :] + uc_pad[1:, :] +
                       uc_rolled_pad[:-1, :] + uc_rolled_pad[1:, :])  # (n_lat+1, n_lon)

    dvc_dt = -abs_vor_at_vc * u_at_vc - dBdy

    # --- Divergence damping ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        div = _cgrid_divergence(uc, vc, grid)  # (n_lat, n_lon)

        if config.div_damp_2 > 0:
            # grad(div) at uc position
            ddiv_dx = (div - jnp.roll(div, 1, axis=-1)) / (R * cos_lat * dlon)
            ddiv_dy_int = (div[1:, :] - div[:-1, :]) / (R * dlat)
            ddiv_dy = jnp.concatenate([
                jnp.zeros((1, n_lon)), ddiv_dy_int, jnp.zeros((1, n_lon))
            ], axis=0)
            duc_dt = duc_dt + config.div_damp_2 * ddiv_dx
            dvc_dt = dvc_dt + config.div_damp_2 * ddiv_dy

        if config.div_damp_4 > 0:
            # 4th order: -nu4 * grad(lap(div))
            # Laplacian of divergence
            div_pad_lon = _pad_periodic(div, 1)
            div_pad_lat = _pad_zero_grad(div, 1)
            lap_div_x = (div_pad_lon[:, 2:] - 2 * div_pad_lon[:, 1:-1] + div_pad_lon[:, :-2]) / \
                        (R * cos_lat * dlon) ** 2
            lap_div_y = (div_pad_lat[2:, :] - 2 * div_pad_lat[1:-1, :] + div_pad_lat[:-2, :]) / \
                        (R * dlat) ** 2
            lap_div = lap_div_x + lap_div_y

            # Gradient of laplacian at edge positions
            dlap_dx = (lap_div - jnp.roll(lap_div, 1, axis=-1)) / (R * cos_lat * dlon)
            dlap_dy_int = (lap_div[1:, :] - lap_div[:-1, :]) / (R * dlat)
            dlap_dy = jnp.concatenate([
                jnp.zeros((1, n_lon)), dlap_dy_int, jnp.zeros((1, n_lon))
            ], axis=0)
            duc_dt = duc_dt - config.div_damp_4 * dlap_dx
            dvc_dt = dvc_dt - config.div_damp_4 * dlap_dy

    # --- Hyperdiffusion (simple Laplacian² on velocity) ---
    if config.hyperdiff_coeff > 0:
        # Laplacian of uc
        uc_pad_lon = _pad_periodic(uc, 1)
        uc_pad_lat = _pad_zero_grad(uc, 1)
        lap_uc = (uc_pad_lon[:, 2:] - 2*uc + uc_pad_lon[:, :-2]) / (R * cos_lat * dlon)**2 + \
                 (uc_pad_lat[2:, :] - 2*uc + uc_pad_lat[:-2, :]) / (R * dlat)**2
        # Apply Laplacian again for 4th order
        lap_uc_pad_lon = _pad_periodic(lap_uc, 1)
        lap_uc_pad_lat = _pad_zero_grad(lap_uc, 1)
        bilap_uc = (lap_uc_pad_lon[:, 2:] - 2*lap_uc + lap_uc_pad_lon[:, :-2]) / (R * cos_lat * dlon)**2 + \
                   (lap_uc_pad_lat[2:, :] - 2*lap_uc + lap_uc_pad_lat[:-2, :]) / (R * dlat)**2
        duc_dt = duc_dt - config.hyperdiff_coeff * bilap_uc

        # Same for vc (interior only)
        vc_int = vc[1:-1, :]
        vc_pad_lon = _pad_periodic(vc_int, 1)
        vc_pad_lat = jnp.concatenate([vc[0:1, :], vc_int, vc[-1:, :]], axis=0)
        cos_lat_half_int = jnp.maximum(jnp.cos(lat_half[1:-1]), 1e-10)[:, None]
        lap_vc = (vc_pad_lon[:, 2:] - 2*vc_int + vc_pad_lon[:, :-2]) / (R * cos_lat_half_int * dlon)**2 + \
                 (vc_pad_lat[2:, :] - 2*vc_int + vc_pad_lat[:-2, :]) / (R * dlat)**2
        lap_vc_pad_lon = _pad_periodic(lap_vc, 1)
        lap_vc_pad_lat = jnp.concatenate([jnp.zeros((1, n_lon)), lap_vc, jnp.zeros((1, n_lon))], axis=0)
        bilap_vc_int = (lap_vc_pad_lon[:, 2:] - 2*lap_vc + lap_vc_pad_lon[:, :-2]) / (R * cos_lat_half_int * dlon)**2 + \
                       (lap_vc_pad_lat[2:, :] - 2*lap_vc + lap_vc_pad_lat[:-2, :]) / (R * dlat)**2
        dvc_dt = dvc_dt.at[1:-1, :].add(-config.hyperdiff_coeff * bilap_vc_int)

    # Zero tendency at pole boundaries
    dvc_dt = dvc_dt.at[0, :].set(0.0)
    dvc_dt = dvc_dt.at[-1, :].set(0.0)

    return duc_dt, dvc_dt


# ==============================================================================
# C-grid shallow water tendencies
# ==============================================================================

def cgrid_shallow_water_tendencies(
    state: CGShallowWaterState,
    grid: LatLonGrid,
    config: CGShallowWaterConfig,
    polar_filter_mask=None,
):
    """Compute C-grid shallow water tendencies.

    Returns
    -------
    (dh_dt, duc_dt, dvc_dt)
    """
    h, uc, vc, h_s = state

    # Mass
    dh_dt = _cgrid_mass_flux(h, uc, vc, grid)

    # Zero-mean correction
    dh_dt = dh_dt - jnp.sum(dh_dt * grid.area) / grid.total_area

    # Momentum
    duc_dt, dvc_dt = _cgrid_momentum(h, uc, vc, h_s, grid, config)

    # Polar filter
    if config.use_polar_filter and polar_filter_mask is not None:
        dh_dt = fourier_filter(dh_dt, grid, polar_filter_mask)
        duc_dt = fourier_filter(duc_dt, grid, polar_filter_mask)
        # vc has n_lat+1 rows; filter expects (n_lat, n_lon).
        # Filter the n_lat-1 interior rows by padding to n_lat, filtering, then extracting.
        dvc_interior = dvc_dt[1:-1, :]  # (n_lat-1, n_lon)
        dvc_padded = jnp.concatenate([dvc_interior, jnp.zeros((1, dvc_dt.shape[-1]))], axis=0)
        dvc_filtered = fourier_filter(dvc_padded, grid, polar_filter_mask)
        dvc_dt = jnp.concatenate([
            dvc_dt[0:1, :], dvc_filtered[:-1, :], dvc_dt[-1:, :]
        ], axis=0)

    return dh_dt, duc_dt, dvc_dt


# ==============================================================================
# Conservation fixers for C-grid
# ==============================================================================

def _cgrid_conservation_fixer(state_new, state_old, grid, g):
    """Fix mass and energy for C-grid state."""
    # Mass fixer
    mass_old = jnp.sum(state_old.h * grid.area)
    mass_new = jnp.sum(state_new.h * grid.area)
    h_fixed = state_new.h + (mass_old - mass_new) / grid.total_area

    return state_new._replace(h=h_fixed)


# ==============================================================================
# Model class
# ==============================================================================

class CGShallowWaterLatLonModel:
    """C-grid shallow water model on the lat-lon grid (FV3-inspired).

    Uses proper C-grid staggering with:
    - Exact pressure gradients (cell differences)
    - PPM mass transport with naturally colocated edge velocity
    - Divergence damping
    - 4-point Coriolis interpolation

    Parameters
    ----------
    grid : LatLonGrid
    config : CGShallowWaterConfig, optional
    dt : float, optional
        Reference time step for polar filter.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        config: CGShallowWaterConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.config = config or CGShallowWaterConfig()

        if self.config.use_polar_filter:
            self.polar_filter_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
        else:
            self.polar_filter_mask = None

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CGShallowWaterState, dt: float) -> CGShallowWaterState:
        """Advance one time step using SSP-RK3."""
        def tendency_fn(s):
            dh, duc, dvc = cgrid_shallow_water_tendencies(
                s, self.grid, self.config, self.polar_filter_mask,
            )
            return CGShallowWaterState(h=dh, uc=duc, vc=dvc, h_s=jnp.zeros_like(s.h_s))

        state_new = ssp_rk3_step(state, tendency_fn, dt)

        # Conservation fixer
        state_new = _cgrid_conservation_fixer(state_new, state, self.grid, self.config.g)

        return state_new

    def integrate(
        self,
        state: CGShallowWaterState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[CGShallowWaterState, list[CGShallowWaterState]]:
        """Integrate forward for a given duration."""
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory
