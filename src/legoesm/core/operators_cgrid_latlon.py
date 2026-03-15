"""C-grid operators on the lat-lon grid.

Provides PPM reconstruction, mass flux, vorticity, divergence, and
momentum tendency computation for a C-grid staggering on lat-lon:

- h : (n_lat, n_lon)   at cell centers
- uc: (n_lat, n_lon)   at western cell edges (periodic in lon)
- vc: (n_lat+1, n_lon) at southern cell edges

References
----------
- Lin & Rood (1997): An explicit flux-form semi-Lagrangian SWE model
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def pad_periodic(q, n_ghost):
    """Pad array periodically along last axis."""
    return jnp.concatenate([q[..., -n_ghost:], q, q[..., :n_ghost]], axis=-1)


def pad_zero_grad(q, n_ghost):
    """Pad array with zero-gradient along first axis (latitude poles)."""
    top = jnp.broadcast_to(q[0:1, :], (n_ghost,) + q.shape[1:])
    bot = jnp.broadcast_to(q[-1:, :], (n_ghost,) + q.shape[1:])
    return jnp.concatenate([top, q, bot], axis=0)


def ppm_face_values_1d(q_pad, n_out):
    """PPM upwind face values from a ghost-padded array along the last axis.

    This is a simplified 4th-order PPM with Colella-Woodward limiting,
    designed for C-grid face values where velocity is colocated at the
    interface. It differs from _ppm_edge_values in operators_fv.py which
    is designed for A-grid PPM reconstruction with halo=2 stencils.

    Parameters
    ----------
    q_pad : (..., M)  cell averages with 2 ghost cells on each side (M = n_out + 4).
    n_out : int        number of face values to return (= number of original cells).

    Returns
    -------
    a_L, a_R : (..., n_out)  left and right parabola edges for original cells.
    """
    edges = (7.0 / 12.0) * (q_pad[..., 1:-2] + q_pad[..., 2:-1]) - \
            (1.0 / 12.0) * (q_pad[..., :-3] + q_pad[..., 3:])
    lo = jnp.minimum(q_pad[..., 1:-2], q_pad[..., 2:-1])
    hi = jnp.maximum(q_pad[..., 1:-2], q_pad[..., 2:-1])
    edges = jnp.clip(edges, lo, hi)

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


def cgrid_mass_flux(h, uc, vc, grid):
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

    # --- Longitude flux: PPM h at uc positions ---
    h_padlon = pad_periodic(h, 2)
    a_L_x, a_R_x = ppm_face_values_1d(h_padlon, n_lon)

    h_left_x = jnp.roll(a_R_x, 1, axis=-1)
    h_right_x = a_L_x
    h_face_x = jnp.where(uc > 0, h_left_x, h_right_x)

    hy = R * dlat
    Phi_x = uc * hy * h_face_x

    # --- Latitude flux: PPM h at vc positions ---
    h_padlat = pad_zero_grad(h, 2)
    a_L_y, a_R_y = ppm_face_values_1d(h_padlat.T, n_lat)
    a_L_y = a_L_y.T
    a_R_y = a_R_y.T

    h_left_y_int = a_R_y[:-1, :]
    h_right_y_int = a_L_y[1:, :]
    vc_int = vc[1:-1, :]
    h_face_y_int = jnp.where(vc_int > 0, h_left_y_int, h_right_y_int)

    h_face_y = jnp.concatenate([
        h[0:1, :],
        h_face_y_int,
        h[-1:, :],
    ], axis=0)

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.maximum(jnp.cos(lat_half), 0.0)
    hx = R * dlon * cos_lat_half[:, None]
    Phi_y = vc * hx * h_face_y

    # --- Net flux divergence ---
    net_x = jnp.roll(Phi_x, -1, axis=-1) - Phi_x
    net_y = Phi_y[1:, :] - Phi_y[:-1, :]

    dh_dt = -(net_x + net_y) / grid.area
    return dh_dt


def cgrid_vorticity(uc, vc, grid):
    """Compute relative vorticity at cell corners.

    Returns shape (n_lat-1, n_lon).
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.maximum(jnp.cos(lat_half), 1e-10)

    vc_interior = vc[1:-1, :]
    dvdx = (vc_interior - jnp.roll(vc_interior, 1, axis=-1)) / \
           (R * cos_lat_half[1:-1, None] * dlon)

    dudy = (uc[1:, :] - uc[:-1, :]) / (R * dlat)

    return dvdx - dudy


def cgrid_divergence(uc, vc, grid):
    """Compute divergence at cell centers.

    Returns shape (n_lat, n_lon).
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    cos_lat_half = jnp.cos(lat_half)
    cos_lat_c = grid.cos_lat[:, None]

    dudlon = (jnp.roll(uc, -1, axis=-1) - uc) / (R * cos_lat_c * dlon)

    vc_coslat = vc * cos_lat_half[:, None]
    dvdlat = (vc_coslat[1:, :] - vc_coslat[:-1, :]) / (R * cos_lat_c * dlat)

    return dudlon + dvdlat


def cgrid_momentum(h, uc, vc, h_s, grid, config):
    """Compute C-grid momentum tendencies.

    Uses the vector-invariant form with exact C-grid pressure gradient
    and 4-point Coriolis interpolation.

    Parameters
    ----------
    h, h_s : (n_lat, n_lon)
    uc : (n_lat, n_lon)
    vc : (n_lat+1, n_lon)
    grid : LatLonGrid
    config : CGShallowWaterConfig

    Returns
    -------
    duc_dt : (n_lat, n_lon)
    dvc_dt : (n_lat+1, n_lon)
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    g = config.g

    # --- Kinetic energy at cell centers ---
    u_center = 0.5 * (uc + jnp.roll(uc, -1, axis=-1))
    v_center = 0.5 * (vc[:-1, :] + vc[1:, :])
    KE = 0.5 * (u_center**2 + v_center**2)

    # --- Bernoulli function ---
    B = KE + g * (h + h_s)

    # --- Pressure gradient (EXACT on C-grid) ---
    cos_lat = grid.cos_lat[:, None]
    dBdx = (B - jnp.roll(B, 1, axis=-1)) / (R * cos_lat * dlon)

    dBdy_interior = (B[1:, :] - B[:-1, :]) / (R * dlat)
    dBdy = jnp.concatenate([
        jnp.zeros((1, n_lon)),
        dBdy_interior,
        jnp.zeros((1, n_lon)),
    ], axis=0)

    # --- Vorticity at corners ---
    zeta_corner = cgrid_vorticity(uc, vc, grid)

    lat_half = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n_lat + 1)
    f_corner = 2.0 * constants.Omega * jnp.sin(lat_half[1:-1])
    abs_vor_corner = zeta_corner + f_corner[:, None]

    # --- Coriolis at uc positions ---
    f_south_pole = 2.0 * constants.Omega * jnp.sin(lat_half[0])
    f_north_pole = 2.0 * constants.Omega * jnp.sin(lat_half[-1])
    abs_vor_padded = jnp.concatenate([
        jnp.full((1, n_lon), f_south_pole),
        abs_vor_corner,
        jnp.full((1, n_lon), f_north_pole),
    ], axis=0)

    abs_vor_at_uc = 0.5 * (abs_vor_padded[:-1, :] + abs_vor_padded[1:, :])

    vc_rolled = jnp.roll(vc, 1, axis=-1)
    v_at_uc = 0.25 * (vc[:-1, :] + vc[1:, :] + vc_rolled[:-1, :] + vc_rolled[1:, :])

    duc_dt = abs_vor_at_uc * v_at_uc - dBdx

    # --- Coriolis at vc positions ---
    abs_vor_at_vc_interior = 0.5 * (abs_vor_corner + jnp.roll(abs_vor_corner, -1, axis=-1))
    abs_vor_at_vc = jnp.concatenate([
        jnp.full((1, n_lon), f_south_pole),
        abs_vor_at_vc_interior,
        jnp.full((1, n_lon), f_north_pole),
    ], axis=0)

    uc_rolled = jnp.roll(uc, -1, axis=-1)
    uc_pad = jnp.concatenate([uc[0:1, :], uc, uc[-1:, :]], axis=0)
    uc_rolled_pad = jnp.concatenate([uc_rolled[0:1, :], uc_rolled, uc_rolled[-1:, :]], axis=0)
    u_at_vc = 0.25 * (uc_pad[:-1, :] + uc_pad[1:, :] +
                       uc_rolled_pad[:-1, :] + uc_rolled_pad[1:, :])

    dvc_dt = -abs_vor_at_vc * u_at_vc - dBdy

    # --- Divergence damping ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        div = cgrid_divergence(uc, vc, grid)

        if config.div_damp_2 > 0:
            ddiv_dx = (div - jnp.roll(div, 1, axis=-1)) / (R * cos_lat * dlon)
            ddiv_dy_int = (div[1:, :] - div[:-1, :]) / (R * dlat)
            ddiv_dy = jnp.concatenate([
                jnp.zeros((1, n_lon)), ddiv_dy_int, jnp.zeros((1, n_lon))
            ], axis=0)
            duc_dt = duc_dt + config.div_damp_2 * ddiv_dx
            dvc_dt = dvc_dt + config.div_damp_2 * ddiv_dy

        if config.div_damp_4 > 0:
            div_pad_lon = pad_periodic(div, 1)
            div_pad_lat = pad_zero_grad(div, 1)
            lap_div_x = (div_pad_lon[:, 2:] - 2 * div_pad_lon[:, 1:-1] + div_pad_lon[:, :-2]) / \
                        (R * cos_lat * dlon) ** 2
            lap_div_y = (div_pad_lat[2:, :] - 2 * div_pad_lat[1:-1, :] + div_pad_lat[:-2, :]) / \
                        (R * dlat) ** 2
            lap_div = lap_div_x + lap_div_y

            dlap_dx = (lap_div - jnp.roll(lap_div, 1, axis=-1)) / (R * cos_lat * dlon)
            dlap_dy_int = (lap_div[1:, :] - lap_div[:-1, :]) / (R * dlat)
            dlap_dy = jnp.concatenate([
                jnp.zeros((1, n_lon)), dlap_dy_int, jnp.zeros((1, n_lon))
            ], axis=0)
            duc_dt = duc_dt - config.div_damp_4 * dlap_dx
            dvc_dt = dvc_dt - config.div_damp_4 * dlap_dy

    # --- Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        uc_pad_lon = pad_periodic(uc, 1)
        uc_pad_lat = pad_zero_grad(uc, 1)
        lap_uc = (uc_pad_lon[:, 2:] - 2*uc + uc_pad_lon[:, :-2]) / (R * cos_lat * dlon)**2 + \
                 (uc_pad_lat[2:, :] - 2*uc + uc_pad_lat[:-2, :]) / (R * dlat)**2
        lap_uc_pad_lon = pad_periodic(lap_uc, 1)
        lap_uc_pad_lat = pad_zero_grad(lap_uc, 1)
        bilap_uc = (lap_uc_pad_lon[:, 2:] - 2*lap_uc + lap_uc_pad_lon[:, :-2]) / (R * cos_lat * dlon)**2 + \
                   (lap_uc_pad_lat[2:, :] - 2*lap_uc + lap_uc_pad_lat[:-2, :]) / (R * dlat)**2
        duc_dt = duc_dt - config.hyperdiff_coeff * bilap_uc

        vc_int = vc[1:-1, :]
        vc_pad_lon = pad_periodic(vc_int, 1)
        vc_pad_lat = jnp.concatenate([vc[0:1, :], vc_int, vc[-1:, :]], axis=0)
        cos_lat_half_int = jnp.maximum(jnp.cos(lat_half[1:-1]), 1e-10)[:, None]
        lap_vc = (vc_pad_lon[:, 2:] - 2*vc_int + vc_pad_lon[:, :-2]) / (R * cos_lat_half_int * dlon)**2 + \
                 (vc_pad_lat[2:, :] - 2*vc_int + vc_pad_lat[:-2, :]) / (R * dlat)**2
        lap_vc_pad_lon = pad_periodic(lap_vc, 1)
        lap_vc_pad_lat = jnp.concatenate([jnp.zeros((1, n_lon)), lap_vc, jnp.zeros((1, n_lon))], axis=0)
        bilap_vc_int = (lap_vc_pad_lon[:, 2:] - 2*lap_vc + lap_vc_pad_lon[:, :-2]) / (R * cos_lat_half_int * dlon)**2 + \
                       (lap_vc_pad_lat[2:, :] - 2*lap_vc + lap_vc_pad_lat[:-2, :]) / (R * dlat)**2
        dvc_dt = dvc_dt.at[1:-1, :].add(-config.hyperdiff_coeff * bilap_vc_int)

    # Zero tendency at pole boundaries
    dvc_dt = dvc_dt.at[0, :].set(0.0)
    dvc_dt = dvc_dt.at[-1, :].set(0.0)

    return duc_dt, dvc_dt
