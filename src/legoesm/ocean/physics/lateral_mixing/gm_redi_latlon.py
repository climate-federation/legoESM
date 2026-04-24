"""GM/Redi isopycnal mixing for lat-lon C-grid ocean.

Adapts the cubed-sphere GM/Redi (gm_redi.py) for the lat-lon grid by
replacing cubed-sphere gradient/divergence operators with their lat-lon
C-grid equivalents. The physics is identical.

**Tensor-consistent discretization**: All components of the Redi/GM
diffusion tensor use the SAME compact C-grid gradient operator
(gradient at faces, divergence back to centers). The horizontal flux
is assembled at faces (where the C-grid gradient naturally lives) and
a single ``divergence_cgrid`` call produces the tendency. This ensures
the discrete operator inherits the positive semi-definiteness of the
continuous Redi tensor, preventing the anti-diffusion that arises when
the diagonal (compact stencil) and off-diagonal/vertical (2dx stencil)
parts use inconsistent gradient operators.

See gm_redi.py for the mathematical formulation and references.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)


def _compute_tapered_slopes_latlon(
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    cfg: GMRediConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute isopycnal slopes at interfaces with DM95 tapering.

    Parameters
    ----------
    rho : (n_lat, n_lon, nlev) in-situ density.
    z_coord : vertical coordinate.
    jacobian : (n_lat, n_lon) z-star Jacobian.
    grid : LatLonGrid.
    cfg : GMRediConfig.

    Returns
    -------
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    taper : (n_lat, n_lon, nlev-1)
        Taper factor in [0, 1].
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Horizontal density gradients at full levels (cell centers -> faces -> back)
    # gradient_x_cgrid returns at u-faces; we need at cell centers.
    # Average adjacent faces to get the 2dx centered gradient at centers.
    drho_dx_faces = gradient_x_cgrid(rho, grid)  # (n_lat, n_lon+1, nlev)
    drho_dx = 0.5 * (drho_dx_faces[:, :-1, :] + drho_dx_faces[:, 1:, :])

    drho_dy_faces = gradient_y_cgrid(rho, grid)  # (n_lat+1, n_lon, nlev)
    drho_dy = 0.5 * (drho_dy_faces[:-1, :, :] + drho_dy_faces[1:, :, :])

    # Average to interfaces (between level k and k+1)
    drho_dx_half = 0.5 * (drho_dx[..., :-1] + drho_dx[..., 1:])
    drho_dy_half = 0.5 * (drho_dy[..., :-1] + drho_dy[..., 1:])

    # Vertical density gradient at interfaces
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / jnp.maximum(dz_half, eps)
    # Must be negative for stable stratification
    drho_dz_safe = jnp.minimum(drho_dz, -eps)

    # Slopes: S_x = -(drho/dx) / (drho/dz)
    S_x = jnp.clip(-drho_dx_half / drho_dz_safe, -cfg.S_max, cfg.S_max)
    S_y = jnp.clip(-drho_dy_half / drho_dz_safe, -cfg.S_max, cfg.S_max)

    # DM95 tapering: smooth taper near S_max
    S_mag = jnp.sqrt(S_x**2 + S_y**2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh((cfg.S_max - S_mag) / (0.1 * cfg.S_max + eps)))

    S_x = S_x * taper
    S_y = S_y * taper

    return S_x, S_y, taper


def _tracer_tendency_gm_redi_latlon(
    q: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_GM: float,
    kappa_Redi: float,
) -> jnp.ndarray:
    """Compute GM+Redi tendency for a single tracer on lat-lon grid.

    **Tensor-consistent C-grid discretization.**

    The full horizontal flux is assembled at face locations (where the
    compact C-grid gradient lives) and passed to a single
    ``divergence_cgrid`` call. This avoids mixing the compact 1dx
    stencil (``laplacian_cgrid``) with the wider 2dx stencil
    (face-averaged-to-center gradients used in the vertical flux),
    which would break the positive semi-definiteness of the discrete
    Redi tensor and cause anti-diffusion on grid-scale modes.

    Horizontal flux at u-faces (x-direction):
        F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * Sx * dq/dz

    Horizontal flux at v-faces (y-direction):
        F_y = kappa_Redi * dq/dy + (kappa_Redi - kappa_GM) * Sy * dq/dz

    dq/dx is naturally at u-faces; Sx and dq/dz are at cell centers
    (interface levels) and interpolated to faces.

    Vertical flux at interfaces:
        F_z = (kR + kG) * (Sx*dq/dx + Sy*dq/dy) + kR * S^2 * dq/dz

    where the horizontal gradients are the SAME face-averaged-to-center
    gradients used to compute slopes.

    Parameters
    ----------
    q : (n_lat, n_lon, nlev) tracer field.
    S_x, S_y : (n_lat, n_lon, nlev-1) tapered isopycnal slopes.

    Returns
    -------
    dq_dt : (n_lat, n_lon, nlev) tendency.
    """
    eps = _EPS
    nlev = q.shape[-1]
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # --- Compact C-grid gradient at faces ---
    dq_dx_u = gradient_x_cgrid(q, grid)   # (n_lat, n_lon+1, nlev)
    dq_dy_v = gradient_y_cgrid(q, grid)   # (n_lat+1, n_lon, nlev)

    # --- Horizontal gradients at cell centers (for vertical flux) ---
    dq_dx_c = 0.5 * (dq_dx_u[:, :-1, :] + dq_dx_u[:, 1:, :])
    dq_dy_c = 0.5 * (dq_dy_v[:-1, :, :] + dq_dy_v[1:, :, :])

    # --- Vertical tracer gradient at interfaces ---
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    dq_dz_half = (q[..., :-1] - q[..., 1:]) / jnp.maximum(dz_half, eps)

    # --- Average horizontal gradients to interfaces (for vertical flux) ---
    dq_dx_half = 0.5 * (dq_dx_c[..., :-1] + dq_dx_c[..., 1:])
    dq_dy_half = 0.5 * (dq_dy_c[..., :-1] + dq_dy_c[..., 1:])

    # =====================================================================
    # Horizontal tendency: div(F_x, F_y)
    #
    # Full horizontal flux (Griffies 1998, small-slope):
    #   F_x = kR * dq/dx + (kR - kG) * Sx * dq/dz
    #   F_y = kR * dq/dy + (kR - kG) * Sy * dq/dz
    #
    # The diagonal part (kR * dq/dx) is already at faces. The off-diagonal
    # part (Sx * dq/dz) lives at cell centers (interface levels), so we
    # average it to full levels then interpolate to faces.
    # =====================================================================

    # Off-diagonal: (kR - kG) * S * dq/dz at interfaces, averaged to full levels
    off_coeff = kappa_Redi - kappa_GM
    off_diag_x = off_coeff * S_x * dq_dz_half  # (n_lat, n_lon, nlev-1)
    off_diag_y = off_coeff * S_y * dq_dz_half

    # Average interface -> full levels (zero at surface/bottom boundaries)
    z_pad = jnp.zeros((*off_diag_x.shape[:-1], 1), dtype=off_diag_x.dtype)
    off_x_full = 0.5 * (
        jnp.concatenate([z_pad, off_diag_x], axis=-1)
        + jnp.concatenate([off_diag_x, z_pad], axis=-1)
    )
    off_y_full = 0.5 * (
        jnp.concatenate([z_pad, off_diag_y], axis=-1)
        + jnp.concatenate([off_diag_y, z_pad], axis=-1)
    )

    # Interpolate off-diagonal from cell centers to faces
    off_x_u = interp_cell_to_uface(off_x_full)   # (n_lat, n_lon+1, nlev)
    off_y_v = interp_cell_to_vface(off_y_full)    # (n_lat+1, n_lon, nlev)

    # Full horizontal flux at faces
    Fx_u = kappa_Redi * dq_dx_u + off_x_u
    Fy_v = kappa_Redi * dq_dy_v + off_y_v

    # Single divergence of the full flux
    dq_h = divergence_cgrid(Fx_u, Fy_v, grid)

    # =====================================================================
    # Vertical flux at interfaces
    #   F_z = (kR + kG) * (Sx*dq/dx + Sy*dq/dy) + kR * S^2 * dq/dz
    # =====================================================================
    S2_half = S_x**2 + S_y**2
    F_z = ((kappa_Redi + kappa_GM) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi * S2_half * dq_dz_half)

    # Vertical flux divergence: (F_z_top - F_z_bot) / dz
    z_pad_v = jnp.zeros((*F_z.shape[:-1], 1), dtype=F_z.dtype)
    F_z_ext = jnp.concatenate([z_pad_v, F_z, z_pad_v], axis=-1)
    dq_vert = (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)

    return dq_h + dq_vert


def gm_redi_tracer_tendency_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    eos: str = "linear",
    eos_linear=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GM/Redi tracer tendencies for the lat-lon C-grid ocean.

    Parameters
    ----------
    T, S : (n_lat, n_lon, nlev) temperature and salinity.
    eta : (n_lat, n_lon) sea surface height.
    H_bathy : (n_lat, n_lon) bathymetry depth.
    grid : LatLonGrid.
    z_coord : OceanZStarCoordinate.
    cfg : GMRediConfig.
    eos : str, "wright" or "linear".
    eos_linear : LinearEOSConfig or None.

    Returns
    -------
    dT_dt, dS_dt : (n_lat, n_lon, nlev) tracer tendencies.
    """
    from legoesm.ocean.eos import compute_ocean_rho_and_pressure
    from legoesm.ocean.vertical import compute_ocean_jacobian

    # Compute density
    J = compute_ocean_jacobian(eta, H_bathy, z_coord)
    z_full = z_coord.z_full_ref * J[..., jnp.newaxis]

    if eos == "linear" and eos_linear is not None:
        # Simple linear EOS: rho = rho_ref * (1 - alpha_T * (T - T_ref))
        rho = eos_linear.rho_ref * (
            1.0 - eos_linear.alpha_T * (T - eos_linear.T_ref))
    else:
        rho, _ = compute_ocean_rho_and_pressure(
            T, S, z_coord, J, eos=eos)

    # Compute tapered slopes
    S_x, S_y, _ = _compute_tapered_slopes_latlon(rho, z_coord, J, grid, cfg)

    # Compute tendencies for T and S
    dT_dt = _tracer_tendency_gm_redi_latlon(
        T, S_x, S_y, z_coord, J, grid, cfg.kappa_GM, cfg.kappa_Redi)
    dS_dt = _tracer_tendency_gm_redi_latlon(
        S, S_x, S_y, z_coord, J, grid, cfg.kappa_GM, cfg.kappa_Redi)

    return dT_dt, dS_dt
