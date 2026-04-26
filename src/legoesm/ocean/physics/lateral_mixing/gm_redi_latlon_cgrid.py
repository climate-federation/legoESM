"""GM/Redi isopycnal mixing for the lat-lon Arakawa C-grid ocean.

Griffies (1998) skew-flux formulation with DM95 slope tapering,
optional Visbeck (1997) adaptive coefficient, and full land-mask
treatment via Neumann fill + face masks.

Grid conventions (LatLonGrid C-grid staggering):
- Scalars (rho, T, S): cell centers, shape (n_lat, n_lon, nlev)
- u-fluxes: lon interfaces, shape (n_lat, n_lon+1, nlev)
- v-fluxes: lat interfaces, shape (n_lat+1, n_lon, nlev)

References
----------
- Griffies, S. M. (1998). The Gent-McWilliams skew flux.
  J. Phys. Oceanogr., 28, 831-841.
- Danabasoglu, G. & McWilliams, J. C. (1995). J. Climate, 8, 2967-2987.
- Visbeck, M. et al. (1997). J. Phys. Oceanogr., 27, 381-402.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _neumann_fill_cgrid
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import make_eos_fn, rho_0 as _RHO_0
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    _EPS,
    compute_visbeck_kappa_gm,
    dm95_taper,
    vertical_flux_divergence,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian

# Division-guard epsilon — larger than float32 machine eps to prevent
# intermediate blow-up in the backward pass (see plan §7, AD safety).
_EPS_DIV = 1e-10


# =====================================================================
# Isopycnal slope computation
# =====================================================================

def compute_isopycnal_slopes_latlon_cgrid(
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    cfg: GMRediConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute tapered isopycnal slopes at vertical interfaces.

    Parameters
    ----------
    rho : (n_lat, n_lon, nlev)
        In-situ density at cell centers.
    mask : (n_lat, n_lon)
        Ocean mask (1 = ocean, 0 = land).
    z_coord : OceanZStarCoordinate
    jacobian : (n_lat, n_lon)
        z-star Jacobian (eta + H) / H.
    grid : LatLonGrid
    cfg : GMRediConfig

    Returns
    -------
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    taper : (n_lat, n_lon, nlev-1)
        DM95 taper factor in [0, 1].
    """
    # Neumann-fill density to prevent garbage gradients at coastlines.
    rho_filled = _neumann_fill_cgrid(rho, mask)

    # --- Horizontal density gradients at full levels ---
    # gradient_x_cgrid handles 3D natively: (n_lat, n_lon, nlev) -> (n_lat, n_lon+1, nlev)
    drho_dx_u = gradient_x_cgrid(rho_filled, grid)  # at u-faces
    drho_dy_v = gradient_y_cgrid(rho_filled, grid)  # at v-faces

    # Average face gradients to cell centers.
    # u-face j is between cell (j-1) and cell j, so cell j's gradient
    # is the average of face j (west) and face j+1 (east).
    drho_dx = 0.5 * (drho_dx_u[:, :-1, :] + drho_dx_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    # v-face i is between cell (i-1) and cell i, so cell i's gradient
    # is the average of face i (south) and face i+1 (north).
    drho_dy = 0.5 * (drho_dy_v[:-1, :, :] + drho_dy_v[1:, :, :])  # (n_lat, n_lon, nlev)

    # Average cell-center horizontal gradients from full levels to interfaces.
    drho_dx_half = 0.5 * (drho_dx[:, :, :-1] + drho_dx[:, :, 1:])  # (n_lat, n_lon, nlev-1)
    drho_dy_half = 0.5 * (drho_dy[:, :, :-1] + drho_dy[:, :, 1:])

    # --- Vertical density gradient at interfaces ---
    # dz_half_ref is the distance between adjacent cell centers (nlev-1,).
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)
    drho_dz = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)
    # Force stable stratification: drho/dz must be negative (density increases downward).
    drho_dz_safe = jnp.minimum(drho_dz, -_EPS_DIV)

    # --- Slopes ---
    S_x = jnp.clip(-drho_dx_half / drho_dz_safe, -cfg.S_max, cfg.S_max)
    S_y = jnp.clip(-drho_dy_half / drho_dz_safe, -cfg.S_max, cfg.S_max)

    # DM95 tapering via shared helper (identical formula across grids).
    return dm95_taper(S_x, S_y, cfg.S_max, _EPS)


# =====================================================================
# Tracer tendency (single tracer)
# =====================================================================

def gm_redi_tracer_tendency_latlon_cgrid(
    q: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_GM,
    kappa_Redi: float,
) -> jnp.ndarray:
    """GM+Redi tendency for a single tracer on the lat-lon C-grid.

    Uses the Griffies (1998) small-slope tensor.  Diagonal and
    off-diagonal horizontal fluxes are combined at faces and passed
    through a single ``divergence_cgrid`` call for conservation.

    Parameters
    ----------
    q : (n_lat, n_lon, nlev)
        Tracer field at cell centers.
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    mask : (n_lat, n_lon)
        Ocean mask.
    u_mask : (n_lat, n_lon+1)
        u-face mask (1 if both adjacent cells are ocean).
    v_mask : (n_lat+1, n_lon)
        v-face mask.
    z_coord : OceanZStarCoordinate
    jacobian : (n_lat, n_lon)
    grid : LatLonGrid
    kappa_GM : float or (n_lat, n_lon)
        GM transport coefficient [m^2/s].
    kappa_Redi : float
        Redi isopycnal diffusivity [m^2/s].

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev)
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)

    # Broadcast kappa_GM for interface-level arrays when it is per-column.
    if isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 2:
        kappa_GM_b = kappa_GM[:, :, jnp.newaxis]
    else:
        kappa_GM_b = kappa_GM

    # --- Neumann-fill tracer before computing gradients ---
    q_filled = _neumann_fill_cgrid(q, mask)

    # --- Horizontal tracer gradients at faces (3D-native) ---
    dq_dx_u = gradient_x_cgrid(q_filled, grid)  # (n_lat, n_lon+1, nlev) at u-faces
    dq_dy_v = gradient_y_cgrid(q_filled, grid)  # (n_lat+1, n_lon, nlev) at v-faces

    # --- Vertical tracer gradient at interfaces ---
    dq_dz_half = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)

    # ================================================================
    # Horizontal fluxes — computed at INTERFACES for exact cancellation
    # ================================================================
    # F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * S_x * dq/dz
    # F_y = kappa_Redi * dq/dy + (kappa_Redi - kappa_GM) * S_y * dq/dz
    #
    # KEY: Both the diagonal (kR * dq/dx) and off-diagonal (kR-kG)*S*dq/dz
    # terms are evaluated at INTERFACE levels before averaging to full
    # levels.  This ensures exact cancellation (dq/dx + S_x*dq/dz = 0)
    # when q is constant along isopycnals (e.g., linear EOS with T as
    # tracer).  The previous approach evaluated the diagonal at full
    # levels and the off-diagonal at interfaces, breaking the
    # cancellation and producing spurious cross-isopycnal diffusion.

    # Average horizontal tracer gradients from full levels to interfaces.
    # u-face gradient → cell center → interface
    dq_dx_center = 0.5 * (dq_dx_u[:, :-1, :] + dq_dx_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    dq_dy_center = 0.5 * (dq_dy_v[:-1, :, :] + dq_dy_v[1:, :, :])
    dq_dx_half = 0.5 * (dq_dx_center[:, :, :-1] + dq_dx_center[:, :, 1:])  # (n_lat, n_lon, nlev-1)
    dq_dy_half = 0.5 * (dq_dy_center[:, :, :-1] + dq_dy_center[:, :, 1:])

    # Total horizontal Redi flux at interfaces (exact cancellation here).
    F_x_half = (kappa_Redi * dq_dx_half
                + (kappa_Redi - kappa_GM_b) * S_x * dq_dz_half)  # (n_lat, n_lon, nlev-1)
    F_y_half = (kappa_Redi * dq_dy_half
                + (kappa_Redi - kappa_GM_b) * S_y * dq_dz_half)

    # Average interface fluxes to full levels (zero-pad at surface/bottom).
    z_pad = jnp.zeros((*F_x_half.shape[:2], 1), dtype=F_x_half.dtype)
    F_x_full = 0.5 * (
        jnp.concatenate([z_pad, F_x_half], axis=-1)
        + jnp.concatenate([F_x_half, z_pad], axis=-1)
    )  # (n_lat, n_lon, nlev)
    F_y_full = 0.5 * (
        jnp.concatenate([z_pad, F_y_half], axis=-1)
        + jnp.concatenate([F_y_half, z_pad], axis=-1)
    )

    # Interpolate cell-center fluxes to faces for divergence.
    F_x_u = interp_cell_to_uface(F_x_full)  # (n_lat, n_lon+1, nlev)
    F_y_v = interp_cell_to_vface(F_y_full)   # (n_lat+1, n_lon, nlev)

    # Apply face masks (zero flux through land boundaries).
    F_x_u = F_x_u * u_mask[:, :, jnp.newaxis]
    F_y_v = F_y_v * v_mask[:, :, jnp.newaxis]

    # Single conservative FV divergence.
    dq_h = divergence_cgrid(F_x_u, F_y_v, grid)

    # ================================================================
    # Vertical flux at interfaces
    # ================================================================
    # F_z = (kR + kG) * (S_x*dq/dx_center + S_y*dq/dy_center) + kR * S^2 * dq/dz
    # dq_dx_half, dq_dy_half already computed above (reused here).

    S2_half = S_x ** 2 + S_y ** 2
    F_z = ((kappa_Redi + kappa_GM_b) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi * S2_half * dq_dz_half)

    # Vertical flux divergence via shared helper (zero-flux BCs at surface/bottom).
    dq_vert = vertical_flux_divergence(F_z, dz_actual, _EPS)

    # ================================================================
    # Total tendency, masked
    # ================================================================
    tendency = (dq_h + dq_vert) * mask[:, :, jnp.newaxis]
    return tendency


# =====================================================================
# Top-level orchestrator (public API matching call site)
# =====================================================================

def gm_redi_tracer_tendency_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    f_coriolis: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-level GM/Redi for lat-lon C-grid.

    Computes density, isopycnal slopes, optional Visbeck coefficient,
    then returns tracer tendencies for T and S.

    Parameters
    ----------
    T, S : (n_lat, n_lon, nlev)
    eta : (n_lat, n_lon)
        Sea-surface height.
    H_bathy : (n_lat, n_lon)
        Bottom depth (positive).
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    cfg : GMRediConfig
    eos : str
        Equation of state ("wright" or "linear").
    eos_linear : LinearEOSConfig or None
    mask : (n_lat, n_lon) ocean mask
    u_mask : (n_lat, n_lon+1) u-face mask
    v_mask : (n_lat+1, n_lon) v-face mask
    f_coriolis : (n_lat, n_lon) Coriolis parameter

    Returns
    -------
    dT_dt, dS_dt : (n_lat, n_lon, nlev)
    """
    # Default masks: all ocean.
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    if u_mask is None:
        u_mask = jnp.ones((T.shape[0], T.shape[1] + 1), dtype=T.dtype)
    if v_mask is None:
        v_mask = jnp.ones((T.shape[0] + 1, T.shape[1]), dtype=T.dtype)

    # Jacobian.
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)

    # Density via 2-iteration EOS coupling.
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: _neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn,
        z_coord.dz_ref, _RHO_0, constants.g,
        n_iter=2,
    )

    # Isopycnal slopes.
    S_x, S_y, taper = compute_isopycnal_slopes_latlon_cgrid(
        rho, mask, z_coord, jacobian, grid, cfg,
    )

    # GM coefficient.
    if cfg.visbeck.enabled:
        if f_coriolis is None:
            # Compute from grid latitude.
            f_coriolis = 2.0 * constants.Omega * jnp.sin(grid.lat[:, jnp.newaxis])
            f_coriolis = jnp.broadcast_to(f_coriolis, mask.shape)
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Tracer tendencies.
    dT_dt = gm_redi_tracer_tendency_latlon_cgrid(
        T, S_x, S_y, mask, u_mask, v_mask,
        z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi,
    )
    dS_dt = gm_redi_tracer_tendency_latlon_cgrid(
        S, S_x, S_y, mask, u_mask, v_mask,
        z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi,
    )

    return dT_dt, dS_dt


# =====================================================================
# LateralMixingOutput wrapper (for future factory integration)
# =====================================================================

def gm_redi_lateral_mixing_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    **kwargs,
) -> LateralMixingOutput:
    """GM/Redi returning ``LateralMixingOutput`` (factory-compatible API).

    Wraps ``gm_redi_tracer_tendency_latlon`` with zero momentum tendencies.
    """
    dT_dt, dS_dt = gm_redi_tracer_tendency_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, **kwargs,
    )
    return LateralMixingOutput(
        du_dt=jnp.zeros_like(dT_dt[:, :1, :]),  # placeholder shape
        dv_dt=jnp.zeros_like(dT_dt[:1, :, :]),
        dT_dt=dT_dt,
        dS_dt=dS_dt,
    )
