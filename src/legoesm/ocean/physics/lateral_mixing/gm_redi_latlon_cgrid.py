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
# Triad slope discretisation (Griffies, Gnanadesikan et al. 1998)
# =====================================================================
#
# The centered scheme above evaluates slopes at vertical interfaces and
# tracer gradients at faces, then averages the two onto interfaces.
# This averaging breaks the algebraic identity
#
#     dq/dx + S_x * dq/dz = 0           when q = f(rho),
#
# leaving a small Redi residual that accumulates through dynamical
# feedback over long integrations.
#
# The triad scheme decomposes the flux at each face into four
# quarter-cell triads.  Each triad uses the SAME three rho/q values for
# both its slope and its tracer gradients, so the cancellation above
# holds *exactly* for every triad — the Redi tendency is zero to
# machine precision when q is constant along isopycnals.
#
# Convention: ``z`` increases UPWARD, ``k = 0`` is the surface, and
# ``k+1`` is the level below ``k``.  ``drho_dz_w[k]`` therefore is the
# (negative) finite difference ``(rho[k] - rho[k+1]) / dz_half`` at the
# w-face between full levels ``k`` and ``k+1``.
#
# u-face triads at face index ``j_face`` (between cells ``j_w`` and
# ``j_e``) and full level ``k`` share the horizontal pair
# ``(j_w, k)-(j_e, k)`` — this gives ``drho_dx_uface(j_face, k)``,
# common to all four triads — and differ only in the vertical pair:
#
#     T1 (W,B): (j_w, k) - (j_w, k+1)   uses drho_dz_w[j_w, k]
#     T2 (W,A): (j_w, k-1) - (j_w, k)   uses drho_dz_w[j_w, k-1]
#     T3 (E,B): (j_e, k) - (j_e, k+1)   uses drho_dz_w[j_e, k]
#     T4 (E,A): (j_e, k-1) - (j_e, k)   uses drho_dz_w[j_e, k-1]
#
# Triads "above" the top level (k = 0 ⇒ T2/T4) and "below" the bottom
# level (k = nlev-1 ⇒ T1/T3) are not defined; their contributions are
# masked out and the per-face average is normalised by the number of
# valid triads, preserving the diagonal Redi flux at full strength
# while still cancelling the off-diagonal exactly when q = f(rho).


def _to_uface_west(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *west* neighbour of every u-face.

    For a ``(n_lat, n_lon, ...)`` array, returns ``(n_lat, n_lon+1, ...)``
    with ``out[:, j_face, ...] = field[:, (j_face - 1) mod n_lon, ...]``
    using periodic wrap in longitude.
    """
    rolled = jnp.roll(field, 1, axis=1)
    return jnp.concatenate([rolled, rolled[:, 0:1]], axis=1)


def _to_uface_east(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *east* neighbour of every u-face.

    Returns ``out[:, j_face, ...] = field[:, j_face mod n_lon, ...]``.
    """
    return jnp.concatenate([field, field[:, 0:1]], axis=1)


def _to_vface_south(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *south* neighbour of every v-face.

    No periodic wrap in latitude — wall BCs at the poles.  The pole
    sentinels are inert because the v-face mask zeroes the flux there.
    """
    return jnp.concatenate([field[0:1], field], axis=0)


def _to_vface_north(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *north* neighbour of every v-face."""
    return jnp.concatenate([field, field[-1:]], axis=0)


def _triad_taper(S: jnp.ndarray, S_max: float) -> jnp.ndarray:
    """DM95 smooth tanh taper applied to a per-triad slope magnitude.

    Identical functional form to :func:`dm95_taper` but operating on a
    single scalar slope (an x-triad or y-triad sees only one direction).
    Returns a factor in [0, 1].
    """
    return 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (0.1 * S_max + _EPS)
    ))


def gm_redi_tracer_tendency_triads_latlon_cgrid(
    q: jnp.ndarray,
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_GM,
    kappa_Redi: float,
    S_max: float,
) -> jnp.ndarray:
    """Triad-based GM+Redi tracer tendency on the lat-lon C-grid.

    Implements the Griffies, Gnanadesikan, Pacanowski, Larichev,
    Dukowicz & Smith (1998) triad decomposition of the small-slope
    isopycnal-tensor fluxes.  Each face flux is built from 4 triads
    (w-face) or up to 4 triads (u/v-face); each triad uses one shared
    horizontal density-and-tracer pair and one shared vertical pair, so
    the algebraic identity ``dq/dx + S_x dq/dz = 0`` (and its z-flux
    counterpart) holds *per triad* when ``q = f(rho)`` — the Redi
    tendency is zero to machine precision.

    Parameters
    ----------
    q, rho : (n_lat, n_lon, nlev)
        Tracer and in-situ density at cell centres.  Both are
        Neumann-filled internally — pass the raw fields exactly as
        ``compute_isopycnal_slopes_latlon_cgrid`` expects them.
    mask, u_mask, v_mask : ocean / face masks.
    z_coord, jacobian, grid : geometry.
    kappa_GM : float or (n_lat, n_lon)
        GM bolus coefficient (scalar or per-column from Visbeck).
    kappa_Redi : float
        Redi isopycnal diffusivity.
    S_max : float
        Slope cap for clipping and DM95 taper.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    n_lat, n_lon, nlev = q.shape
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]

    if isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 2:
        kappa_GM_c = kappa_GM[:, :, jnp.newaxis]                   # (n_lat, n_lon, 1)
        kappa_GM_u = interp_cell_to_uface(kappa_GM_c)              # (n_lat, n_lon+1, 1)
        kappa_GM_v = interp_cell_to_vface(kappa_GM_c)              # (n_lat+1, n_lon, 1)
    else:
        kappa_GM_c = kappa_GM
        kappa_GM_u = kappa_GM
        kappa_GM_v = kappa_GM
    # w-face triads sit at cell-centers horizontally → cell-centered kappa.
    kappa_GM_w = kappa_GM_c

    # Neumann-fill BOTH rho and q so the gradients across coastlines do
    # not pick up jumps between ocean and land sentinel values.  This
    # is essential for the per-triad cancellation when q = f(rho):
    # without it, a land cell that has T = 0 sentinel produces a
    # spurious vertical drho/dz = 0 in the land column, which clips the
    # slope to S_max and breaks the algebraic identity.
    rho_filled = _neumann_fill_cgrid(rho, mask)
    q_filled = _neumann_fill_cgrid(q, mask)

    # -----------------------------------------------------------------
    # 1. Density and tracer gradients on the native C-grid
    # -----------------------------------------------------------------
    drho_dx_u = gradient_x_cgrid(rho_filled, grid)        # (n_lat, n_lon+1, nlev)
    drho_dy_v = gradient_y_cgrid(rho_filled, grid)        # (n_lat+1, n_lon, nlev)
    dq_dx_u = gradient_x_cgrid(q_filled, grid)
    dq_dy_v = gradient_y_cgrid(q_filled, grid)

    drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )
    # Stable-strat floor: drho_dz must be negative (z UPWARD ⇒ rho denser below).
    drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)        # (n_lat, n_lon, nlev-1)
    dq_dz_w = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )

    # -----------------------------------------------------------------
    # 2. Pad drho_dz_w / dq_dz_w along the level axis so that boundary
    #    triads can be expressed by a single jnp.where / multiplication.
    #    "below" array at level k = drho_dz_w at w-face (k+1/2);
    #    "above" array at level k = drho_dz_w at w-face (k-1/2).
    # -----------------------------------------------------------------
    sentinel_rho = jnp.full(
        (n_lat, n_lon, 1), -_EPS_DIV, dtype=drho_dz_w.dtype,
    )
    drho_dz_below_lev = jnp.concatenate([drho_dz_w, sentinel_rho], axis=-1)
    drho_dz_above_lev = jnp.concatenate([sentinel_rho, drho_dz_w], axis=-1)

    sentinel_q = jnp.zeros((n_lat, n_lon, 1), dtype=dq_dz_w.dtype)
    dq_dz_below_lev = jnp.concatenate([dq_dz_w, sentinel_q], axis=-1)
    dq_dz_above_lev = jnp.concatenate([sentinel_q, dq_dz_w], axis=-1)

    valid_below = jnp.concatenate([
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
    ], axis=-1)
    valid_above = jnp.concatenate([
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
    ], axis=-1)

    # -----------------------------------------------------------------
    # 3. U-FACE triads — flux F_x at (n_lat, n_lon+1, nlev)
    #
    # IMPORTANT: the DM95 taper is applied to the *whole* per-triad
    # flux contribution, NOT to the raw slope.  Tapering the slope
    # before the cancellation
    #
    #     F_x^(m) = K_R · dq/dx + (K_R − K_GM) · taper · S · dq/dz^(m)
    #
    # would produce a residual ``K_R · (1 − taper) · dq/dx`` for
    # ``q = f(ρ)``, because the diagonal stays at full κ_R while the
    # off-diagonal is reduced by ``taper``.  Even at slopes well below
    # S_max the DM95 ``tanh`` saturates to ``1 − O(exp)``, so this
    # residual is *much* larger than float64 round-off.  Tapering the
    # whole triad — diagonal + off-diagonal together — preserves the
    # algebraic identity ``F_x^(m) = 0`` for ``q = f(ρ)`` and just
    # damps the genuine Redi flux when slopes saturate.
    # -----------------------------------------------------------------
    drho_dz_T1 = _to_uface_west(drho_dz_below_lev)
    drho_dz_T2 = _to_uface_west(drho_dz_above_lev)
    drho_dz_T3 = _to_uface_east(drho_dz_below_lev)
    drho_dz_T4 = _to_uface_east(drho_dz_above_lev)

    dq_dz_T1 = _to_uface_west(dq_dz_below_lev)
    dq_dz_T2 = _to_uface_west(dq_dz_above_lev)
    dq_dz_T3 = _to_uface_east(dq_dz_below_lev)
    dq_dz_T4 = _to_uface_east(dq_dz_above_lev)

    valid_T1 = _to_uface_west(valid_below)
    valid_T2 = _to_uface_west(valid_above)
    valid_T3 = _to_uface_east(valid_below)
    valid_T4 = _to_uface_east(valid_above)

    # Raw clipped slopes — DO NOT taper here; taper goes on the whole flux.
    S_T1 = jnp.clip(-drho_dx_u / drho_dz_T1, -S_max, S_max)
    S_T2 = jnp.clip(-drho_dx_u / drho_dz_T2, -S_max, S_max)
    S_T3 = jnp.clip(-drho_dx_u / drho_dz_T3, -S_max, S_max)
    S_T4 = jnp.clip(-drho_dx_u / drho_dz_T4, -S_max, S_max)

    taper_T1 = _triad_taper(S_T1, S_max)
    taper_T2 = _triad_taper(S_T2, S_max)
    taper_T3 = _triad_taper(S_T3, S_max)
    taper_T4 = _triad_taper(S_T4, S_max)

    N_valid_u = valid_T1 + valid_T2 + valid_T3 + valid_T4
    N_valid_u_safe = jnp.maximum(N_valid_u, 1.0)
    w_T1 = valid_T1 / N_valid_u_safe
    w_T2 = valid_T2 / N_valid_u_safe
    w_T3 = valid_T3 / N_valid_u_safe
    w_T4 = valid_T4 / N_valid_u_safe

    # Per-triad full flux (cancels exactly when q = f(ρ)).
    flux_T1 = kappa_Redi * dq_dx_u + (kappa_Redi - kappa_GM_u) * S_T1 * dq_dz_T1
    flux_T2 = kappa_Redi * dq_dx_u + (kappa_Redi - kappa_GM_u) * S_T2 * dq_dz_T2
    flux_T3 = kappa_Redi * dq_dx_u + (kappa_Redi - kappa_GM_u) * S_T3 * dq_dz_T3
    flux_T4 = kappa_Redi * dq_dx_u + (kappa_Redi - kappa_GM_u) * S_T4 * dq_dz_T4

    F_x_u = (w_T1 * taper_T1 * flux_T1
           + w_T2 * taper_T2 * flux_T2
           + w_T3 * taper_T3 * flux_T3
           + w_T4 * taper_T4 * flux_T4)
    F_x_u = F_x_u * u_mask[:, :, jnp.newaxis]

    # -----------------------------------------------------------------
    # 4. V-FACE triads — flux F_y at (n_lat+1, n_lon, nlev)
    # -----------------------------------------------------------------
    drho_dz_V1 = _to_vface_south(drho_dz_below_lev)
    drho_dz_V2 = _to_vface_south(drho_dz_above_lev)
    drho_dz_V3 = _to_vface_north(drho_dz_below_lev)
    drho_dz_V4 = _to_vface_north(drho_dz_above_lev)

    dq_dz_V1 = _to_vface_south(dq_dz_below_lev)
    dq_dz_V2 = _to_vface_south(dq_dz_above_lev)
    dq_dz_V3 = _to_vface_north(dq_dz_below_lev)
    dq_dz_V4 = _to_vface_north(dq_dz_above_lev)

    valid_V1 = _to_vface_south(valid_below)
    valid_V2 = _to_vface_south(valid_above)
    valid_V3 = _to_vface_north(valid_below)
    valid_V4 = _to_vface_north(valid_above)

    # At pole v-faces drho_dy_v = 0 ⇒ all S_V* = 0 ⇒ flux = K_R·dq_dy_v = 0
    # (gradient_y_cgrid sets dq_dy_v = 0 there); v_mask zeros the result.
    S_V1 = jnp.clip(-drho_dy_v / drho_dz_V1, -S_max, S_max)
    S_V2 = jnp.clip(-drho_dy_v / drho_dz_V2, -S_max, S_max)
    S_V3 = jnp.clip(-drho_dy_v / drho_dz_V3, -S_max, S_max)
    S_V4 = jnp.clip(-drho_dy_v / drho_dz_V4, -S_max, S_max)

    taper_V1 = _triad_taper(S_V1, S_max)
    taper_V2 = _triad_taper(S_V2, S_max)
    taper_V3 = _triad_taper(S_V3, S_max)
    taper_V4 = _triad_taper(S_V4, S_max)

    N_valid_v = valid_V1 + valid_V2 + valid_V3 + valid_V4
    N_valid_v_safe = jnp.maximum(N_valid_v, 1.0)
    w_V1 = valid_V1 / N_valid_v_safe
    w_V2 = valid_V2 / N_valid_v_safe
    w_V3 = valid_V3 / N_valid_v_safe
    w_V4 = valid_V4 / N_valid_v_safe

    flux_V1 = kappa_Redi * dq_dy_v + (kappa_Redi - kappa_GM_v) * S_V1 * dq_dz_V1
    flux_V2 = kappa_Redi * dq_dy_v + (kappa_Redi - kappa_GM_v) * S_V2 * dq_dz_V2
    flux_V3 = kappa_Redi * dq_dy_v + (kappa_Redi - kappa_GM_v) * S_V3 * dq_dz_V3
    flux_V4 = kappa_Redi * dq_dy_v + (kappa_Redi - kappa_GM_v) * S_V4 * dq_dz_V4

    F_y_v = (w_V1 * taper_V1 * flux_V1
           + w_V2 * taper_V2 * flux_V2
           + w_V3 * taper_V3 * flux_V3
           + w_V4 * taper_V4 * flux_V4)
    F_y_v = F_y_v * v_mask[:, :, jnp.newaxis]

    # Horizontal divergence (single conservative call).
    dq_h = divergence_cgrid(F_x_u, F_y_v, grid)

    # -----------------------------------------------------------------
    # 5. W-FACE triads — flux F_z at (n_lat, n_lon, nlev-1)
    #
    # All eight triads at a w-face share the same drho_dz_w(k+1/2)
    # because the vertical pair (k, k+1) is fixed; they differ in their
    # horizontal pair.  The 4 x-triads use drho_dx_u at u-face j or j+1
    # and at level k (above the w-face) or k+1 (below).  The 4 y-triads
    # use drho_dy_v at v-face i or i+1 and at level k or k+1.
    # -----------------------------------------------------------------
    # u-face j == drho_dx_u[:, :n_lon, :] (west face of cell j, internal index)
    # u-face j+1 == drho_dx_u[:, 1:n_lon+1, :] (east face of cell j)
    drho_dx_west = drho_dx_u[:, :n_lon, :]
    drho_dx_east = drho_dx_u[:, 1:n_lon + 1, :]
    dq_dx_west = dq_dx_u[:, :n_lon, :]
    dq_dx_east = dq_dx_u[:, 1:n_lon + 1, :]

    # Slice to (n_lat, n_lon, nlev-1) — "above" = level k, "below" = k+1.
    drho_dx_west_A = drho_dx_west[:, :, :-1]
    drho_dx_west_B = drho_dx_west[:, :, 1:]
    drho_dx_east_A = drho_dx_east[:, :, :-1]
    drho_dx_east_B = drho_dx_east[:, :, 1:]

    dq_dx_west_A = dq_dx_west[:, :, :-1]
    dq_dx_west_B = dq_dx_west[:, :, 1:]
    dq_dx_east_A = dq_dx_east[:, :, :-1]
    dq_dx_east_B = dq_dx_east[:, :, 1:]

    drho_dy_south = drho_dy_v[:n_lat, :, :]
    drho_dy_north = drho_dy_v[1:n_lat + 1, :, :]
    dq_dy_south = dq_dy_v[:n_lat, :, :]
    dq_dy_north = dq_dy_v[1:n_lat + 1, :, :]

    drho_dy_south_A = drho_dy_south[:, :, :-1]
    drho_dy_south_B = drho_dy_south[:, :, 1:]
    drho_dy_north_A = drho_dy_north[:, :, :-1]
    drho_dy_north_B = drho_dy_north[:, :, 1:]

    dq_dy_south_A = dq_dy_south[:, :, :-1]
    dq_dy_south_B = dq_dy_south[:, :, 1:]
    dq_dy_north_A = dq_dy_north[:, :, :-1]
    dq_dy_north_B = dq_dy_north[:, :, 1:]

    # x-triad slopes at w-face (drho_dz_w shared across all four).
    # Same per-triad-full-flux × per-triad-taper structure as above —
    # tapering S² before adding cross_x would re-introduce a residual
    # ``K_R · taper · (1 − taper) · b · drho_dx² / drho_dz_w`` for
    # ``q = f(ρ)``.
    S_Wx1 = jnp.clip(-drho_dx_west_A / drho_dz_w, -S_max, S_max)  # W,A
    S_Wx2 = jnp.clip(-drho_dx_east_A / drho_dz_w, -S_max, S_max)  # E,A
    S_Wx3 = jnp.clip(-drho_dx_west_B / drho_dz_w, -S_max, S_max)  # W,B
    S_Wx4 = jnp.clip(-drho_dx_east_B / drho_dz_w, -S_max, S_max)  # E,B

    taper_Wx1 = _triad_taper(S_Wx1, S_max)
    taper_Wx2 = _triad_taper(S_Wx2, S_max)
    taper_Wx3 = _triad_taper(S_Wx3, S_max)
    taper_Wx4 = _triad_taper(S_Wx4, S_max)

    S_Wy1 = jnp.clip(-drho_dy_south_A / drho_dz_w, -S_max, S_max)
    S_Wy2 = jnp.clip(-drho_dy_north_A / drho_dz_w, -S_max, S_max)
    S_Wy3 = jnp.clip(-drho_dy_south_B / drho_dz_w, -S_max, S_max)
    S_Wy4 = jnp.clip(-drho_dy_north_B / drho_dz_w, -S_max, S_max)

    taper_Wy1 = _triad_taper(S_Wy1, S_max)
    taper_Wy2 = _triad_taper(S_Wy2, S_max)
    taper_Wy3 = _triad_taper(S_Wy3, S_max)
    taper_Wy4 = _triad_taper(S_Wy4, S_max)

    # Per-triad full vertical-flux contribution.  For q = f(ρ) and
    # K_GM = 0, each ``flux_W*_m`` is exactly zero (per-triad
    # algebraic cancellation, see derivation in module header).
    # Multiplying each by its taper and averaging keeps that exact
    # zero while still damping the genuine GM transport in tapered
    # boundary regions.
    flux_Wx1 = ((kappa_Redi + kappa_GM_w) * S_Wx1 * dq_dx_west_A
                + kappa_Redi * S_Wx1 ** 2 * dq_dz_w)
    flux_Wx2 = ((kappa_Redi + kappa_GM_w) * S_Wx2 * dq_dx_east_A
                + kappa_Redi * S_Wx2 ** 2 * dq_dz_w)
    flux_Wx3 = ((kappa_Redi + kappa_GM_w) * S_Wx3 * dq_dx_west_B
                + kappa_Redi * S_Wx3 ** 2 * dq_dz_w)
    flux_Wx4 = ((kappa_Redi + kappa_GM_w) * S_Wx4 * dq_dx_east_B
                + kappa_Redi * S_Wx4 ** 2 * dq_dz_w)

    flux_Wy1 = ((kappa_Redi + kappa_GM_w) * S_Wy1 * dq_dy_south_A
                + kappa_Redi * S_Wy1 ** 2 * dq_dz_w)
    flux_Wy2 = ((kappa_Redi + kappa_GM_w) * S_Wy2 * dq_dy_north_A
                + kappa_Redi * S_Wy2 ** 2 * dq_dz_w)
    flux_Wy3 = ((kappa_Redi + kappa_GM_w) * S_Wy3 * dq_dy_south_B
                + kappa_Redi * S_Wy3 ** 2 * dq_dz_w)
    flux_Wy4 = ((kappa_Redi + kappa_GM_w) * S_Wy4 * dq_dy_north_B
                + kappa_Redi * S_Wy4 ** 2 * dq_dz_w)

    F_z = 0.25 * (taper_Wx1 * flux_Wx1 + taper_Wx2 * flux_Wx2
                 + taper_Wx3 * flux_Wx3 + taper_Wx4 * flux_Wx4
                 + taper_Wy1 * flux_Wy1 + taper_Wy2 * flux_Wy2
                 + taper_Wy3 * flux_Wy3 + taper_Wy4 * flux_Wy4)

    dq_vert = vertical_flux_divergence(F_z, dz_actual, _EPS)

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

    # Centred interface slopes — used for Visbeck (only ⟨N|S|⟩_z is
    # needed; the cancellation property of triads is irrelevant there).
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(
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

    scheme = getattr(cfg, "slope_scheme", "triads")
    if scheme == "triads":
        dT_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi, cfg.S_max,
        )
        dS_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            S, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi, cfg.S_max,
        )
    elif scheme == "centered":
        dT_dt = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi,
        )
        dS_dt = gm_redi_tracer_tendency_latlon_cgrid(
            S, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi,
        )
    else:
        raise ValueError(
            f"Unknown GMRediConfig.slope_scheme={scheme!r}; "
            f"expected 'centered' or 'triads'."
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
