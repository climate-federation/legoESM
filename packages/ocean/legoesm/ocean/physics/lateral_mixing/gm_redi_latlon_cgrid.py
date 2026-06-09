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
    interp_wface_to_center,
    laplacian_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _neumann_fill_cgrid
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import (
    compute_hydrostatic_pressure,
    eos_density_derivatives,
    make_eos_fn,
    rho_0 as _RHO_0,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    _EPS,
    compute_eke_kappa_gm,
    compute_visbeck_kappa_gm,
    dm95_taper,
    dm95_taper_scalar,
    vertical_flux_divergence,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian

# Division-guard epsilon — larger than float32 machine eps to prevent
# intermediate blow-up in the backward pass (see plan §7, AD safety).
_EPS_DIV = 1e-10


def _kappa_is_interface_3d(kappa, nlev: int) -> bool:
    """True iff ``kappa`` is a depth-resolved *interface* (W-grid) diffusivity.

    The 3-D EKE closure (Stages 1-2) supplies ``kappa_GM`` / ``kappa_Redi`` at
    the ``nlev-1`` interior interfaces.  Distinguishing this from the legacy
    ``(n_lat, n_lon, 1)`` broadcast (a 2-D per-column kappa lifted with a
    trailing length-1 axis) is by the last-axis length:

    - last axis ``== nlev-1`` **and** ``nlev > 2`` ⇒ interface kappa (new path).
    - last axis ``== 1`` ⇒ broadcast (legacy path), so a ``(n,n,1)`` kappa is
      ALWAYS treated as a broadcast.

    The ``nlev > 2`` guard resolves the only ambiguous case: when ``nlev == 2``
    the interior-interface count ``nlev-1 == 1`` collides with the ``(n,n,1)``
    broadcast shape, so we never interpret a length-1 trailing axis as an
    interface field.  An interface kappa may therefore only be passed when
    ``nlev > 2`` (every real ocean config has ``nlev >= 10``).  A scalar or a
    ``(n_lat, n_lon)`` 2-D kappa is not an ndarray-with-ndim-3 and returns
    False here, taking the bit-identical broadcast path downstream.
    """
    return (
        isinstance(kappa, jnp.ndarray)
        and kappa.ndim == 3
        and kappa.shape[-1] == nlev - 1
        and nlev > 2
    )


def _kappa_center_uvw(kappa, nlev: int):
    """Resolve a kappa argument to its center / u-face / v-face / w-face forms.

    Returns ``(kappa_c, kappa_u, kappa_v, kappa_w)`` where:

    - **scalar / 2-D ``(n_lat, n_lon)``** (the legacy path) — the 2-D array is
      lifted to ``(n_lat, n_lon, 1)`` cell-centred and interpolated to u/v-faces
      exactly as before; ``kappa_w = kappa_c``.  A scalar passes through
      unchanged on all four outputs.  **Bit-identical to the prior code.**
    - **3-D interface ``(n_lat, n_lon, nlev-1)``** (the 3-D EKE path) — the
      interface kappa is the w-face value DIRECTLY (``kappa_w = kappa``; it
      already lives at the w-faces, so no lossy round-trip), while the
      horizontal forms come from a vertical interface→center interpolation
      (``interp_wface_to_center``) followed by the usual horizontal
      cell→u/v-face interpolation.

    Shared by the triad scheme, the centered scheme, and the K_33 getter so the
    three can never diverge.
    """
    if _kappa_is_interface_3d(kappa, nlev):
        kappa_w = kappa                              # (n_lat, n_lon, nlev-1)
        kappa_c = interp_wface_to_center(kappa)      # (n_lat, n_lon, nlev)
        kappa_u = interp_cell_to_uface(kappa_c)      # (n_lat, n_lon+1, nlev)
        kappa_v = interp_cell_to_vface(kappa_c)      # (n_lat+1, n_lon, nlev)
        return kappa_c, kappa_u, kappa_v, kappa_w
    if isinstance(kappa, jnp.ndarray) and kappa.ndim == 2:
        kappa_c = kappa[:, :, jnp.newaxis]           # (n_lat, n_lon, 1)
        kappa_u = interp_cell_to_uface(kappa_c)      # (n_lat, n_lon+1, 1)
        kappa_v = interp_cell_to_vface(kappa_c)      # (n_lat+1, n_lon, 1)
        return kappa_c, kappa_u, kappa_v, kappa_c
    # Scalar (or already a (n,n,1) broadcast array): pass through unchanged.
    return kappa, kappa, kappa, kappa


# =====================================================================
# Neutral (locally-referenced) density-gradient ingredients
# =====================================================================

_VALID_SLOPE_DENSITY = frozenset({"in_situ", "neutral"})


def _validate_slope_density(slope_density: str) -> None:
    """Fail-fast on an unknown ``slope_density`` literal (Dispatch Discipline).

    The in-situ / neutral selection is a binary ``if`` rather than a factory,
    but an unrecognised value must NOT silently fall through to the in-situ
    branch (a typo would then mask the neutral option entirely).
    """
    if slope_density not in _VALID_SLOPE_DENSITY:
        raise ValueError(
            f"Unknown GMRediConfig.slope_density={slope_density!r}; "
            f"expected one of {sorted(_VALID_SLOPE_DENSITY)}."
        )


def _neutral_drho_derivs(T, S, mask, z_coord, jacobian, eos_fn, rho_0, g):
    """Cell-centred EOS partials ``(∂ρ/∂T, ∂ρ/∂S)`` at the LOCAL cell pressure.

    The ingredients of the locally-referenced *neutral* density gradient used
    by the ``slope_density="neutral"`` isoneutral slope build (Veros
    ``get_drhodT`` / ``get_drhodS`` at ``abs(zt)``;
    ``veros/core/isoneutral/isoneutral.py:40-41``).

    The LOCAL pressure is built EXACTLY as the in-situ EOS iteration
    (``iterate_eos_and_pressure_anomaly``) builds it — the *reference*
    hydrostatic pressure ``compute_hydrostatic_pressure(ρ, η=0, dz_ref, J=1)``
    — so the neutral derivatives reference the SAME cell pressure the in-situ
    ρ (and hence the in-situ slope) already uses (plan §"local pressure must
    be the SAME hydrostatic pressure the slope builder uses").  For a
    hydrostatic column this is ``ρ₀·g·z``, the legoESM analogue of Veros's
    ``abs(zt)``.  T, S are Neumann-filled so the partials on land take an
    ocean-neighbour value (the gradients across coastlines are still masked by
    the face masks downstream).

    Returns ``(drdT, drdS)`` each ``(n_lat, n_lon, nlev)`` at cell centres,
    masked to the wet domain.
    """
    T_filled = _neumann_fill_cgrid(T, mask)
    S_filled = _neumann_fill_cgrid(S, mask)
    # In-situ density via the SAME 2-iteration EOS coupling the slope builder
    # uses, then the reference hydrostatic pressure (η=0, J=1) — identical to
    # iterate_eos_and_pressure_anomaly's internal p_hydro.
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)
    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)
    drdT, drdS = eos_density_derivatives(eos_fn, T_filled, S_filled, p_hydro)
    return drdT * mask[:, :, jnp.newaxis], drdS * mask[:, :, jnp.newaxis]


def _slope_density_face_grads(
    rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
    eos_fn, rho_0, g,
):
    """Density gradients at the C-grid faces used to BUILD isoneutral slopes.

    Returns ``(drho_dx_u, drho_dy_v, drho_dz_w)`` — the eastward (u-face),
    northward (v-face) and vertical (w-face) density gradients with the
    stable-strat floor on ``drho_dz_w``.  Two modes:

    - ``"in_situ"`` (default, BIT-IDENTICAL): finite-difference the in-situ
      ``rho_filled``.
    - ``"neutral"``: the locally-referenced neutral form
      ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S``.  The horizontal face gradients use ``∂ρ/∂T``
      interpolated to the u/v-face (``interp_cell_to_uface`` / ``..._vface``)
      times the raw tracer face gradient; the vertical w-face gradient uses the
      UPPER-cell ``∂ρ/∂T`` (``drdT[:, :, :-1]``) — the validated probe form,
      matching Veros's ``drodzb`` with the cell's own derivative (the kr-sum
      over both triad levels is the documented metric-faithful follow-up).

    Shared by the centred slope builder (Visbeck/EKE diagnostics) so the two
    modes never diverge there.  The TRIAD / K_33 path uses
    :func:`_w_triad_slope_density_grads` instead (per-triad cell references).
    """
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]
    if slope_density == "neutral":
        drdT, drdS = _neutral_drho_derivs(
            T, S, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = _neumann_fill_cgrid(T, mask)
        S_filled = _neumann_fill_cgrid(S, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        drdT_u = interp_cell_to_uface(drdT)
        drdS_u = interp_cell_to_uface(drdS)
        drdT_v = interp_cell_to_vface(drdT)
        drdS_v = interp_cell_to_vface(drdS)
        drho_dx_u = drdT_u * dTdx_u + drdS_u * dSdx_u
        drho_dy_v = drdT_v * dTdy_v + drdS_v * dSdy_v
        drdT_w = drdT[:, :, :-1]
        drdS_w = drdS[:, :, :-1]
        drho_dz_w = drdT_w * dTdz_w + drdS_w * dSdz_w
    else:
        drho_dx_u = gradient_x_cgrid(rho_filled, grid)
        drho_dy_v = gradient_y_cgrid(rho_filled, grid)
        drho_dz_w = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
    # Stable-strat floor: ∂_zρ must be negative (z UPWARD ⇒ ρ denser below).
    drho_dz_w = jnp.minimum(drho_dz_w, -_EPS_DIV)
    return drho_dx_u, drho_dy_v, drho_dz_w


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
    *,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    eos_fn=None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
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
    T, S : (n_lat, n_lon, nlev) or None
        Tracer fields, REQUIRED when ``cfg.slope_density == "neutral"`` so the
        neutral density gradient ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` can be built; ignored
        for the default ``"in_situ"`` mode (then only ``rho`` is used).
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> ρ`` (needed only for the neutral mode; the local
        cell pressure / EOS partials are built from it).
    rho_0, g : float
        Reference density / gravity for the local hydrostatic pressure.

    Returns
    -------
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    taper : (n_lat, n_lon, nlev-1)
        DM95 taper factor in [0, 1].
    """
    slope_density = getattr(cfg, "slope_density", "in_situ")
    _validate_slope_density(slope_density)
    if slope_density == "neutral" and (T is None or S is None or eos_fn is None):
        raise ValueError(
            "compute_isopycnal_slopes_latlon_cgrid: slope_density='neutral' "
            "requires T, S and eos_fn to build the neutral density gradient."
        )
    # Neumann-fill density to prevent garbage gradients at coastlines.
    rho_filled = _neumann_fill_cgrid(rho, mask)

    # Face density gradients (in-situ FD of rho, or the neutral
    # ∂ρ/∂T·∇T+∂ρ/∂S·∇S form) — shared with the centred K_33 / triad builders.
    drho_dx_u, drho_dy_v, drho_dz_safe = _slope_density_face_grads(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )

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

    # drho_dz_safe already carries the stable-strat floor (min(0,·)-eps) for
    # both modes (applied inside _slope_density_face_grads).

    # --- Slopes ---
    # in_situ clips to ±S_max (legoESM safety); neutral leaves the slope
    # UNCLIPPED and lets the DM95 taper suppress steep slopes (Veros never
    # clips — see _w_triad_slopes_tapers).  dm95_taper returns S·taper, which is
    # bounded for |S| → ∞ (taper decays faster than S grows).
    S_x_raw = -drho_dx_half / drho_dz_safe
    S_y_raw = -drho_dy_half / drho_dz_safe
    if slope_density != "neutral":
        S_x_raw = jnp.clip(S_x_raw, -cfg.S_max, cfg.S_max)
        S_y_raw = jnp.clip(S_y_raw, -cfg.S_max, cfg.S_max)

    # DM95 tapering via shared helper (identical formula across grids).
    return dm95_taper(S_x_raw, S_y_raw, cfg.S_max, _EPS, cfg.taper_width_frac)


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
    kappa_Redi,
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
    kappa_GM : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        GM transport coefficient [m^2/s].  Scalar, per-column 2-D, or a 3-D
        **interface** field (the 3-D EKE closure, used directly at the nlev-1
        interfaces where the centered scheme evaluates all fluxes).
    kappa_Redi : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        Redi isopycnal diffusivity [m^2/s].  Same three forms as ``kappa_GM``.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    nlev = q.shape[-1]
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev)
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)

    # The centered scheme evaluates ALL fluxes (F_x/F_y/F_z) at the nlev-1
    # interior interfaces, so kappa is needed at the interfaces (the W-grid).
    # Three cases (see ``_kappa_is_interface_3d``):
    #   * scalar              → pass through unchanged (bit-identical).
    #   * 2-D (n_lat, n_lon)  → broadcast over interfaces via a trailing axis
    #                            (the prognostic-EKE-2D / Visbeck / K_iso=K_gm
    #                            per-column override) — bit-identical to before.
    #   * 3-D (…, nlev-1)     → the 3-D EKE interface kappa, used DIRECTLY (it
    #                            already lives at the interfaces).
    if _kappa_is_interface_3d(kappa_GM, nlev):
        kappa_GM_b = kappa_GM
    elif isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 2:
        kappa_GM_b = kappa_GM[:, :, jnp.newaxis]
    else:
        kappa_GM_b = kappa_GM
    if _kappa_is_interface_3d(kappa_Redi, nlev):
        kappa_Redi_b = kappa_Redi
    elif isinstance(kappa_Redi, jnp.ndarray) and kappa_Redi.ndim == 2:
        kappa_Redi_b = kappa_Redi[:, :, jnp.newaxis]
    else:
        kappa_Redi_b = kappa_Redi

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
    F_x_half = (kappa_Redi_b * dq_dx_half
                + (kappa_Redi_b - kappa_GM_b) * S_x * dq_dz_half)  # (n_lat, n_lon, nlev-1)
    F_y_half = (kappa_Redi_b * dq_dy_half
                + (kappa_Redi_b - kappa_GM_b) * S_y * dq_dz_half)

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
    F_z = ((kappa_Redi_b + kappa_GM_b) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi_b * S2_half * dq_dz_half)

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



def _w_face_slope_density_inputs(
    rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
    eos_fn, rho_0, g,
):
    """Build the W-face slope-density inputs for :func:`_w_triad_slopes_tapers`.

    Returns a dict of keyword arguments — ``drho_dx_u``, ``drho_dy_v``,
    ``drho_dz_w`` and (for the neutral mode) ``drdT_w``, ``drdS_w``, ``dTdx_u``,
    ``dSdx_u``, ``dTdy_v``, ``dSdy_v`` — so the K_33 getter and the realized
    GM-skew conversion (both W-face-only consumers) build the slopes through the
    SAME path as the triad ``F_z`` and never diverge.

    ``"in_situ"`` (BIT-IDENTICAL): the in-situ face gradients + floored vertical
    gradient (exactly the prior inline code).  ``"neutral"``: the locally-
    referenced ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` ingredients (W-cell ``drdT_w`` + tracer
    face gradients), with the floor applied to the neutral ``drho_dz_w``.

    kr-sum (Veros ``isoneutral.py:176-198``): each w-face triad references one
    of the TWO cells adjacent to the face, and Veros builds BOTH ``drodxb`` and
    ``drodzb`` from that reference cell's own EOS derivatives (the ``kr`` loop).
    The neutral dict therefore carries both the UPPER-cell (``drdT_w``/
    ``drho_dz_w``, the A-triads) and the LOWER-cell (``drdT_wb``/
    ``drho_dz_w_b``, the B-triads) variants.
    """
    _validate_slope_density(slope_density)
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]
    if slope_density == "neutral":
        drdT_c, drdS_c = _neutral_drho_derivs(
            T, S, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = _neumann_fill_cgrid(T, mask)
        S_filled = _neumann_fill_cgrid(S, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        drdT_w = drdT_c[:, :, :-1]
        drdS_w = drdS_c[:, :, :-1]
        drdT_wb = drdT_c[:, :, 1:]
        drdS_wb = drdS_c[:, :, 1:]
        drho_dz_w = jnp.minimum(drdT_w * dTdz_w + drdS_w * dSdz_w, -_EPS_DIV)
        drho_dz_w_b = jnp.minimum(
            drdT_wb * dTdz_w + drdS_wb * dSdz_w, -_EPS_DIV)
        return dict(
            drho_dx_u=None, drho_dy_v=None, drho_dz_w=drho_dz_w,
            slope_density="neutral", drdT_w=drdT_w, drdS_w=drdS_w,
            drdT_wb=drdT_wb, drdS_wb=drdS_wb, drho_dz_w_b=drho_dz_w_b,
            dTdx_u=dTdx_u, dSdx_u=dSdx_u, dTdy_v=dTdy_v, dSdy_v=dSdy_v,
        )
    drho_dx_u = gradient_x_cgrid(rho_filled, grid)
    drho_dy_v = gradient_y_cgrid(rho_filled, grid)
    drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV)
    drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)
    return dict(
        drho_dx_u=drho_dx_u, drho_dy_v=drho_dy_v, drho_dz_w=drho_dz_w,
        slope_density="in_situ",
    )


def _w_triad_numerators(slope_density, drho_dx_u, drho_dy_v,
                        drdT_w, drdS_w, dTdx_u, dSdx_u, dTdy_v, dSdy_v,
                        n_lat, n_lon, drdT_wb=None, drdS_wb=None):
    """The 8 W-face triad horizontal density-gradient *numerators* ``-∇_hρ``.

    Returns ``(nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb)`` where the
    A-suffix (W/E/S/N) is the upper-level (k) horizontal gradient and the
    b-suffix is the lower-level (k+1) one, all at the ``nlev-1`` w-faces.

    - ``"in_situ"`` (BIT-IDENTICAL): the corresponding slices of the in-situ
      face gradient ``drho_dx_u`` / ``drho_dy_v`` (exactly the slices the prior
      inline code used).
    - ``"neutral"``: the locally-referenced ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` form, with
      the PER-TRIAD reference cell's own EOS derivatives (Veros's ``kr`` loop,
      ``isoneutral.py:176-198``): the A-triads (upper-level tracer gradients)
      use the UPPER cell's ``drdT_w``/``drdS_w``; the B-triads (lower-level
      gradients) use the LOWER cell's ``drdT_wb``/``drdS_wb`` — Veros pairs
      ``drdT[..., kr]`` with ``dTdx[..., kr]`` for both ``drodxb`` and
      ``drodzb``.  (Before the kr-sum refinement all 8 used the upper cell.)
    """
    if slope_density == "neutral":
        if drdT_wb is None or drdS_wb is None:
            raise ValueError(
                "_w_triad_numerators: slope_density='neutral' requires the "
                "lower-cell drdT_wb/drdS_wb (the kr-sum pairing); got None. "
                "Build inputs via _w_face_slope_density_inputs.")
        nx_W = drdT_w * dTdx_u[:, :n_lon, :-1] + drdS_w * dSdx_u[:, :n_lon, :-1]
        nx_E = (drdT_w * dTdx_u[:, 1:n_lon + 1, :-1]
                + drdS_w * dSdx_u[:, 1:n_lon + 1, :-1])
        nx_Wb = (drdT_wb * dTdx_u[:, :n_lon, 1:]
                 + drdS_wb * dSdx_u[:, :n_lon, 1:])
        nx_Eb = (drdT_wb * dTdx_u[:, 1:n_lon + 1, 1:]
                 + drdS_wb * dSdx_u[:, 1:n_lon + 1, 1:])
        ny_S = drdT_w * dTdy_v[:n_lat, :, :-1] + drdS_w * dSdy_v[:n_lat, :, :-1]
        ny_N = (drdT_w * dTdy_v[1:n_lat + 1, :, :-1]
                + drdS_w * dSdy_v[1:n_lat + 1, :, :-1])
        ny_Sb = (drdT_wb * dTdy_v[:n_lat, :, 1:]
                 + drdS_wb * dSdy_v[:n_lat, :, 1:])
        ny_Nb = (drdT_wb * dTdy_v[1:n_lat + 1, :, 1:]
                 + drdS_wb * dSdy_v[1:n_lat + 1, :, 1:])
    else:
        nx_W = drho_dx_u[:, :n_lon, :-1]
        nx_E = drho_dx_u[:, 1:n_lon + 1, :-1]
        nx_Wb = drho_dx_u[:, :n_lon, 1:]
        nx_Eb = drho_dx_u[:, 1:n_lon + 1, 1:]
        ny_S = drho_dy_v[:n_lat, :, :-1]
        ny_N = drho_dy_v[1:n_lat + 1, :, :-1]
        ny_Sb = drho_dy_v[:n_lat, :, 1:]
        ny_Nb = drho_dy_v[1:n_lat + 1, :, 1:]
    return nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb


def _w_triad_slopes_tapers(drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon,
                           S_max, taper_width_frac,
                           slope_density="in_situ",
                           drdT_w=None, drdS_w=None,
                           drdT_wb=None, drdS_wb=None, drho_dz_w_b=None,
                           dTdx_u=None, dSdx_u=None, dTdy_v=None, dSdy_v=None):
    """W-face (vertical-flux) triad isopycnal slopes + DM95 tapers.

    Shared by the explicit ``F_z`` assembly (in
    ``gm_redi_tracer_tendency_triads_latlon_cgrid``), the implicit-K_33
    diffusivity getter (``compute_isoneutral_K33_latlon``), and the realized
    GM-skew EKE source so the slope numerics can never diverge.  ``A`` = upper
    level k, ``B`` = lower level k+1; the x-triads use the west/east u-faces,
    the y-triads the south/north v-faces.  Returns the 8 per-triad slopes
    followed by their 8 DM95 tapers, each at the w-faces (n_lat, n_lon, nlev-1).

    ``slope_density`` selects the density gradient that builds the slopes:
    ``"in_situ"`` (default, BIT-IDENTICAL) slices ``drho_dx_u`` / ``drho_dy_v``
    and CLIPS the slope to ±S_max before the DM95 taper (the legoESM safety
    convention); ``"neutral"`` builds ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` from the W-cell
    ``drdT_w`` / ``drdS_w`` and the raw tracer face gradients, and does NOT clip
    — Veros never clips the slope, it relies on the DM95 taper (``dm_taper``) to
    suppress steep slopes (``taper → 0`` as ``|S| → ∞``).  The neutral slope is
    ~4× steeper than the in-situ one (compressibility removed from ``∂_zρ``), so
    it routinely exceeds S_max; clipping it would saturate the taper at its
    S_max value (0.5) and inflate K_33 ~10× over Veros — the unclipped taper is
    what makes the neutral K_33 track Veros (ratio ~1.0 at the thermocline).
    """
    (nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb) = _w_triad_numerators(
        slope_density, drho_dx_u, drho_dy_v, drdT_w, drdS_w,
        dTdx_u, dSdx_u, dTdy_v, dSdy_v, n_lat, n_lon,
        drdT_wb=drdT_wb, drdS_wb=drdS_wb)
    clip = slope_density != "neutral"
    # kr-sum: the B-triads divide by the LOWER cell's drodzb (Veros pairs the
    # same kr-cell derivatives in numerator and denominator).  in_situ has no
    # per-cell derivative, so both levels share the single face denominator
    # (a neutral call without the b-variant already raised in the numerators).
    drho_dz_w_B = drho_dz_w_b if drho_dz_w_b is not None else drho_dz_w

    def _slope(num, dz=drho_dz_w):
        s = -num / dz
        return jnp.clip(s, -S_max, S_max) if clip else s

    S_Wx1 = _slope(nx_W)                    # W,A
    S_Wx2 = _slope(nx_E)                    # E,A
    S_Wx3 = _slope(nx_Wb, drho_dz_w_B)      # W,B
    S_Wx4 = _slope(nx_Eb, drho_dz_w_B)      # E,B
    S_Wy1 = _slope(ny_S)                    # S,A
    S_Wy2 = _slope(ny_N)                    # N,A
    S_Wy3 = _slope(ny_Sb, drho_dz_w_B)      # S,B
    S_Wy4 = _slope(ny_Nb, drho_dz_w_B)      # N,B
    tw = lambda s: dm95_taper_scalar(s, S_max, transition_width_frac=taper_width_frac)[1]
    return (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
            tw(S_Wx1), tw(S_Wx2), tw(S_Wx3), tw(S_Wx4),
            tw(S_Wy1), tw(S_Wy2), tw(S_Wy3), tw(S_Wy4))


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
    kappa_Redi,
    S_max: float,
    taper_width_frac: float = 0.1,
    implicit_K33: bool = False,
    K_iso_steep: float = 0.0,
    slope_density: str = "in_situ",
    T_tracer: jnp.ndarray | None = None,
    S_tracer: jnp.ndarray | None = None,
    eos_fn=None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
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
    kappa_GM : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        GM bolus coefficient [m^2/s].  Scalar (constant), per-column 2-D
        (Visbeck), or a depth-resolved 3-D **interface** field (the 3-D EKE
        closure): the interface kappa lives at the ``nlev-1`` interior
        interfaces (W-grid) and is used DIRECTLY for the vertical flux, with a
        vertical interface→center interpolation feeding the horizontal flux.
        Dispatch is by ``_kappa_is_interface_3d`` (only when ``nlev > 2``).
    kappa_Redi : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        Redi isopycnal diffusivity [m^2/s].  Same three forms as ``kappa_GM``.
        Per-column or 3-D for the K_iso=K_gm coupling (pass kappa_Redi ==
        kappa_GM): then the (kappa_Redi-kappa_GM) horizontal off-diagonal
        cancels, as in Veros enable_eke_isopycnal_diffusion.  A scalar keeps the
        path bit-identical.
    S_max : float
        Slope cap for clipping and DM95 taper.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    _validate_slope_density(slope_density)
    n_lat, n_lon, nlev = q.shape
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]

    # Resolve kappa to its center / u-face / v-face / w-face forms (shared
    # helper, identical dispatch for kappa_GM, kappa_Redi, and the K_33 getter):
    #   * scalar / 2-D (n_lat, n_lon)  → cell-centred broadcast + horizontal
    #     interp to faces, ``kappa_w = kappa_c`` — BIT-IDENTICAL to the prior
    #     code (the K_iso=K_gm coupling passes kappa_Redi == kappa_GM, so the
    #     (kappa_Redi-kappa_GM) horizontal off-diagonal still cancels, as in
    #     Veros enable_eke_isopycnal_diffusion).
    #   * 3-D interface (…, nlev-1)    → the 3-D EKE interface kappa: it is the
    #     w-face value DIRECTLY (the faithful placement for the vertical flux —
    #     no lossy round-trip), while the u/v-face horizontal forms come from a
    #     vertical interface→center interp then the usual cell→face interp.
    # The w-face triads share the SAME drho_dz_w(k+1/2) at a w-face, so a single
    # interface-level kappa_w is the correct per-w-face coefficient.
    kappa_GM_c, kappa_GM_u, kappa_GM_v, kappa_GM_w = _kappa_center_uvw(kappa_GM, nlev)
    kappa_Redi_c, kappa_Redi_u, kappa_Redi_v, kappa_Redi_w = _kappa_center_uvw(
        kappa_Redi, nlev)

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
    dq_dx_u = gradient_x_cgrid(q_filled, grid)
    dq_dy_v = gradient_y_cgrid(q_filled, grid)
    dq_dz_w = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )

    # Density gradients that BUILD the slopes.  For the per-triad u/v-face
    # slopes, the horizontal density gradient references the TRIAD's cell (Veros
    # K_11 ``drodxe`` uses ``drdT[1+ip]`` — west cell for the W triads, east for
    # the E triads); the W-face triads use the W-cell's own derivative (Veros
    # K_33 uses ``drdT[2:-2]`` for all triads).  So we build the neutral
    # vertical w-face gradient with each cell's own drdT (so the cell-shifted
    # ``_to_uface_west/east`` already select the right per-triad drho_dz), and
    # the neutral horizontal numerators per reference cell below.
    if slope_density == "neutral":
        if T_tracer is None or S_tracer is None or eos_fn is None:
            raise ValueError(
                "gm_redi_tracer_tendency_triads_latlon_cgrid: "
                "slope_density='neutral' requires T_tracer, S_tracer, eos_fn."
            )
        drdT_c, drdS_c = _neutral_drho_derivs(
            T_tracer, S_tracer, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = _neumann_fill_cgrid(T_tracer, mask)
        S_filled = _neumann_fill_cgrid(S_tracer, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        # Per-cell neutral vertical gradient at the w-face, floored for stable
        # strat — BOTH adjacent-cell variants (the Veros kr loop): the upper
        # cell's (A-triads / faces BELOW their reference cell) and the lower
        # cell's (B-triads / faces ABOVE their reference cell).  The west/east
        # (south/north) shifts below pick the per-triad reference COLUMN; the
        # up/down variant picks the reference CELL within the column, so every
        # triad pairs its own cell's EOS derivatives in numerator AND
        # denominator (Veros K_11 ``drodze`` / K_33 ``drodzb``).
        drdT_w = drdT_c[:, :, :-1]
        drdS_w = drdS_c[:, :, :-1]
        drdT_wb = drdT_c[:, :, 1:]
        drdS_wb = drdS_c[:, :, 1:]
        drho_dz_w = jnp.minimum(drdT_w * dTdz_w + drdS_w * dSdz_w, -_EPS_DIV)
        drho_dz_w_b = jnp.minimum(
            drdT_wb * dTdz_w + drdS_wb * dSdz_w, -_EPS_DIV)
        # Per-reference-cell neutral horizontal gradients at the u/v-faces, used
        # by the u/v-face slopes (S_T*/S_V*).  drdT lifted to the west/east
        # (south/north) neighbour of each u-face (v-face).
        drdT_uw = _to_uface_west(drdT_c)   # (n_lat, n_lon+1, nlev)
        drdS_uw = _to_uface_west(drdS_c)
        drdT_ue = _to_uface_east(drdT_c)
        drdS_ue = _to_uface_east(drdS_c)
        drho_dx_u_west = drdT_uw * dTdx_u + drdS_uw * dSdx_u   # W-triad numerator
        drho_dx_u_east = drdT_ue * dTdx_u + drdS_ue * dSdx_u   # E-triad numerator
        drdT_vs = _to_vface_south(drdT_c)  # (n_lat+1, n_lon, nlev)
        drdS_vs = _to_vface_south(drdS_c)
        drdT_vn = _to_vface_north(drdT_c)
        drdS_vn = _to_vface_north(drdS_c)
        drho_dy_v_south = drdT_vs * dTdy_v + drdS_vs * dSdy_v
        drho_dy_v_north = drdT_vn * dTdy_v + drdS_vn * dSdy_v
        # Single drho_dx_u / drho_dy_v are unused in the neutral path (the W-face
        # builder takes the tracer gradients + drdT_w directly); set to None so
        # any accidental in-situ slice raises.
        drho_dx_u = None
        drho_dy_v = None
    else:
        drho_dx_u = gradient_x_cgrid(rho_filled, grid)        # (n_lat, n_lon+1, nlev)
        drho_dy_v = gradient_y_cgrid(rho_filled, grid)        # (n_lat+1, n_lon, nlev)
        drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV
        )
        # Stable-strat floor: drho_dz must be negative (z UPWARD ⇒ rho denser below).
        drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)        # (n_lat, n_lon, nlev-1)
        # In-situ: the same single face gradient feeds all triads at a face.
        drho_dx_u_west = drho_dx_u
        drho_dx_u_east = drho_dx_u
        drho_dy_v_south = drho_dy_v
        drho_dy_v_north = drho_dy_v
        drdT_w = None
        drdS_w = None
        drdT_wb = None
        drdS_wb = None
        # In-situ has no per-cell EOS derivative; both adjacent cells share
        # the single face gradient (bit-identical to the prior code).
        drho_dz_w_b = drho_dz_w
        dTdx_u = dSdx_u = dTdy_v = dSdy_v = None

    # -----------------------------------------------------------------
    # 2. Pad drho_dz_w / dq_dz_w along the level axis so that boundary
    #    triads can be expressed by a single jnp.where / multiplication.
    #    "below" array at level k = drho_dz_w at w-face (k+1/2);
    #    "above" array at level k = drho_dz_w at w-face (k-1/2).
    #
    # kr pairing (neutral): a triad referencing cell k is the UPPER cell of
    # its below-face (k+1/2) ⇒ ``below_lev`` uses the upper-cell variant
    # ``drho_dz_w``; it is the LOWER cell of its above-face (k-1/2) ⇒
    # ``above_lev`` uses the lower-cell variant ``drho_dz_w_b`` (Veros
    # ``drodze``/``drodzn`` build each face gradient from the reference
    # cell's own drdT — isoneutral.py kr/ki loops).  in_situ: identical
    # arrays, bit-identical.
    # -----------------------------------------------------------------
    sentinel_rho = jnp.full(
        (n_lat, n_lon, 1), -_EPS_DIV, dtype=drho_dz_w.dtype,
    )
    drho_dz_below_lev = jnp.concatenate([drho_dz_w, sentinel_rho], axis=-1)
    drho_dz_above_lev = jnp.concatenate([sentinel_rho, drho_dz_w_b], axis=-1)

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

    # Raw slopes — DO NOT taper here; taper goes on the whole flux.
    # T1/T2 reference the WEST cell, T3/T4 the EAST cell (Veros K_11 drodxe uses
    # drdT[1+ip]); in-situ both branches share the single face gradient so this
    # is bit-identical there.  in_situ clips to ±S_max (legoESM safety); neutral
    # leaves the slope UNCLIPPED (Veros relies on the DM95 taper — see
    # _w_triad_slopes_tapers) since neutral slopes routinely exceed S_max.
    _clip_slope = slope_density != "neutral"

    def _uvslope(num, dz):
        s = -num / dz
        return jnp.clip(s, -S_max, S_max) if _clip_slope else s

    S_T1 = _uvslope(drho_dx_u_west, drho_dz_T1)
    S_T2 = _uvslope(drho_dx_u_west, drho_dz_T2)
    S_T3 = _uvslope(drho_dx_u_east, drho_dz_T3)
    S_T4 = _uvslope(drho_dx_u_east, drho_dz_T4)

    taper_T1 = dm95_taper_scalar(S_T1, S_max, transition_width_frac=taper_width_frac)[1]
    taper_T2 = dm95_taper_scalar(S_T2, S_max, transition_width_frac=taper_width_frac)[1]
    taper_T3 = dm95_taper_scalar(S_T3, S_max, transition_width_frac=taper_width_frac)[1]
    taper_T4 = dm95_taper_scalar(S_T4, S_max, transition_width_frac=taper_width_frac)[1]

    N_valid_u = valid_T1 + valid_T2 + valid_T3 + valid_T4
    N_valid_u_safe = jnp.maximum(N_valid_u, 1.0)
    w_T1 = valid_T1 / N_valid_u_safe
    w_T2 = valid_T2 / N_valid_u_safe
    w_T3 = valid_T3 / N_valid_u_safe
    w_T4 = valid_T4 / N_valid_u_safe

    # Per-triad full flux (cancels exactly when q = f(ρ)).
    flux_T1 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T1 * dq_dz_T1
    flux_T2 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T2 * dq_dz_T2
    flux_T3 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T3 * dq_dz_T3
    flux_T4 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T4 * dq_dz_T4

    F_x_u = (w_T1 * taper_T1 * flux_T1
           + w_T2 * taper_T2 * flux_T2
           + w_T3 * taper_T3 * flux_T3
           + w_T4 * taper_T4 * flux_T4)
    if K_iso_steep > 0.0:
        # Veros K_11 steep-slope floor (isoneutral.py:128): lift the along-
        # isopycnal diffusivity to K_iso_steep where DM95 tapered it below,
        # reverting to horizontal diffusion at steep slopes.  Additive deficit on
        # the diagonal (dq/dx) only; the skew is untouched.  K_iso_steep=0 ⇒ no-op.
        F_x_u = F_x_u + dq_dx_u * (
            w_T1 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T1)
            + w_T2 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T2)
            + w_T3 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T3)
            + w_T4 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T4))
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
    # V1/V2 reference the SOUTH cell, V3/V4 the NORTH cell (Veros K_22 drodyb);
    # in-situ both share the single face gradient (bit-identical).  neutral
    # leaves the slope unclipped (see S_T* above / _w_triad_slopes_tapers).
    S_V1 = _uvslope(drho_dy_v_south, drho_dz_V1)
    S_V2 = _uvslope(drho_dy_v_south, drho_dz_V2)
    S_V3 = _uvslope(drho_dy_v_north, drho_dz_V3)
    S_V4 = _uvslope(drho_dy_v_north, drho_dz_V4)

    taper_V1 = dm95_taper_scalar(S_V1, S_max, transition_width_frac=taper_width_frac)[1]
    taper_V2 = dm95_taper_scalar(S_V2, S_max, transition_width_frac=taper_width_frac)[1]
    taper_V3 = dm95_taper_scalar(S_V3, S_max, transition_width_frac=taper_width_frac)[1]
    taper_V4 = dm95_taper_scalar(S_V4, S_max, transition_width_frac=taper_width_frac)[1]

    N_valid_v = valid_V1 + valid_V2 + valid_V3 + valid_V4
    N_valid_v_safe = jnp.maximum(N_valid_v, 1.0)
    w_V1 = valid_V1 / N_valid_v_safe
    w_V2 = valid_V2 / N_valid_v_safe
    w_V3 = valid_V3 / N_valid_v_safe
    w_V4 = valid_V4 / N_valid_v_safe

    flux_V1 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V1 * dq_dz_V1
    flux_V2 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V2 * dq_dz_V2
    flux_V3 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V3 * dq_dz_V3
    flux_V4 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V4 * dq_dz_V4

    F_y_v = (w_V1 * taper_V1 * flux_V1
           + w_V2 * taper_V2 * flux_V2
           + w_V3 * taper_V3 * flux_V3
           + w_V4 * taper_V4 * flux_V4)
    if K_iso_steep > 0.0:
        # Veros K_22 steep-slope floor (isoneutral.py:165), as for F_x_u above.
        F_y_v = F_y_v + dq_dy_v * (
            w_V1 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V1)
            + w_V2 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V2)
            + w_V3 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V3)
            + w_V4 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V4))
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
    # Tracer gradients at the 8 triad positions ("A" = level k, "B" = k+1;
    # west/east u-faces and south/north v-faces).
    dq_dx_west_A = dq_dx_u[:, :n_lon, :-1]
    dq_dx_west_B = dq_dx_u[:, :n_lon, 1:]
    dq_dx_east_A = dq_dx_u[:, 1:n_lon + 1, :-1]
    dq_dx_east_B = dq_dx_u[:, 1:n_lon + 1, 1:]
    dq_dy_south_A = dq_dy_v[:n_lat, :, :-1]
    dq_dy_south_B = dq_dy_v[:n_lat, :, 1:]
    dq_dy_north_A = dq_dy_v[1:n_lat + 1, :, :-1]
    dq_dy_north_B = dq_dy_v[1:n_lat + 1, :, 1:]

    # rho-based per-triad slopes + DM95 tapers (shared with the K_33 getter so
    # the explicit F_z and the implicit K_33 use bit-identical slopes/tapers).
    # Neutral mode passes the W-cell drdT_w/drdS_w + tracer face gradients so the
    # W-face slopes use the same ∂ρ/∂T·∇T+∂ρ/∂S·∇S form Veros K_33 uses.
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     taper_Wx1, taper_Wx2, taper_Wx3, taper_Wx4,
     taper_Wy1, taper_Wy2, taper_Wy3, taper_Wy4) = _w_triad_slopes_tapers(
        drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon, S_max, taper_width_frac,
        slope_density=slope_density, drdT_w=drdT_w, drdS_w=drdS_w,
        drdT_wb=drdT_wb, drdS_wb=drdS_wb,
        drho_dz_w_b=(drho_dz_w_b if slope_density == "neutral" else None),
        dTdx_u=dTdx_u, dSdx_u=dSdx_u, dTdy_v=dTdy_v, dSdy_v=dSdy_v)

    # Per-triad vertical flux.  The off-diagonal skew (kR+kG)·S·dq/dx is ALWAYS
    # explicit.  The DIAGONAL K_33 term (kR·S²·dq/dz — the "enhanced vertical
    # mixing ∝ S²") is added here only when NOT ``implicit_K33``; otherwise it is
    # applied implicitly by the model step (Veros core/isoneutral/diffusion.py),
    # with ``compute_isoneutral_K33_latlon`` supplying the matching diffusivity.
    # The per-triad algebraic cancellation for q = f(ρ) is preserved either way
    # (each term cancels independently; taper × averaging keeps the exact zero).
    flux_Wx1 = (kappa_Redi_w + kappa_GM_w) * S_Wx1 * dq_dx_west_A
    flux_Wx2 = (kappa_Redi_w + kappa_GM_w) * S_Wx2 * dq_dx_east_A
    flux_Wx3 = (kappa_Redi_w + kappa_GM_w) * S_Wx3 * dq_dx_west_B
    flux_Wx4 = (kappa_Redi_w + kappa_GM_w) * S_Wx4 * dq_dx_east_B
    flux_Wy1 = (kappa_Redi_w + kappa_GM_w) * S_Wy1 * dq_dy_south_A
    flux_Wy2 = (kappa_Redi_w + kappa_GM_w) * S_Wy2 * dq_dy_north_A
    flux_Wy3 = (kappa_Redi_w + kappa_GM_w) * S_Wy3 * dq_dy_south_B
    flux_Wy4 = (kappa_Redi_w + kappa_GM_w) * S_Wy4 * dq_dy_north_B
    if not implicit_K33:
        flux_Wx1 = flux_Wx1 + kappa_Redi_w * S_Wx1 ** 2 * dq_dz_w
        flux_Wx2 = flux_Wx2 + kappa_Redi_w * S_Wx2 ** 2 * dq_dz_w
        flux_Wx3 = flux_Wx3 + kappa_Redi_w * S_Wx3 ** 2 * dq_dz_w
        flux_Wx4 = flux_Wx4 + kappa_Redi_w * S_Wx4 ** 2 * dq_dz_w
        flux_Wy1 = flux_Wy1 + kappa_Redi_w * S_Wy1 ** 2 * dq_dz_w
        flux_Wy2 = flux_Wy2 + kappa_Redi_w * S_Wy2 ** 2 * dq_dz_w
        flux_Wy3 = flux_Wy3 + kappa_Redi_w * S_Wy3 ** 2 * dq_dz_w
        flux_Wy4 = flux_Wy4 + kappa_Redi_w * S_Wy4 ** 2 * dq_dz_w

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
    rho_0: float = _RHO_0,
    g: float = constants.g,
    kappa_gm_override: jnp.ndarray | None = None,
    kappa_redi_override: jnp.ndarray | None = None,
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
        z_coord.dz_ref, rho_0, g,
        n_iter=2,
    )

    slope_density = getattr(cfg, "slope_density", "in_situ")

    # Centred interface slopes — used for Visbeck (only ⟨N|S|⟩_z is
    # needed; the cancellation property of triads is irrelevant there).
    # T, S, eos_fn are passed so the neutral mode can build the
    # locally-referenced gradient; ignored for the default in-situ mode.
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(
        rho, mask, z_coord, jacobian, grid, cfg,
        T=T, S=S, eos_fn=eos_fn, rho_0=rho_0, g=g,
    )

    # GM coefficient. Precedence: prognostic-EKE override (computed by the step
    # from the evolving eddy-energy field) > Visbeck diagnostic > constant.
    if kappa_gm_override is not None:
        kappa_GM = kappa_gm_override
    elif cfg.visbeck.enabled:
        if f_coriolis is None:
            # Use the grid's Coriolis field (f = 2·Ω·sin(lat), already built
            # with the grid's pinned rotation rate) rather than re-deriving from
            # the module constant — keeps Visbeck consistent with the pinned Ω
            # (e.g. a Veros recipe) and avoids an inline constants.Omega read.
            f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Redi isopycnal diffusivity. K_iso = K_gm (prognostic) when the override is
    # supplied (Veros enable_eke_isopycnal_diffusion -> the step passes
    # kappa_redi_override = kappa_gm_override); else the constant cfg.kappa_Redi.
    kappa_Redi_eff = cfg.kappa_Redi if kappa_redi_override is None else kappa_redi_override

    scheme = getattr(cfg, "slope_scheme", "triads")
    if scheme == "triads":
        dT_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff, cfg.S_max,
            cfg.taper_width_frac, cfg.implicit_K33, cfg.K_iso_steep,
            slope_density=slope_density, T_tracer=T, S_tracer=S,
            eos_fn=eos_fn, rho_0=rho_0, g=g,
        )
        dS_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            S, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff, cfg.S_max,
            cfg.taper_width_frac, cfg.implicit_K33, cfg.K_iso_steep,
            slope_density=slope_density, T_tracer=T, S_tracer=S,
            eos_fn=eos_fn, rho_0=rho_0, g=g,
        )
    elif scheme == "centered":
        dT_dt = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff,
        )
        dS_dt = gm_redi_tracer_tendency_latlon_cgrid(
            S, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff,
        )
        # --- Near-surface horizontal diffusion complement ---
        # In the mixed layer, DM95 tapers Redi to zero, leaving no
        # horizontal tracer mixing.  Following Ferrari et al. (2008,
        # J. Climate, 21, 2770-2789), add horizontal diffusion with
        # coefficient kappa_Redi in the boundary layer so total
        # diffusivity is always kappa_Redi.
        #
        # Use a fixed depth proxy rather than the DM95 taper,
        # because the taper-based complement was ineffective (taper ≈ 1
        # where the 2Δy feedback operates).  When KPP is active, this
        # should be replaced with the KPP-diagnosed boundary layer depth.
        if cfg.surface_complement:
            z_full = z_coord.z_full_ref  # (nlev,) — negative depths
            complement = jnp.where(
                jnp.abs(z_full) < cfg.surface_complement_depth, 1.0, 0.0
            )  # (nlev,) — broadcast over (n_lat, n_lon)
            for q_field, tend_ref in [(T, 'dT_dt'), (S, 'dS_dt')]:
                q_filled = _neumann_fill_cgrid(q_field, mask)
                dq_dx_u = gradient_x_cgrid(q_filled, grid) * u_mask[:, :, jnp.newaxis]
                dq_dy_v = gradient_y_cgrid(q_filled, grid) * v_mask[:, :, jnp.newaxis]
                # complement is (nlev,) — broadcasts over spatial dims. This
                # boundary-layer term deliberately uses the CONSTANT cfg.kappa_Redi,
                # NOT kappa_Redi_eff: the K_iso=K_gm override applies to the
                # isopycnal-tensor fluxes (above); the Ferrari (2008) surface
                # complement is a separate fixed-diffusivity term. Scoped limitation
                # for a centered + surface_complement + prognostic-override config
                # (making it override-aware needs cell->u/v-face interp of the
                # array kappa); INERT for ACC, which uses slope_scheme="triads".
                F_x = cfg.kappa_Redi * complement * dq_dx_u
                F_y = cfg.kappa_Redi * complement * dq_dy_v
                dq_complement = divergence_cgrid(F_x, F_y, grid) * mask[:, :, jnp.newaxis]
                if tend_ref == 'dT_dt':
                    dT_dt = dT_dt + dq_complement
                else:
                    dS_dt = dS_dt + dq_complement
    else:
        raise ValueError(
            f"Unknown GMRediConfig.slope_scheme={scheme!r}; "
            f"expected 'centered' or 'triads'."
        )

    return dT_dt, dS_dt


def compute_isoneutral_K33_latlon(
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
    rho_0: float = _RHO_0,
    g: float = constants.g,
    kappa_redi_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Vertical isoneutral diffusivity K_33 at w-faces, for the implicit solve.

    ``K_33 = 0.25 · kappa_Redi · Σ_8 taper_W · S_W²``  →  (n_lat, n_lon, nlev-1), ≥ 0.

    This is EXACTLY the coefficient of dq/dz in the explicit triad ``F_z`` diagonal
    term that ``gm_redi_tracer_tendency_triads_latlon_cgrid`` drops when
    ``implicit_K33=True`` (same rho via the 2-iteration EOS coupling, same shared
    ``_w_triad_slopes_tapers``, same ``kappa_Redi`` w-face placement via
    ``_kappa_center_uvw`` — so the explicit-drop and the implicit-add are
    consistent; ``kappa_Redi`` may be a scalar, a 2-D per-column array, or a 3-D
    interface field, just as in the triad ``F_z``).  The lat-lon C-grid model
    step adds it to the implicit vertical-diffusion ``K_v`` (Veros
    core/isoneutral/diffusion.py:154, ``delta = dt/dzw · K_33``), applying the
    vertical isoneutral diffusion backward-Euler-implicitly as in Veros rather than
    explicitly.  Density-independent of the tracer, so one call serves both T and S.
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: _neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    kappa_Redi = cfg.kappa_Redi if kappa_redi_override is None else kappa_redi_override
    nlev = T.shape[-1]
    # K_33 is evaluated at the nlev-1 w-faces, so kappa_Redi is needed there.
    # The shared dispatch (``_kappa_center_uvw``) puts a 3-D interface kappa on
    # the w-faces DIRECTLY (== the 4th return) and a scalar / 2-D per-column
    # kappa as its cell-centred broadcast (bit-identical to the prior inline
    # 2-D handling).  Take only the w-face form here.
    kappa_Redi_w = _kappa_center_uvw(kappa_Redi, nlev)[3]
    n_lat, n_lon = mask.shape
    rho_filled = _neumann_fill_cgrid(rho, mask)
    slope_density = getattr(cfg, "slope_density", "in_situ")
    w_inputs = _w_face_slope_density_inputs(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     tWx1, tWx2, tWx3, tWx4, tWy1, tWy2, tWy3, tWy4) = _w_triad_slopes_tapers(
        n_lat=n_lat, n_lon=n_lon, S_max=cfg.S_max,
        taper_width_frac=cfg.taper_width_frac, **w_inputs)
    K_33 = 0.25 * kappa_Redi_w * (
        tWx1 * S_Wx1 ** 2 + tWx2 * S_Wx2 ** 2 + tWx3 * S_Wx3 ** 2 + tWx4 * S_Wx4 ** 2
        + tWy1 * S_Wy1 ** 2 + tWy2 * S_Wy2 ** 2 + tWy3 * S_Wy3 ** 2 + tWy4 * S_Wy4 ** 2)
    return K_33 * mask[:, :, jnp.newaxis]


# =====================================================================
# EKE SOURCE augmentation (Veros K_diss_h + realized -P_diss_skew)
# =====================================================================

def harmonic_lateral_kediss_eke_source(
    visc_u: jnp.ndarray,
    visc_v: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    kdiss_h_cell: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mean-KE removal by the harmonic lateral viscosity, routed to the EKE source.

    Returns the rate at which the harmonic lateral viscosity removes mean kinetic
    energy, as a NON-NEGATIVE EKE source [m²/s³] at the ``nlev-1`` interior
    interfaces (the W-grid) — legoESM's analogue of Veros's ``K_diss_h``
    (``veros/core/friction.py`` ``harmonic_friction`` → ``calc_diss_u``/``calc_diss_v``,
    fed into the EKE forcing at ``veros/core/eke.py:110``).

    TWO config-selectable discretisations (``EKEConfig.kdiss_h_flux_form``):

    **Flux form (FAITHFUL, positive-definite; ``kdiss_h_flux_form=True``, ACC recipe).**
    ``kdiss_h_cell`` is the pre-computed cell-centre positive-definite dissipation
    density [m²/s³], MATCHED to the selected lateral-viscosity operator (each is the
    exact KE-removal of its own operator, so it pairs with whichever viscosity the
    config selects):
      * VECTOR Laplacian → ``A_h·(div² + <ζ²>)`` — the Helmholtz KE-removal
        ``-∫u·∇²_vec u = ∫(div²+|ζ|²) ≥ 0`` — from
        :func:`...latlon_cgrid_operators.vector_laplacian_dissipation_cgrid`.
      * FLUX-DIVERGENCE → the componentwise ``A_h·|∇u|² = 0.5·Σ(Δu·flux)``
        (Veros ``calc_diss_u/v``) from
        :func:`...latlon_cgrid_operators.flux_divergence_viscosity_cgrid`.
    This branch only averages that density to the ``nlev-1`` interior interfaces (the
    SAME W-grid mapping Veros's ``dissipation_on_wgrid`` uses for the interior).  It is
    ``≥ 0`` EVERYWHERE by construction so NO clamp is applied, and (probe-verified on
    the spun-up ACC state) its domain integral equals the mean KE actually removed by
    ``A_h`` to ~0.3% and matches Veros's captured ``K_diss_h`` to ~7%.  (The
    flux-divergence operator IS Veros's componentwise form exactly; the vector-Laplacian
    form replaces the squared one-sided gradients with squared div/curl because that
    operator carries the metric/curvature coupling the componentwise form omits.)

    **Dynamical form (DEFAULT, ``kdiss_h_flux_form=False``; ``kdiss_h_cell=None``).**
    Uses the KE-tendency ``-u·(A_h∇²_vec u) - v·(A_h∇²_vec v)`` from the supplied
    ``visc_u``/``visc_v`` momentum tendencies, averaged faces→centres→interfaces and
    CLAMPED ``≥ 0``.  This form is NOT positive-definite (it carries the transport
    divergence ``-∇·(A_h u∇u)`` ⇒ ~35% of wet cells negative); the per-cell clamp
    truncates those, OVER-CREDITING the domain-integrated KE dissipation by ~11–20%
    on the ACC channel (probe-measured; see
    .physics-validator/eke_source/probe_kdiss_conservation.py).  Retained as the
    default so existing runs stay BIT-IDENTICAL; the ACC recipe opts into the flux
    form for the exact, clamp-free Veros analogue.

    Either way the energy added to EKE is the mean KE removed by ``A_h``: the sink
    already happens in the momentum tendency (``du_dt`` carries ``A_h∇²_vec u``), and
    this term credits that lost KE to the eddy field (Veros's mean→eddy pathway).

    Parameters
    ----------
    visc_u : (n_lat, n_lon+1, nlev) — harmonic-viscosity u-tendency ``A_h∇²u`` [m/s²]
        (used only by the dynamical-form / ``kdiss_h_cell=None`` path).
    visc_v : (n_lat+1, n_lon, nlev) — harmonic-viscosity v-tendency ``A_h∇²v`` [m/s²]
        (dynamical-form path only).
    u : (n_lat, n_lon+1, nlev) — zonal velocity at u-faces [m/s] (dynamical-form path).
    v : (n_lat+1, n_lon, nlev) — meridional velocity at v-faces [m/s] (dynamical path).
    grid : LatLonGrid (unused metric-wise — the C-grid averaging is index-based; kept
        for signature parity with the other EKE source builders).
    mask : (n_lat, n_lon) — ocean mask (1 = ocean).
    kdiss_h_cell : (n_lat, n_lon, nlev) — OPTIONAL pre-computed positive-definite
        cell-centre dissipation density [m²/s³], matched to the lateral-viscosity
        operator (``A_h(div²+<ζ²>)`` vector Laplacian / ``A_h|∇u|²`` flux-divergence).
        When provided (the flux-form path), it is mapped to the W-grid with NO clamp;
        ``visc_*``, ``u``, ``v`` are then unused.  ``None`` (default) = dynamical form.

    Returns
    -------
    K_diss_h : (n_lat, n_lon, nlev-1) — non-negative EKE source [m²/s³] at interfaces.
    """
    if kdiss_h_cell is not None:
        # FAITHFUL FLUX FORM: kdiss_h_cell is already the positive-definite per-cell
        # dissipation density ≥ 0 matched to the operator (A_h·(div²+<ζ²>) from
        # vector_laplacian_dissipation_cgrid, or A_h·|∇u|² from
        # flux_divergence_viscosity_cgrid) — built from the SAME velocities / A_h
        # scaling the applied viscous tendency uses.  Average the full-level (nlev) cell
        # density to the nlev-1 interior interfaces — the same interior W-grid
        # mapping Veros's dissipation_on_wgrid uses (0.5·(c[:-1]+c[1:])).  No clamp:
        # the density is ≥ 0 everywhere by construction, so the W-grid average is too.
        diss_cell = kdiss_h_cell * mask[:, :, jnp.newaxis]
        K_diss_h_w = 0.5 * (diss_cell[:, :, :-1] + diss_cell[:, :, 1:])
        return K_diss_h_w * mask[:, :, jnp.newaxis]

    # DYNAMICAL FORM (default): per-face KE-dissipation rate -u·(A_h∇²u) [m²/s³ on
    # the face].  visc_* are already face-masked by the caller.
    p_u = -u * visc_u                                   # (n_lat, n_lon+1, nlev)
    p_v = -v * visc_v                                   # (n_lat+1, n_lon, nlev)
    # Average the face products to cell centres (inverse of cell->face interp).
    # u-face j and j+1 straddle cell j: cell value = 0.5*(p_u[:, :-1] + p_u[:, 1:]).
    diss_cell = 0.5 * (p_u[:, :-1, :] + p_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    # v-face i and i+1 straddle cell i: cell value = 0.5*(p_v[:-1] + p_v[1:]).
    diss_cell = diss_cell + 0.5 * (p_v[:-1, :, :] + p_v[1:, :, :])
    diss_cell = diss_cell * mask[:, :, jnp.newaxis]
    # Average full-level cell dissipation to the nlev-1 interior interfaces (W-grid),
    # matching where E lives, then clamp >= 0 to make this a pure source.  NB: the
    # clamp is NOT inactive — -u·A_h∇²u is locally negative in transport regions, so
    # the clamp over-credits the column-integrated KE dissipation by ~11-20% vs the
    # positive-definite flux form above (set kdiss_h_flux_form=True for the exact,
    # clamp-free Veros analogue).
    K_diss_h_w = 0.5 * (diss_cell[:, :, :-1] + diss_cell[:, :, 1:])
    return jnp.maximum(K_diss_h_w, 0.0) * mask[:, :, jnp.newaxis]


def compute_realized_gm_skew_conversion(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    kappa_gm_w: jnp.ndarray,
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Realized GM-skew buoyancy conversion ``-P_diss_skew`` as an EKE source [m²/s³].

    The Gent-McWilliams skew (bolus) flux releases mean available potential energy
    into eddy energy at the rate ``-P_diss_skew = -(g/ρ₀)·∇₃ρ·F_skew`` (Veros
    ``veros/core/isoneutral/diffusion.py:234-281``), where ``F_skew = κ_GM·S·∂ρ/∂z``
    is the per-triad GM skew flux of density.  Using ``S = -∇_h ρ / ∂_z ρ`` per
    triad and contracting with ``∇₃ρ`` gives the closed W-grid form

        -P_diss_skew(z) = κ_GM(z) · (g/ρ₀)·|∂ρ/∂z|_w · ( 0.25·Σ_8 taper·S² )
                        = κ_GM(z) · N²_w · <S²>_triad,                        (≥ 0)

    i.e. the GM coefficient times the true local buoyancy frequency ``N²_w =
    (g/ρ₀)|∂ρ/∂z|`` times the per-triad slope-variance ``<S²>_triad = 0.25·Σ taper·S²``
    (which is exactly ``K_33/κ_Redi`` from :func:`compute_isoneutral_K33_latlon`).

    This is the *realized* conversion — it differs from the *parameterized* EKE
    production ``κ_GM·σ²`` with ``σ = <N|S|>`` (a squared slope AVERAGE) because the
    per-triad slope VARIANCE ``<S²> ≥ <S>²`` (Jensen): the realized form keeps the
    discrete slope variance the GM tracer flux actually transports, which the
    pre-averaged Visbeck σ under-counts.  It reuses the SHARED per-triad W-face
    slopes/tapers (``_w_triad_slopes_tapers``) — the SAME slopes the GM/Redi skew
    flux and the implicit ``K_33`` use — so no slope numerics are duplicated.

    Sign / positivity: ``N²_w ≥ 0`` (stable-strat floor ``|∂ρ/∂z| ≥ _EPS_DIV``),
    ``S² ≥ 0``, ``taper ≥ 0``, ``κ_GM ≥ 0`` ⇒ the source is ≥ 0 everywhere (slumping
    isopycnals release mean APE into EKE). The energy comes from the mean APE that
    the GM skew flux flattens — the same flux already applied to the tracer
    tendency; documenting the routing: this credits that released APE to EKE.

    Parameters
    ----------
    T, S, eta, H_bathy, grid, z_coord, cfg, eos/eos_linear/mask/rho_0/g : as in
        :func:`compute_isoneutral_K33_latlon` (same rho via the 2-iteration EOS
        coupling, same shared slopes).
    kappa_gm_w : (n_lat, n_lon, nlev-1) — the GM coefficient ``κ_GM(z)`` at the
        interior W-faces (the prognostic-EKE 3-D override; the SAME kappa fed to the
        GM tracer flux), so the released APE is consistent with the applied skew flux.

    Returns
    -------
    P_skew : (n_lat, n_lon, nlev-1) — non-negative realized GM-skew EKE source [m²/s³].
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: _neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    n_lat, n_lon = mask.shape
    rho_filled = _neumann_fill_cgrid(rho, mask)
    slope_density = getattr(cfg, "slope_density", "in_situ")
    w_inputs = _w_face_slope_density_inputs(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )
    drho_dz_w = w_inputs["drho_dz_w"]      # (n_lat, n_lon, nlev-1)
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     tWx1, tWx2, tWx3, tWx4, tWy1, tWy2, tWy3, tWy4) = _w_triad_slopes_tapers(
        n_lat=n_lat, n_lon=n_lon, S_max=cfg.S_max,
        taper_width_frac=cfg.taper_width_frac, **w_inputs)
    # Per-triad slope variance <S²>_triad = 0.25·Σ taper·S²  (= K_33/κ_Redi).
    S2_triad = 0.25 * (
        tWx1 * S_Wx1 ** 2 + tWx2 * S_Wx2 ** 2 + tWx3 * S_Wx3 ** 2 + tWx4 * S_Wx4 ** 2
        + tWy1 * S_Wy1 ** 2 + tWy2 * S_Wy2 ** 2 + tWy3 * S_Wy3 ** 2 + tWy4 * S_Wy4 ** 2)
    # True local buoyancy frequency at the W-faces: N²_w = (g/ρ₀)|∂ρ/∂z| ≥ 0.
    # In the neutral mode this is the locally-referenced N² (compressibility
    # bias removed), consistent with the neutral slope variance above.
    N2_w = (g / rho_0) * jnp.abs(drho_dz_w)
    P_skew = jnp.maximum(kappa_gm_w, 0.0) * N2_w * S2_triad
    return jnp.maximum(P_skew, 0.0) * mask[:, :, jnp.newaxis]


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


def eke_horizontal_transport(E, U_bar, V_bar, grid, eke_cfg, mask, u_mask, v_mask):
    """Conservative 2-D EKE transport tendency [m^2/s^3]: flux-form advection of the
    eddy-energy field ``E`` by the depth-mean flow (upwind) + lateral diffusion
    (``k_iso``). Returns ``dE/dt|transport``.

    Conserves the area-integral of E by construction: the advective flux divergence
    telescopes (periodic in lon; v-flux = 0 at the N/S walls) and the lateral
    diffusion is flux-form (``laplacian_cgrid`` = div of grad, no-flux walls).
    Reuses ``divergence_cgrid`` + ``laplacian_cgrid`` — no duplicate numerics.

    Parameters
    ----------
    E : (n_lat, n_lon) eddy kinetic energy.
    U_bar : (n_lat, n_lon+1) depth-mean zonal velocity at u-faces.
    V_bar : (n_lat+1, n_lon) depth-mean meridional velocity at v-faces (0 at poles).
    grid, eke_cfg (k_iso), mask/u_mask/v_mask.
    """
    # E upwinded to u-faces by U_bar sign (periodic in lon).
    E_west = jnp.roll(E, 1, axis=1)                          # E[:, j-1]
    E_uface_core = jnp.where(U_bar[:, :-1] > 0.0, E_west, E)  # upwind
    E_uface = jnp.concatenate([E_uface_core, E_uface_core[:, 0:1]], axis=1)
    # E upwinded to interior v-faces by V_bar sign; poles are walls (V_bar=0).
    E_vface_int = jnp.where(V_bar[1:-1, :] > 0.0, E[:-1, :], E[1:, :])
    zero_row = jnp.zeros((1, E.shape[1]), dtype=E.dtype)
    E_vface = jnp.concatenate([zero_row, E_vface_int, zero_row], axis=0)
    # Flux-form advection (conservative).
    flux_u = E_uface * U_bar * u_mask
    flux_v = E_vface * V_bar * v_mask
    adv = -divergence_cgrid(flux_u, flux_v, grid)
    # Lateral diffusion (conservative).
    diff = eke_cfg.k_iso * laplacian_cgrid(E, grid, mask=mask)
    return (adv + diff) * mask


def eke_3d_horizontal_transport(E, U, V, grid, eke_cfg, mask, u_mask, v_mask):
    """Conservative 3-D EKE transport tendency [m^2/s^3] at the interior interfaces
    (the ``eke_3d=True`` path): flux-form advection of the eddy-energy field ``E``
    by the flow at EACH interface level (upwind) + per-level lateral diffusion
    (``k_iso``). Returns ``dE/dt|transport`` with shape ``(n_lat, n_lon, nlev-1)``.

    This is the depth-resolved generalisation of the 2-D
    :func:`eke_horizontal_transport`: ``E`` is advected by the LOCAL flow ``u(z),
    v(z)`` at each interface (Veros advects ``vs.eke`` by the W-grid velocities),
    NOT by the depth-mean flow. The conservative flux-form, upwind face values,
    and the reuse of ``divergence_cgrid`` (3-D-native) + ``laplacian_cgrid``
    (3-D-native, per level) are identical to the 2-D path — only the arrays carry an
    extra level axis, so the per-level numerics are bit-identical to the 2-D scheme
    at each level and no numerics are duplicated.

    Conservation: at every interface level the advective flux divergence telescopes
    (periodic in lon; v-flux = 0 at the N/S walls) and the lateral diffusion is
    flux-form (div of grad, no-flux walls), so the volume-integral
    ``Σ E·area·dz`` is conserved by construction (each level's area-integral is).

    NOTE (parity with Veros): Veros uses ``max(500, K_gm)·∇E`` for the lateral
    diffusivity of EKE (``veros/core/eke.py:173,183``); legoESM keeps the validated
    2-D path's constant ``eke_cfg.k_iso·∇E`` here for continuity (a depth-/flow-
    independent lateral diffusivity). The W-grid VERTICAL advection of E (Veros's
    ``adv_flux_top`` / ``flux_top`` Adams-Bashforth term) is NOT included here — see
    the module/stage notes; it is a documented known omission (the dominant 3-D
    effects — depth-resolved source/sink, implicit vertical diffusion, and per-level
    horizontal advection + lateral diffusion — are all present).

    Parameters
    ----------
    E : (n_lat, n_lon, nlev-1) eddy kinetic energy at interior interfaces.
    U : (n_lat, n_lon+1, nlev-1) zonal velocity at u-faces, per interface level.
    V : (n_lat+1, n_lon, nlev-1) meridional velocity at v-faces (0 at poles).
    grid, eke_cfg (k_iso), mask/u_mask/v_mask (2-D; broadcast over levels).
    """
    # E upwinded to u-faces by U sign at each level (periodic in lon).
    E_west = jnp.roll(E, 1, axis=1)                          # E[:, j-1, :]
    E_uface_core = jnp.where(U[:, :-1, :] > 0.0, E_west, E)   # upwind, (n_lat, n_lon, nlev-1)
    E_uface = jnp.concatenate([E_uface_core, E_uface_core[:, 0:1, :]], axis=1)
    # E upwinded to interior v-faces by V sign; poles are walls (V=0).
    E_vface_int = jnp.where(V[1:-1, :, :] > 0.0, E[:-1, :, :], E[1:, :, :])
    zero_row = jnp.zeros((1, E.shape[1], E.shape[2]), dtype=E.dtype)
    E_vface = jnp.concatenate([zero_row, E_vface_int, zero_row], axis=0)
    # Flux-form advection (conservative); face masks broadcast over the level axis.
    flux_u = E_uface * U * u_mask[:, :, jnp.newaxis]
    flux_v = E_vface * V * v_mask[:, :, jnp.newaxis]
    adv = -divergence_cgrid(flux_u, flux_v, grid)            # 3-D-native
    # Lateral diffusion (conservative), per level (laplacian_cgrid is 3-D-native).
    diff = eke_cfg.k_iso * laplacian_cgrid(E, grid, mask=mask)
    return (adv + diff) * mask[:, :, jnp.newaxis]


def eke_3d_vertical_diffusion(E, A_v_profile, dz_w, dz_half_w, dt, eke_cfg):
    """Backward-Euler implicit vertical diffusion of the 3-D eddy-energy field ``E``
    (the ``eke_3d=True`` path), with diffusivity ``K = alpha_eke · A_v`` — Veros's
    ``delta = dt/dzt · 0.5(kappaM[k]+kappaM[k+1]) · alpha_eke`` (``veros/core/eke.py``
    :134-142), here as a clean reuse of the shared
    :func:`implicit_vertical_diffusion_ocean` (no duplicate tridiagonal numerics).

    ``E`` lives on the interior interfaces (the W-grid, ``M = nlev-1`` levels), so the
    implicit diffusion operates over those ``M`` levels with zero-flux BCs at the top
    and bottom of the W-grid column. The caller (the model step, a later build stage)
    supplies the W-grid metrics ``dz_w`` (M layer thicknesses) + ``dz_half_w`` (M-1
    spacings between adjacent W-levels) and the vertical viscosity ``A_v_profile`` at
    the ``M-1`` interior W-interfaces. ``alpha_eke`` scales it (Veros's vertical-
    friction factor). Unconditionally stable + AD-safe (the underlying Thomas solve
    is differentiable).

    NOTE: Veros's EKE dissipation ``c_int = c_eps·√E/eke_len`` is folded into the
    SAME tridiagonal implicit solve (its ``b_tri`` diagonal). legoESM keeps the
    validated split treatment: the local source/sink (incl. the semi-implicit
    dissipation, :func:`legoesm.ocean.physics.lateral_mixing.eke.eke_apply_local_source`)
    is applied separately by the step; this function applies ONLY the vertical
    diffusion. The two operator-split sub-steps are each unconditionally stable.

    Parameters
    ----------
    E : (..., M) eddy kinetic energy on the W-grid (M = nlev-1 interior interfaces).
    A_v_profile : (..., M-1) or float — vertical viscosity at the interior W-grid
        interfaces (>= 0). Scaled by ``alpha_eke``. A scalar/profile is fine.
    dz_w : (..., M) or (M,) — W-grid layer thicknesses [m].
    dz_half_w : (..., M-1) or (M-1,) — spacing between adjacent W-levels [m].
    dt : float — time step [s].
    eke_cfg : EKEConfig (uses ``alpha_eke``).
    """
    from legoesm.ocean.physics.vertical_mixing import (
        implicit_vertical_diffusion_ocean,
    )
    K = eke_cfg.alpha_eke * jnp.asarray(A_v_profile)
    return implicit_vertical_diffusion_ocean(E, K, dz_w, dz_half_w, dt)


def compute_eke_step_kappa(
    T, S, eta, H_bathy, eke, grid, z_coord, cfg, *,
    eos="wright", eos_linear=None, mask=None, rho_0=_RHO_0, g=constants.g,
    omega=constants.Omega, r_earth=constants.R_earth,
    depth_resolved=False,
):
    """Prognostic GM coefficient + Eady growth rate + mixing length from the
    eddy-energy field, for the EKE-active model step. Returns ``(kappa_GM,
    sigma, L)``: ``kappa_GM`` is the override fed into the GM/Redi tracer
    tendency; ``sigma``/``L`` drive the EKE local source/sink.

    Two shapes, selected by the static Python ``depth_resolved`` flag (the
    ``EKEConfig.eke_3d`` switch in the model step):

    - ``depth_resolved=False`` (default, the 2-D path): ``eke`` is 2-D
      ``(n_lat, n_lon)`` and the return is the 2-D ``(kappa_GM, sigma_bar, L)``
      — ``kappa_GM`` the 2-D GM override, ``sigma_bar`` the depth-AVERAGED Eady
      growth ``<N|S|>_z``, ``L`` the 2-D mixing length. Bit-identical to before.
    - ``depth_resolved=True`` (the 3-D ``eke_3d`` path): ``eke`` is 3-D
      ``(n_lat, n_lon, nlev-1)`` at the interior interfaces (W-grid) and the
      return is the depth-resolved ``(kappa_GM(z), sigma(z), L(z))`` all at the
      ``nlev-1`` interfaces — the 3-D GM-skew override AND the per-level Eady
      growth / mixing length the depth-resolved EKE source/sink consumes
      (:func:`legoesm.ocean.physics.lateral_mixing.eke.eke_3d_local_tendency`).

    Recomputes rho + isopycnal slopes with the SAME shared helpers GM/Redi uses
    internally (a redundant recompute — correct; compute-once is a future
    optimization), then ``compute_eke_kappa_gm`` (which reuses the shared
    Eady-length machinery and itself dispatches on ``depth_resolved``). ``cfg``
    is the GMRediConfig (uses ``cfg.visbeck`` for the Rossby-length params and
    ``cfg.eke`` for the closure params).

    For ``cfg.eke.mixing_length_scheme == "rhines"`` the eke_len needs ``β =
    df/dy``; it is computed analytically as ``2Ω·cosφ/R`` (exact on the sphere
    where ``f = 2Ω sinφ``; equals Veros's discrete ``df/dy`` to O(dφ²)). ``Ω``/``R``
    come from the model constants (``omega``/``r_earth``; Veros-pinned in the ACC
    recipe), never literals. The ``"rossby"`` scheme ignores ``β``.
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: _neumann_fill_cgrid(field, mask)
    rho, _rp, _pp = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(
        rho, mask, z_coord, jacobian, grid, cfg,
        T=T, S=S, eos_fn=eos_fn, rho_0=rho_0, g=g,
    )
    f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
    # β = df/dy = 2Ω cosφ/R (analytic; grid.cos_lat is cosφ). Broadcast (n_lat,)
    # -> (n_lat, n_lon) to match f. Used only by the "rhines" eke_len scheme.
    beta = jnp.broadcast_to(
        (2.0 * omega * grid.cos_lat / r_earth)[:, None], mask.shape,
    )
    return compute_eke_kappa_gm(
        eke, rho, S_x, S_y, z_coord, jacobian, f_coriolis,
        cfg.visbeck, cfg.eke, rho_ref=rho_0, beta=beta,
        depth_resolved=depth_resolved,
    )
