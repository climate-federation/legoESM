"""Grid-agnostic helpers shared by cubed-sphere and lat-lon GM/Redi.

Functions in this module operate on ``(..., nlev)`` arrays and make no
reference to a specific grid type.  Both ``gm_redi.py`` (cubed-sphere)
and ``gm_redi_latlon_cgrid.py`` (lat-lon C-grid) import from here.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0_DEFAULT
from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


# ---------------------------------------------------------------------------
# DM95 slope tapering
# ---------------------------------------------------------------------------

def dm95_taper(
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
    transition_width_frac: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Danabasoglu & McWilliams (1995) smooth slope tapering.

    Returns tapered ``(S_x, S_y, taper)`` where *taper* is a smooth
    factor in [0, 1] computed as::

        taper = 0.5 * (1 + tanh((S_max - |S|) / (width_frac * S_max)))

    Parameters
    ----------
    S_x, S_y : array (..., nlev-1)
        Raw (clipped but un-tapered) isopycnal slopes at interfaces.
    S_max : float
        Slope at which the taper crosses 0.5. Equivalent to Veros's
        ``iso_slopec``.
    eps : float
        Small constant for sqrt regularisation.
    transition_width_frac : float
        Tanh transition half-width as a fraction of ``S_max``.
        Default ``0.1`` matches the legoESM pre-2026 convention.
        Veros's ``iso_dslope`` parameter maps via
        ``transition_width_frac = iso_dslope / iso_slopec``.

    Returns
    -------
    S_x_tapered, S_y_tapered, taper : same shapes as inputs.
    """
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - S_mag) / (transition_width_frac * S_max + eps)
    ))
    return S_x * taper, S_y * taper, taper


def dm95_taper_scalar(
    S: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
    transition_width_frac: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-component variant of :func:`dm95_taper`.

    Used by grids that carry a scalar slope along each face's own normal
    (e.g. MPAS/Voronoi edges).  Identical functional form, with ``|S|``
    replaced by ``|S_n|``. See :func:`dm95_taper` for parameter
    semantics (in particular ``transition_width_frac`` ↔ Veros's
    ``iso_dslope / iso_slopec``).

    Returns
    -------
    S_tapered, taper : same shape as ``S``.
    """
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (transition_width_frac * S_max + eps)
    ))
    return S * taper, taper


# ---------------------------------------------------------------------------
# Vertical flux divergence with zero-flux BCs
# ---------------------------------------------------------------------------

def vertical_flux_divergence(
    F_z: jnp.ndarray,
    dz_actual: jnp.ndarray,
    eps: float = _EPS,
) -> jnp.ndarray:
    """Compute vertical flux divergence at full levels.

    ``dq/dt[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]``

    with F_z = 0 at the surface and bottom boundaries (zero-flux BCs).

    Parameters
    ----------
    F_z : array (..., nlev-1)
        Vertical flux at interior interfaces.
    dz_actual : array (..., nlev)
        Layer thicknesses at full levels.

    Returns
    -------
    tendency : array (..., nlev)
    """
    # Single Pad HLO op (replaces alloc-zeros + concatenate of three).
    pad_axes = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)


# ---------------------------------------------------------------------------
# Visbeck (1997) adaptive GM coefficient
# ---------------------------------------------------------------------------

def compute_visbeck_kappa_gm(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg: VisbeckConfig,
    rho_ref: float = _RHO_0_DEFAULT,
) -> jnp.ndarray:
    """Visbeck (1997) adaptive GM coefficient.

    ``kappa(x, y) = alpha * L^2 * <N * |S|>_z`` with the depth-average
    weighted by the local interface thickness, optionally using the
    local first-baroclinic Rossby radius as the mixing length.

    Parameters
    ----------
    rho : (..., nlev) in-situ density.
    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
    z_coord : OceanZStarCoordinate.
    jacobian : (...,) z* Jacobian at cell centres.
    f_coriolis : (...,) Coriolis parameter.
    cfg : VisbeckConfig.
    rho_ref : float
        Boussinesq reference density [kg/m^3].

    Returns
    -------
    kappa : (...,) horizontally-varying kappa_GM [m^2/s], clamped to
        the configured bounds.
    """
    sigma_bar, L, wet_col, _int_N_dz = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg, rho_ref,
    )
    # Apply the wet-column mask AFTER clipping — otherwise dry columns
    # get lifted to ``kappa_min`` rather than 0 (Codex review caught
    # this).  A dry column should contribute exactly zero diffusivity
    # so it cannot leak gradients through the GM/Redi tendencies.
    kappa = jnp.clip(cfg.alpha * L ** 2 * sigma_bar, cfg.kappa_min, cfg.kappa_max)
    return jnp.where(wet_col, kappa, 0.0)


def _eady_growth_and_length(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg,
    rho_ref: float = _RHO_0_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Depth-averaged Eady growth rate ``sigma_bar = <N|S|>_z`` and mixing length
    ``L`` (first-baroclinic Rossby radius, or ``cfg.L_fixed``), the wet-column mask,
    and the column buoyancy integral ``int_N_dz = ∫N dz`` [m/s]. Shared by the
    Visbeck diagnostic ``kappa_GM`` (``alpha·L²·sigma_bar``) and the prognostic-EKE
    closure (``kappa_GM = c_k·L·√E``, production ``∝ sigma_bar²``, and — for the
    ``"rhines"`` eke_len — the deformation radius ``c1=int_N_dz/π``) so the
    N²/slope/length numerics live in ONE place. ``cfg`` is a VisbeckConfig (uses
    ``L_min``, ``L_max``, ``f_min``, ``use_rossby_radius``, ``L_fixed``).

    Returns ``(sigma_bar, L, wet_col, int_N_dz)``.
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # Local growth rate sigma_Eady ~ N * |S| at each interior interface.
    N2 = compute_buoyancy_frequency(
        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
    )
    # Use a small positive floor on N² before sqrt, NOT a hard zero.
    # ``sqrt(0)`` has an infinite gradient in JAX; combined with the
    # ``maximum(N²,0)`` mask whose gradient is zero on the unstable
    # side, the backward pass evaluates ``inf * 0`` and produces NaN.
    # ``maximum(N², 1e-30)`` keeps N tiny but positive in unstable
    # layers, so ``sqrt`` has a finite (but very large) derivative
    # which is then multiplied by zero from ``maximum``'s VJP — a
    # well-defined zero rather than NaN.  Forward effect is at most
    # ``sqrt(1e-30) ≈ 1e-15``, negligible.
    N = jnp.sqrt(jnp.maximum(N2, 1e-30))
    # Regularise sqrt at zero slope — 1e-30 avoids spurious |S| ~ 3e-4
    # that the float32 eps (~1.19e-7) would produce.
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
    sigma = N * S_mag

    # Fused column reduction of ``ones``, ``sigma`` and ``N`` (shared ``dz_half``
    # weight). ``int_N_dz = Σ N·dz_half = ∫N dz`` [m/s] is the column buoyancy
    # integral (= Veros's ``C_rossby·π``); it is returned so the prognostic-EKE
    # eke_len deformation radius (``c1 = int_N_dz/π``) reuses the shared N WITHOUT
    # re-deriving N²/N anywhere (no duplicate numerics). The N reduction is shared by
    # both length modes; the ``L_fixed`` branch ignores ``N_bar`` but the fused sum
    # is identical, so ``sigma_bar``/``L``/``wet_col`` stay bit-identical to the
    # pre-int_N_dz code.
    #
    # Dry-column safeguard: when ``dz_half`` is all zero (jacobian = 0 over land /
    # dry cells), ``w_total = 0`` and N² = 0/0 = NaN, so EVERY column-reduced
    # quantity is explicitly zeroed on dry columns via ``wet_col`` — ``sigma_bar``,
    # ``N_bar`` AND ``int_N_dz`` alike — so the wet mask propagates cleanly through
    # forward values and gradients. (A raw NaN ``int_N_dz`` would otherwise poison
    # the "rhines" eke_len + its VJP on any config with true land; bit-identical on
    # wet columns, where ``wet_col`` is True.)
    _stack = jnp.stack([jnp.ones_like(sigma), sigma, N], axis=-1)
    _col = jnp.sum(_stack * dz_half[..., None], axis=-2)
    w_total = _col[..., 0]
    w_safe = jnp.maximum(w_total, eps)
    wet_col = w_total > eps
    sigma_bar = jnp.where(wet_col, _col[..., 1] / w_safe, 0.0)
    int_N_dz = jnp.where(wet_col, _col[..., 2], 0.0)  # ∫N dz [m/s]; masked like sigma_bar
    if cfg.use_rossby_radius:
        N_bar = jnp.where(wet_col, int_N_dz / w_safe, 0.0)
        H_col = jnp.sum(dz_actual, axis=-1)
        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
    else:
        L = jnp.full_like(sigma_bar, cfg.L_fixed)

    return sigma_bar, L, wet_col, int_N_dz


def compute_eke_kappa_gm(
    E: jnp.ndarray,
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    visbeck_cfg,
    eke_cfg,
    rho_ref: float = _RHO_0_DEFAULT,
    *,
    beta: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prognostic GM coefficient from the eddy-energy field ``E`` (Eden-Greatbatch).

    Reuses the SHARED Eady-rate/Rossby-length machinery (``_eady_growth_and_length``,
    using ``visbeck_cfg`` for the length params) and the EKE closure. Returns
    ``(kappa_GM, sigma_bar, L)``: ``kappa_GM = c_k·L·√E`` (2-D, masked to wet
    columns) for the GM/Redi tendency, and ``sigma_bar`` (depth-averaged Eady growth
    rate) + ``L`` (mixing length) for the EKE local source/sink
    (``eke_local_tendency``). Pure.

    The mixing length ``L`` follows ``eke_cfg.mixing_length_scheme``:

    - ``"rossby"`` (default) — ``L = max(L_rossby, l_min)``, the Visbeck
      first-baroclinic length (legoESM's pre-eke_len behaviour). ``beta`` unused.
    - ``"rhines"`` — Veros ``eke_len = max(l_min, min(eke_cross·L_rossby,
      eke_crhin·L_rhines))`` from the deformation radius ``c1=int_N_dz/π`` and the
      eddy-energy Rhines scale ``√(√E/β)``. Requires ``beta`` (df/dy); raises
      otherwise.

    Dispatch is on the static ``mixing_length_scheme`` Python string, so the
    ``if/elif/else`` runs at trace time (not on a traced value).
    """
    from legoesm.ocean.physics.lateral_mixing.eke import (
        eke_deformation_radius, eke_kappa_gm, eke_len_composite,
        eke_mixing_length, eke_rhines_length,
    )

    sigma_bar, L_rossby, wet_col, int_N_dz = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis, visbeck_cfg, rho_ref,
    )
    scheme = eke_cfg.mixing_length_scheme
    if scheme == "rossby":
        L = eke_mixing_length(L_rossby, eke_cfg)
    elif scheme == "rhines":
        if beta is None:
            raise ValueError(
                "compute_eke_kappa_gm: mixing_length_scheme='rhines' requires "
                "`beta` (df/dy [1/(m·s)]); none was passed."
            )
        L = eke_len_composite(
            eke_deformation_radius(int_N_dz, f_coriolis, beta, eke_cfg),
            eke_rhines_length(E, beta, eke_cfg),
            eke_cfg,
        )
    else:
        raise ValueError(
            "EKEConfig.mixing_length_scheme must be 'rossby' or 'rhines', got "
            f"{scheme!r}"
        )
    kappa = eke_kappa_gm(E, L, eke_cfg)
    return jnp.where(wet_col, kappa, 0.0), sigma_bar, L
