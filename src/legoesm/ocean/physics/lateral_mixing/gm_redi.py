"""Gent-McWilliams / Redi isopycnal diffusion (small-slope formulation).

Implements the standard small-slope GM/Redi scheme following Griffies (1998):

**Redi (isopycnal diffusion):**
  Full small-slope tensor with off-diagonal slope terms:
  F_h = kappa_Redi * (dq/dx + Sx * dq/dz)     (horizontal flux, x)
  F_v = kappa_Redi * (Sx*dq/dx + Sy*dq/dy + S^2*dq/dz)  (vertical flux)

**GM (bolus transport via skew flux):**
  Skew-flux form equivalent to streamfunction psi = kappa_GM * S:
  F_h_skew = -kappa_GM * Sx * dq/dz            (horizontal skew flux)
  F_v_skew = +kappa_GM * (Sx*dq/dx + Sy*dq/dy) (vertical skew flux)

**Combined tensor (when kappa_GM == kappa_Redi == kappa):**
  Horizontal: kappa * dq/dx  (off-diagonal terms cancel)
  Vertical: 2*kappa*(Sx*dq/dx + Sy*dq/dy) + kappa*S^2*dq/dz

The scheme supports DM95 slope tapering via smooth tanh transition.

References
----------
- Gent, P. R. & McWilliams, J. C. (1990). Isopycnal mixing in ocean
  circulation models. J. Phys. Oceanogr., 20, 150-155.
- Redi, M. H. (1982). Oceanic isopycnal mixing by coordinate rotation.
  J. Phys. Oceanogr., 12, 1154-1158.
- Griffies, S. M. (1998). The Gent-McWilliams skew flux. J. Phys.
  Oceanogr., 28, 831-841.
- Danabasoglu, G. & McWilliams, J. C. (1995). Sensitivity of the global
  ocean circulation to parameterizations of mesoscale tracer transports.
  J. Climate, 8, 2967-2987.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_3d import gradient_x_3d, gradient_y_3d, divergence_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, VisbeckConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0_DEFAULT
from legoesm import constants

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _compute_tapered_slopes(
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: GMRediConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute isopycnal slopes at interfaces with DM95 tapering.

    Returns
    -------
    S_x, S_y : (6, n, n, nlev-1)
        Tapered isopycnal slopes at interfaces.
    taper : (6, n, n, nlev-1)
        Taper factor in [0, 1].
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Horizontal density gradients at full levels
    drho_dx = gradient_x_3d(rho, grid)
    drho_dy = gradient_y_3d(rho, grid)

    # Average to interfaces
    drho_dx_half = 0.5 * (drho_dx[..., :-1] + drho_dx[..., 1:])
    drho_dy_half = 0.5 * (drho_dy[..., :-1] + drho_dy[..., 1:])

    # Vertical density gradient at interfaces
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / jnp.maximum(dz_half, eps)
    # Must be negative for stable stratification
    drho_dz_safe = jnp.minimum(drho_dz, -eps)

    # Slopes: S_x = -(drho/dx) / (drho/dz)
    # Guard denominator to prevent huge slopes and NaN gradients
    S_x = jnp.clip(-drho_dx_half / drho_dz_safe, -cfg.S_max, cfg.S_max)
    S_y = jnp.clip(-drho_dy_half / drho_dz_safe, -cfg.S_max, cfg.S_max)

    # DM95 tapering: smooth taper near S_max
    S_mag = jnp.sqrt(S_x**2 + S_y**2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh((cfg.S_max - S_mag) / (0.1 * cfg.S_max + eps)))

    S_x = S_x * taper
    S_y = S_y * taper

    return S_x, S_y, taper


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

    ``κ_Visbeck(x, y) = α · L² · ⟨N · |S|⟩_z`` with the depth-average
    weighted by the local interface thickness, optionally using the
    local first-baroclinic Rossby radius as the mixing length.

    Parameters
    ----------
    rho : (..., nlev) in-situ density.
    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
    z_coord : OceanZStarCoordinate.
    jacobian : (...,) z* Jacobian at cell centres.
    f_coriolis : (...,) Coriolis parameter (same horizontal shape as
        ``jacobian``, broadcastable).
    cfg : VisbeckConfig.

    Returns
    -------
    kappa : (...,) horizontally-varying κ_GM [m²/s], clamped to the
        configured bounds.  Shape matches ``jacobian`` — i.e. one value
        per column.
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Interior interface thicknesses (used as weights for the depth-avg).
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # --- Local growth rate σ_Eady ≈ N · |S| at each interior interface ---
    # compute_buoyancy_frequency returns ``-(g/rho_ref) · drho/dz``; we
    # must pass the Boussinesq reference density ``rho_0`` so that N²
    # has the correct O(1e-5) magnitude for seawater.  Using rho_ref=1
    # overstates N² by ~10³ (Boussinesq reference density) and destroys
    # both σ and the Rossby-radius length scale.
    N2 = compute_buoyancy_frequency(
        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
    )
    N = jnp.sqrt(jnp.maximum(N2, 0.0))
    # Regularise the square-root gradient at zero slope with a value much
    # smaller than any realistic ocean slope (≥ 1e-6).  The float32 eps
    # (1.19e-7) was too large and produced an apparent |S| ≈ 3e-4 even
    # for exactly-zero inputs, which pushed κ well above ``kappa_min``.
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
    sigma = N * S_mag                                   # (..., nlev-1)

    # --- Depth-weighted average of σ_Eady ---
    w_total = jnp.sum(dz_half, axis=-1)
    sigma_bar = jnp.sum(sigma * dz_half, axis=-1) / jnp.maximum(w_total, eps)

    # --- Mixing length L ---
    if cfg.use_rossby_radius:
        # Arithmetic depth-average of N itself — NOT sqrt(<N²>), which is
        # always ≥ <N> and would over-estimate the Rossby radius for any
        # vertically varying stratification.  Using <N> matches the
        # standard WKB scale (1/H)·∫N dz used in textbook Rossby-radius
        # definitions (Chelton et al. 1998, Gill 1982).
        N_bar = jnp.sum(N * dz_half, axis=-1) / jnp.maximum(w_total, eps)
        H_col = jnp.sum(dz_actual, axis=-1)
        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
    else:
        L = jnp.full_like(sigma_bar, cfg.L_fixed)

    kappa = cfg.alpha * L ** 2 * sigma_bar
    return jnp.clip(kappa, cfg.kappa_min, cfg.kappa_max)


def _tracer_tendency_gm_redi(
    q: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    kappa_GM,
    kappa_Redi: float,
) -> jnp.ndarray:
    """Compute GM+Redi tendency for a single tracer.

    Uses the full small-slope tensor formulation:

    Horizontal flux (x-dir):
        F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * Sx * dq/dz
    Vertical flux:
        F_z = (kappa_Redi + kappa_GM) * (Sx*dq/dx + Sy*dq/dy)
              + kappa_Redi * S^2 * dq/dz

    Parameters
    ----------
    q : (6, n, n, nlev)
        Tracer field.
    S_x, S_y : (6, n, n, nlev-1)
        Tapered isopycnal slopes at interfaces.
    """
    eps = _EPS
    nlev = q.shape[-1]
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Accept ``kappa_GM`` as either a scalar or an array.  When it is a
    # per-column field (shape matching ``jacobian``) add a trailing
    # singleton so it broadcasts against the (..., nlev-1) slope and
    # flux arrays; scalar and arrays that already carry the nlev axis
    # are left unchanged.
    if isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == jacobian.ndim:
        kappa_GM_b = kappa_GM[..., jnp.newaxis]
    else:
        kappa_GM_b = kappa_GM

    # Horizontal tracer gradients at full levels
    dq_dx = gradient_x_3d(q, grid)
    dq_dy = gradient_y_3d(q, grid)

    # Vertical tracer gradient at interfaces
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    dq_dz_half = (q[..., :-1] - q[..., 1:]) / jnp.maximum(dz_half, eps)

    # Average horizontal gradients to interfaces
    dq_dx_half = 0.5 * (dq_dx[..., :-1] + dq_dx[..., 1:])
    dq_dy_half = 0.5 * (dq_dy[..., :-1] + dq_dy[..., 1:])

    # === Horizontal Redi + GM off-diagonal contribution ===
    # Full horizontal flux (Griffies 1998, small-slope):
    #   F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * Sx * dq/dz
    #   F_y = kappa_Redi * dq/dy + (kappa_Redi - kappa_GM) * Sy * dq/dz
    #
    # The diagonal part (kappa_Redi * nabla^2 q) is computed via laplacian.
    # The off-diagonal part needs: div[(kR-kG) * S * dq/dz].
    # When kappa_GM == kappa_Redi the off-diagonal vanishes identically.

    # Off-diagonal flux at interfaces: (kR - kG) * S * dq/dz
    off_diag_x = (kappa_Redi - kappa_GM_b) * S_x * dq_dz_half
    off_diag_y = (kappa_Redi - kappa_GM_b) * S_y * dq_dz_half

    # Average interface values to full levels (pad boundaries with zero).
    # ``jnp.pad`` lowers to one Pad HLO op per pad and avoids the
    # alloc-zeros + concatenate pair (2 HLO ops each).
    pad_axes = ((0, 0),) * (off_diag_x.ndim - 1)
    off_diag_x_full = 0.5 * (
        jnp.pad(off_diag_x, (*pad_axes, (1, 0)))
        + jnp.pad(off_diag_x, (*pad_axes, (0, 1)))
    )
    off_diag_y_full = 0.5 * (
        jnp.pad(off_diag_y, (*pad_axes, (1, 0)))
        + jnp.pad(off_diag_y, (*pad_axes, (0, 1)))
    )

    # Diagonal: kappa_Redi * nabla^2(q)
    dq_h = laplacian_viscosity_3d(q, grid, kappa_Redi)

    # Off-diagonal: div[(kR-kG) * S * dq/dz]
    dq_h = dq_h + divergence_3d(off_diag_x_full, off_diag_y_full, grid)

    # === Vertical flux ===
    # F_z at interfaces = (kR + kG) * (Sx*dq/dx + Sy*dq/dy) + kR * S^2 * dq/dz
    S2_half = S_x**2 + S_y**2
    F_z = ((kappa_Redi + kappa_GM_b) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi * S2_half * dq_dz_half)

    # Vertical flux divergence at full levels: dF_z/dz
    # dq/dt_vert[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]
    # with F_z = 0 at surface and bottom boundaries.  Single Pad HLO op
    # replaces the alloc-zeros + concatenate-of-three.
    pad_axes_v = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes_v, (1, 1)))
    dq_vert = (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)

    return dq_h + dq_vert


def gm_redi_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: GMRediConfig,
) -> LateralMixingOutput:
    """Apply GM-Redi lateral mixing with full small-slope tensor.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    grid : CubedSphereGrid
    cfg : GMRediConfig

    Returns
    -------
    LateralMixingOutput
    """
    # Compute tapered isopycnal slopes at interfaces
    S_x, S_y, taper = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)

    # GM coefficient: scalar from config, or Visbeck-adaptive field.
    if cfg.visbeck.enabled:
        f_coriolis = jnp.asarray(grid.grid_coriolis)
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Tracer tendencies with full GM+Redi tensor
    dT_dt = _tracer_tendency_gm_redi(
        T, S_x, S_y, z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi
    )
    dS_dt = _tracer_tendency_gm_redi(
        S, S_x, S_y, z_coord, jacobian, grid, kappa_GM, cfg.kappa_Redi
    )

    # GM/Redi does not produce momentum tendencies
    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v)

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
