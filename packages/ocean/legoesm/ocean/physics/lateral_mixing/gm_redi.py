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
from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d

__physics_contract__ = {
    "summary": (
        "Gent-McWilliams / Redi isopycnal mixing (Griffies 1998 small-slope "
        "skew-flux tensor) on the cubed sphere: adiabatic eddy (bolus) "
        "advection + isoneutral diffusion of T and S, with DM95 slope tapering "
        "and optional Visbeck adaptive kappa."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu", "rho": "kg/m^3",
        "jacobian": "1 (z-star dimensionless)",
        "cfg.kappa_GM": "m^2/s", "cfg.kappa_Redi": "m^2/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
    },
    "sign_convention": (
        "kappa_GM, kappa_Redi >= 0; the GM skew flux is adiabatic "
        "(streamfunction psi = kappa_GM*S along isopycnals, flattening slopes) "
        "and Redi diffuses down-gradient ALONG isopycnals; both are "
        "flux-divergence form so the volume integral of T and S is conserved "
        "(adiabatic redistribution; no momentum tendency, du_dt=dv_dt=0); slopes "
        "tapered (DM95) where steep; z positive up."
    ),
    # Adiabatic tracer redistribution: conserves volume-integrated tracer.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Gent & McWilliams (1990) JPO 20, 150-155; Redi (1982) JPO 12, "
        "1154-1158; Griffies (1998) JPO 28, 831-841; Danabasoglu & McWilliams "
        "(1995) J. Climate 8, 2967-2987"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_visbeck_gm.py + "
        "tests/ocean/unit/test_gm_redi_eady_physics.py — GM flattens an "
        "isopycnal slope (releasing APE) while conserving volume-integrated T, "
        "S; flat isopycnals give zero tendency."
    ),
}


def _pad_for_gradient(field, grid):
    """Single-call pad for gradient_x_3d/gradient_y_3d sharing.

    Both gradient operators accept ``padded=`` to skip their internal
    halo exchange.  Pre-padding here lets paired (∂/∂x, ∂/∂y) calls on
    the same input issue ONE ``pad_halo_4d`` MPI exchange instead of
    two.
    """
    dg = getattr(grid, 'duogrid', None)
    offsets = None if dg is not None else grid.halo_interp_offsets
    return _pad_halo_4d(field, interp_offsets=offsets, duogrid=dg)
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS,
    compute_visbeck_kappa_gm,
    dm95_taper,
    gm_resolution_scaled_kappa,
    validate_adjoint_stabilization,
    vertical_flux_divergence,
)
from legoesm.ocean.vertical import OceanZStarCoordinate


# GMRediConfig fields actually honored by the cubed-sphere leaf. Every other
# scheme-altering field is silently inert on this path (the lat-lon C-grid
# implementation honors them via fail-fast dispatch in gm_redi_latlon_cgrid).
# Cross-checked against the `cfg.<field>` reads in this module:
#   kappa_GM, kappa_Redi, S_max, taper_width_frac, visbeck (all paths) and
#   adjoint_stabilization (the stop_gradient knob in _compute_tapered_slopes).
_CUBED_SPHERE_SUPPORTED_FIELDS = frozenset({
    "kappa_GM",
    "kappa_Redi",
    "S_max",
    "taper_width_frac",
    "visbeck",
    "adjoint_stabilization",
    # Hallberg (2013) resolution taper -- honored on the cube (wired below),
    # so it is a supported (not silently-inert) field.
    "resolution_function",
    "resfn_gamma",
    "resfn_cbcl_ms",
})


def validate_cubed_sphere_gm_redi_config(cfg: GMRediConfig) -> None:
    """Fail fast if a cubed-sphere-UNSUPPORTED GMRediConfig field is set.

    The cubed-sphere GM/Redi leaf implements only the centered small-slope
    tensor with DM95 tapering and the (Visbeck-adaptive) ``kappa_GM``.  The
    Veros-faithful triad options (``slope_scheme``, ``slope_density``,
    ``implicit_K33``, ``K_iso_steep``, ``veros_triad_weights``,
    ``double_redi_diagonal``) and prognostic EKE (``eke``) are only available
    on the lat-lon C-grid path.

    The Ferrari (2008) surface complement (``surface_complement``,
    ``surface_complement_depth``) is INHERENT on the cubed-sphere path, not
    absent.  The lat-lon tapered tensor multiplies the Redi DIAGONAL by the DM95
    taper, which vanishes in the mixed layer, so it needs the explicit
    depth-masked complement to restore ``kappa_Redi`` horizontal mixing there.
    The cubed-sphere diagonal (``laplacian_viscosity_3d(q, grid, kappa_Redi)``
    in ``_tracer_tendency_gm_redi``) is NEVER tapered — it applies
    ``kappa_Redi * nabla^2 q`` at every level, mixed layer included — so the
    total boundary-layer horizontal diffusivity is already ``kappa_Redi`` by
    construction.  The default ``surface_complement=True`` is therefore honored;
    a non-default value (``False``, or a changed ``surface_complement_depth``)
    is rejected below because the cube has no depth-masked TOGGLE to match — its
    diagonal mixing is structurally always-on.

    To avoid a silent no-op, this raises ``ValueError`` for any unsupported
    field whose value differs from its ``GMRediConfig`` default.  A default
    config (and any config that touches only supported fields) is unaffected.

    The comparison is on the static Python config (mirrors the fail-fast
    ``slope_scheme`` / triad-option dispatch in ``gm_redi_latlon_cgrid``);
    it adds no traced ops and is safe under JIT/grad.
    """
    defaults = GMRediConfig()
    offending = [
        name
        for name in cfg._fields
        if name not in _CUBED_SPHERE_SUPPORTED_FIELDS
        and getattr(cfg, name) != getattr(defaults, name)
    ]
    if offending:
        raise ValueError(
            "The cubed-sphere GM/Redi path does not support these "
            f"GMRediConfig field(s): {', '.join(sorted(offending))}. "
            "They are only honored by the lat-lon C-grid path "
            "(gm_redi_latlon_cgrid); use that path or leave these fields "
            "at their GMRediConfig defaults."
        )


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
    eps = EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Horizontal density gradients at full levels.
    # Pre-pad once so both gradients share the halo MPI exchange.
    rho_pad = _pad_for_gradient(rho, grid)
    drho_dx = gradient_x_3d(rho, grid, padded=rho_pad)
    drho_dy = gradient_y_3d(rho, grid, padded=rho_pad)

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

    # Adjoint stabilization (primal-invisible; see _gm_redi_common note and
    # the GMRediConfig field doc). Static Python gating on the config literal.
    adj_stab = getattr(cfg, "adjoint_stabilization", "none")
    validate_adjoint_stabilization(adj_stab)
    if adj_stab == "stop_gradient_slopes":
        S_x = jax.lax.stop_gradient(S_x)
        S_y = jax.lax.stop_gradient(S_y)

    # DM95 tapering: smooth taper near S_max. ``taper_width_frac`` maps
    # to Veros's ``iso_dslope / iso_slopec``.
    return dm95_taper(
        S_x, S_y, cfg.S_max, eps, cfg.taper_width_frac,
        stop_gradient_taper=(adj_stab == "stop_gradient_taper"),
    )


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
    eps = EPS
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

    # Horizontal tracer gradients at full levels.
    # Pre-pad once so both gradients share the halo MPI exchange.
    q_pad = _pad_for_gradient(q, grid)
    dq_dx = gradient_x_3d(q, grid, padded=q_pad)
    dq_dy = gradient_y_3d(q, grid, padded=q_pad)

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

    # Diagonal: kappa_Redi * nabla^2(q). UNTAPERED (constant kappa_Redi at every
    # level) — this is the Ferrari (2008) surface complement made inherent: the
    # mixed-layer horizontal diffusivity is always kappa_Redi, so no separate
    # depth-masked complement term is needed (unlike the lat-lon tapered tensor).
    dq_h = laplacian_viscosity_3d(q, grid, kappa_Redi)

    # Off-diagonal: div[(kR-kG) * S * dq/dz]
    dq_h = dq_h + divergence_3d(off_diag_x_full, off_diag_y_full, grid)

    # === Vertical flux ===
    # F_z at interfaces = (kR + kG) * (Sx*dq/dx + Sy*dq/dy) + kR * S^2 * dq/dz
    S2_half = S_x**2 + S_y**2
    F_z = ((kappa_Redi + kappa_GM_b) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi * S2_half * dq_dz_half)

    # Vertical flux divergence at full levels (uses jnp.pad inside the
    # shared helper for a single Pad HLO op vs alloc-zeros + concatenate).
    dq_vert = vertical_flux_divergence(F_z, dz_actual, eps)

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
    # Fail fast on cubed-sphere-unsupported config (static Python check on the
    # config literal; no traced ops). Mirrors the lat-lon path's fail-fast
    # dispatch so a user-set unsupported field is never silently ignored.
    validate_cubed_sphere_gm_redi_config(cfg)

    # Compute tapered isopycnal slopes at interfaces
    S_x, S_y, taper = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)

    # GM coefficient: scalar from config, or Visbeck-adaptive field.
    if getattr(cfg, "treguier", None) is not None and cfg.treguier.enabled:
        raise NotImplementedError(
            "GMRediConfig.treguier (NEMO nn_aei_ijk_t=21 adaptive kappa) is "
            "implemented on the lat-lon C-grid path only; the cubed-sphere "
            "GM/Redi would silently fall back to constant kappa. Use "
            "visbeck or constant kappa_GM here.")
    if cfg.visbeck.enabled:
        f_coriolis = jnp.asarray(grid.grid_coriolis)
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Hallberg (2013) resolution taper of the GM coefficient (default off =>
    # byte-identical). Static Python gate on the config bool. GM-only: the Redi
    # diffusivity cfg.kappa_Redi is left unscaled. dx = sqrt(cell area).
    if getattr(cfg, "resolution_function", False):
        f_coriolis = jnp.asarray(grid.grid_coriolis)
        kappa_GM = gm_resolution_scaled_kappa(
            kappa_GM, f_coriolis, jnp.sqrt(jnp.asarray(grid.grid_area)),
            cfg.resfn_gamma, cfg.resfn_cbcl_ms,
        )

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
