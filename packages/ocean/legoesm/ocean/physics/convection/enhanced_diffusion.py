"""Enhanced diffusion for convective adjustment.

Applies large vertical diffusivity where N^2 < 0 (statically unstable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing._shared import vmap_vertical_diffusion
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def convective_K_A_flag(
    rho: jnp.ndarray,
    dz_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
):
    """Convective tracer diffusivity, momentum viscosity, and flag.

    Single source of truth for the enhanced-diffusion (Oceananigans
    ``ConvectiveAdjustmentVerticalDiffusivity``) coefficients, shared by
    the explicit kernel ``enhanced_diffusion_convection`` and the implicit
    fallback ``compute_vertical_K_profiles`` so the two paths are bit-for-
    bit identical (no duplicated numerics).

    AD safety on dry / land columns: ``compute_buoyancy_frequency`` divides
    by the interface thickness, which is **zero** when ``jacobian == 0``.
    Masking ``N²`` *after* that division still leaves a ``num / 0`` node in
    the graph, and reverse-mode then evaluates ``0 · ∞ = NaN`` for any
    control that ``rho`` depends on.  We therefore substitute *safe* inputs
    **before** the division — a unit jacobian and a reference density on
    dry columns — so both forward and backward stay finite, then zero the
    diffusivity / viscosity on dry interfaces (no mixing through land).

    Returns
    -------
    (K_v, A_v, flag) : arrays at interior interfaces, shape
        ``rho.shape[:-1] + (nlev - 1,)``.  ``K_v`` is the tracer
        diffusivity (``convective_κz``), ``A_v`` the momentum viscosity
        (``convective_νz``), ``flag`` the convective activity in [0, 1].
    """
    dz_actual = dz_ref * jacobian[..., jnp.newaxis]               # (..., nlev)
    dry_iface = (dz_actual[..., :-1] <= 0.0) | (dz_actual[..., 1:] <= 0.0)
    dry_col = jacobian <= 0.0                                     # (...,)

    # Substitute safe inputs BEFORE the N² interior division (AD-safe).
    jacobian_safe = jnp.where(dry_col, 1.0, jacobian)
    rho_safe = jnp.where(dry_col[..., jnp.newaxis], rho_0, rho)
    N2 = compute_buoyancy_frequency(rho_safe, dz_ref, jacobian_safe)

    if cfg.smooth_transition:
        sig = jax.nn.sigmoid(-N2 * cfg.sigmoid_sharpness)
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * sig
        A = cfg.nu_bg + (cfg.nu_conv - cfg.nu_bg) * sig
        flag = sig
    else:
        K = jnp.where(N2 < 0.0, cfg.K_conv, cfg.K_bg)
        A = jnp.where(N2 < 0.0, cfg.nu_conv, cfg.nu_bg)
        flag = jnp.where(N2 < 0.0, 1.0, 0.0)

    # Zero mixing + flag on dry interfaces (no mixing through land).
    K = jnp.where(dry_iface, 0.0, K)
    A = jnp.where(dry_iface, 0.0, A)
    flag = jnp.where(dry_iface, 0.0, flag)
    return K, A, flag


def enhanced_diffusion_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
    apply_diffusion: bool = True,
    dt: float | None = None,
    u: jnp.ndarray | None = None,
    v: jnp.ndarray | None = None,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    Oceananigans ``ConvectiveAdjustmentVerticalDiffusivity``: large
    vertical **tracer diffusivity** ``K_conv`` (and **momentum
    viscosity** ``nu_conv``) wherever ``N² < 0``, background values
    elsewhere.  Tracer and momentum coefficients are independent.

    When ``dt`` is supplied, the explicit-branch CFL cap uses the
    actual physics step instead of ``cfg.cfl_dt_estimate`` — preserves
    stability and correct convection strength when runtime ``dt``
    differs from the config-time estimate.  Codex iter-3 finding #7.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig
    apply_diffusion : bool
        Run the explicit diffusion branch.  When False, K_v / A_v are
        returned in the output for downstream implicit solvers.
    dt : float, optional
        Physics step [s].  When provided, used in the CFL cap instead
        of ``cfg.cfl_dt_estimate``.
    u, v : array (6, n, n, nlev), optional
        Velocity components.  Required for the explicit momentum-mixing
        branch (``apply_diffusion=True``); when ``None`` the explicit
        momentum tendencies are zero (the implicit path re-derives ``A_v``
        in ``compute_vertical_K_profiles`` and applies it via the
        unconditionally-stable backward-Euler solve).

    Returns
    -------
    OceanConvectionOutput
    """
    # Tracer diffusivity (convective_κz), momentum viscosity (convective_νz)
    # and convective flag at interfaces — shared, AD-safe on dry columns.
    K, A, flag = convective_K_A_flag(rho, z_coord.dz_ref, jacobian, cfg)

    # CFL safety cap on the EXPLICIT branch.  Backward-Euler (implicit)
    # mixing is unconditionally stable; explicit-Euler vertical
    # diffusion requires ``K · dt / dz² ≤ 1/2``.  Without the cap a
    # ``K_conv = 1 m²/s`` over ``dz = 10 m`` would need ``dt ≤ 50 s`` —
    # 70× tighter than typical ocean physics steps.  See
    # ``EnhancedDiffusionConfig`` for ``cfl_dt_estimate`` and
    # ``cfl_safety``.
    #
    # The earlier coarse cap used the reference ``dz_ref`` alone and
    # ignored ``jacobian`` (z-star contracts layers when ``eta + H``
    # shrinks): a column with jacobian = 0.5 has dz_actual half of
    # dz_ref, so K must be 4× tighter.  Delegating the per-interface
    # cap to ``vertical_diffusion_variable_K(..., dt=dt_eff)`` uses the
    # actual layer thicknesses ``dz_ref · jacobian`` and the per-
    # interface dz_min that the leaf already computes.
    du_dt = None
    dv_dt = None
    if apply_diffusion:
        dt_eff = cfg.cfl_dt_estimate if dt is None else dt
        diffuse = lambda q, c: vertical_diffusion_variable_K(
            q, z_coord, jacobian, c, dt=dt_eff, cfl_safety=cfg.cfl_safety,
        )
        if u is not None and v is not None:
            # Mix momentum (A) and tracers (K) through the shared kernel
            # — same operator/CFL cap the Richardson scheme uses.
            vel_tend, tr_tend = vmap_vertical_diffusion(
                u, v, T, S, A, K, diffuse, apply_diffusion=True,
            )
            du_dt = vel_tend[0]
            dv_dt = vel_tend[1]
        else:
            # Fail closed: a tracer-only explicit branch with a nonzero
            # convective momentum viscosity would silently drop the requested
            # momentum mixing.  Tracer-only is legal only for nu_*=0 (the MPAS
            # path, which guards nu_*=0 at its factory before calling here).
            if cfg.nu_conv != 0.0 or cfg.nu_bg != 0.0:
                raise ValueError(
                    "enhanced_diffusion_convection: nu_conv/nu_bg is nonzero "
                    "but u/v were not supplied, so the explicit convective "
                    "momentum tendency cannot be computed and would be "
                    "silently dropped. Pass cell-centred u and v, or set "
                    "nu_conv=nu_bg=0.0 for tracer-only convective adjustment."
                )
            # Velocity not supplied + nu_*=0: tracer-only explicit branch.
            tr_tend = jax.vmap(
                lambda q: diffuse(q, K), in_axes=0, out_axes=0,
            )(jnp.stack([T, S], axis=0))
        dT_dt = tr_tend[0]
        dS_dt = tr_tend[1]
    else:
        dT_dt = jnp.zeros_like(T)
        dS_dt = jnp.zeros_like(S)

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
        K_v=K,
        du_dt=du_dt,
        dv_dt=dv_dt,
        A_v=A,
    )
