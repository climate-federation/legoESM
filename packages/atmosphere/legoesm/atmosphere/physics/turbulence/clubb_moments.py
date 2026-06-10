"""CLUBB prognostic moment advances (CAM-default tree).

Ports the implicit moment-advance modules from CLUBB-JAX for the CAM-default
flags. Each advance = LHS band assembly (Crank-Nicholson diffusion + 1/dt +
mean advection + boundary terms) → solve (tridiag/penta, :mod:`clubb_solve`) →
clipping. This file starts with the simplest: ``advance_windm_edsclrm`` (the
u/v wind advance via eddy diffusion, the CAM ``l_predict_upwp_vpwp = .false.``
path); the wp2/wp3, xp2/xpyp and xm/wpxp advances follow.

CAM-default branches taken here: ``l_predict_upwp_vpwp = .false.`` (u/v advanced
by eddy diffusion, not the prognostic upwp/vpwp path), ``l_upwind_xm_ma =
.true.`` (upwind mean advection), ``l_imp_sfc_momentum_flux = .true.`` (implicit
surface momentum flux). The centered mean-advection branch (``l_upwind_xm_ma =
.false.``, needs ``weights_zt2zm``) is out of the CAM-default tree and not ported.

These kernels use no physical constants (only the tunable ``c_K10``/``nu10``
params and the ``max_mag_correlation = 0.99`` numerical bound) → bit-exact to
the reference. All arrays on the ascending CLUBB grid; ``zt`` = thermodynamic
(nzt), ``zm`` = momentum (nzm = nzt + 1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid
from legoesm.atmosphere.physics.turbulence.clubb_solve import tridiag_solve

_EPS = 1.0e-10
_MAX_MAG_CORRELATION = 0.99   # Cauchy-Schwarz correlation bound (constants_clubb)


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x,0))`` with a finite (0) gradient at ``x<=0`` (double-where)."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


# ---------------------------------------------------------------------------
# LHS band builders
# ---------------------------------------------------------------------------

def diffusion_zt_lhs(K_zm, nu, invrs_rho_ds_zt, rho_ds_zm, gr: CLUBBGrid):
    """Tridiagonal LHS for implicit eddy diffusion of a zt-level variable.

    Faithful port of ``diffusion.F90:diffusion_zt_lhs`` (non-upwind path):
    discretizes ``d/dz[(K_zm + nu) d(var_zt)/dz]`` at zt levels with zero-flux
    boundaries. Returns ``(3, ngrdcol, nzt)`` = ``[super, main, sub]``.
    """
    K_zm_nu = K_zm + nu[:, None]
    invrs_dzt = gr.invrs_dzt
    invrs_dzm = gr.invrs_dzm

    common_bot = (invrs_dzt[:, :1] * invrs_rho_ds_zt[:, :1]
                  * K_zm_nu[:, 1:2] * rho_ds_zm[:, 1:2] * invrs_dzm[:, 1:2])
    super_bot, main_bot, sub_bot = -common_bot, common_bot, jnp.zeros_like(common_bot)

    scale_int = invrs_dzt[:, 1:-1] * invrs_rho_ds_zt[:, 1:-1]
    super_int = -scale_int * K_zm_nu[:, 2:-1] * rho_ds_zm[:, 2:-1] * invrs_dzm[:, 2:-1]
    sub_int = -scale_int * K_zm_nu[:, 1:-2] * rho_ds_zm[:, 1:-2] * invrs_dzm[:, 1:-2]
    main_int = -(super_int + sub_int)

    common_top = (invrs_dzt[:, -1:] * invrs_rho_ds_zt[:, -1:]
                  * K_zm_nu[:, -2:-1] * rho_ds_zm[:, -2:-1] * invrs_dzm[:, -2:-1])
    super_top, sub_top, main_top = jnp.zeros_like(common_top), -common_top, common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag = jnp.concatenate([main_bot, main_int, main_top], axis=1)
    subdiag = jnp.concatenate([sub_bot, sub_int, sub_top], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


def term_ma_zt_lhs_upwind(wm_zt, gr: CLUBBGrid):
    """Upwind mean-advection LHS for a zt-level variable (CAM ``l_upwind_xm_ma``).

    Faithful port of the upwind branch of ``mean_adv.F90:term_ma_zt_lhs``
    (ascending grid). ``(3, ngrdcol, nzt)`` = ``[super, main, sub]``. The
    centered branch is out of the CAM-default tree and not ported.
    """
    ngrdcol, nzt = wm_zt.shape
    invrs_dzm = gr.invrs_dzm

    wm_int = wm_zt[:, 1:-1]
    idzm_k = invrs_dzm[:, 1:-2]
    idzm_kp1 = invrs_dzm[:, 2:-1]
    mask = wm_int >= 0.0
    sup_int = jnp.where(mask, 0.0, wm_int * idzm_kp1)
    mid_int = jnp.where(mask, wm_int * idzm_k, -wm_int * idzm_kp1)
    sub_int = jnp.where(mask, -wm_int * idzm_k, 0.0)

    # Lower boundary k=0 (Fortran k=1): uses invrs_dzm[1]; the upward-wind
    # contribution is dropped (zero-flux from below), matching mean_adv.F90.
    wm0 = wm_zt[:, 0]
    m0 = wm0 >= 0.0
    idzm_2 = invrs_dzm[:, 1]
    sup0 = jnp.where(m0, 0.0, wm0 * idzm_2)
    mid0 = jnp.where(m0, 0.0, -wm0 * idzm_2)
    sub0 = jnp.zeros_like(wm0)

    # Upper boundary k=nzt-1 (Fortran k=nzt): uses invrs_dzm[nzm-2]=invrs_dzm[nzt-1];
    # the downward-wind contribution is dropped (zero-flux from above).
    wmt = wm_zt[:, -1]
    mt = wmt >= 0.0
    idzm_top = invrs_dzm[:, nzt - 1]
    supt = jnp.zeros_like(wmt)
    midt = jnp.where(mt, wmt * idzm_top, 0.0)
    subt = jnp.where(mt, -wmt * idzm_top, 0.0)

    sup = jnp.concatenate([sup0[:, None], sup_int, supt[:, None]], axis=1)
    mid = jnp.concatenate([mid0[:, None], mid_int, midt[:, None]], axis=1)
    sub = jnp.concatenate([sub0[:, None], sub_int, subt[:, None]], axis=1)
    return jnp.stack([sup, mid, sub], axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calc_xpwp(Km_zm, xm, invrs_dzm):
    """Down-gradient eddy flux ``x'w'`` on momentum levels (``calc_xpwp``).

    ``xpwp[k] = Km_zm[k]*invrs_dzm[k]*(xm[k]-xm[k-1])`` for interior k; top and
    bottom levels are zero. ``Km_zm``/``invrs_dzm`` are ``(ngrdcol, nzm)``;
    ``xm`` is ``(ngrdcol, nzt)``.
    """
    ng, nzm = Km_zm.shape
    interior = Km_zm[:, 1:nzm - 1] * invrs_dzm[:, 1:nzm - 1] * (xm[:, 1:] - xm[:, :-1])
    return jnp.zeros((ng, nzm), dtype=xm.dtype).at[:, 1:nzm - 1].set(interior)


def clip_covar(wpxp, wp2, xp2, max_mag_corr=_MAX_MAG_CORRELATION):
    """Cauchy-Schwarz clip of a covariance after the solve (``clip_covar``).

    Clips ``wpxp`` to ``±max_mag_corr·sqrt(wp2·xp2)`` at interior levels; the
    top/bottom boundaries are left unchanged. ``_safe_sqrt`` keeps the gradient
    finite where a variance is zero (forward-identical, variances ≥ 0).
    """
    bound = max_mag_corr * _safe_sqrt(wp2 * xp2)
    clipped = jnp.clip(wpxp, -bound, bound)
    clipped = clipped.at[:, 0].set(wpxp[:, 0])
    clipped = clipped.at[:, -1].set(wpxp[:, -1])
    return clipped


def compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing):
    """Coriolis + geostrophic + prescribed-forcing wind tendencies (``compute_uv_tndcy``).

    ``d(um)/dt = -fcor·vg + fcor·vm + um_forcing``;
    ``d(vm)/dt = +fcor·ug - fcor·um + vm_forcing``. ``fcor`` is ``(ngrdcol,)``.
    """
    f = fcor[:, None]
    return (-f * vg + f * vm + um_forcing, f * ug - f * um + vm_forcing)


def windm_edsclrm_rhs(lhs_diff, xm, xm_tndcy, dt):
    """RHS of the um/vm tridiagonal solve (``windm_edsclrm_rhs``, implicit sfc flux).

    ``rhs[k] = 0.5·explicit_diffusion[k] + xm_tndcy[k] + xm[k]/dt`` (no explicit
    surface-flux term — it is implicit in the LHS).
    """
    invrs_dt = 1.0 / dt
    rhs_bot = (0.5 * (-lhs_diff[1, :, 0] * xm[:, 0] - lhs_diff[0, :, 0] * xm[:, 1])
               + xm_tndcy[:, 0] + invrs_dt * xm[:, 0])[:, None]
    rhs_int = (0.5 * (-lhs_diff[2, :, 1:-1] * xm[:, :-2]
                      - lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                      - lhs_diff[0, :, 1:-1] * xm[:, 2:])
               + xm_tndcy[:, 1:-1] + invrs_dt * xm[:, 1:-1])
    rhs_top = (0.5 * (-lhs_diff[2, :, -1] * xm[:, -2] - lhs_diff[1, :, -1] * xm[:, -1])
               + xm_tndcy[:, -1] + invrs_dt * xm[:, -1])[:, None]
    return jnp.concatenate([rhs_bot, rhs_int, rhs_top], axis=1)


def windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                      u_star_sqd, wind_speed, gr: CLUBBGrid):
    """Assemble the windm/edsclrm tridiagonal LHS (``windm_edsclrm_lhs``).

    CN diffusion (0.5·lhs_diff) + 1/dt accumulation + mean advection (interior
    only) + the implicit surface-momentum-flux term at the bottom level
    (``l_imp_sfc_momentum_flux = .true.``). ``k_lb_zt = k_lb_zm = 0`` (ascending).
    """
    lhs = 0.5 * lhs_diff
    lhs = lhs.at[1].add(1.0 / dt)
    lhs = lhs.at[:, :, :-1].add(lhs_ma_zt[:, :, :-1])
    sfc_term = (invrs_rho_ds_zt[:, 0] * gr.invrs_dzt[:, 0] * rho_ds_zm[:, 0]
                * (u_star_sqd / wind_speed[:, 0]))
    return lhs.at[1, :, 0].add(sfc_term)


# ---------------------------------------------------------------------------
# advance_windm_edsclrm
# ---------------------------------------------------------------------------

def advance_windm_edsclrm(um, vm, upwp, vpwp, wp2, up2, vp2, wm_zt, Kh_zm,
                          ug, vg, um_forcing, vm_forcing,
                          rho_ds_zm, rho_ds_zt, invrs_rho_ds_zt, fcor,
                          c_K10, nu10, dt, gr: CLUBBGrid, l_tke_aniso=True):
    """Advance um/vm (and the diagnostic upwp/vpwp) via eddy diffusion.

    Faithful port of ``advance_windm_edsclrm`` for the CAM ``l_predict_upwp_vpwp
    = .false.`` path (standalone, ascending grid, upwind MA, implicit surface
    momentum flux). Two Crank-Nicholson half-steps update the momentum fluxes
    around the implicit um/vm tridiagonal solve, then the fluxes are
    Cauchy-Schwarz clipped (``l_tke_aniso`` selects up2/vp2 vs wp2).

    Parameters (all ascending CLUBB grid)
    ----------
    um, vm, wm_zt, ug, vg, um_forcing, vm_forcing, rho_ds_zt, invrs_rho_ds_zt :
        zt-level fields ``(ngrdcol, nzt)``.
    upwp, vpwp, wp2, up2, vp2, Kh_zm, rho_ds_zm : zm-level fields ``(ngrdcol, nzm)``.
    fcor : Coriolis parameter ``(ngrdcol,)``.
    c_K10 : float
        Momentum-diffusivity coefficient (``CLUBBParams.c_K10``; ``Km = c_K10·Kh``).
    nu10 : float
        Background momentum diffusivity (``CLUBBParams.nu10``).
    dt : float
        Time step [s].
    l_tke_aniso : bool
        CAM default True → clip upwp/vpwp against up2/vp2 (else against wp2).

    Returns
    -------
    tuple of jax.Array
        ``(um_new, vm_new, upwp_new, vpwp_new)``.
    """
    nzm = gr.zm.shape[1]
    k_ub_zm = nzm - 1

    Km_zm = Kh_zm * c_K10
    Km_zm_p_nu10 = Km_zm + nu10

    nu10_arr = jnp.full((um.shape[0],), nu10, dtype=um.dtype)
    lhs_diff = diffusion_zt_lhs(Km_zm, nu10_arr, invrs_rho_ds_zt, rho_ds_zm, gr)
    lhs_ma_zt = term_ma_zt_lhs_upwind(wm_zt, gr)

    um_tndcy, vm_tndcy = compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing)

    # sqrt(max(s^2, eps^2)) is forward-identical to the reference's
    # max(sqrt(s^2), eps) but AD-safe: it never differentiates through sqrt(0),
    # so calm-wind columns (um=vm=0, which feed the implicit sfc-flux LHS term)
    # keep finite gradients.
    wind_speed = jnp.sqrt(jnp.maximum(um ** 2 + vm ** 2, _EPS ** 2))
    u_star_sqd = _safe_sqrt(upwp[:, 0] ** 2 + vpwp[:, 0] ** 2)

    # First Crank-Nicholson half (explicit) for upwp/vpwp.
    xpwp_u = calc_xpwp(Km_zm_p_nu10, um, gr.invrs_dzm)[:, 1:-1]
    upwp_new = upwp.at[:, 1:-1].set(-0.5 * xpwp_u).at[:, k_ub_zm].set(0.0)
    xpwp_v = calc_xpwp(Km_zm_p_nu10, vm, gr.invrs_dzm)[:, 1:-1]
    vpwp_new = vpwp.at[:, 1:-1].set(-0.5 * xpwp_v).at[:, k_ub_zm].set(0.0)

    lhs = windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                            u_star_sqd, wind_speed, gr)
    rhs_um = windm_edsclrm_rhs(lhs_diff, um, um_tndcy, dt)
    rhs_vm = windm_edsclrm_rhs(lhs_diff, vm, vm_tndcy, dt)
    um_new = tridiag_solve(lhs, rhs_um)
    vm_new = tridiag_solve(lhs, rhs_vm)

    # Second Crank-Nicholson half (implicit component) for upwp/vpwp.
    xpwp_u_new = calc_xpwp(Km_zm_p_nu10, um_new, gr.invrs_dzm)[:, 1:-1]
    xpwp_v_new = calc_xpwp(Km_zm_p_nu10, vm_new, gr.invrs_dzm)[:, 1:-1]
    upwp_new = upwp_new.at[:, 1:-1].add(-0.5 * xpwp_u_new)
    vpwp_new = vpwp_new.at[:, 1:-1].add(-0.5 * xpwp_v_new)

    xp2_u = up2 if l_tke_aniso else wp2
    xp2_v = vp2 if l_tke_aniso else wp2
    upwp_new = clip_covar(upwp_new, wp2, xp2_u)
    vpwp_new = clip_covar(vpwp_new, wp2, xp2_v)
    return um_new, vm_new, upwp_new, vpwp_new


__all__ = [
    "diffusion_zt_lhs",
    "term_ma_zt_lhs_upwind",
    "calc_xpwp",
    "clip_covar",
    "compute_uv_tndcy",
    "windm_edsclrm_rhs",
    "windm_edsclrm_lhs",
    "advance_windm_edsclrm",
]
