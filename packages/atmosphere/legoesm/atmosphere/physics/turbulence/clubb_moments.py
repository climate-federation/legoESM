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
from legoesm.atmosphere.physics.turbulence.clubb_fill_holes import fill_holes_vertical
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid
from legoesm.atmosphere.physics.turbulence.clubb_solve import tridiag_solve

from legoesm import constants

_EPS = 1.0e-10
_MAX_MAG_CORRELATION = 0.99   # Cauchy-Schwarz correlation bound (constants_clubb)
_ZERO_THRESHOLD = 0.0
_ONE_THIRD = 1.0 / 3.0
_GAMMA_OVER_IMPLICIT_TS = 1.5   # over-implicit weight (constants_clubb)


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


def diffusion_zm_lhs(K_zt, nu, invrs_rho_ds_zm, rho_ds_zt, gr: CLUBBGrid):
    """Tridiagonal LHS for implicit eddy diffusion of a zm-level variable.

    Faithful port of ``diffusion.F90:diffusion_zm_lhs`` (non-upwind): discretizes
    ``d/dz[(K_zt + nu) d(var_zm)/dz]`` at zm levels with zero-flux boundaries.
    Returns ``(3, ngrdcol, nzm)`` = ``[super, main, sub]``. (The k=0 row is not
    used by the solver, per the Fortran note, but is filled for shape.)
    """
    K_zt_nu = K_zt + nu[:, None]
    invrs_dzm = gr.invrs_dzm
    invrs_dzt = gr.invrs_dzt

    common_bot = (invrs_dzm[:, :1] * invrs_rho_ds_zm[:, :1]
                  * K_zt_nu[:, :1] * rho_ds_zt[:, :1] * invrs_dzt[:, :1])
    super_bot, main_bot, sub_bot = -common_bot, common_bot, jnp.zeros_like(common_bot)

    scale_int = invrs_dzm[:, 1:-1] * invrs_rho_ds_zm[:, 1:-1]
    super_int = -scale_int * K_zt_nu[:, 1:] * rho_ds_zt[:, 1:] * invrs_dzt[:, 1:]
    sub_int = -scale_int * K_zt_nu[:, :-1] * rho_ds_zt[:, :-1] * invrs_dzt[:, :-1]
    main_int = -(super_int + sub_int)

    common_top = (invrs_dzm[:, -1:] * invrs_rho_ds_zm[:, -1:]
                  * K_zt_nu[:, -1:] * rho_ds_zt[:, -1:] * invrs_dzt[:, -1:])
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


def term_ma_zm_lhs(wm_zm, gr: CLUBBGrid):
    """Centered mean-advection LHS for a zm-level variable (``term_ma_zm_lhs``).

    Faithful port of ``mean_adv.F90:term_ma_zm_lhs``: discretizes
    ``w·d(var_zm)/dz`` implicitly at interior momentum levels with the
    zm→zt interpolation weights. The xp2/xpyp moments live on zm, so their
    mean advection always uses this centered form (the ``l_upwind_xm_ma`` flag
    gates only the *zt*-level scalar/wind advance, not the zm-level moments).

    The zm→zt weights are computed inline from the grid geometry — exactly
    ``calc_zm2zt_weights`` (``grid_class.F90``), ascending grid (grid_dir=+1):

      ``w_above[k] = (zt[k] - zm[k])   / (zm[k+1] - zm[k])``  (weight of zm[k]),
      ``w_below[k] = (zm[k+1] - zt[k]) / (zm[k+1] - zm[k])``  (weight of zm[k+1]),

    for ``k = 0 .. nzt-1``. On a uniform grid both are 1/2. Boundary rows
    (k=0, k=nzm-1) are zero (fixed-value BCs applied by the assembler).
    ``(3, ngrdcol, nzm)`` = ``[super, main, sub]``.
    """
    invrs_dzm = gr.invrs_dzm                       # (ng, nzm)
    total_dist = (gr.zm[:, 1:] - gr.zm[:, :-1]) + 1.0e-30   # (ng, nzt)
    w_above = (gr.zt - gr.zm[:, :-1]) / total_dist  # M_ABOVE, (ng, nzt)
    w_below = (gr.zm[:, 1:] - gr.zt) / total_dist   # M_BELOW, (ng, nzt)

    # Interior momentum levels k = 1 .. nzm-2 (Fortran k = 2 .. nzm-1).
    fac = wm_zm[:, 1:-1] * invrs_dzm[:, 1:-1]       # (ng, nzm-2)
    super_int = fac * w_above[:, 1:]                # weights_zm2zt[:, 1:, M_ABOVE]
    main_int = fac * (w_below[:, 1:] - w_above[:, :-1])
    sub_int = -fac * w_below[:, :-1]                # weights_zm2zt[:, :-1, M_BELOW]

    zeros_bnd = jnp.zeros((wm_zm.shape[0], 1), dtype=wm_zm.dtype)
    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag = jnp.concatenate([zeros_bnd, main_int, zeros_bnd], axis=1)
    subdiag = jnp.concatenate([zeros_bnd, sub_int, zeros_bnd], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


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


# ---------------------------------------------------------------------------
# advance_xp2_xpyp term builders (scalar/horizontal-velocity variance equations)
# ---------------------------------------------------------------------------

def term_dp1_lhs(Cn, invrs_tau_zm):
    """Main-diagonal dissipation-term-1 coefficient for x_a'x_b' (``term_dp1_lhs``).

    Implicit ``+(C_n/tau_zm)·x_a'x_b'(t+1)`` — main diagonal only, interior
    levels; boundaries zero. ``Cn``/``invrs_tau_zm`` are ``(ngrdcol, nzm)``.
    """
    interior = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]
    zeros_bnd = jnp.zeros((Cn.shape[0], 1), dtype=Cn.dtype)
    return jnp.concatenate([zeros_bnd, interior, zeros_bnd], axis=1)


def term_dp1_rhs(Cn, invrs_tau_zm, threshold):
    """Explicit dissipation-term-1 RHS for x'y' (``term_dp1_rhs``), all levels.

    The explicit part of ``-(C_n/tau_zm)·(x'y' - threshold)`` is
    ``+(C_n/tau_zm)·threshold``.
    """
    return Cn * invrs_tau_zm * threshold


def term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm):
    """Turbulent production of x_a'x_b' (explicit) on interior zm levels (``term_tp_rhs``).

    ``rhs = -w'x_b'·d(x_am)/dz - w'x_a'·d(x_bm)/dz``. ``x_am``/``x_bm`` are zt
    (nzt); returns the interior slice ``(ngrdcol, nzm-2)``.
    """
    return (-wpxbp[:, 1:-1] * invrs_dzm[:, 1:-1] * (xam[:, 1:] - xam[:, :-1])
            - wpxap[:, 1:-1] * invrs_dzm[:, 1:-1] * (xbm[:, 1:] - xbm[:, :-1]))


def term_pr1(C4, C14, xbp2, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd):
    """Explicit pressure/dissipation term 1 for up2/vp2 (``term_pr1``), interior.

    ``rhs = (1/3)C4(xbp2+wp2)/tau_C4 - (1/3)C14(xbp2+wp2)/tau_C14
            + C14·w_tol²/tau_C14``; ``xbp2`` is the *other* horizontal variance.
    Returns the interior slice ``(ngrdcol, nzm-2)``.
    """
    return (_ONE_THIRD * C4 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]
            - _ONE_THIRD * C14 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C14_zm[:, 1:-1]
            + C14 * invrs_tau_C14_zm[:, 1:-1] * w_tol_sqd)


def term_pr2(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, vpwp, um, vm, gr: CLUBBGrid):
    """Explicit pressure term 2 (PR2) for up2/vp2 (``term_pr2``), interior, floored ≥0.

    ``rhs = (2/3)[C_uu_buoy·(g/thv_ds)·w'thv' + C_uu_shr·(-u'w'·d(um)/dz
            - v'w'·d(vm)/dz)]`` clamped to 0. Uses ``constants.g``. Returns the
    interior slice ``(ngrdcol, nzm-2)``.
    """
    invrs_dzm = gr.invrs_dzm
    du_dz = invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
    dv_dz = invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
    pr2 = (2.0 / 3.0) * (
        C_uu_buoy * (constants.g / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
        + C_uu_shr * (-upwp[:, 1:-1] * du_dz - vpwp[:, 1:-1] * dv_dz))
    return jnp.maximum(pr2, _ZERO_THRESHOLD)


# ---------------------------------------------------------------------------
# Turbulent advection of xp2/xpyp (CAM l_upwind_xpyp_ta = True; ascending grid)
# ---------------------------------------------------------------------------
# CAM uses the upwind (Godunov-style one-sided) turbulent-advection operator on
# the ascending grid (grid_dir = +1). The centered branch (needs weights_zm2zt)
# is out of the CAM-default tree and not ported.


def _xpyp_ta_pdf_lhs_upwind(coef_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Upwind turbulent-advection LHS for xp2/xpyp (``xpyp_term_ta_pdf_lhs``).

    One-sided stencil keyed on ``sgn`` (grid_dir=+1): ``(3, ngrdcol, nzm)`` =
    ``[super, main, sub]``; boundaries zero.
    """
    invrs_dzt = gr.invrs_dzt
    irho = invrs_rho_ds_zm[:, 1:-1]
    s = sgn[:, 1:-1]
    rho_k, coef_k = rho_ds_zm[:, 1:-1], coef_zm[:, 1:-1]
    rho_km1, coef_km1 = rho_ds_zm[:, :-2], coef_zm[:, :-2]
    rho_kp1, coef_kp1 = rho_ds_zm[:, 2:], coef_zm[:, 2:]
    idzt_km1, idzt_k = invrs_dzt[:, :-1], invrs_dzt[:, 1:]
    zint = jnp.zeros_like(rho_k)

    sup_up = zint
    main_up = irho * idzt_km1 * rho_k * coef_k
    sub_up = -irho * idzt_km1 * rho_km1 * coef_km1
    sup_dn = irho * idzt_k * rho_kp1 * coef_kp1
    main_dn = -irho * idzt_k * rho_k * coef_k
    sub_dn = zint
    is_up = s > 0.0
    super_int = jnp.where(is_up, sup_up, sup_dn)
    main_int = jnp.where(is_up, main_up, main_dn)
    sub_int = jnp.where(is_up, sub_up, sub_dn)

    zb = jnp.zeros((rho_ds_zm.shape[0], 1), dtype=rho_ds_zm.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([zb, main_int, zb], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def _xpyp_ta_pdf_rhs_upwind(term_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Upwind turbulent-advection explicit RHS for xp2/xpyp (``xpyp_term_ta_pdf_rhs``).

    Returns ``(ngrdcol, nzm)``; boundaries zero.
    """
    invrs_dzt = gr.invrs_dzt
    irho = invrs_rho_ds_zm[:, 1:-1]
    s = sgn[:, 1:-1]
    rho_k, term_k = rho_ds_zm[:, 1:-1], term_zm[:, 1:-1]
    rho_km1, term_km1 = rho_ds_zm[:, :-2], term_zm[:, :-2]
    rho_kp1, term_kp1 = rho_ds_zm[:, 2:], term_zm[:, 2:]
    idzt_km1, idzt_k = invrs_dzt[:, :-1], invrs_dzt[:, 1:]

    rhs_up = -irho * idzt_km1 * (rho_k * term_k - rho_km1 * term_km1)
    rhs_dn = -irho * idzt_k * (rho_kp1 * term_kp1 - rho_k * term_k)
    rhs_int = jnp.where(s > 0.0, rhs_up, rhs_dn)
    zb = jnp.zeros((rho_ds_zm.shape[0], 1), dtype=rho_ds_zm.dtype)
    return jnp.concatenate([zb, rhs_int, zb], axis=1)


def calc_xp2_xpyp_ta_lhs(wp3_on_wp2, sigma_sqd_w, beta, rho_ds_zm, invrs_rho_ds_zm,
                         gr: CLUBBGrid):
    """Shared implicit turbulent-advection LHS for xp2/xpyp (ADG1 upwind path).

    The operator depends only on the w-PDF, so it is the SAME ``(3, ngrdcol,
    nzm)`` for all five moments. ``beta`` is a scalar/per-column param.
    """
    beta_c = jnp.asarray(beta)
    beta_c = beta_c[:, None] if beta_c.ndim == 1 else beta_c
    a1 = 1.0 / (1.0 - sigma_sqd_w)
    sgn = jnp.where(wp3_on_wp2 >= 0.0, 1.0, -1.0)
    coef_zm = _ONE_THIRD * beta_c * a1 * wp3_on_wp2
    return _xpyp_ta_pdf_lhs_upwind(coef_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr)


def calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta, flux_a_zm, flux_b_zm,
                         rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Turbulent-advection explicit RHS for one xp2/xpyp moment (ADG1 upwind).

    Explicit term ``wp_coef·<w'a'><w'b'>`` with ``wp_coef = (1 - beta/3)·a1²·
    wp3_on_wp2 / wp2``; variance uses ``flux_a = flux_b``, covariance uses the
    two distinct fluxes (e.g. rtpthlp: wprtp, wpthlp).
    """
    beta_c = jnp.asarray(beta)
    beta_c = beta_c[:, None] if beta_c.ndim == 1 else beta_c
    a1 = 1.0 / (1.0 - sigma_sqd_w)
    sgn = jnp.where(wp3_on_wp2 >= 0.0, 1.0, -1.0)
    wp_coef = (1.0 - _ONE_THIRD * beta_c) * a1 ** 2 * wp3_on_wp2 / wp2
    return _xpyp_ta_pdf_rhs_upwind(wp_coef * flux_a_zm * flux_b_zm, sgn,
                                   rho_ds_zm, invrs_rho_ds_zm, gr)


def xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, dt, gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Assemble the full xp2/xpyp tridiagonal LHS (``xp2_xpyp_lhs``).

    Interior: ``diff + ma + gamma·ta`` (+ ``lhs_dp1`` + ``1/dt`` on the main
    diagonal); boundaries are fixed-value BCs ``[0, 1, 0]``. ``lhs_dp1`` is the
    dissipation main-diagonal pre-scaled by the caller. ``(3, ngrdcol, nzm)``.
    """
    super_int = lhs_diff[0, :, 1:-1] + lhs_ma[0, :, 1:-1] + lhs_ta[0, :, 1:-1] * gamma
    main_int = (lhs_diff[1, :, 1:-1] + lhs_ma[1, :, 1:-1] + lhs_ta[1, :, 1:-1] * gamma
                + lhs_dp1[:, 1:-1] + 1.0 / dt)
    sub_int = lhs_diff[2, :, 1:-1] + lhs_ma[2, :, 1:-1] + lhs_ta[2, :, 1:-1] * gamma

    ng = lhs_ta.shape[1]
    zb = jnp.zeros((ng, 1), dtype=lhs_ta.dtype)
    ob = jnp.ones((ng, 1), dtype=lhs_ta.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([ob, main_int, ob], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def xp2_xpyp_rhs(lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold, xapxbp, xam, xbm,
                 wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt, gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Explicit RHS of the x'^2 / x'y' equations (``xp2_xpyp_rhs``).

    Interior: ``rhs_ta + (1-gamma)·(over-implicit TA) + turbulent-production
    + Cn/tau·threshold + (1-gamma)·(over-implicit DP1) + forcing + xapxbp/dt``.
    BCs: lower carries the current value, upper is set to ``threshold``.
    ``(ngrdcol, nzm)``.
    """
    one_minus_gamma = 1.0 - gamma
    rhs_tp_int = term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm)
    rhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1] * threshold
    lhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]

    rhs_int = (rhs_ta[:, 1:-1]
               + one_minus_gamma * (-lhs_ta[0, :, 1:-1] * xapxbp[:, 2:]
                                    - lhs_ta[1, :, 1:-1] * xapxbp[:, 1:-1]
                                    - lhs_ta[2, :, 1:-1] * xapxbp[:, :-2])
               + rhs_tp_int + rhs_dp1_int
               + one_minus_gamma * (-lhs_dp1_int * xapxbp[:, 1:-1])
               + xpyp_forcing[:, 1:-1] + (1.0 / dt) * xapxbp[:, 1:-1])

    rhs_lb = xapxbp[:, 0:1]
    rhs_ub = jnp.full((Cn.shape[0], 1), threshold, dtype=Cn.dtype)
    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


def calc_xp2_xpyp_lhs(lhs_ta, lhs_ma, Kh_zt, c_K2, nu2, invrs_rho_ds_zm,
                      rho_ds_zt, Cn, invrs_tau_xp2_zm, gamma, dt, gr: CLUBBGrid):
    """Shared implicit LHS for rtp2/thlp2/rtpthlp (``calc_xp2_xpyp_lhs``).

    ``Kw2 = c_K2·Kh_zt`` eddy diffusion + the ``Cn`` pressure-damping (dp1) term,
    combined with the shared turbulent-advection (``lhs_ta``) and mean-advection
    (``lhs_ma``) operators. The SAME LHS solves all three second moments under
    ADG1. ``Cn``/``invrs_tau_xp2_zm`` are ``(ngrdcol, nzm)``; ``nu2`` is
    ``(ngrdcol,)``. Returns ``(lhs, lhs_diff, dp1)`` — ``lhs_diff``/``dp1`` are
    reused by the budget diagnostics.
    """
    Kw2 = c_K2 * Kh_zt
    lhs_diff = diffusion_zm_lhs(Kw2, nu2, invrs_rho_ds_zm, rho_ds_zt, gr)
    dp1 = term_dp1_lhs(Cn, invrs_tau_xp2_zm)
    lhs = xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, dp1 * gamma, dt)
    return lhs, lhs_diff, dp1


def calc_up2_vp2_lhs(lhs_ta, lhs_ma, Kh_zt, c_K9, nu9, invrs_rho_ds_zm,
                     rho_ds_zt, C4, C14, invrs_tau_C4_zm, invrs_tau_C14_zm,
                     gamma, dt, gr: CLUBBGrid):
    """Shared implicit LHS for up2/vp2 (``calc_up2_vp2_lhs``).

    The shared TA/MA operators plus the up2/vp2-specific ``Kw9 = c_K9·Kh_zt``
    eddy diffusion and the ``C4``/``C14`` pressure-damping (dp1) terms scaled by
    ``gamma``. The same LHS solves both up2 and vp2 (ADG1). ``c_K9`` may be a
    scalar or per-column ``(ngrdcol,)``. Returns
    ``(lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14)`` — the latter three are reused by
    the up2/vp2 RHS build and the dp2 budget diagnostic.
    """
    c_K9 = jnp.asarray(c_K9)
    c_K9 = c_K9[:, None] if c_K9.ndim == 1 else c_K9
    Kw9_zt = c_K9 * Kh_zt
    lhs_diff = diffusion_zm_lhs(Kw9_zt, nu9, invrs_rho_ds_zm, rho_ds_zt, gr)

    ng, nzm = invrs_tau_C4_zm.shape
    c4_1d = (2.0 / 3.0) * C4 * jnp.ones((ng, nzm), dtype=invrs_tau_C4_zm.dtype)
    c14_1d = _ONE_THIRD * C14 * jnp.ones((ng, nzm), dtype=invrs_tau_C14_zm.dtype)
    lhs_dp1_C4 = term_dp1_lhs(c4_1d, invrs_tau_C4_zm)
    lhs_dp1_C14 = term_dp1_lhs(c14_1d, invrs_tau_C14_zm)
    lhs_dp1 = (lhs_dp1_C4 + lhs_dp1_C14) * gamma
    lhs = xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, dt)
    return lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14


def xp2_xpyp_uv_rhs(rhs_ta_this, this_pre, other_pre, this_wp, this_dvel_dz,
                    lhs_splat, wp2, lhs_ta, C_uu_shr, C4, C14,
                    invrs_tau_C4_zm, invrs_tau_C14_zm, lhs_dp1_C4, lhs_dp1_C14,
                    pr2, omg, dt, w_tol_sqd, l_coriolis=False, fcor_y_col=None):
    """Explicit RHS for the up2 (or vp2) equation (``xp2_xpyp_uv_rhs``).

    The pressure-rotation (C_uu) form. Symmetric: for vp2 pass the v-quantities
    as ``this_*``/``this_wp``/``this_dvel_dz`` and the *pre-solve* up2 as
    ``other_pre`` (the C4/C14 isotropization couples the two horizontal
    variances). Interior terms: shared turbulent advection (over-implicit
    ``omg``), shear production ``(1-C_uu_shr)·(-2 u'w' d(vel)/dz)``, ``term_pr1``
    (C4/C14 isotropization), over-implicit dp1 damping, ``pr2`` (buoyancy/shear
    pressure), and ``xp2/dt``. The splat term ``½·lhs_splat·wp2`` is zero in the
    CAM default (``C_wp2_splat = 0``) but kept for faithfulness.

    ``l_coriolis`` (CAM default ``l_ho_nontrad_coriolis = .false.``) is a
    **compile-time static** feature gate (Python branch, not ``jnp.where``):
    when on, subtracts ``2·fcor_y·this_wp`` (``fcor_y_col`` is ``(ncol, 1)``).
    In normal use it is a static ``CLUBBFlags`` field closed over by the
    enclosing ``jax.jit``; if this helper is jitted directly, pass it via
    ``static_argnums=(19,)`` / ``static_argnames=("l_coriolis",)``. BCs: lower
    row carries the current value, upper row is ``w_tol_sqd``.
    ``this_dvel_dz``/``pr2`` are interior slices ``(ncol, nzm-2)``. Returns
    ``(ncol, nzm)``.
    """
    ng = this_pre.shape[0]
    rhs_int = (
        rhs_ta_this[:, 1:-1]
        + 0.5 * lhs_splat[:, 1:-1] * wp2[:, 1:-1]
        + omg * (-lhs_ta[0, :, 1:-1] * this_pre[:, 2:]
                 - lhs_ta[1, :, 1:-1] * this_pre[:, 1:-1]
                 - lhs_ta[2, :, 1:-1] * this_pre[:, :-2])
        + (1.0 - C_uu_shr) * (-this_wp[:, 1:-1] * this_dvel_dz
                              - this_wp[:, 1:-1] * this_dvel_dz)
        + term_pr1(C4, C14, other_pre, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd)
        + omg * (-lhs_dp1_C4[:, 1:-1] - lhs_dp1_C14[:, 1:-1]) * this_pre[:, 1:-1]
        + pr2
        + (1.0 / dt) * this_pre[:, 1:-1])
    if l_coriolis:
        rhs_int = rhs_int - 2.0 * fcor_y_col * this_wp[:, 1:-1]

    rhs_lb = this_pre[:, 0:1]
    rhs_ub = jnp.full((ng, 1), w_tol_sqd, dtype=this_pre.dtype)
    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


def pos_definite_variances(field, rho_ds_zm, dzm, threshold, hf_lower, hf_upper,
                           fill_holes_type):
    """Mass-conserving hole-fill of one variance field (``pos_definite_variances``).

    Thin wrapper over :func:`clubb_fill_holes.fill_holes_vertical` (CAM default
    ``fill_holes_type = 2``): restores ``field >= threshold`` over the zm
    interior ``[hf_lower, hf_upper]`` while conserving ``sum(rho_ds·dz·field)``.
    ``hf_lower``/``hf_upper``/``fill_holes_type`` are **compile-time static**
    (closed over by the enclosing ``jax.jit``; if jitted directly, pass via
    ``static_argnums=(4, 5, 6)``).
    """
    return fill_holes_vertical(field, rho_ds_zm, dzm, threshold,
                               hf_lower, hf_upper, fill_holes_type)


__all__ = [
    "diffusion_zt_lhs",
    "diffusion_zm_lhs",
    "xp2_xpyp_lhs",
    "xp2_xpyp_rhs",
    "term_ma_zt_lhs_upwind",
    "term_ma_zm_lhs",
    "calc_xp2_xpyp_lhs",
    "calc_up2_vp2_lhs",
    "calc_xpwp",
    "clip_covar",
    "compute_uv_tndcy",
    "windm_edsclrm_rhs",
    "windm_edsclrm_lhs",
    "advance_windm_edsclrm",
    "term_dp1_lhs",
    "term_dp1_rhs",
    "term_tp_rhs",
    "term_pr1",
    "term_pr2",
    "calc_xp2_xpyp_ta_lhs",
    "calc_xp2_xpyp_ta_rhs",
    "xp2_xpyp_uv_rhs",
    "pos_definite_variances",
]
