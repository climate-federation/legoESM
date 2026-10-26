"""Holtslag-Boville (1993) nonlocal K-profile boundary-layer turbulence.

Differentiable JAX port of E3SM v3.0.1 ``hb_diff.F90`` (``eddy_scheme='HB'``,
bit-identical to CAM ``cam_cesm2_1_rel_60``) + ``pbl_utils.F90``, followed by
the CAM ``diffusion_solver.F90::compute_vdiff`` tendency solve exactly as
``vertical_diffusion.F90::vertical_diffusion_tend`` sets it up for HB.
Validated by RUNNING those Fortran sources (fixture
``tests/atmosphere/oracle_fixtures/holtslag_boville.npz``):

1. Friction velocity from the surface stress with the dry bottom density
   ``rrho = R_d T/p`` (``calc_ustar``), kinematic fluxes and Obukhov length
   (``calc_obklen``); gradient Ri (``trbintd``).
2. Bulk-Richardson PBL height (``pblintd``): two-pass scan limited to the
   bottom ``npbl`` levels, ``npbl`` = number of REFERENCE-pressure levels
   with ``pref_mid >= pblmaxp``; no-crossing fallback ``z(pverp-npbl)``;
   ``700 u*`` floor; ``wstar`` from that height; then the ``zi(pver)+50 m``
   floor.
3. Free-atmosphere ``kvf = max(zkmin, ml2 sqrt(s2) f(Ri))`` with ``ml2 = 0``
   at the first interior interface (``austausch_atm``).
4. The nonlocal K-profile, Prandtl number, countergradient ``cgs``/``cgh``
   and the ``tpert``/``qpert``/``tke`` diagnostics on the bottom ``npbl-1``
   interfaces (``austausch_pbl``).
5. Tendencies: :func:`~legoesm.atmosphere.physics.turbulence.vertical_diffusion.cam_compute_vdiff`
   (dry static energy with the heat countergradient and frictional KE
   heating, moisture with the constituent countergradient and CAM's
   ``qmincg`` rollback, momentum with the explicit surface stress; CAM
   pressure metrics).

The oracle's hard ``if`` switches that already had a sharpness field in
:class:`HoltslagBovilleConfig` stay sigmoid blends that reduce to the oracle
at their hard limit; the switches without one (search region, PBL loop
range, ``tke`` gate, rollback) are ported hard.

NOT-YET-FAITHFUL (only when the caller passes no ``pref_mid``): the PBL search
region then falls back to the legacy actual-pressure sigmoid of 10 hPa width
with no PBL loop bound, which on elevated terrain gives a shallower search
region than the oracle (see the module PR).  Every lane currently calls the
kernel without ``pref_mid``.

References
----------
- Holtslag, A. A. M., & Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. J. Climate, 6,
  1825-1842.
- E3SM v3.0.1 (d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84)
  ``components/eam/src/physics/cam/hb_diff.F90``, ``pbl_utils.F90``;
  CAM cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56)
  ``diffusion_solver.F90``, ``vertical_diffusion.F90``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    exner_function,
    half_to_full,
    surface_buoyancy_flux,
    virtual_temperature,
    half_to_iface_zero_ends,
)
from legoesm.atmosphere.physics.turbulence.config import HoltslagBovilleConfig
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    first_crossing_height,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    cam_compute_vdiff,
    diagnostic_heat_flux_full,
    implicit_vertical_diffusion,
)

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Holtslag-Boville (1993) nonlocal K-profile PBL turbulence: a "
        "bulk-Richardson PBL height sets a nonlocal eddy-diffusivity profile "
        "with countergradient heat and moisture transport, blended with a "
        "free-atmosphere local-Ri diffusivity; port of E3SM/CAM hb_diff + "
        "the CAM compute_vdiff solver (dry static energy, KE heating)."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
        "pref_mid": "Pa",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Down-gradient K-profile (Km, Kh >= 0) plus positive countergradient "
        "fluxes rho*Kh*cgh (heat) and rho*Kh*cgs*kqfs (moisture) that carry "
        "heat/moisture UP in an unstable PBL. Surface shflx, qflx > 0 upward "
        "(into the air); stress taux,tauy is the stress ON the air. Top and "
        "surface interface diffusivities are zero. z increases upward, level 0 "
        "is the model top. Column dry static energy + KE changes only by the "
        "surface sensible flux (frictional dissipation returns the KE removed "
        "by diffusion and surface drag as heat); column water changes only by "
        "the surface water flux."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": (
        "Holtslag & Boville (1993), J. Climate 6, 1825-1842; "
        "E3SM v3.0.1 hb_diff.F90 + pbl_utils.F90; CAM cam_cesm2_1_rel_60 "
        "diffusion_solver.F90 compute_vdiff"
    ),
    "idealized_test": (
        "oracle fixture (21 columns) reproduced to <= 1e-10 rel at the hard "
        "limit; column water and DSE+KE budgets close to the surface fluxes."
    ),
}

_ONET = 1.0 / 3.0  # 1/3 power in the MO gradient expressions (oracle ``onet``)

# --- CAM constituent table (physpkg.F90:178, cnst_add('Q', ..., qmin)) ---
# Minimum specific humidity below which compute_vdiff rolls back the
# explicit constituent countergradient for the whole column (``qmincg``).
_QMINCG_Q = 1.0e-12


# Power floor for the gradient only: the cube root / square root have an
# INFINITE derivative at 0, so we floor the argument INSIDE the JVP while the
# forward value stays the exact oracle expression ``max(x, 0)**p``.  This is
# the key to (a) oracle fidelity -- the previous softplus floor shifted the
# value by ~0.69 at the floor, inflating neutral MO factors 4-8% (codex
# iter-1 finding 1) -- and (b) finite gradients in stable columns where the
# cube-root argument is exactly 0 (codex iter-1 finding 3).
_GRAD_FLOOR = 1.0e-6


@jax.custom_jvp
def _pos_cbrt(x: jax.Array) -> jax.Array:
    """``max(x, 0)**(1/3)`` with a finite gradient (floored only in the JVP).

    Forward value is the EXACT oracle expression (so a neutral argument of 1
    returns exactly 1, and ``wstar`` is exactly 0 for ``kbfs<=0``).  The JVP
    floors the argument so the ``(1/3)x**(-2/3)`` slope never blows up.
    """
    return jnp.maximum(x, 0.0) ** _ONET


@_pos_cbrt.defjvp
def _pos_cbrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = jnp.maximum(x, 0.0) ** _ONET
    # Derivative of max(x,0)**(1/3): exactly 0 for x<0 (the forward is flat),
    # and (1/3)x**(-2/3) for x>0 with x floored to _GRAD_FLOOR so the slope
    # near 0 stays finite (codex iter-2 finding 4: don't propagate a positive
    # slope through the x<0 flat region).
    xf = jnp.maximum(x, _GRAD_FLOOR)
    slope = jnp.where(x > 0.0, _ONET * xf ** (_ONET - 1.0), 0.0)
    return y, slope * dx


@jax.custom_jvp
def _pos_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x, 0))`` with a finite gradient (floored only in the JVP)."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


@_pos_sqrt.defjvp
def _pos_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = jnp.sqrt(jnp.maximum(x, 0.0))
    xf = jnp.maximum(x, _GRAD_FLOOR)
    slope = jnp.where(x > 0.0, 0.5 / jnp.sqrt(xf), 0.0)
    return y, slope * dx


def _safe_cbrt(x: jax.Array) -> jax.Array:
    """Cube root with exact forward value ``max(x,0)**(1/3)``, finite grad."""
    return _pos_cbrt(x)


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """Square root with exact forward value ``sqrt(max(x,0))``, finite grad."""
    return _pos_sqrt(x)


class HBDiffusivities(NamedTuple):
    """Output of :func:`hb_diffusivities` (oracle ``compute_hb_diff``).

    Interface arrays have shape ``(ncol, nlev-1)``: ``[j]`` is the interior
    interface between full levels ``j`` and ``j+1`` (oracle interface
    ``j+2``, 1-based); the oracle's model-top and surface interfaces are 0
    for every field below.
    """
    Km_half: jax.Array        # kvm [m2/s]
    Kh_half: jax.Array        # kvh (= kvq) [m2/s]
    cgh_half: jax.Array       # countergradient term for heat [J/kg/m]
    cgs_half: jax.Array       # countergradient star (cg/flux) [s/m2]
    tke_half: jax.Array       # crude TKE estimate (kvm/pblh)^2 [m2/s2]
    h_pbl: jax.Array          # pblh [m], (ncol,)
    wstar: jax.Array          # convective velocity scale [m/s], (ncol,)
    wm: jax.Array             # momentum velocity scale u*/phiminv [m/s], (ncol,)
    tpert: jax.Array          # convective temperature excess [K], (ncol,)
    qpert: jax.Array          # convective humidity excess [kg/kg], (ncol,)


def holtslag_boville_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: HoltslagBovilleConfig,
    pref_mid: jax.Array | None = None,
) -> TurbulenceOutput:
    """Holtslag-Boville turbulence: CAM ``compute_hb_diff`` + ``compute_vdiff``.

    Index convention: level 0 is the model top, level ``nlev-1`` is the
    surface-adjacent layer (CAM ``k=1..pver``).

    Parameters
    ----------
    u, v : (ncol, nlev) winds [m/s].
    T : (ncol, nlev) temperature [K].
    q_v : (ncol, nlev) specific humidity [kg/kg].
    p_full, p_half : (ncol, nlev) / (ncol, nlev+1) pressure [Pa]; the solver
        uses CAM's ``Coords1D(p_half)`` metrics, ``p_full`` feeds the Exner
        function and the bottom-level density of ``calc_ustar``.
    z_full, z_half : full/half-level height above the surface [m].
    T_sfc, q_sfc : (ncol,) surface temperature / humidity for the surface
        layer that produces the stress and fluxes.
    rho : (ncol, nlev) air density; consumed only by the surface layer (the
        scheme itself uses the oracle's dry ``R_d T / p`` densities).
    dt : time step [s].
    config : HoltslagBovilleConfig
    pref_mid : (nlev,) reference full-level pressures [Pa] (CAM ``pref_mid``
        = ``hyam*ps0 + hybm*psr``), or ``None``.  ``None`` keeps the legacy,
        NOT-faithful actual-pressure search region (module docstring).
    """
    g = constants.g
    cpair = constants.c_pd

    # ----- thv = virtem(th, q) (oracle compute_hb_diff) -----
    theta, theta_v = _theta_thetav(T, q_v, p_full)
    thv_bot = theta_v[:, -1]
    th_bot = theta[:, -1]

    # ----- trbintd: interface shear^2 and gradient Ri -----
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dvdz2 = (u[:, :-1] - u[:, 1:]) ** 2 + (v[:, :-1] - v[:, 1:]) ** 2
    dvdz2 = jnp.maximum(dvdz2, 1.0e-36)
    s2 = dvdz2 / dz_half ** 2
    thv_sum = theta_v[:, :-1] + theta_v[:, 1:]
    n2 = g * 2.0 * (theta_v[:, :-1] - theta_v[:, 1:]) / (
        jnp.clip(thv_sum, 1.0, None) * dz_half
    )
    Ri = n2 / s2  # (ncol, nlev-1)

    # ----- surface stress / fluxes from the surface layer -----
    tau_x, tau_y, shflx, lhflx, _ustar_sfc = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)   # qflx [kg/m2/s]

    # ----- calc_ustar / calc_obklen (pbl_utils) -----
    ustar, khfs, kqfs, kbfs, obklen = _calc_ustar_obklen(
        T[:, -1], p_full[:, -1], tau_x, tau_y, shflx, sflx_q, th_bot, thv_bot,
        config.ustar_min)

    # ----- pblintd + austausch_atm + austausch_pbl -----
    hb = hb_diffusivities(
        u, v, theta_v, z_full, z_half, p_full, s2, Ri,
        ustar, khfs, kqfs, kbfs, obklen, config, pref_mid,
    )

    # ----- CAM compute_vdiff (eddy_scheme='HB', do_iss=.false.) -----
    # Heat BC: shflx plus the legoESM latent-enthalpy term (user decision
    # 2026-09-28; zero when the surface charged the constant L_v).
    shflx_bc = shflx + latent_enthalpy_correction(lhflx, sflx_q)
    # In the winds' precision, as the solve (#2110): s_new - s_dse then
    # cancels the rounding of s exactly as in a uniform-precision run.
    s_dse = (cpair * T + g * z_full).astype(u.dtype)
    u_new, v_new, s_new, q_new = cam_compute_vdiff(
        u, v, s_dse, q_v, T, p_half, hb.Km_half, hb.Kh_half,
        hb.cgh_half, hb.cgs_half, tau_x, tau_y, shflx_bc, sflx_q, dt,
        _QMINCG_Q,
    )

    # Q1 diagnostic (LES-suite): nonlocal heat flux -Kh*(dtheta/dz - gamma)
    # with gamma = cgh/cpair = khfs*cgs.  Pure diagnostic.
    wtheta_flux = diagnostic_heat_flux_full(
        theta, dz_half, hb.Kh_half, hb.cgh_half / cpair,
    )

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(s_new - s_dse) / (cpair * dt),
        dq_v_dt=(q_new - q_v) / dt,
        Km=half_to_full(hb.Km_half),
        Kh=half_to_full(hb.Kh_half),
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=hb.h_pbl,
        wtheta_flux=wtheta_flux,
        Kh_iface=half_to_iface_zero_ends(hb.Kh_half),
    )


def _theta_thetav(T, q_v, p_full):
    """``th = t*exner`` and ``thv = virtem(th, q)`` (oracle input of pblintd).

    ``exner`` is the repo's ``exner_function`` (``p0``-referenced, as for the
    HB lane)."""
    exner = exner_function(p_full)
    theta = T / jnp.clip(exner, 1.0e-6, None)
    return theta, virtual_temperature(theta, q_v)


def _calc_ustar_obklen(T_bot, p_bot, tau_x, tau_y, shflx, qflx, th_bot,
                       thv_bot, ustar_min):
    """Oracle ``pbl_utils`` ``calc_ustar`` + ``calc_obklen``.

    Returns ``(ustar, khfs, kqfs, kbfs, obklen)`` from the bottom-level
    temperature/pressure (dry ``rrho = rair*t/pmid``), the surface stress
    [Pa], sensible heat flux [W/m2] and water flux [kg/m2/s].
    """
    g = constants.g
    vk = constants.kappa_von_karman
    rrho = constants.R_d * T_bot / jnp.clip(p_bot, 1.0, None)
    # ustar = max(sqrt(sqrt(taux^2+tauy^2)*rrho), ustar_min), written so the
    # max picks ustar_min BEFORE the square roots are differentiated (exact
    # forward value and exact gradient; no NaN at zero stress).
    tau2 = tau_x ** 2 + tau_y ** 2
    stressed = tau2 > 0.0
    ustar_tau = jnp.sqrt(jnp.sqrt(jnp.where(stressed, tau2, 1.0)) * rrho)
    ustar = jnp.where(stressed & (ustar_tau > ustar_min), ustar_tau, ustar_min)
    khfs = shflx * rrho / constants.c_pd             # [m K/s]
    kqfs = qflx * rrho                               # [m/s]
    kbfs = surface_buoyancy_flux(khfs, kqfs, th_bot)  # khfs + zvir*ths*kqfs
    # Obukhov length: L = -thvs*u*^3 / (g*vk*(kbfs + sign(1e-10,kbfs))).
    kbfs_signed = kbfs + jnp.sign(kbfs) * 1.0e-10 + (kbfs == 0) * 1.0e-10
    obklen = -thv_bot * ustar ** 3 / (g * vk * kbfs_signed)
    return ustar, khfs, kqfs, kbfs, obklen


def _unstable_indicator(kbfs, config):
    """Oracle ``unstbl = kbfs > 0``; biased so kbfs = 0 is stable."""
    return jax.nn.sigmoid(
        config.unstable_blend_sharpness * (kbfs - config.unstable_kbfs_threshold)
    )


def cam_pblintd_height(
    T, q_v, u, v, p_full, z_full, z_half, tau_x, tau_y, shflx, qflx,
    pref_mid, config: HoltslagBovilleConfig,
):
    """PBL height CAM6 diagnoses after CLUBB (``clubb_intr.F90:2594-2617``).

    ``th = t*exner``, ``thv = th*(1+zvir*q)``, ``calc_ustar`` from the
    surface stress, ``calc_obklen`` from the sensible heat flux and the water
    flux, then ``hb_diff::pblintd`` -- the same oracle routine the
    Holtslag-Boville lane ports (:func:`_pbl_height`, scan limited to the
    ``npbl`` levels with ``pref_mid >= pblmaxp``; cloud argument >= 0, so the
    ``zi(pver)+50 m`` floor is unconditional).  ``clubb_intr`` also sets
    ``kbfs = 0`` where it is ``-0``; no value downstream depends on the sign
    of a zero.  Inputs top-first ``(ncol, nlev)`` (``z_half`` ``nlev+1``,
    surface last), the per-column surface stress ``tau_x, tau_y`` [Pa],
    ``shflx`` [W/m2] and ``qflx`` [kg/m2/s] (CAM ``cam_in%wsx, wsy, shf,
    cflx(:,1)``), ``pref_mid`` (nlev,) [Pa].  Returns ``pblh`` [m], (ncol,).
    """
    theta, theta_v = _theta_thetav(T, q_v, p_full)
    ustar, _, _, kbfs, obklen = _calc_ustar_obklen(
        T[:, -1], p_full[:, -1], tau_x, tau_y, shflx, qflx, theta[:, -1],
        theta_v[:, -1], config.ustar_min)
    h_pbl, *_ = _pbl_height(
        u, v, theta_v, z_full, z_half, p_full, ustar, obklen, kbfs,
        _unstable_indicator(kbfs, config), theta_v[:, -1], constants.g,
        config.betam * config.sffrac, config.betah * config.sffrac,
        config.fak, config.Ri_crit, config, _npbl(pref_mid, config.pblmaxp),
    )
    return h_pbl


def _npbl(pref_mid, pblmaxp):
    """Oracle ``init_hb_diff``: ``npbl = max(count(pref_mid >= pblmaxp), 1)``."""
    return jnp.maximum(jnp.sum(jnp.asarray(pref_mid) >= pblmaxp), 1)


def hb_diffusivities(
    u, v, theta_v, z_full, z_half, p_full, s2, Ri,
    ustar, khfs, kqfs, kbfs, obklen, config: HoltslagBovilleConfig,
    pref_mid=None,
) -> HBDiffusivities:
    """Oracle ``pblintd`` + ``austausch_atm`` + ``austausch_pbl``.

    Driven by the kinematic surface fluxes so the oracle harness can compare
    the core directly.  Top-first ``(ncol, nlev)`` profiles; ``s2``/``Ri`` on
    the ``nlev-1`` interior interfaces; ``ustar``, ``khfs``, ``kqfs``,
    ``kbfs``, ``obklen`` per column.  ``pref_mid`` as in
    :func:`holtslag_boville_turbulence`.
    """
    g = constants.g
    vk = constants.kappa_von_karman
    cpair = constants.c_pd
    ncol, nlev = z_full.shape
    sffrac = config.sffrac
    betas = config.betas
    binm = config.betam * sffrac
    binh = config.betah * sffrac
    fak = config.fak
    fakn = config.fakn
    ccon = fak * sffrac * vk
    thv_bot = theta_v[:, -1]

    # Unstable indicator (oracle unstbl = kbfs > 0); biased so kbfs = 0 is stable.
    unstbl = _unstable_indicator(kbfs, config)

    # Oracle level ranges.  pref_mid given: hard masks from npbl (interface j
    # is oracle interface j+2; the K loop is k = pver..pver-npbl+2 and the
    # tke loop k = max(pverp-npbl,2)..pverp).  None: legacy, no bound.
    j_ifc = jnp.arange(nlev - 1)
    if pref_mid is None:
        npbl = None
        pbl_ifc_ok = jnp.ones((nlev - 1,), dtype=bool)
        tke_ifc_ok = pbl_ifc_ok
    else:
        npbl = _npbl(pref_mid, config.pblmaxp)
        pbl_ifc_ok = j_ifc >= nlev - npbl
        tke_ifc_ok = j_ifc >= nlev - 1 - npbl

    h_pbl, wstar, phiminv, phihinv, wm = _pbl_height(
        u, v, theta_v, z_full, z_half, p_full, ustar, obklen, kbfs, unstbl,
        thv_bot, g, binm, binh, fak, config.Ri_crit, config, npbl,
    )

    # ----- austausch_atm: free-atmosphere diffusivity -----
    # ml2(ntop_turb) = 0: the first interior interface has no mixing length.
    ml2 = jnp.where(j_ifc == 0, 0.0, config.ml_free ** 2)
    kvn = ml2 * _safe_sqrt(s2)
    f_unstable = _safe_sqrt(1.0 - config.free_ri_unstable_coeff * Ri)
    f_stable = 1.0 / (
        1.0 + config.free_ri_stable_c1 * Ri * (1.0 + config.free_ri_stable_c2 * Ri)
    )
    w_ri = jax.nn.sigmoid(config.stable_blend_sharpness * Ri)
    fofri = (1.0 - w_ri) * f_unstable + w_ri * f_stable
    kvf = jnp.maximum(config.kvf_min, kvn * fofri)

    # ----- austausch_pbl: K-profile inside the PBL -----
    z_lower = z_full[:, 1:]                          # oracle z(k)
    zp = z_full[:, :-1]                              # oracle zp = z(k-1)
    # if (zkmin == 0 .and. zp > pblh) zp = pblh
    zp = jnp.where((config.kvf_min == 0.0) & (zp > h_pbl[:, None]),
                   h_pbl[:, None], zp)
    zmzp = 0.5 * (z_lower + zp)
    zh = zmzp / jnp.clip(h_pbl[:, None], 1.0, None)
    zl = zmzp / obklen[:, None]
    zzh = zh * jnp.maximum(0.0, 1.0 - zh) ** 2

    fak1 = (ustar * h_pbl * vk)[:, None]
    fak2 = (wm * h_pbl * vk)[:, None]
    fak3 = (fakn * wstar / jnp.clip(wm, 1.0e-6, None))[:, None]

    # unstable surface layer (zh < sffrac)
    term = _safe_cbrt(1.0 - config.betam * zl)
    pblk_sfc_u = fak1 * zzh * term
    pr_sfc_u = term / jnp.clip(_safe_sqrt(1.0 - config.betah * zl), config.arg_floor, None)
    # unstable outer layer (zh >= sffrac)
    pblk_out_u = fak2 * zzh
    pr_out_u = (phiminv / jnp.clip(phihinv, 1.0e-6, None))[:, None] + (
        ccon * fak3 / fak
    )
    cgs_outer = fak3 / jnp.clip((h_pbl * wm)[:, None], 1.0e-12, None)

    w_outer = jax.nn.sigmoid(config.sfc_blend_sharpness * (zh - sffrac))
    pblk_u = (1.0 - w_outer) * pblk_sfc_u + w_outer * pblk_out_u
    pr_u = (1.0 - w_outer) * pr_sfc_u + w_outer * pr_out_u
    cgs_u = w_outer * cgs_outer

    # stable (kbfs <= 0); denominators floored so the inactive branch stays
    # finite in unstable columns (zl < 0).
    den_lo = jnp.maximum(1.0 + betas * zl, config.arg_floor)
    den_hi = jnp.maximum(betas + zl, config.arg_floor)
    w_zl = jax.nn.sigmoid(config.stable_blend_sharpness * (zl - 1.0))
    pblk_s = (1.0 - w_zl) * (fak1 * zzh / den_lo) + w_zl * (fak1 * zzh / den_hi)

    w_uns = unstbl[:, None]
    pblk = w_uns * pblk_u + (1.0 - w_uns) * pblk_s
    pr = w_uns * pr_u + (1.0 - w_uns) * jnp.ones_like(pblk_s)
    cgs = w_uns * cgs_u

    # pblpt = z(k) < pblh (smooth gates, existing sharpness fields) and the
    # oracle loop range (hard).
    in_pbl = jax.nn.sigmoid(config.sfc_blend_sharpness * (1.0 - zh))
    in_pbl_cg = jax.nn.sigmoid(
        config.cgs_gate_sharpness * (1.0 - z_lower / jnp.clip(h_pbl[:, None], 1.0, None)))
    loop = pbl_ifc_ok[None, :]
    Km_half = jnp.where(loop, jnp.maximum(pblk * in_pbl, kvf), kvf)
    Kh_half = jnp.where(
        loop, jnp.maximum(pblk * in_pbl / jnp.clip(pr, 1.0e-6, None), kvf), kvf)
    cgs_half = jnp.where(loop, cgs * in_pbl_cg, 0.0)
    cgh_half = khfs[:, None] * cgs_half * cpair

    # tpert / qpert (austausch_pbl)
    tpert = w_uns[:, 0] * jnp.maximum(khfs * fak / jnp.clip(wm, 1.0e-6, None), 0.0) + (
        1.0 - w_uns[:, 0]) * jnp.maximum(khfs * fak / ustar, 0.0)
    qpert = w_uns[:, 0] * jnp.maximum(kqfs * fak / jnp.clip(wm, 1.0e-6, None), 0.0) + (
        1.0 - w_uns[:, 0]) * jnp.maximum(kqfs * fak / ustar, 0.0)
    # tke = (kvm/pblh)^2 where z(k-1) < pblh (hard gate, oracle loop range)
    tke_half = jnp.where(
        tke_ifc_ok[None, :] & (z_full[:, :-1] < h_pbl[:, None]),
        (Km_half / h_pbl[:, None]) ** 2, 0.0)

    return HBDiffusivities(Km_half, Kh_half, cgh_half, cgs_half, tke_half,
                           h_pbl, wstar, wm, tpert, qpert)


def _pbl_height(
    u, v, theta_v, z_full, z_half, p_full, ustar, obklen, kbfs, unstbl,
    thv_bot, g, binm, binh, fak, ricr, config: HoltslagBovilleConfig, npbl,
):
    """Bulk-Richardson PBL height (oracle ``pblintd``).

    Returns ``(h_pbl, wstar, phiminv, phihinv, wm)``; ``phiminv``, ``phihinv``
    and ``wm`` are evaluated at the final ``h_pbl`` as in ``austausch_pbl``.
    The first-crossing scan + linear interpolation is the shared
    ``first_crossing_height`` (``pbl_crossing_sharpness``).  ``npbl`` (from
    ``pref_mid``) limits the scan to the bottom ``npbl`` levels with fallback
    ``z(pverp-npbl)``; ``npbl=None`` is the legacy NOT-faithful region
    (actual-pressure sigmoid, 10 hPa width).
    """
    ncol, nlev = z_full.shape
    z_bot = z_full[:, -1:]
    vvk = (u - u[:, -1:]) ** 2 + (v - v[:, -1:]) ** 2 + config.pblh_ustar_fac * ustar[:, None] ** 2
    vvk = jnp.maximum(vvk, 1.0e-36)
    dz_sfc = z_full - z_bot
    thv_bot_c = thv_bot[:, None]

    if npbl is None:
        # LEGACY (not faithful): smooth region on ACTUAL pressure.
        search_ok = jax.nn.sigmoid((p_full - config.pblmaxp) / 1.0e3)
        d_search = search_ok - jnp.concatenate(
            [jnp.zeros((ncol, 1), search_ok.dtype), search_ok[:, :-1]], axis=1,
        )
        edge_w = jnp.maximum(d_search, 0.0) + 1.0e-20
        z_top_search = jnp.sum(edge_w * z_full, axis=1) / jnp.sum(edge_w, axis=1)
    else:
        # Oracle: scan k = pver-1 .. pver-npbl+1, fallback z(pverp-npbl).
        lev_ok = jnp.arange(nlev) >= nlev - npbl
        search_ok = jnp.broadcast_to(lev_ok, (ncol, nlev)).astype(z_full.dtype)
        z_top_search = jnp.take(z_full, nlev - npbl, axis=1)

    # Pass 1 bulk Ri relative to thv_bot.
    rino1 = g * (theta_v - thv_bot_c) * dz_sfc / (jnp.clip(thv_bot_c, 1.0, None) * vvk)
    h1 = first_crossing_height(
        rino1, z_full, ricr, config.pbl_crossing_sharpness,
        search_ok=search_ok, z_top=z_top_search,
    )

    # Pass 2 (unstable): surface excess tlv = thv_bot + kbfs*fak/(u* phiminv).
    phiminv1 = _safe_cbrt(1.0 - binm * h1 / obklen)
    tlv = thv_bot + kbfs * fak / jnp.clip(ustar * phiminv1, 1.0e-6, None)
    rino2 = g * (theta_v - tlv[:, None]) * dz_sfc / (jnp.clip(thv_bot_c, 1.0, None) * vvk)
    h2 = first_crossing_height(
        rino2, z_full, ricr, config.pbl_crossing_sharpness,
        search_ok=search_ok, z_top=z_top_search,
    )
    h_pbl = unstbl * h2 + (1.0 - unstbl) * h1

    # pblh = max(pblh, 700 u*); wstar from THIS height; then the
    # unconditional cloud/ventilation floor zi(pver) + 50 m.
    h_pbl = jnp.maximum(h_pbl, config.pblh_mech_coeff * ustar)
    wstar = _safe_cbrt(
        jnp.maximum(0.0, kbfs) * g * h_pbl / jnp.clip(thv_bot, 1.0, None)
    )
    h_pbl = jnp.maximum(h_pbl, z_half[:, -2] + config.cloud_pbl_floor_m)

    phiminv = _safe_cbrt(1.0 - binm * h_pbl / obklen)
    phihinv = _safe_sqrt(1.0 - binh * h_pbl / obklen)
    wm = ustar * phiminv
    return h_pbl, wstar, phiminv, phihinv, wm


def diffuse_theta_with_countergradient(
    T, Kh_half, rho, dz, dz_half, p_full, dt, surface_flux_T, gamma_theta_half,
):
    """Implicit theta diffusion with the HB nonlocal countergradient.

    The countergradient enters as ``flux = -rho*Kh*(dtheta/dz - gamma)``.
    The ``-Kh*gamma`` part is an explicit (known) flux added to the column;
    its divergence is an explicit source applied before the implicit solve,
    exactly as CAM treats ``cgh`` as a known countergradient flux.

    Implementation: convert T->theta, add the explicit countergradient flux
    divergence as a theta source (forward), then run the standard implicit
    theta diffusion of the local gradient with the surface flux, and convert
    back.  This keeps the implicit part identical to the no-cg path while
    adding the nonlocal term consistently with the oracle.

    Conserved-variable caveat.  The E3SM oracle diffuses dry static energy
    s = c_p T + g z (enthalpy-conserving in flux form); this routine diffuses
    θ instead, so it reproduces the oracle's K-profile and countergradient
    structure but conserves mass-weighted θ (``Σ ρ dz θ``) rather than column
    enthalpy (``Σ ρ dz c_p T``), since the Exner π = (p/p_ref)^κ varies with
    height.  An energy-conserving variant would diffuse s; this is an accepted
    approximation and a validated follow-up.

    Surface-Exner proxy.  ``exner_sfc`` below uses the lowest FULL-level Exner
    (exner[:, -1]) as a surface-Exner stand-in.  Since p_low < p_surface this
    proxy is biased low, so the injected surface θ-flux F_θ_sfc =
    F_T_sfc / exner_sfc is biased slightly HIGH; threading a true p_surface
    and using (p_sfc/p_ref)^κ would remove the bias.
    """
    # Exner Pi = (p/p_ref)^kappa via the canonical helper (same 1 Pa floor).
    exner = exner_function(p_full)
    exner_safe = jnp.clip(exner, 1.0e-6, None)
    theta = T / exner_safe
    exner_sfc = exner_safe[:, -1]

    # Explicit countergradient flux on interfaces: F_cg = rho_half*Kh*gamma.
    # Positive gamma (warm, unstable BL) drives an UPWARD theta flux, which
    # warms the layers below the flux convergence -> mixes heat up against
    # the local gradient (the defining HB nonlocal effect).
    rho_half = 0.5 * (rho[:, :-1] + rho[:, 1:])
    F_cg = rho_half * Kh_half * gamma_theta_half  # (ncol, nlev-1), upward +

    # Flux divergence -> theta source: d(theta)/dt = -(1/rho) dF/dz.
    # Interface k between full levels k and k+1. For full level k, the net
    # flux convergence is (F_top - F_bot)/(rho*dz). Interior levels see two
    # interfaces; top/bottom see one (zero-flux top, surface handled by the
    # surface flux BC, so countergradient flux at the surface interface is 0).
    ncol, nlev = T.shape
    # F at interface above level k is F_cg[k-1]; below level k is F_cg[k].
    F_above = jnp.concatenate([jnp.zeros((ncol, 1), F_cg.dtype), F_cg], axis=1)  # (ncol,nlev)
    F_below = jnp.concatenate([F_cg, jnp.zeros((ncol, 1), F_cg.dtype)], axis=1)  # (ncol,nlev)
    # d(theta)/dt = -(F_above - F_below)/(rho*dz)  [upward-positive flux,
    # level 0 = top]. Flux convergence warms the layer where F decreases
    # upward.
    dtheta_cg = -(F_above - F_below) / (jnp.clip(rho, 1.0e-6, None) * dz)
    theta_star = theta + dt * dtheta_cg

    # Implicit local-gradient diffusion with the surface theta flux.
    surface_flux_theta = surface_flux_T / exner_sfc
    theta_new = implicit_vertical_diffusion(
        theta_star, Kh_half, rho, dz, dz_half, dt, surface_flux_theta,
    )
    return theta_new * exner
