"""YSU (Yonsei University) PBL scheme: WRF v4.8.0 YSU fed by the WRF revised
MM5 surface layer.

:func:`ysu_turbulence` is the scheme every lane selects with
``turbulence: ysu``.  It runs, per step and in WRF's order:

1. :func:`~legoesm.atmosphere.physics.turbulence.sfclayrev_wrf.sfclayrev_run`
   -- JAX port of WRF v4.8.0 ``sf_sfclayrev`` (``sf_sfclay_physics = 1``),
   oracle-tested (``test_sfclayrev_oracle.py``), on its carried state;
2. :func:`bl_ysu_run` -- JAX port of WRF v4.8.0
   ``phys/physics_mmm/bl_ysu.F90::bl_ysu_run`` (WRF
   06d4240ae989cc3e50af412bb472df3d9048783c, physics_mmm
   550b5b494758ad2756394e93cb08b5709b8feb83) on its clear-sky path
   (``ysu_topdown_pblmix=.false.``, no cloud water/ice, no BEP urban canopy,
   ``ctopo`` absent, no ocean current), oracle-tested
   (``test_ysu_oracle.py``), called at the hard limit of its four sharpness
   arguments (``jnp.finfo(dtype).max``): the oracle's hard switches, a
   user-approved exception to the smooth-switch rule (R4, 2026-11-03).

YSU reads ``ust``, ``hfx``, ``qfx``, ``br``, ``psim = fm``, ``psih = fh``,
``wspd``, ``znt``, ``u10``, ``v10`` from the surface layer, as WRF's
``module_pbl_driver`` passes them.  The surface layer's sensible heat flux,
water flux and friction velocity are the fluxes this scheme publishes
(``TurbulenceOutput.shflx/lhflx/evap_sfc/ustar``).  Where the lane prescribes
the heat and water fluxes (``SurfaceLayerConfig.prescribed_*``), the surface
layer runs WRF's ``scm_force_flux = 1`` path and the prescribed values are used
verbatim; a prescribed stress is refused (WRF has no such path).

The surface layer is stateful, so the scheme carries ``PhysicsState.sfclay``
(ncol, 6) = ``[ust, mol, znt, hfx, qfx, pblh]`` (see :data:`SFCLAY_FIELDS`)
and returns the updated carry, like the TKE-carrying schemes.  The legacy
surrogate this module used to run on every lane (NOT WRF YSU; tier-2 audit:
PBL height 40 % median off the oracle) is deleted.

References
----------
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon.
  Wea. Rev., 134, 2318-2341.
- Jimenez, P. A., et al. (2012). A revised scheme for the WRF surface layer
  formulation. Mon. Wea. Rev., 140, 898-918.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.physics._shared import (
    exner_function,
    half_to_iface_zero_ends,
)
from legoesm.atmosphere.physics.turbulence.config import YSUConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.sfclayrev_wrf import (
    SfclayState,
    sfclayrev_run,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    surface_moisture_flux,
)


__physics_contract__ = {
    "summary": (
        "YSU PBL scheme: WRF v4.8.0 bl_ysu_run (clear-sky path, oracle hard "
        "switches) fed by the WRF v4.8.0 revised MM5 surface layer "
        "(sfclayrev), both oracle-tested; nonlocal K-profile with explicit "
        "PBL-top entrainment, implicit vertical diffusion over 2*dt as WRF."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg", "sfclay": "carry",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "shflx, lhflx, evap_sfc positive UPWARD (surface to atmosphere), "
        "injected as the lower boundary source of the heat / moisture solves "
        "(the column is NOT closed); surface drag implicit from ust. Km, Kh "
        ">= 0. z increases upward; level index 0 is the top, -1 the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "WRF v4.8.0 phys/physics_mmm/bl_ysu.F90 and sf_sfclayrev.F90; "
        "Hong, Noh & Dudhia (2006), Mon. Wea. Rev. 134, 2318-2341"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_ysu_oracle.py and "
        "test_sfclayrev_oracle.py (running-oracle match, column budgets); "
        "tests/atmosphere/hydrostatic/unit/test_ysu_route.py (sfclayrev -> "
        "YSU coupling equals the two oracle-tested kernels chained, carry, "
        "refusals)."
    ),
}


# ===========================================================================
# WRF YSU: JAX port of WRF v4.8.0 phys/physics_mmm/bl_ysu.F90::bl_ysu_run
# (WRF 06d4240ae989cc3e50af412bb472df3d9048783c, physics_mmm submodule
# 550b5b494758ad2756394e93cb08b5709b8feb83).  Line numbers below refer to
# that bl_ysu.F90.  The oracle builds with RWORDSIZE=8 (-r8), so every literal
# is fp64 as written (h1 = 0.33333335, not 1/3).
# ===========================================================================
# --- bl_ysu_run parameters (l.141-155) ---
_XKZMINM = 0.1          # background momentum diffusivity xkzom [m2/s]
_XKZMINH = 0.01         # background heat/moisture diffusivity xkzoh [m2/s]
_XKZMAX = 1000.0        # diffusivity cap [m2/s]
_RIMIN = -100.0         # lower bound on z/L and on the local Ri
_RLAM = 30.0            # minimum asymptotic mixing length [m]
_PRMIN = 0.25           # Prandtl-number bounds
_PRMAX = 4.0
_BRCR_UB = 0.0          # critical bulk Ri, unstable/neutral passes
_BRCR_SB = 0.25         # critical bulk Ri, stable surface over land
_CORI = 1.0e-4          # Coriolis parameter in the surface Rossby number [1/s]
_AFAC = 6.8             # thermal-excess coefficients
_BFAC = 6.8
_PFAC = 2.0             # K-profile exponents (momentum/heat and moisture)
_PFAC_Q = 2.0
_PHIFAC = 8.0           # convective coefficient in the velocity scale
_SFCFRAC = 0.1          # surface-layer fraction of the PBL depth
_D1 = 0.02              # entrainment-layer thickness coefficients
_D2 = 0.05
_D3 = 0.001
_H1 = 0.33333335        # cube-root exponent as written in the oracle
_H2 = 0.6666667
_ZFMIN = 1.0e-8         # floors on z/L and on the K-profile factor
_APHI5 = 5.0            # Businger-Dyer stability-function coefficients
_APHI16 = 16.0
_TMIN = 1.0e-2          # minimum PBL-top jumps (theta, theta_v, wind)
_GAMCRT = 3.0           # caps on the heat / moisture countergradient [K], [kg/kg]
_GAMCRQ = 2.0e-3
# --- literals inside bl_ysu_run ---
_SPDK2_MIN = 1.0        # max(u^2+v^2, 1.) in the bulk Ri (l.629)
_WSPD1_EPS = 1.0e-9     # added to the lowest-level wind speed (l.551)
_ENT_BFX_COEFF = 0.15   # bfxpbl = -0.15*thv1/g*wm3/h (l.837)
_WM3_UST_COEFF = 5.0    # wm3 = wstar3 + 5*ust^3 (l.835)
_HGAMU_COEFF = 15.9     # momentum countergradient (l.694) and prfac2 (l.949)
_PRFAC2_COEFF = 4.0     # 1 + 4*karman*wstar3/ust3 (l.949)
_PRNUMFAC_COEFF = 3.0   # prnumfac = -3*(...)^2/h^2 (l.950)
_PRPBL = 1.0            # momentum/heat entrainment ratio prpbl (l.910)
_DELTA_MAX = 100.0      # entrainment-layer thickness cap [m] (l.924)
_ROSS_A = 0.16          # brcr over water: min(0.16*(1e-7*Ro)^-0.18, 0.3) (l.787)
_ROSS_SCALE = 1.0e-7
_ROSS_EXP = -0.18
_BRCR_WATER_MAX = 0.3
_ENTFAC_MAX = 4.6       # entrainment layer: entfac < 4.6 (l.1059)
_SS_EPS = 1.0e-9        # added to the squared shear (l.989)
_RLAMDZ_FRAC = 0.1      # rlamdz = min(max(0.1*dz, rlam), 300, dz) (l.1004)
_RLAMDZ_MAX = 300.0
_KF_UNST = 8.0          # free-atmosphere unstable K (l.1012-1013)
_KF_UNST_M = 1.746
_KF_UNST_H = 1.286
_KF_STAB = 5.0          # free-atmosphere stable K and Prandtl number (l.1016-1017)
_KF_PR_STAB = 2.1
_WSCALEK_MIN = 0.001    # stable-surface velocity-scale floor (l.957)
_THX_OFFSET = 300.0     # theta offset of the heat solve (l.1045, l.1103)
_XLAND_WATER = 1.5      # water when xland - 1.5 >= 0 (l.783)


class YSUSurfaceLayer(NamedTuple):
    """Surface-layer state ``bl_ysu_run`` reads from WRF's surface-layer scheme.

    Every field has shape (ncol,).  Names and meanings are the oracle's
    argument names: ``psim``/``psih`` are what WRF passes under those names
    since 2012, the INTEGRATED stability functions ``ln(z1/z0) - psi_m`` and
    ``ln(z1/z0) - psi_h`` (sfclay's ``fm``/``fh``); YSU only uses them through
    ``z/L = br*psim^2/psih``.

    Fields
    ------
    br : surface-layer bulk Richardson number (sign sets stable/unstable).
    psim, psih : integrated stability functions (see above).
    wspd : surface-layer wind speed [m/s].
    znt : roughness length [m].
    u10, v10 : 10 m wind [m/s].
    xland : land/water flag (1 land, 2 water; water when ``xland >= 1.5``).
    """
    br: jax.Array
    psim: jax.Array
    psih: jax.Array
    wspd: jax.Array
    znt: jax.Array
    u10: jax.Array
    v10: jax.Array
    xland: jax.Array


class YSUOutput(NamedTuple):
    """Outputs of :func:`bl_ysu_run`, level index 0 = top (repo convention).

    ``exch_h``/``exch_m`` are the oracle's ``exch_hx``/``exch_mx``: entry k
    holds the diffusivity of the interface BELOW level k (the oracle's
    ``exch(k+1) = xkz(k)``), so the surface entry (index -1) is 0.
    ``kpbl`` is the oracle's 1-based level index counted from the surface.
    """
    du_dt: jax.Array
    dv_dt: jax.Array
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    hpbl: jax.Array
    kpbl: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array
    exch_q: jax.Array
    wstar: jax.Array
    delta: jax.Array
    pblflg: jax.Array


def _soft_gt0(x, sharpness, at_zero):
    """Smooth ``x > 0``: ``sigmoid(sharpness*x)``; at ``x == 0`` exactly the
    oracle's hard value ``at_zero`` (the sigmoid alone gives 1/2 there at any
    sharpness)."""
    return jnp.where(x == 0.0, at_zero, jax.nn.sigmoid(sharpness * x))


def _pbl_weights(zr, above, p_gt1, inv_gt1, region_sharpness):
    """In-PBL weight ``wp`` and its conditional on ``kpbl > 1`` (``in_pbl``)
    per interface, from ``zr = za/hpbl - 1``, ``above = P(kpbl <= k)``,
    ``p_gt1 = P(kpbl > 1)`` and ``inv_gt1 = 1/p_gt1`` (0 where p_gt1 is 0).

    Given kpbl > 1: k < kpbl at the exact tie ``zr == 0`` (P(kpbl > k) already
    implies kpbl > 1, hence the division and no second factor p_gt1), else the
    height sigmoid."""
    eq = zr == 0.0
    in_pbl = jnp.where(eq, (1.0 - above) * inv_gt1,
                       1.0 - jax.nn.sigmoid(region_sharpness * zr))
    wp = jnp.where(eq, 1.0 - above, p_gt1 * in_pbl)
    return wp, in_pbl


def bl_ysu_run(u, v, T, q_v, p_full, p_half, exner, dz, ust, hfx, qfx,
               sfc: YSUSurfaceLayer, dt, *, pbl_sharpness, region_sharpness,
               ri_sharpness, entrainment_cutoff_sharpness) -> YSUOutput:
    """WRF v4.8.0 ``bl_ysu_run`` (clear-sky path), batched over columns.

    Ported path: ``ysu_topdown_pblmix=.false.``, ``f_qc=f_qi=.false.`` (so the
    cloud-top entrainment and in-cloud Ri branches, which need condensate, are
    not reached), ``nmix=0``, no BEP, ``ctopo``/``ctopo2``/``uox``/``vox``
    absent.  Constants are ``legoesm.constants`` (cp=c_pd, g, rd=R_d,
    ep1=R_v/R_d-1, karman=kappa_von_karman).

    Arrays are (ncol, nlev) with level 0 = TOP (``p_half`` is (ncol, nlev+1));
    the port flips to the oracle's bottom-up order internally.  ``dz`` is the
    layer thickness (the oracle's ``dz8w``), ``exner`` its ``pi2d``, ``ust`` /
    ``hfx`` [W/m2] / ``qfx`` [kg/m2/s] the surface friction velocity and
    fluxes, ``q_v`` is passed where the oracle reads ``qvx``.

    Smoothing (each reduces to the oracle's hard switch as its sharpness
    goes to infinity; tested at 1e300):

    * ``pbl_sharpness``: the first-crossing selection of the bulk-Ri scans
      (oracle: first level with ``brup > brcr``; an exact tie is not a
      crossing at any sharpness); the in-pair interpolation
      is the oracle's ``brint`` exactly.
    * ``region_sharpness``: the in-PBL (``k < kpbl``) vs above-PBL selection of
      the diffusivities and of the in-PBL nonlocal fluxes, as
      ``1 - sigmoid(s*(za(k)/hpbl - 1))`` (``k < kpbl`` iff ``za(k) < hpbl``
      when ``kpbl > 1``, except at equality, where ``k < kpbl`` itself is used
      at any sharpness).
    * ``ri_sharpness``: the ``ri < 0`` unstable/stable selection of the
      free-atmosphere diffusivity, as ``sigmoid(s*ri)``; each branch is
      evaluated at ``min(ri,0)`` / ``max(ri,0)``, its value at ``ri = 0``,
      where both branches equal ``dk``.
    * The PBL-top level ``kpbl`` (#2231) is a distribution over levels 1..n:
      each pass's first-crossing weights (``pbl_sharpness``, the same weights
      that give hpbl), with the no-crossing mass on level n and the
      ``kpbl = 1`` rule (``h < zq(2)``) as ``sigmoid(region_sharpness*(h/zq(2)
      - 1))``.  The PBL flag is a weight in [0, 1]; ``a & b`` -> ``a*b``,
      ``where(f, A, B)`` -> ``f*A + (1-f)*B`` on hpbl and on the distribution
      (pass selections, the stable-surface ``hpbl < zq(2)`` test likewise).
      The entrainment-level picks apply the oracle's max/min/products per
      candidate level, then average over the distribution; "interface above
      kpbl" is the cumulative distribution; ``delta`` is the PBL weight times
      its PBL value, and ``entfac`` uses that PBL value everywhere.  The PBL
      flag implies ``kpbl > 1``, so what it multiplies (the level averages,
      the entrainment layer, the in-PBL nonlocal fluxes) is conditioned on
      ``kpbl > 1``.
    * ``entrainment_cutoff_sharpness``: the entrainment layer ``entfac < 4.6``
      as ``sigmoid(s*(4.6 - entfac))``.
    The hard ``kpbl`` and ``pblflg`` outputs are diagnostics computed as the
    oracle does (on the smooth hpbl), with the hard stable test they need;
    none of these feeds hpbl, delta, the diffusivities or the tendencies.
    The ``zfac`` denominator (``hpbl - zl1`` if ``kpbl > 1``, else 1): both
    branches' raw in-PBL diffusivities and ``zfacent`` are mixed by
    P(kpbl > 1), and the caps applied once after the mix.

    Hard switches (user-approved exception for the R1-R4 oracle routing,
    2026-11-03): :func:`ysu_turbulence` calls this kernel with all four
    sharpness arguments at ``jnp.finfo(dtype).max``, the oracle's hard limit;
    the remaining ``jnp.where`` / ``max`` / ``min`` / ``clip`` switches on
    traced state (``br > 0``, ``sflux > 0``, the zero-10 m-wind guard, the
    momentum entrainment thresholds, floors and caps, the Ri-sign branch
    arguments) are the oracle's as written.  Operands of unselected branches
    are replaced by benign values only so their gradients stay finite.
    """
    flip = lambda a: a[:, ::-1]  # noqa: E731  top-first -> oracle bottom-up
    ux, vx, tx, qx = flip(u), flip(v), flip(T), flip(q_v)
    p2d, pi2d, dz8w, p2di = flip(p_full), flip(exner), flip(dz), flip(p_half)
    g, cp, rd, karman = constants.g, constants.c_pd, constants.R_d, constants.kappa_von_karman
    ep1 = constants.R_v / constants.R_d - 1.0
    ncol, n = tx.shape
    dt2 = 2.0 * dt                       # l.560: implicit step over 2*dt
    rdt = 1.0 / dt2
    cont = cp / g

    # --- preliminaries (l.418-555) ---
    thx = tx / pi2d
    thvx = thx * (1.0 + ep1 * qx)
    rhox = p2di[:, 0] / (rd * tx[:, 0] * (1.0 + ep1 * qx[:, 0]))
    govrth = g / thx[:, 0]
    zq = jnp.concatenate([jnp.zeros_like(dz8w[:, :1]), jnp.cumsum(dz8w, axis=1)], axis=1)
    za = 0.5 * (zq[:, :-1] + zq[:, 1:])
    dele = p2di[:, :-1] - p2di[:, 1:]
    dza = jnp.concatenate([za[:, :1], za[:, 1:] - za[:, :-1]], axis=1)
    zl1 = za[:, 0]
    wspd1 = jnp.sqrt(ux[:, 0] * ux[:, 0] + vx[:, 0] * vx[:, 0]) + _WSPD1_EPS
    sflux = hfx / rhox / cp + qfx / rhox * ep1 * thx[:, 0]
    sfcflg = ~(sfc.br > 0.0)
    thermal1 = thvx[:, 0]

    spdk2 = jnp.maximum(ux[:, 1:] ** 2 + vx[:, 1:] ** 2, _SPDK2_MIN)
    gza = g * za[:, 1:] / thvx[:, :1]

    def scan(thermal, brcr):
        """One bulk-Ri pass (l.624-648): first level k >= 2 with brup > brcr,
        hpbl interpolated with the oracle's brint; kpbl = n if none.

        Returns hpbl, the hard kpbl (diagnostic), the smooth kpbl distribution
        P (ncol, n) over levels 1..n, and the smooth flag P(kpbl > 1)."""
        brcr = jnp.broadcast_to(brcr, (ncol,))[:, None]
        brup = (thvx[:, 1:] - thermal[:, None]) * gza / spdk2
        crossed = brup > brcr
        kpbl = jnp.where(jnp.any(crossed, axis=1),
                         jnp.argmax(crossed, axis=1) + 2, n)
        vals = jnp.concatenate([sfc.br[:, None], brup], axis=1)
        lo, hi = vals[:, :-1], vals[:, 1:]
        mid = (lo < brcr) & (hi > brcr)
        brint = jnp.where(lo >= brcr, 0.0, jnp.where(
            hi <= brcr, 1.0, (brcr - lo) / jnp.where(mid, hi - lo, 1.0)))
        h_pair = za[:, :-1] + brint * (za[:, 1:] - za[:, :-1])
        # an exact tie (brup == brcr, e.g. a layer of constant theta_v against
        # brcr = 0) is NOT a crossing, as the oracle's strict '>' has it; the
        # sigmoid alone would give it weight 1/2 at any sharpness
        w_c = jnp.where(brup == brcr, 0.0, jax.nn.sigmoid(pbl_sharpness * (brup - brcr)))
        not_yet = jnp.cumprod(jnp.concatenate(
            [jnp.ones((ncol, 1), tx.dtype), 1.0 - w_c[:, :-1]], axis=1), axis=1)
        # no crossing: the oracle interpolates in the top pair (kpbl = n)
        p_cross = not_yet * w_c                      # kpbl = 2..n
        p_none = jnp.prod(1.0 - w_c, axis=1)
        h = jnp.sum(p_cross * h_pair, axis=1) + p_none * h_pair[:, -1]
        # kpbl = 1 when h < zq(2) (l.648): smooth flag for h >= zq(2)
        f_gt1 = _soft_gt0(h / zq[:, 1] - 1.0, region_sharpness, 1.0)
        dist = jnp.concatenate([(1.0 - f_gt1)[:, None], f_gt1[:, None]
                                * p_cross.at[:, -1].add(p_none)], axis=1)
        return h, jnp.where(h < zq[:, 1], 1, kpbl), dist, f_gt1

    # --- first guess (l.617-648) ---
    # Hard kpbl / pblflg are kept as diagnostics (outputs); the computation
    # uses their smooth counterparts: dist (kpbl distribution) and pf
    # (pblflg weight), composed as a&b -> a*b, where(f, A, B) -> f*A+(1-f)*B.
    h_guess, kpbl, dist, f1 = scan(thermal1, _BRCR_UB)
    pblflg = kpbl > 1
    pf = f1

    # --- surface-layer scales from the first guess (l.650-678) ---
    zol1 = jnp.maximum(sfc.br * sfc.psim * sfc.psim / sfc.psih, _RIMIN)
    zol1 = jnp.where(sfcflg, jnp.minimum(zol1, -_ZFMIN), jnp.maximum(zol1, _ZFMIN))
    hol1 = zol1 * h_guess / zl1 * _SFCFRAC
    base_u = jnp.where(sfcflg, 1.0 - _APHI16 * hol1, 1.0)
    phim = jnp.where(sfcflg, base_u ** (-0.25), 1.0 + _APHI5 * hol1)
    phih = jnp.where(sfcflg, base_u ** (-0.5), 1.0 + _APHI5 * hol1)
    wstar3 = jnp.where(sfcflg, govrth * jnp.maximum(sflux, 0.0) * h_guess, 0.0)
    wstar = jnp.where(wstar3 > 0.0, jnp.where(wstar3 > 0.0, wstar3, 1.0) ** _H1, 0.0)
    ust3 = ust ** 3.0
    wscale = (ust3 + _PHIFAC * karman * wstar3 * 0.5) ** _H1
    wscale = jnp.maximum(jnp.minimum(wscale, ust * _APHI16), ust / _APHI5)

    # --- unstable surface: thermal excess, countergradients (l.683-700) ---
    unst = sfcflg & (sflux > 0.0)
    gamfac = _BFAC / rhox / wscale
    hgamt = jnp.minimum(gamfac * hfx / cp, _GAMCRT)
    hgamq = jnp.minimum(gamfac * qfx, _GAMCRQ)
    vpert = (hgamt + ep1 * thx[:, 0] * hgamq) / _BFAC * _AFAC
    thermal2 = thermal1 + jnp.maximum(vpert, 0.0) * jnp.minimum(
        za[:, 0] / (_SFCFRAC * h_guess), 1.0)
    hgamt = jnp.where(unst, jnp.maximum(hgamt, 0.0), 0.0)
    brint_u = (-_HGAMU_COEFF * ust * ust / jnp.where(unst, sfc.wspd, 1.0)
               * wstar3 / (wscale ** 4.0))
    hgamu = jnp.where(unst, brint_u * ux[:, 0], 0.0)
    hgamv = jnp.where(unst, brint_u * vx[:, 0], 0.0)
    pblflg = pblflg & unst
    pf = pf * jnp.where(unst, 1.0, 0.0)

    # --- second pass with the thermal (l.704-768, topdown off) ---
    h2, k2, dist2, f2 = scan(thermal2, _BRCR_UB)
    hpbl_h = jnp.where(pblflg, h2, h_guess)
    hpbl = pf * h2 + (1.0 - pf) * h_guess
    dist = pf[:, None] * dist2 + (1.0 - pf)[:, None] * dist
    kpbl = jnp.where(pblflg, k2, kpbl)
    pblflg = pblflg & (k2 > 1)
    pf = pf * f2

    # --- stable boundary layer (l.770-822) ---
    stab = (~sfcflg) & (hpbl_h < zq[:, 1])
    st = jnp.where(sfcflg, 0.0, 1.0) * (
        1.0 - _soft_gt0(hpbl / zq[:, 1] - 1.0, region_sharpness, 1.0))
    water = (sfc.xland - _XLAND_WATER) >= 0.0
    s10 = sfc.u10 * sfc.u10 + sfc.v10 * sfc.v10
    # the Rossby brcr is pass 3's whenever over water; st weighs pass 3 in
    ok = water & (s10 > 0.0)
    ross = jnp.sqrt(jnp.where(ok, s10, 1.0)) / (_CORI * sfc.znt)
    # zero 10 m wind: (1e-7*ross)**(-0.18) is +inf and min() returns 0.3
    brcr_sbro = jnp.minimum(jnp.where(
        ok, _ROSS_A * (_ROSS_SCALE * jnp.where(ok, ross, 1.0)) ** _ROSS_EXP, jnp.inf),
        _BRCR_WATER_MAX)
    h3, k3, dist3, f3 = scan(thermal1, jnp.where(water, brcr_sbro, _BRCR_SB))
    hpbl = st * h3 + (1.0 - st) * hpbl
    dist = st[:, None] * dist3 + (1.0 - st)[:, None] * dist
    kpbl = jnp.where(stab, k3, kpbl)
    pblflg = pblflg & ~(stab & (k3 <= 1))
    pf = pf * (1.0 - st * (1.0 - f3))

    # --- entrainment parameters, clear-sky branch (l.827-926) ---
    # The oracle picks the jumps at k = kpbl-1 (0-based kb = kpbl-2) and then
    # applies its max/min/products: done here per candidate level L = 1..n
    # (kb = clip(L-2, 0, n-2)), then averaged over the kpbl distribution.
    # pblflg implies kpbl > 1, and pf already carries that probability, so
    # everything that multiplies pf (these averages, the entrainment-layer and
    # in-PBL nonlocal weights) is conditioned on kpbl > 1.
    p_gt1 = 1.0 - dist[:, :1]
    inv_gt1 = jnp.where(p_gt1 > 0.0, 1.0 / jnp.where(p_gt1 > 0.0, p_gt1, 1.0), 0.0)
    lev = jnp.arange(1, n + 1)
    kbl = jnp.clip(lev - 2, 0, n - 2)
    jump = lambda a: a[:, kbl + 1] - a[:, kbl]  # noqa: E731
    expect = lambda a: jnp.sum(dist[:, 1:] * a[:, 1:], axis=1) * inv_gt1[:, 0]  # noqa: E731
    wm3 = wstar3 + _WM3_UST_COEFF * ust3
    wm2 = wm3 ** _H2
    bfxpbl = -_ENT_BFX_COEFF * thvx[:, 0] / g * wm3 / hpbl
    dthvx = jnp.maximum(jump(thvx), _TMIN)
    we = jnp.maximum(bfxpbl[:, None] / dthvx, -jnp.sqrt(wm2)[:, None])
    dthx = jnp.maximum(jump(thx), _TMIN)
    dqx = jnp.minimum(jump(qx), 0.0)
    hfxpbl = expect(we * dthx)
    qfxpbl = expect(we * dqx)

    def mom_ent_flux(d):
        return jnp.where(d > _TMIN, jnp.maximum(_PRPBL * we * d, -ust[:, None] * ust[:, None]),
                         jnp.where(d < -_TMIN, jnp.minimum(_PRPBL * we * d,
                                                           ust[:, None] * ust[:, None]),
                                   0.0))
    ufxpbl = expect(mom_ent_flux(jump(ux)))
    vfxpbl = expect(mom_ent_flux(jump(vx)))
    we_dza = expect(we * dza[:, jnp.clip(lev - 1, 0, n - 1)])
    delb = govrth * _D3 * hpbl
    delta_full = jnp.minimum(_D1 * hpbl + _D2 * wm2 / delb, _DELTA_MAX)
    delta = pf * delta_full

    # interface k (1..n-1, Fortran) between levels k and k+1 sits at zq(k+1)
    zqi = zq[:, 1:n]
    above = jnp.cumsum(dist, axis=1)[:, :n - 1]    # P(kpbl <= k)
    entfac = ((zqi - hpbl[:, None]) / delta_full[:, None]) ** 2.0

    # --- diffusivities below the PBL top, k < kpbl (l.937-980) ---
    sf = sfcflg[:, None]
    # The oracle's zfac denominator switches on kpbl > 1 (hpbl - zl1, else 1):
    # both branches are evaluated and their raw (uncapped) diffusivities and
    # zfacent mixed by P(kpbl > 1), then capped once (user pick 2026-10-29).
    # hpbl - zl1 <= 0 (unreachable at the hard limit: kpbl > 1 gives
    # hpbl >= zq(2) > zl1) uses 1.0, i.e. branch B there.
    prfac = jnp.where(sfcflg, _BFAC * karman * _SFCFRAC, 0.0)[:, None]
    prfac2 = jnp.where(sfcflg, _HGAMU_COEFF * wstar3 / ust3 / (
        1.0 + _PRFAC2_COEFF * karman * wstar3 / ust3), 0.0)[:, None]
    prnumfac = jnp.where(sf, -_PRNUMFAC_COEFF * jnp.maximum(
        zqi - _SFCFRAC * hpbl[:, None], 0.0) ** 2.0 / hpbl[:, None] ** 2.0, 0.0)
    phim8z = jnp.where(sf, 1.0, 1.0 + _APHI5 * zol1[:, None] * zqi / zl1[:, None])
    prnum0 = jnp.clip((phih / phim)[:, None] + prfac, _PRMIN, _PRMAX)
    prnum0h = prnum0 / (1.0 + prfac2 * karman * _SFCFRAC)

    def pbl_k(den):
        """Raw in-PBL km, kh, kq and zfacent for zfac denominator ``den``."""
        zfac = jnp.clip(1.0 - (zqi - zl1[:, None]) / den[:, None], _ZFMIN, 1.0)
        wscalek = (ust3[:, None] + _PHIFAC * karman * wstar3[:, None]
                   * (1.0 - zfac)) ** _H1
        # wscalek2 = (phifac*karman*wstar3_2*zfac)**h1 is 0: wstar3_2 is set
        # only on the cloud-top branch, which needs condensate (not ported).
        wscalek = jnp.where(sf, wscalek, jnp.maximum(ust[:, None] / phim8z, _WSCALEK_MIN))
        km = wscalek * karman * zqi * zfac ** _PFAC
        kq = km / (1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)) * zfac ** (_PFAC_Q - _PFAC)
        kh = km / (1.0 + (prnum0h - 1.0) * jnp.exp(prnumfac))
        return km, kh, kq, (1.0 - zfac) ** 3.0

    d_a = hpbl - zl1
    k_a = pbl_k(jnp.where(d_a > 0.0, d_a, 1.0))
    k_b = pbl_k(jnp.ones_like(hpbl))
    km_p, kh_p, kq_p, zfacent = (p_gt1 * a + (1.0 - p_gt1) * b for a, b in zip(k_a, k_b))
    km_p = jnp.minimum(km_p + _XKZMINM, _XKZMAX)
    kh_p = jnp.minimum(kh_p + _XKZMINH, _XKZMAX)
    kq_p = jnp.minimum(kq_p + _XKZMINH, _XKZMAX)

    # --- free atmosphere, k >= kpbl (l.984-1028; dry ri, imvdif in-cloud
    #     branch needs condensate) ---
    du = ux[:, 1:] - ux[:, :-1]
    dv = vx[:, 1:] - vx[:, :-1]
    dzi = dza[:, 1:]
    ss = (du * du + dv * dv) / (dzi * dzi) + _SS_EPS
    govrthv = g / (0.5 * (thvx[:, 1:] + thvx[:, :-1]))
    ri = govrthv * (thvx[:, 1:] - thvx[:, :-1]) / (ss * dzi)
    zk = karman * zqi
    rlamdz = jnp.minimum(dzi, jnp.minimum(jnp.maximum(_RLAMDZ_FRAC * dzi, _RLAM), _RLAMDZ_MAX))
    dk = (zk * rlamdz / (rlamdz + zk)) ** 2 * jnp.sqrt(ss)
    neg = ri < 0.0
    ri_u = jnp.where(neg, jnp.maximum(ri, _RIMIN), -1.0)
    sri = jnp.sqrt(-ri_u)
    km_u = jnp.where(neg, dk * (1.0 + _KF_UNST * (-ri_u) / (1.0 + _KF_UNST_M * sri)), dk)
    kh_u = jnp.where(neg, dk * (1.0 + _KF_UNST * (-ri_u) / (1.0 + _KF_UNST_H * sri)), dk)
    ri_s = jnp.maximum(ri, 0.0)
    kh_s = dk / (1.0 + _KF_STAB * ri_s) ** 2
    km_s = kh_s * jnp.minimum(1.0 + _KF_PR_STAB * ri_s, _PRMAX)
    w_st = jax.nn.sigmoid(ri_sharpness * ri)
    km_f = jnp.minimum((1.0 - w_st) * km_u + w_st * km_s + _XKZMINM, _XKZMAX)
    kh_f = jnp.minimum((1.0 - w_st) * kh_u + w_st * kh_s + _XKZMINH, _XKZMAX)

    # --- entrainment layer above the PBL top (l.1059-1063, l.1339-1342) ---
    # pf times P(2 <= kpbl <= k | kpbl > 1) (a product, not one joint expectation)
    ent = pf[:, None] * (above - dist[:, :1]) * inv_gt1 * _soft_gt0(
        _ENTFAC_MAX - entfac, entrainment_cutoff_sharpness, 0.0)
    # >= 0 (we <= 0); ke2 == 0 gives sqrt(0) -> xkzminh as in the oracle (do
    # not substitute the old benign 1.0 there: it would break the hard limit)
    ke2 = -we_dza[:, None] * jnp.exp(-entfac) * kh_f
    kh_e = jnp.clip(jnp.where(ke2 > 0.0, jnp.sqrt(jnp.where(ke2 > 0.0, ke2, 1.0)), 0.0),
                    _XKZMINH, _XKZMAX)
    km_e = jnp.clip(jnp.sqrt(_PRPBL * kh_e * km_f), _XKZMINM, _XKZMAX)

    # --- PBL / above-PBL selection ---
    # za(k) == hpbl (hpbl exactly on level kpbl-1 after an exact tie, or on
    # level kpbl when brint rounds to 1): the sigmoid would give 1/2 at any
    # sharpness, so equality takes the oracle's k < kpbl
    zr = za[:, :-1] / hpbl[:, None] - 1.0
    wp, in_pbl = _pbl_weights(zr, above, p_gt1, inv_gt1, region_sharpness)
    kh_a = ent * kh_e + (1.0 - ent) * kh_f
    Kh = wp * kh_p + (1.0 - wp) * kh_a
    Kq = wp * kq_p + (1.0 - wp) * kh_a
    Km = wp * km_p + (1.0 - wp) * (ent * km_e + (1.0 - ent) * km_f)
    fw = pf[:, None] * in_pbl     # in-PBL nonlocal fluxes (conditioned as ent)

    # --- implicit solves over 2*dt (l.1031-1405) ---
    dsig = p2d[:, :-1] - p2d[:, 1:]
    rdz = 1.0 / dzi
    dtodsd = dt2 / dele[:, :-1]
    dtodsu = dt2 / dele[:, 1:]
    zero = jnp.zeros((ncol, 1), tx.dtype)
    h_ = hpbl[:, None]

    def solve(base, K, dsdz, extra0):
        """Rows: f(k) += dtodsd*dsdz(k), f(k+1) -= dtodsu*dsdz(k); matrix
        au = -dtodsd*dsig*K*rdz^2, al = -dtodsu*(...), ad = 1-al(k-1)-au(k)."""
        rhs = base + jnp.concatenate([dtodsd * dsdz, zero], axis=1) \
            - jnp.concatenate([zero, dtodsu * dsdz], axis=1)
        dsdz2 = dsig * K * rdz * rdz
        au = jnp.concatenate([-dtodsd * dsdz2, zero], axis=1)
        al = jnp.concatenate([zero, -dtodsu * dsdz2], axis=1)
        diag = (1.0 - al - au).at[:, 0].add(extra0)
        return thomas_solve_batched(al, diag, au, rhs)

    th_base = (thx - _THX_OFFSET).at[:, 0].add(hfx / cont / dele[:, 0] * dt2)
    dsdzt = fw * dsig * Kh * rdz * (-hgamt[:, None] / h_ - hfxpbl[:, None] * zfacent / Kh)
    th_new = solve(th_base, Kh, dsdzt, 0.0)
    ttend = (th_new - thx + _THX_OFFSET) * rdt * pi2d

    q_base = qx.at[:, 0].add(qfx * g / dele[:, 0] * dt2)
    dsdzq = fw * dsig * Kq * rdz * (-qfxpbl[:, None] * zfacent / Kq)
    qtend = (solve(q_base, Kq, dsdzq, 0.0) - qx) * rdt

    fric = ust ** 2 / wspd1 * rhox * g / dele[:, 0] * dt2 * (wspd1 / sfc.wspd) ** 2
    dsdzu = fw * dsig * Km * rdz * (-hgamu[:, None] / h_ - ufxpbl[:, None] * zfacent / Km)
    dsdzv = fw * dsig * Km * rdz * (-hgamv[:, None] / h_ - vfxpbl[:, None] * zfacent / Km)
    utend = (solve(ux, Km, dsdzu, fric) - ux) * rdt
    vtend = (solve(vx, Km, dsdzv, fric) - vx) * rdt

    exch = lambda K: flip(jnp.concatenate([zero, K], axis=1))  # noqa: E731
    return YSUOutput(
        du_dt=flip(utend), dv_dt=flip(vtend), dT_dt=flip(ttend), dq_v_dt=flip(qtend),
        hpbl=hpbl, kpbl=kpbl, exch_h=exch(Kh), exch_m=exch(Km), exch_q=exch(Kq),
        wstar=wstar, delta=delta, pblflg=pblflg)




# ===========================================================================
# ysu_turbulence: sfclayrev -> bl_ysu_run, the scheme every lane runs
# ===========================================================================
# Carried surface-layer state, PhysicsState.sfclay[:, i] (R4 vote J3).
SFCLAY_FIELDS = ("ust", "mol", "znt", "hfx", "qfx", "pblh")
# WRF cold start (module_physics_init.F l.1263-1267): ust = 0.0001, mol = pblh
# = hfx = qfx = 0; znt is the lane's roughness (vote X6/X9).
_UST_COLD_START = 1.0e-4
_XLAND_LAND_FLAG = 1.0      # WRF xland: 1 land, 2 water
_XLAND_WATER_FLAG = 2.0
_LAND_FRACTION_LAND = 0.5   # xland = 1 where land fraction >= 0.5 (vote X7)
_SURFACE_TYPES = {"land": _XLAND_LAND_FLAG, "water": _XLAND_WATER_FLAG}


def init_sfclay_carry(ncol: int, config: YSUConfig, dtype=jnp.float64) -> jax.Array:
    """WRF cold-start surface-layer carry, shape (ncol, len(SFCLAY_FIELDS))."""
    seed = {"ust": _UST_COLD_START, "znt": config.surface.z0}
    return jnp.stack([jnp.full((ncol,), seed.get(k, 0.0), dtype=dtype)
                      for k in SFCLAY_FIELDS], axis=1)


def _land_water_flag(config: YSUConfig, land_fraction, ncol, dtype):
    if land_fraction is not None:
        if config.surface_type is not None:
            raise ValueError(
                "YSUConfig.surface_type is set but this lane supplies a land "
                "fraction, which sets xland; the field would be inert. Unset it.")
        lf = jnp.reshape(land_fraction, (ncol,)).astype(dtype)
        return jnp.where(lf >= _LAND_FRACTION_LAND, _XLAND_LAND_FLAG, _XLAND_WATER_FLAG)
    if config.surface_type not in _SURFACE_TYPES:
        raise ValueError(
            "turbulence=ysu needs the land/water flag of the WRF surface layer: "
            "this lane has no land fraction, so set YSUConfig.surface_type to "
            f"one of {sorted(_SURFACE_TYPES)} (got {config.surface_type!r}).")
    return jnp.full((ncol,), _SURFACE_TYPES[config.surface_type], dtype=dtype)


def _grid_spacing(config: YSUConfig, cell_area, ncol, dtype):
    if cell_area is not None:
        if config.dx_m is not None:
            raise ValueError(
                "YSUConfig.dx_m is set but this lane supplies cell areas, which "
                "set the grid spacing; the field would be inert. Unset it.")
        return jnp.sqrt(jnp.reshape(cell_area, (ncol,)).astype(dtype))
    if config.dx_m is None:
        raise ValueError(
            "turbulence=ysu needs the grid spacing of the WRF surface layer "
            "(Mahrt-Sun subgrid wind): this lane has no cell areas, so set "
            "YSUConfig.dx_m [m].")
    return jnp.full((ncol,), config.dx_m, dtype=dtype)


def _forced_fluxes(surf, T_sfc):
    """(forced, hfx, lhflx, qfx) from SurfaceLayerConfig.prescribed_*.

    WRF's scm_force_flux keeps BOTH hfx and qfx, so heat and latent heat must
    both be prescribed; a prescribed stress has no WRF path (vote X4)."""
    if surf.prescribed_tau_x_pa is not None or surf.prescribed_tau_y_pa is not None:
        raise ValueError(
            "turbulence=ysu takes its friction velocity from the WRF surface "
            "layer; a prescribed surface stress has no counterpart there and is "
            "refused (R4 vote X4).")
    sh, lh, ev = (surf.prescribed_shflx_w_m2, surf.prescribed_lhflx_w_m2,
                  surf.prescribed_evap_kg_m2_s)
    if sh is None and lh is None and ev is None:
        return False, None, None, None
    if sh is None or lh is None:
        raise ValueError(
            "turbulence=ysu with prescribed surface fluxes runs WRF's "
            "scm_force_flux path, which keeps BOTH the heat and the water flux: "
            "prescribe both prescribed_shflx_w_m2 and prescribed_lhflx_w_m2.")
    qfx = ev if ev is not None else surface_moisture_flux(surf, lh, T_sfc)
    return True, sh, lh, qfx


def ysu_turbulence(u, v, T, q_v, sfclay, p_full, p_half, z_full, z_half, T_sfc,
                   q_sfc, rho, dt, config: YSUConfig, *, land_fraction=None,
                   cell_area=None):
    """WRF surface layer + WRF YSU for one step; returns (TurbulenceOutput,
    sfclay_new).

    Arrays as every turbulence kernel: (ncol, nlev) with level 0 = TOP,
    ``p_half``/``z_half`` (ncol, nlev+1); ``T_sfc`` [K] is the surface layer's
    skin temperature (TSK) and ``q_sfc`` its land surface humidity (over water
    the surface layer computes its own, vote X10); moisture availability is 1
    (vote X11).  ``sfclay`` is the carry (ncol, 6) in :data:`SFCLAY_FIELDS`
    order (seed :func:`init_sfclay_carry`).  ``land_fraction`` / ``cell_area``
    (ncol,) come from the lane; without them ``config.surface_type`` /
    ``config.dx_m`` must be set.  ``z_full`` and ``rho`` are not read: the
    oracle's layer geometry comes from ``z_half`` and its density from the
    surface pressure.
    """
    del z_full, rho
    ncol = T.shape[0]
    want = (ncol, len(SFCLAY_FIELDS))
    if sfclay is None or tuple(sfclay.shape) != want:
        raise ValueError(
            f"turbulence=ysu carries the WRF surface-layer state "
            f"PhysicsState.sfclay of shape {want}; got "
            f"{None if sfclay is None else tuple(sfclay.shape)}. Seed it with "
            "init_physics_state and thread the returned carry (no silent reseed).")
    dtype = T.dtype
    surf = config.surface
    forced, sh, lh, qf = _forced_fluxes(surf, T_sfc)
    xland = _land_water_flag(config, land_fraction, ncol, dtype)
    dx = _grid_spacing(config, cell_area, ncol, dtype)
    c = {k: sfclay[:, i] for i, k in enumerate(SFCLAY_FIELDS)}
    water = xland >= _XLAND_WATER_FLAG
    state = SfclayState(
        ust=c["ust"], mol=c["mol"],
        znt=jnp.where(water, c["znt"], jnp.asarray(surf.z0, dtype)),
        hfx=jnp.broadcast_to(sh, (ncol,)).astype(dtype) if forced else c["hfx"],
        qfx=jnp.broadcast_to(qf, (ncol,)).astype(dtype) if forced else c["qfx"],
        qsfc=q_sfc)
    dz = z_half[:, :-1] - z_half[:, 1:]
    s = sfclayrev_run(u[:, -1], v[:, -1], T[:, -1], q_v[:, -1], p_full[:, -1], dz[:, -1],
                      p_half[:, -1], T_sfc, xland, jnp.ones((ncol,), dtype), c["pblh"], dx,
                      state, scm_force_flux=forced)
    hard = float(jnp.finfo(dtype).max)
    o = bl_ysu_run(
        u, v, T, q_v, p_full, p_half, exner_function(p_full), dz, s.ust, s.hfx, s.qfx,
        YSUSurfaceLayer(br=s.br, psim=s.fm, psih=s.fh, wspd=s.wspd, znt=s.znt,
                        u10=s.u10, v10=s.v10, xland=xland), dt,
        pbl_sharpness=hard, region_sharpness=hard, ri_sharpness=hard,
        entrainment_cutoff_sharpness=hard)
    lhflx = jnp.broadcast_to(lh, (ncol,)).astype(dtype) if forced else s.lh
    out = TurbulenceOutput(
        du_dt=o.du_dt, dv_dt=o.dv_dt, dT_dt=o.dT_dt, dq_v_dt=o.dq_v_dt,
        Km=o.exch_m, Kh=o.exch_h, shflx=s.hfx, lhflx=lhflx, evap_sfc=s.qfx,
        ustar=s.ust, h_pbl=o.hpbl,
        Kh_iface=half_to_iface_zero_ends(o.exch_h[:, :-1]))
    new = {"ust": s.ust, "mol": s.mol, "znt": s.znt, "hfx": s.hfx, "qfx": s.qfx,
           "pblh": o.hpbl}
    return out, jnp.stack([new[k] for k in SFCLAY_FIELDS], axis=1)
