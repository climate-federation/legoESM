"""CAM6 Zhang-McFarlane deep convection: faithful JAX port of ``zm_conv.F90``.

Oracle: CESM2.1 ``cam_cesm2_1_rel_60`` ``src/physics/cam/zm_conv.F90``
(subroutines ``zm_convr``, ``buoyan_dilute``/``parcel_dilute``, ``cldprp``,
``closure``, ``q1q2_pjr``, ``zm_conv_evap`` + ``cldfrc_fice``, ``momtran``,
``convtran``) with the CAM6 defaults ``zmconv_microp = .false.`` and
``zm_org = .false.``: every ``zmconv_microp`` / ``zm_org`` branch of the
Fortran is dead in the CAM6 configuration and is NOT ported.

Every routine is vectorised over columns: the Fortran's gathered ``ideep``
subset becomes a per-column mask (``cape > capelmt``), level loops become
``lax.scan`` (recursions) or ``jnp.where`` over the Fortran 1-based level
number ``k`` (pointwise loops), and integer level indices (``mx``, ``lcl``,
``lel``, ``jt``, ``j0``, ``jd``, ``jlcl``, ``limcnv``) are carried as
1-based ``int32`` arrays exactly as in the Fortran so the code reads
against the oracle line by line.  Internally the Fortran units are kept:
pressures in hPa (``p = pap*0.01``), mass fluxes in hPa/s, ``dz`` in m.

Departures from the oracle (all documented, measured where numeric):

* ``ientropy`` Brent inversion (data-dependent iteration count) is replaced
  by the fixed-iteration Newton solve of ``_zm_dilute.invert_entropy``
  (converges to the same root; AD-safe).
* Saturation: CAM ``qsat_water`` (Goff-Gratch) / ``qsat`` (mixed phase) are
  replaced by the shared ``legoesm.thermo.saturation_mixing_ratio``
  (Tetens, smooth cap) per CLAUDE.md; the curve difference is measured in
  ``tests/atmosphere/hydrostatic/unit/test_zm_cam6_oracle.py``.
* Constants come from ``legoesm.constants`` (CAM's liquid/vapour heat
  capacities, dry gas constant, epsilon and vapour gas constant differ in
  the third or fourth significant figure).
* Divisions the Fortran performs only on gathered deep columns are guarded
  (``_safe_div``) so non-deep columns stay finite under reverse-mode AD;
  the guarded branch is never selected on a deep column.
* The Fortran ``pblt`` needs ``pblh``; when the host cannot supply it the
  launch-level search is bounded by ``pbl_top_pa`` instead (see
  ``zm_convr``).
* ``limcnv``: CAM fixes the 40 hPa cap from the REFERENCE interface
  pressures once at init; hosts without a reference vertical coordinate
  (``pref_edge=None``) get it from each column's own interfaces instead.
* ``closure``: a launch-level vapour pressure ``eb <= 0`` is ``log(0)`` in
  the Fortran; it stays NaN in a deep column and is 0 in a non-deep one
  (masked anyway; keeps the reverse-mode pass finite).
* ``cldprp``: when no level satisfies ``hsat <= hmin`` the Fortran leaves
  ``j0`` uninitialised; the port takes ``jt + 2`` (then the same clamps).
* ``buoyan_dilute``: the CAPE integral floors the interface pressure at
  1e-2 hPa (a 0 Pa model top is ``log(0)`` in the Fortran).
* The downdraft recursions' ``num / min(md, -1e-20)`` is evaluated as
  ``num * (-1e20)`` on the floor branch (same value, finite float32 VJP).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_vapor_pressure
from legoesm.atmosphere.physics.convection._zm_dilute import (
    invert_entropy,
    moist_entropy,
)

__all__ = (
    "ZMConvr",
    "ZMEvap",
    "ZMMomtran",
    "buoyan_dilute",
    "cldfrc_fice",
    "cldprp",
    "closure",
    "convtran",
    "momtran",
    "q1q2_pjr",
    "zm_conv_evap",
    "zm_convr",
)

__physics_contract__ = {
    "summary": (
        "CAM6 Zhang-McFarlane deep convection kernels (zm_conv.F90 port): "
        "dilute-parcel CAPE, cloud model (cldprp), quasi-equilibrium closure, "
        "q1q2 tendencies, rain evaporation, momentum and tracer transport."
    ),
    "inputs": {
        "T": "K", "q": "kg/kg", "p_full": "Pa", "p_half": "Pa", "z": "m",
        "zf": "m", "u": "m/s", "v": "m/s", "land_frac": "1", "dt": "s",
    },
    "outputs": {
        "dqdt": "kg/kg/s", "heat": "J/kg/s", "dlf": "kg/kg/s",
        "rprd": "kg/kg/s", "prec": "kg/m^2/s", "mu": "hPa/s", "md": "hPa/s",
        "dudt": "m/s^2", "dvdt": "m/s^2", "seten": "J/kg/s",
    },
    "sign_convention": (
        "Surface at [:, -1]; Fortran level k = python index k-1. Updraft mass "
        "flux mu >= 0, downdraft md <= 0 (hPa/s, normalised by mb before "
        "scaling). heat > 0 warms; dqdt > 0 moistens; rprd > 0 is rain "
        "production (net of downdraft evaporation)."
    ),
    # moisture: prec = -sum dp (dqdt + dlf)/g to rounding; momentum: momtran
    # is flux-form with zero flux at the cloud top and the surface.
    "conserves": ["moisture", "momentum"],
    "differentiable": True,
    "reference": (
        "Zhang & McFarlane (1995) Atmos.-Ocean 33; Neale et al. (2008) dilute "
        "plume; Richter & Rasch (2008) momentum transport; CESM2.1 zm_conv.F90"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_zm_cam6_oracle.py pins every "
        "routine against a Python transcription of the Fortran (rel 1e-10) and "
        "closes the column water budget."
    ),
}

# --- oracle module parameters (zm_conv.F90 line refs) ------------------------
_MU_MIN = 0.02                 # updraft cut-off, zm_conv.F90:3221/3253
_HU_TOP_JUMP = -2000.0         # hu-hsthat < -2000 J/kg top rule, zm_conv.F90:3255
_EXPDIF_MIN = 100.0            # expdif > 100 J/kg gate, zm_conv.F90:3157
_F_MAX = 0.0002                # f <= 2e-4 1/m, zm_conv.F90:3165
_F_TINY = 1.0e-6               # f(j0) < 1e-6 bump, zm_conv.F90:3171
_LOGMEAN_TOL = 1.0e-6          # interface log-mean switch, zm_conv.F90:892/3098
_MD_SMALL = 1.0e-20            # small, zm_conv.F90:3446
# CAM floors water vapour at this value after every parameterization
# (qneg3 in physics_update, physics_types.F90:340; cnst_add('Q', ..., 1.E-12)
# in physpkg.F90:178), so zm_convr never sees q = 0: the interface log-mean
# log(q(k-1)/q(k)) (zm_conv.F90:900) would be NaN on an exact zero.
Q_MIN_VAPOR = 1.0e-12
_HMIN_INIT = 1.0e6             # hmin init, zm_conv.F90:3049
_MBSTH = 1.0e-15               # convtran/momtran mass-flux threshold, zm_conv.F90:1754/2093
_CONVTRAN_SMALL = 1.0e-36      # convtran small, zm_conv.F90:1752
_CHAT_FLOOR_FRAC = 1.0e-12     # cabv/cbel floor, zm_conv.F90:1811
_NETFLUX_TOL = 1.0e-12         # netflux round-off zeroing, zm_conv.F90:1880
_VIRT_1608 = 1.608             # 1 + 1.608 q virtual factor, zm_conv.F90:4090
_VIRT_0608 = 0.608             # 1 + 0.608 q virtual factor, zm_conv.F90:2894
_P_LCL_MAX_HPA = 600.0         # plge600: no deep convection if LCL above 600 hPa, zm_conv.F90:4152
# Bolton (1980) LCL-temperature derivative constants, closure(), zm_conv.F90:3708
_BOLTON_A = 2840.0
_BOLTON_B = 3.5
_BOLTON_C = 4.805
# cldfrc_fice (cloud_fraction.F90:768-771)
_FICE_TMAX_OFFSET = 10.0
_FICE_RANGE = 30.0
_FSNOW_RANGE = 5.0
# sqrt(flxprec) floor so the AD of sqrt at exactly zero flux stays finite; the
# evaporation limiter selects 0 there anyway (zm_conv_evap).
_FLX_SQRT_FLOOR = 1.0e-30


# ---------------------------------------------------------------------------
# small vectorisation helpers
# ---------------------------------------------------------------------------
def _lev(nlev):
    """Fortran level numbers ``k = 1..pver`` as a ``(1, nlev)`` int32 row."""
    return jnp.arange(1, nlev + 1, dtype=jnp.int32)[None, :]


def _at(a, k):
    """``a(i, k(i))`` for a 1-based per-column level index ``k`` (clipped)."""
    idx = jnp.clip(k - 1, 0, a.shape[1] - 1)
    return jnp.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def _kp1(a):
    """Value at level ``k+1`` (zero past the last level)."""
    return jnp.concatenate([a[:, 1:], jnp.zeros_like(a[:, :1])], axis=1)


def _km1(a):
    """Value at level ``k-1`` (zero above the first level)."""
    return jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)


def _safe_div(num, den, cond):
    """``num/den`` where ``cond`` else 0, with the division itself guarded."""
    den_safe = jnp.where(cond, den, jnp.ones_like(den))
    return jnp.where(cond, num / den_safe, jnp.zeros_like(num))


def _log_mean(a_km1, a_k, diff_gate):
    """Fortran interface log-mean ``log(a(k-1)/a(k))*a(k-1)*a(k)/(a(k-1)-a(k))``."""
    den = a_km1 - a_k
    num_safe = jnp.where(diff_gate, a_km1, jnp.ones_like(a_km1))
    den_k = jnp.where(diff_gate, a_k, jnp.ones_like(a_k))
    den_safe = jnp.where(diff_gate, den, jnp.ones_like(den))
    return jnp.log(num_safe / den_k) * num_safe * den_k / den_safe


def _scan_levels(body, init, xs, *, descending):
    """Run ``body(carry, (k, *x_k)) -> (carry, outs)`` over Fortran levels.

    ``xs`` are ``(ncol, nlev)`` arrays; ``k`` is the 1-based level number
    (int32 scalar).  Outputs come back as ``(ncol, nlev)`` in level order.
    """
    nlev = xs[0].shape[1]
    ks = jnp.arange(1, nlev + 1, dtype=jnp.int32)
    stacked = tuple(jnp.moveaxis(x, 1, 0) for x in xs)
    if descending:
        ks = ks[::-1]
        stacked = tuple(s[::-1] for s in stacked)
    carry, outs = jax.lax.scan(body, init, (ks,) + stacked)

    def back(o):
        o = o[::-1] if descending else o
        return jnp.moveaxis(o, 0, 1)

    return carry, jax.tree_util.tree_map(back, outs)


def _qsat_hpa(t, p_hpa):
    """Oracle ``qsat_hPa``: saturation mixing ratio at ``p`` in hPa."""
    return saturation_mixing_ratio(t, p_hpa * 100.0)


def _qst_cldprp(t, p_hpa):
    """``cldprp``'s environment saturation: ``qst = 1`` where ``p - es <= 0``
    (zm_conv.F90:2937-2941; the only qsat call in the oracle with this guard)."""
    p_pa = p_hpa * 100.0
    return jnp.where(p_pa - saturation_vapor_pressure(t) <= 0.0, 1.0, _qsat_hpa(t, p_hpa))


def _div_md_floor(num, md):
    """``num / min(md, -1e-20)`` (zm_conv.F90:3511-3517, 3544-3548), AD-safe.

    The literal form divides by the floor where ``md`` is not yet negative;
    its reverse-mode derivative carries ``num/md**2 = 0 * 1e40`` which is NaN
    in float32.  Multiplying by the reciprocal of the floor on that branch
    gives the same value and a finite derivative.
    """
    at_floor = md >= -_MD_SMALL
    den = jnp.where(at_floor, -1.0, md)
    return jnp.where(at_floor, num * (-1.0 / _MD_SMALL), num / den)


def limcnv_from_edges(p_edge, limcnv_p_pa):
    """``zm_conv_intr.F90:334-345``: 1-based interface index ``k`` with
    ``p_edge(k) < limcnv_p_pa <= p_edge(k+1)`` (1 if the top interface is
    already below the cap, ``nlev+1`` if no interface is), per row of the
    ``(ncol, nlev+1)`` interface pressures [Pa]."""
    nlev = p_edge.shape[1] - 1
    n_above = jnp.sum((p_edge < limcnv_p_pa).astype(jnp.int32), axis=1)
    limcnv = jnp.where(p_edge[:, 0] >= limcnv_p_pa, 1,
                       jnp.where(n_above == 0, nlev + 1, n_above)).astype(jnp.int32)
    return jnp.minimum(limcnv, nlev + 1)


# ---------------------------------------------------------------------------
# buoyan_dilute + parcel_dilute (zm_conv.F90:3970-4563)
# ---------------------------------------------------------------------------
class DiluteBuoyancy(NamedTuple):
    tp: jax.Array      # parcel temperature (K), (ncol, nlev)
    qstp: jax.Array    # parcel saturation / retained vapour (kg/kg)
    tl: jax.Array      # LCL temperature (K), (ncol,)
    cape: jax.Array    # J/kg, (ncol,)
    lcl: jax.Array     # int32 1-based, (ncol,)
    lel: jax.Array
    mx: jax.Array
    buoy: jax.Array    # K, (ncol, nlev) (diagnostic)


def buoyan_dilute(q, t, p, z, pf, pblt, tpert, msg, *, num_cin, dmpdz,
                  tiedke_add, lwmax):
    """Port of ``buoyan_dilute``/``parcel_dilute`` (p, pf in hPa)."""
    ncol, nlev = t.shape
    K = _lev(nlev)
    cp, g, rl, rd = constants.c_pd, constants.g, constants.L_v, constants.R_d
    cpliq, latice, tfreez = constants.c_pw, constants.L_f, constants.T_freeze
    dtype = t.dtype
    msg1 = (msg + 1)[:, None]

    # launch level mx: bottom-most maximum of moist static energy at or below
    # the PBL top (zm_conv.F90:4113-4121; ties resolved by the descending
    # loop's strict ">" -> the lowest level of equal maxima).
    hmn = cp * t + g * z + rl * q
    cand = (K >= pblt[:, None]) & (K >= msg1)
    hm = jnp.where(cand, hmn, -jnp.inf)
    mx = (nlev - jnp.argmax(hm[:, ::-1], axis=1)).astype(jnp.int32)

    q_mx = _at(q, mx)
    t_mx = _at(t, mx)
    p_mx = _at(p, mx)
    q_kp1 = _kp1(q)
    t_kp1 = _kp1(t)
    p_kp1 = _kp1(p)

    # --- entraining ascent (parcel_dilute lines 4355-4447) ------------------
    def ascend(carry, xk):
        (sp0, qtp0, sp, qtp, mp, smix_p, qtmix_p, tmix_p, qsmix_p,
         lcl, pl, tl) = carry
        k, q_k, t_k, p_k, q_n, t_n, p_n = xk
        launch = k == mx
        entr = (k < mx) & (k >= msg + 1)
        # launch
        s0_l = moist_entropy(t_k, p_k * 100.0, q_k)
        tmix_l = invert_entropy(s0_l, p_k * 100.0, q_k, t_k)
        qsmix_l = _qsat_hpa(tmix_l, p_k)
        # entrainment of the layer between k+1 and k
        dp = p_k - p_n
        qtenv = 0.5 * (q_k + q_n)
        tenv = 0.5 * (t_k + t_n)
        penv = 0.5 * (p_k + p_n)
        senv = moist_entropy(tenv, penv * 100.0, qtenv)
        dpdz = -(penv * g) / (rd * tenv)
        dmpdp = dmpdz / dpdz
        sp_e = sp - dmpdp * dp * senv
        qtp_e = qtp - dmpdp * dp * qtenv
        mp_e = mp - dmpdp * dp
        smix_e = (sp0 + sp_e) / (1.0 + mp_e)
        qtmix_e = (qtp0 + qtp_e) / (1.0 + mp_e)
        # first guess tmix(k+1); below the launch the carry is unset, so
        # seed with t(k) there to keep the (masked-out) Newton solve finite.
        tfg = jnp.where(entr, tmix_p, t_k)
        tmix_e = invert_entropy(smix_e, p_k * 100.0, qtmix_e, tfg)
        qsmix_e = _qsat_hpa(tmix_e, p_k)
        # LCL: first level (ascending) where qsmix <= qtmix (line 4419)
        is_lcl = entr & (qsmix_e <= qtmix_e) & (qsmix_p > qtmix_p)
        qxsk = qtmix_e - qsmix_e
        qxskp1 = qtmix_p - qsmix_p
        dqxsdp = _safe_div(qxsk - qxskp1, dp, is_lcl)
        pl_new = p_n - _safe_div(qxskp1, dqxsdp, is_lcl)
        dsdp = _safe_div(smix_e - smix_p, dp, is_lcl)
        dqtdp = _safe_div(qtmix_e - qtmix_p, dp, is_lcl)
        slcl = smix_p + dsdp * (pl_new - p_n)
        qtlcl = qtmix_p + dqtdp * (pl_new - p_n)
        pl_safe = jnp.where(is_lcl, pl_new, p_k)
        tl_new = invert_entropy(slcl, pl_safe * 100.0, qtlcl, tmix_e)

        sel = lambda a, b, c: jnp.where(launch, a, jnp.where(entr, b, c))
        smix_k = sel(s0_l, smix_e, jnp.zeros_like(s0_l))
        qtmix_k = sel(q_k, qtmix_e, jnp.zeros_like(q_k))
        tmix_k = sel(tmix_l, tmix_e, jnp.zeros_like(tmix_l))
        qsmix_k = sel(qsmix_l, qsmix_e, jnp.zeros_like(qsmix_l))
        new_carry = (
            jnp.where(launch, s0_l, sp0), jnp.where(launch, q_k, qtp0),
            jnp.where(entr, sp_e, sp), jnp.where(entr, qtp_e, qtp),
            jnp.where(entr, mp_e, mp),
            jnp.where(launch | entr, smix_k, smix_p),
            jnp.where(launch | entr, qtmix_k, qtmix_p),
            jnp.where(launch | entr, tmix_k, tmix_p),
            jnp.where(launch | entr, qsmix_k, qsmix_p),
            jnp.where(is_lcl, k, lcl),
            jnp.where(is_lcl, pl_new, pl),
            jnp.where(is_lcl, tl_new, tl),
        )
        new_carry = tuple(c.astype(dtype) if c.dtype != jnp.int32 else c
                          for c in new_carry)
        return new_carry, (smix_k.astype(dtype), qtmix_k.astype(dtype),
                           tmix_k.astype(dtype), qsmix_k.astype(dtype))

    zero = jnp.zeros((ncol,), dtype)
    init = (zero, zero, zero, zero, zero, zero, zero, zero, zero,
            mx, p_mx.astype(dtype), t_mx.astype(dtype))
    carry, (smix, qtmix, tmix, qsmix) = _scan_levels(
        ascend, init, (q, t, p, q_kp1, t_kp1, p_kp1), descending=True)
    lcl, pl, tl = carry[9], carry[10], carry[11]

    # --- condensate loss + freezing (lines 4474-4559) -----------------------
    def lheat(carry, xk):
        xsh2o_p, dsx_p, dsf_p, qsmix_p = carry
        k, q_k, p_k, smix_k, qtmix_k, tmix_k, qsmix_k, t_k = xk
        launch = k == mx
        above = (k < mx) & (k >= msg + 1)
        # outside the parcel column tmix/qsmix are unset (0); seed with the
        # environment so the masked-out branch stays finite under AD.
        tm = jnp.where(launch | above, tmix_k, t_k)
        qsm = jnp.where(launch | above, qsmix_k, q_k)
        dsf = jnp.zeros_like(tm)
        xs = xsh2o_p
        dsx = dsx_p
        new_q = qtmix_k
        for _ in range(2):  # nit_lheat = 2, zm_conv.F90:4327
            xs = jnp.maximum(0.0, qtmix_k - qsm - lwmax)
            dsx = dsx_p - cpliq * jnp.log(tm / tfreez) * jnp.maximum(0.0, xs - xsh2o_p)
            frz = tm <= tfreez
            dsf = jnp.where(
                frz & (dsf_p == 0.0),
                (latice / tm) * jnp.maximum(0.0, qtmix_k - qsm - xs),
                jnp.where(frz & (dsf_p != 0.0),
                          dsf_p + (latice / tm) * jnp.maximum(0.0, qsmix_p - qsm),
                          dsf))
            new_s = smix_k + dsx + dsf
            new_q = qtmix_k - xs
            tm = invert_entropy(new_s, p_k * 100.0, new_q, tm)
            qsm = _qsat_hpa(tm, p_k)
        tp_a = tm
        qstp_a = jnp.where(new_q > qsm, qsm, new_q)
        tpv_a = (tp_a + tpert) * (1.0 + _VIRT_1608 * qstp_a) / (1.0 + new_q)
        tpv_l = (tmix_k + tpert) * (1.0 + _VIRT_1608 * q_k) / (1.0 + q_k)
        tp = jnp.where(launch, tmix_k, jnp.where(above, tp_a, t_k))
        qstp = jnp.where(launch, q_k, jnp.where(above, qstp_a, q_k))
        tpv = jnp.where(launch, tpv_l, jnp.where(above, tpv_a, t_k))
        z0 = jnp.zeros_like(tm)
        new_carry = (
            jnp.where(launch, z0, jnp.where(above, xs, xsh2o_p)),
            jnp.where(launch, z0, jnp.where(above, dsx, dsx_p)),
            jnp.where(launch, z0, jnp.where(above, dsf, dsf_p)),
            jnp.where(launch, qsmix_k, jnp.where(above, qsm, qsmix_p)),
        )
        new_carry = tuple(c.astype(dtype) for c in new_carry)
        return new_carry, (tp.astype(dtype), qstp.astype(dtype), tpv.astype(dtype))

    _, (tp, qstp, tpv) = _scan_levels(
        lheat, (zero, zero, zero, zero),
        (q, p, smix, qtmix, tmix, qsmix, t), descending=True)

    # --- buoyancy (lines 4151-4169) -----------------------------------------
    plge600 = pl >= _P_LCL_MAX_HPA
    tv = t * (1.0 + _VIRT_1608 * q) / (1.0 + q)
    inbuoy = (K <= mx[:, None]) & plge600[:, None] & (K >= msg1)
    buoy = jnp.where(inbuoy, tpv - tv + tiedke_add, 0.0)
    tp = jnp.where(inbuoy, tp, t)
    qstp = jnp.where(inbuoy, qstp, q)

    # --- tentative tops (lines 4180-4189) -----------------------------------
    buoy_kp1 = _kp1(buoy)
    cross = ((K >= msg1 + 1) & (K < lcl[:, None]) & plge600[:, None]
             & (buoy_kp1 > 0.0) & (buoy <= 0.0))
    cnt = jnp.cumsum(cross.astype(jnp.int32), axis=1)
    slot = jnp.minimum(num_cin, cnt)
    pf_floor = jnp.maximum(pf, 1.0e-2)  # coeff-ok: 1 Pa floor guards a 0 Pa model-top interface in log(pf)
    dlnp = jnp.log(pf_floor[:, 1:] / pf_floor[:, :-1])
    cape = jnp.zeros((ncol,), dtype)
    lel = jnp.full((ncol,), nlev, jnp.int32)
    for n in range(1, num_cin + 1):
        sel = cross & (slot == n)
        k_sel = jnp.where(sel, K, 0)
        lelten_n = jnp.where(sel.any(axis=1), k_sel.max(axis=1), nlev).astype(jnp.int32)
        in_cape = plge600[:, None] & (K <= mx[:, None]) & (K > lelten_n[:, None]) & (K >= msg1)
        capeten_n = jnp.sum(jnp.where(in_cape, rd * buoy * dlnp, 0.0), axis=1)
        better = capeten_n > cape
        cape = jnp.where(better, capeten_n, cape)
        lel = jnp.where(better, lelten_n, lel)
    cape = jnp.maximum(cape, 0.0)
    return DiluteBuoyancy(tp=tp, qstp=qstp, tl=tl, cape=cape, lcl=lcl, lel=lel,
                          mx=mx, buoy=buoy)


# ---------------------------------------------------------------------------
# cldprp (zm_conv.F90:2689-3607), zmconv_microp = .false.
# ---------------------------------------------------------------------------
class CloudProps(NamedTuple):
    mu: jax.Array
    eu: jax.Array
    du: jax.Array
    md: jax.Array
    ed: jax.Array
    sd: jax.Array
    qd: jax.Array
    mc: jax.Array
    qu: jax.Array
    su: jax.Array
    qst: jax.Array
    hmn: jax.Array
    hsat: jax.Array
    ql: jax.Array
    cmeg: jax.Array
    jt: jax.Array
    jlcl: jax.Array
    j0: jax.Array
    jd: jax.Array
    pflx: jax.Array
    evp: jax.Array
    cu: jax.Array
    rprd: jax.Array
    qcde: jax.Array
    eps0: jax.Array


def cldprp(q, t, p, z, s, zf, shat, qhat, jb, lel, limcnv, msg, landfrac, *,
           c0_lnd, c0_ocn, tiedke_add, alfa):
    """Port of ``cldprp`` (p in hPa; mass fluxes normalised by mb, 1/m units)."""
    ncol, nlev = t.shape
    K = _lev(nlev)
    cp, g, rl, rd, eps1 = (constants.c_pd, constants.g, constants.L_v,
                           constants.R_d, constants.epsilon)
    dtype = t.dtype
    msg1 = (msg + 1)[:, None]
    mx = jb

    c0mask = c0_ocn * (1.0 - landfrac) + c0_lnd * landfrac
    dz = zf[:, :-1] - zf[:, 1:]

    qst = _qst_cldprp(t, p)
    gamma = qst * (1.0 + qst / eps1) * eps1 * rl / (rd * t ** 2) * rl / cp
    hmn = cp * t + g * z + rl * q
    hsat = cp * t + g * z + rl * qst

    # interface values (lines 3095-3108)
    qst_km1, gam_km1 = _km1(qst), _km1(gamma)
    upper = K >= msg1 + 1
    gate_q = upper & (jnp.abs(qst_km1 - qst) > _LOGMEAN_TOL)
    qsthat = jnp.where(gate_q, _log_mean(qst_km1, qst, gate_q), qst)
    hsthat = jnp.where(upper, cp * shat + rl * qsthat, hsat)
    gate_g = upper & (jnp.abs(gam_km1 - gamma) > _LOGMEAN_TOL)
    gamhat = jnp.where(gate_g, _log_mean(gam_km1, gamma, gate_g), gamma)

    # cloud top / detrainment-start initialisation (lines 3113-3138)
    jt = jnp.minimum(jnp.maximum(lel, limcnv + 1), nlev).astype(jnp.int32)
    jlcl = lel
    in_j0 = (K >= jt[:, None]) & (K <= jb[:, None]) & (K >= msg1)
    hs_m = jnp.where(in_j0, hsat, jnp.inf)
    j0 = (nlev - jnp.argmin(hs_m[:, ::-1], axis=1)).astype(jnp.int32)  # last "<="
    j0 = jnp.where(in_j0.any(axis=1), j0, jt + 2)
    j0 = jnp.minimum(j0, jb - 2)
    j0 = jnp.maximum(j0, jt + 2)
    j0 = jnp.minimum(j0, nlev)

    hmn_mx = _at(hmn, mx)
    s_mx = _at(s, mx)
    q_mx = _at(q, mx)
    incloud = (K >= jt[:, None]) & (K <= jb[:, None]) & (K >= msg1)
    hu = jnp.where(incloud, hmn_mx[:, None] + cp * tiedke_add, hmn)
    su = jnp.where(incloud, s_mx[:, None] + tiedke_add, s)

    # Taylor series (lines 3143-3155)
    def taylor(carry, xk):
        k1p, i2p, i3p, i4p = carry
        k, hmn_k, dz_k = xk
        cond = (k < jb) & (k >= jt) & (k >= msg + 1)
        k1 = jnp.where(cond, k1p + (hmn_mx - hmn_k) * dz_k, 0.0)
        i2 = jnp.where(cond, i2p + 0.5 * (k1p + k1) * dz_k, 0.0)
        i3 = jnp.where(cond, i3p + 0.5 * (i2p + i2) * dz_k, 0.0)
        i4 = jnp.where(cond, i4p + 0.5 * (i3p + i3) * dz_k, 0.0)
        out = tuple(a.astype(dtype) for a in (k1, i2, i3, i4))
        return out, out

    zero = jnp.zeros((ncol,), dtype)
    _, (k1, i2, i3, i4) = _scan_levels(
        taylor, (zero, zero, zero, zero), (hmn, dz), descending=True)

    # expdif (lines 3163-3172)
    in_hmin = (K >= j0[:, None]) & (K <= jb[:, None]) & (K >= msg1)
    hmin = jnp.min(jnp.where(in_hmin, hmn, _HMIN_INIT), axis=1)
    expdif = jnp.where(in_hmin.any(axis=1), hmn_mx - hmin, 0.0)

    # f(k) 4th-order series (lines 3177-3197)
    z_km1, hsat_km1 = _km1(z), _km1(hsat)
    zf_k = zf[:, :-1]
    inrange = (K >= jt[:, None]) & (K < jb[:, None])
    k1 = jnp.where(inrange, k1, 0.0)
    den_z = z_km1 - z
    expnum = jnp.where(
        inrange & upper,
        hmn_mx[:, None] - _safe_div(hsat_km1 * (zf_k - z) + hsat * (z_km1 - zf_k),
                                    den_z, inrange & upper),
        0.0)
    gate_f = upper & (expdif[:, None] > _EXPDIF_MIN) & (expnum > 0.0) & (k1 > expnum * dz)
    ftemp = _safe_div(expnum, k1, gate_f)
    k1s = jnp.where(gate_f, k1, 1.0)
    f_series = (ftemp + i2 / k1s * ftemp ** 2
                + (2.0 * i2 ** 2 - k1s * i3) / k1s ** 2 * ftemp ** 3
                + (-5.0 * k1s * i2 * i3 + 5.0 * i2 ** 3 + k1s ** 2 * i4) / k1s ** 3 * ftemp ** 4)  # coeff-ok: 4th-order Taylor-series coefficients, zm_conv.F90:3160
    f = jnp.where(gate_f, jnp.clip(f_series, 0.0, _F_MAX), 0.0)

    # j0 bump (line 3199-3203)
    f_j0 = _at(f, j0)
    f_j0p1 = _at(f, j0 + 1)
    j0 = jnp.where((j0 < jb) & (f_j0 < _F_TINY) & (f_j0p1 > f_j0), j0 + 1, j0)

    # running max f(k) = max(f(k), f(k-1)) for k in [jt, j0] (lines 3204-3210)
    fmax_src = jnp.where(K >= (jt - 1)[:, None], f, -jnp.inf)
    cmax = jax.lax.cummax(fmax_src, axis=1)
    in_rmax = (K >= jt[:, None]) & (K <= j0[:, None]) & upper
    f = jnp.where(in_rmax, cmax, f)

    eps0 = _at(f, j0)
    eps = jnp.where((K >= j0[:, None]) & (K <= jb[:, None]) & (K >= msg1), eps0[:, None],
                    jnp.where((K < j0[:, None]) & (K >= jt[:, None]) & (K >= msg1), f, 0.0))
    eps = jnp.where(K == jb[:, None], eps0[:, None], eps)
    pos0 = eps0 > 0.0

    # updraft mass flux (lines 3233-3252)
    zf_jb = _at(zf, jb)
    dz_jb = _at(dz, jb)
    up_rng = pos0[:, None] & (K >= jt[:, None]) & (K < jb[:, None]) & (K >= msg1)
    zuef = zf_k - zf_jb[:, None]
    zuef_s = jnp.where(up_rng, zuef, 1.0)
    eps0_s = jnp.where(pos0, eps0, 1.0)[:, None]
    mu_cf = (1.0 / eps0_s) * jnp.expm1(eps * zuef_s) / zuef_s
    rmue = (1.0 / eps0_s) * jnp.expm1(_kp1(eps) * zuef_s) / zuef_s
    mu = jnp.where(K == jb[:, None], jnp.where(pos0, 1.0, 0.0)[:, None],
                   jnp.where(up_rng, mu_cf, 0.0))
    mu_kp1 = _kp1(mu)
    eu = jnp.where(K == jb[:, None], jnp.where(pos0, 1.0 / dz_jb, 0.0)[:, None],
                   jnp.where(up_rng, (rmue - mu_kp1) / dz, 0.0))
    du = jnp.where(up_rng, (rmue - mu) / dz, 0.0)

    # hu recursion with the mu < 0.02 cut-off (lines 3262-3282)
    def hu_rec(carry, xk):
        mu_p, hu_p = carry
        k, mu_k, eu_k, du_k, hu_k, hmn_k, hsat_k, dz_k = xk
        cond = (k <= jb - 1) & (k >= lel) & pos0
        low = mu_k < _MU_MIN
        mu_n = jnp.where(cond & low, 0.0, mu_k)
        eu_n = jnp.where(cond & low, 0.0, eu_k)
        du_n = jnp.where(cond & low, mu_p / dz_k, du_k)
        hu_full = (_safe_div(mu_p, mu_k, ~low) * hu_p
                   + _safe_div(dz_k, mu_k, ~low) * (eu_k * hmn_k - du_k * hsat_k))
        hu_n = jnp.where(cond, jnp.where(low, hmn_k, hu_full), hu_k)
        out = tuple(a.astype(dtype) for a in (mu_n, eu_n, du_n, hu_n))
        return (out[0], out[3]), out

    _, (mu, eu, du, hu) = _scan_levels(
        hu_rec, (zero, zero), (mu, eu, du, hu, hmn, hsat, dz), descending=True)

    # cloud-top reset (lines 3287-3314)
    hu_jb = _at(hu, jb)
    rng = (K <= (jb - 2)[:, None]) & (K >= (lel - 1)[:, None])
    condA = (hu <= hsthat) & (_kp1(hu) > _kp1(hsthat)) & (mu >= _MU_MIN)
    condB = (hu > hu_jb[:, None]) | (mu < _MU_MIN)
    hit = rng & (condA | condB)
    any_hit = hit.any(axis=1)
    k_hit = (nlev - jnp.argmax(hit[:, ::-1], axis=1)).astype(jnp.int32)
    a_hit = _at(condA, k_hit)
    diff_hit = _at(hu - hsthat, k_hit)
    jt_hit = jnp.where(a_hit, jnp.where(diff_hit < _HU_TOP_JUMP, k_hit + 1, k_hit), k_hit + 1)
    jt = jnp.where(any_hit, jt_hit, jt).astype(jnp.int32)

    # zero above the top; du at the top (lines 3316-3329)
    above = (K >= lel[:, None]) & (K <= jt[:, None]) & pos0[:, None] & (K >= msg1)
    mu = jnp.where(above, 0.0, mu)
    eu = jnp.where(above, 0.0, eu)
    du = jnp.where(above, 0.0, du)
    hu = jnp.where(above, hmn, hu)
    at_top = (K == jt[:, None]) & pos0[:, None]
    du = jnp.where(at_top, _kp1(mu) / dz, du)
    eu = jnp.where(at_top, 0.0, eu)
    mu = jnp.where(at_top, 0.0, mu)

    # updraft q, s bottom-up to the LCL (lines 3334-3356); qu init = q (line 3086)
    p_km1 = _km1(p)
    qu = q

    def qs_up(carry, xk):
        su_p, qu_p, done, jlcl_c = carry
        k, mu_k, mu_n, eu_k, du_k, s_k, q_k, qst_k, hu_k, zf_k_, p_k, p_m, dz_k, su_k, qu_k = xk
        at_jb = (k == jb) & pos0 & (k >= msg + 2)
        mid = (~done) & (k > jt) & (k < jb) & pos0 & (k >= msg + 2)
        su_jb = (hu_k - rl * q_mx) / cp
        r = _safe_div(mu_n, mu_k, mid)
        dzr = _safe_div(dz_k, mu_k, mid)
        su_mid = r * su_p + dzr * (eu_k - du_k) * s_k
        qu_mid = r * qu_p + dzr * (eu_k * q_k - du_k * qst_k)
        su_n = jnp.where(at_jb, su_jb, jnp.where(mid, su_mid, su_k))
        qu_n = jnp.where(at_jb, q_mx, jnp.where(mid, qu_mid, qu_k))
        tu = su_n - g / cp * zf_k_
        qstu = _qsat_hpa(tu, 0.5 * (p_k + p_m))
        sat = mid & (qu_n >= qstu)
        new_carry = (su_n.astype(dtype), qu_n.astype(dtype), done | sat,
                     jnp.where(sat, k, jlcl_c).astype(jnp.int32))
        return new_carry, (su_n.astype(dtype), qu_n.astype(dtype))

    carry, (su, qu) = _scan_levels(
        qs_up, (zero, zero, jnp.zeros((ncol,), bool), jlcl),
        (mu, _kp1(mu), eu, du, s, q, qst, hu, zf_k, p, p_km1, dz, su, qu),
        descending=True)
    jlcl = carry[3]

    # saturated updraft above the LCL (lines 3358-3366)
    sat_rng = (K > jt[:, None]) & (K <= jlcl[:, None]) & pos0[:, None] & upper
    su = jnp.where(sat_rng, shat + (hu - hsthat) / (cp * (1.0 + gamhat)), su)
    qu = jnp.where(sat_rng, qsthat + gamhat * (hu - hsthat) / (rl * (1.0 + gamhat)), qu)

    # condensation (lines 3373-3388)
    cu_rng = (K >= jt[:, None]) & (K < jb[:, None]) & pos0[:, None] & upper
    cu = ((mu * su - _kp1(mu) * _kp1(su)) / dz - (eu - du) * s) / (rl / cp)
    cu = jnp.where(K == jt[:, None], 0.0, cu)
    cu = jnp.where(cu_rng, jnp.maximum(0.0, cu), 0.0)

    # liquid water, rain production (lines 3502-3524)
    def ql_rec(carry, xk):
        ql_p, totpcp = carry
        k, mu_k, mu_n, du_k, cu_k, dz_k = xk
        cond = (k >= jt) & (k < jb) & pos0 & (mu_k >= 0.0) & (k >= msg + 2)
        pos = cond & (mu_k > 0.0)
        ql1 = _safe_div(mu_n * ql_p - dz_k * du_k * ql_p + dz_k * cu_k, mu_k, pos)
        ql_n = jnp.where(pos, ql1 / (1.0 + dz_k * c0mask), 0.0)
        totpcp = totpcp + jnp.where(cond, dz_k * (cu_k - du_k * ql_p), 0.0)
        rprd_k = jnp.where(cond, c0mask * mu_k * ql_n, 0.0)
        return ((ql_n.astype(dtype), totpcp.astype(dtype)),
                (ql_n.astype(dtype), rprd_k.astype(dtype)))

    (_, totpcp), (ql, rprd) = _scan_levels(
        ql_rec, (zero, zero), (mu, _kp1(mu), du, cu, dz), descending=True)
    qcde = ql

    # downdraft (lines 3412-3496)
    jt = jnp.minimum(jt, jb - 1).astype(jnp.int32)
    jd = jnp.minimum(jnp.maximum(j0, jt + 1), jb).astype(jnp.int32)
    dd_on = (jd < jb) & pos0
    hd = jnp.where(K == jd[:, None], _at(hmn, jd - 1)[:, None], hmn)
    md = jnp.where((K == jd[:, None]) & dd_on[:, None], -alfa, 0.0)
    zf_jd = _at(zf, jd)
    md_rng = (K > jd[:, None]) & (K <= jb[:, None]) & pos0[:, None] & (K >= msg1)
    zdef = jnp.where(md_rng, zf_jd[:, None] - zf_k, 1.0)
    md = jnp.where(md_rng, -alfa / (2.0 * eps0_s) * jnp.expm1(2.0 * eps0_s * zdef) / zdef, md)
    md_jb = _at(md, jb)
    ratmjb = jnp.minimum(jnp.abs(_safe_div(_at(mu, jb), md_jb, dd_on)), 1.0)
    md = jnp.where((K >= jt[:, None]) & (K <= jb[:, None]) & dd_on[:, None] & (K >= msg1),
                   md * ratmjb[:, None], md)
    # ed(k-1) = (md(k-1)-md(k))/dz(k-1) for k >= jt  ->  ed(j) for j >= jt-1
    ed = jnp.where((K >= (jt - 1)[:, None]) & (K <= nlev - 1) & pos0[:, None],
                   (md - _kp1(md)) / dz, 0.0)

    def hd_rec(carry, xk):
        hd_p = carry
        k, md_k, md_m, ed_m, hmn_m, dz_m, hd_k = xk
        cond = (k >= jt) & pos0 & (k >= msg + 1)
        hd_n = jnp.where(cond, _div_md_floor(md_m * hd_p - dz_m * ed_m * hmn_m, md_k), hd_k)
        return hd_n.astype(dtype), hd_n.astype(dtype)

    _, hd = _scan_levels(
        hd_rec, zero, (md, _km1(md), _km1(ed), _km1(hmn), _km1(dz), hd), descending=False)

    qds_rng = (K >= jd[:, None]) & (K <= jb[:, None]) & dd_on[:, None] & upper
    qds = jnp.where(qds_rng, qsthat + gamhat * (hd - hsthat) / (rl * (1.0 + gamhat)), q)
    qd_rng = (K == jd[:, None]) | ((K > jd[:, None]) & (K <= jb[:, None]) & pos0[:, None])
    qd = jnp.where(qd_rng, qds, q)
    hd_jd, qd_jd = _at(hd, jd), _at(qd, jd)
    sd_init = jnp.where(K == jd[:, None], ((hd_jd - rl * qd_jd) / cp)[:, None], s)

    def evp_rec(carry, xk):
        sd_k, totevp = carry
        k, ed_k, q_k, md_k, qd_k, md_n, qd_n, dz_k, s_k, sd_init_n = xk
        cond = (k >= jd) & (k < jb) & pos0 & (k >= msg + 2)
        evp_k = jnp.where(cond, jnp.maximum(
            -ed_k * q_k + (md_k * qd_k - md_n * qd_n) / dz_k, 0.0), 0.0)
        sd_n = jnp.where(cond, _div_md_floor((rl / cp * evp_k - ed_k * s_k) * dz_k + md_k * sd_k, md_n),
                         sd_init_n)
        totevp = totevp - jnp.where(cond, dz_k * ed_k * q_k, 0.0)
        return (sd_n.astype(dtype), totevp.astype(dtype)), (evp_k.astype(dtype), sd_k.astype(dtype))

    (_, totevp), (evp, sd) = _scan_levels(
        evp_rec, (sd_init[:, 0], zero),
        (ed, q, md, qd, _kp1(md), _kp1(qd), dz, s, _kp1(sd_init)), descending=False)
    totevp = totevp + _at(md, jd) * qd_jd - _at(md, jb) * _at(qd, jb)

    totpcp = jnp.maximum(totpcp, 0.0)
    totevp = jnp.maximum(totevp, 0.0)
    both = (totevp > 0.0) & (totpcp > 0.0)
    fac = jnp.where(both, jnp.minimum(1.0, _safe_div(totpcp, totevp + totpcp, both)), 0.0)
    md = jnp.where(upper, md * fac[:, None], md)
    ed = jnp.where(upper, ed * fac[:, None], ed)
    evp = jnp.where(upper, evp * fac[:, None], evp)
    cmeg = jnp.where(upper, cu - evp, 0.0)
    rprd = jnp.where(upper, rprd - evp, rprd)

    pflx = jnp.concatenate(
        [jnp.zeros((ncol, 1), dtype), jnp.cumsum(rprd * dz, axis=1)], axis=1)
    mc = jnp.where(K >= msg1, mu + md, 0.0)
    return CloudProps(mu=mu, eu=eu, du=du, md=md, ed=ed, sd=sd, qd=qd, mc=mc,
                      qu=qu, su=su, qst=qst, hmn=hmn, hsat=hsat, ql=ql,
                      cmeg=cmeg, jt=jt, jlcl=jlcl, j0=j0, jd=jd, pflx=pflx,
                      evp=evp, cu=cu, rprd=rprd, qcde=qcde, eps0=eps0)


# ---------------------------------------------------------------------------
# closure (zm_conv.F90:3609-3819)
# ---------------------------------------------------------------------------
def closure(q, t, p, z, s, tp, qs, qu, su, mc, du, mu, md, qd, sd, qhat, shat,
            dp, qstp, zf, ql, dsubcld, cape, tl, lcl, lel, jt, mx, msg, *,
            capelmt, tau, deep):
    """Port of ``closure``: cloud-base mass flux ``mb`` [hPa/s].

    ``deep`` marks the columns the Fortran gathers.  A launch-level vapour
    pressure ``eb <= 0`` is a NaN in the Fortran (``log(eb)``); it stays NaN
    here in a deep column and is 0 in a non-deep one, whose ``mb`` is masked
    anyway (keeps the reverse-mode pass finite).
    """
    ncol, nlev = t.shape
    K = _lev(nlev)
    cp, g, rl, rd, eps1 = (constants.c_pd, constants.g, constants.L_v,
                           constants.R_d, constants.epsilon)
    msg1 = (msg + 1)[:, None]
    q_mx, p_mx, t_mx = _at(q, mx), _at(p, mx), _at(t, mx)
    ok = dsubcld > 0.0
    eb = p_mx * q_mx / (eps1 + q_mx)
    dtbdt = _safe_div(_at(mu, mx) * (_at(shat, mx) - _at(su, mx))
                      + _at(md, mx) * (_at(shat, mx) - _at(sd, mx)), dsubcld, ok)
    dqbdt = _safe_div(_at(mu, mx) * (_at(qhat, mx) - _at(qu, mx))
                      + _at(md, mx) * (_at(qhat, mx) - _at(qd, mx)), dsubcld, ok)
    debdt = eps1 * p_mx / (eps1 + q_mx) ** 2 * dqbdt
    eb_ok = eb > 0.0
    eb_s = jnp.where(eb_ok, eb, 1.0)
    dtldt = jnp.where(
        eb_ok,
        -_BOLTON_A * (_BOLTON_B / t_mx * dtbdt - debdt / eb_s)
        / (_BOLTON_B * jnp.log(t_mx) - jnp.log(eb_s) - _BOLTON_C) ** 2,
        jnp.where(deep, jnp.nan, 0.0))

    mu_n, md_n, su_n, sd_n, shat_n = _kp1(mu), _kp1(md), _kp1(su), _kp1(sd), _kp1(shat)
    qu_n, qd_n, qhat_n, ql_n, mc_n = _kp1(qu), _kp1(qd), _kp1(qhat), _kp1(ql), _kp1(mc)
    at_jt = (K == jt[:, None]) & (K <= nlev - 1) & (K >= msg1)
    mid = (K > jt[:, None]) & (K < mx[:, None]) & (K <= nlev - 1) & (K >= msg1)
    dtmdt_jt = (mu_n * (su_n - shat_n - rl / cp * ql_n) + md_n * (sd_n - shat_n)) / dp
    dqmdt_jt = (mu_n * (qu_n - qhat_n + ql_n) + md_n * (qd_n - qhat_n)) / dp
    dtmdt_mid = ((mc * (shat - s) + mc_n * (s - shat_n)) / dp - rl / cp * du * ql_n)
    dqmdt_mid = ((mu_n * (qu_n - qhat_n + cp / rl * (su_n - s))
                  - mu * (qu - qhat + cp / rl * (su - s))
                  + md_n * (qd_n - qhat_n + cp / rl * (sd_n - s))
                  - md * (qd - qhat + cp / rl * (sd - s))) / dp + du * ql_n)
    dtmdt = jnp.where(at_jt, dtmdt_jt, jnp.where(mid, dtmdt_mid, 0.0))
    dqmdt = jnp.where(at_jt, dqmdt_jt, jnp.where(mid, dqmdt_mid, 0.0))

    kap = rd / cp
    pi_fac = (1000.0 / p) ** kap
    thetavm = t * pi_fac * (1.0 + _VIRT_0608 * q)
    qmx = q_mx[:, None]
    tl_c = tl[:, None]
    # lel <= k <= lcl (lines 3742-3760)
    r1 = (K >= lel[:, None]) & (K <= lcl[:, None]) & (K >= msg1)
    thetavp1 = tp * pi_fac * (1.0 + _VIRT_1608 * qstp - qmx)
    dqsdtp = qstp * (1.0 + qstp / eps1) * eps1 * rl / (rd * tp ** 2)
    dtpdt = (tp / (1.0 + rl / cp * (dqsdtp - qstp / tp))
             * (dtbdt[:, None] / t_mx[:, None]
                + rl / cp * (dqbdt[:, None] / tl_c - qmx / tl_c ** 2 * dtldt[:, None])))
    dboydt1 = (((dtpdt / tp + 1.0 / (1.0 + _VIRT_1608 * qstp - qmx)
                 * (_VIRT_1608 * dqsdtp * dtpdt - dqbdt[:, None]))
                - (dtmdt / t + _VIRT_0608 / (1.0 + _VIRT_0608 * q) * dqmdt))
               * g * thetavp1 / thetavm)
    # lcl < k < mx (lines 3764-3775)
    r2 = (K > lcl[:, None]) & (K < mx[:, None]) & (K >= msg1)
    thetavp2 = tp * pi_fac * (1.0 + _VIRT_0608 * qmx)
    dboydt2 = ((dtbdt[:, None] / t_mx[:, None]
                + _VIRT_0608 / (1.0 + _VIRT_0608 * qmx) * dqbdt[:, None]
                - dtmdt / t - _VIRT_0608 / (1.0 + _VIRT_0608 * q) * dqmdt)
               * g * thetavp2 / thetavm)
    dboydt = jnp.where(r1, dboydt1, jnp.where(r2, dboydt2, 0.0))
    dzf = zf[:, :-1] - zf[:, 1:]
    in_dadt = (K >= lel[:, None]) & (K <= (mx - 1)[:, None])
    dadt = jnp.sum(jnp.where(in_dadt, dboydt * dzf, 0.0), axis=1)
    dltaa = -(cape - capelmt)
    nz = dadt != 0.0
    mb = jnp.where(nz, jnp.maximum(_safe_div(dltaa / tau, dadt, nz), 0.0), 0.0)
    return mb


# ---------------------------------------------------------------------------
# q1q2_pjr (zm_conv.F90:3821-3968)
# ---------------------------------------------------------------------------
def q1q2_pjr(q, qs, qu, su, du, qhat, shat, dp, mu, md, sd, qd, ql, dsubcld,
             jt, mx, msg, evp, cu):
    """Port of ``q1q2_pjr``: (dqdt [kg/kg/s], dsdt [K/s], dl [kg/kg/s])."""
    ncol, nlev = q.shape
    K = _lev(nlev)
    cp, rl = constants.c_pd, constants.L_v
    msg1 = (msg + 1)[:, None]
    mu_n, md_n, su_n, sd_n, shat_n = _kp1(mu), _kp1(md), _kp1(su), _kp1(sd), _kp1(shat)
    qu_n, qd_n, qhat_n, ql_n = _kp1(qu), _kp1(qd), _kp1(qhat), _kp1(ql)
    emc = -cu + evp
    top = (K <= nlev - 1) & (K >= msg1)
    dsdt = jnp.where(top, -rl / cp * emc + (mu_n * (su_n - shat_n) - mu * (su - shat)
                                            + md_n * (sd_n - shat_n) - md * (sd - shat)) / dp, 0.0)
    dqdt = jnp.where(top, emc + (mu_n * (qu_n - qhat_n) - mu * (qu - qhat)
                                 + md_n * (qd_n - qhat_n) - md * (qd - qhat)) / dp, 0.0)
    dl = jnp.where(top, du * ql_n, 0.0)
    ok = dsubcld > 0.0
    dsdt_mx = _safe_div(-_at(mu, mx) * (_at(su, mx) - _at(shat, mx))
                        - _at(md, mx) * (_at(sd, mx) - _at(shat, mx)), dsubcld, ok)
    dqdt_mx = _safe_div(-_at(mu, mx) * (_at(qu, mx) - _at(qhat, mx))
                        - _at(md, mx) * (_at(qd, mx) - _at(qhat, mx)), dsubcld, ok)
    below = (K >= mx[:, None]) & (K >= msg1)
    dsdt = jnp.where(below, dsdt_mx[:, None], dsdt)
    dqdt = jnp.where(below, dqdt_mx[:, None], dqdt)
    return dqdt, dsdt, dl


# ---------------------------------------------------------------------------
# zm_convr (zm_conv.F90:142-1393)
# ---------------------------------------------------------------------------
class ZMConvr(NamedTuple):
    dqdt: jax.Array      # kg/kg/s
    heat: jax.Array      # J/kg/s
    dlf: jax.Array       # detrained cloud water, kg/kg/s
    rprd: jax.Array      # rain production net of downdraft evaporation, kg/kg/s
    prec: jax.Array      # kg/m^2/s (Fortran prec [m/s] * 1000)
    cape: jax.Array      # J/kg
    mb: jax.Array        # hPa/s
    ideep: jax.Array     # bool
    mu: jax.Array        # hPa/s (scaled by mb)
    md: jax.Array
    du: jax.Array        # 1/s (scaled by mb)
    eu: jax.Array
    ed: jax.Array
    mc: jax.Array
    cmeg: jax.Array
    pflx: jax.Array      # kg/m^2/s at interfaces (nlev+1)
    dp: jax.Array        # hPa
    dsubcld: jax.Array   # hPa
    jt: jax.Array
    maxg: jax.Array
    ql: jax.Array
    limcnv: jax.Array
    msg: jax.Array


def zm_convr(t, qh, p_full, p_half, z, zf, landfrac, dt, *, pblh, tpert,
             capelmt, tau, num_cin, dmpdz, tiedke_add, lwmax, c0_lnd, c0_ocn,
             alfa, limcnv_p_pa, pbl_top_pa, pref_edge=None):
    """Port of ``zm_convr`` (``delt = 0.5*dt`` as in ``zm_conv_tend``).

    ``pblh`` is the PBL height [m] (``None`` -> the launch-level search is
    bounded by ``pbl_top_pa`` [Pa] instead of the Fortran ``pblt``, a
    documented departure because the host does not carry ``pblh``).

    ``pref_edge`` are the REFERENCE interface pressures [Pa] (``nlev+1``, top
    first): CAM caps deep convection at the interface where they cross
    ``limcnv_p_pa`` ONCE at init.  ``None`` (hosts without a reference
    vertical coordinate) falls back to each column's own ``p_half``, so the
    cap can move with surface pressure -- a documented departure.
    """
    ncol, nlev = t.shape
    K = _lev(nlev)
    dtype = t.dtype
    # Hosts may hand mixed dtypes (float32 T with float64 pressures); the
    # level scans need one carry dtype, so everything follows T.
    qh, p_full, p_half, z, zf, landfrac = (
        jnp.asarray(a, dtype) for a in (qh, p_full, p_half, z, zf, landfrac))
    cp, g = constants.c_pd, constants.g
    delt = 0.5 * dt

    # limcnv: interface index k with pref_edge(k) < 40 hPa <= pref_edge(k+1)
    # (zm_conv_intr.F90:334-345); msg = limcnv - 1 (zm_conv.F90:450).
    if pref_edge is None:
        limcnv = limcnv_from_edges(p_half, limcnv_p_pa)
    else:
        pe = jnp.asarray(pref_edge, dtype).reshape(1, nlev + 1)
        limcnv = limcnv_from_edges(jnp.broadcast_to(pe, (ncol, nlev + 1)), limcnv_p_pa)
    msg = limcnv - 1
    msg1 = (msg + 1)[:, None]

    p = p_full * 0.01  # coeff-ok: Pa -> hPa (zm_conv.F90:731)
    pf = p_half * 0.01  # coeff-ok: Pa -> hPa
    dpp = p_half[:, 1:] - p_half[:, :-1]
    dp = 0.01 * dpp  # coeff-ok: Pa -> hPa
    dz = zf[:, :-1] - zf[:, 1:]

    if pblh is None:
        in_pbl = (p_full >= pbl_top_pa) & (K >= msg1)
        pblt = jnp.where(in_pbl.any(axis=1),
                         jnp.min(jnp.where(in_pbl, K, nlev), axis=1), nlev)
    else:
        match = (jnp.abs(z - pblh[:, None]) < 0.5 * dz) & (K <= nlev - 1) & (K >= msg1)
        pblt = jnp.where(match.any(axis=1),
                         jnp.min(jnp.where(match, K, nlev), axis=1), nlev)
    pblt = pblt.astype(jnp.int32)

    q = qh
    s = t + (g / cp) * z
    tpert_arr = jnp.broadcast_to(jnp.asarray(tpert, dtype), (ncol,))
    bd = buoyan_dilute(q, t, p, z, pf, pblt, tpert_arr, msg, num_cin=num_cin,
                       dmpdz=dmpdz, tiedke_add=tiedke_add, lwmax=lwmax)
    cape, lcl, lel, maxg, tl, tp, qstp = (bd.cape, bd.lcl, bd.lel, bd.mx, bd.tl,
                                           bd.tp, bd.qstp)
    ideep = cape > capelmt

    dsubcld = jnp.sum(jnp.where((K >= maxg[:, None]) & (K >= msg1), dp, 0.0), axis=1)

    # interface s, q (lines 889-909)
    s_km1, q_km1 = _km1(s), _km1(q)
    upper = K >= msg1 + 1
    sdifr = jnp.where((s > 0.0) | (s_km1 > 0.0),
                      jnp.abs(_safe_div(s - s_km1, jnp.maximum(s_km1, s), upper)), 0.0)
    qdifr = jnp.where((q > 0.0) | (q_km1 > 0.0),
                      jnp.abs(_safe_div(q - q_km1, jnp.maximum(q_km1, q), upper)), 0.0)
    # The Fortran forms these only on the gathered deep columns (lines
    # 795-804): a zero humidity next to a finite one in a non-convecting
    # column must not reach the log.
    gs = upper & ideep[:, None] & (sdifr > _LOGMEAN_TOL)
    gq = upper & ideep[:, None] & (qdifr > _LOGMEAN_TOL)
    shat = jnp.where(gs, _log_mean(s_km1, s, gs), jnp.where(upper, 0.5 * (s + s_km1), s))
    qhat = jnp.where(gq, _log_mean(q_km1, q, gq), jnp.where(upper, 0.5 * (q + q_km1), q))

    cpz = cldprp(q, t, p, z, s, zf, shat, qhat, maxg, lel, limcnv, msg, landfrac,
                 c0_lnd=c0_lnd, c0_ocn=c0_ocn, tiedke_add=tiedke_add, alfa=alfa)
    jt = cpz.jt

    # 1/m -> 1/hPa (lines 936-946)
    conv = jnp.where(K >= msg1, dz / dp, 1.0)
    du, eu, ed = cpz.du * conv, cpz.eu * conv, cpz.ed * conv
    cug, cmeg, rprdg, evpg = cpz.cu * conv, cpz.cmeg * conv, cpz.rprd * conv, cpz.evp * conv

    mb = closure(q, t, p, z, s, tp, cpz.qst, cpz.qu, cpz.su, cpz.mc, du, cpz.mu,
                 cpz.md, cpz.qd, cpz.sd, qhat, shat, dp, qstp, zf, cpz.ql, dsubcld,
                 cape, tl, lcl, lel, jt, maxg, msg, capelmt=capelmt, tau=tau,
                 deep=ideep)

    # theoretical upper bound (lines 972-985)
    mumax = jnp.max(jnp.where(upper, cpz.mu / dp, 0.0), axis=1)
    mb = jnp.where(mumax > 0.0,
                   jnp.minimum(mb, _safe_div(0.5, delt * mumax, mumax > 0.0)), 0.0)
    mb = jnp.where(ideep, mb, 0.0)

    scale = jnp.where(K >= msg1, mb[:, None], 0.0)
    mu, md, mc = cpz.mu * scale, cpz.md * scale, cpz.mc * scale
    du, eu, ed = du * scale, eu * scale, ed * scale
    cmeg, rprdg, cug, evpg = cmeg * scale, rprdg * scale, cug * scale, evpg * scale
    pflx = cpz.pflx * jnp.concatenate(
        [jnp.zeros((ncol, 1), dtype), jnp.where(K >= msg1, mb[:, None] * 100.0 / g, 0.0)],
        axis=1)

    dqdt, dsdt, dlg = q1q2_pjr(q, cpz.qst, cpz.qu, cpz.su, du, qhat, shat, dp, mu,
                               md, cpz.sd, cpz.qd, cpz.qcde, dsubcld, jt, maxg, msg,
                               evpg, cug)
    heat = dsdt * cp
    # prec = -sum dpp (q - qh) - sum dpp dlf 2 delt  ->  /(2 delt g) [kg/m^2/s]
    prec = jnp.maximum(-jnp.sum(jnp.where(K >= msg1, dpp * (dqdt + dlg), 0.0), axis=1),
                       0.0) / g
    return ZMConvr(dqdt=dqdt, heat=heat, dlf=dlg, rprd=rprdg, prec=prec, cape=cape,
                   mb=mb, ideep=ideep, mu=mu, md=md, du=du, eu=eu, ed=ed, mc=mc,
                   cmeg=cmeg, pflx=pflx, dp=dp, dsubcld=dsubcld, jt=jt, maxg=maxg,
                   ql=cpz.ql, limcnv=limcnv, msg=msg)


# ---------------------------------------------------------------------------
# zm_conv_evap (zm_conv.F90:1396-1639, old_snow path) + cldfrc_fice
# ---------------------------------------------------------------------------
def cldfrc_fice(t):
    """``cloud_fraction.F90::cldfrc_fice`` -> (fice, fsnow)."""
    tmelt = constants.T_freeze
    tmax_fice = tmelt - _FICE_TMAX_OFFSET
    tmin_fice = tmax_fice - _FICE_RANGE
    fice = jnp.where(t > tmax_fice, 0.0,
                     jnp.where(t < tmin_fice, 1.0, (tmax_fice - t) / (tmax_fice - tmin_fice)))
    tmin_fsnow = tmelt - _FSNOW_RANGE
    fsnow = jnp.where(t > tmelt, 0.0,
                      jnp.where(t < tmin_fsnow, 1.0, (tmelt - t) / (tmelt - tmin_fsnow)))
    return fice, fsnow


class ZMEvap(NamedTuple):
    tend_s: jax.Array    # J/kg/s
    tend_q: jax.Array    # kg/kg/s
    prec: jax.Array      # kg/m^2/s at the surface
    snow: jax.Array      # kg/m^2/s
    ntprprd: jax.Array   # net precip production, kg/kg/s
    ntsnprd: jax.Array
    flxprec: jax.Array   # kg/m^2/s at interfaces
    flxsnow: jax.Array


def zm_conv_evap(t, pmid, pdel, q, prdprec, cldfrc, deltat, prec, *, ke):
    """Port of ``zm_conv_evap`` (``prec`` in kg/m^2/s in and out)."""
    ncol, nlev = t.shape
    dtype = t.dtype
    g, latvap, latice, tmelt = constants.g, constants.L_v, constants.L_f, constants.T_freeze
    qs = saturation_mixing_ratio(t, pmid)
    _, fsnow_conv = cldfrc_fice(t)

    def body(carry, xk):
        flxprec, flxsnow, evpvint = carry
        k, t_k, q_k, qs_k, pdel_k, prd_k, cld_k, fsn_k = xk
        warm = t_k > tmelt
        flxsntm = jnp.where(warm, 0.0, flxsnow)
        snowmlt = jnp.where(warm, flxsnow * g / pdel_k, 0.0)
        evplimit = jnp.maximum(1.0 - q_k / qs_k, 0.0)
        evpprec = ke * (1.0 - cld_k) * evplimit * jnp.sqrt(jnp.maximum(flxprec, _FLX_SQRT_FLOOR))
        evplimit = jnp.maximum(0.0, (qs_k - q_k) / deltat)
        evplimit = jnp.minimum(evplimit, flxprec * g / pdel_k)
        evplimit = jnp.minimum(evplimit, (prec - evpvint) * g / pdel_k)
        evpprec = jnp.minimum(evplimit, evpprec)
        pos = flxprec > 0.0
        work1 = jnp.clip(_safe_div(flxsntm, flxprec, pos), 0.0, 1.0)
        evpsnow = jnp.where(pos, evpprec * work1, 0.0)
        evpvint = evpvint + evpprec * pdel_k / g
        ntprprd = prd_k - evpprec
        work1 = jnp.where(pos, jnp.clip(_safe_div(flxsnow, flxprec, pos), 0.0, 1.0), 0.0)
        work2 = jnp.where(snowmlt > 0.0, 0.0, jnp.maximum(fsn_k, work1))
        ntsnprd = prd_k * work2 - evpsnow - snowmlt
        flxprec_n = jnp.maximum(flxprec + ntprprd * pdel_k / g, 0.0)
        flxsnow_n = jnp.maximum(flxsnow + ntsnprd * pdel_k / g, 0.0)
        tend_s = -evpprec * latvap + ntsnprd * latice
        tend_q = evpprec
        new_carry = tuple(a.astype(dtype) for a in (flxprec_n, flxsnow_n, evpvint))
        outs = tuple(a.astype(dtype) for a in
                     (tend_s, tend_q, ntprprd, ntsnprd, flxprec_n, flxsnow_n))
        return new_carry, outs

    zero = jnp.zeros((ncol,), dtype)
    (flx_p, flx_s, _), (tend_s, tend_q, ntprprd, ntsnprd, flxprec_lo, flxsnow_lo) = \
        _scan_levels(body, (zero, zero, zero),
                     (t, q, qs, pdel, prdprec, cldfrc, fsnow_conv), descending=False)
    flxprec = jnp.concatenate([jnp.zeros((ncol, 1), dtype), flxprec_lo], axis=1)
    flxsnow = jnp.concatenate([jnp.zeros((ncol, 1), dtype), flxsnow_lo], axis=1)
    return ZMEvap(tend_s=tend_s, tend_q=tend_q, prec=flx_p, snow=flx_s,
                  ntprprd=ntprprd, ntsnprd=ntsnprd, flxprec=flxprec, flxsnow=flxsnow)


# ---------------------------------------------------------------------------
# momtran (zm_conv.F90:1982-2382)
# ---------------------------------------------------------------------------
class ZMMomtran(NamedTuple):
    dudt: jax.Array
    dvdt: jax.Array
    seten: jax.Array   # KE-dissipation heating, J/kg/s
    pguall: jax.Array  # (2, ncol, nlev)
    pgdall: jax.Array
    icwu: jax.Array
    icwd: jax.Array


def _momtran_one(const, mu, md, du, eu, ed, dp, jt, mx, msg, momcu, momcd):
    ncol, nlev = const.shape
    K = _lev(nlev)
    dtype = const.dtype
    const_km1 = jnp.concatenate([const[:, :1], const[:, :-1]], axis=1)   # km1 = max(1,k-1)
    const_kp1 = jnp.concatenate([const[:, 1:], const[:, -1:]], axis=1)   # kp1 = min(pver,k+1)
    dp_km1 = jnp.concatenate([dp[:, :1], dp[:, :-1]], axis=1)
    chat = 0.5 * (const + const_km1)
    mu_kp1 = jnp.concatenate([mu[:, 1:], mu[:, -1:]], axis=1)
    md_kp1 = jnp.concatenate([md[:, 1:], md[:, -1:]], axis=1)
    interior = (K >= 2) & (K <= nlev - 1)
    mududp = mu * (const - const_km1) / dp_km1 + mu_kp1 * (const_kp1 - const) / dp
    mddudp = md * (const - const_km1) / dp_km1 + md_kp1 * (const_kp1 - const) / dp
    pgu = jnp.where(interior, -momcu * 0.5 * mududp, 0.0)
    pgd = jnp.where(interior, -momcd * 0.5 * mddudp, 0.0)
    mududp_b = mu * (const - const_km1) / dp_km1
    mddudp_b = md * (const - const_km1) / dp_km1
    pgu = jnp.where(K == nlev, -momcu * mududp_b, pgu)
    pgd = jnp.where(K == nlev, -momcd * mddudp_b, pgd)

    # updraft from the bottom (lines 2178-2199)
    def up(carry, xk):
        conu_p = carry
        k, mu_k, mu_n, du_k, eu_k, const_k, dp_k, pgu_k, chat_k = xk
        mupdudp = mu_k + du_k * dp_k
        on = mupdudp > _MBSTH
        bottom = k == nlev
        num = jnp.where(bottom, 0.0, mu_n * conu_p) + eu_k * const_k * dp_k + pgu_k * dp_k
        conu_k = jnp.where(on, _safe_div(num, mupdudp, on), chat_k)
        return conu_k.astype(dtype), conu_k.astype(dtype)

    zero = jnp.zeros((ncol,), dtype)
    _, conu = _scan_levels(up, zero, (mu, _kp1(mu), du, eu, const, dp, pgu, chat),
                           descending=True)

    # downdraft from the top (lines 2190-2216); note the oracle's k=2 form
    # divides ONLY the pgd term by md (zm_conv.F90:2189), reproduced as is.
    def down(carry, xk):
        cond_p = carry
        k, md_k, md_m, ed_m, const_m, dp_m, pgd_m, chat_k = xk
        on = (md_k < -_MBSTH) & (k >= 2)
        md_s = jnp.where(on, md_k, 1.0)
        c2 = (-ed_m * const_m * dp_m) - pgd_m * dp_m / md_s
        c3 = (md_m * cond_p - ed_m * const_m * dp_m - pgd_m * dp_m) / md_s
        cond_k = jnp.where(on, jnp.where(k == 2, c2, c3), chat_k)
        return cond_k.astype(dtype), cond_k.astype(dtype)

    _, cond = _scan_levels(down, zero, (md, _km1(md), _km1(ed), _km1(const), _km1(dp),
                                        _km1(pgd), chat), descending=False)

    conu_kp1 = jnp.concatenate([conu[:, 1:], conu[:, -1:]], axis=1)
    cond_kp1 = jnp.concatenate([cond[:, 1:], cond[:, -1:]], axis=1)
    chat_kp1 = jnp.concatenate([chat[:, 1:], chat[:, -1:]], axis=1)
    dcondt = (mu_kp1 * (conu_kp1 - chat_kp1) - mu * (conu - chat)
              + md_kp1 * (cond_kp1 - chat_kp1) - md * (cond - chat)) / dp
    dcondt_mx = (-mu * (conu - chat) - md * (cond - chat)) / dp
    dcondt = jnp.where(K == mx[:, None], dcondt_mx, dcondt)
    dcondt = jnp.where(K >= jt[:, None], dcondt, 0.0)
    mflux = jnp.where(K >= jt[:, None], -mu * (conu - chat) - md * (cond - chat), 0.0)
    mflux = jnp.concatenate([mflux, jnp.zeros((ncol, 1), dtype)], axis=1)
    return dcondt, mflux, -pgu, -pgd, conu, cond


def momtran(u, v, mu, md, du, eu, ed, dp, jt, mx, msg, dt, *, momcu, momcd):
    """Port of ``momtran`` for the (u, v) pair; ``dt`` = ztodt."""
    ncol, nlev = u.shape
    K = _lev(nlev)
    dudt, mfu, pgu_u, pgd_u, icwu_u, icwd_u = _momtran_one(
        u, mu, md, du, eu, ed, dp, jt, mx, msg, momcu, momcd)
    dvdt, mfv, pgu_v, pgd_v, icwu_v, icwd_v = _momtran_one(
        v, mu, md, du, eu, ed, dp, jt, mx, msg, momcu, momcd)
    in_top = K >= jt[:, None]
    windf_u = jnp.where(in_top, u - (mfu[:, 1:] - mfu[:, :-1]) * dt / dp, 0.0)
    windf_v = jnp.where(in_top, v - (mfv[:, 1:] - mfv[:, :-1]) * dt / dp, 0.0)
    u_km1 = jnp.concatenate([u[:, :1], u[:, :-1]], axis=1)
    v_km1 = jnp.concatenate([v[:, :1], v[:, :-1]], axis=1)
    u_kp1 = jnp.concatenate([u[:, 1:], u[:, -1:]], axis=1)
    v_kp1 = jnp.concatenate([v[:, 1:], v[:, -1:]], axis=1)
    utop, vtop = 0.5 * (u + u_km1), 0.5 * (v + v_km1)
    ubot, vbot = 0.5 * (u_kp1 + u), 0.5 * (v_kp1 + v)
    fket = utop * mfu[:, :-1] + vtop * mfv[:, :-1]
    fkeb = ubot * mfu[:, 1:] + vbot * mfv[:, 1:]
    ketend_cons = (fket - fkeb) / dp
    ketend = ((windf_u ** 2 + windf_v ** 2) - (u ** 2 + v ** 2)) * 0.5 / dt
    seten = jnp.where(in_top, ketend_cons - ketend, 0.0)
    return ZMMomtran(dudt=dudt, dvdt=dvdt, seten=seten,
                     pguall=jnp.stack([pgu_u, pgu_v]), pgdall=jnp.stack([pgd_u, pgd_v]),
                     icwu=jnp.stack([icwu_u, icwu_v]), icwd=jnp.stack([icwd_u, icwd_v]))


# ---------------------------------------------------------------------------
# convtran (zm_conv.F90:1643-1978), moist constituent, fracis = 1
# ---------------------------------------------------------------------------
def convtran(const, mu, md, du, eu, ed, dp, jt, mx, msg):
    """Port of ``convtran`` for one moist tracer (``dqdt`` [1/s])."""
    ncol, nlev = const.shape
    K = _lev(nlev)
    dtype = const.dtype
    const_km1 = jnp.concatenate([const[:, :1], const[:, :-1]], axis=1)
    const_kp1 = jnp.concatenate([const[:, 1:], const[:, -1:]], axis=1)
    minc = jnp.minimum(const_km1, const)
    maxc = jnp.maximum(const_km1, const)
    cdifr = jnp.where(minc < 0.0, 0.0,
                      jnp.abs(const - const_km1) / jnp.maximum(maxc, _CONVTRAN_SMALL))
    geo = cdifr > _LOGMEAN_TOL
    cabv = jnp.maximum(const_km1, maxc * _CHAT_FLOOR_FRAC)
    cbel = jnp.maximum(const, maxc * _CHAT_FLOOR_FRAC)
    cabv_s = jnp.where(geo, cabv, 1.0)
    cbel_s = jnp.where(geo, cbel, 2.0)
    chat = jnp.where(geo, jnp.log(cabv_s / cbel_s) / (cabv_s - cbel_s) * cabv_s * cbel_s,
                     0.5 * (const + const_km1))

    def up(carry, xk):
        conu_p = carry
        k, mu_k, mu_n, du_k, eu_k, const_k, dp_k, chat_k = xk
        mupdudp = mu_k + du_k * dp_k
        on = mupdudp > _MBSTH
        num = jnp.where(k == nlev, 0.0, mu_n * conu_p) + eu_k * const_k * dp_k
        conu_k = jnp.where(on, _safe_div(num, mupdudp, on), chat_k)
        return conu_k.astype(dtype), conu_k.astype(dtype)

    zero = jnp.zeros((ncol,), dtype)
    _, conu = _scan_levels(up, zero, (mu, _kp1(mu), du, eu, const, dp, chat), descending=True)

    def down(carry, xk):
        cond_p = carry
        k, md_k, md_m, ed_m, const_m, dp_m, chat_k = xk
        on = (md_k < -_MBSTH) & (k >= 2)
        md_s = jnp.where(on, md_k, 1.0)
        c2 = (-ed_m * const_m * dp_m) / md_s
        c3 = (md_m * cond_p - ed_m * const_m * dp_m) / md_s
        cond_k = jnp.where(on, jnp.where(k == 2, c2, c3), chat_k)
        return cond_k.astype(dtype), cond_k.astype(dtype)

    _, cond = _scan_levels(down, zero, (md, _km1(md), _km1(ed), _km1(const), _km1(dp), chat),
                           descending=False)

    mu_kp1 = jnp.concatenate([mu[:, 1:], mu[:, -1:]], axis=1)
    md_kp1 = jnp.concatenate([md[:, 1:], md[:, -1:]], axis=1)
    conu_kp1 = jnp.concatenate([conu[:, 1:], conu[:, -1:]], axis=1)
    cond_kp1 = jnp.concatenate([cond[:, 1:], cond[:, -1:]], axis=1)
    chat_kp1 = jnp.concatenate([chat[:, 1:], chat[:, -1:]], axis=1)
    fluxin = (mu_kp1 * conu_kp1 + mu * jnp.minimum(chat, const_km1)
              - (md * cond + md_kp1 * jnp.minimum(chat_kp1, const_kp1)))
    fluxout = (mu * conu + mu_kp1 * jnp.minimum(chat_kp1, const)
               - (md_kp1 * cond_kp1 + md * jnp.minimum(chat, const)))
    netflux = fluxin - fluxout
    netflux = jnp.where(jnp.abs(netflux) < jnp.maximum(fluxin, fluxout) * _NETFLUX_TOL,
                        0.0, netflux)
    dcondt = netflux / dp
    fin_mx = mu * jnp.minimum(chat, const_km1) - md * cond
    fout_mx = mu * conu - md * jnp.minimum(chat, const)
    net_mx = fin_mx - fout_mx
    net_mx = jnp.where(jnp.abs(net_mx) < jnp.maximum(fin_mx, fout_mx) * _NETFLUX_TOL, 0.0, net_mx)
    dcondt = jnp.where(K == mx[:, None], net_mx / dp, dcondt)
    dcondt = jnp.where(K > mx[:, None], 0.0, dcondt)
    dcondt = jnp.where(K >= jt[:, None], dcondt, 0.0)
    return dcondt
