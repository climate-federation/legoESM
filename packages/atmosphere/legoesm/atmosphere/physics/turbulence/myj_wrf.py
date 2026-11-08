"""WRF v4.8.0 Mellor-Yamada-Janjic (MYJ) PBL: JAX port of the column core.

Oracle: WRF ``v4.8.0`` (06d4240ae989cc3e50af412bb472df3d9048783c), file
``phys/module_bl_myjpbl.F``, subroutine ``MYJPBL`` and everything it calls
(MIXLEN, PRODQ2, DIFCOF, VDIFQ, VDIFH, VDIFV): Mellor-Yamada level 2.5 as
extended by Janjic (2002, NCEP Office Note 437).  One column, ``LOWLYR = 1``
(WRF-ARW), cloud ice present (``NSPEC = 4``: THE, q, qc, qi are mixed).

The MYJSFC surface-layer fields (``ustar``, ``akhs``, ``akms``, ``thz0``,
``qz0``, ``uz0``, ``vz0``, ``ct``, ``chklowq``, ``elflx``, ``tsk``, ``qsfc``)
are INPUTS; MYJSFC itself is not ported here.

Level arrays are BOTTOM-UP on input and output (index 0 = lowest mass level,
WRF ``k = 1``), moisture as WRF mixing ratios.  Inside, the port follows the
oracle's flipped, TOP-DOWN column index by index: Python index ``i`` is MYJ's
``K = i + 1``; interface heights ``z`` hold ``Z(K)`` at ``i = K - 1`` (``Z(L+1)``
is the ground).  Sequential Fortran loops (sums, Thomas recursions, the
interface-height accumulation) are ``lax.scan`` in the oracle's order.

Differentiability (RULE 5): every clamp is ``core.smooth`` smooth_max/min and
every regime branch a ``sigmoid_switch`` blend, each with its window width
from :class:`MYJConfig`; width 0 is the oracle's hard switch (the oracle test
runs all widths 0).  Integer level searches (PBL top ``LPBL``, mixed-layer
level ``LMXL``, the counter-gradient inversion level), the exact-equality
tests (``Q2 == EPSQ2``, ``ELM == EPSL``, ``ELOQN > EPS1``) and the data
branches (land/sea, snow/ice, ``QFC1 > 0``) stay hard: they are piecewise
constant, and at nonzero widths the smooth clamps keep the equality tests
from ever firing.  A branch evaluated but not selected takes a safe argument.

Constants: the scheme's own are module-level below (module_bl_myjpbl.F
header); the host constants it USEs from WRF's ``module_model_constants`` are
legoESM's physical constants and WRF's MYJ scheme constants (``EPSQ2``,
``PQ0``, ``A2``-``A4``), which is how the oracle fixture was generated.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from legoesm import constants as _C
from legoesm.core.smooth import sigmoid_switch, smooth_max, smooth_min, smoothstep_quintic

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 MYJ PBL column core (MYJPBL): MY level-2.5 mixing length "
        "(Blackadar asymptote from the TKE integral), implicit-in-time q^2 "
        "production/dissipation, level-2.5 Sm/Sh exchange coefficients, "
        "implicit vertical diffusion of q^2, THE, q, qc, qi, u, v with the "
        "MYJSFC surface exchange as lower boundary condition."
    ),
    "inputs": {
        "dz": "m", "u": "m/s", "v": "m/s", "th": "K", "t": "K", "exner": "1",
        "pmid": "Pa", "pint": "Pa", "qv": "kg/kg (mixing ratio)",
        "qcw": "kg/kg (mixing ratio)", "qci": "kg/kg (mixing ratio)",
        "tke": "m^2/s^2", "ustar": "m/s", "akhs": "m/s", "akms": "m/s",
        "thz0": "K", "qz0": "kg/kg (specific)", "uz0": "m/s", "vz0": "m/s",
        "ct": "K", "chklowq": "1", "elflx": "W/m^2", "tsk": "K",
        "qsfc": "kg/kg (specific)", "xland": "1 land / 2 water",
        "snow": "kg/m^2", "sice": "1", "ht": "m", "dt": "s",
    },
    "outputs": {
        "rublten": "m/s^2", "rvblten": "m/s^2", "rthblten": "K/s",
        "rqvblten": "kg/kg/s (mixing ratio)", "rqcblten": "kg/kg/s",
        "rqiblten": "kg/kg/s", "tke": "m^2/s^2", "el": "m",
        "exch_h": "m^2/s", "pblh": "m", "kpbl": "1", "mixht": "m",
        "ct": "K", "thz0": "K", "qz0": "kg/kg", "qsfc": "kg/kg",
    },
    "sign_convention": (
        "z up, arrays bottom-up; surface exchange AKHS*rho*(THZ0 - TH1) > 0 "
        "heats the column (upward heat flux); tendencies are d/dt of the "
        "column state; exchange coefficients >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Janjic (2002) NCEP Office Note 437; Mellor & Yamada (1982) Rev. "
        "Geophys. 20, 851-875; WRF v4.8.0 phys/module_bl_myjpbl.F"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_myj_oracle.py: every output "
        "matches the running WRF oracle on 24 columns, 3 steps"
    ),
}

__param_spec__ = {
    "MYJConfig": {
        "scheme_key": "atm.turb.MYJConfig",
        "params": {},
        "excluded": {
            "width_gh_K_m": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_gm_s2": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_ratio": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_el_m": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_el0_m": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_q2_m2_s2": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_loq_s": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_eqol2_s2": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_dth_K": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_snow_kg_m2": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_sice": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
            "width_rel": "RULE 5 smoothing window width (0 = oracle hard switch); numerics, not trainable",
        },
    },
}

# --- MYJ scheme constants (module_bl_myjpbl.F header) ---
_VKARMAN = 0.4
_EPS1 = 1.0e-12
_EPS2 = 0.0
_EPSL = 0.32
_EPSRU = 1.0e-7
_EPSRS = 1.0e-7
_EPSTRB = 1.0e-24
_FH = 1.01
_ALPH = 0.30
_BETA = 1.0 / 273.0
_EL0MAX = 1000.0
_EL0MIN = 1.0
_ELFC = 0.23 * 0.5
_A1 = 0.659888514560862645
_A2X = 0.6574209922667784586
_B1 = 11.87799326209552761
_B2 = 7.226971804046074028
_C1 = 0.000830955950095854396
_ELZ0 = 0.0
_ESQ = 5.0
_SEAFC = 0.98
_EPSGH = 1.0e-9
# --- port numerics (not in the oracle) ---
_EQOL2_FLOOR = 1.0e-12   # [s^-2] turbulent-branch (q/L)^2 floor, fp32 overflow guard (vote X2)
# --- host scheme constants (WRF share/module_model_constants.F) ---
_EPSQ2 = 0.2
_PQ0 = 379.90516
_A2 = 17.2693882
_A3 = 273.16
_A4 = 35.86
# --- derived (module_bl_myjpbl.F header, same expressions) ---
_CP = _C.c_pd
_CAPA = _C.R_d / _CP
_RLIVWV = _C.L_s / _C.L_v  # latent-ok: WRF MYJ oracle constant
_ELOCP = 2.72e6 / _CP
_P608 = _C.R_v / _C.R_d - 1.0
_PQ0SEA = _PQ0 * _SEAFC
_BTG = _BETA * _C.g
_RB1 = 1.0 / _B1
_ESQHF = 0.5 * _ESQ
_ADNH = 9. * _A1 * _A2X * _A2X * (12. * _A1 + 3. * _B2) * _BTG * _BTG
_ADNM = 18. * _A1 * _A1 * _A2X * (_B2 - 3. * _A2X) * _BTG
_ANMH = -9. * _A1 * _A2X * _A2X * _BTG * _BTG
_ANMM = -3. * _A1 * _A2X * (3. * _A2X + 3. * _B2 * _C1 + 18. * _A1 * _C1 - _B2) * _BTG
_BDNH = 3. * _A2X * (7. * _A1 + _B2) * _BTG
_BDNM = 6. * _A1 * _A1
_BEQH = _A2X * _B1 * _BTG + 3. * _A2X * (7. * _A1 + _B2) * _BTG
_BEQM = -_A1 * _B1 * (1. - 3. * _C1) + 6. * _A1 * _A1
_BNMH = -_A2X * _BTG
_BNMM = _A1 * (1. - 3. * _C1)
_BSHH = 9. * _A1 * _A2X * _A2X * _BTG
_BSHM = 18. * _A1 * _A1 * _A2X * _C1
_BSMH = -3. * _A1 * _A2X * (3. * _A2X + 3. * _B2 * _C1 + 12. * _A1 * _C1 - _B2) * _BTG
_CESH = _A2X
_CESM = _A1 * (1. - 3. * _C1)
_AEQH = (9. * _A1 * _A2X * _A2X * _B1 * _BTG * _BTG
         + 9. * _A1 * _A2X * _A2X * (12. * _A1 + 3. * _B2) * _BTG * _BTG)
_AEQM = (3. * _A1 * _A2X * _B1 * (3. * _A2X + 3. * _B2 * _C1 + 18. * _A1 * _C1 - _B2) * _BTG
         + 18. * _A1 * _A1 * _A2X * (_B2 - 3. * _A2X) * _BTG)
_REQU = -_AEQH / _AEQM
_EPSGM = _REQU * _EPSGH
_UBRYL = ((18. * _REQU * _A1 * _A1 * _A2X * _B2 * _C1 * _BTG
           + 9. * _A1 * _A2X * _A2X * _B2 * _BTG * _BTG) / (_REQU * _ADNM + _ADNH))
_UBRY = (1. + _EPSRS) * _UBRYL
_UBRY3 = 3. * _UBRY
_AUBH = 27. * _A1 * _A2X * _A2X * _B2 * _BTG * _BTG - _ADNH * _UBRY3
_AUBM = 54. * _A1 * _A1 * _A2X * _B2 * _C1 * _BTG - _ADNM * _UBRY3
_BUBH = (9. * _A1 * _A2X + 3. * _A2X * _B2) * _BTG - _BDNH * _UBRY3
_BUBM = 18. * _A1 * _A1 * _C1 - _BDNM * _UBRY3
_CUBR = 1. - _UBRY3
_RCUBR = 1. / _CUBR
_B1_23 = _B1 ** (2. / 3.)


class MYJConfig(NamedTuple):
    """RULE 5 window widths (0 = the oracle's hard switch), each 0.01 x the
    quantity's scale near its switch (GM: its floor EPSGM).  GH, GM, q^2 and
    (q/L)^2 sit at the bottom of their distributions where they are switched
    (q^2 is AT EPSQ2 above the PBL): windows of 0.01 x their bulk scale moved
    exch_h by 80% and broke FD/AD agreement on the oracle columns.  These
    widths, measured on the 24 oracle columns: worst per-column gap 1.2e-2
    (EL, cold start), else <= 1.4e-3; FD/AD 1.000-1.003."""
    width_gh_K_m: float = 1.0e-7       # GH, scale near switch 1e-5 K/m
    width_gm_s2: float = 7.1e-13       # GM, scale = its floor EPSGM 7.1e-11 s^-2
    width_ratio: float = 7.1e-4        # GM/GH vs REQU, scale REQU
    width_el_m: float = 0.1            # EL, ELM, scale 10 m
    width_el0_m: float = 1.0           # EL0, scale 100 m
    width_q2_m2_s2: float = 1.0e-4     # q^2, scale near EPSQ2 1e-2 m^2/s^2
    width_loq_s: float = 0.1           # L/q, scale 10 s
    width_eqol2_s2: float = 1.0e-6     # (q/L)^2 equilibrium, scale near 0 1e-4 s^-2
    width_rel: float = 1.0e-3          # EL/ELM smoothing ratio
    width_dth_K: float = 1.0e-5        # THE layer difference at the inversion test
    width_snow_kg_m2: float = 0.01     # snow water equivalent, scale 1 kg/m^2
    width_sice: float = 0.01           # sea-ice fraction, scale 1


ORACLE_HARD = MYJConfig(*([0.0] * len(MYJConfig._fields)))


def _sh(w):
    return math.inf if w == 0 else 1.0 / w


def _ge0(x, w):
    """x >= 0 (hard at w = 0)."""
    return sigmoid_switch(x, _sh(w))


def _le0_incl(x, w):
    """x <= 0 with the boundary INCLUDED at any width: exactly 1 for x <= 0,
    C2 ramp to 0 at x = w (hard at w = 0).  For tests the oracle relies on
    being inclusive at an exact tie (calm neutral layers sit exactly on
    GM/GH = REQU and on |GH| <= EPSGH by construction of EPSGM, EPSGH)."""
    if w == 0:
        return (x <= 0).astype(x.dtype)
    return 1.0 - smoothstep_quintic(x / w)


def _gt0_incl0(x, w):
    """x > 0 that is exactly 0 for x <= 0 at any width (C2 ramp to 1 at x = w)."""
    if w == 0:
        return (x > 0).astype(jnp.result_type(x, float))
    return smoothstep_quintic(x / w)


def _gt0(x, w):
    """x > 0 (hard at w = 0)."""
    return (x > 0).astype(x.dtype) if w == 0 else sigmoid_switch(x, _sh(w))


def _mx(a, b, w):
    return smooth_max(a, b, _sh(w))


def _mn(a, b, w):
    return smooth_min(a, b, _sh(w))


def _blend(w, a, b):
    """w*a + (1-w)*b, exactly a / b where w is exactly 1 / 0 (the hard
    switch; also keeps a non-finite unselected branch out)."""
    return jnp.where(w == 1.0, a, jnp.where(w == 0.0, b, w * a + (1.0 - w) * b))


def _sqrt0(x):
    """sqrt of x >= 0, derivative 0 (not NaN) where x <= 0 (unselected branch)."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _safe(x):
    """Nonzero denominator for a branch evaluated but not selected."""
    return jnp.where(x != 0.0, x, 1.0)


def _seq_sum(x):
    s, _ = lax.scan(lambda c, xi: (c + xi, None), jnp.zeros((), x.dtype), x)
    return s


def _last_true(mask, default):
    """Largest MYJ index K (1-based) where ``mask`` (Python index K-1) holds,
    else ``default``."""
    k = jnp.arange(1, mask.shape[0] + 1)
    return jnp.where(jnp.any(mask), jnp.max(jnp.where(mask, k, 0)), default)


def _rev_cumprod_after(a):
    """out[i] = prod(a[i+1:]) (1 for the last entry), length len(a) + 1 with
    out[len(a)] = 1: 'no later entry holds' for soft 0/1 flags 1 - a."""
    rp = jnp.cumprod(a[::-1])[::-1]
    return jnp.concatenate([rp[1:], jnp.ones((2,), a.dtype)])


def mixlen(u, v, t, the, q, cwm, q2, z, ct, cfg):
    """MIXLEN: GM, GH, EL (K = 1..L-1), PBL top LPBL, mixed-layer LMXL,
    PBLH, MIXHT; returns ct = 0 as the oracle does."""
    L = u.shape[0]
    # PBL top: lowest K in 1..L-1 with Q2(K) <= EPSQ2*FH, else 1
    lpbl = _last_true(q2[:L - 1] <= _EPSQ2 * _FH, 1)
    pblh = z[lpbl] - z[L]                                   # Z(LPBL+1) - Z(L+1)
    dth = the[:-1] - the[1:]                                # DTH(K), K = 1..L-1
    # counter-gradient increment at the lowest K in 1..L-2 with DTH(K) > 0 >= DTH(K+1)
    # soft one-hot at the lowest such K (hard: exactly the oracle's level)
    cond = (_gt0(dth[:-1], cfg.width_dth_K) * _le0_incl(dth[1:], cfg.width_dth_K))
    w_inv = cond * _rev_cumprod_after(1.0 - cond)[:L - 2]
    dth = dth + jnp.concatenate([w_inv * ct, jnp.zeros((1,), dth.dtype)])
    rdz = 2.0 / (z[:L - 1] - z[2:])
    gml = ((u[:-1] - u[1:]) ** 2 + (v[:-1] - v[1:]) ** 2) * rdz * rdz
    gm = _mx(gml, _EPSGM, cfg.width_gm_s2)
    tem = (t[:-1] + t[1:]) * 0.5
    thm = (the[:-1] + the[1:]) * 0.5
    a = thm * _P608
    b = (_ELOCP / tem - 1. - _P608) * thm
    ghl = (dth * ((q[:-1] + q[1:] + cwm[:-1] + cwm[1:]) * (0.5 * _P608) + 1.)
           + (q[:-1] - q[1:] + cwm[:-1] - cwm[1:]) * a
           + (cwm[:-1] - cwm[1:]) * b) * rdz
    # |GHL| <= EPSGH -> EPSGH, then GH >= EPSGH is the stable branch: so the
    # stable branch is GHL >= -EPSGH with GH = max(GHL, EPSGH), the unstable
    # one GHL < -EPSGH with GH = GHL (same values as the oracle's two steps,
    # and neither formula is ever fed 0 < GH < EPSGH when smoothed)
    stab = _le0_incl(-(ghl + _EPSGH), cfg.width_gh_K_m)
    gh_st = _mx(ghl, _EPSGH, cfg.width_gh_K_m)
    gh = _blend(stab, gh_st, ghl)
    # maximum mixing length: stable / forbidden / unstable regimes
    forb = _le0_incl(gm / gh_st - _REQU, cfg.width_ratio)
    aubr = (_AUBM * gm + _AUBH * gh_st) * gh_st
    bubr = _BUBM * gm + _BUBH * gh_st
    qol2st = (-0.5 * bubr + _sqrt0(bubr * bubr * 0.25 - aubr * _CUBR)) * _RCUBR
    elm_st = _mx(_sqrt0(1. / _safe(qol2st) * q2[:-1]), _EPSL, cfg.width_el_m)
    aden = (_ADNM * gm + _ADNH * ghl) * ghl
    bden = _BDNM * gm + _BDNH * ghl
    qol2un = -0.5 * bden + _sqrt0(bden * bden * 0.25 - aden)
    elm_un = _mx(_sqrt0(1. / _safe(qol2un + _EPSRU) * q2[:-1]), _EPSL, cfg.width_el_m)
    elm = _blend(stab, _blend(forb, _EPSL, elm_st), elm_un)
    forb_hard = (gh >= _EPSGH) & (gm / jnp.where(gh > 0.0, gh, 1.0) <= _REQU)
    lmxl = _last_true(forb_hard, L)
    # ELM(LMH-1) == EPSL <=> forbidden, or the raw length <= EPSL (hard test)
    st_raw = _sqrt0(1. / _safe(qol2st) * q2[:-1])
    un_raw = _sqrt0(1. / _safe(qol2un + _EPSRU) * q2[:-1])
    at_floor = jnp.where(ghl >= -_EPSGH, forb_hard | (st_raw <= _EPSL), un_raw <= _EPSL)
    lmxl = jnp.where(at_floor[L - 2], L, lmxl)
    mixht = z[lmxl - 1] - z[L]                              # Z(LMXL) - Z(L+1)
    # Blackadar asymptote from the q-weighted height of the PBL
    # K >= LPBL <=> no K' in (K, L-1] has Q2 <= EPSQ2*FH (soft, exact when hard)
    inpbl = _rev_cumprod_after(1.0 - _le0_incl(q2[:L - 1] - _EPSQ2 * _FH, cfg.width_q2_m2_s2))
    q1 = inpbl * _sqrt0(q2)
    qdzl = (q1[:-1] + q1[1:]) * (z[1:L] - z[2:])
    szq = _seq_sum((z[1:L] + z[2:] - z[L] - z[L]) * qdzl)
    sq = _seq_sum(qdzl)
    el0 = _mn(_ALPH * szq * 0.5 / sq, _EL0MAX, cfg.width_el0_m)
    el0 = _mx(el0, _EL0MIN, cfg.width_el0_m)
    el_above = _mn((z[:L - 1] - z[2:]) * _ELFC, elm, cfg.width_el_m)
    vkrmz = (z[1:L] - z[L]) * _VKARMAN
    el_in = _mn(vkrmz / (vkrmz / el0 + 1.), elm, cfg.width_el_m)
    el = _blend(inpbl[:L - 1], el_in, el_above)
    rel = el / elm
    rel_m = jnp.concatenate([rel[:1], rel[:-1]])
    rel_p = jnp.concatenate([rel[1:], rel[-1:]])
    srel = _mn(((rel_m + rel_p) * 0.5 + rel) * 0.5, rel, cfg.width_rel)
    w_srel = jnp.concatenate([jnp.zeros((1,), el.dtype), inpbl[:L - 3], jnp.zeros((1,), el.dtype)])
    el = _blend(w_srel, _mx(srel * elm, _EPSL, cfg.width_el_m), el)
    return gm, gh, el, pblh, lpbl, lmxl, mixht, jnp.zeros_like(ct), stab, gh_st


def prodq2(dt, ustar, gm, gh, el, q2, cfg, stab, gh_st):
    """PRODQ2: implicit q^2 production/dissipation, K = 1..L-1, then the
    surface value Q2(L).  ``stab``/``gh_st`` are MIXLEN's stable-regime
    weight and stable GH, so the forbidden-region test smooths exactly as
    MIXLEN's does."""
    aequ = (_AEQM * gm + _AEQH * gh) * gh
    bequ = _BEQM * gm + _BEQH * gh
    disc = bequ * bequ * 0.25 - aequ
    eqol2 = -0.5 * bequ + _sqrt0(disc)
    h1 = _ge0(_EPSTRB - (gm + gh * gh), cfg.width_gm_s2)
    h2 = stab * _le0_incl(gm / gh_st - _REQU, cfg.width_ratio)
    h3 = _ge0(_EPS2 - eqol2, cfg.width_eqol2_s2)
    noturb = 1.0 - (1.0 - h1) * (1.0 - h2) * (1.0 - h3)
    anum = (_ANMM * gm + _ANMH * gh) * gh
    bnum = _BNMM * gm + _BNMH * gh
    aden = (_ADNM * gm + _ADNH * gh) * gh
    bden = _BDNM * gm + _BDNH * gh
    cden = 1.
    arhs = -(anum * bden - bnum * aden) * 2.
    brhs = -anum * 4.
    crhs = -bnum * 2.
    q2l = q2[:-1]
    dloq1 = el / _sqrt0(jnp.where(q2l > 0.0, q2l, 1.0))
    # smoothed only: floor the turbulent branch's (q/L)^2, whose -5/2 power
    # overflows fp32 below ~3.9e-16 inside the blend; hard (oracle) form keeps 1/EQOL2
    if cfg.width_eqol2_s2 > 0:
        eloq21 = 1. / _mx(eqol2, _EQOL2_FLOOR, _EQOL2_FLOOR)
    else:
        eloq21 = 1. / jnp.where(eqol2 > 0.0, eqol2, 1.0)
    eloq11 = jnp.sqrt(eloq21)
    eloq31 = eloq21 * eloq11
    eloq41 = eloq21 * eloq21
    eloq51 = eloq21 * eloq31
    rden1 = 1. / (aden * eloq41 + bden * eloq21 + cden)
    rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1
    eloq12 = eloq11 + (dloq1 - eloq11) * jnp.exp(rhsp1 * dt)
    eloq12 = _mx(eloq12, _EPS1, cfg.width_loq_s)
    eloq22 = eloq12 * eloq12
    eloq32 = eloq22 * eloq12
    eloq42 = eloq22 * eloq22
    eloq52 = eloq22 * eloq32
    rden2 = 1. / (aden * eloq42 + bden * eloq22 + cden)
    rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + _RB1
    rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2
    rhst2 = rhs2 / _safe(rhsp2)
    eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * jnp.exp(rhsp2 * dt)
    eloqn = _mx(eloq13, _EPS1, cfg.width_loq_s)
    # ELOQN > EPS1 after the AMAX1 <=> ELOQ13 > EPS1; Q2 == EPSQ2 after the
    # AMAX1 <=> the raw Q2 <= EPSQ2: both written on the raw values, so the
    # smoothed tests are blends rather than equalities that never fire
    ok = _gt0(eloq13 - _EPS1, cfg.width_loq_s)
    q2raw = el * el / (eloqn * eloqn)
    floor = _ge0(_EPSQ2 - q2raw, cfg.width_q2_m2_s2)
    q2t = _blend(ok, _mx(q2raw, _EPSQ2, cfg.width_q2_m2_s2), _EPSQ2)
    elt = _blend(ok, _blend(floor, _EPSL, el), _EPSL)
    q2_new = _blend(noturb, _EPSQ2, q2t)
    el_new = _blend(noturb, _EPSL, elt)
    q2_sfc = _mx(_B1_23 * ustar * ustar, _EPSQ2, cfg.width_q2_m2_s2)
    return jnp.concatenate([q2_new, q2_sfc[None]]), el_new


def difcof(gm, gh, el, q2, z):
    """DIFCOF: level-2.5 Sm, Sh and the exchange coefficients AKM, AKH
    (K/dz, m/s) on the bottoms of layers K = 1..L-1."""
    L = q2.shape[0]
    eloq2 = el * el / q2[:-1]
    eloq4 = eloq2 * eloq2
    aden = (_ADNM * gm + _ADNH * gh) * gh
    bden = _BDNM * gm + _BDNH * gh
    besm = _BSMH * gh
    besh = _BSHM * gm + _BSHH * gh
    rden = 1. / (aden * eloq4 + bden * eloq2 + 1.)
    esm = (besm * eloq2 + _CESM) * rden
    esh = (besh * eloq2 + _CESH) * rden
    rdz = 2. / (z[:L - 1] - z[2:])
    elqdz = el * jnp.sqrt(q2[:-1]) * rdz
    return elqdz * esm, elqdz * esh


def vdifq(dt, q2, el, z):
    """VDIFQ: implicit vertical diffusion of q^2 (Q2(L) is the boundary value)."""
    L = q2.shape[0]
    dtoz = (dt + dt) / (z[:L - 2] - z[2:L])                 # K = 1..L-2
    akq = (jnp.sqrt((q2[:L - 2] + q2[1:L - 1]) * 0.5) * (el[:L - 2] + el[1:L - 1]) * _ESQHF
           / (z[1:L - 1] - z[2:L]))
    cr = -dtoz * akq
    cm1 = dtoz[0] * akq[0] + 1.

    def fwd(c, x):
        cm_p, rs_p = c
        dtozk, akq_p, akq_k, cr_p, q2k = x
        cf = -dtozk * akq_p / cm_p
        cm = -cr_p * cf + (akq_p + akq_k) * dtozk + 1.
        rs = -rs_p * cf + q2k
        return (cm, rs), (cm, rs)

    _, (cmr, rsr) = lax.scan(fwd, (cm1, q2[0]),
                             (dtoz[1:], akq[:-1], akq[1:], cr[:-1], q2[1:L - 2]))
    cm = jnp.concatenate([cm1[None], cmr])
    rsq2 = jnp.concatenate([q2[:1], rsr])
    dtozs = (dt + dt) / (z[L - 2] - z[L])
    akqs = jnp.sqrt((q2[L - 2] + q2[L - 1]) * 0.5) * (el[L - 2] + _ELZ0) * _ESQHF / (z[L - 1] - z[L])
    cf = -dtozs * akq[L - 3] / cm[L - 3]
    q2lm1 = ((dtozs * akqs * q2[L - 1] - rsq2[L - 3] * cf + q2[L - 2])
             / ((akq[L - 3] + akqs) * dtozs - cr[L - 3] * cf + 1.))

    def bwd(qn, x):
        crk, rsk, cmk = x
        qk = (-crk * qn + rsk) / cmk
        return qk, qk

    _, qs = lax.scan(bwd, q2lm1, (cr, rsq2, cm), reverse=True)
    return jnp.concatenate([qs, q2lm1[None], q2[L - 1:]])


def _thomas(dt, rkh, rho, s, z, rkct=None):
    """Shared forward sweep of VDIFH/VDIFV for one or more species
    s[..., K]; returns (cm, cr, rss, dtozs, cf) for the boundary/back steps."""
    L = rho.shape[0]
    dtoz = dt / (z[:L - 1] - z[1:L])                        # K = 1..L-1
    cr = -dtoz * rkh
    cm1 = dtoz[0] * rkh[0] + rho[0]
    if rkct is None:
        rs1 = s[..., 0] * rho[0]
        src = jnp.zeros_like(s[..., 1:L - 1])
    else:
        rs1 = -rkct[..., 0] * dtoz[0] + s[..., 0] * rho[0]
        src = (rkct[..., :L - 2] - rkct[..., 1:L - 1]) * dtoz[1:]

    def fwd(c, x):
        cm_p, rs_p = c
        dtozl, rkh_p, rkh_k, cr_p, rhok, sk, srck = x
        cf = -dtozl * rkh_p / cm_p
        cm = -cr_p * cf + (rkh_p + rkh_k) * dtozl + rhok
        if rkct is None:
            rs = -rs_p * cf + sk * rhok
        else:
            rs = -rs_p * cf + srck + sk * rhok
        return (cm, rs), (cm, rs)

    sm = jnp.moveaxis(s[..., 1:L - 1], -1, 0)
    srcm = jnp.moveaxis(src, -1, 0)
    _, (cmr, rsr) = lax.scan(fwd, (cm1, rs1),
                             (dtoz[1:], rkh[:-1], rkh[1:], cr[:-1], rho[1:L - 1], sm, srcm))
    cm = jnp.concatenate([cm1[None], cmr])
    rss = jnp.concatenate([rs1[None], rsr])                 # (L-1, ...)
    dtozs = dt / (z[L - 1] - z[L])
    cf = -dtozs * rkh[L - 2] / cm[L - 2]
    return cm, cr, rss, dtozs, cf


def _back(cm, cr, rss, s_bottom):
    def bwd(sn, x):
        crk, rsk, cmk = x
        sk = (-crk * sn + rsk) * (1. / cmk)
        return sk, sk

    _, ss = lax.scan(bwd, s_bottom, (cr, rss, cm), reverse=True)
    return jnp.concatenate([jnp.moveaxis(ss, 0, -1), s_bottom[..., None]], axis=-1)


def vdifh(dt, lpbl, sz0, rkhs, clow, cts, species, rkh, z, rho):
    """VDIFH: implicit diffusion of the mass species (rows of ``species``)."""
    L = rho.shape[0]
    kk = jnp.arange(1, L)
    rkhz = rkh * (z[:L - 1] - z[2:])
    rkct = jnp.where(kk[None, :] < lpbl, 0.0, rkhz[None, :] * cts[:, None] * 0.5)
    cm, cr, rss, dtozs, cf = _thomas(dt, rkh, rho, species, z, rkct)
    rkhh = rkh[L - 2]
    cmb = cr[L - 2] * cf
    rkss = rkhs * clow
    cmsb = -cmb + (rkhh + rkss) * dtozs + rho[L - 1]
    rssb = -rss[L - 2] * cf + rkct[:, L - 2] * dtozs + species[:, L - 1] * rho[L - 1]
    s_bot = (dtozs * rkss * sz0 + rssb) / cmsb
    return _back(cm, cr, rss, s_bot)


def vdifv(dt, uz0, vz0, rkms, uv, rkm, z, rho):
    """VDIFV: implicit diffusion of u, v (rows of ``uv``)."""
    L = rho.shape[0]
    cm, cr, rss, dtozs, cf = _thomas(dt, rkm, rho, uv, z)
    rkmh = rkm[L - 2]
    rcmvb = 1. / ((rkmh + rkms) * dtozs - cr[L - 2] * cf + rho[L - 1])
    dtozak = dtozs * rkms
    s_bot = (dtozak * jnp.stack([uz0, vz0]) - rss[L - 2] * cf + uv[:, L - 1] * rho[L - 1]) * rcmvb
    return _back(cm, cr, rss, s_bot)


class MYJOutput(NamedTuple):
    rublten: jnp.ndarray
    rvblten: jnp.ndarray
    rthblten: jnp.ndarray
    rqvblten: jnp.ndarray
    rqcblten: jnp.ndarray
    rqiblten: jnp.ndarray
    tke: jnp.ndarray
    el: jnp.ndarray          # EL_MYJ, bottom-up, 0 at k = 1
    exch_h: jnp.ndarray      # bottom-up, k = 1..L-1
    exch_m: jnp.ndarray      # momentum analogue of EXCH_H (AKM*DELTAZ; not an oracle output)
    dqdt: jnp.ndarray        # DQDT, specific-humidity tendency before the RQVBLTEN conversion (not an oracle output)
    pblh: jnp.ndarray
    kpbl: jnp.ndarray
    mixht: jnp.ndarray
    ct: jnp.ndarray
    thz0: jnp.ndarray
    qz0: jnp.ndarray
    qsfc: jnp.ndarray


def myjpbl_column(dt, ht, dz, pmid, pint, th, t, exner, qv, qcw, qci, u, v, tke,
                  ustar, akhs, akms, thz0, qz0, uz0, vz0, ct, chklowq, elflx, tsk,
                  qsfc, xland, snow, sice, cfg: MYJConfig = MYJConfig()):
    """One MYJPBL call on one column (STEPBL = 1, LOWLYR = 1); level inputs
    bottom-up, ``pint`` has L+1 interfaces (surface first)."""
    L = dz.shape[0]
    rdt = 1. / dt
    f = lambda x: x[::-1]                                   # bottom-up <-> MYJ K
    # ZINT(K) = ZINT(K+1) + DZ(KFLIP), from the ground up
    _, zup = lax.scan(lambda c, d: (c + d, c + d), jnp.asarray(ht, dz.dtype), dz)
    z = jnp.concatenate([f(zup), jnp.asarray(ht, dz.dtype)[None]])
    ape = 1. / exner
    cwm = qcw + qci
    the = (cwm * (-_ELOCP / t) + 1.) * th
    qk = f(qv / (1. + qv))
    thek, cwmk, tk, uk, vk = f(the), f(cwm), f(t), f(u), f(v)
    q2 = f(2. * tke)
    gm, gh, el, pblh, lpbl, lmxl, mixht, ct_out, stab, gh_st = mixlen(
        uk, vk, tk, thek, qk, cwmk, q2, z, ct, cfg)
    q2, el = prodq2(dt, ustar, gm, gh, el, q2, cfg, stab, gh_st)
    akm, akh = difcof(gm, gh, el, q2, z)
    kpbl = L - lpbl + 1
    deltaz = 0.5 * (z[:L - 1] - z[2:])                      # at MYJ K = 1..L-1
    exch_h = f(akh * deltaz)                                # bottom-up k = 1..L-1
    exch_m = f(akm * deltaz)
    q2 = vdifq(dt, q2, el, z)
    q2 = _mx(q2, _EPSQ2, cfg.width_q2_m2_s2)
    tke_out = 0.5 * f(q2)
    el_myj = jnp.concatenate([jnp.zeros((1,), el.dtype), f(el)])
    # --- main integration: mass species ---
    rhok = f(pmid) / (_C.R_d * tk * (1. + _P608 * qk - cwmk))
    akhk = akh * 0.5 * (rhok[:-1] + rhok[1:])
    psfc = pint[0]
    thsk = tsk * (_C.p_ref / psfc) ** _CAPA
    seamask = xland - 1.
    thz0 = (1. - seamask) * thsk + seamask * thz0
    akhs_dens = akhs * rhok[L - 1]
    qfc1 = _C.L_v * chklowq * akhs_dens  # latent-ok: WRF MYJ oracle constant
    # SNOW > 0 .OR. SICE > 0.5 -> sublimation heat (soft OR; exactly 0 at snow <= 0)
    w_ice = 1.0 - (1.0 - _gt0_incl0(snow, cfg.width_snow_kg_m2)) * (1.0 - _gt0(sice - 0.5, cfg.width_sice))
    qfc1 = _blend(w_ice, qfc1 * _RLIVWV, qfc1)
    qsfc_land = jnp.where(qfc1 > 0., qk[L - 1] + elflx / jnp.where(qfc1 > 0., qfc1, 1.), qsfc)
    exnsfc = (_C.p_ref / psfc) ** _CAPA
    qsfc_sea = _PQ0SEA / psfc * jnp.exp(_A2 * (thsk - _A3 * exnsfc) / (thsk - _A4 * exnsfc))
    qsfc = jnp.where(seamask < 0.5, qsfc_land, qsfc_sea)
    qz0 = (1. - seamask) * qsfc + seamask * qz0
    zero = jnp.zeros_like(thz0)
    sz0 = jnp.stack([thz0, qz0, zero, zero])
    clow = jnp.stack([jnp.ones_like(thz0), jnp.asarray(chklowq, thz0.dtype), zero, zero])
    cts = jnp.stack([ct_out, zero, zero, zero])
    species = jnp.stack([thek, qk, f(qcw), f(qci)])
    sp = vdifh(dt, lpbl, sz0, akhs_dens, clow, cts, species, akhk, z, rhok)
    thek2, qk2, qcwk2, qcik2 = sp[0], sp[1], sp[2], sp[3]
    cwmk2 = qcwk2 + qcik2
    thnew = f(thek2) + f(cwmk2) * _ELOCP * ape
    rthblten = (thnew - th) * rdt
    qold = qv / (1. + qv)
    qnew = f(qk2)
    dqdt = (qnew - qold) * rdt
    rqvblten = dqdt / (1. - qnew) ** 2
    rqcblten = (f(qcwk2) - qcw) * rdt
    rqiblten = (f(qcik2) - qci) * rdt
    # --- momentum ---
    akmk = akm * (rhok[:-1] + rhok[1:]) * 0.5
    akms_dens = akms * rhok[L - 1]
    uv = vdifv(dt, uz0, vz0, akms_dens, jnp.stack([uk, vk]), akmk, z, rhok)
    rublten = (f(uv[0]) - u) * rdt
    rvblten = (f(uv[1]) - v) * rdt
    return MYJOutput(rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, tke_out,
                     el_myj, exch_h, exch_m, dqdt, pblh, kpbl, mixht, ct_out, thz0, qz0, qsfc)
