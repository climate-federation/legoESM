"""CAM7 CLUBB-MF plume ensemble (opt-in mass flux on prognostic CLUBB), REDUCED setting.

Port of ``integrate_mf`` (ESCOMP/atmospheric_physics atmos_phys0_34_002,
cec8c93ed2, ``schemes/clubb/clubb_mf.F90``; the tag CAM cam_development
417212dff0 pins) for ONE column on CLUBB's ascending grid (index 0 = surface
momentum level ``zm[0]``; thermo level ``zt[k]`` lies between ``zm[k]`` and
``zm[k+1]``), its Poisson entrainment draws (``poisson``/``knuth``/``hormann``
with CESM_share share1.1.22 KISS, seeded from the near-surface wind as CAM
does), and CAM's coupling (``clubb.F90`` ``compute_mf_plumes``): the
mass-flux convergence of thl and qt, CFL-limited, as CLUBB's explicit
``thlm``/``rtm`` forcing.  First scored against our EDMF (user decision
2026-11-19, option 2); a model option since 2026-11-23 (user: CAM's exact
generator, fixed L0 = 200 m, no new carried state unless buoyancy sorting is selected).

Setting (CAM cam7 namelist defaults except where the user's reduced setting
says otherwise): ``do_clubb_mf_aloft = F`` and ``clubb_mf_alphturb = 0`` (user:
no CLUBB inputs), ``do_clubb_mf_precip = F`` (vote V1), ``nup = 25``,
``do_clubb_mf_invswitch = T``; ``Lopt = 0`` (L0 = 200 m: vote V2b, user
2026-11-23) because Lopt 6 needs plume-top memory carried in the model
state; every other flag at its CAM default, which in this setting leaves the
following CAM code unreachable or inert, so it is not ported (exclusions,
each from the source):

* aloft start, ``w``/``wp*_env`` inputs (aloft off); TKE (``eturb == 1``
  exactly with ``alphturb = 0``, so ``tke`` drops out);
* precipitation, downdrafts, cold pools: ``supqt = supthl = 0``, no
  downdraft, ``ddcp`` returned 0, hence ``cpfac = 1`` (vote V24, asserted on
  the fixture); with ``mf_sqt = 0`` CAM's "MF drying exceeds rtm" abort can
  never fire, so it is not ported;
* ``dynamic_L0`` is the fixed L0 (Lopt 0); with optional ``bsort``,
  ``ztopm1`` is carried per column/plume for the stopping distance;
  otherwise the historical call-local 500 m memory remains unused by physics;
* convective momentum transport (``do_clubb_mf_cmt`` cam7 default F);
* ``aspd``/``scalesrf``/``do_implicit``/``fixent`` are
  compile-time ``.false.``; ``rhinv`` is unused (``rhtke`` off, Lopt 0);
* dry/moist aggregates, mcape, ent/det diagnostics (vote V8).

RULE 5: every switch on traced state is smooth, sharpness field in
:class:`MFSmoothing` (``math.inf`` = CAM exactly; defaults voted V15-V22).
The fixed-point exit inside ``condensation_mf`` (|dqc| < 2e-5) is a switch on
state too (the iteration count jumps with it), so it is smoothed as a carried
done-weight; its hard limit is CAM's exit exactly (vote V7').  The integer
entrainment draws are piecewise constant in the state that seeds them, so
they carry no gradient (``stop_gradient``); the fluxes stay differentiable
given the draws.

KNOWN DIFFERENCES at the smooth defaults are stored per fixture column in
``tests/atmosphere/oracle_fixtures/clubb_mf_default_gaps.json``.

Optional buoyancy sorting follows the pinned ``clubb_mf.F90`` at
``oracle_audit/bsort_build`` (cec8c93ed2 with ``poisson`` made public and
``bsort = .true.``): lines 739-742, 1068-1210, 1612-1622, 1792-1810,
1929-2040, 2497-2538. Three successive-substitution passes and the total-area
cap of 0.6 are included. KNOWN DIFFERENCES: smooth case selection, neutral
fraction clamps, discriminant selection/square root, segment equality, and
area cap replace CAM's hard switches. The root sign and its reciprocal
are regularised by ``bsort_root`` (inf recovers CAM's stable quadratic). ``bsort_sat`` extends the two-segment
formula smoothly using regularised absolute water excesses (CAM's ratio
when the excesses have opposite signs). This keeps the otherwise inactive
same-sign and endpoint evaluations finite. Defaults and denominator guards
are TODO-VOTE; ``HARD`` recovers CAM. Per-column/per-call gaps belong in
``tests/atmosphere/oracle_fixtures/clubb_mf_bsort_default_gaps.json``; run the
new test module as a script on a compute node to measure them. No measured
default gap is claimed before that generator has run. The cap rescales the
fluxes before convergence, retaining zero boundary fluxes and column thl/qt
conservation; it does not discard prognostic water or energy.

"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from legoesm.core.smooth import expm1_over_x, sigmoid_switch, smooth_max, smooth_min, smooth_relu
from legoesm.thermo import saturation_specific_humidity_cam_table

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "CAM7 CLUBB-MF stochastic plume ensemble (reduced setting) and its "
        "mass-flux convergence of thl and qt as CLUBB's explicit forcing."
    ),
    "inputs": {
        "thl": "K", "qt": "kg/kg", "thv": "K", "u": "m/s", "v": "m/s",
        "p_zm": "Pa", "zm": "m", "wthl_sfc": "K m/s", "wqt_sfc": "kg/kg m/s",
        "pblh": "m", "lts": "K",
    },
    "outputs": {"thlflx": "K m/s", "qtflx": "kg/kg m/s",
                "thlm_forcing": "K/s", "rtm_forcing": "kg/kg/s"},
    "sign_convention": (
        "Ascending CLUBB grid; fluxes positive upward on momentum levels, zero "
        "at the surface and model top; forcing = -(1/rho_ds_zt) d(rho_ds_zm F)/dz."
    ),
    "conserves": ["moisture", "energy"],  # flux form, zero boundary fluxes: column thl and qt
    "differentiable": True,
    "reference": (
        "Suselj et al. (2019), J. Atmos. Sci. 76, 2373-2395; CAM "
        "atmos_phys0_34_002 schemes/clubb/clubb_mf.F90 and clubb.F90"
    ),
    "idealized_test": "tests/atmosphere/hydrostatic/unit/test_clubb_mf_oracle.py",
}

__param_spec__ = {
    "MFSmoothing": {
        "scheme_key": "atm.turb.MFSmoothing",
        "params": {},
        "excluded": {
            "alive": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "qc": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "trig_wthv": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "trig_wqt": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "trig_lts": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "wstar": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "convh": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "kspan": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "conv": "numerics: RULE 5 sharpness of a CLUBB-MF switch on traced state (inf = CAM), not a tunable",
            "bsort_case": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
            "bsort_sat": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
            "bsort_xc": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
            "bsort_disc": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
            "bsort_root": "numerics: RULE 5 stable-root sign smoothing (inf = CAM), not a tunable",
            "bsort_equal": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
            "bsort_area": "numerics: RULE 5 buoyancy-sorting smoothing (inf = CAM), not a tunable",
        },
    },
}

# --- clubb_mf.F90 integrate_mf PARAMETERs and cam7 namelist defaults ---
NUP = 25  # clubb_mf_nup (namelist_defaults_cam.xml)
ENT0 = 0.2  # clubb_mf_ent0
PWMIN, PWMAX = 1.5, 3.0  # clubb_mf_pwmin / pwmax
ALPHW, ALPHQT, ALPHTHV = 0.572, 2.890, 2.890
CWQT, CWTHV = 0.32, 0.58
WA, WB = 1.0, 1.5
WSTARMIN, PBLHMIN = 1.0e-3, 100.0
ENT_MOM_REDUCTION = 3.0
KSPAN_MIN = 4
LTS_INHIBIT_THRESH = 20.0  # [K]
COND_MAX_ITER, COND_TOL = 50, 2.0e-5  # condensation_mf
L0 = 200.0  # clubb_mf_L0 [m], Lopt 0 (user 2026-11-23)
KSEED = 1  # clubb_mf_kseed: offset of the 2nd..4th seed winds
ZTOPM1_COLD = 500.0  # plume-top memory [m] each call starts from (fixture call 1)
_BSORT_ITER = 3  # CAM integrate_mf:741, static successive-substitution count
_BSORT_RLE, _BSORT_AMAX = 0.1, 0.6  # CAM :471, :467
CFLVAL = 1.0  # clubb.F90 compute_mf_plumes: Courant-number limit
CFL_SHARPNESS = 100.0  # RULE 5 default of the Courant limiter's max(1, .) [1]^-1 (voted)

# --- CESM_share share1.1.22 kissvec.c (Marsaglia KISS) ---
_KISS_LCG_A, _KISS_LCG_C = 69069, 1327217885
_KISS_MWC3, _KISS_MWC4, _KISS_LOW16 = 18000, 30903, 65535
_KISS_SCALE = 2.328306e-10  # int32 -> (0, 1)
# --- clubb_mf.F90 poisson / hybridRNG / hormann (PTRS, Hormann 1993) ---
_SEED_SCALE = 1.0e9
_KNUTH_MAX_LAMBDA = 10.0
_H_B = (0.931, 2.53)
_H_A = (-0.059, 0.02483)
_H_VR = (0.9277, 3.6224)
_H_AI = (1.1239, 1.1328)
_H_SHIFT, _H_US_OK, _H_US_BAD = 0.43, 0.07, 0.013
_H_B_POLE = 3.4  # alphinv = 1.1239 + 1.1328/(b - 3.4)
_STIRLING_K, _STIRLING_C, _STIRLING_A = 10, 360.0, 12.0
_KISS_SHIFTS = (13, 17, 5, 16)  # kissvec.c xorshift and MWC shifts
# --- clubb.F90 host LTS (mf_pinterp targets) ---
P_LTS_TOP, P_LTS_BOT = 70000.0, 100000.0  # [Pa]


class MFSmoothing(NamedTuple):
    """Sharpness of each smooth switch (1/width, argument units); inf = CAM."""

    alive: float = 1.0e4  # wn2 > 0 keeps the plume going [s2/m2]      V15'
    qc: float = 1.0e5  # qc = max(qt - qs, 0); cloudy test [kg/kg]^-1 V16
    trig_wthv: float = 1.0e4  # wthv > 0 [(K m/s)^-1]                       V17'
    trig_wqt: float = 1.0e7  # wqt > 0 [(kg/kg m/s)^-1]                    V18
    trig_lts: float = 100.0  # lts < 20 K [1/K]                            V19
    wstar: float = 1.0e11  # max(wstarmin**3, .) cubed [s3/m3]           V25
    convh: float = 1.0  # max(pblh, pblhmin) [1/m]                    V20
    kspan: float = 100.0  # alive-level count >= kspan_min [1/level]    V22'
    conv: float = 1.0e7  # condensation exit |dqc| < 2e-5 [kg/kg]^-1   V7'
    # TODO-VOTE defaults: case=1e10 [(kg/kg)^-2], sat=1e5 [(kg/kg)^-1],
    # xc=100 [1], disc=100 [s4/m4], root=100 [s2/m2], equal=100 [1], area=100 [1].
    bsort_case: float = 1.0e10
    bsort_sat: float = 1.0e5
    bsort_xc: float = 100.0
    bsort_disc: float = 100.0
    bsort_root: float = 100.0  # stable-root sign choice [s2/m2], TODO-VOTE
    bsort_equal: float = 100.0
    bsort_area: float = 100.0


HARD = MFSmoothing(*(math.inf,) * len(MFSmoothing._fields))


def _strict_pos(x, s):
    """``x > 0`` (CAM's strict test) smoothed; exactly ``x > 0`` at s = inf."""
    return 1.0 - sigmoid_switch(-x, s)


def _qsat(t, p, hard):
    if hard:
        return saturation_specific_humidity_cam_table(
            t, p, clamp_sharpness=math.inf, cap_sharpness=math.inf
        )
    return saturation_specific_humidity_cam_table(t, p)


def condensation_mf(qt, thl, p, iex, sm: MFSmoothing):
    """``condensation_mf`` (noice = .true., so wf = 1 and L = latvap).
    Returns thv, qc, th, qt - qs (the pre-clip excess, for the cloudy test) and the
    iteration count (test placement only)."""
    hard = math.isinf(sm.qc)  # the CAM table's own clamp/cap follow the qc switch
    lcp = constants.L_v / constants.c_pd

    def body(_, c):
        qc, done, n = c  # done = 1 once CAM would have exited (the update that converged is kept)
        t = thl / iex + lcp * qc
        qs = _qsat(t, p, hard)
        qc_upd = smooth_relu(0.5 * qc + 0.5 * (qt - qs), sm.qc)
        qc_new = done * qc + (1.0 - done) * qc_upd
        conv = _strict_pos(COND_TOL - jnp.abs(qc_upd - qc), sm.conv)
        return qc_new, done + (1.0 - done) * conv, n + (1.0 - done)

    z0 = jnp.zeros_like(qt)
    qc, _, n_iter = jax.lax.fori_loop(0, COND_MAX_ITER, body, (z0, z0, z0))
    t = thl / iex + lcp * qc
    excess = qt - _qsat(t, p, hard)
    qc = smooth_relu(excess, sm.qc)
    thv = (thl + lcp * iex * qc) * (1.0 + (constants.R_v / constants.R_d - 1.0) * (qt - qc) - qc)
    return thv, qc, t * iex, excess, n_iter


def bsort_roots(a, b, c, sm):
    """CAM roots :2497-2538, general quadratic on the alive ``a=wn**2>0`` branch.

    b==0 uses the same general formula (Fortran sign(1,0)=1). The sign is
    smooth by default; TODO-VOTE: regularise c/q with q/(q*q + width*width),
    width=1/bsort_root, since smoothing sign makes q cross zero. At inf the
    only q==0 case with a real root is b=c=0; its double root is zero.
    """
    disc = b**2 - 4.0 * a * c
    rd = smooth_relu(disc, sm.bsort_disc)
    rootdisc = jnp.where(rd > 0.0, jnp.sqrt(jnp.where(rd > 0.0, rd, 1.0)), 0.0)
    sign = 2.0 * sigmoid_switch(b, sm.bsort_root) - 1.0
    q = -0.5 * (b + sign * rootdisc)
    r1 = q / a
    if math.isinf(sm.bsort_root):
        r2 = c / jnp.where(q != 0.0, q, 1.0)
    else:
        r2 = c * q / (q**2 + (1.0 / sm.bsort_root)**2)
    return smooth_min(r1, r2, sm.bsort_xc), disc


def bsort_clip(x, sharpness):
    """CAM neutral-fraction min(1,max(0,x)), with a RULE 5 smooth default."""
    return smooth_min(jnp.asarray(1.0), smooth_max(jnp.asarray(0.0), x, sharpness), sharpness)


def bsort_choose(x_cu, x_en, xsat, sm):
    """CAM x_cu==xsat branch, smoothed with an equality kernel by default."""
    eq = ((x_cu == xsat).astype(x_cu.dtype) if math.isinf(sm.bsort_equal)
          else jnp.exp(-(sm.bsort_equal * (x_cu - xsat))**2))
    return eq * smooth_max(x_cu, x_en, sm.bsort_xc) + (1.0 - eq) * x_cu


def buoyancy_sort_mixfrac(qtn, thln, wn, thl_e, qt_e, thv_e, iex, p, ztopm1, sm):
    """CAM :1929-2040, Bretherton et al. (2004), returns ``ee2, ud2``.

    ``a = wn**2`` is positive on the alive CAM branch (including the
    averaged trials of passes 2/3). Zero velocity is evaluated only for
    already dead scan lanes: replace it by 1 there, without a physics floor.
    CAM's stable general quadratic is used (see :func:`bsort_roots`);
    no a==0/b==0 branch is needed on the alive branch.
    Safe exact-zero denominators protect inactive endpoint evaluations only.
    TODO-VOTE: at the degenerate both-zero excess, use xsat=1/2, and at
    endpoint xsat=0/1 use zero segment slope (CAM itself divides by zero).
    """
    hard_sat = math.isinf(sm.bsort_sat)
    hard_qc = math.isinf(sm.qc)
    em = qt_e - _qsat(thl_e / iex, p, hard_qc)
    en = qtn - _qsat(thln / iex, p, hard_qc)
    thvn = condensation_mf(qtn, thln, p, iex, sm)[0]
    w2 = jnp.where(wn != 0.0, wn, 1.0)**2
    cridis = _BSORT_RLE * ztopm1
    coeff = 2.0 * WA * constants.g * cridis

    xc_same = bsort_clip(1.0 - coeff / w2 * (1.0 - thvn / thv_e), sm.bsort_xc)
    # On CAM's mixed-sign branch |en|/(|en|+|em|) = en/(en-em).
    # Positive regularised magnitudes also define a bounded continuation on
    # the same-sign branch, which a smooth case blend must evaluate.
    if hard_sat:
        an, am = jnp.abs(en), jnp.abs(em)
    else:
        width = 1.0 / sm.bsort_sat
        an, am = jnp.sqrt(en**2 + width**2), jnp.sqrt(em**2 + width**2)
    den = an + am
    xsat = jnp.where(den != 0.0, an / jnp.where(den != 0.0, den, 1.0), 0.5)
    thv_sat = condensation_mf(qtn + xsat * (qt_e - qtn),
                              thln + xsat * (thl_e - thln), p, iex, sm)[0]
    inv_x = 1.0 / jnp.where(xsat != 0.0, xsat, 1.0)
    inv_y = 1.0 / jnp.where(xsat != 1.0, 1.0 - xsat, 1.0)

    def segment(t0, t1, cumulus):
        a = w2
        b = coeff * (t1 - t0) / thv_e - 2.0 * w2
        cc = coeff * (t0 - thv_e) / thv_e + w2
        small, disc = bsort_roots(a, b, cc, sm)
        candidate = (smooth_min(xsat, small, sm.bsort_xc) if cumulus
                     else smooth_max(xsat, small, sm.bsort_xc))
        has_roots = sigmoid_switch(disc, sm.bsort_disc)  # CAM >=, not >
        return has_roots * bsort_clip(candidate, sm.bsort_xc) + (1.0 - has_roots) * (xsat if cumulus else 1.0)

    x_cu = segment(thvn, (1.0 - inv_x) * thvn + inv_x * thv_sat, True)
    x_en = segment((1.0 - inv_y) * thv_e + inv_y * thv_sat, thv_e, False)
    xc_mixed = bsort_choose(x_cu, x_en, xsat, sm)
    same = _strict_pos(em * en, sm.bsort_case)
    xc = same * xc_same + (1.0 - same) * xc_mixed
    return xc**2, 1.0 - 2.0 * xc + xc**2


def limit_bsort_area(area, sharpness):
    """CAM :1612-1622, same per-level rescaling for all plumes, except top."""
    ac = jnp.sum(area, axis=1, keepdims=True)
    factor = _BSORT_AMAX / smooth_max(jnp.asarray(_BSORT_AMAX), ac, sharpness)
    return area.at[:-1].multiply(factor[:-1])


def integrate_mf(c, enti, ztopm1_in, sm: MFSmoothing = MFSmoothing(), *, bsort=False):
    """One column.  ``c``: CLUBB-grid inputs (keys as the fixture's ``c_*``
    arrays without the prefix, one column; bsort also needs thl_zm, qt_zm,
    thv_zm and rho_zm); ``enti`` (nzt, NUP) int draws;
    ``ztopm1_in`` (NUP,) plume-top memory.  Returns the compared outputs."""
    g, zvir = constants.g, constants.R_v / constants.R_d - 1.0
    zm, dzt = c["zm"], c["dzt"]
    nzm = zm.shape[0]
    wthv = c["wthl_sfc"] + zvir * c["ths"] * c["wqt_sfc"]
    wqt = c["wqt_sfc"]
    trig = (
        _strict_pos(wthv, sm.trig_wthv)
        * _strict_pos(wqt, sm.trig_wqt)
        * _strict_pos(LTS_INHIBIT_THRESH - c["lts"], sm.trig_lts)
    )

    convh = smooth_max(c["pblh"], jnp.asarray(PBLHMIN), sm.convh)
    # max(wstarmin, x**(1/3)) == max(wstarmin**3, x)**(1/3) (monotone), and the cube-root argument
    # then never reaches 0, where its derivative is infinite (a zero-flux column; GLM review)
    wstar = jnp.cbrt(smooth_max(jnp.asarray(WSTARMIN**3), g / c["thv"][0] * wthv * convh, sm.wstar))
    qstar, thvstar = wqt / wstar, wthv / wstar
    sigmaw, sigmaqt, sigmathv = ALPHW * wstar, ALPHQT * jnp.abs(qstar), ALPHTHV * jnp.abs(thvstar)
    wmin, wmax = sigmaw * PWMIN, sigmaw * PWMAX
    i = jnp.arange(NUP, dtype=zm.dtype)
    wlv = wmin + (wmax - wmin) / NUP * i
    wtv = wmin + (wmax - wmin) / NUP * (i + 1.0)
    w0 = 0.5 * (wlv + wtv)
    a = 0.5 * erf(wtv / (jnp.sqrt(2.0) * sigmaw)) - 0.5 * erf(wlv / (jnp.sqrt(2.0) * sigmaw))
    qt0 = CWQT * w0 * sigmaqt / sigmaw
    thv0 = CWTHV * w0 * sigmathv / sigmaw
    den = 0.5 * (dzt[2] + 2.0 * dzt[1] + dzt[0])
    beta_qt = (c["qt"][2] - c["qt"][0]) / den
    beta_thv = (c["thv"][2] - c["thv"][0]) / den
    qt0 = c["qt"][0] - beta_qt * dzt[0] + qt0
    thv0 = c["thv"][0] - beta_thv * dzt[0] + thv0
    thl0 = thv0 / (1.0 + zvir * qt0)
    thv0, qc0, _, ex0, _ = condensation_mf(qt0, thl0, c["p_zm"][0], c["iex_zm"][0], sm)
    u0 = jnp.full((NUP,), c["u"][0])
    v0 = jnp.full((NUP,), c["v"][0])

    def step(carry, xk):
        w, thl, qt, thv, u, v = carry[:6]
        dz, qt_e, thl_e, thv_e, u_e, v_e, p_n, iex_n, ent_k = xk[:9]
        mix = ent_k * ENT0 / dz
        entn = mix
        qtn, thln, wn = qt, thl, w
        qtn0, thln0, wn0 = qtn, thln, wn
        for iteration in range(_BSORT_ITER if bsort else 1):
            if bsort:
                if iteration:
                    qtn, thln, wn = (0.5 * (qtn + qtn0), 0.5 * (thln + thln0),
                                     0.5 * (wn + wn0))
                qtn0, thln0, wn0 = qtn, thln, wn
                ee2, ud2 = buoyancy_sort_mixfrac(
                    qtn, thln, wn, *xk[9:12], iex_n, p_n, ztopm1_in, sm)
                entn, detn = mix * ee2, mix * ud2
            entexp = jnp.exp(-entn * dz)
            entexpu = jnp.exp(-entn * dz / ENT_MOM_REDUCTION)
            qtn = qt_e * (1.0 - entexp) + qt * entexp
            thln = thl_e * (1.0 - entexp) + thl * entexp
            un = u_e * (1.0 - entexpu) + u * entexpu
            vn = v_e * (1.0 - entexpu) + v * entexpu
            thvn, qcn, _, exn, _ = condensation_mf(qtn, thln, p_n, iex_n, sm)
            b = g * (0.5 * (thvn + thv) / thv_e - 1.0)
            wp = WB * entn
            noent = (entn == 0.0) if bsort else (ent_k == 0)
            wp_safe = jnp.where(noent, 1.0, wp)
            entw = jnp.exp(-2.0 * wp_safe * dz)
            if bsort:
                # Analytic zero-entrainment continuation, including its slope.
                # expm1 avoids losing the acceleration at tiny traced entn.
                x = -2.0 * wp * dz
                wn2 = jnp.exp(x) * w**2 + 2.0 * WA * b * dz * expm1_over_x(x)
            else:
                wn2 = jnp.where(
                    noent, w**2 + 2.0 * WA * b * dz, entw * w**2 + (1.0 - entw) * WA * b / wp_safe
                )
            r = smooth_relu(wn2, sm.alive)
            wn = jnp.where(r > 0.0, jnp.sqrt(jnp.where(r > 0.0, r, 1.0)), 0.0)
        gate = _strict_pos(wn2, sm.alive)
        new = (wn, thln, qtn, thvn, un, vn)
        prof = (gate, wn, thln, qtn, thvn, qcn, un, vn, exn)
        if bsort:
            # CAM has exited exactly-zero-survival lanes. Avoid evaluating
            # their exponential (0*inf in tall fp32 columns); fractional
            # survival is applied only through cumulative alive below.
            live = (gate > 0.0) & (carry[6] != 0.0)
            mfn = carry[6] * jnp.exp(jnp.where(live, dz * (entn - detn), 0.0))
            mfn = jnp.where(live, mfn, 0.0)
            area = mfn / (jnp.where(wn > 0.0, wn, 1.0) * xk[12])
            new += (mfn,)
            prof += (area,)
        return new, prof

    k = slice(0, nzm - 1)
    xs = (dzt, c["qt"], c["thl"], c["thv"], c["u"], c["v"], c["p_zm"][1:], c["iex_zm"][1:])
    xs = tuple(jnp.broadcast_to(x[k][:, None], (nzm - 1, NUP)) for x in xs) + (
        enti[k].astype(zm.dtype),
    )
    initial = (w0, thl0, qt0, thv0, u0, v0)
    if bsort:
        extra = (c["thl_zm"][1:], c["qt_zm"][1:], c["thv_zm"][1:], c["rho_zm"][1:])
        xs += tuple(jnp.broadcast_to(x[:, None], (nzm - 1, NUP)) for x in extra)
        initial += (c["rho_zm"][0] * a * w0,)
    _, (gate, *prof) = jax.lax.scan(step, initial, xs)
    area = None
    if bsort:
        area = jnp.concatenate([a[None], prof.pop()], axis=0)
    cat = lambda x0, x: jnp.concatenate([x0[None], x], axis=0)  # noqa: E731
    w, thl, qt, thv, qc, u, v, ex = (
        cat(x0, x) for x0, x in zip((w0, thl0, qt0, thv0, qc0, u0, v0, ex0), prof)
    )
    alive = jnp.concatenate([jnp.ones((1, NUP), zm.dtype), jnp.cumprod(gate, axis=0)], axis=0)
    # not culled: CAM keeps a plume alive on >= KSPAN_MIN levels; the count is an integer at the
    # hard limit, so the transition is centred half a level below the threshold
    keep = sigmoid_switch(jnp.sum(alive, axis=0) - (KSPAN_MIN - 0.5), sm.kspan)
    on = alive * keep  # (nzm, NUP)
    up = {
        "upa": on * (area if bsort else a),
        "upw": on * w,
        "upthl": on * thl,
        "upqt": on * qt,
        "upthv": on * thv,
        "upqc": on * qc,
    }

    if bsort:
        up["upa"] = limit_bsort_area(up["upa"], sm.bsort_area)

    aw = jnp.sum(up["upa"] * up["upw"], axis=1)
    flux = {}
    for name, x, env in (
        ("thlflx", thl, c["thl"]),
        ("qtflx", qt, c["qt"]),
        ("thvflx", thv, c["thv"]),
        ("uflx", u, c["u"]),
        ("vflx", v, c["v"]),
    ):
        f = jnp.sum(up["upa"] * up["upw"] * on * x, axis=1)[1:-1] - aw[1:-1] * env[1:]
        flux[name] = jnp.concatenate([jnp.zeros(1, zm.dtype), f, jnp.zeros(1, zm.dtype)])

    early = jnp.prod(1.0 - on[1])  # CAM: ac(ksfcm+kdir) == 0 -> early return
    nxt = jnp.concatenate([alive[1:], jnp.zeros((1, NUP), zm.dtype)], axis=0)
    top = jnp.sum(zm[:, None] * (alive - nxt), axis=0)
    ztop = (1.0 - early) * (keep * top + (1.0 - keep) * ztopm1_in) + early * zm[0]
    cl = on * _strict_pos(ex, sm.qc)
    first_cl = cl * jnp.cumprod(
        jnp.concatenate([jnp.ones((1, NUP), zm.dtype), 1.0 - cl[:-1]]), axis=0
    )
    first_dead = (1.0 - on) * jnp.cumprod(
        jnp.concatenate([jnp.ones((1, NUP), zm.dtype), on[:-1]]), axis=0
    )
    cb = jnp.sum(first_cl * zm[:, None], axis=0) + (1.0 - jnp.sum(first_cl, axis=0)) * (
        jnp.sum(first_dead * zm[:, None], axis=0) + jnp.prod(on, axis=0) * zm[-1]
    )
    # moist-plume aggregates (CAM "updraft properties for output"): plumes
    # with upqc > 0 (the qc switch); in-cloud condensate = area-weighted mean
    moist = up["upa"] * _strict_pos(ex, sm.qc)
    moist_a = jnp.sum(moist, axis=1)
    # double where: finite reverse-mode gradient at zero moist area (fp32 too)
    _has = moist_a > 0.0
    moist_qc = jnp.where(
        _has, jnp.sum(moist * up["upqc"], axis=1) / jnp.where(_has, moist_a, 1.0), 0.0)
    out = dict(
        up,
        **flux,
        moist_a=moist_a,
        moist_qc=moist_qc,
        ztopm1=ztop,
        cbm1=(1.0 - early) * jnp.mean(cb) + early * zm[0],
        n_alive=jnp.sum(alive, axis=0),
    )  # pre-cull alive-level count (test placement only)
    off = zm[0]  # CAM's no-plume return: ztopm1 = cbm1 = zm(ksfcm), everything else 0
    return {
        k_: trig * v_ + (1.0 - trig) * (off if k_ in ("ztopm1", "cbm1") else 0.0)
        for k_, v_ in out.items()
    }


# ---------------------------------------------------------------------------
# Poisson entrainment draws: CAM's own generator, bit for bit
# ---------------------------------------------------------------------------
def _u32(x):
    return jnp.uint32(x)


def _kiss_next(seeds):
    """One ``kiss_rng`` step (uint32 wrap-around, as the C source)."""
    s1, s2, s3, s4 = seeds
    s1 = s1 * _u32(_KISS_LCG_A) + _u32(_KISS_LCG_C)
    a, b, c, h = (_u32(x) for x in _KISS_SHIFTS)
    s2 = s2 ^ (s2 << a)
    s2 = s2 ^ (s2 >> b)
    s2 = s2 ^ (s2 << c)
    s3 = _u32(_KISS_MWC3) * (s3 & _u32(_KISS_LOW16)) + (s3 >> h)
    s4 = _u32(_KISS_MWC4) * (s4 & _u32(_KISS_LOW16)) + (s4 >> h)
    r = jax.lax.bitcast_convert_type(s1 + s2 + (s3 << h) + s4, jnp.int32)
    return (s1, s2, s3, s4), r.astype(jnp.float64) * _KISS_SCALE + 0.5


def kiss_seeds(u_seed):
    """``poisson``'s seed: int((x - int(x)) * 1e9) per wind (Fortran int
    truncates toward zero), the int32 bits read as uint32 by ``kiss_rng``."""
    u_seed = jnp.asarray(u_seed, jnp.float64)
    frac = u_seed - jnp.trunc(u_seed)
    i32 = jnp.trunc(frac * _SEED_SCALE).astype(jnp.int32)
    return tuple(jax.lax.bitcast_convert_type(i32[n], jnp.uint32) for n in range(4))


def _knuth(seeds, lam):
    explam = jnp.exp(-1.0 * lam)

    def cond(c):
        return c[2] > explam

    def body(c):
        sd, k, puni = c
        sd, u = _kiss_next(sd)
        return sd, k + 1, puni * u

    sd, k, _ = jax.lax.while_loop(cond, body, (seeds, jnp.int32(0), jnp.float64(1.0)))
    return sd, k - 1


def _loggam(k):
    kf = k.astype(jnp.float64)

    def small(_):
        return jax.lax.fori_loop(
            1, _STIRLING_K, lambda i, a: a + jnp.where(i <= k, jnp.log(1.0 * i), 0.0),
            jnp.float64(0.0))

    def big(_):
        return (jnp.log(jnp.sqrt(2.0 * math.pi)) + (kf + 0.5) * jnp.log(1.0 * kf) - kf
                + (1.0 / _STIRLING_A - 1.0 / (_STIRLING_C * kf * kf)) / kf)

    return jax.lax.cond(k < _STIRLING_K, small, big, None)


def _hormann(seeds, lam):
    b = _H_B[0] + _H_B[1] * jnp.sqrt(lam)
    a = _H_A[0] + _H_A[1] * b
    vr = _H_VR[0] - _H_VR[1] / (b - 2.0)
    alphinv = _H_AI[0] + _H_AI[1] / (b - _H_B_POLE)

    def body(c):
        sd, _, _ = c
        sd, U = _kiss_next(sd)
        sd, V = _kiss_next(sd)
        U = U - 0.5
        us = 0.5 - jnp.abs(U)
        k = jnp.floor((2.0 * a / us + b) * U + lam + _H_SHIFT).astype(jnp.int32)
        quick = (us >= _H_US_OK) & (V <= vr)
        skip = (k <= 0) | ((us < _H_US_BAD) & (V > us))
        kp = jnp.maximum(k, 1)  # log(k) is only read when k > 0
        lhs = jnp.log(V * alphinv / (a / (us * us) + b))
        rhs = -1.0 * lam + kp.astype(jnp.float64) * jnp.log(lam) - _loggam(kp)
        done = quick | (~skip & (lhs <= rhs))
        return sd, k, done

    sd, k, _ = jax.lax.while_loop(lambda c: ~c[2], body,
                                  (seeds, jnp.int32(0), jnp.bool_(False)))
    return sd, k


def entrainment_draws(lam, u_seed):
    """``poisson`` (ksfc = 1 to ktop = nzt, ascending): integer draws
    ``(nzt, NUP)`` at rates ``lam``, levels outer and plumes inner, one KISS
    stream seeded from ``u_seed`` (4,).  CAM's ``lam = 0`` level gives -1."""
    lam = jax.lax.stop_gradient(jnp.asarray(lam, jnp.float64))
    seeds = kiss_seeds(jax.lax.stop_gradient(u_seed))

    def one(sd, l_):
        # Under vmap the cond runs BOTH branches; Hormann's rejection loop does
        # not terminate for small rates (seen at lam = 0.38), so each branch is
        # fed a rate inside its own range.  The selected branch sees l_ itself.
        # A NaN rate (already-NaN state) would hang either loop: draw at 0.
        l_ = jnp.where(jnp.isnan(l_), 0.0, l_)
        return jax.lax.cond(
            l_ < _KNUTH_MAX_LAMBDA,
            lambda s, x: _knuth(s, jnp.minimum(x, _KNUTH_MAX_LAMBDA)),
            lambda s, x: _hormann(s, jnp.maximum(x, _KNUTH_MAX_LAMBDA)), sd, l_)

    _, draws = jax.lax.scan(one, seeds, lam.reshape(-1))
    return draws.reshape(lam.shape)


# ---------------------------------------------------------------------------
# Host coupling (clubb.F90 compute_mf_plumes, reduced setting)
# ---------------------------------------------------------------------------
def mf_lts(T, q_liq, p_full, p_sfc):
    """CAM's host LTS: thl(700 hPa) - thl(1000 hPa), thl = (T - L/cp q_l)
    (p_s/p)^kappa, ``mf_pinterp`` = linear in p, end values held (one
    column, top-down)."""
    thl = (T - constants.L_v / constants.c_pd * q_liq) * (p_sfc / p_full) ** constants.kappa
    return jnp.interp(P_LTS_TOP, p_full, thl) - jnp.interp(P_LTS_BOT, p_full, thl)


def clubb_mf_forcing(c, dt, rho_ds_zm, invrs_rho_ds_zt, invrs_dzt,
                     sm: MFSmoothing = MFSmoothing(), cfl_sharpness: float = CFL_SHARPNESS,
                     *, bsort=False, ztopm1=None):
    """One column: the plumes, then CAM's explicit forcing on CLUBB's thermo
    levels, ``-(1/rho_ds_zt) d(rho_ds_zm F)/dz * cflfac`` for thl and qt.
    ``c`` as :func:`integrate_mf` (plus ``u`` for the seed); returns
    ``(thlm_forcing, rtm_forcing, out)``.  CAM's limiter ``cflfac =
    cflval/max_cfl if max_cfl > cflval`` is ``cflval/max(cflval, max_cfl)``,
    smoothed with ``cfl_sharpness`` (inf = CAM)."""
    nzt = c["dzt"].shape[0]
    lam = (c["dzt"] / L0).at[nzt - 1].set(0.0)[:, None] * jnp.ones((1, NUP))
    u = c["u"]
    enti = entrainment_draws(lam, jnp.stack([u[0], u[KSEED], u[1 + KSEED], u[2 + KSEED]]))
    memory = (jnp.zeros((NUP,), c["zm"].dtype) if ztopm1 is None else ztopm1) if bsort else jnp.full((NUP,), ZTOPM1_COLD)
    out = integrate_mf(c, enti, memory, sm, bsort=bsort)
    aw = jnp.sum(out["upa"] * out["upw"], axis=1).at[0].set(0.0)
    max_cfl = jnp.max(dt * invrs_dzt * jnp.maximum(jnp.abs(aw[:-1]), jnp.abs(aw[1:])))
    cflfac = CFLVAL / smooth_max(jnp.asarray(CFLVAL), max_cfl, cfl_sharpness)

    def conv(f):
        rf = rho_ds_zm * f
        return -invrs_rho_ds_zt * invrs_dzt * cflfac * (rf[1:] - rf[:-1])

    return conv(out["thlflx"]), conv(out["qtflx"]), dict(out, cflfac=cflfac)


# ---------------------------------------------------------------------------
# Deep-cloud hookup (atmos_phys clubb.F90 clubb3_run, do_clubb_mf_rad)
# ---------------------------------------------------------------------------
MF_CONCLD_MAX = 0.8  # clubb3_run: concld = min(deepcu, 0.80)


def _select(a, b, sharpness):
    """Weight of ``a`` in the larger-of-two choice: exactly ``a > b`` at inf;
    a convex blend otherwise, so two zero clouds stay exactly zero."""
    return _strict_pos(a - b, sharpness)


def mf_deep_cloud(deepcu_zm, icwmr_zm, mf_cloudfrac, mf_qc, cfg):
    """ZM's deep cloud combined with the plume cloud, top-down (ncol, nlev).

    CAM7 sets ``deepcu = clubb_mf_cldfrac_fac * mf_cloudfrac``, ``dp_icwmr =
    mf_qc``, zeroed where ``deepcu <= frac_limit`` or ``dp_icwmr < ic_limit``
    (layers 1..pver-1); here that cloud and ZM's are combined as the larger
    fraction per level, with that cloud's in-cloud water (user 2026-11-23).
    Returns ``(deepcu, icwmr)``; the 0.8 cap is :func:`mf_concld`."""
    dc = cfg.mf_cldfrac_fac * mf_cloudfrac
    keep = (_strict_pos(dc - cfg.cam6_deepcu_frac_limit, cfg.mf_deepcu_frac_gate_sharpness)
            * sigmoid_switch(mf_qc - cfg.cam6_deepcu_ic_limit,
                             cfg.mf_deepcu_ic_gate_sharpness))  # kept iff qc >= ic_limit
    dc = (dc * keep).at[:, -1].set(0.0)
    w = _select(dc, deepcu_zm, cfg.mf_deepcu_select_sharpness)
    return w * dc + (1.0 - w) * deepcu_zm, w * mf_qc + (1.0 - w) * icwmr_zm


def mf_concld(deepcu, cfg):
    """``min(deepcu, 0.8)`` (CAM concld, the cloud-fraction share), smooth."""
    w = _select(deepcu, jnp.asarray(MF_CONCLD_MAX), cfg.mf_deepcu_cap_sharpness)
    return w * MF_CONCLD_MAX + (1.0 - w) * deepcu
