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
generator, fixed L0 = 200 m, no new carried state).

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
* ``dynamic_L0`` is the fixed L0 (Lopt 0); the plume-top memory ``ztopm1``
  feeds only ``dynamic_L0`` under other Lopt values and the culled-plume
  output, so it is not carried (the column always starts from the 500 m
  memory the fixture's first call uses);
* convective momentum transport (``do_clubb_mf_cmt`` cam7 default F);
* ``bsort``/``aspd``/``limarea``/``scalesrf``/``do_implicit``/``fixent`` are
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
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from legoesm.core.smooth import sigmoid_switch, smooth_max, smooth_relu
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


def integrate_mf(c, enti, ztopm1_in, sm: MFSmoothing = MFSmoothing()):
    """One column.  ``c``: CLUBB-grid inputs (keys as the fixture's ``c_*``
    arrays without the prefix, one column); ``enti`` (nzt, NUP) int draws;
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
        w, thl, qt, thv, u, v = carry
        dz, qt_e, thl_e, thv_e, u_e, v_e, p_n, iex_n, ent_k = xk
        entn = ent_k * ENT0 / dz
        entexp = jnp.exp(-entn * dz)
        entexpu = jnp.exp(-entn * dz / ENT_MOM_REDUCTION)
        qtn = qt_e * (1.0 - entexp) + qt * entexp
        thln = thl_e * (1.0 - entexp) + thl * entexp
        un = u_e * (1.0 - entexpu) + u * entexpu
        vn = v_e * (1.0 - entexpu) + v * entexpu
        thvn, qcn, _, exn, _ = condensation_mf(qtn, thln, p_n, iex_n, sm)
        b = g * (0.5 * (thvn + thv) / thv_e - 1.0)
        wp = WB * entn
        noent = ent_k == 0
        wp_safe = jnp.where(noent, 1.0, wp)
        entw = jnp.exp(-2.0 * wp_safe * dz)
        wn2 = jnp.where(
            noent, w**2 + 2.0 * WA * b * dz, entw * w**2 + (1.0 - entw) * WA * b / wp_safe
        )
        r = smooth_relu(wn2, sm.alive)
        wn = jnp.where(r > 0.0, jnp.sqrt(jnp.where(r > 0.0, r, 1.0)), 0.0)
        gate = _strict_pos(wn2, sm.alive)
        new = (wn, thln, qtn, thvn, un, vn)
        return new, (gate, wn, thln, qtn, thvn, qcn, un, vn, exn)

    k = slice(0, nzm - 1)
    xs = (dzt, c["qt"], c["thl"], c["thv"], c["u"], c["v"], c["p_zm"][1:], c["iex_zm"][1:])
    xs = tuple(jnp.broadcast_to(x[k][:, None], (nzm - 1, NUP)) for x in xs) + (
        enti[k].astype(zm.dtype),
    )
    _, (gate, *prof) = jax.lax.scan(step, (w0, thl0, qt0, thv0, u0, v0), xs)
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
        "upa": on * a,
        "upw": on * w,
        "upthl": on * thl,
        "upqt": on * qt,
        "upthv": on * thv,
        "upqc": on * qc,
    }

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
                     sm: MFSmoothing = MFSmoothing(), cfl_sharpness: float = CFL_SHARPNESS):
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
    out = integrate_mf(c, enti, jnp.full((NUP,), ZTOPM1_COLD), sm)
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
