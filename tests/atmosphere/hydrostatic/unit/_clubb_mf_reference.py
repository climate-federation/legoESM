"""Test-only JAX reference of CAM's CLUBB-MF plume ensemble, REDUCED setting.

Port of ``integrate_mf`` (ESCOMP/atmospheric_physics atmos_phys0_34_002,
cec8c93ed2, ``schemes/clubb/clubb_mf.F90``; the tag CAM cam_development
417212dff0 pins) for ONE column on CLUBB's ascending grid (index 0 = surface
momentum level ``zm[0]``; thermo level ``zt[k]`` lies between ``zm[k]`` and
``zm[k+1]``).  It is the oracle our EDMF (``turbulence/edmf.py``) is scored
against, not a model scheme (user decision 2026-11-19, option 2).

Setting (CAM cam7 namelist defaults except where the user's reduced setting
says otherwise): ``do_clubb_mf_aloft = F`` and ``clubb_mf_alphturb = 0`` (user:
no CLUBB inputs), ``do_clubb_mf_precip = F`` (vote V1), ``nup = 25``,
``do_clubb_mf_invswitch = T``; the fixture runs ``Lopt = 0`` (L0 = 200 m, vote V2b)
because Lopt 6 from CAM's cold start culls every plume; every other flag at its CAM
default, which in this setting leaves the following CAM code unreachable or
inert, so it is not ported (exclusions, each from the source):

* aloft start, ``w``/``wp*_env`` inputs (aloft off); TKE (``eturb == 1``
  exactly with ``alphturb = 0``, so ``tke`` drops out);
* precipitation, downdrafts, cold pools: ``supqt = supthl = 0``, no
  downdraft, ``ddcp`` returned 0, hence ``cpfac = 1`` (vote V24, asserted on
  the fixture);
* the entrainment length ``dynamic_L0`` enters only the Poisson rate, and the
  integer draws ``enti`` are an INPUT here (vote V5: replayed from CAM's own
  KISS generator by the oracle driver), so L0 and the RNG are not ported;
* ``bsort``/``aspd``/``limarea``/``scalesrf``/``do_implicit``/``fixent`` are
  compile-time ``.false.``; ``rhinv`` is unused (``rhtke`` off, Lopt 0);
* dry/moist aggregates, mcape, ent/det diagnostics (vote V8).

RULE 5: every switch on traced state is smooth, sharpness field in
:class:`MFSmoothing` (``math.inf`` = CAM exactly; defaults voted V15-V22).
The fixed-point exit inside ``condensation_mf`` (|dqc| < 2e-5) is a switch on
state too (the iteration count jumps with it), so it is smoothed as a carried
done-weight; its hard limit is CAM's exit exactly (vote V7').
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
    out = dict(
        up,
        **flux,
        ztopm1=ztop,
        cbm1=(1.0 - early) * jnp.mean(cb) + early * zm[0],
        n_alive=jnp.sum(alive, axis=0),
    )  # pre-cull alive-level count (test placement only)
    off = zm[0]  # CAM's no-plume return: ztopm1 = cbm1 = zm(ksfcm), everything else 0
    return {
        k_: trig * v_ + (1.0 - trig) * (off if k_ in ("ztopm1", "cbm1") else 0.0)
        for k_, v_ in out.items()
    }
