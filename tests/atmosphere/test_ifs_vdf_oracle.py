"""Oracle match: OpenIFS 49r1 VDF column kernels (``turbulence.ifs_vdf``), stage 1.

Fixture ``oracle_fixtures/ifs_vdf_kernels.npz``: the tier-2 audit's 24
atmosphere columns (16 L32, 5 L137, 3 stress) plus 9 analytic SYN columns
(dry mixed layers and cloud-topped layers on an L137 grid), each with seven
surface sets, and the outputs of RUNNING OpenIFS 8f6f722f73 arpifs/phys_ec/{vdfdpbl,
vdfhghtn,vdfexcu,vdftofdc}.F90 (gfortran 8.5, fp64) in two arms: ``N`` 49r1
native constants, ``L`` legoESM g/R_d/c_pd/R_v/L_v
(scripts/validate/oracle/ifs_vdf/{build.sh,driver.F90,make_fixture.py}).
VDFEXCU is fed VDFHGHTN's outputs as the oracle produced them.

Error metric (as the IFS convection and SSO oracles): per field, max|ours -
oracle| / max|oracle| over all columns and levels.  Smooth-default gaps are
stored per column in ``oracle_fixtures/ifs_vdf_kernels_gaps.json``
(RULE 5; regenerate with ``IFS_VDF_REGEN_GAPS=1``).
"""

from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics._ifs_thermo import ifs_thermo
from legoesm.atmosphere.physics.turbulence.ifs_vdf import (
    IFSTOFDConfig, IFSVDFConfig, hard_config, hard_tofd_config, ifs_hghtn_parcel, ifs_slgs,
    ifs_tofd_coefficient, ifs_vdfdifc, ifs_vdfdifh_flux, ifs_vdfdifm, ifs_vdfdpbl, ifs_vdfexcu,
    IFSVDFLaneConfig, IFSVDFMainConfig, ifs_vdf_turbulence, ifs_vdfhghtn, ifs_vdfincr,
    ifs_vdfmain,
)
from legoesm.atmosphere.physics.convection._ifs_cumastrn import njkt_levels

_DIR = Path(__file__).parent / "oracle_fixtures"
_FIX = _DIR / "ifs_vdf_kernels.npz"
_GAPS = _DIR / "ifs_vdf_kernels_gaps.json"
_TOL = 1.0e-12                     # oracle bar (vote 3 T2)
_GAP_RTOL, _GAP_ATOL = 1.0e-6, 1.0e-12   # RULE 5 stored-gap tolerance
_GROUPS = ("L32", "L137", "STRESS", "SYN")
_ARMS = ("N", "L")
_EXCU = ("cfm", "cfh", "km", "kh")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _consts(fix, arm, g):
    m = re.search(r"CONST MODE RG RD RCPD RETV RLVTT RTICECU\s+(\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
                  str(fix[f"{arm}_{g}_log"]))
    mode = int(m.group(1))
    rg, rd, cpd, retv, lv = (float(x.replace("D", "E")) for x in m.groups()[1:6])
    if arm == "L":
        assert mode == 1 and (rg, rd, cpd, lv) == (constants.g, constants.R_d, constants.c_pd, constants.L_v)
        assert retv == pytest.approx(constants.R_v / constants.R_d - 1.0, rel=1e-15)
    return dict(g=rg, rd=rd, cpd=cpd, retv=retv, lv=lv)


def _a(fix, g, k):
    return jnp.asarray(fix[f"{g}_{k}"], jnp.float64)


def _dpbl(fix, arm, g, cfg):
    c = _consts(fix, arm, g)
    f = jax.vmap(lambda u, v, T, q, geo: ifs_vdfdpbl(u, v, T, q, geo, cfg, c["g"], c["cpd"], c["retv"]))
    return f(*(_a(fix, g, k) for k in ("u", "v", "T", "q", "geo_full")))


def _excu_args(fix, arm, g):
    c = _consts(fix, arm, g)
    cptgz = _a(fix, g, "geo_full") + _a(fix, g, "T") * c["cpd"]        # RVTMP2 = 0
    o = lambda k: jnp.asarray(fix[f"{arm}_{g}_{k}"])
    cols = (_a(fix, g, "u"), _a(fix, g, "v"), _a(fix, g, "T"), _a(fix, g, "q"),
            _a(fix, g, "p_half"), _a(fix, g, "geo_full"), _a(fix, g, "geo_half"), cptgz,
            _a(fix, g, "hrlw"), _a(fix, g, "hrsw"), _a(fix, g, "z0m"),
            o("kmfl"), o("khfl"), o("kqfl"), o("khvfl"), o("ri"), o("zinv"),
            o("kinv"), o("kcbase"), o("kctop"), o("kpbltype"))
    return cols, c


def _excu(fix, arm, g, cfg, cols=None):
    base, c = _excu_args(fix, arm, g)
    cols = base if cols is None else cols
    f = lambda *a: ifs_vdfexcu(*a, cfg=cfg, g=c["g"], rd=c["rd"], cpd=c["cpd"], retv=c["retv"], lv=c["lv"])
    return jax.vmap(f)(*cols)


def _err(v, ref):
    v = np.asarray(v, np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ref))):
        return np.inf
    return float(np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300))


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS)
def test_dpbl_matches_openifs(fix, arm, g):
    assert _err(_dpbl(fix, arm, g, hard_config()), fix[f"{arm}_{g}_pblh"]) <= _TOL


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS)
def test_excu_matches_openifs(fix, arm, g):
    out = _excu(fix, arm, g, hard_config())
    for k in _EXCU:
        e = _err(getattr(out, k)[:, :-1], fix[f"{arm}_{g}_{k}"][:, :-1])
        assert e <= _TOL, (k, e)


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS)
def test_tofd_matches_openifs(fix, arm, g):
    c = _consts(fix, arm, g)
    tofdc = ifs_tofd_coefficient(_a(fix, g, "geo_full"), _a(fix, g, "u"), _a(fix, g, "v"),
                                 _a(fix, g, "sigflt"), hard_tofd_config(), c["g"])
    assert _err(tofdc, fix[f"{arm}_{g}_tofdc"]) <= _TOL


def test_fixture_exercises_every_pbl_type(fix):
    seen = set()
    for g in _GROUPS:
        seen |= set(np.unique(fix[f"N_{g}_kpbltype"]).tolist())
    assert seen == {0, 1, 2, 3}


def _col_gaps(soft, hard):
    soft, hard = np.asarray(soft), np.asarray(hard)
    den = np.maximum(np.max(np.abs(hard), axis=1), 1e-300)
    return (np.max(np.abs(soft - hard), axis=1) / den).tolist()


def _measured_gaps(fix):
    gaps = {}
    for g in _GROUPS:
        c = _consts(fix, "N", g)
        gaps[f"{g}_pblh"] = (np.abs(np.asarray(_dpbl(fix, "N", g, IFSVDFConfig()))
                                    - np.asarray(_dpbl(fix, "N", g, hard_config())))
                             / np.maximum(np.abs(np.asarray(_dpbl(fix, "N", g, hard_config()))), 1e-300)).tolist()
        soft, hard = _excu(fix, "N", g, IFSVDFConfig()), _excu(fix, "N", g, hard_config())
        for k in _EXCU:
            gaps[f"{g}_{k}"] = _col_gaps(getattr(soft, k)[:, :-1], getattr(hard, k)[:, :-1])
        args = (_a(fix, g, "geo_full"), _a(fix, g, "u"), _a(fix, g, "v"), _a(fix, g, "sigflt"))
        gaps[f"{g}_tofdc"] = _col_gaps(ifs_tofd_coefficient(*args, IFSTOFDConfig(), c["g"]),
                                       ifs_tofd_coefficient(*args, hard_tofd_config(), c["g"]))
        soft, hard = _hghtn(fix, "N", g, IFSVDFConfig()), _hghtn(fix, "N", g, hard_config())
        for k in _HG_SCAL + _HG_ROWS:
            s_, h_ = (np.asarray(getattr(o, k)).reshape(len(fix[f"{g}_names"]), -1) for o in (soft, hard))
            gaps[f"{g}_hg_{k}"] = _col_gaps(s_, h_)
    return gaps


def test_smooth_default_gap_per_column(fix):
    """RULE 5: the smooth default's mismatch against the hard (oracle-exact) form,
    per fixture column and field, equals the stored measurement."""
    gaps = _measured_gaps(fix)
    if os.environ.get("IFS_VDF_REGEN_GAPS") == "1":
        _GAPS.write_text(json.dumps(gaps, indent=1, sort_keys=True))
    stored = json.loads(_GAPS.read_text())
    assert sorted(stored) == sorted(gaps)
    for k, v in gaps.items():
        np.testing.assert_allclose(v, stored[k], rtol=_GAP_RTOL, atol=_GAP_ATOL, err_msg=k)


def test_every_width_is_live():
    """Non-vacuity of every smoothing width: on a column placed inside that
    switch's band, turning ONLY that width on (all others hard) changes the
    VDF output, and the smooth result stays finite."""
    for f in IFSVDFConfig._fields:
        assert f in _SWITCH_CASES, f"no targeted case for {f}"
        hard = np.asarray(_SWITCH_CASES[f](hard_config()))
        only = np.asarray(_SWITCH_CASES[f](hard_config()._replace(**{f: getattr(IFSVDFConfig(), f)})))
        assert np.all(np.isfinite(only)), f
        assert np.any(only != hard), f


_K = 12


def _col():
    """A 12-level idealised column (top -> surface) for the targeted cases."""
    ph = 1.0e5 - 9.0e4 * (1.0 - jnp.linspace(0.0, 1.0, _K + 1)) ** 3   # thin near the surface
    pf = 0.5 * (ph[:-1] + ph[1:])
    T = 300.0 - 50.0 * (1.0 - pf / 1.0e5)
    q = 0.012 * (pf / 1.0e5) ** 3
    zh = 8000.0 * jnp.log(1.0e5 / ph)
    zf = 8000.0 * jnp.log(1.0e5 / pf)
    u = 2.0 + 10.0 * (1.0 - pf / 1.0e5)
    return dict(T=T, q=q, ph=ph, gh=constants.g * zh, gf=constants.g * zf, u=u, v=jnp.zeros(_K))


def _excu_case(cfg, *, khvfl=-0.05, khfl=-0.05, kqfl=-5.0e-5, kmfl=0.09, ri=None, kinv=9,
               kcbase=11, kctop=9, kpbltype=1, hr=None, u=None, T=None, gh=None, out="cf"):
    c = _col()
    T = c["T"] if T is None else T
    u = c["u"] if u is None else u
    gh = c["gh"] if gh is None else gh
    cptgz = c["gf"] + constants.c_pd * T
    ri = jnp.full((_K,), -0.1) if ri is None else ri
    hr = jnp.full((_K,), -1.0e-5) if hr is None else hr
    o = ifs_vdfexcu(u, c["v"], T, c["q"], c["ph"], c["gf"], gh, cptgz, hr, jnp.zeros(_K), 0.01,
                    kmfl, khfl, kqfl, khvfl, ri, 0.0, kinv, kcbase, kctop, kpbltype, cfg)
    return o.kh if out == "kh" else jnp.concatenate([o.cfm, o.cfh])


def _bisect(fn, target, lo, hi, n=80):
    """Scalar x in [lo, hi] with fn(x) = target (fn monotone), hard config."""
    flo = fn(lo) - target
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = fn(mid) - target
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stable_pblh(u_scale):
    from legoesm.atmosphere.physics.turbulence import ifs_vdf as m
    c = _col()
    sv = constants.c_pd * c["T"] * (1.0 + (constants.R_v / constants.R_d - 1.0) * c["q"]) + c["gf"]
    h, found = m._pbl_height_crossing(sv, c["gf"], c["u"] * u_scale, c["v"], sv[-1] + m._EPS_SV,
                                      hard_config(), constants.g)
    return float(h) if bool(found) else 0.0


def _stable_case(cfg, target_h):
    s = _bisect(_stable_pblh, target_h, 30.0, 0.3)
    return _excu_case(cfg, khvfl=1.0e-2, ri=jnp.full((_K,), 0.5), kinv=_K, kpbltype=0,
                      u=_col()["u"] * s)


def _level_edge_case(cfg):
    # PBL height 2 m below the third-lowest full level
    z = float(_col()["gf"][-3] / constants.g)
    return _stable_case(cfg, z - 2.0)


def _dtv_case(cfg):
    c = _col()
    i = 8 - 2                                       # python index of level IINV-1 (KINV 9)
    def dtv(dT):
        T = c["T"].at[i].add(dT)
        cp = c["gf"] + constants.c_pd * T
        retv = constants.R_v / constants.R_d - 1.0
        return float((cp[i] - cp[i + 1]) / constants.c_pd
                     + retv * 0.5 * (c["q"][i] - c["q"][i + 1]) * (T[i + 1] + T[i]))
    dT = _bisect(dtv, 1.0e-3 + 2.0e-6, -20.0, 0.0)
    return _excu_case(cfg, T=c["T"].at[i].add(dT))


def _kcap_case(cfg):
    # MIN(K, 1e6) is smooth on the clipped side: unclipped K = 1e6 + 5e3
    from legoesm.atmosphere.physics.turbulence import ifs_vdf as m
    cap = m._KCAP_M2_S
    try:
        m._KCAP_M2_S = 1.0e30
        kmax = lambda lk: float(jnp.max(_excu_case(hard_config(), kmfl=10.0 ** lk, out="kh")))
        lk = _bisect(kmax, cap + 5.0e3, 0.0, 14.0)
    finally:
        m._KCAP_M2_S = cap
    return _excu_case(cfg, kmfl=10.0 ** lk)


def _sqrt_case(cfg):
    gh = _col()["gh"]
    gh = gh.at[9].set(0.995 * gh[8])                # PGEOH(IINV+1) just below PGEOH(IINV)
    return _excu_case(cfg, kpbltype=2, hr=jnp.full((_K,), -1.0e-4), gh=gh)


def _cbase_case(cfg):
    # MIN(1, z/z_cloudbase) is smooth on the clipped side: z/z_cb = 0.995 at JK = 11
    # (z_cb = PGEOH(KCBASE-1) = PGEOH(10))
    gh = _col()["gh"]
    gh = gh.at[11].set(0.995 * gh[10])
    return _excu_case(cfg, kpbltype=3, khfl=-0.01, kqfl=-1.0e-4, kcbase=11, kctop=8, gh=gh)


def _shear_floor_case(cfg):
    c = _col()
    du = (0.01 + 5.0e-5) ** 0.5                    # |dV|^2 = REPDU2 + width/2
    u = 2.0 + du * jnp.arange(_K)[::-1]
    return _excu_case(cfg, u=u)


_SWITCH_CASES = {
    "repdu2_width": _shear_floor_case,
    "repust_width_m_s": lambda cfg: _excu_case(cfg, kmfl=(1.0e-4 + 5.0e-7) ** 2),
    "khvfl_gate_width": lambda cfg: _excu_case(
        cfg, khvfl=1.0e-4, ri=jnp.full((_K,), 0.5), kinv=_K, kpbltype=0,
        u=_col()["u"] * _bisect(_stable_pblh, 1500.0, 30.0, 0.3)),
    "ri_width": lambda cfg: _excu_case(cfg, ri=jnp.full((_K,), 5.0e-4), kinv=_K, kpbltype=0),
    "pblh_switch_width_m": _level_edge_case,
    "lmin_width_m": lambda cfg: _stable_case(cfg, 298.5),   # 0.1 h just below 30 m
    "lmax_width_m": lambda cfg: _stable_case(cfg, 3015.0),   # 0.1 h just above 300 m
    # MIN(hr(IINV), hr(IINV-1)) is smooth on the clipped side: hr(IINV) 5e-7 K/s above
    "hr_min_width": lambda cfg: _excu_case(
        cfg, kpbltype=2, hr=jnp.full((_K,), -1.0e-4).at[7].add(5.0e-7)),
    "dradflx_width": lambda cfg: _excu_case(cfg, kpbltype=2, hr=jnp.full((_K,), -1.0e-7)),
    "flux_test_width_w_m2": lambda cfg: _excu_case(cfg, kpbltype=3, khfl=-0.1 / constants.c_pd,
                                                   kqfl=-1.0e-4, kcbase=11, kctop=8),
    "cbase_ratio_width": _cbase_case,
    "wtventr_width": lambda cfg: _excu_case(cfg, khvfl=-1.0e-9),
    "dtv_width_k": _dtv_case,
    "kcap_width": _kcap_case,
    "cbrt_width": lambda cfg: _excu_case(cfg, kpbltype=2, hr=jnp.full((_K,), -3.0e-8)),
    "sqrt_width": _sqrt_case,
    "guard_width_frac": lambda cfg: _excu_case(
        cfg, kmfl=0.0, khvfl=-1.005 * _guard_khvfl()),
}


def _guard_khvfl():
    """KHVFL that puts |1/L| at its 1/ZEPSILON cap (u* at REPUST)."""
    from legoesm.atmosphere.physics.turbulence import ifs_vdf as m
    T = float(_col()["T"][-1])
    return (1.0 / m._EPS_MACH) * m._REPUST ** 3 * T / (m._RKAP * constants.g)


@pytest.mark.parametrize("g", ("L32",))
def test_excu_gradients_at_default(fix, g):
    """check_grads through VDFEXCU at the smooth default on one column per PBL type."""
    base, c = _excu_args(fix, "N", g)
    types = np.asarray(fix[f"N_{g}_kpbltype"])
    picks = [int(np.argmax(types == t)) for t in (0, 1, 2, 3) if np.any(types == t)]
    for i in picks:
        col = [a[i] for a in base]

        def f(u, T, khvfl, kmfl):
            a = list(col)
            a[0], a[2], a[14], a[11] = u, T, khvfl, kmfl
            o = ifs_vdfexcu(*a, cfg=IFSVDFConfig(), g=c["g"], rd=c["rd"], cpd=c["cpd"],
                            retv=c["retv"], lv=c["lv"])
            return jnp.sum(o.cfm) + jnp.sum(o.cfh)

        args = (col[0], col[2], col[14], col[11])
        check_grads(f, args, order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-5, eps=1e-6)
        gr = jax.grad(f, argnums=(0, 1))(*args)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in gr)
        assert any(np.any(np.asarray(x) != 0.0) for x in gr)


def test_dpbl_and_tofd_gradients_at_default(fix):
    g = "L32"
    u, v, T, q, geo = (_a(fix, g, k)[0] for k in ("u", "v", "T", "q", "geo_full"))
    check_grads(lambda u, T: ifs_vdfdpbl(u, v, T, q, geo), (u, T), order=1, modes=("fwd", "rev"),
                atol=1e-6, rtol=1e-5, eps=1e-6)
    sf = _a(fix, g, "sigflt")[:4]
    f = lambda uu, s: jnp.sum(ifs_tofd_coefficient(_a(fix, g, "geo_full")[:4], uu, _a(fix, g, "v")[:4], s))
    check_grads(f, (_a(fix, g, "u")[:4], sf), order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-5, eps=1e-6)


def test_gradient_finite_at_zero_stress():
    """u* floor: reverse-mode gradient at KMFL = 0 is finite (review 1, P1)."""
    g = jax.grad(lambda x: jnp.sum(_excu_case(IFSVDFConfig(), kmfl=x)))(0.0)
    assert np.isfinite(float(g))


def test_dpbl_gradient_finite_without_crossing():
    """No Ri crossing (1000 m default): the discarded interpolation must not
    poison the gradient (review 1, P1)."""
    geo = jnp.array([2048.0, 1024.0])
    target = constants.c_pd * 300.0 + geo[-1] + 1.0e-10
    T = jnp.array([(target - geo[0]) / constants.c_pd, 300.0])
    f = lambda t: ifs_vdfdpbl(jnp.ones(2), jnp.zeros(2), t, jnp.zeros(2), geo)
    assert float(f(T)) == 1000.0
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(T))))


def test_ri_blend_is_c1_at_zero():
    """The stable/unstable blend has no derivative jump at Ri = 0 (review 1, P2)."""
    f = lambda r: jnp.sum(_excu_case(IFSVDFConfig(), ri=jnp.full((_K,), r), kinv=_K, kpbltype=0))
    lo, hi = float(jax.grad(f)(-1.0e-10)), float(jax.grad(f)(1.0e-10))
    assert abs(hi - lo) <= 1.0e-6 * abs(lo)


def test_finite_where_stable_denominator_vanishes():
    """Ri = -0.0951: the INACTIVE stable LTG momentum denominator 1 + 10 Ri/sqrt(1+Ri)
    vanishes there; output and gradient must stay finite (review 2, P1)."""
    f = lambda r: jnp.sum(_excu_case(IFSVDFConfig(), ri=jnp.full((_K,), r), kinv=_K, kpbltype=0))
    r = -0.09512492197250393
    assert np.isfinite(float(f(r))) and np.isfinite(float(jax.grad(f)(r)))


# ---------------------------------------------------------------------------
# Stage 1b: VDFHGHTN
# ---------------------------------------------------------------------------
_HG_INT = ("kinv", "kcbase", "kctop", "kpbltype")
_HG_SCAL = ("zinv", "zcldbase", "eis", "khvfl")
_HG_ROWS = ("ri", "mflx", "slguh", "qtuh", "uuh", "vuh")
_KNAMES = ("RG", "RD", "RV", "RCPD", "RLVTT", "RTT", "RA", "RLMLT", "R2ES", "R3LES", "R4LES",
           "R3IES", "R4IES", "RPI")


def _th(fix, arm, g):
    line = [ln for ln in str(fix[f"{arm}_{g}_log"]).splitlines() if ln.startswith("K ")
            and not ln.startswith("K RG")][-1]
    c = dict(zip(_KNAMES, (float(x.replace("D", "E")) for x in line.split()[1:])))
    th = ifs_thermo(**{k: c[k] for k in _KNAMES[:11]}, R3IES=c["R3IES"], R4IES=c["R4IES"])
    assert th.RPI == c["RPI"]
    return th


def _hghtn(fix, arm, g, cfg):
    th = _th(fix, arm, g)
    T, q, u, v, gf = (_a(fix, g, k) for k in ("T", "q", "u", "v", "geo_full"))
    cptgz = gf + T * th.RCPD                                           # RVTMP2 = 0
    slgs = ifs_slgs(T, q, _a(fix, g, "ql"), gf, th.RCPD, th.RLVTT)
    o = lambda k: jnp.asarray(fix[f"{arm}_{g}_{k}"])
    f = lambda *a: ifs_vdfhghtn(th, *a, dt=float(fix["ptsphy"]), cfg=cfg)
    return jax.vmap(f)(T, q, u, v, slgs, cptgz, _a(fix, g, "qe"), _a(fix, g, "p_half"),
                       _a(fix, g, "p_full"), gf, _a(fix, g, "geo_half"), o("kmfl"), o("khfl"),
                       o("kqfl"), _a(fix, g, "land"))


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS)
def test_hghtn_matches_openifs(fix, arm, g):
    out = _hghtn(fix, arm, g, hard_config())
    for k in _HG_INT:
        np.testing.assert_array_equal(np.asarray(getattr(out, k)), fix[f"{arm}_{g}_{k}"], err_msg=k)
    for k in _HG_SCAL + _HG_ROWS:
        e = _err(getattr(out, k), fix[f"{arm}_{g}_{k}"])
        assert e <= _TOL, (k, e)


_HG_FIX = {}


def _hg_fix():
    if not _HG_FIX:
        _HG_FIX.update(dict(np.load(_FIX)))
    return _HG_FIX


def _hg1(cfg, name, *, dT=0.0, khfl=None, kqfl=None, dt=None, slgs_fn=None, parcel=False):
    """One SYN fixture column through VDFHGHTN (arm N) with targeted overrides;
    returns every real output concatenated (or the CUBASEN parcel)."""
    fx = _hg_fix()
    i = list(fx["SYN_names"]).index(name)
    th = _th(fx, "N", "SYN")
    col = {k: _a(fx, "SYN", k)[i] for k in ("T", "q", "u", "v", "ql", "qe", "p_half", "p_full",
                                             "geo_full", "geo_half", "land")}
    o = lambda k: jnp.asarray(fx[f"N_SYN_{k}"][i])
    T = col["T"] + dT
    khfl = o("khfl") if khfl is None else khfl
    kqfl = o("kqfl") if kqfl is None else kqfl
    if parcel:
        return th, ifs_hghtn_parcel(th, T, col["q"], col["u"], col["v"], col["qe"], col["p_half"],
                                    col["p_full"], col["geo_full"], col["geo_half"], o("kmfl"),
                                    khfl, kqfl, col["land"])[2]
    slgs = ifs_slgs(T, col["q"], col["ql"], col["geo_full"], th.RCPD, th.RLVTT)
    slgs = slgs if slgs_fn is None else slgs_fn(slgs, th, col["p_full"])
    out = ifs_vdfhghtn(th, T, col["q"], col["u"], col["v"], slgs, col["geo_full"] + T * th.RCPD,
                       col["qe"], col["p_half"], col["p_full"], col["geo_full"], col["geo_half"],
                       o("kmfl"), khfl, kqfl, col["land"],
                       dt=float(fx["ptsphy"]) if dt is None else dt, cfg=cfg)
    return jnp.concatenate([jnp.ravel(getattr(out, k)) for k in _HG_SCAL + _HG_ROWS])


def _eis_case(cfg):
    """The two EIS branches 0.5 width apart (smooth side of MAX)."""
    def fn(slgs, th, p_full):
        K = slgs.shape[0]
        _, _, n3, _, n6, _ = njkt_levels(K, jnp.concatenate([p_full[:1], p_full]))
        b1 = slgs[n6 - 1] - slgs[n3 - 1]
        return slgs.at[-1].set(slgs[n3 - 1] - (b1 - 0.5 * IFSVDFConfig().eis_width_k * th.RCPD))
    return _hg1(cfg, "sc_eis_a_sfc0", slgs_fn=fn)


def _mf_cap_case(cfg):
    """Lowest-level mass flux 0.5 width above its CFL cap (clipped side of MIN)."""
    from legoesm.atmosphere.physics.turbulence import ifs_vdf as m
    fx = _hg_fix()
    i = list(fx["SYN_names"]).index("dry1500_sfc0")
    K = fx["SYN_T"].shape[1]
    mf_k = float(_hg1(hard_config(), "dry1500_sfc0", dt=1.0)[len(_HG_SCAL) + K + K - 1])
    pf = fx["SYN_p_full"][i]
    dt = (pf[-1] - pf[-2]) * m._MF_CFL / (_th(fx, "N", "SYN").RG
                                          * (mf_k - 0.5 * IFSVDFConfig().mf_cap_width))
    return _hg1(cfg, "dry1500_sfc0", dt=dt)


def _foedelta_dT():
    """Column shift that puts the lowest cloudy parcel level 0.0025 K above RTT
    (inside the centred +-0.005 K band of the 0.01 K step)."""
    th, b = _hg1(hard_config(), "sc_weak_sfc0", parcel=True)
    k = int(np.max(np.nonzero(np.asarray(b["plu"]) > 0.0)[0]))
    f = lambda d: float(_hg1(hard_config(), "sc_weak_sfc0", dT=d, parcel=True)[1]["ptu"][k]) - th.RTT
    d0 = th.RTT - float(b["ptu"][k])
    d = _bisect(f, 0.0025, d0 - 3.0, d0 + 3.0, n=40)
    _, bb = _hg1(hard_config(), "sc_weak_sfc0", dT=d, parcel=True)
    assert float(bb["plu"][k]) > 0.0 and 0.0 < float(bb["ptu"][k]) - th.RTT < 0.005
    return d


_SWITCH_CASES.update({
    "eis_width_k": _eis_case,
    "khvfl_pos_width": lambda cfg: _hg1(cfg, "dry1500_sfc0", khfl=-5.0e-4, kqfl=0.0),
    "wsfc_pow_width": lambda cfg: _hg1(cfg, "dry1500_sfc0", khfl=-5.0e-4, kqfl=0.0),
    "mf_cap_width": _mf_cap_case,
    # weak flux: level-to-level mass-flux changes inside the 5e-4 band
    "dmf_split_width": lambda cfg: _hg1(cfg, "dry1500_sfc0", khfl=-1.0e-6, kqfl=0.0),
    "foedelta_width_k": lambda cfg: _hg1(cfg, "sc_weak_sfc0", dT=_foedelta_dT()),
})


@pytest.mark.parametrize("name", ("dry1500_sfc1", "dry1500_sfc0", "sc_eis_b_sfc0", "sc_weak_sfc0"))
def test_hghtn_gradients_at_default(fix, name):
    """check_grads through VDFHGHTN at the smooth default, one SYN column per PBL type."""
    i = list(fix["SYN_names"]).index(name)
    assert int(fix["N_SYN_kpbltype"][i]) == ("dry1500_sfc1", "dry1500_sfc0", "sc_eis_b_sfc0",
                                             "sc_weak_sfc0").index(name)
    f = lambda dT, khfl: jnp.sum(_hg1(IFSVDFConfig(), name, dT=dT, khfl=khfl) ** 2)
    args = (jnp.zeros(fix["SYN_T"].shape[1]), jnp.asarray(fix["N_SYN_khfl"][i]))
    check_grads(f, args, order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-5, eps=1e-6)
    gr = jax.grad(f, argnums=(0, 1))(*args)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in gr)
    assert any(np.any(np.asarray(x) != 0.0) for x in gr)


# ---------------------------------------------------------------- stage 2
# Inputs as the driver read them (make_fixture.py, vote 7 D5); the solvers are
# fed the oracle's own stage-1 intermediates (CFM/CFH/TOFDC/MFLX/SLGUH/QTUH).
_KTOP = 1
_DIFM = ("udif", "vdif", "dtutofd", "dtvtofd", "dtuso", "dtvso", "dtuvdf", "dtvvdf")
_DIFH = ("tdif", "qdif", "slgevdf", "qtevdf", "tskh", "tskte")
_INCR = ("vom", "vol", "slge", "qte", "slgewodis", "vdis", "vdisg", "strtu", "strtv", "strsou",
         "strsov", "strtofdu", "strtofdv", "tofdu", "tofdv", "disgw3d")


def _s2_inputs(fix, arm, g, i):
    th = _th(fix, arm, g)
    a = lambda k: _a(fix, g, k)[i]
    o = lambda k: jnp.asarray(fix[f"{arm}_{g}_{k}"][i])
    tile, sfc2 = a("tile"), a("sfc2")
    T, q, ql = a("T"), a("q"), a("ql")
    return dict(
        th=th, u=a("u"), v=a("v"), p_half=a("p_half"), T=T, qe=a("qe"),
        cfm=o("cfm").at[-1].set(sfc2[3]), cfh=o("cfh"), tofdc=o("tofdc"), mflx=o("mflx"),
        sluh=o("slguh"), qtuh=o("qtuh"), sobeta=a("sobeta"), soteu=a("soteu"), sotev=a("sotev"),
        frti=tile[:8], cfhti=tile[8:16], tskti=tile[16:], evapsnw=sfc2[0], ucurr=sfc2[1],
        vcurr=sfc2[2], cflx=sfc2[4:6], wd=sfc2[6:8], c=jnp.stack([a("c1"), a("c2")], axis=1),
        slg=a("geo_full") + T * th.RCPD - th.RLVTT * ql, qt=q + ql, shf=a("shf"), lhf=a("lhf"),
        dt=float(fix["ptsphy"]))


@jax.jit   # one compile per grid: eager op-by-op runs exhaust XLA's CPU code space
def _s2_run(x):
    th, dt, g = x["th"], x["dt"], x["th"].RG
    z = jnp.zeros_like(x["u"])
    m = ifs_vdfdifm(x["u"], x["v"], x["p_half"], x["cfm"], x["tofdc"], x["soteu"], x["sotev"],
                    x["sobeta"], z, z, x["ucurr"], x["vcurr"], dt, g, _KTOP)
    h = ifs_vdfdifh_flux(th, x["slg"], x["qt"], x["p_half"], x["cfh"], x["mflx"], x["sluh"],
                         x["qtuh"], z, x["qe"], x["frti"], x["evapsnw"], x["tskti"], x["cfhti"],
                         x["shf"], x["lhf"], dt, _KTOP)
    n = ifs_vdfincr(x["u"], x["v"], x["p_half"], m.dtutofd, m.dtvtofd, m.dtuso, m.dtvso,
                    m.dtuvdf, m.dtvvdf, z, z, z, x["qe"], h.slgevdf, h.qtevdf, dt, g)
    c = ifs_vdfdifc(x["c"], jnp.zeros_like(x["c"]), x["p_half"], x["cfh"], x["wd"], x["cflx"],
                    dt, g, _KTOP)
    return m, h, n, c


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS)
def test_solvers_match_openifs(fix, arm, g):
    n = len(fix[f"{g}_names"])
    res = [_s2_run(_s2_inputs(fix, arm, g, i)) for i in range(n)]
    for j, names in ((0, _DIFM), (1, _DIFH), (2, _INCR)):
        for k in names:
            ours = np.stack([np.asarray(getattr(r[j], k)) for r in res])
            e = _err(ours, fix[f"{arm}_{g}_{k}"])
            assert e <= _TOL, (k, e)
    tenc = np.stack([np.asarray(r[3].tenc) for r in res])
    for t in (0, 1):
        assert _err(tenc[:, :, t], fix[f"{arm}_{g}_tenc{t + 1}"]) <= _TOL, t
    assert _err(np.stack([np.asarray(r[3].cflxt) for r in res]), fix[f"{arm}_{g}_cflxt"]) <= _TOL


_EPS = float(np.finfo(np.float64).eps)


def _close(w, tend, flux, new, old, dt):
    """sum(dp/g tend) == flux to the fp64 round-off of differencing the column
    state over dt: 100 eps sum(dp/g (|new| + |old|)) / dt (vote 8; measured
    worst 45 eps on the fixture)."""
    lhs = float(np.sum(w * np.asarray(tend)))
    scale = float(np.sum(w * (np.abs(np.asarray(new)) + np.abs(np.asarray(old))))) / dt
    assert abs(lhs - flux) <= 100.0 * _EPS * scale, (lhs, flux, scale)


@pytest.mark.parametrize("g", _GROUPS)
def test_solver_budgets_close(fix, g):
    """Per column: water, s_lg (+ VDFINCR dissipation), momentum and tracer
    budgets close against the surface fluxes."""
    for i in range(len(fix[f"{g}_names"])):
        x = _s2_inputs(fix, "N", g, i)
        m, h, n, c = _s2_run(x)
        dt, th = x["dt"], x["th"]
        w = np.asarray(x["p_half"][1:] - x["p_half"][:-1]) / th.RG
        shf = -float(x["shf"])
        # water: -J_q - snow-tile evaporation
        _close(w, h.qtevdf, -float(x["lhf"]) / th.RLVTT - float(x["frti"][6] * x["evapsnw"]),
               h.qdif, x["qt"], dt)
        # s_lg: -SHF; with VDFINCR the dissipated kinetic energy returns as heat
        _close(w, h.slgevdf, shf, h.tdif, x["slg"], dt)
        u1, v1 = np.asarray(x["u"]), np.asarray(x["v"])
        u3 = u1 + dt * np.asarray(m.dtuvdf + m.dtuso)
        v3 = v1 + dt * np.asarray(m.dtvvdf + m.dtvso)
        dke = 0.5 * (u3 ** 2 + v3 ** 2 - u1 ** 2 - v1 ** 2) / dt
        ke = 0.5 * (u1 ** 2 + v1 ** 2 + u3 ** 2 + v3 ** 2)
        _close(w, np.asarray(n.slge) + dke, shf, np.asarray(h.tdif) + ke, x["slg"], dt)
        # momentum: turbulent stress at the surface against the current (TOFD drag apart)
        for d, t, cur, sol, old in ((m.dtuvdf, m.dtutofd, x["ucurr"], m.udif, x["u"]),
                                    (m.dtvvdf, m.dtvtofd, x["vcurr"], m.vdif, x["v"])):
            _close(w, d - t, -float(x["cfm"][-1] * (sol[-1] - cur)), sol, old, dt)
        # tracers: -(flux + dry deposition)
        for t in (0, 1):
            _close(w, c.tenc[:, t], -float(c.cflxt[t]), x["c"][:, t] + dt * c.tenc[:, t],
                   x["c"][:, t], dt)


def test_solver_gradients(fix):
    """Reverse-mode check_grads through each solver (thomas_solve's adjoint is
    reverse-only)."""
    x = _s2_inputs(fix, "N", "L32", 0)
    kw = dict(order=1, modes=("rev",), atol=1e-6, rtol=1e-5, eps=1e-6)
    z = jnp.zeros_like(x["u"])
    check_grads(lambda u, cfm: jnp.sum(ifs_vdfdifm(
        u, x["v"], x["p_half"], cfm, x["tofdc"], x["soteu"], x["sotev"], x["sobeta"], z, z,
        x["ucurr"], x["vcurr"], x["dt"], x["th"].RG, _KTOP).udif ** 2), (x["u"], x["cfm"]), **kw)
    check_grads(lambda slg, cfh, shf: jnp.sum(jnp.concatenate(ifs_vdfdifh_flux(
        x["th"], slg, x["qt"], x["p_half"], cfh, x["mflx"], x["sluh"], x["qtuh"], z, x["qe"],
        x["frti"], x["evapsnw"], x["tskti"], x["cfhti"], shf, x["lhf"], x["dt"], _KTOP)) ** 2),
        (x["slg"], x["cfh"].at[-1].set(0.0), x["shf"]), **kw)
    check_grads(lambda cc, cfh: jnp.sum(ifs_vdfdifc(
        cc, jnp.zeros_like(cc), x["p_half"], cfh, x["wd"], x["cflx"], x["dt"], x["th"].RG,
        _KTOP).tenc ** 2), (x["c"], x["cfh"].at[-1].set(0.0)), **kw)
    check_grads(lambda d: jnp.sum(ifs_vdfincr(
        x["u"], x["v"], x["p_half"], z, z, z, z, d, z, z, z, z, x["qe"], z, z, x["dt"],
        x["th"].RG).slge ** 2), (z + 1e-3,), **kw)



# ---------------------------------------------------------------- stage 3
# VDFMAIN itself (build_main.sh / driver_main.F90, make_fixture.py --main):
# the same inputs plus the RQM group (top levels nearly dry under cloud).
# Snow evaporation is SURFEXCDRIVER's output there (stubbed 0), not the input.
_FIXM = _DIR / "ifs_vdf_main.npz"
_GAPS_M = _DIR / "ifs_vdf_main_gaps.json"
_GROUPS_M = _GROUPS + ("RQM",)
_M_ROWS = ("te", "qe", "vom", "vol", "kh", "km", "ri", "strtu", "strtv")
_M_SCAL = ("zinv", "pblh", "vdis", "vdisg", "tofdu", "tofdv")


@pytest.fixture(scope="module")
def fixm():
    return dict(np.load(_FIXM))


def _m_cols(fx, arm, g):
    a = lambda k: _a(fx, g, k)
    tile, sfc2 = a("tile"), a("sfc2")
    z = jnp.zeros_like(a("T"))
    return (a("u"), a("v"), a("T"), a("q"), a("ql"), a("qi"), a("p_half"), a("p_full"),
            a("geo_full"), a("geo_half"), a("hrlw"), a("hrsw"), a("z0m"), a("sigflt"), a("shf"),
            a("lhf"), a("ustar"), sfc2[:, 3], tile[:, :8], 0.0 * sfc2[:, 0], tile[:, 16:],
            tile[:, 8:16],
            a("sobeta"), a("soteu"), a("sotev"), sfc2[:, 1], sfc2[:, 2], z, a("qe"), z, z, z, z)


@partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))   # widths pick hard/smooth in Python
def _m_run(th, cols, dt, cfg, tofd_cfg, main_cfg):
    f = lambda *c: ifs_vdfmain(th, *c, dt, cfg, tofd_cfg, main_cfg)
    return jax.vmap(f)(*cols)


def _main(fx, arm, g, cfg, tofd_cfg, main_cfg):
    return _m_run(_th(fx, arm, g), _m_cols(fx, arm, g), float(fx["ptsphy"]), cfg, tofd_cfg,
                  main_cfg)


_HARD_M = (hard_config(), hard_tofd_config(), IFSVDFMainConfig(repqmi_width=0.0))
_SOFT_M = (IFSVDFConfig(), IFSTOFDConfig(), IFSVDFMainConfig())


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("g", _GROUPS_M)
def test_vdfmain_matches_openifs(fixm, arm, g):
    o = _main(fixm, arm, g, *_HARD_M)
    np.testing.assert_array_equal(np.asarray(o.kpbltype), fixm[f"{arm}_{g}_kpbltype"])
    for k in _M_ROWS + _M_SCAL:
        e = _err(getattr(o, k), fixm[f"{arm}_{g}_{k}"])
        assert e <= _TOL, (k, e)
    # liquid and ice tendencies are the two shares of one condensate transfer:
    # scaled together (vote 12 G18), so the oracle's ~1e-27 round-off ice share
    # of an all-liquid transfer is not read against itself
    e = _err(jnp.stack([o.le, o.ie]), np.stack([fixm[f"{arm}_{g}_le"], fixm[f"{arm}_{g}_ie"]]))
    assert e <= _TOL, ("le/ie", e)


def test_vdfmain_fixture_exercises_glue(fixm, fix):
    """The two glue switches fire on the fixture: the REPQMI correction (RQM
    levels end at q + dt qe = REPQMI with condensate removed) and the cloud/fog
    PBL correction (VDFHGHTN type 1 promoted to type 2 by VDFMAIN)."""
    dt = float(fixm["ptsphy"])
    w1 = fixm["RQM_q"][:, :10] + dt * fixm["N_RQM_qe"][:, :10]
    fired = np.abs(w1 - 1.0e-8) < 1e-20          # most RQM top levels (diffusion moistens a few)
    assert fired.mean() > 0.9
    assert np.all(fixm["N_RQM_le"][:, :10][fired] < 0.0) and np.all(fixm["N_RQM_ie"][:, :10][fired] < 0.0)
    promoted = sum(int(np.sum((fix[f"N_{g}_kpbltype"] == 1) & (fixm[f"N_{g}_kpbltype"] == 2)))
                   for g in _GROUPS)
    assert promoted > 0


def _measured_main_gaps(fx):
    gaps = {}
    for g in _GROUPS_M:
        hard = _main(fx, "N", g, *_HARD_M[:2], IFSVDFMainConfig(repqmi_width=0.0))
        soft = _main(fx, "N", g, *_HARD_M[:2], IFSVDFMainConfig())
        for k in ("te", "qe", "le", "ie"):
            gaps[f"{g}_{k}"] = _col_gaps(getattr(soft, k), getattr(hard, k))
    return gaps


def test_repqmi_smooth_gap_per_column(fixm):
    """RULE 5: the smooth REPQMI correction's mismatch against the hard form,
    per fixture column, equals the stored measurement (VDF widths held hard)."""
    gaps = _measured_main_gaps(fixm)
    if os.environ.get("IFS_VDF_REGEN_GAPS") == "1":
        _GAPS_M.write_text(json.dumps(gaps, indent=1, sort_keys=True))
    stored = json.loads(_GAPS_M.read_text())
    assert sorted(stored) == sorted(gaps)
    for k, v in gaps.items():
        np.testing.assert_allclose(v, stored[k], rtol=_GAP_RTOL, atol=_GAP_ATOL, err_msg=k)


def test_repqmi_width_is_live():
    """Non-vacuity of repqmi_width: inside its band (updated humidity 3e-11
    below REPQMI under cloud) the smooth correction differs from the hard one
    and stays finite; far outside both agree."""
    from legoesm.atmosphere.physics._ifs_thermo import lego_thermo
    from legoesm.atmosphere.physics.turbulence import ifs_vdf as m
    th, dt = lego_thermo(), 900.0
    q = jnp.array([m._REPQMI - 3.0e-11, 2.0e-9])
    c = jnp.array([1.0e-6, 1.0e-6])
    z = jnp.zeros(2)
    hard = m._repqmi_correction(th, q, c, 0.3 * c, z, z, z, z, dt, 0.0)
    soft = m._repqmi_correction(th, q, c, 0.3 * c, z, z, z, z, dt, IFSVDFMainConfig().repqmi_width)
    for h, s in zip(hard, soft):
        assert np.all(np.isfinite(np.asarray(s)))
        assert float(h[0]) != float(s[0])
        assert float(h[1]) == float(s[1])


def test_vdfmain_water_budget(fixm):
    """Column total water: tendency integral = surface flux + input tendency
    (the REPQMI correction only moves water between phases)."""
    for g in _GROUPS_M:
        th = _th(fixm, "N", g)
        o = _main(fixm, "N", g, *_SOFT_M)
        dt = float(fixm["ptsphy"])
        for i in range(len(fixm[f"{g}_names"])):
            a = lambda k: np.asarray(fixm[f"{g}_{k}"][i])
            w = (a("p_half")[1:] - a("p_half")[:-1]) / th.RG
            tend = np.asarray(o.qe[i] + o.le[i] + o.ie[i]) - a("qe")
            flux = -float(a("lhf")) / th.RLVTT
            old = a("q") + a("ql") + a("qi")
            _close(w, tend, flux, old + dt * tend, old, dt)


def test_vdfmain_gradient_through_repqmi(fixm):
    """check_grads (reverse) at the smooth default through the REPQMI correction
    on an RQM column: d(top-level tendencies)/d(q, ql) finite and non-zero."""
    th = _th(fixm, "N", "RQM")
    cols = [c[0] for c in _m_cols(fixm, "N", "RQM")]
    dt = float(fixm["ptsphy"])

    def f(q, ql):
        c = list(cols)
        c[3], c[4] = q, ql
        o = ifs_vdfmain(th, *c, dt, *_SOFT_M)
        return jnp.sum((o.qe[:10] + 2.0 * o.le[:10] + 3.0 * o.ie[:10]) * 1e8)

    f = jax.jit(f)
    gq, gl = jax.grad(f, argnums=(0, 1))(cols[3], cols[4])
    assert np.all(np.isfinite(gq)) and np.any(np.asarray(gq[:10]) != 0.0)
    assert np.any(np.asarray(gl[:10]) != 0.0)
    check_grads(f, (cols[3], cols[4]), order=1, modes=("rev",), atol=1e-6, rtol=1e-4,
                eps=1e-12)


def _lane_cols(fx, g="SYN", n=3):
    a = lambda k: np.asarray(fx[f"{g}_{k}"][:n])
    g0 = constants.g
    zf, zh = a("geo_full") / g0, a("geo_half") / g0
    T, q = a("T"), a("q")
    rho = a("p_full") / (constants.R_d * T)
    return dict(u=a("u"), v=a("v"), T=T, q_v=q, p_full=a("p_full"), p_half=a("p_half"),
                z_full=zf, z_half=zh, T_sfc=T[:, -1] + 1.0, q_sfc=q[:, -1] * 1.1, rho=rho,
                q_c=a("ql"), q_i=a("qi"), rad_heating=a("hrlw"), sigflt=a("sigflt"),
                land_fraction=a("land"))


def test_lane_kernel_runs_and_refuses_missing_inputs(fixm):
    x = {k: jnp.asarray(v) for k, v in _lane_cols(fixm).items()}
    pos = [x.pop(k) for k in ("u", "v", "T", "q_v", "p_full", "p_half", "z_full", "z_half",
                              "T_sfc", "q_sfc", "rho")]
    flux = (jnp.full(3, 0.1), jnp.full(3, -0.05), jnp.full(3, 20.0), jnp.full(3, 80.0),
            jnp.full(3, 0.3))
    o = ifs_vdf_turbulence(*pos, 900.0, IFSVDFLaneConfig(), flux, **x)
    for k in ("du_dt", "dT_dt", "dq_v_dt", "dq_c_dt", "dq_i_dt", "h_pbl"):
        assert np.all(np.isfinite(np.asarray(getattr(o, k)))), k
    o2 = ifs_vdf_turbulence(*pos, 900.0, IFSVDFLaneConfig(), None, **x)   # bulk fluxes
    assert np.all(np.isfinite(np.asarray(o2.dT_dt)))
    for k in x:
        with pytest.raises(ValueError, match=k):
            ifs_vdf_turbulence(*pos, 900.0, IFSVDFLaneConfig(), flux, **{**x, k: None})


def test_lane_kernel_gradient_at_rest(fixm):
    """The surface momentum coefficient's wind floor keeps the gradient finite
    with the lowest-level wind at rest (vote 11 G15)."""
    x = {k: jnp.asarray(v) for k, v in _lane_cols(fixm, n=1).items()}
    u = x.pop("u").at[:, -1].set(0.0)
    v = x.pop("v").at[:, -1].set(0.0)
    pos = [x.pop(k) for k in ("T", "q_v", "p_full", "p_half", "z_full", "z_half", "T_sfc",
                              "q_sfc", "rho")]
    flux = (jnp.full(1, 0.0), jnp.full(1, 0.0), jnp.full(1, 20.0), jnp.full(1, 80.0),
            jnp.full(1, 0.3))
    g = jax.grad(lambda uu: jnp.sum(ifs_vdf_turbulence(
        uu, v, *pos, 900.0, IFSVDFLaneConfig(), flux, **x).du_dt ** 2))(u)
    assert np.all(np.isfinite(np.asarray(g)))
