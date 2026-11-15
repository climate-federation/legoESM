"""Oracle match: OpenIFS 49r1 VDF column kernels (``turbulence.ifs_vdf``), stage 1a.

Fixture ``oracle_fixtures/ifs_vdf_kernels.npz``: the tier-2 audit's 24
atmosphere columns (16 L32, 5 L137, 3 stress), each with four surface sets,
and the outputs of RUNNING OpenIFS 8f6f722f73 arpifs/phys_ec/{vdfdpbl,
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
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.ifs_vdf import (
    IFSTOFDConfig, IFSVDFConfig, hard_config, hard_tofd_config, ifs_tofd_coefficient,
    ifs_vdfdpbl, ifs_vdfexcu,
)

_DIR = Path(__file__).parent / "oracle_fixtures"
_FIX = _DIR / "ifs_vdf_kernels.npz"
_GAPS = _DIR / "ifs_vdf_kernels_gaps.json"
_TOL = 1.0e-12                     # oracle bar (vote 3 T2)
_GAP_RTOL, _GAP_ATOL = 1.0e-6, 1.0e-12   # RULE 5 stored-gap tolerance
_GROUPS = ("L32", "L137", "STRESS")
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
    kmax = lambda lk: float(jnp.max(_excu_case(hard_config(), kmfl=10.0 ** lk, out="kh")))
    lk = _bisect(kmax, 1.0e6 - 5.0e3, 0.0, 14.0)
    return _excu_case(cfg, kmfl=10.0 ** lk)


def _sqrt_case(cfg):
    gh = _col()["gh"]
    gh = gh.at[9].set(0.995 * gh[8])                # PGEOH(IINV+1) just below PGEOH(IINV)
    return _excu_case(cfg, kpbltype=2, hr=jnp.full((_K,), -1.0e-4), gh=gh)


def _cbase_case(cfg):
    # MIN(1, z/z_cloudbase) is one-sided: z/z_cb = 1.005 at JK = 10 (z_cb = PGEOH(11))
    gh = _col()["gh"]
    gh = gh.at[11].set(gh[10] / 1.005)
    return _excu_case(cfg, kpbltype=3, khfl=-0.01, kqfl=-1.0e-4, kcbase=12, kctop=8, gh=gh)


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
    "lmax_width_m": lambda cfg: _stable_case(cfg, 2985.0),
    # MIN(hr(IINV), hr(IINV-1)) is one-sided: the lower level's rate 5e-7 K/s below
    "hr_min_width": lambda cfg: _excu_case(
        cfg, kpbltype=2, hr=jnp.full((_K,), -1.0e-4).at[7].add(-5.0e-7)),
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
        cfg, kmfl=0.0, khvfl=-0.995 * _guard_khvfl()),
}


def _guard_khvfl():
    """KHVFL that puts |1/L| at 99.5 % of its 1/ZEPSILON cap (u* at REPUST)."""
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
