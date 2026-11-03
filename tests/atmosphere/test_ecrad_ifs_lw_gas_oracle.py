"""ecRad 49r1 RRTMG LONGWAVE gas optics vs the 49r1 Fortran oracle.

Stage by stage (each stage fed the ORACLE's previous-stage arrays, so a failure
localises) and end to end from the gas-optics inputs, at sharpness=inf, to
1e-12 relative (|a-b|/max(|a|,|b|)) in fp64; integer intermediates exactly.
Plus: eager vs jit, the RULE-5 per-column gaps at the default sharpness, and
gradients through every smoothed transition.

Fixture: tests/atmosphere/oracle_fixtures/ecrad_ifs49.npz (8 of 32 columns);
set ECRAD_IFS49_FIXTURE to the 32-column file to run the same checks on it.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_lw as lw
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_prepare as prep

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
FIXTURE = os.environ.get("ECRAD_IFS49_FIXTURE",
                         os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49.npz"))
TOL = 1e-12
INF = math.inf
# 49r1 constants the oracle ran with (oracle values, not legoESM ones):
RG_49R1 = 9.80665          # bld49/src/yomcst_ecrad.F90:33, RPLRG = 1 (yomdyncore.F90:23)
MIN_GAS_OD_LW = 1.0e-15    # radiation_config.F90:235 default (namelist does not set it)


@pytest.fixture(scope="module")
def z():
    with np.load(FIXTURE) as f:
        return {k: f[k] for k in f.files}


def dump(z, k):
    return z["dump:" + k]


def reldiff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den == 0.0, 0.0, np.abs(a - b) / np.where(den == 0.0, 1.0, den))


def assert_close(name, got, ref, mask=None):
    r = reldiff(got, ref)
    if mask is not None:
        r = np.where(mask, r, 0.0)
    mx = float(r.max())
    print(f"{name}: max rel {mx:.3e}")
    assert np.all(np.isfinite(np.asarray(got))), name
    assert mx <= TOL, f"{name}: max rel {mx:.3e} at {np.unravel_index(r.argmax(), r.shape)}"


def inputs(z):
    gas = jnp.asarray(dump(z, "in_gas_mmr"))
    return dict(p_hl=jnp.asarray(dump(z, "in_pressure_hl")),
                t_hl=jnp.asarray(dump(z, "in_temperature_hl")),
                p_fl=jnp.asarray(dump(z, "in_pressure_fl")),
                t_fl=jnp.asarray(dump(z, "in_temperature_fl")),
                gas=gas, tskin=jnp.asarray(dump(z, "in_skin_temperature")),
                alb=jnp.asarray(dump(z, "in_lw_albedo").T))


def oracle_prep(z):
    return prep.PreparedGases(*(jnp.asarray(dump(z, "lw_prep_z" + n)) for n in
                                ("coldry", "wbrodl", "wkl", "wx", "pavel", "tavel", "pz", "tz")),
                              jnp.asarray(dump(z, "lw_prep_ireflect")))


_COEF_REAL = ["fac00", "fac01", "fac10", "fac11", "forfac", "forfrac", "colh2o", "colco2",
              "colo3", "coln2o", "colch4", "colo2", "co2mult", "colbrd", "selffac", "selffrac",
              "scaleminor", "scaleminorn2", "minorfrac"]
_COEF_INT = ["indfor", "jp", "jt", "jt1", "indself", "indminor"]
_RATS = ["h2oco2", "h2oco2_1", "h2oo3", "h2oo3_1", "h2on2o", "h2on2o_1", "h2och4",
         "h2och4_1", "n2oco2", "n2oco2_1", "o3co2", "o3co2_1"]
# Oracle leaves these undefined (stack garbage) outside one branch (vote V8-lw):
_LOWER_ONLY = {"selffrac", "indself", "rat_h2oo3", "rat_h2oo3_1", "rat_h2on2o",
               "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1", "rat_n2oco2", "rat_n2oco2_1"}
_UPPER_ONLY = {"rat_o3co2", "rat_o3co2_1"}


def lower_mask(z):
    lt = dump(z, "lw_coef_ilaytrop")
    nlev = dump(z, "lw_coef_jp").shape[1]
    return np.arange(nlev)[None, :] < lt[:, None]


def oracle_coef(z):
    """LwCoef from the oracle setcoef dump: both branches = the oracle's
    branch-selected arrays, w = (layer <= laytrop) exactly as the taumols do."""
    def g(n):
        return jnp.asarray(dump(z, "lw_coef_" + n))

    br = lw.Branch(jp=g("jp"), jt=g("jt"), jt1=g("jt1"), fac00=g("zfac00"), fac01=g("zfac01"),
                   fac10=g("zfac10"), fac11=g("zfac11"), indfor=g("indfor"), forfrac=g("zforfrac"),
                   **{"rat_" + r: g("zrat_" + r) for r in _RATS})
    kw = {n: g("z" + n) for n in _COEF_REAL}
    kw.update({n: g(n) for n in _COEF_INT})
    kw.update({"rat_" + r: g("zrat_" + r) for r in _RATS})
    p = oracle_prep(z)
    return lw.LwCoef(laytrop=g("ilaytrop"), layswtch=g("ilayswtch"), laylow=g("ilaylow"),
                     w=jnp.asarray(lower_mask(z), jnp.float64), lo=br, up=br,
                     pavel=p.pavel, coldry=p.coldry, wx=p.wx, **kw)


def compare_prep(z, p):
    for n in ("coldry", "wbrodl", "wkl", "wx", "pavel", "tavel", "pz", "tz"):
        assert_close("prep " + n, getattr(p, n), dump(z, "lw_prep_z" + n))
    np.testing.assert_array_equal(np.asarray(p.ireflect), dump(z, "lw_prep_ireflect"))


def compare_coef(z, c):
    lo = lower_mask(z)
    for n in _COEF_REAL + ["rat_" + r for r in _RATS]:
        m = lo if n in _LOWER_ONLY else (~lo if n in _UPPER_ONLY else None)
        key = "lw_coef_z" + n.replace("rat_", "rat_")
        assert_close("coef " + n, getattr(c, n), dump(z, key), m)
    for n in _COEF_INT:
        m = lo if n in _LOWER_ONLY else np.ones_like(lo)
        np.testing.assert_array_equal(np.asarray(getattr(c, n))[m], dump(z, "lw_coef_" + n)[m], n)
    for n in ("laytrop", "layswtch", "laylow"):
        np.testing.assert_array_equal(np.asarray(getattr(c, n)), dump(z, "lw_coef_i" + n), n)


_setcoef = jax.jit(lw.rrtm_setcoef_140gp, static_argnames="sharpness")
_god = jax.jit(lw.rrtm_gas_optical_depth, static_argnames="sharpness")
_pla = jax.jit(lw.planck_function_atmos, static_argnames="sharpness")
_pls = jax.jit(lw.planck_function_surf, static_argnames="sharpness")


def run_all(i, sharpness):
    p = prep.rrtm_prepare_gases(i["p_hl"], i["p_fl"], i["t_hl"], i["t_fl"], i["gas"], RG_49R1)
    return p, lw.gas_optics_lw(p, i["t_hl"], i["tskin"], i["alb"], MIN_GAS_OD_LW, sharpness)


_run_all = jax.jit(run_all, static_argnames="sharpness")


# ---------------------------------------------------------------- stages
def test_full_levels(z):
    i = inputs(z)
    pfl, tfl = prep.full_levels(i["p_hl"], i["t_hl"])
    assert_close("pressure_fl", pfl, dump(z, "in_pressure_fl"))
    assert_close("temperature_fl", tfl, dump(z, "in_temperature_fl"))


def test_stage_prepare(z):
    i = inputs(z)
    compare_prep(z, prep.rrtm_prepare_gases(i["p_hl"], i["p_fl"], i["t_hl"], i["t_fl"],
                                            i["gas"], RG_49R1))


def test_stage_setcoef(z):
    p = oracle_prep(z)
    compare_coef(z, _setcoef(p.coldry, p.wbrodl, p.wkl, p.pavel, p.tavel, p.wx, sharpness=INF))


def test_stage_gas_optical_depth(z):
    tau, pf = _god(oracle_coef(z), sharpness=INF)
    assert_close("zod_lw", tau, dump(z, "lw_od_zod_lw").transpose(2, 1, 0))
    assert_close("zpfrac", pf, dump(z, "lw_od_zpfrac").transpose(0, 2, 1))


def test_stage_planck(z):
    pf = jnp.asarray(dump(z, "lw_od_zpfrac").transpose(0, 2, 1))
    assert_close("planck_hl", _pla(jnp.asarray(dump(z, "in_temperature_hl")), pf, sharpness=INF),
                 dump(z, "lw_planck_planck_hl").transpose(2, 1, 0))
    surf = _pls(jnp.asarray(dump(z, "in_skin_temperature")), pf[:, 0], sharpness=INF)
    assert_close("planck_surf", surf, dump(z, "lw_planck_lw_emission").T)


def test_stage_outputs(z):
    tau = jnp.asarray(dump(z, "lw_od_zod_lw").transpose(2, 1, 0))
    assert_close("od_lw", lw.lw_outputs(tau, MIN_GAS_OD_LW, INF), z["props:od_lw"])
    emis = dump(z, "lw_planck_lw_emission").T * (1.0 - dump(z, "in_lw_albedo").T)
    assert_close("lw_emission (oracle planck x (1-albedo))", emis, z["props:lw_emission"])


def test_end_to_end(z):
    p, o = _run_all(inputs(z), sharpness=INF)
    compare_prep(z, p)
    compare_coef(z, o.coef)
    assert_close("e2e zod_lw", o.tau, dump(z, "lw_od_zod_lw").transpose(2, 1, 0))
    assert_close("e2e zpfrac", o.pfrac, dump(z, "lw_od_zpfrac").transpose(0, 2, 1))
    assert_close("e2e planck_hl", o.planck_hl, dump(z, "lw_planck_planck_hl").transpose(2, 1, 0))
    assert_close("e2e od_lw", o.od_lw, z["props:od_lw"])
    assert_close("e2e lw_emission", o.lw_emission, z["props:lw_emission"])


@pytest.mark.parametrize("sharpness", [INF, 100.0])
def test_eager_vs_jit(z, sharpness):
    i = inputs(z)
    _, a = run_all(i, sharpness)
    _, b = _run_all(i, sharpness=sharpness)
    for n in ("od_lw", "planck_hl", "lw_emission"):
        r = float(reldiff(getattr(a, n), getattr(b, n)).max())
        print(f"eager vs jit {n} (sharpness={sharpness}): max rel {r:.3e}")
        assert r <= TOL


# ------------------------------------------------- RULE 5: default-sharpness gaps
# KNOWN DIFFERENCE at the production sharpness (100), measured per fixture column
# (columns 0,4,...,28 of the 32) against the 49r1 oracle. Metrics per column:
#   od_abs / planck_abs: max |a-b| over levels and g-points (relative error is 1.0
#     where the oracle od is the 1e-15 floor or the Planck fraction is 0, e.g.
#     bands 12/15 above the tropopause, so it carries no information);
#   od_rel: max |a-b|/max(|a|,|b|) over elements with oracle od > 1e-10;
#   emis_rel: max relative difference of lw_emission.
# Sources (measured): the tropopause blend (od_abs and od_rel, at the layer
# nearest ln p = 4.56; its tail still gives 46 % on a band-13 g-point 0.13 in
# ln p above it); the specparm cap near ONEMINUS carries all of the lw_emission
# gap (0 with that cap hard). Band 7 lower od differs by up to 19 % in the mid
# troposphere (source not isolated).
# Re-measure and update, with the reason, whenever a smooth form or the default
# sharpness changes.
GAPS_DEFAULT = {   # re-measured 2026-11-03 (job 11021161) after V4/V5/V9/V10/V11-lw
    "od_abs": [2.9752129047178073, 2.878973060028386, 2.9057193713563265, 3.045143499010919,
               3.062832230693232, 3.003224331108356, 2.8918367511618044, 2.7341814108439166],
    "planck_abs": [0.03279346977665476, 0.03648409415473575, 0.03862084165333046,
                   0.026137649688367848, 0.023153170090435715, 0.027579309507234306,
                   0.03992268941463184, 0.04455002313970802],
    "od_rel": [0.9997447459789051, 0.9997611340613465, 0.9998622404291261, 0.9999789100094723,
               0.9999829812353078, 0.9999683164699835, 0.9998963737805034, 0.9998351091690104],
    "emis_rel": [0.0021580012614706456, 0.00023060623037458948, 0.006178327611027848,
                 0.0073662530111327385, 0.007387146371731142, 0.007171973440249537,
                 0.007162772367775337, 0.004515671777382144],
}


def column_gaps(z, o):
    od, odr = np.asarray(o.od_lw), z["props:od_lw"]
    pl, plr = np.asarray(o.planck_hl), dump(z, "lw_planck_planck_hl").transpose(2, 1, 0)
    n = od.shape[0]
    return {
        "od_abs": np.abs(od - odr).reshape(n, -1).max(1),
        "planck_abs": np.abs(pl - plr).reshape(n, -1).max(1),
        "od_rel": np.where(odr > 1e-10, reldiff(od, odr), 0.0).reshape(n, -1).max(1),
        "emis_rel": reldiff(o.lw_emission, z["props:lw_emission"]).max(1),
    }


def test_default_sharpness_gap(z):
    _, o = _run_all(inputs(z), sharpness=100.0)
    g = column_gaps(z, o)
    for n, v in g.items():
        print(f"GAP {n!r}: {[float(x) for x in v]!r},")
    if GAPS_DEFAULT is None or FIXTURE.endswith("_all.npz"):
        pytest.skip("gap measurement printed (stored values apply to the 8-column fixture)")
    for n, v in g.items():
        np.testing.assert_allclose(v, GAPS_DEFAULT[n], rtol=1e-6, atol=1e-12, err_msg=n)


# ---------------------------------------------------------------- gradients
# Every smoothed transition: at the default sharpness the gradient at the
# transition point is finite, non-zero and passes check_grads with a step far
# below the transition width; the hard form (sharpness=inf) FAILS the same
# check_grads (jump or kink), so each test would go red if the smoothing were
# removed (non-vacuity control).
WT = jnp.linspace(0.5, 1.5, 140)


def _column(z):
    p = oracle_prep(z)
    return prep.PreparedGases(*(a[:1] for a in p))


def _chain(p, sharpness):
    c = lw.rrtm_setcoef_140gp(p.coldry, p.wbrodl, p.wkl, p.pavel, p.tavel, p.wx, sharpness)
    return lw.rrtm_gas_optical_depth(c, sharpness)[0]


def _layer_tau(z, j, x0):
    p = _column(z)

    def f(x, sharpness):
        q = p._replace(pavel=p.pavel.at[0, j].set(x))
        return jnp.sum(_chain(q, sharpness)[0, j] * WT)
    return f, x0


def _transitions(z):
    """name -> (f(x, sharpness), x0 = transition point, h = transition width
    in x units (1/sharpness in the switched argument), kinked_at_inf)."""
    pref = np.asarray(lw.tables.table("yoerrtrf.PREFLOG"))
    pl = np.log(np.asarray(oracle_prep(z).pavel[0]))
    jt = int(np.argmin(np.abs(pl - 4.56)))
    tab = jnp.asarray(lw.tables.table("yoerrta3.ABSA"))
    base = (2 * 5 + 1) * 9
    one = jnp.ones((1, 1))
    w0 = 0.01                                                         # 1/sharpness

    def major3(x, sharpness):
        sp, _, _ = lw._eta_param(x[None, None], 8.0, sharpness)
        return lw._major3(one, sp, 8.0 * sp, 0.6 * one, 0.4 * one, tab, base, 9,
                          sharpness).sum()

    def adj(spec):
        chi, coldry = 3.0e-7, 1.0e24
        x0 = spec[0] * chi * coldry / lw._ADJ_SCALE                  # rat = threshold
        return ((lambda x, s: lw._adjcol(x, coldry, chi, chi, spec, s)),
                x0, x0 / spec[0] * w0, True)

    c0 = oracle_coef(z)

    def band1(x, sharpness):
        c = c0._replace(pavel=c0.pavel.at[0, 0].set(x))
        return lw._band(1, c, False, sharpness)[0][0, 0].sum()

    p = _column(z)
    fl = float(p.coldry[0, 0]) * 1e-32 / 1e-20                        # floor in wkl units

    def zero_co2(x, sharpness):
        q = p._replace(wkl=p.wkl.at[0, 1, 0].set(x))
        return lw.rrtm_setcoef_140gp(q.coldry, q.wbrodl, q.wkl, q.pavel, q.tavel, q.wx,
                                     sharpness).colco2[0, 0]

    pf = jnp.ones((1, 140)) / 16.0
    pk = np.exp((36.0 - 5) / 5.0 - 0.04)                              # u = 5: jp boundary
    pt, pb = np.exp(4.56), np.exp(pref[0] + 0.2)
    out = {
        "tropopause blend": _layer_tau(z, jt, pt) + (pt * w0, True),
        "fp clamp (ln p bound)": _layer_tau(z, 0, pb) + (pb * w0, True),
        "chi(jp+1) staircase": ((lambda x, s: lw._chi_step(3, x[None], False, s)[0]),
                                pk, pk * w0 / 5.0, True),
        "zero-column floor": (zero_co2, 0.0, fl * w0, True),
        "band1 250 hPa": (band1, lw._B1_P_CORR, w0, True),
        "planck 160 K": ((lambda x, s: lw.planck_function_surf(x[None], pf, s).sum()),
                         160.0, w0, True),
        "od floor": ((lambda x, s: lw.lw_outputs(x[None, None, None], MIN_GAS_OD_LW, s).sum()),
                     MIN_GAS_OD_LW, MIN_GAS_OD_LW * w0, True),
        "specparm cap ONEMINUS": (major3, lw._ONEMINUS, w0, True),
        # the 3-point and 2-point formulas join C1 at 0.125/0.875 (measured: the
        # hard form passes check_grads there), so the control is a value check
        "3-point blend 0.125": (major3, 0.125, w0, False),
        "3-point blend 0.875": (major3, 0.875, w0, False),
    }
    for spec in (lw._ADJ_N2O, lw._ADJ_B6, lw._ADJ_B7_LO, lw._ADJ_B8, lw._ADJ_B13):
        out[f"ratio adjustment {spec}"] = adj(spec)
    return out


_NAMES = ["tropopause blend", "fp clamp (ln p bound)", "chi(jp+1) staircase",
          "zero-column floor", "band1 250 hPa", "planck 160 K", "od floor",
          "specparm cap ONEMINUS", "3-point blend 0.125", "3-point blend 0.875"] + [
          f"ratio adjustment {s}" for s in (lw._ADJ_N2O, lw._ADJ_B6, lw._ADJ_B7_LO,
                                            lw._ADJ_B8, lw._ADJ_B13)]
_EPS = 1e-3          # check_grads step in units of the transition width
_T0 = 0.5 * _EPS     # evaluation point: the hard kink lies inside the FD stencil


@pytest.mark.parametrize("name", _NAMES)
def test_grad_smoothed_transition(z, name):
    f, x0, h, kinked = _transitions(z)[name]
    x0 = jnp.asarray(x0, jnp.float64)
    scale = abs(float(jax.grad(lambda x: f(x, 100.0))(x0))) * h
    g = np.asarray(jax.grad(lambda x: f(x, 100.0))(x0))
    print(f"grad {name}: {g}")
    assert np.isfinite(g) and g != 0.0 and scale > 0.0

    def fn(t, s):            # dimensionless: t in widths, output in gradient*width
        return f(x0 + h * t, s) / scale
    t0 = jnp.asarray(_T0)
    check_grads(lambda t: fn(t, 100.0), (t0,), order=1, modes=("fwd", "rev"), eps=_EPS)
    if kinked:               # control: the hard form has a jump/kink inside the stencil
        with pytest.raises(AssertionError):
            check_grads(lambda t: fn(t, INF), (t0,), order=1, modes=("fwd", "rev"), eps=_EPS)
    else:                    # control: the blend is active one width past the threshold
        t1 = jnp.asarray(1.0)
        assert abs(float(fn(t1, 100.0) - fn(t1, INF))) > 1e-6


def test_safe_pow_grads_and_underflow():
    """V12-lw custom_jvp: check_grads order 2 at interior points; zero (not NaN)
    tangent at base == 0; fp32 band-7-lower adjustment below threshold (base
    underflows) has finite gradients."""
    for b in (1e-3, 0.37, 2.5):
        check_grads(lambda x: lw._safe_pow(x, 0.79), (jnp.asarray(b),), order=2,
                    modes=("fwd", "rev"), eps=1e-6)
    assert float(jax.grad(lambda x: lw._safe_pow(x, 0.79))(jnp.asarray(0.0))) == 0.0
    z0 = jnp.asarray(0.0)
    jj = jax.jvp(lambda x: jax.jvp(lambda y: lw._safe_pow(y, 0.79), (x,), (1.0,))[1], (z0,), (1.0,))
    assert np.all(np.isfinite(np.asarray(jj)))                       # jvp of jvp at 0
    for dt in (jnp.float32, jnp.float64):                            # composed 2nd order at 0
        h = jax.grad(jax.grad(lambda x: lw._safe_pow(x, 0.79) ** 2))(jnp.asarray(0.0, dt))
        assert np.isfinite(float(h)), dt
    chi, coldry = jnp.float32(3.0e-7), jnp.float32(1.0e24)
    for rat in (0.0, 1.0, 2.0):
        col = jnp.float32(rat * 3.0e-7 * 1.0e24 / 1.0e20)
        g = jax.grad(lambda c: lw._adjcol(c, coldry, chi, chi, lw._ADJ_B7_LO, 100.0))(col)
        print(f"fp32 band-7 adjcol grad at rat={rat}: {g}")
        assert g.dtype == jnp.float32 and np.isfinite(float(g))


def test_check_grads_end_to_end(z):
    i = inputs(z)
    i1 = {k: (v[:1] if k != "alb" else v[:1]) for k, v in i.items()}
    wt = jnp.linspace(0.5, 1.5, 140)
    def f(dt, sq):
        j = dict(i1, t_hl=i1["t_hl"] + dt, t_fl=i1["t_fl"] + dt,
                 gas=i1["gas"].at[..., prep.IH2O].multiply(sq))
        _, o = run_all(j, 100.0)
        return (jnp.sum(o.od_lw * wt) + 1e-2 * jnp.sum(o.planck_hl * wt)
                + 1e-2 * jnp.sum(o.lw_emission))
    check_grads(f, (jnp.asarray(0.0123), jnp.asarray(1.0)), order=1, modes=("fwd", "rev"), eps=1e-6)
