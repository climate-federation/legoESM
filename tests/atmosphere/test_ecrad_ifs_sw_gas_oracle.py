"""ecRad 49r1 RRTMG SW gas optics (ecrad_ifs/rrtmg_sw.py) vs the Fortran oracle.

Each stage is fed the oracle's previous-stage arrays, then the chain runs end to
end. Bar: max |a-b|/max(|a|,|b|) <= 1e-12 (fp64), integers exact. Night columns
(cos_sza <= 0) are skipped for the SRTM_SETCOEF / SRTM_GAS_OPTICAL_DEPTH dumps
because the oracle does not compute them there (stale LW values); the final
props are checked on every column.

ECRAD_IFS49_FIXTURE overrides the fixture (the 32-column harness file).
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_sw as sw
from legoesm.atmosphere.physics.radiation.ecrad_ifs import tables

jax.config.update("jax_enable_x64", True)

_FIX = os.environ.get(
    "ECRAD_IFS49_FIXTURE",
    os.path.join(os.path.dirname(__file__), "oracle_fixtures", "ecrad_ifs49.npz"))
_TOL = 1e-12
_BLOCK = 8  # config49r1_rrtmg.nam nblocksize


@pytest.fixture(scope="module")
def z():
    with np.load(_FIX) as f:
        return {k: f[k] for k in f.files}


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), "non-finite in comparison"
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)))


def test_rel_rejects_nan():
    with pytest.raises(AssertionError):
        _rel([np.nan], [1.0])


def _prep(z):
    return (z["dump:lw_prep_zpavel"], z["dump:lw_prep_ztavel"],
            z["dump:lw_prep_zcoldry"], z["dump:lw_prep_zwkl"])


def _day(z):
    return z["dump:in_cos_sza"] > 0.0


def _coef_from_dump(z):
    d = lambda n: jnp.asarray(z["dump:sw_coef_" + n])  # noqa: E731
    jp = d("jp")
    return sw.SwCoef(
        laytrop=d("ilaytrop"), jp=jp, jt=d("jt"), jt1=d("jt1"),
        fac00=d("zfac00"), fac01=d("zfac01"), fac10=d("zfac10"), fac11=d("zfac11"),
        colh2o=d("zcolh2o"), colco2=d("zcolco2"), colo3=d("zcolo3"),
        colch4=d("zcolch4"), colo2=d("zcolo2"), colmol=d("zcolmol"),
        forfac=d("zforfac"), forfrac_lo=d("zforfrac"), indfor_lo=d("indfor"),
        forfrac_up=d("zforfrac"), indfor_up=d("indfor"),
        selffac_lo=d("zselffac"), selffrac_lo=d("zselffrac"), indself_lo=d("indself"),
        w_up=(jp >= 13).astype(jnp.float64), jpc=jp.astype(jnp.float64))


def test_setcoef_matches_oracle(z):
    c = sw.srtm_setcoef(*map(jnp.asarray, _prep(z)), sharpness=math.inf)
    h = c.hard()
    day = _day(z)
    got = dict(ilaytrop=c.laytrop, jp=c.jp, jt=c.jt, jt1=c.jt1, indfor=h["indfor"],
               indself=h["indself"], zfac00=c.fac00, zfac01=c.fac01, zfac10=c.fac10,
               zfac11=c.fac11, zcolh2o=c.colh2o, zcolco2=c.colco2, zcolo3=c.colo3,
               zcolch4=c.colch4, zcolo2=c.colo2, zcolmol=c.colmol, zforfac=c.forfac,
               zforfrac=h["forfrac"], zselffac=h["selffac"], zselffrac=h["selffrac"])
    for name, v in got.items():
        ref = z["dump:sw_coef_" + name][day]
        v = np.asarray(v)[day]
        if np.issubdtype(ref.dtype, np.integer):
            np.testing.assert_array_equal(v, ref, err_msg=name)
        else:
            assert _rel(v, ref) <= _TOL, (name, _rel(v, ref))
    # split weight at the hard limit is exactly the oracle's layer split
    lay = np.arange(1, c.jp.shape[1] + 1)
    np.testing.assert_array_equal(
        np.asarray(c.w_up)[day] == 1.0, lay[None] > z["dump:sw_coef_ilaytrop"][day, None])


def test_gas_optical_depth_from_oracle_coef(z):
    zod, zssa, zinc = sw.srtm_gas_optical_depth(_coef_from_dump(z), sharpness=math.inf)
    day = _day(z)
    for v, n in ((zod, "zod_sw"), (zssa, "zssa_sw"), (zinc, "zincsol")):
        r = _rel(np.asarray(v)[day], z["dump:sw_od_" + n][day])
        assert r <= _TOL, (n, r)


def test_outputs_from_oracle_od(z):
    od, ssa, inc = sw.sw_outputs(
        jnp.asarray(z["dump:sw_od_zod_sw"]), jnp.asarray(z["dump:sw_od_zssa_sw"]),
        jnp.asarray(z["dump:sw_od_zincsol"]), jnp.asarray(z["dump:in_cos_sza"]),
        float(z["dump:in_solar_irradiance"]), 0.0, sharpness=math.inf)
    for v, n in ((od, "od_sw"), (ssa, "ssa_sw"), (inc, "incoming_sw")):
        assert _rel(v, z["props:" + n]) <= _TOL, (n, _rel(v, z["props:" + n]))


def test_outputs_gradient_finite_with_night_zero_flux(z):
    """Oracle-stage night columns carry ZINCSOL = 0: gradients must stay finite."""
    zod, zssa, zinc = (jnp.asarray(z["dump:sw_od_" + n]) for n in ("zod_sw", "zssa_sw", "zincsol"))
    mu = jnp.asarray(z["dump:in_cos_sza"])
    assert not bool(_day(z).all()) and float(jnp.abs(zinc[~_day(z)]).max()) == 0.0

    def f(s, zi):
        return jnp.sum(sw.sw_outputs(zod, zssa, zi, mu, s, 0.0)[2])

    gs, gz = jax.grad(f, argnums=(0, 1))(float(z["dump:in_solar_irradiance"]), zinc)
    assert bool(jnp.isfinite(gs)) and bool(jnp.all(jnp.isfinite(gz)))


def _end_to_end(z, sharpness):
    return sw.gas_optics_sw(*map(jnp.asarray, _prep(z)), jnp.asarray(z["dump:in_cos_sza"]),
                            float(z["dump:in_solar_irradiance"]), 0.0, sharpness=sharpness)


def test_end_to_end_matches_oracle(z):
    out = _end_to_end(z, math.inf)
    for v, n in zip(out, ("od_sw", "ssa_sw", "incoming_sw")):
        r = _rel(v, z["props:" + n])
        print(f"end-to-end {n} ({v.shape[0]} cols): max rel {r:.3e}")
        assert r <= _TOL, (n, r)
    day = _day(z)
    c = sw.srtm_setcoef(*map(jnp.asarray, _prep(z)), sharpness=math.inf)
    zod, zssa, zinc = sw.srtm_gas_optical_depth(c, sharpness=math.inf)
    for v, n in ((zod, "zod_sw"), (zssa, "zssa_sw"), (zinc, "zincsol")):
        assert _rel(np.asarray(v)[day], z["dump:sw_od_" + n][day]) <= _TOL, n


def test_jit_matches_eager(z):
    for s in (math.inf, 100.0):
        eager = _end_to_end(z, s)
        jitted = jax.jit(lambda *a: sw.gas_optics_sw(*a, sharpness=s))(
            *map(jnp.asarray, _prep(z)), jnp.asarray(z["dump:in_cos_sza"]),
            float(z["dump:in_solar_irradiance"]), 0.0)
        d = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(eager, jitted))
        print(f"eager vs jit (sharpness={s}): max abs diff {d:.3e}")
        assert d <= 1e-12 * max(float(jnp.max(jnp.abs(a))) for a in eager)


# --- V37: scalar (49r1 block) vs per-column I_LAYSOLFR -------------------------


def _scalar_block(cand, jp, laytrop, day, layreffr, blocks):
    """49r1 srtm_taumol16/17/27/28/29 upper loop: one I_LAYSOLFR per block,
    layer-outer / column-inner. Unset (stale in the oracle) -> NaN."""
    nlev = jp.shape[1]
    out = np.full((jp.shape[0], cand.shape[2]), np.nan)
    for blk in blocks:
        laysolfr = nlev
        for lay in range(1, nlev + 1):
            for col in blk:
                if not day[col] or lay < laytrop[col] + 1:
                    continue
                if jp[col, lay - 2] < layreffr and jp[col, lay - 1] >= layreffr:
                    laysolfr = lay
                if lay == laysolfr:
                    out[col] = cand[col, lay - 1]
    return out


def test_laysolfr_scalar_equals_per_column_on_oracle(z):
    c = sw.srtm_setcoef(*map(jnp.asarray, _prep(z)), sharpness=math.inf)
    day = _day(z)
    cols = z["columns"]
    blocks = [np.flatnonzero(cols // _BLOCK == b) for b in np.unique(cols // _BLOCK)]
    jp, lt = np.asarray(c.jp), np.asarray(c.laytrop)
    for b in sw._TOP_BANDS:
        _, _, cand, mode = sw.band_terms(b, c, math.inf)
        lr = int(tables.table(f"yoesrta{b}.LAYREFFR")[0])
        per_col = np.asarray(sw.solar_source(cand, c.jpc, c.w_up, mode, lr, math.inf))
        scalar = _scalar_block(np.asarray(cand), jp, lt, day, lr, blocks)
        np.testing.assert_array_equal(per_col[day], scalar[day], err_msg=f"band {b}")


def _synthetic(jp, laytrop):
    """(jpc, w_up) at the hard limit from integer jp and laytrop."""
    jp = jnp.asarray(jp)
    lay = jnp.arange(1, jp.shape[1] + 1)
    return jp.astype(jnp.float64), (lay[None] > jnp.asarray(laytrop)[:, None]).astype(jnp.float64)


def test_laysolfr_scalar_differs_on_heterogeneous_block():
    """Two columns of one block whose crossings differ. Column A crosses
    LAYREFFR=18 at layer 4, column B never does. Per column (V37), B takes its
    default layer nlev=6. 49r1 scalar: order (A, B) -> B takes layer 4 (A's
    crossing); order (B, A) -> B is never set (stale ZSFLXZEN in the oracle)."""
    jp = np.array([[10, 14, 16, 18, 20, 22], [10, 14, 15, 16, 17, 17]])
    lt = np.array([1, 1])
    cand = np.arange(12.0).reshape(2, 6, 1) + 1.0
    day = np.array([True, True])
    per_col = np.asarray(sw.solar_source(jnp.asarray(cand), *_synthetic(jp, lt), "top", 18, math.inf))
    np.testing.assert_array_equal(per_col[:, 0], [cand[0, 3, 0], cand[1, 5, 0]])
    ab = _scalar_block(cand, jp, lt, day, 18, [np.array([0, 1])])
    np.testing.assert_array_equal(ab[:, 0], [cand[0, 3, 0], cand[1, 3, 0]])
    ba = _scalar_block(cand, jp, lt, day, 18, [np.array([1, 0])])
    assert ba[0, 0] == cand[0, 3, 0] and np.isnan(ba[1, 0])


# --- differentiability -------------------------------------------------------


def _day_cols(z):
    d = _day(z)
    p, t, cd, w = _prep(z)
    return p[d], t[d], cd[d], w[d], z["dump:in_cos_sza"][d], float(z["dump:in_solar_irradiance"])


def _loss(p, t, cd, w, mu, s, sharpness=100.0):
    od, ssa, inc = sw.gas_optics_sw(p, t, cd, w, mu, s, 0.0, sharpness=sharpness)
    return jnp.sum(jnp.log(od)) + jnp.sum(ssa) + jnp.sum(inc * jnp.arange(inc.shape[-1]))


def test_gradient_finite_nonzero(z):
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    g = jax.grad(_loss, argnums=(0, 1, 3))(p, t, cd, w, mu, s)
    for gi in g:
        assert bool(jnp.all(jnp.isfinite(gi))) and float(jnp.max(jnp.abs(gi))) > 0


def test_gradient_across_tropopause_split(z):
    """Put one layer exactly on jp = 13 (the split): d od / d p finite, nonzero."""
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    k = int(z["dump:sw_coef_ilaytrop"][_day(z)][0])
    p_split = float(np.exp((sw._JP_OFFSET - sw._JP_TROP) / sw._JP_SLOPE - sw._JP_PLOG_SHIFT))

    def f(x):
        pp = p.at[0, k].set(x)
        od, _, _ = sw.gas_optics_sw(pp, t, cd, w, mu, s, 0.0, sharpness=100.0)
        return jnp.sum(od[0, -1 - k])

    g = jax.grad(f)(p_split)
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_gradient_across_specparm_clamp(z):
    """Scale H2O in one layer so band-16 specparm sits on ONEMINUS."""
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    strrat = float(tables.table("yoesrta16.STRRAT1")[0])
    h2o, ch4 = w[0, 0, 0], w[0, 5, 0]
    a_on = strrat * ch4 * sw._ONEMINUS / (1.0 - sw._ONEMINUS)  # specparm == ONEMINUS

    def f(a):
        ww = w.at[0, 0, 0].set(a)
        od, _, _ = sw.gas_optics_sw(p, t, cd, ww, mu, s, 0.0, sharpness=100.0)
        return jnp.sum(od[0, -1, :6])

    g = jax.grad(f)(a_on)
    assert bool(jnp.isfinite(g)) and float(g) != 0.0 and float(h2o) > 0


def test_check_grads_smooth_point(z):
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    p, t, cd, w, mu = p[:2], t[:2], cd[:2], w[:2], mu[:2]
    check_grads(lambda tt: _loss(p, tt, cd, w, mu, s), (t,), order=1,
                modes=("fwd", "rev"), eps=1e-4, rtol=1e-4, atol=1e-6)


def test_solar_source_out_of_bounds_cases():
    """Votes: no layer above laytrop -> default layer nlev; lay=1 with laytrop=0
    is not a crossing (K_JP(lay-1) is out of bounds in the oracle)."""
    cand = jnp.arange(4.0).reshape(1, 4, 1) + 1.0
    jp = jnp.array([[20, 21, 22, 23]])
    got = sw.solar_source(cand, *_synthetic(jp, [4]), "top", 18, math.inf)
    assert float(got[0, 0]) == 4.0
    got = sw.solar_source(cand, *_synthetic(jp, [0]), "top", 18, math.inf)
    assert float(got[0, 0]) == 4.0  # jp(1) >= 18 but no previous layer


def test_gradient_across_day_night(z):
    """cos_sza inside the night-side band (-1/sharpness, 0) (vote T5): d incoming /
    d mu finite, nonzero."""
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))

    def f(m):
        _, _, inc = sw.gas_optics_sw(p, t, cd, w, mu.at[0].set(m), s, 0.0, sharpness=100.0)
        return jnp.sum(inc[0])

    g = jax.grad(f)(-0.005)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0


def test_gradient_across_solar_source_crossing(z):
    """Band 18 (trop): put the layer above the LAYREFFR crossing exactly on jp = LAYREFFR;
    d(band-18 incoming)/d p is finite and nonzero."""
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    c = sw.srtm_setcoef(p, t, cd, w, sharpness=math.inf)
    lr = sw._layreffr(18)
    jp = np.asarray(c.jp[0])
    k = int(np.flatnonzero((jp[:-1] < lr) & (jp[1:] >= lr))[0]) + 1
    p_on = float(np.exp((sw._JP_OFFSET - lr) / sw._JP_SLOPE - sw._JP_PLOG_SHIFT))
    g0 = sum(sw._ngc()[b] for b in sw._BANDS if b < 18)

    def f(x):
        _, _, inc = sw.gas_optics_sw(p.at[0, k].set(x), t, cd, w, mu, s, 0.0, sharpness=100.0)
        return jnp.sum(inc[0, g0:g0 + sw._ngc()[18]])

    g = jax.grad(f)(p_on)
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def _fortran_solar_source(cand, jp, laytrop, mode, lr):
    """Independent sequential port of the srtm_taumol I_LAYSOLFR loops (per
    column), with the voted out-of-bounds conventions."""
    nlev = jp.shape[1]
    out = np.zeros((jp.shape[0], cand.shape[2]))
    for col in range(jp.shape[0]):
        lt = laytrop[col]
        if mode == "top":
            L = nlev
            for lay in range(lt + 1, nlev + 1):
                if lay > 1 and jp[col, lay - 2] < lr and jp[col, lay - 1] >= lr:
                    L = lay
                if lay == L:
                    out[col] = cand[col, lay - 1]
            if lt >= nlev:
                out[col] = cand[col, nlev - 1]
        else:
            L = lt
            for lay in range(1, lt + 1):
                nxt = min(nlev, lay + 1)
                if mode == "trop" and jp[col, lay - 1] < lr and jp[col, nxt - 1] >= lr:
                    L = min(lay + 1, lt)
                if lay == L:
                    out[col] = cand[col, lay - 1]
    return out


@pytest.mark.parametrize("mode", ["top", "trop", "trop26"])
@pytest.mark.parametrize("lr", [1, 6, 18, 58])
def test_solar_source_matches_sequential_reference(mode, lr):
    rng = np.random.default_rng(lr)
    nlev, ncol = 12, 64
    jp = np.sort(rng.integers(1, 59, size=(ncol, nlev)), axis=1)
    jp[: ncol // 4] = np.sort(rng.integers(1, 13, size=(ncol // 4, nlev)), axis=1)  # laytrop = nlev
    jp[ncol // 4: ncol // 2] = np.sort(rng.integers(13, 59, size=(ncol // 4, nlev)), axis=1)  # laytrop = 0
    jp[ncol // 2: ncol // 2 + 4, :] = lr  # constant at the threshold
    laytrop = np.sum(jp < 13, axis=1)
    cand = rng.random((ncol, nlev, 3)) + np.arange(nlev)[None, :, None]
    got = np.asarray(sw.solar_source(jnp.asarray(cand), *_synthetic(jp, laytrop), mode,
                                     None if mode == "trop26" else lr, math.inf))
    np.testing.assert_array_equal(got, _fortran_solar_source(cand, jp, laytrop, mode, lr))


def _grad_hard_and_smooth(f, x):
    return float(jax.grad(lambda v: f(v, 100.0))(x)), float(jax.grad(lambda v: f(v, math.inf))(x))


def test_switch_derivatives_isolated(z):
    """Each smoothed switch alone carries a nonzero derivative at its transition,
    and the same derivative is exactly 0 in the hard form (the test can fail)."""
    p, t, cd, w, mu, s = map(jnp.asarray, _day_cols(z))
    # tropopause split weight w.r.t. pressure at jpc = 13
    p13 = float(np.exp((sw._JP_OFFSET - sw._JP_TROP) / sw._JP_SLOPE - sw._JP_PLOG_SHIFT))
    f = lambda x, sh: sw.srtm_setcoef(p.at[0, 0].set(x), t, cd, w, sharpness=sh).w_up[0, 0]  # noqa: E731
    g, gh = _grad_hard_and_smooth(f, p13)
    assert np.isfinite(g) and g != 0.0 and gh == 0.0
    # taug depends on the split weight (upper and lower formulas differ there)
    c = sw.srtm_setcoef(p, t, cd, w, sharpness=100.0)
    k = int(c.laytrop[0])
    gw = jax.grad(lambda v: jnp.sum(sw.band_terms(16, c._replace(w_up=c.w_up.at[0, k].set(v)), 100.0)[0][0, k]))(0.5)
    assert np.isfinite(float(gw)) and float(gw) != 0.0
    # specparm clamp just above ONEMINUS
    f = lambda a, sh: sw._spec(a, jnp.asarray(1.0), 1.0, sw._SPEC_LO, sh)[2]  # noqa: E731
    a_on = sw._ONEMINUS / (1.0 - sw._ONEMINUS) * (1.0 + 1e-3)
    g, gh = _grad_hard_and_smooth(f, a_on)
    assert np.isfinite(g) and g != 0.0 and gh == 0.0
    # solar-source crossing weight w.r.t. the pressure index at jp = LAYREFFR
    jp = np.array([[3, 5, 6, 8]], float)
    cand = jnp.arange(4.0).reshape(1, 4, 1) + 1.0
    w_lo = jnp.zeros((1, 4))
    f = lambda x, sh: sw.solar_source(cand, jnp.asarray(jp).at[0, 2].set(x), w_lo, "trop", 6, sh)[0, 0]  # noqa: E731
    g, gh = _grad_hard_and_smooth(f, 6.0)
    assert np.isfinite(g) and g != 0.0 and gh == 0.0
    # day/night weight inside the night-side band (-1/sharpness, 0)
    f = lambda m, sh: sw._day_weight(jnp.asarray([m]), sh)[0]  # noqa: E731
    g, gh = _grad_hard_and_smooth(f, -0.005)
    assert np.isfinite(g) and g != 0.0 and gh == 0.0


def test_eta_index_stays_in_table_fp32():
    """Vote F2: a / speccomb = 1 with the smoothed cap rounds to 1.0 in fp32; JS
    must stay <= mult (8 lower, 4 upper)."""
    for mult in (sw._SPEC_LO, sw._SPEC_UP):
        for dt in (jnp.float32, jnp.float64):
            for s in (100.0, math.inf):
                _, js, fs = sw._spec(jnp.ones((1,), dt), jnp.zeros((1,), dt), 1.0, mult, s)
                assert int(js[0]) <= int(mult) and 0.0 <= float(fs[0]) <= 1.0, (mult, dt, s)
