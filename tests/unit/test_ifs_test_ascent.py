import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.convection import _ifs_test_ascent as ta
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.thermo import saturation_mixing_ratio


def column(kind, nlev=30, ps=101300.0, sigma_half=None):
    if sigma_half is None:
        sig_h = np.linspace(0.0, 1.0, nlev + 1)
    else:
        sig_h = np.asarray(sigma_half, dtype=np.float64)
        nlev = len(sig_h) - 1
    p_half = (sig_h * ps)[None, :].astype(np.float32).copy()
    p_half[0, 0] = max(p_half[0, 0], 1000.0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    pf = p_full[0].astype(np.float64)
    T0, q0 = 300.0, 0.017
    rcpl = constants.R_d / constants.c_pd
    if kind == "deep":
        Tad = np.asarray(compute_moist_adiabat(jnp.array([T0]), jnp.array(pf)[None, :], jnp.array([q0])))[0]
        T = np.where(pf > 95000, T0 * (pf / ps) ** rcpl, Tad - 1.0)
        T = np.where(pf < 15000, np.maximum(T, 200.0), T)
        qs = np.asarray(saturation_mixing_ratio(jnp.array(T), jnp.array(pf)))
        q = np.where(pf > 95000, q0, 0.8 * qs)
        q = np.minimum(q, q0)
    else:  # "trade"
        th = np.where(pf > 95000, 298.7,
                      np.where(pf > 85000, 298.7 + 2.0 * (95000 - pf) / 10000,
                               np.where(pf > 80000, 300.7 + 6.0 * (85000 - pf) / 5000,
                                        306.7 + 3.0 * (80000 - pf) / 10000)))
        T = th * (pf / 1e5) ** rcpl
        qs = np.asarray(saturation_mixing_ratio(jnp.array(T), jnp.array(pf)))
        q = np.where(pf > 95000, 0.017, np.where(pf > 85000, 0.85 * qs, 0.3 * qs))
        q = np.minimum(q, 0.017)
    return T[None, :].astype(np.float32), q[None, :].astype(np.float32), p_full, p_half


def _geopotential(T, q, p_half):
    Tv = T * (1.0 + (1.0 / constants.epsilon - 1.0) * q)
    dphi = constants.R_d * Tv * np.log(p_half[:, 1:] / p_half[:, :-1])
    geo_half = np.zeros_like(p_half, dtype=np.float64)
    geo_half[:, :-1] = np.flip(np.cumsum(np.flip(dphi, axis=1), axis=1), axis=1)
    geo_full = 0.5 * (geo_half[:, :-1] + geo_half[:, 1:])
    return geo_half, geo_full


def run(kind, nlev, r, sigma_half=None, T_perturb=None):
    if sigma_half is None:
        T, q, p_full, p_half = column(kind, nlev=nlev)
    else:
        T, q, p_full, p_half = column(kind, sigma_half=sigma_half)
    if T_perturb is not None:
        T = T + T_perturb.astype(np.float32)
    geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64), p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    nlev_actual = T.shape[1]
    cfg = ta.IFSTestAscentConfig(column_refine=r)
    res = ta.ifs_departure_search_refined(
        f32(T), f32(q), f32(p_full), f32(p_half), f32(geo_full), f32(geo_half),
        jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32), jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
        jnp.zeros((1, nlev_actual), jnp.float32), cfg)
    return res, p_full, T, q


def _hpa(p_full, k):
    return float(p_full[0, int(k)]) / 100.0


def test_deep_sounding_native_l60_is_deep():
    res, p_full, _, _ = run("deep", 60, 1)
    assert int(res.ktype[0]) == 1
    assert abs(_hpa(p_full, res.k_dpl[0]) - 988.0) <= 2.0
    assert abs(_hpa(p_full, res.k_cbot[0]) - 954.0) <= 2.0
    assert abs(_hpa(p_full, res.k_ctop[0]) - 650.0) <= 2.0
    assert abs(float(res.w_base[0]) - 3.42) <= 0.05


def test_l30_misclassifies_deep_as_shallow():
    res, p_full, _, _ = run("deep", 30, 1)
    assert int(res.ktype[0]) == 2
    assert abs(_hpa(p_full, res.k_dpl[0]) - 996.0) <= 2.0
    assert abs(_hpa(p_full, res.k_cbot[0]) - 962.0) <= 2.0
    assert abs(_hpa(p_full, res.k_ctop[0]) - 895.0) <= 2.0
    assert abs(float(res.w_base[0]) - 1.53) <= 0.05


def test_refined_column_recovers_deep():
    res2, p_full, _, _ = run("deep", 30, 2)
    assert int(res2.ktype[0]) == 1
    assert abs(_hpa(p_full, res2.k_dpl[0]) - 996.0) <= 2.0
    assert abs(_hpa(p_full, res2.k_cbot[0]) - 962.0) <= 2.0
    assert abs(_hpa(p_full, res2.k_ctop[0]) - 625.0) <= 5.0
    assert abs(float(res2.w_base[0]) - 2.88) <= 0.05
    res4, p_full4, _, _ = run("deep", 30, 4)
    assert int(res4.ktype[0]) == 1
    assert abs(_hpa(p_full4, res4.k_ctop[0]) - 591.0) <= 10.0
    assert abs(float(res4.w_base[0]) - 2.89) <= 0.05


def test_refined_gate_native_l60_vs_l30_x2():
    for kind, top_l60, top_l30 in (("deep", 650.0, 625.0), ("trade", 802.0, 794.0)):
        r60, pf60, _, _ = run(kind, 60, 1)
        r30, pf30, _, _ = run(kind, 30, 2)
        assert int(r60.ktype[0]) == int(r30.ktype[0])
        assert abs(_hpa(pf60, r60.k_dpl[0]) - _hpa(pf30, r30.k_dpl[0])) <= 34.0
        assert abs(_hpa(pf60, r60.k_ctop[0]) - _hpa(pf30, r30.k_ctop[0])) <= 50.0
        assert abs(_hpa(pf60, r60.k_ctop[0]) - top_l60) <= 2.0
        assert abs(_hpa(pf30, r30.k_ctop[0]) - top_l30) <= 5.0
    r60, _, _, _ = run("deep", 60, 1)
    assert int(r60.ktype[0]) == 1


def test_warm_cap_control_is_not_deep():
    _, p_full, T, q = run("deep", 60, 1)
    pert = np.where((p_full[0] >= 80000.0) & (p_full[0] <= 85000.0), 3.0, 0.0)[None, :]
    res, _, _, _ = run("deep", 60, 1, T_perturb=pert)
    assert int(res.ktype[0]) in (0, 2)


def test_refine_one_is_identity():
    for kind in ("deep", "trade"):
        T, q, p_full, p_half = column(kind, nlev=30)
        geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                           p_half.astype(np.float64))
        args = (T, q, p_full, p_half, geo_full.astype(np.float32), geo_half.astype(np.float32),
                jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32), jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
                np.zeros_like(T))
        a = ta.ifs_departure_search_refined(*args, ta.IFSTestAscentConfig(column_refine=1))
        b = ta.ifs_departure_search(*args, ta.IFSTestAscentConfig())
        for field in ta.TestAscent._fields:
            assert np.array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field))), field


def test_jit_eager_parity():
    T, q, p_full, p_half = column("deep", nlev=30)
    geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                       p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    args = (f32(T), f32(q), f32(p_full), f32(p_half), f32(geo_full), f32(geo_half),
            jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32), jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
            jnp.zeros_like(f32(T)))
    cfg = ta.IFSTestAscentConfig(column_refine=2)
    eager = ta.ifs_departure_search_refined(*args, cfg)
    jitted = jax.jit(lambda *a: ta.ifs_departure_search_refined(*a, cfg))(*args)
    for field in ("ktype", "k_dpl", "k_cbot", "k_ctop", "klab", "k_botsc"):
        assert np.array_equal(np.asarray(getattr(eager, field)), np.asarray(getattr(jitted, field))), field
    for field in ("ldcum", "w_base", "T_u", "q_u", "l_u", "cape_test", "w2", "w2_surface", "ldsc"):
        ea, ja = np.asarray(getattr(eager, field), np.float64), np.asarray(getattr(jitted, field), np.float64)
        assert np.allclose(ea, ja, rtol=5e-3, atol=1e-5), field


def test_output_validity_ensemble():
    rng = np.random.default_rng(0)
    Ts, qs, phs = [], [], []
    for kind in ("deep", "trade"):
        for _ in range(25):
            T, q, p_full, p_half = column(kind, nlev=30)
            Ts.append(T + rng.uniform(-0.3, 0.3, T.shape).astype(np.float32))
            qs.append(q * rng.uniform(0.98, 1.02, q.shape).astype(np.float32))
            phs.append(p_half)
    T = np.concatenate(Ts, 0); q = np.concatenate(qs, 0); p_half = np.concatenate(phs, 0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                       p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    res = ta.ifs_departure_search_refined(
        f32(T), f32(q), f32(p_full), f32(p_half), f32(geo_full), f32(geo_half),
        np.full(50, -20.0, np.float32), np.full(50, -150.0, np.float32),
        np.full(50, 0.1, np.float32), np.zeros(50, np.float32),
        np.zeros_like(f32(T)), ta.IFSTestAscentConfig(column_refine=1))
    for i in range(50):
        kt = int(res.ktype[i])
        assert kt in (0, 1, 2)
        if kt > 0:
            assert 0 <= int(res.k_ctop[i]) <= int(res.k_cbot[i]) < 30
            assert float(res.w_base[i]) >= 0.0 and np.isfinite(float(res.w_base[i]))
        else:
            assert int(res.k_cbot[i]) == -1 and int(res.k_ctop[i]) == -1
            assert int(res.k_dpl[i]) == -1 and float(res.w_base[i]) == 0.0


@pytest.mark.xfail(strict=True, reason="masked-arithmetic gradient hazard, tracked")
def test_gradient_finite():
    T, q, p_full, p_half = column("deep", nlev=30)
    geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                       p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))

    def loss(Ta):
        out = ta.ifs_departure_search(Ta, f32(q), f32(p_full), f32(p_half),
                                      f32(geo_full), f32(geo_half),
                                      jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32),
                                      jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
                                      jnp.zeros((1, 30), jnp.float32),
                                      ta.IFSTestAscentConfig())
        return jnp.mean(out.w_base)

    grad = jax.grad(loss)(f32(T))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_config_validation():
    T, q, p_full, p_half = column("deep", nlev=30)
    geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                       p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    args = (f32(T), f32(q), f32(p_full), f32(p_half), f32(geo_full), f32(geo_half),
            jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32), jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
            jnp.zeros_like(f32(T)))
    with pytest.raises(ValueError):
        ta.ifs_departure_search(*args, ta.IFSTestAscentConfig(mixed_layer_gate="bogus"))
    with pytest.raises(ValueError):
        ta.ifs_departure_search_refined(*args, ta.IFSTestAscentConfig(column_refine=0))
    with pytest.raises(ValueError):
        ta.ifs_departure_search_refined(*args, ta.IFSTestAscentConfig(column_refine=-1))


def _l45_sigma_half():
    old = np.linspace(0.01, 1.0, 31)
    upper = 0.002 * (old[3] / 0.002) ** (np.arange(10) / 9.0)
    upper[0], upper[-1] = 0.002, old[3]
    mid = old[3:22]
    low = np.linspace(old[21], 1.0, 19)
    return np.concatenate([upper[:-1], mid, low[1:]])


def test_l45_draft_grid():
    sig = _l45_sigma_half()
    res_d, pf_d, _, _ = run("deep", None, 1, sigma_half=sig)
    assert int(res_d.ktype[0]) == 1
    assert abs(_hpa(pf_d, res_d.k_ctop[0]) - 662.0) <= 20.0
    res_t, pf_t, _, _ = run("trade", None, 1, sigma_half=sig)
    assert int(res_t.ktype[0]) == 2
    assert abs(_hpa(pf_t, res_t.k_ctop[0]) - 804.0) <= 20.0