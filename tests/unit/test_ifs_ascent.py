
"""Unit tests for _ifs_ascent.ifs_updraught_ascent (IFS main updraught ascent).

Chains exactly as in the reference harness: departure search from
_ifs_test_ascent (refined, column_refine=2), followed by the main ascent
with the cloud-base state from the trigger module.  Sounding builders are
copied verbatim from tests/unit/test_ifs_test_ascent.py to stay
self-contained.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.convection import _ifs_test_ascent as ta
from legoesm.atmosphere.physics.convection import _ifs_ascent as asc
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.thermo import saturation_specific_humidity


# --------------------------------------------------------------------------
# Copied verbatim from tests/unit/test_ifs_test_ascent.py
# --------------------------------------------------------------------------
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
        qs = np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(pf)))
        q = np.where(pf > 95000, q0, 0.8 * qs)
        q = np.minimum(q, q0)
    else:  # "trade"
        th = np.where(pf > 95000, 298.7,
                      np.where(pf > 85000, 298.7 + 2.0 * (95000 - pf) / 10000,
                               np.where(pf > 80000, 300.7 + 6.0 * (85000 - pf) / 5000,
                                        306.7 + 3.0 * (80000 - pf) / 10000)))
        T = th * (pf / 1e5) ** rcpl
        qs = np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(pf)))
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


# --------------------------------------------------------------------------
# Harness (identical chaining to the reference run)
# --------------------------------------------------------------------------
def _run(T, q, p_full, p_half, cfg_t=None, cfg_a=None):
    cfg_t = cfg_t or ta.IFSTestAscentConfig(column_refine=2)
    cfg_a = cfg_a or asc.IFSAscentConfig()
    geo_half, geo_full = _geopotential(
        T.astype(np.float64), q.astype(np.float64),
        p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    T, q, p_full, p_half = map(f32, (T, q, p_full, p_half))
    geo_full, geo_half = f32(geo_full), f32(geo_half)

    tst = ta.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((1,), -20.0, jnp.float32),
        jnp.full((1,), -150.0, jnp.float32),
        jnp.full((1,), 0.1, jnp.float32),
        jnp.zeros((1,), jnp.float32),
        jnp.zeros((1, T.shape[1]), jnp.float32),
        cfg_t)

    qsp = q      # q is already PQEN specific humidity: no seam conversion
    T_h, q_h, s_h = ta.half_level_env(T, qsp, p_full, p_half,
                                       geo_full, geo_half, cfg_t)
    qs = saturation_specific_humidity(T, p_full)

    q_u0 = tst.q_u   # PQU / PLU are on the same moist-mass basis as PQEN
    l_u0 = tst.l_u

    out = asc.ifs_updraught_ascent(
        T, qsp, qs, p_full, p_half, geo_full, geo_half, T_h, q_h,
        plitot=jnp.zeros_like(T),
        ldcum=tst.ldcum > 0.5,
        ktype=tst.ktype,
        k_dpl=jnp.maximum(tst.k_dpl, 0),
        k_cbot=jnp.maximum(tst.k_cbot, 0),
        klab0=tst.klab,
        T_u0=tst.T_u, q_u0=q_u0, l_u0=l_u0,
        w_base=tst.w_base,
        M_b=jnp.asarray(0.02, jnp.float32),
        dt=jnp.asarray(112.5, jnp.float32),
        cfg=cfg_a)
    return out, tst, (T, qsp, qs, p_full, p_half, geo_full, geo_half,
                      T_h, q_h)


def _run_kind(kind):
    T, q, p_full, p_half = column(kind)
    return _run(T, q, p_full, p_half)


# --------------------------------------------------------------------------
# Measured profiles
# --------------------------------------------------------------------------
def test_deep_sounding_profile():
    out, tst, _ = _run_kind("deep")
    p_full = column("deep")[2]
    base = int(tst.k_cbot[0])
    top = int(out.k_ctop[0])
    assert base == 28
    assert abs(float(p_full[0, base]) - 962.0 * 100.0) < 200.0
    assert top == 24
    assert abs(float(p_full[0, top]) - 827.0 * 100.0) < 200.0
    M_ratio = np.asarray(out.M[0] / 0.02)
    expected_M = [1.00, 0.98, 1.16, 1.34, 0.83]
    for i, k in enumerate(range(28, 23, -1)):
        assert M_ratio[k] == pytest.approx(expected_M[i], abs=0.03), (k, i)
    assert np.all(M_ratio[:24] == pytest.approx(0.0, abs=1e-8))
    K = np.asarray(out.PKINEU[0])
    expected_K = [2.04, 4.51, 4.80, 3.26, 1.08]
    for i, k in enumerate(range(28, 23, -1)):
        assert K[k] == pytest.approx(expected_K[i], abs=0.15), (k, i)
    assert np.all(K[:24] == pytest.approx(0.0, abs=1e-6))
    pdmfup_sum = float(np.sum(np.asarray(out.PDMFUP)))
    assert pdmfup_sum > 0.0
    assert pdmfup_sum == pytest.approx(2.04e-5, rel=0.20)


def test_trade_sounding_profile():
    out, tst, _ = _run_kind("trade")
    p_full = column("trade")[2]
    base = int(tst.k_cbot[0])
    top = int(out.k_ctop[0])
    assert base == 28
    assert top == 25
    assert abs(float(p_full[0, top]) - 861.0 * 100.0) < 200.0
    M_ratio = np.asarray(out.M[0] / 0.02)
    expected_M = [1.00, 1.00, 1.09, 1.16]
    for i, k in enumerate(range(28, 24, -1)):
        assert M_ratio[k] == pytest.approx(expected_M[i], abs=0.03), (k, i)
    K = np.asarray(out.PKINEU[0])
    expected_K = [1.26, 5.53, 6.06, 6.41]
    for i, k in enumerate(range(28, 24, -1)):
        assert K[k] == pytest.approx(expected_K[i], abs=0.15), (k, i)
    pdmfup_sum = float(np.sum(np.asarray(out.PDMFUP)))
    assert pdmfup_sum > 0.0
    assert pdmfup_sum == pytest.approx(1.09e-5, rel=0.20)


def test_mass_flux_nonnegative_and_zero_above_top():
    for kind in ("deep", "trade"):
        out, tst, _ = _run_kind(kind)
        M = np.asarray(out.M[0])
        assert np.all(M >= -1e-8)
        top = int(out.k_ctop[0])
        assert np.all(M[:top] == pytest.approx(0.0, abs=1e-8))


def test_flux_consistency():
    from legoesm.constants import c_pd
    for kind in ("deep", "trade"):
        out, tst, env = _run_kind(kind)
        geo_half = env[6]
        M = np.asarray(out.M[0])
        mask = M > 0
        assert np.allclose(
            np.asarray(out.PMFUS[0])[mask],
            (c_pd * np.asarray(out.T_u[0])[mask]
             + np.asarray(geo_half[0])[:M.size][mask]) * M[mask],
            rtol=1e-5)
        assert np.allclose(np.asarray(out.PMFUQ[0])[mask],
                           np.asarray(out.q_u[0])[mask] * M[mask], rtol=1e-5)
        assert np.allclose(np.asarray(out.PMFUL[0])[mask],
                           np.asarray(out.l_u[0])[mask] * M[mask], rtol=1e-5)


def test_terminal_deposition():
    for kind in ("deep", "trade"):
        out, tst, _ = _run_kind(kind)
        top = int(out.k_ctop[0])
        assert top >= 1
        l_at_top = float(out.l_u[0, top])
        M_at_top = float(out.M[0, top])
        assert float(out.PLUDE[0, top - 1]) == pytest.approx(
            l_at_top * M_at_top, rel=1e-5)
        assert float(out.PMFUDE_RATE[0, top - 1]) == pytest.approx(
            M_at_top, rel=1e-5)


def test_klab_labels():
    for kind in ("deep", "trade"):
        out, tst, _ = _run_kind(kind)
        top = int(out.k_ctop[0])
        bot = int(tst.k_cbot[0])
        klab = np.asarray(out.klab[0])
        assert np.all(klab[top:bot + 1] == 2)
        assert np.all(klab[:top] == 0)


def test_inactive_column_untouched():
    T, q, p_full, p_half = column("deep")
    geo_half, geo_full = _geopotential(
        T.astype(np.float64), q.astype(np.float64),
        p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    T, q, p_full, p_half = map(f32, (T, q, p_full, p_half))
    geo_full, geo_half = f32(geo_full), f32(geo_half)
    cfg_t = ta.IFSTestAscentConfig(column_refine=2)
    tst = ta.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((1,), -20.0, jnp.float32),
        jnp.full((1,), -150.0, jnp.float32),
        jnp.full((1,), 0.1, jnp.float32),
        jnp.zeros((1,), jnp.float32),
        jnp.zeros((1, T.shape[1]), jnp.float32),
        cfg_t)
    qsp = q      # q is already PQEN specific humidity: no seam conversion
    T_h, q_h, s_h = ta.half_level_env(T, qsp, p_full, p_half,
                                       geo_full, geo_half, cfg_t)
    qs = saturation_specific_humidity(T, p_full)
    q_u0 = tst.q_u   # PQU / PLU are on the same moist-mass basis as PQEN
    l_u0 = tst.l_u
    nlev = T.shape[1]
    out = asc.ifs_updraught_ascent(
        T, qsp, qs, p_full, p_half, geo_full, geo_half, T_h, q_h,
        plitot=jnp.zeros_like(T),
        ldcum=jnp.zeros((1,), bool),
        ktype=jnp.zeros((1,), jnp.int32),
        k_dpl=jnp.zeros((1,), jnp.int32),
        k_cbot=jnp.maximum(tst.k_cbot, 0),
        klab0=jnp.zeros((1, nlev), jnp.int32),
        T_u0=tst.T_u, q_u0=q_u0, l_u0=l_u0,
        w_base=tst.w_base,
        M_b=jnp.asarray(0.02, jnp.float32),
        dt=jnp.asarray(112.5, jnp.float32),
        cfg=asc.IFSAscentConfig())
    for fld in ("M", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFEN",
                "PMFUDE_RATE", "PDMFUP", "PLRAIN", "PKINEU"):
        assert np.all(np.asarray(getattr(out, fld)) == 0.0), fld
    assert np.allclose(np.asarray(out.T_u), np.asarray(tst.T_u), rtol=1e-6)
    assert np.allclose(np.asarray(out.q_u), np.asarray(q_u0), rtol=1e-6)
    assert np.allclose(np.asarray(out.l_u), np.asarray(l_u0), rtol=1e-6)
    assert int(out.k_ctop[0]) == int(jnp.maximum(tst.k_cbot, 0)[0])


def test_jit_eager_parity():
    T, q, p_full, p_half = column("deep")
    out_eager, tst, env = _run(T, q, p_full, p_half,
                               ta.IFSTestAscentConfig(column_refine=2),
                               asc.IFSAscentConfig())
    (Tj, qsp, qs, pf, ph, gf, gh, T_h, q_h) = env
    q_u0 = tst.q_u   # PQU / PLU are on the same moist-mass basis as PQEN
    l_u0 = tst.l_u
    cfg_a = asc.IFSAscentConfig()

    def model(Tj, qsp, qs, pf, ph, gf, gh, T_h, q_h, ldcum, ktype, k_dpl, k_cbot, klab0, T_u0, q_u0, l_u0, w_base):
        return asc.ifs_updraught_ascent(Tj, qsp, qs, pf, ph, gf, gh, T_h, q_h, jnp.zeros_like(Tj),
                                        ldcum, ktype, k_dpl, k_cbot, klab0, T_u0, q_u0, l_u0, w_base,
                                        jnp.asarray(0.02, jnp.float32), jnp.asarray(112.5, jnp.float32), cfg_a)
    args = (Tj, qsp, qs, pf, ph, gf, gh, T_h, q_h, tst.ldcum > 0.5, tst.ktype, jnp.maximum(tst.k_dpl, 0),
            jnp.maximum(tst.k_cbot, 0), tst.klab, tst.T_u, q_u0, l_u0, tst.w_base)
    out_jit = jax.jit(model)(*args)
    assert np.array_equal(np.asarray(out_jit.k_ctop), np.asarray(out_eager.k_ctop))
    assert np.array_equal(np.asarray(out_jit.klab), np.asarray(out_eager.klab))
    float_fields = ("M", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFEN",
                    "PMFUDE_RATE", "PDMFUP", "PLRAIN", "PKINEU",
                    "T_u", "q_u", "l_u")
    for fld in float_fields:
        a = np.asarray(getattr(out_eager, fld), np.float64)
        b = np.asarray(getattr(out_jit, fld), np.float64)
        assert np.allclose(a, b, rtol=5e-3, atol=1e-8), fld


def test_no_condensation_stop():
    T, q, p_full, p_half = column("deep")
    qs = np.asarray(saturation_specific_humidity(
        jnp.asarray(T), jnp.asarray(p_full)))
    q_dry = np.where(p_full < 940.0 * 100.0, 0.3 * qs, q).astype(np.float32)
    out, tst, _ = _run(T, q_dry, p_full, p_half)
    base = int(tst.k_cbot[0])
    top = int(out.k_ctop[0])
    # plume stops within two layers of the base
    assert top > base - 3
    M = np.asarray(out.M[0])
    assert np.all(M >= -1e-8)
    assert np.all(M[:top] == pytest.approx(0.0, abs=1e-8))


def test_shapes_dtypes():
    T, q, p_full, p_half = column("deep")
    out, tst, _ = _run(T, q, p_full, p_half)
    nlev = T.shape[1]
    for fld in ("M", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFEN",
                "PMFUDE_RATE", "PDMFUP", "PLRAIN", "PKINEU",
                "T_u", "q_u", "l_u"):
        arr = getattr(out, fld)
        assert arr.shape == (1, nlev), fld
        assert arr.dtype == jnp.float32, fld
    for fld in ("klab", "k_ctop"):
        arr = getattr(out, fld)
        assert arr.dtype == jnp.int32, fld
    assert out.k_ctop.shape == (1,)
    assert out.klab.shape == (1, nlev)


@pytest.mark.xfail(strict=True, reason="masked-arithmetic gradient hazard near the plume top, tracked")
def test_gradient_finite():
    """d mean(PKINEU) / dT through the main ascent only (trigger outputs held fixed)."""
    T, q, p_full, p_half = column("deep")
    out_eager, tst, env = _run(T, q, p_full, p_half,
                               ta.IFSTestAscentConfig(column_refine=2),
                               asc.IFSAscentConfig())
    (Tj, qsp, qs, pf, ph, gf, gh, T_h, q_h) = env
    q_u0 = tst.q_u   # PQU / PLU are on the same moist-mass basis as PQEN
    l_u0 = tst.l_u
    cfg_a = asc.IFSAscentConfig()

    def loss(T_):
        out = asc.ifs_updraught_ascent(T_, qsp, qs, pf, ph, gf, gh, T_h, q_h, jnp.zeros_like(T_),
                                       tst.ldcum > 0.5, tst.ktype, jnp.maximum(tst.k_dpl, 0),
                                       jnp.maximum(tst.k_cbot, 0), tst.klab, tst.T_u, q_u0, l_u0, tst.w_base,
                                       jnp.asarray(0.02, jnp.float32), jnp.asarray(112.5, jnp.float32), cfg_a)
        return jnp.mean(out.PKINEU)

    grad = jax.grad(loss)(Tj)
    assert np.all(np.isfinite(np.asarray(grad)))
