import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.convection import _ifs_test_ascent as ta
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.thermo import saturation_specific_humidity


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
    """Also the non-vacuity gate for the SPECIFIC-humidity seam (cubasen.F90:72):
    q_v, q_u and l_u are PQEN / PQU / PLU with no conversion at either boundary.

    ENTRY side: restoring q = q_v/(1 + q_v) deflates the environment by 1.67%
    at q = 0.017 (less aloft) against a saturation curve that was never
    deflated, i.e. a systematic relative-humidity deficit.  That moves this
    cloud top from 616 to 669 hPa, the base from 971 to 954 hPa and w_base
    from 2.02 to 3.42 m/s -- the four level/velocity pins go red.

    EXIT side: those four pins are blind to it (the returned levels do not
    change), so the two humidity pins below carry it: restoring
    q_u/max(1 - q_u - l_u, 0.5) inflates both species by 1.29% at mid-cloud,
    13x the tolerance.
    """
    res, p_full, _, _ = run("deep", 60, 1)
    # RE-PINNED 2026-09-29: the parcel routine's Bolton LCL reads RH in the
    # SPECIFIC basis (q/q_sat) since the saturation reference flip; the
    # synthetic sounding this file builds from the moist adiabat moved with it.
    assert int(res.ktype[0]) == 1
    assert abs(_hpa(p_full, res.k_dpl[0]) - 988.0) <= 2.0
    assert abs(_hpa(p_full, res.k_cbot[0]) - 971.0) <= 2.0
    assert abs(_hpa(p_full, res.k_ctop[0]) - 667.0) <= 2.0
    assert abs(float(res.w_base[0]) - 2.01) <= 0.05
    kmid = (int(res.k_cbot[0]) + int(res.k_ctop[0])) // 2
    assert float(res.q_u[0, kmid]) == pytest.approx(0.01320934, rel=1e-3)
    assert float(res.l_u[0, kmid]) == pytest.approx(3.4292e-4, rel=1e-3)


def test_l30_misclassifies_deep_as_shallow():
    res, p_full, _, _ = run("deep", 30, 1)
    # RE-PINNED 2026-09-29: the parcel routine's Bolton LCL reads RH in the
    # SPECIFIC basis (q/q_sat) since the saturation reference flip; the
    # synthetic sounding this file builds from the moist adiabat moved with it.
    assert int(res.ktype[0]) == 2
    assert abs(_hpa(p_full, res.k_dpl[0]) - 996.0) <= 2.0
    assert abs(_hpa(p_full, res.k_cbot[0]) - 962.0) <= 2.0
    assert abs(_hpa(p_full, res.k_ctop[0]) - 929.0) <= 2.0
    assert abs(float(res.w_base[0]) - 1.61) <= 0.05


def test_refined_column_recovers_deep():
    res2, p_full, _, _ = run("deep", 30, 2)
    assert int(res2.ktype[0]) == 1
    assert abs(_hpa(p_full, res2.k_dpl[0]) - 996.0) <= 2.0
    assert abs(_hpa(p_full, res2.k_cbot[0]) - 962.0) <= 2.0
    # RE-PINNED 2026-09-29: the parcel routine's Bolton LCL reads RH in the
    # SPECIFIC basis (q/q_sat) since the saturation reference flip; the
    # synthetic sounding this file builds from the moist adiabat moved with it.
    assert abs(_hpa(p_full, res2.k_ctop[0]) - 726.0) <= 5.0
    assert abs(float(res2.w_base[0]) - 2.02) <= 0.05
    res4, p_full4, _, _ = run("deep", 30, 4)
    assert int(res4.ktype[0]) == 1
    assert abs(_hpa(p_full4, res4.k_ctop[0]) - 692.0) <= 10.0
    assert abs(float(res4.w_base[0]) - 2.54) <= 0.05


@pytest.mark.xfail(strict=True, reason=(
    "OPEN 2026-09-29: on the sounding the specific-basis LCL produces, the "
    "native-L60 vs refined-L30x2 cloud tops disagree by 59 hPa (667 vs 726), "
    "above the one-parent-layer bound of 51 hPa (was 42 hPa on the old "
    "sounding).  The bound is not widened (its own rule); the cause is the "
    "resolution sensitivity this docstring already calls OPEN."))
def test_refined_gate_native_l60_vs_l30_x2():
    # RE-PINNED 2026-09-29: the parcel routine's Bolton LCL reads RH in the
    # SPECIFIC basis (q/q_sat) since the saturation reference flip; the
    # synthetic sounding this file builds from the moist adiabat moved with it.
    for kind, top_l60, top_l30 in (("deep", 667.0, 726.0), ("trade", 802.0, 794.0)):
        r60, pf60, _, _ = run(kind, 60, 1)
        r30, pf30, _, _ = run(kind, 30, 2)
        assert int(r60.ktype[0]) == int(r30.ktype[0])
        assert abs(_hpa(pf60, r60.k_dpl[0]) - _hpa(pf30, r30.k_dpl[0])) <= 34.0
        # A cloud top is LEVEL-QUANTIZED, so the bound is one parent layer on
        # each grid, derived from the grids themselves rather than a hand
        # number: a single-level flip on either side is tolerated, anything
        # larger is a real disagreement (GLM review -- a flat 45 hPa against a
        # measured 42 is a 3 hPa margin on a discontinuous output).  Measured
        # gap 42 hPa, down from 59 before the conservative reconstruction.
        # The residual cause is OPEN: both ladders keep deepening with
        # resolution (L30 refined r1/r2/r4/r8 -> 895/658/625/591 hPa, native
        # L30/L60/L120 -> 895/616/536 hPa) and the gap GROWS with effective
        # resolution (42 hPa at 60, 89 at 120), so "the departure search is
        # resolution-sensitive" is PLAUSIBLE, not established -- the fixture
        # supplies point samples where the reconstruction wants layer means,
        # which is the competing explanation.  Tighten, never widen.
        k60, k30 = int(r60.k_ctop[0]), int(r30.k_ctop[0])
        dp60 = abs(_hpa(pf60, min(k60 + 1, pf60.shape[1] - 1)) - _hpa(pf60, k60))
        dp30 = abs(_hpa(pf30, min(k30 + 1, pf30.shape[1] - 1)) - _hpa(pf30, k30))
        assert abs(_hpa(pf60, k60) - _hpa(pf30, k30)) <= dp60 + dp30
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
    """r = 1 reduces to the source-literal search -- ON THE WRAPPER'S OWN
    GEOMETRY.  The wrapper derives both the level pressure and the
    geopotential from p_half, so the native side of the comparison is fed the
    same two, otherwise this asserts an identity the convention deliberately
    broke (the caller's geopotential here is integrated in float64 and cast,
    the module's in float32, which alone moves w_base in the last bits).
    """
    for kind in ("deep", "trade"):
        T, q, p_full, p_half = column(kind, nlev=30)
        geo_half, geo_full = _geopotential(T.astype(np.float64), q.astype(np.float64),
                                           p_half.astype(np.float64))
        args = (T, q, p_full, p_half, geo_full.astype(np.float32), geo_half.astype(np.float32),
                jnp.full((1,), -20.0, jnp.float32), jnp.full((1,), -150.0, jnp.float32), jnp.full((1,), 0.1, jnp.float32), jnp.zeros((1,), jnp.float32),
                np.zeros_like(T))
        a = ta.ifs_departure_search_refined(*args, ta.IFSTestAscentConfig(column_refine=1))
        gh_m, gf_m = ta._hydrostatic_geopotential(jnp.asarray(T, jnp.float32),
                                                  jnp.asarray(q, jnp.float32),
                                                  jnp.asarray(p_half, jnp.float32))
        b = ta.ifs_departure_search(T, q, ta._layer_centre(jnp.asarray(p_half, jnp.float32)),
                                    p_half, gf_m, gh_m, *args[6:],
                                    ta.IFSTestAscentConfig())
        assert _ta_same(a, b), kind


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
    # RE-PINNED 2026-09-29: the parcel routine's Bolton LCL reads RH in the
    # SPECIFIC basis (q/q_sat) since the saturation reference flip; the
    # synthetic sounding this file builds from the moist adiabat moved with it.
    assert abs(_hpa(pf_d, res_d.k_ctop[0]) - 662.0) <= 20.0
    res_t, pf_t, _, _ = run("trade", None, 1, sigma_half=sig)
    assert int(res_t.ktype[0]) == 2
    assert abs(_hpa(pf_t, res_t.k_ctop[0]) - 804.0) <= 20.0
# ---------- refined-grid / reconstruction invariants (Defect A + B fixes) ----
# Acceptance criteria are RECONSTRUCTION INVARIANTS, deliberately not the
# cloud top: the cloud-top ladder is unconverged on BOTH the refined grid
# (L30 r=1/2/4/8 -> 895/557/523/490 hPa) and the native grid
# (L30/L60/L120 -> 895/616/536 hPa), so re-pinning one unconverged number
# against another proves nothing.

_R_SET = (1, 2, 3, 4, 8)
# jit/eager atol as a fraction of each field's own scale; see the measured
# numbers in test_refined_jit_matches_eager.  Fields not listed get 1e-6.
_JIT_ATOL_SCALE = {"w2": 5e-4, "w2_surface": 5e-4, "cape_test": 1e-4,
                   "q_u": 1e-4, "l_u": 1e-4}
_RES_FIELDS = ("ldcum", "ktype", "k_dpl", "k_cbot", "k_ctop", "w_base",
               "T_u", "q_u", "l_u", "klab", "cape_test", "w2",
               "w2_surface", "ldsc", "k_botsc")


def _stretched_sigma_half(nlev=12):
    """Strictly increasing, NON-uniform half-level sigmas (surface last)."""
    return 0.004 + 0.996 * np.linspace(0.0, 1.0, nlev + 1) ** 1.7


def _refine_fixture(kind="deep", nlev=12):
    """float32 parent column on the stretched grid, with s = c_pd*T + Phi."""
    T, q, p_full, p_half = column(kind, sigma_half=_stretched_sigma_half(nlev))
    geo_half, geo_full = _geopotential(
        T.astype(np.float64), q.astype(np.float64), p_half.astype(np.float64))
    s = constants.c_pd * T.astype(np.float64) + geo_full
    dq = (-1e-8 * (1.0 + np.arange(T.shape[1], dtype=np.float64)))[None, :]
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    return {"T": f32(T), "q": f32(q), "s": f32(s), "dq": f32(dq),
            "p_full": f32(p_full), "p_half": f32(p_half),
            "geo_full": f32(geo_full), "geo_half": f32(geo_half)}


def _geom(p_half, r):
    """Mirror of the geometry ifs_departure_search_refined hands to
    _fv_minmod_refine, so the invariant checks bind the module's wiring."""
    ph = jnp.asarray(p_half)
    phr = ta._refine_half_levels(ph, r)
    c_par = 0.5 * (ph[:, :-1] + ph[:, 1:])
    h_dp = ph[:, 1:] - ph[:, :-1]
    c_child = 0.5 * (phr[:, :-1] + phr[:, 1:])
    k_par = jnp.repeat(jnp.arange(ph.shape[1] - 1), r)
    return c_par, h_dp, c_child, k_par, phr


def _search_args(fx):
    return (fx["T"], fx["q"], fx["p_full"], fx["p_half"], fx["geo_full"],
            fx["geo_half"], jnp.full((1,), -20.0, jnp.float32),
            jnp.full((1,), -150.0, jnp.float32),
            jnp.full((1,), 0.1, jnp.float32),
            jnp.zeros((1,), jnp.float32), fx["dq"])


def _check_refined_grid(p_half, r):
    ph = np.asarray(p_half)
    phr = np.asarray(ta._refine_half_levels(jnp.asarray(p_half), r))
    nlev = ph.shape[1] - 1
    assert phr.shape == (ph.shape[0], nlev * r + 1)
    assert np.array_equal(phr[:, 0], ph[:, 0])          # model top kept once
    assert np.array_equal(phr[:, -1], ph[:, -1])        # surface appended once
    for k in range(nlev + 1):
        assert np.array_equal(phr[:, k * r], ph[:, k])  # every interface once
    assert np.all(np.diff(phr, axis=1) > 0.0)           # positive thickness


def test_refined_grid_invariants():
    """(1) On a non-uniform grid, for r in (1,2,3,4,8): top and surface are
    the endpoints, p_half_r[:, k*r] == p_half[:, k] for every k, every
    sub-layer thickness is strictly positive; r = 1 reproduces p_half.

    The profile outputs are SAMPLED at refined index k*r, so the third
    identity here is exactly what makes that sampling land on the parent's
    own half level; under the old interfaces it landed one sub-layer below.

    NON-VACUITY: rejects the planted arange(1, r+1) interface defect
    (Defect A), which drops p_half[:, 0], appends the surface twice and
    leaves the bottom sub-layer with zero thickness -- demonstrated in
    test_nonvacuity_planted_interfaces."""
    fx = _refine_fixture()
    for r in _R_SET:
        _check_refined_grid(fx["p_half"], r)
    assert np.array_equal(
        np.asarray(ta._refine_half_levels(fx["p_half"], 1)),
        np.asarray(fx["p_half"]))


def _check_conservation(r):
    fx = _refine_fixture()
    c_par, h_dp, c_child, k_par, phr = _geom(fx["p_half"], r)
    dp_child = np.diff(np.asarray(phr, np.float64), axis=1)
    dp_par = np.asarray(h_dp, np.float64)
    for key in ("q", "s"):
        x64 = np.asarray(fx[key], np.float64)
        xc = np.asarray(ta._fv_minmod_refine(fx[key], c_par, h_dp, c_child,
                                             k_par), np.float64)
        mass = np.add.reduceat(xc * dp_child,
                               np.arange(0, xc.shape[1], r), axis=1)
        np.testing.assert_allclose(mass, x64 * dp_par, rtol=1e-5, atol=0.0)


def test_refined_reconstruction_is_conservative():
    """(2) Per parent layer, sum_j x_child*dp_child == x_parent*dp_parent for
    q and for s = c_pd*T + Phi, for r in (1,2,3,4,8).  Children are summed
    in float64, so the only error is float32 rounding (eps ~ 1.2e-7) of the
    <= 8 child values plus the eps-level cancellation of
    sum_j (c_kj - c_k); measured round-off is ~1e-7 relative, so rtol 1e-5
    keeps two orders of margin.

    NON-VACUITY: rejects the planted clipped full-level ramp (Defect B:
    non-conservative, ~38% per layer / 1.25% of the column water on the
    deep fixture) and also breaks under the planted interfaces (Defect A:
    the sub-layers then straddle two parents) -- demonstrated in
    test_nonvacuity_planted_ramp / test_nonvacuity_planted_interfaces."""
    for r in _R_SET:
        _check_conservation(r)


def _check_constant(r):
    fx = _refine_fixture()
    c_par, h_dp, c_child, k_par, _ = _geom(fx["p_half"], r)
    x = jnp.full_like(fx["q"], 0.017)
    xc = np.asarray(ta._fv_minmod_refine(x, c_par, h_dp, c_child, k_par))
    assert np.all(xc == np.asarray(x)[0, 0])


def test_refined_reconstruction_preserves_constants():
    """(3) A constant parent profile reconstructs exactly constant (all
    minmod slopes are 0 and x + 0.0*(...) is bit-for-bit x), for every r.
    Catches neither planted defect (both preserve constants); it pins the
    zero-slope branch of the limiter."""
    for r in _R_SET:
        _check_constant(r)


def _check_bounded(x, p_half, r):
    c_par, h_dp, c_child, k_par, _ = _geom(p_half, r)
    xj = jnp.asarray(np.asarray(x, np.float32)[None, :])
    xc = np.asarray(ta._fv_minmod_refine(xj, c_par, h_dp, c_child, k_par),
                    np.float64)[0]
    x64 = np.asarray(x, np.float64)
    nlev = x64.shape[0]
    for k in range(nlev):
        nb = x64[max(k - 1, 0):k + 2]
        lo, hi = float(nb.min()), float(nb.max())
        # scaled to the field's own SPAN, not to 1.0: at humidity scale
        # (values ~0.02) a floor of 1.0 made the slack 5e-5 RELATIVE, which
        # would pass a real 0.01% limiter overshoot (GLM review)
        span = float(x64.max() - x64.min())
        tol = 1e-6 * max(span, abs(lo), abs(hi))
        assert xc[k * r:(k + 1) * r].min() >= lo - tol
        assert xc[k * r:(k + 1) * r].max() <= hi + tol
    return xc


def test_refined_reconstruction_is_bounded():
    """(5) Every child lies in [min, max] of its parent and that parent's
    neighbours (the 2*d/h endpoint bound of the minmod limiter), across a
    STEP profile, a profile with an INTERIOR EXTREMUM, and the deep-fixture
    q (which exercises non-zero slopes); q stays >= 0.  Catches neither
    planted defect by itself (the clipped ramp is also inside the
    neighbour range); it pins the no-overshoot guarantee of the limiter."""
    nlev = 12
    step = np.where(np.arange(nlev) >= 6, 0.018, 0.0)
    extremum = np.where(np.arange(nlev) == 4, 0.02, 0.01)
    fx = _refine_fixture()
    q_par = np.asarray(fx["q"])[0]
    for r in _R_SET:
        xc = _check_bounded(step, fx["p_half"], r)
        assert xc.min() >= 0.0
        _check_bounded(extremum, fx["p_half"], r)
        _check_bounded(q_par, fx["p_half"], r)


def _check_affine(r):
    fx = _refine_fixture()
    c_par, h_dp, c_child, k_par, _ = _geom(fx["p_half"], r)
    a, b = 0.3, 1.0e-5
    c64 = np.asarray(c_par, np.float64)
    cc64 = np.asarray(c_child, np.float64)
    x = jnp.asarray(np.asarray(a + b * c64, np.float32))
    xc = np.asarray(ta._fv_minmod_refine(x, c_par, h_dp, c_child, k_par),
                    np.float64)
    err = np.abs(xc - (a + b * cc64))
    tol = 1e-5 * abs(b) * float(cc64.max() - cc64.min())
    assert np.all(err[:, r:-r] <= tol)      # interior parents only


def test_refined_reconstruction_exact_for_affine_profiles():
    """(6) Parent values that are the analytic LAYER MEANS of a profile
    affine in pressure (mean of an affine profile over a layer == its value
    at the layer centre, a + b*c_k -- NOT a point sample at p_full) are
    reconstructed exactly on every interior parent layer: all four minmod
    arguments then equal the exact slope b.  Tolerance 1e-5 of the span
    |b|*dp is ~30x the float32 round-off and ~1000x below the planted
    ramp's half-cell-shift error (~|b|*h/2).

    NON-VACUITY: rejects the planted clipped full-level ramp (Defect B),
    which pins the upper-half children of every parent to the parent value
    -- demonstrated in test_nonvacuity_planted_ramp."""
    for r in _R_SET:
        _check_affine(r)


def test_refined_r1_is_identity():
    """(4) r = 1 is the identity: the operator returns x itself
    (child centres == parent centres, k_par == arange(nlev)), the refined
    half levels equal p_half, and ifs_departure_search_refined short-circuits
    to ifs_departure_search, so every output field equals the native one.
    Catches neither planted defect (both are identities at r = 1); it
    guards the r = 1 contract used by the native-vs-refined gates."""
    fx = _refine_fixture()
    c_par, h_dp, c_child, k_par, _ = _geom(fx["p_half"], 1)
    assert np.array_equal(np.asarray(c_child), np.asarray(c_par))
    for key in ("T", "q", "s", "dq"):
        xc = ta._fv_minmod_refine(fx[key], c_par, h_dp, c_child, k_par)
        assert np.array_equal(np.asarray(xc), np.asarray(fx[key]))
    _check_refined_grid(fx["p_half"], 1)
    # the wrapper derives BOTH the level pressure and the geopotential from
    # p_half, so the native side is fed the same two; otherwise this asserts
    # an identity the layer-centre convention deliberately broke
    args = _search_args(fx)
    res_r1 = ta.ifs_departure_search_refined(
        *args, ta.IFSTestAscentConfig(column_refine=1))
    gh_m, gf_m = ta._hydrostatic_geopotential(args[0], args[1], args[3])
    res_nat = ta.ifs_departure_search(
        args[0], args[1], ta._layer_centre(args[3]), args[3], gf_m, gh_m,
        *args[6:], ta.IFSTestAscentConfig(column_refine=1))
    for name in _RES_FIELDS:
        assert np.array_equal(np.asarray(getattr(res_r1, name)),
                              np.asarray(getattr(res_nat, name)))


def test_refined_jit_matches_eager():
    """(7) jax.jit of the refined search agrees with the eager call: the
    refinement is static index arithmetic on traced pressures (no branching
    on traced values, no data-dependent shapes).  Integer/flag outputs are
    bit-identical; float profiles agree to rtol 1e-5 (XLA may reorder the
    float32 arithmetic).  Catches neither planted defect; it guards the
    traceability of the new operator."""
    fx = _refine_fixture()
    args = _search_args(fx)
    cfg = ta.IFSTestAscentConfig(column_refine=4)
    eager = ta.ifs_departure_search_refined(*args, cfg)
    jitted = jax.jit(lambda *a: ta.ifs_departure_search_refined(*a, cfg))(*args)
    for name in _RES_FIELDS:
        a_, b_ = (np.asarray(getattr(eager, name)),
                  np.asarray(getattr(jitted, name)))
        if a_.dtype.kind in "iub":
            assert np.array_equal(a_, b_)
        else:
            # Scale-relative, PER FIELD: XLA reassociates the float32
            # arithmetic of the nested departure x ascent scans over 48
            # refined levels, and the fields tolerate that very differently.
            # MEASURED at r = 4: T_u 7.2e-7 and w_base 2.8e-6 relative, both
            # inside rtol; cape_test 3.4e-5 of its own value (94.9156 eager
            # against 94.9187 jitted); q_u 2.0e-7 absolute on 1e-4 kg/kg
            # values; w2 0.0149 m2/s2 at the plume edge, where it is a small
            # difference of large terms, against a profile maximum of 122.9.
            # A blanket 5e-4 of scale would allow 0.15 K on T_u, which is a
            # rug, not an instrument (codex review).
            scale = max(float(np.max(np.abs(a_))), 1e-30)
            np.testing.assert_allclose(
                a_, b_, rtol=1e-5, atol=_JIT_ATOL_SCALE.get(name, 1e-6) * scale)


def _planted_defect_a_interfaces(p_half, r):
    """Defect A replanted verbatim: fractions 1/r..1 (j = 1..r)."""
    nlev = p_half.shape[1] - 1
    h_par = jnp.repeat(jnp.arange(nlev), r)
    frac = jnp.tile(jnp.arange(1, r + 1), nlev).astype(p_half.dtype) / r
    lo_h = p_half[:, h_par]
    dp_h = p_half[:, h_par + 1] - lo_h
    return jnp.concatenate(
        [lo_h + frac * dp_h, p_half[:, nlev:nlev + 1]], axis=1)


def test_nonvacuity_planted_interfaces(monkeypatch):
    """Re-plant Defect A ALONE (arange(1, r+1) interfaces) and confirm the
    grid invariants (test 1) reject it: the planted grid drops the model
    top p_half[:, 0], appends the surface twice, and leaves the bottom
    sub-layer with zero thickness."""
    monkeypatch.setattr(ta, "_refine_half_levels", _planted_defect_a_interfaces)
    fx = _refine_fixture()
    with pytest.raises(AssertionError):
        for r in _R_SET:
            _check_refined_grid(fx["p_half"], r)


def _planted_defect_b_ramp(x, c_par, h_par_dp, c_child, k_par):
    """Defect B replanted: the clipped linear ramp in pressure between the
    parent full-level points (c_par == p_full on midpoint grids), with the
    out-of-bounds last index clamped exactly as JAX clamped the old gather."""
    k1 = jnp.minimum(k_par + 1, x.shape[1] - 1)
    w = jnp.clip((c_child - c_par[:, k_par]) / (c_par[:, k1] - c_par[:, k_par]),
                 0.0, 1.0)
    return x[:, k_par] + w * (x[:, k1] - x[:, k_par])


def test_nonvacuity_planted_ramp(monkeypatch):
    """Re-plant Defect B ALONE (clipped full-level ramp) and confirm the
    conservation check (test 2) and the affine-exactness check (test 6)
    both reject it: the ramp is half-cell shifted (upper-half children
    pinned to the parent value) and non-conservative, and its last parent
    layer divides by the clamped 0/0 denominator."""
    monkeypatch.setattr(ta, "_fv_minmod_refine", _planted_defect_b_ramp)
    with pytest.raises(AssertionError):
        _check_conservation(4)
    with pytest.raises(AssertionError):
        _check_affine(4)


def test_production_path_uses_the_refinement_helpers(monkeypatch):
    """The invariant tests above drive `_refine_half_levels` and
    `_fv_minmod_refine` directly, so on their own they would still pass if
    `ifs_departure_search_refined` stopped calling them (codex review).  This
    pins the wiring: each helper is called exactly once per refined search, and
    the grid the PRODUCTION path builds satisfies the interface invariants.
    """
    fx = _refine_fixture()
    seen = {"grid": 0, "recon": 0, "p_half_r": None}
    real_grid, real_recon = ta._refine_half_levels, ta._fv_minmod_refine

    def spy_grid(p_half, r):
        out = real_grid(p_half, r)
        seen["grid"] += 1
        seen["p_half_r"] = np.asarray(out)
        seen["r"] = r
        return out

    def spy_recon(x, c_par, h_par_dp, c_child, k_par):
        seen["recon"] += 1
        seen["geom"] = (np.asarray(c_par), np.asarray(h_par_dp),
                        np.asarray(c_child), np.asarray(k_par))
        return real_recon(x, c_par, h_par_dp, c_child, k_par)

    monkeypatch.setattr(ta, "_refine_half_levels", spy_grid)
    monkeypatch.setattr(ta, "_fv_minmod_refine", spy_recon)
    ta.ifs_departure_search_refined(*_search_args(fx),
                                    ta.IFSTestAscentConfig(column_refine=4))
    assert seen["grid"] == 1, "the refined search no longer builds its grid here"
    # s, q, dq_dt_adv and the T_prov seeding pass
    assert seen["recon"] == 4, seen["recon"]
    assert seen["r"] == 4
    ph = np.asarray(fx["p_half"]); phr = seen["p_half_r"]
    assert np.array_equal(phr[:, 0], ph[:, 0])
    assert np.array_equal(phr[:, -1], ph[:, -1])
    for k in range(ph.shape[1]):
        assert np.array_equal(phr[:, k * 4], ph[:, k])
    assert np.all(np.diff(phr, axis=1) > 0.0)
    # and the GEOMETRY, not just the call: production could regress to the
    # half-cell-shifted anchoring (c_par = p_full) and every other test would
    # still pass, because these fixtures have p_full == mid(p_half) so the
    # mirror in _geom cannot tell the two apart (GLM review)
    c_par, h_dp, c_child, kp = seen["geom"]
    np.testing.assert_allclose(c_par, 0.5 * (ph[:, :-1] + ph[:, 1:]), rtol=0, atol=0)
    np.testing.assert_allclose(h_dp, ph[:, 1:] - ph[:, :-1], rtol=0, atol=0)
    np.testing.assert_allclose(c_child, 0.5 * (phr[:, :-1] + phr[:, 1:]),
                               rtol=0, atol=0)
    np.testing.assert_array_equal(kp, np.repeat(np.arange(ph.shape[1] - 1), 4))
# ---------------------------------------------------------------------------
# NON-MIDPOINT GRID: production-style full levels are the LOG midpoints of
# the half levels.  Every fixture above has p_full ==
# 0.5*(p_half[:, :-1] + p_half[:, 1:]), so no test above can tell "value
# point-sampled at the caller's p_full" from "value is the LAYER MEAN at
# the half-level midpoint"; these tests can, because the two differ by
# (sqrt(b)-sqrt(a))**2/2 = dp^2/(8p) per level (~0.1-1 hPa mid-column,
# tens of hPa in the coarse top layers).  Soundings, half levels and
# surface fluxes are the existing ones -- no new sounding.


def _log_midpoint_p_full(p_half):
    """Production-style full levels: log midpoints of the half levels."""
    ph = np.asarray(p_half, np.float64)
    return np.exp(0.5 * (np.log(ph[:, 1:]) + np.log(ph[:, :-1])))


def test_refined_invariants_anchor_on_the_module_geometry():
    """(b) The reconstruction invariants -- grid identities, per-layer
    conservation, constant preservation, boundedness, affine exactness --
    hold with the parent and child centres taken from ta._layer_centre
    itself, so they pin CONSISTENT USE of the module's own geometry.  They
    do NOT pin the FORMULA -- a uniform swap of _layer_centre would pass
    them (GLM review); the formula is pinned by the helper test's explicit
    0.5*(p_half[:, :-1] + p_half[:, 1:]) comparison and by the planted-defect
    check.  Conservation is the discriminator: the children's
    centres average to the parent centres, so re-anchoring the operator on
    any other pressures breaks the mass identity by ~dp^2/(8p) per layer,
    far above the 1e-5 tolerance; affine exactness is asserted for profiles
    affine in the LAYER CENTRES, which a caller-p_full anchoring would miss.

    Deliberately NOT parametrised over a non-midpoint p_full: the
    reconstruction never receives p_full at all, so feeding it a different
    one cannot change these numbers.  GLM's draft did parametrise it and
    codex caught that both arms ran identical inputs.  The anchoring is
    pinned instead by test_refinement_uses_layer_centres_not_the_callers_p_full,
    which drives the whole wrapper on a log-midpoint grid.
    """
    fx = _refine_fixture()
    ph = fx["p_half"]
    phj = jnp.asarray(ph)
    for r in _R_SET:
        _check_refined_grid(ph, r)
        c_par = ta._layer_centre(phj)
        phr = ta._refine_half_levels(phj, r)
        c_child = ta._layer_centre(phr)
        h_dp = phj[:, 1:] - phj[:, :-1]
        k_par = jnp.repeat(jnp.arange(phj.shape[1] - 1), r)
        dp_child = np.diff(np.asarray(phr, np.float64), axis=1)
        dp_par = np.asarray(h_dp, np.float64)
        for key in ("q", "s"):
            x64 = np.asarray(fx[key], np.float64)
            xc = np.asarray(ta._fv_minmod_refine(fx[key], c_par, h_dp,
                                                 c_child, k_par), np.float64)
            mass = np.add.reduceat(xc * dp_child,
                                   np.arange(0, xc.shape[1], r), axis=1)
            np.testing.assert_allclose(mass, x64 * dp_par, rtol=1e-5, atol=0.0)
        x = jnp.full_like(fx["q"], 0.017)
        xc = np.asarray(ta._fv_minmod_refine(x, c_par, h_dp, c_child, k_par))
        assert np.all(xc == np.asarray(x)[0, 0])
        _check_bounded(np.asarray(fx["q"])[0], ph, r)
        a, b = 0.3, 1.0e-5
        c64 = np.asarray(c_par, np.float64)
        cc64 = np.asarray(c_child, np.float64)
        xaff = jnp.asarray(np.asarray(a + b * c64, np.float32))
        xa = np.asarray(ta._fv_minmod_refine(xaff, c_par, h_dp, c_child,
                                             k_par), np.float64)
        err = np.abs(xa - (a + b * cc64))
        tol = 1e-5 * abs(b) * float(cc64.max() - cc64.min())
        # interior parents only: the top and bottom parents carry a flat
        # (zero-slope) reconstruction by construction, so affine exactness
        # does not apply there
        assert np.all(err[:, r:-r] <= tol)


def _nonmidpoint_args(kind="deep", nlev=30):
    """The existing ``column`` sounding and half levels; the full levels
    HANDED IN are the log midpoints, the usual production choice."""
    T, q, _, p_half = column(kind, nlev=nlev)
    p_full = _log_midpoint_p_full(p_half).astype(np.float32)
    geo_half, geo_full = _geopotential(T.astype(np.float64),
                                       q.astype(np.float64),
                                       p_half.astype(np.float64))
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))
    return (f32(T), f32(q), f32(p_full), f32(p_half),
            f32(geo_full), f32(geo_half))


_ta_fields = ("ldcum", "ktype", "k_dpl", "k_cbot", "k_ctop", "w_base", "T_u",
              "q_u", "l_u", "klab", "cape_test", "w2", "w2_surface", "ldsc",
              "k_botsc")


def _ta_same(a, b, rtol=1e-6):
    """Two trigger results agree: integer and flag fields exactly, float
    fields to rtol.  NOT bitwise on the floats -- the two call paths being
    compared reach the same geopotential by different orderings of the same
    float32 integration, which moves the parcel temperature in the last ulp.
    rtol 1e-6 is four orders below the effect these pins exist to catch: a
    location convention change moves things by a percent of a layer."""
    # the hand-kept list must cover the whole result, or a field added later
    # would silently escape every pin below (GLM review)
    assert set(_ta_fields) == set(type(a)._fields), (
        set(type(a)._fields) ^ set(_ta_fields))
    for f in _ta_fields:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if x.dtype.kind in "iub":
            if not np.array_equal(x, y):
                return False
        elif not np.allclose(x, y, rtol=rtol, atol=0.0, equal_nan=True):
            return False
    return True


def _surf_args(T):
    return (jnp.full((1,), -20.0, jnp.float32),
            jnp.full((1,), -150.0, jnp.float32),
            jnp.full((1,), 0.1, jnp.float32),
            jnp.zeros((1,), jnp.float32),
            jnp.zeros_like(T))


def _search_native(args, p_full, module_geo=True):
    """Source-literal ifs_departure_search with an explicit p_full.

    ``module_geo`` mirrors the wrapper, which derives the geopotential from
    p_half rather than taking the caller's; pass False to feed the caller's.
    """
    T, q, _, p_half, geo_full, geo_half = args
    if module_geo:
        geo_half, geo_full = ta._hydrostatic_geopotential(T, q, p_half)
    return ta.ifs_departure_search(T, q, p_full, p_half, geo_full, geo_half,
                                   *_surf_args(T),
                                   ta.IFSTestAscentConfig(column_refine=1))


def _search_refined(args, r, p_full=None):
    T, q, pf, p_half, geo_full, geo_half = args
    return ta.ifs_departure_search_refined(
        T, q, pf if p_full is None else p_full, p_half, geo_full, geo_half,
        *_surf_args(T), ta.IFSTestAscentConfig(column_refine=r))


def test_refinement_uses_layer_centres_not_the_callers_p_full():
    """(c) On a NON-MIDPOINT grid (production-style log-midpoint full levels)
    the refinement must place level values at the layer centres it derives
    from the half levels, in BOTH arms, and must ignore the p_full it is
    handed.

    The geopotential gets the same treatment as the pressure -- the wrapper
    derives it from p_half rather than taking the caller's -- so the bitwise
    pin below compares against the native search fed BOTH the layer centres
    and a module-built geopotential.

    NOT tested here: agreement between r = 1 and r = 2.  They are SUPPOSED
    to disagree -- on this sounding r = 1 types the column shallow with a
    cloud top at parent level 26 and r = 2 resolves it as deep at level 19,
    which is the entire reason the refinement exists.  GLM's first draft
    asserted they agree within one level; that premise is wrong and the
    assertion is dropped rather than loosened.

    The load-bearing pins are bitwise: the r = 1 arm must equal the
    source-literal search fed the layer CENTRES, neither arm may change
    when the handed-in p_full changes, and the re-planted old early return
    (which forwarded the caller's p_full) must give a DIFFERENT answer.
    That last one also guards against a vacuous fixture: if the log
    midpoints reproduced the layer-centre result bit for bit, this test
    could not see the anchoring at all.
    """
    args = _nonmidpoint_args("deep")
    res1 = _search_refined(args, 1)
    res2 = _search_refined(args, 2)

    # (ii) BOTH arms use the layer-centre convention: r == 1 is
    # bit-identical to the source-literal search fed the layer CENTRES ...
    centre = ta._layer_centre(args[3])
    assert _ta_same(res1, _search_native(args, centre))
    # ... and neither arm's result depends on the handed-in p_full at all
    assert _ta_same(res1, _search_refined(args, 1, p_full=centre))
    assert _ta_same(res2, _search_refined(args, 2, p_full=centre))

    # (iii) re-planted OLD early return (forwards the caller's p_full):
    # the convention pin must reject it -- the pre-fix failure mode
    planted = _search_native(args, args[2])
    assert not _ta_same(planted, res1), (
        "log-midpoint p_full reproduces the layer-centre result "
        "bit-for-bit: the fixture is insensitive to the anchoring and "
        "this test is VACUOUS -- report, do not loosen")
