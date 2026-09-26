"""CAM6 cloud fraction on the CLUBB path (cloud_scheme='cam6_clubb').

Oracle: CESM2.1 ``cam_cesm2_1_rel_60`` -- ``cldfrc2m.F90`` ``aist_vector``
(iceopt=5, the CAM6/CLUBB ice-stratus fraction), ``clubb_intr.F90``
deep-convective fraction + max/sum assembly, ``tropopause.F90`` analytic
tropopause.  The pins below are scalar Python transcriptions of the Fortran
control flow, compared at rel 1e-10 (x64); the wiring tests prove the MPAS
combined-physics path resolves the scheme with prognostic CLUBB and that the
radiation cloud call actually receives the CLUBB carry.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    cam6_deep_convective_fraction,
    cam6_ice_stratus_fraction,
    cam6_tropopause_analytic,
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import (
    saturation_mixing_ratio_goff,
    saturation_vapor_pressure_goff,
    saturation_vapor_pressure_ice_flatau,
)

jax.config.update("jax_enable_x64", True)

RTOL = 1e-10
NCOL, NLEV = 6, 8
STRADDLE_K = 3   # column 0's layer spanning 24000-30000 Pa (tP = 24548 Pa)


# ---------------------------------------------------------------------------
# Fortran transcriptions (scalar, one cell at a time)
# ---------------------------------------------------------------------------
def _aist_fortran(qv, T, p, qi, rhmini, rhmaxi):
    """cldfrc2m.F90 aist_vector, iceopt=5 branch (lines 844-895)."""
    qs = float(saturation_mixing_ratio_goff(jnp.asarray(T), jnp.asarray(p)))
    esl = float(saturation_vapor_pressure_goff(jnp.asarray(T)))
    esi = float(saturation_vapor_pressure_ice_flatau(jnp.asarray(T)))
    minice, mincld, qist_min, qist_max = 1.0e-12, 1.0e-4, 1.0e-7, 5.0e-3
    rhi = (qv + qi) / qs * (esl / esi)
    if rhmaxi == rhmini:
        rhdif = 1.0 if rhi > rhmini else 0.0
    else:
        rhdif = (rhi - rhmini) / (rhmaxi - rhmini)
    aist = min(1.0, max(rhdif, 0.0) ** 2)
    if qi < minice:
        aist = 0.0
    else:
        aist = max(mincld, aist)
    if qi >= minice:
        icimr = qi / aist
        if icimr < qist_min:
            aist = max(0.0, min(1.0, qi / qist_min))
        if icimr > qist_max:
            aist = max(0.0, min(1.0, qi / qist_max))
    return max(0.0, min(aist, 0.999))


def _deepcu_fortran(cmfmc_col, icwmr_col, dp1=0.1, dp2=500.0, dmax=0.6,
                    frac_limit=0.01, ic_limit=1.0e-12):
    """clubb_intr.F90:2492-2506 (cmfmc_sh == 0 under CLUBB)."""
    pver = len(icwmr_col)
    out = np.zeros(pver)
    for k in range(pver - 1):            # k = 1 .. pver-1 (1-based)
        d = max(0.0, min(dp1 * math.log(1.0 + dp2 * cmfmc_col[k + 1]), dmax))
        if d <= frac_limit or icwmr_col[k] < ic_limit:
            d = 0.0
        out[k] = d
    out[pver - 1] = 0.0                  # deepcu(:, pver) = 0
    return out


def _grid(seed=0):
    rng = np.random.default_rng(seed)
    p_half = np.sort(rng.uniform(2.0e3, 1.0e5, (NCOL, NLEV + 1)), axis=1)
    # column 0 (lat -80 deg, tP = 24548 Pa): layer 4 spans 24000-30000 Pa, so
    # its top face is above the tropopause and its midpoint below it
    p_half[0] = [2.0e3, 8.0e3, 1.5e4, 2.4e4, 3.0e4, 4.5e4, 6.0e4, 8.0e4, 1.0e5]
    p = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = 200.0 + 90.0 * (p / 1.0e5) + rng.uniform(-5.0, 5.0, (NCOL, NLEV))
    T[0, STRADDLE_K] = 220.0
    qs = np.asarray(saturation_mixing_ratio_goff(jnp.asarray(T), jnp.asarray(p)))
    # RH spanning below / inside / above the [0.80, 1.0] ramp
    q_v = qs * rng.uniform(0.5, 1.3, (NCOL, NLEV))
    q_i = rng.choice([0.0, 5.0e-13, 1.0e-9, 1.0e-6, 1.0e-4, 2.0e-2],
                     size=(NCOL, NLEV))
    # The straddling layer is made DISCRIMINATING: rhi = 0.9 with q_i in the
    # limiter-inactive band, so the stratospheric step (rhminis = rhmaxis =
    # 1) yields aist = mincld = 1e-4 while the tropospheric ramp would give
    # ((0.9-0.8)/0.2)^2 = 0.25.  A midpoint-based layer selection fails here.
    q_i[0, STRADDLE_K] = 2.0e-7
    esl = float(saturation_vapor_pressure_goff(jnp.asarray(T[0, STRADDLE_K])))
    esi = float(saturation_vapor_pressure_ice_flatau(jnp.asarray(T[0, STRADDLE_K])))
    q_v[0, STRADDLE_K] = 0.9 * qs[0, STRADDLE_K] * esi / esl - q_i[0, STRADDLE_K]
    lat = np.deg2rad(np.array([-80.0, -40.0, 0.0, 20.0, 60.0, 89.0]))
    return T, p, q_v, q_i, lat, p_half


def _troplev_fortran(p_half_col, tP):
    """tropopause.F90:398-403: scan k = pver..2 upward, the first layer whose
    TOP interface pint(k) <= tP is troplev (1-based); NOTFOUND -> 0."""
    pver = len(p_half_col) - 1
    for k in range(pver, 1, -1):          # Fortran k = pver .. 2
        if tP >= p_half_col[k - 1]:       # pint(k) is the top face of layer k
            return k
    return 0


def test_tropopause_analytic_pins_the_fortran_formula():
    lat = jnp.asarray(np.deg2rad([0.0, 45.0, 90.0]))
    np.testing.assert_allclose(
        np.asarray(cam6_tropopause_analytic(lat)),
        [10000.0, 17500.0, 25000.0], rtol=RTOL)


def test_aist_matches_fortran_transcription():
    T, p, q_v, q_i, lat, p_half = _grid()
    cfg = CloudConfig(scheme="cam6_clubb")
    got = np.asarray(cam6_ice_stratus_fraction(
        jnp.asarray(q_v), jnp.asarray(T), jnp.asarray(p), jnp.asarray(q_i),
        jnp.asarray(lat), cfg, jnp.asarray(p_half[:, :-1])))
    p_trop = 25000.0 - 15000.0 * np.cos(lat) ** 2
    n_strat = 0
    n_straddle = 0
    for i in range(NCOL):
        troplev = _troplev_fortran(p_half[i], p_trop[i])
        for k in range(NLEV):
            strat = (k + 1) <= troplev            # clubb_intr: k <= troplev
            n_strat += int(strat)
            # the straddling layer: top face above tP, midpoint below it --
            # CAM counts it as stratosphere, a midpoint test would not
            n_straddle += int(strat and p[i, k] > p_trop[i])
            rhmini = cfg.cam6_rhminis if strat else cfg.cam6_rhmini
            rhmaxi = cfg.cam6_rhmaxis if strat else cfg.cam6_rhmaxi
            want = _aist_fortran(q_v[i, k], T[i, k], p[i, k], q_i[i, k],
                                 rhmini, rhmaxi)
            assert math.isclose(got[i, k], want, rel_tol=RTOL, abs_tol=1e-300), (
                i, k, got[i, k], want)
    # Non-vacuity: both regimes, the straddling layer and every limiter
    # branch are exercised.
    assert 0 < n_strat < NCOL * NLEV
    assert n_straddle > 0
    # The straddling layer discriminates the interface rule from a midpoint
    # rule: CAM (top face above tP) -> stratospheric step -> mincld; the
    # tropospheric ramp would give 0.25.
    assert p[0, STRADDLE_K] > p_trop[0] > p_half[0, STRADDLE_K]
    assert math.isclose(got[0, STRADDLE_K], 1.0e-4, rel_tol=RTOL)
    trop_alt = _aist_fortran(q_v[0, STRADDLE_K], T[0, STRADDLE_K],
                             p[0, STRADDLE_K], q_i[0, STRADDLE_K],
                             cfg.cam6_rhmini, cfg.cam6_rhmaxi)
    assert math.isclose(trop_alt, 0.25, rel_tol=1e-6)
    assert (got == 0.0).any() and (got == 0.999).any()
    assert ((got > 0.0) & (got < 0.999)).any()


def test_aist_step_when_ramp_is_degenerate():
    """rhmaxi == rhmini => rhdif is a step at rhmini (Fortran :847-852)."""
    T = jnp.full((1, 2), 230.0)
    p = jnp.full((1, 2), 3.0e4)
    # q_i chosen so the in-cloud ice limiters (qist_min/qist_max) stay
    # inactive: icimr = 2e-7/1e-4 = 2e-3 and 2e-7/0.999, both inside.
    q_i = jnp.full((1, 2), 2.0e-7)
    qs = saturation_mixing_ratio_goff(T, p)
    esl = saturation_vapor_pressure_goff(T)
    esi = saturation_vapor_pressure_ice_flatau(T)
    # rhi just below / above 0.9
    q_v = jnp.asarray([[0.89, 0.91]]) * qs * esi / esl - q_i
    cfg = CloudConfig(scheme="cam6_clubb", cam6_rhmini=0.9, cam6_rhmaxi=0.9)
    lat = jnp.zeros((1,))
    got = np.asarray(cam6_ice_stratus_fraction(
        q_v, T, p, q_i, lat, cfg, jnp.full((1, 2), 2.5e4)))
    np.testing.assert_allclose(got, [[1.0e-4, 0.999]], rtol=RTOL)


def test_deepcu_matches_fortran_transcription():
    rng = np.random.default_rng(1)
    mf = rng.uniform(0.0, 0.05, (NCOL, NLEV + 1))
    mf[:, 0] = 0.0
    mf[0, :] = 0.0                                   # no convection column
    mf[1, 3] = 1.0e-5                                 # below frac_limit
    mf[3, 2:5] = 1.0                                  # hits the 0.6 cap
    icwmr = rng.uniform(1.0e-4, 1.0e-3, (NCOL, NLEV))
    icwmr[2, :4] = 0.0                                # below ic_limit
    cfg = CloudConfig(scheme="cam6_clubb")
    got = np.asarray(cam6_deep_convective_fraction(
        jnp.asarray(mf), jnp.asarray(icwmr), cfg))
    want = np.stack([_deepcu_fortran(mf[i], icwmr[i]) for i in range(NCOL)])
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
    assert (got == 0.6).any() and ((got > 0) & (got < 0.6)).any()
    assert (got[:, -1] == 0.0).all()


def test_deepcu_negative_mass_flux_is_finite_zero():
    cfg = CloudConfig(scheme="cam6_clubb")
    got = cam6_deep_convective_fraction(
        jnp.full((1, 4), -1.0), jnp.full((1, 3), 1.0e-3), cfg)
    assert np.all(np.asarray(got) == 0.0)


def _column_inputs():
    T, p, q_v, q_i, lat, p_half = _grid(2)
    dp = p_half[:, 1:] - p_half[:, :-1]
    return (jnp.asarray(T), jnp.asarray(p), jnp.asarray(q_v), jnp.asarray(dp),
            jnp.asarray(q_i), jnp.asarray(lat), jnp.asarray(p_half))


def test_assembly_is_max_alst_aist_plus_deepcu_capped():
    """clubb_intr.F90:2575 ast = max(alst, aist); :2586 min(ast+deepcu, 1)."""
    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    cfg = CloudConfig(scheme="cam6_clubb")
    alst = jnp.asarray(np.random.default_rng(3).uniform(0, 1, T.shape))
    mf = jnp.asarray(np.random.default_rng(4).uniform(0, 0.05, (NCOL, NLEV + 1)))
    icwmr = jnp.full(T.shape, 1.0e-3)
    q_c = jnp.full(T.shape, 1.0e-5)
    out = compute_cloud_properties(
        T, p, q_v, dp, cfg, q_cloud=q_c, q_ice=q_i,
        cloud_fraction_override=alst, lat=lat, p_half=p_half,
        conv_mass_flux_up=mf, conv_icwmr=icwmr)
    aist = cam6_ice_stratus_fraction(q_v, T, p, q_i, lat, cfg, p_half[:, :-1])
    deepcu = cam6_deep_convective_fraction(mf, icwmr, cfg)
    want = jnp.minimum(jnp.maximum(alst, aist) + deepcu, 1.0)
    np.testing.assert_allclose(np.asarray(out.cloud_fraction), np.asarray(want),
                               rtol=RTOL)
    # deepcu really contributed (non-vacuous) and the cap bit somewhere
    assert (np.asarray(want) > np.asarray(jnp.maximum(alst, aist))).any()
    # No deep-convection carries => deepcu exactly 0
    out0 = compute_cloud_properties(
        T, p, q_v, dp, cfg, q_cloud=q_c, q_ice=q_i,
        cloud_fraction_override=alst, lat=lat, p_half=p_half)
    np.testing.assert_allclose(np.asarray(out0.cloud_fraction),
                               np.asarray(jnp.maximum(alst, aist)), rtol=RTOL)
    # CAM6 has no diagnostic condensate floor: radiative condensate is the
    # prognostic one.
    np.testing.assert_allclose(np.asarray(out.lwp),
                               np.asarray(q_c * dp) / constants.g, rtol=1e-3)


def test_rh_blend_floor_gate_knobs_are_bypassed_for_cam6():
    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    alst = jnp.full(T.shape, 0.3)
    q_c = jnp.full(T.shape, 1.0e-5)
    base = CloudConfig(scheme="cam6_clubb")
    knobs = base._replace(clubb_cf_override_strength=0.5,
                          clubb_cf_override_floor=0.2,
                          clubb_cf_override_p_min_pa=7.0e4,
                          clubb_cf_override_ramp_pa=1.0e4)
    kw = dict(q_cloud=q_c, q_ice=q_i, cloud_fraction_override=alst, lat=lat,
              p_half=p_half)
    a = compute_cloud_properties(T, p, q_v, dp, base, **kw).cloud_fraction
    b = compute_cloud_properties(T, p, q_v, dp, knobs, **kw).cloud_fraction
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Non-vacuity: the same knobs DO change a sundqvist override
    s0 = CloudConfig(scheme="sundqvist")
    s1 = s0._replace(clubb_cf_override_strength=0.5)
    kw_s = dict(q_cloud=q_c, q_ice=q_i, cloud_fraction_override=alst)
    assert not np.array_equal(
        np.asarray(compute_cloud_properties(T, p, q_v, dp, s0, **kw_s).cloud_fraction),
        np.asarray(compute_cloud_properties(T, p, q_v, dp, s1, **kw_s).cloud_fraction))


@pytest.mark.parametrize("drop, match", [
    ("cloud_fraction_override", "requires the CLUBB PDF cloud fraction"),
    ("lat", "needs lat"),
    ("p_half", "needs lat"),
    ("q_cloud", "requires explicit q_cloud/q_ice"),
])
def test_missing_inputs_raise(drop, match):
    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    kw = dict(q_cloud=jnp.full(T.shape, 1.0e-5),
              cloud_fraction_override=jnp.full(T.shape, 0.3), lat=lat,
              p_half=p_half)
    kw[drop] = None
    with pytest.raises(ValueError, match=match):
        compute_cloud_properties(T, p, q_v, dp, CloudConfig(scheme="cam6_clubb"),
                                 **kw)


def test_convective_cloud_overlay_refused_and_half_carries_refused():
    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    kw = dict(q_cloud=jnp.full(T.shape, 1.0e-5),
              cloud_fraction_override=jnp.full(T.shape, 0.3), lat=lat,
              p_half=p_half)
    with pytest.raises(ValueError, match="convective_cloud"):
        compute_cloud_properties(
            T, p, q_v, dp, CloudConfig(scheme="cam6_clubb", convective_cloud=True),
            conv_precip=jnp.zeros((NCOL,)), **kw)
    with pytest.raises(ValueError, match="BOTH conv_mass_flux_up"):
        compute_cloud_properties(
            T, p, q_v, dp, CloudConfig(scheme="cam6_clubb"),
            conv_mass_flux_up=jnp.zeros((NCOL, NLEV + 1)), **kw)


def test_cam6_cloud_fraction_is_differentiable():
    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    cfg = CloudConfig(scheme="cam6_clubb")

    def f(qv, alst):
        return jnp.sum(compute_cloud_properties(
            T, p, qv, dp, cfg, q_cloud=jnp.full(T.shape, 1.0e-5), q_ice=q_i,
            cloud_fraction_override=alst, lat=lat, p_half=p_half).cloud_fraction)
    g_qv, g_alst = jax.grad(f, argnums=(0, 1))(q_v, jnp.full(T.shape, 0.3))
    assert np.all(np.isfinite(np.asarray(g_qv)))
    assert np.all(np.isfinite(np.asarray(g_alst)))
    assert (np.asarray(g_alst) != 0).any()


# ---------------------------------------------------------------------------
# Wiring: driver config, MPAS combined physics, radiation reads the carry
# ---------------------------------------------------------------------------
def _exp_cfg(**over):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    base = dict(cloud_scheme="cam6_clubb", turbulence="clubb",
                use_clubb_cloud_fraction=True, microphysics="sundqvist",
                radiation="rrtmgp",
                dycore=DycoreConfig(discretization="mpas"))
    base.update(over)
    return ExperimentConfig(**base)


def test_validate_strict_accepts_the_full_cam6_selection():
    _exp_cfg().validate_strict()


@pytest.mark.parametrize("over, match", [
    (dict(turbulence="louis"), "requires turbulence='clubb'"),
    (dict(use_clubb_cloud_fraction=False), "use_clubb_cloud_fraction=True"),
    (dict(convective_cloud=True), "convective_cloud=True is not part of CAM6"),
    (dict(microphysics="none"), "requires a microphysics scheme"),
    (dict(radiation="gray"), "needs a cloud-path radiation"),
])
def test_validate_strict_refuses_partial_cam6_selections(over, match):
    with pytest.raises(ValueError, match=match):
        _exp_cfg(**over).validate_strict()


def test_validate_strict_refuses_cam6_off_the_mpas_lane():
    from legoesm.driver.config import DycoreConfig
    with pytest.raises(ValueError, match="MPAS lane only"):
        _exp_cfg(dycore=DycoreConfig(discretization="cdgrid")).validate_strict()


def test_use_clubb_cloud_fraction_now_legal_on_mpas():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    ExperimentConfig(turbulence="clubb", use_clubb_cloud_fraction=True,
                     dycore=DycoreConfig(discretization="mpas")).validate_strict()
    with pytest.raises(ValueError, match="requires turbulence='clubb'"):
        ExperimentConfig(turbulence="louis", use_clubb_cloud_fraction=True,
                         dycore=DycoreConfig(discretization="mpas")
                         ).validate_strict()
    with pytest.raises(ValueError, match="not wired into the 'spectral'"):
        ExperimentConfig(turbulence="clubb", use_clubb_cloud_fraction=True,
                         dycore=DycoreConfig(discretization="spectral")
                         ).validate_strict()


def _rad_cfg(use_clubb_cf=True, scheme="cam6_clubb", rad="gray",
             nested=True):
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, RRTMGPConfig,
    )
    return RadiationConfig(scheme=rad, cloud_scheme=scheme,
                           cloud_config=CloudConfig(scheme=scheme) if nested else None,
                           use_clubb_cloud_fraction=use_clubb_cf,
                           rrtmgp=RRTMGPConfig(include_clouds=True))


class _Captured(Exception):
    """Raised by the cloud-call spy once it has recorded its kwargs, so the
    chain physics_fn -> radiation_column -> compute_cloud_properties is
    exercised without building an RRTMGP solver (gray radiation returns
    before the cloud call, so it cannot carry this test)."""


def test_mpas_combined_physics_resolves_cam6_with_prognostic_clubb():
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    fn = make_physics(PhysicsConfig(
        radiation=_rad_cfg(),
        turbulence=TurbulenceConfig(scheme="clubb",
                                    clubb=CLUBBConfig(prognostic=True)),
    ), model_type="mpas")
    assert callable(fn)


def test_radiation_factory_mpas_accepts_flag_and_refuses_cam6_without_it():
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )
    fn = make_radiation_physics(_rad_cfg(), model_type="mpas",
                                use_clubb_cloud_fraction=True)
    assert fn._wants_phys_state_ro is True
    for nested in (True, False):
        with pytest.raises(ValueError, match="use_clubb_cloud_fraction=True"):
            make_radiation_physics(
                _rad_cfg(use_clubb_cf=False, nested=nested), model_type="mpas",
                use_clubb_cloud_fraction=False)
    with pytest.raises(NotImplementedError, match="hydrostatic"):
        make_radiation_physics(_rad_cfg(), model_type="nonhydrostatic",
                               use_clubb_cloud_fraction=True)


def test_radiation_cloud_call_receives_the_clubb_and_deepcu_carries(monkeypatch):
    """End to end on a real column state: the radiation physics_fn reads
    ``phys_state.cloud_fraction`` / ``conv_mass_flux_up`` / ``conv_icwmr``
    and hands them (plus lat) to the cloud call.  Non-vacuous: the seeded
    carry values are asserted, and dropping the flag leaves the override
    None."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation import integration as radint
    from tests.unit.test_physics_smoke import make_hydrostatic_setup

    state, grid, sigma = make_hydrostatic_setup(n=2, nlev=NLEV)
    ncol = 6 * 2 * 2
    ps = init_physics_state(ncol, NLEV, PhysicsConfig())
    ps = ps._replace(
        cloud_fraction=jnp.full((ncol, NLEV), 0.37),
        conv_mass_flux_up=jnp.full((ncol, NLEV + 1), 0.02),
        conv_icwmr=jnp.full((ncol, NLEV), 2.0e-4),
    )
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        raise _Captured()
    monkeypatch.setattr(radint, "compute_cloud_properties", spy)

    for nested in (True, False):
        seen.clear()
        fn = radint.make_radiation_physics(
            _rad_cfg(rad="rrtmgp", nested=nested), model_type="mpas",
            use_clubb_cloud_fraction=True)
        with pytest.raises(_Captured):
            fn(state, grid, sigma, phys_state=ps)
        assert np.allclose(np.asarray(seen["cloud_fraction_override"]), 0.37)
        assert np.allclose(np.asarray(seen["conv_mass_flux_up"]), 0.02)
        assert np.allclose(np.asarray(seen["conv_icwmr"]), 2.0e-4)
        assert seen["lat"].shape == (ncol,)
        assert seen["p_half"].shape == (ncol, NLEV + 1)

    # The real MPAS chain: combined.make_physics forwards phys_state to the
    # radiation slot (radiation runs first, so the spy fires before CLUBB).
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    seen.clear()
    combined_fn = make_physics(PhysicsConfig(
        radiation=_rad_cfg(rad="rrtmgp"),
        turbulence=TurbulenceConfig(scheme="clubb",
                                    clubb=CLUBBConfig(prognostic=True)),
    ), model_type="mpas")
    with pytest.raises(_Captured):
        combined_fn(state, grid, sigma, phys_state=ps)
    assert np.allclose(np.asarray(seen["cloud_fraction_override"]), 0.37)
    assert np.allclose(np.asarray(seen["conv_mass_flux_up"]), 0.02)

    seen.clear()
    fn_off = radint.make_radiation_physics(
        _rad_cfg(use_clubb_cf=False, scheme="sundqvist", rad="rrtmgp"),
        model_type="mpas", use_clubb_cloud_fraction=False)
    with pytest.raises(_Captured):
        fn_off(state, grid, sigma)
    assert seen["cloud_fraction_override"] is None
    assert seen["conv_mass_flux_up"] is None


def test_physics_state_carries_deepcu_inputs_forward():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import (
        init_physics_state, update_physics_state,
    )
    ps = init_physics_state(3, 4, PhysicsConfig())
    assert ps.conv_mass_flux_up.shape == (3, 5) and ps.conv_icwmr.shape == (3, 4)
    seeded = ps._replace(conv_mass_flux_up=jnp.ones((3, 5)))
    assert np.all(np.asarray(update_physics_state(seeded, {}).conv_mass_flux_up) == 1.0)
    out = update_physics_state(seeded, {"conv_mass_flux_up": jnp.zeros((3, 5)),
                                        "conv_icwmr": jnp.ones((3, 4))})
    assert np.all(np.asarray(out.conv_mass_flux_up) == 0.0)
    assert np.all(np.asarray(out.conv_icwmr) == 1.0)


def test_convection_bridge_publishes_mass_flux_when_scheme_exposes_it(monkeypatch):
    """A ConvectionOutput carrying ``mass_flux_up``/``icwmr`` reaches the
    carry dict under the names radiation reads."""
    from legoesm.atmosphere.physics.convection import integration as convint
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    from tests.unit.test_physics_smoke import make_hydrostatic_setup

    state, grid, sigma = make_hydrostatic_setup(n=2, nlev=NLEV)
    ncol = 6 * 2 * 2

    def fake_scheme(*, T, q_v, p_full, p_half, dt, config):
        z = jnp.zeros((ncol, NLEV))
        return ConvectionOutput(
            dT_dt=z, dq_v_dt=z, dq_c_conv_dt=z, cape=jnp.zeros((ncol,)),
            convective_mask=jnp.zeros((ncol,)),
            mass_flux_up=jnp.full((ncol, NLEV + 1), 0.01),
            icwmr=jnp.full((ncol, NLEV), 3.0e-4))
    # 'dca' takes the plain (non-prognostic) bridge branch.
    monkeypatch.setattr(convint, "_get_convection_fn",
                        lambda cfg: ("dca", fake_scheme, cfg.dca))
    fn = convint.make_convection_physics(ConvectionConfig(scheme="dca"),
                                         "mpas", 600.0)
    _tend, prog = fn(state, grid, sigma)
    assert isinstance(prog, dict)
    assert np.allclose(np.asarray(prog["conv_mass_flux_up"]), 0.01)
    assert prog["conv_mass_flux_up"].shape == (ncol, NLEV + 1)
    assert np.allclose(np.asarray(prog["conv_icwmr"]), 3.0e-4)


def test_cmor_cloud_diagnostics_use_the_carry_or_refuse():
    from legoesm.driver.diagnostics import DiagnosticCollector
    dc = DiagnosticCollector.__new__(DiagnosticCollector)
    from legoesm.grids.vertical import create_sigma_coordinate
    dc._cloud_config = CloudConfig(scheme="cam6_clubb")
    dc.vcoord = create_sigma_coordinate(3)
    p_s = np.full(2, 1.0e5)
    with pytest.raises(ValueError, match="cam6_clubb"):
        dc._cam6_cloud_kwargs(None, None, None, None, p_s, 2, 3)
    kw = dc._cam6_cloud_kwargs(np.zeros((2, 3)), np.zeros((2, 4)),
                               np.zeros((2, 3)), np.zeros(2), p_s, 2, 3)
    assert set(kw) == {"cloud_fraction_override", "lat", "conv_mass_flux_up",
                       "conv_icwmr", "p_half"}
    assert kw["p_half"].shape == (2, 4)
    dc._cloud_config = CloudConfig(scheme="sundqvist")
    assert dc._cam6_cloud_kwargs(None, None, None, None, p_s, 2, 3) == {}
