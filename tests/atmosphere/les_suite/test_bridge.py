"""Unit tests for the LES→SCM bridge (les_suite/bridge.py)."""
from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.atmosphere.les_suite.bridge import (
    BridgeError,
    LESReferenceArtifact,
    LESTruth,
    artifact_to_scm_forcing,
    diagnostic_truth,
    load_artifact,
    prognostic_truth,
    save_artifact,
    surface_roughness,
    total_turbulent_flux,
)

NZ = 8
NT = 4


def _heights(nz: int = NZ):
    return jnp.linspace(10.0, 1600.0, nz)


def _times(nt: int = NT):
    return jnp.linspace(0.0, 3600.0, nt)


def _profile_series(value: float, nt: int = NT, nz: int = NZ):
    return jnp.full((nt, nz), value)


def _dry_artifact(**overrides) -> LESReferenceArtifact:
    base = dict(
        case_name="cbl_test",
        sgs="lasd",
        heights_m=_heights(),
        times_s=_times(),
        theta=_profile_series(300.0),
        u=_profile_series(1.0),
        v=_profile_series(0.0),
        wtheta_resolved=_profile_series(0.05),
        wtheta_sgs=_profile_series(0.01),
        subsidence_w=jnp.full((NZ,), -0.01),
        theta_adv=jnp.zeros((NZ,)),
        prescribe="fluxes",
        w_theta_s=jnp.full((NT,), 0.06),
        f_c=1.0e-4,
    )
    base.update(overrides)
    return LESReferenceArtifact(**base)


def _moist_artifact(**overrides) -> LESReferenceArtifact:
    base = dict(
        case_name="bomex_test",
        sgs="lasd",
        heights_m=_heights(),
        times_s=_times(),
        theta=_profile_series(300.0),
        u=_profile_series(-8.0),
        v=_profile_series(0.0),
        wtheta_resolved=_profile_series(0.02),
        wtheta_sgs=_profile_series(0.005),
        qt=_profile_series(0.016),
        wqt_resolved=_profile_series(1.0e-4),
        wqt_sgs=_profile_series(2.0e-5),
        u_geo=jnp.full((NZ,), -8.75),
        subsidence_w=jnp.full((NZ,), -0.005),
        prescribe="fluxes",
        w_theta_s=jnp.full((NT,), 0.02),
        w_qv_s=jnp.full((NT,), 5.0e-5),
        f_c=3.76e-5,
        les_setup={"subsidence_species": ["q_v"]},
    )
    base.update(overrides)
    return LESReferenceArtifact(**base)


# --- construction / validation ------------------------------------------------
def test_dry_artifact_constructs():
    art = _dry_artifact()
    assert art.nz == NZ and art.nt == NT
    assert not art.is_moist


def test_moist_artifact_is_moist():
    assert _moist_artifact().is_moist


def test_empty_case_name_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(case_name="")


def test_bad_prescribe_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="bogus")


def test_qc_channel_validation():
    # A valid recorded cloud-water channel (0 <= qc <= qt) is accepted.
    assert _moist_artifact(qc=_profile_series(0.001)).is_moist
    # qc exceeding total water would seed a negative q_v at SCM init — rejected.
    with pytest.raises(BridgeError, match="qc must not exceed qt"):
        _moist_artifact(qc=_profile_series(0.02))       # qt is 0.016
    # negative cloud water is unphysical — rejected.
    with pytest.raises(BridgeError, match="qc must be non-negative"):
        _moist_artifact(qc=_profile_series(-0.001))
    # a dry artifact must not carry cloud water.
    with pytest.raises(BridgeError):
        _dry_artifact(qc=_profile_series(0.001))


def test_non_monotone_heights_rejected():
    bad = jnp.asarray([100.0, 50.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    with pytest.raises(BridgeError):
        _dry_artifact(heights_m=bad)


def test_profile_shape_mismatch_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(theta=_profile_series(300.0, nz=NZ + 1))


def test_forcing_profile_shape_mismatch_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(subsidence_w=jnp.zeros((NZ + 2,)))


def test_surface_series_shape_mismatch_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(w_theta_s=jnp.zeros((NT + 1,)))


def test_moist_without_flux_rejected():
    with pytest.raises(BridgeError):
        _moist_artifact(wqt_resolved=None)


def test_prescribe_fluxes_without_surface_flux_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="fluxes", w_theta_s=None)


def test_prescribe_ts_requires_ts():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="T_s", w_theta_s=None, T_s=None)


def test_dry_artifact_with_moisture_field_rejected():
    # dry (qt=None) must carry NO moisture field, else it is silently discarded
    with pytest.raises(BridgeError):
        _dry_artifact(wqt_resolved=_profile_series(1.0e-4))


def test_dry_artifact_with_qv_flux_rejected():
    with pytest.raises(BridgeError):
        _dry_artifact(w_qv_s=jnp.full((NT,), 1.0e-5))


@pytest.mark.parametrize("name,value", [
    ("precip_s", jnp.zeros((NT,))), ("precip_s_cum", jnp.zeros((NT,))),
    ("w_qv_s_cum", jnp.zeros((NT,))), ("qr", jnp.zeros((NT, NZ))),
    ("cloud_frac", jnp.zeros((NT, NZ)))])
def test_dry_artifact_with_moist_channel_rejected(name, value):
    with pytest.raises(BridgeError):
        _dry_artifact(**{name: value})


def test_moist_series_need_their_instantaneous_series_and_shapes():
    with pytest.raises(BridgeError):                    # integral without the series
        _moist_artifact(w_qv_s=None, w_qv_s_cum=jnp.zeros((NT,)))
    with pytest.raises(BridgeError):
        _moist_artifact(precip_s_cum=jnp.zeros((NT,)))
    with pytest.raises(BridgeError):                    # wrong shape
        _moist_artifact(cloud_frac=jnp.zeros((NT,)))


def test_moist_channels_round_trip(tmp_path):
    art = _moist_artifact(
        qr=_profile_series(1e-5), cloud_frac=_profile_series(0.1),
        precip_s=jnp.full((NT,), 1e-6), precip_s_cum=jnp.linspace(0.0, 3e-3, NT),
        w_qv_s_cum=jnp.linspace(0.0, 0.1, NT))
    p = tmp_path / "moist.npz"
    save_artifact(art, str(p))
    loaded = load_artifact(str(p))
    for name in ("qr", "cloud_frac", "precip_s", "precip_s_cum", "w_qv_s_cum"):
        assert jnp.allclose(getattr(loaded, name), getattr(art, name)), name


def test_forcing_moisture_flux_uses_recorded_integral():
    from legoesm.atmosphere.les_suite.bridge import surface_flux_fn
    t = _times()
    w = jnp.linspace(1e-5, 4e-5, NT)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w[:-1] * jnp.diff(t))])
    art = _moist_artifact(w_qv_s=w, w_qv_s_cum=cum)
    f = artifact_to_scm_forcing(art)
    mid = 0.5 * (t[1] + t[2])
    assert float(f.w_qv_s(mid)) == pytest.approx(float(w[1]))   # interval mean
    assert float(surface_flux_fn(art, "w_qv_s")(mid)) == pytest.approx(float(w[1]))
    with pytest.raises(BridgeError):
        surface_flux_fn(art, "tau_x_s")


def test_moist_wqt_sgs_without_resolved_rejected():
    with pytest.raises(BridgeError):
        _moist_artifact(wqt_resolved=None, wqt_sgs=_profile_series(1e-5))


def test_prescribe_fluxes_forbids_ts():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="fluxes", T_s=jnp.full((NT,), 300.0))


def test_prescribe_ts_forbids_surface_flux():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="T_s", T_s=jnp.full((NT,), 300.0),
                      w_theta_s=jnp.full((NT,), 0.06))


def test_prescribe_none_forbids_surface_fields():
    with pytest.raises(BridgeError):
        _dry_artifact(prescribe="none", w_theta_s=jnp.full((NT,), 0.06))


# --- total turbulent flux -----------------------------------------------------
def test_total_flux_sums_resolved_and_sgs():
    res = jnp.array([[1.0, 2.0]])
    sgs = jnp.array([[0.1, 0.2]])
    tot = total_turbulent_flux(res, sgs)
    assert jnp.allclose(tot, jnp.array([[1.1, 2.2]]))


def test_total_flux_none_sgs_returns_resolved():
    res = jnp.array([[1.0, 2.0]])
    assert jnp.allclose(total_turbulent_flux(res, None), res)


# --- diagnostic / prognostic truth --------------------------------------------
def test_diagnostic_truth_last_snapshot_by_default():
    art = _dry_artifact()
    truth = diagnostic_truth(art)
    assert isinstance(truth, LESTruth)
    assert truth.theta.shape == (NZ,)
    # total flux = resolved (0.05) + sgs (0.01)
    assert jnp.allclose(truth.wtheta, 0.06)
    assert truth.qt is None


def test_diagnostic_truth_selects_nearest_time():
    art = _dry_artifact()
    truth = diagnostic_truth(art, time_s=0.0)
    assert float(truth.times_s[0]) == pytest.approx(0.0)


def test_diagnostic_truth_moist_has_wqt():
    truth = diagnostic_truth(_moist_artifact())
    assert truth.qt is not None
    assert jnp.allclose(truth.wqt, 1.0e-4 + 2.0e-5)


def test_prognostic_truth_full_series():
    truth = prognostic_truth(_dry_artifact())
    assert truth.theta.shape == (NT, NZ)
    assert truth.wtheta.shape == (NT, NZ)
    assert jnp.allclose(truth.wtheta, 0.06)


def test_prognostic_truth_trims_before_t0():
    art = _dry_artifact()
    truth = prognostic_truth(art, t0_s=1200.0)
    # times are [0, 1200, 2400, 3600]; t0=1200 keeps 3 samples
    assert truth.theta.shape[0] == 3
    assert float(truth.times_s[0]) == pytest.approx(1200.0)


def test_prognostic_truth_t0_between_samples_keeps_at_or_after():
    # t0 between samples must keep the FIRST sample at-or-after (not the nearest).
    art = _dry_artifact()  # times [0, 1200, 2400, 3600]
    truth = prognostic_truth(art, t0_s=600.0)  # between 0 and 1200
    assert float(truth.times_s[0]) == pytest.approx(1200.0)
    assert truth.theta.shape[0] == 3


def test_prognostic_truth_t0_after_last_raises():
    art = _dry_artifact()
    with pytest.raises(BridgeError):
        prognostic_truth(art, t0_s=1.0e9)


# --- forcing reconstruction ---------------------------------------------------
def test_forcing_reconstructs_channels():
    art = _dry_artifact()
    forcing = artifact_to_scm_forcing(art)
    assert isinstance(forcing, SCMForcing)
    assert forcing.f_c == pytest.approx(1.0e-4)
    assert forcing.prescribe == "fluxes"
    # subsidence channel present and returns the stored profile
    assert forcing.subsidence_w is not None
    assert jnp.allclose(forcing.subsidence_w(0.0), -0.01)
    # surface flux interpolates to the stored constant
    assert forcing.w_th_s is not None
    assert jnp.allclose(forcing.w_th_s(1800.0), 0.06)
    # no geostrophic wind in this dry case
    assert forcing.u_geo is None


def test_forcing_surface_flux_time_interpolates():
    # a linearly-ramped surface cooling should interpolate at a mid time
    ramp = jnp.linspace(-0.01, -0.001, NT)
    art = _dry_artifact(w_theta_s=ramp)
    forcing = artifact_to_scm_forcing(art)
    times = _times()
    mid_t = 0.5 * (float(times[1]) + float(times[2]))
    expected = 0.5 * (float(ramp[1]) + float(ramp[2]))
    assert jnp.allclose(forcing.w_th_s(mid_t), expected, atol=1e-9)


def test_forcing_moist_has_geostrophic_and_qv_flux():
    forcing = artifact_to_scm_forcing(_moist_artifact())
    assert forcing.u_geo is not None
    assert jnp.allclose(forcing.u_geo(0.0), -8.75)
    assert forcing.w_qv_s is not None
    assert jnp.allclose(forcing.w_qv_s(0.0), 5.0e-5)


def test_forcing_channels_are_time_independent_profiles():
    art = _dry_artifact()
    forcing = artifact_to_scm_forcing(art)
    # profile channels ignore t (idealized cases hold LS forcing fixed)
    assert jnp.allclose(forcing.subsidence_w(0.0), forcing.subsidence_w(9999.0))


def test_npz_roundtrip_dry(tmp_path):
    art = _dry_artifact()
    p = tmp_path / "cbl.npz"
    save_artifact(art, str(p))
    loaded = load_artifact(str(p))
    assert loaded.case_name == art.case_name
    assert loaded.sgs == art.sgs
    assert loaded.prescribe == art.prescribe
    assert loaded.f_c == pytest.approx(art.f_c)
    assert jnp.allclose(loaded.theta, art.theta)
    assert jnp.allclose(loaded.wtheta_resolved, art.wtheta_resolved)
    assert loaded.qt is None  # dry: optional channel omitted and stays None


def test_npz_roundtrip_moist(tmp_path):
    art = _moist_artifact()
    p = tmp_path / "bomex.npz"
    save_artifact(art, str(p))
    loaded = load_artifact(str(p))
    assert loaded.is_moist
    assert jnp.allclose(loaded.qt, art.qt)
    assert jnp.allclose(loaded.wqt_resolved, art.wqt_resolved)
    assert jnp.allclose(loaded.u_geo, art.u_geo)
    assert jnp.allclose(loaded.w_qv_s, art.w_qv_s)


def test_forcing_reverses_profile_for_top_to_bottom_scm():
    # a vertically-varying subsidence must be reversed surface-first -> top-to-bottom
    prof = jnp.linspace(-0.02, -0.001, NZ)  # surface-first
    art = _dry_artifact(subsidence_w=prof)
    top2bot = artifact_to_scm_forcing(art)  # default flips
    assert jnp.allclose(top2bot.subsidence_w(0.0), prof[::-1])
    surf_first = artifact_to_scm_forcing(art, scm_top_to_bottom=False)
    assert jnp.allclose(surf_first.subsidence_w(0.0), prof)


# --- turbulence statistics + realised surface series (optional channels) --------
def test_turbulence_stats_round_trip_and_old_artifacts_still_load(tmp_path):
    import numpy as np
    from legoesm.atmosphere.les_suite.bridge import (
        SURFACE_SERIES_FIELDS,
        TURBULENCE_STAT_FIELDS,
    )
    old = _dry_artifact()
    save_artifact(old, tmp_path / "old.npz")
    back = load_artifact(tmp_path / "old.npz")
    assert all(getattr(back, n) is None for n in TURBULENCE_STAT_FIELDS)
    extra = {n: _profile_series(0.1 * (i + 1)) for i, n in enumerate(TURBULENCE_STAT_FIELDS)}
    extra.update({n: jnp.full((NT,), -0.07) for n in SURFACE_SERIES_FIELDS})
    art = _dry_artifact(**extra)
    save_artifact(art, tmp_path / "new.npz")
    back = load_artifact(tmp_path / "new.npz")
    for n in TURBULENCE_STAT_FIELDS + SURFACE_SERIES_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(back, n)),
                                      np.asarray(getattr(art, n)))


@pytest.mark.parametrize("name,shape", [("uw_resolved", (NT, NZ + 1)),
                                        ("pres_uw", (NZ,)),
                                        ("tau_x_s", (NT + 1,)),
                                        ("w_theta_s_cum", (NT, NZ))])
def test_turbulence_stat_shape_mismatch_rejected(name, shape):
    with pytest.raises(BridgeError):
        _dry_artifact(**{name: jnp.zeros(shape)})


def test_surface_flux_fn_interval_means_and_fallback():
    from legoesm.atmosphere.les_suite.bridge import surface_flux_fn
    t = jnp.asarray([0.0, 100.0, 300.0, 600.0])
    samples = jnp.asarray([0.0, -0.5e-3, -1e-3, -4e-3])
    cum = jnp.asarray([0.0, -0.1, -0.5, -1.4])            # means -1e-3, -2e-3, -3e-3
    art = _dry_artifact(times_s=t, w_theta_s=samples, w_theta_s_cum=cum)
    fn = surface_flux_fn(art, "w_theta_s")
    for tt, want in ((0.0, -1e-3), (99.0, -1e-3), (100.0, -2e-3), (450.0, -3e-3),
                     (900.0, -3e-3)):
        assert float(fn(tt)) == pytest.approx(want)
    # without the integral: linear interpolation of the instantaneous samples
    legacy = surface_flux_fn(_dry_artifact(times_s=t, w_theta_s=samples), "w_theta_s")
    assert float(legacy(50.0)) == pytest.approx(-0.25e-3)
    # the bridge's forcing uses the same reconstruction
    assert float(artifact_to_scm_forcing(art).w_th_s(450.0)) == pytest.approx(-3e-3)


def test_surface_heat_flux_step_mean_conserves_across_frames():
    # frames 0/100/300 s, recorded heat −0.1 then −0.4 K m; a 60 s forward-Euler step
    # straddles t=100: sampling interval means at step starts gives −0.48 K m over
    # 300 s, the step means telescope to exactly −0.5.
    from legoesm.atmosphere.les_suite.bridge import surface_flux_fn
    t = jnp.asarray([0.0, 100.0, 300.0])
    art = _dry_artifact(times_s=t, w_theta_s=jnp.asarray([0.0, -1e-3, -2e-3]),
                        w_theta_s_cum=jnp.asarray([0.0, -0.1, -0.5]),
                        theta=_profile_series(300.0, nt=3), u=_profile_series(1.0, nt=3),
                        v=_profile_series(0.0, nt=3),
                        wtheta_resolved=_profile_series(0.05, nt=3),
                        wtheta_sgs=_profile_series(0.01, nt=3))
    steps = [60.0 * k for k in range(5)]
    sampled = sum(float(surface_flux_fn(art, "w_theta_s")(s)) * 60.0 for s in steps)
    assert sampled == pytest.approx(-0.48)           # the defect the step mean removes
    fn = surface_flux_fn(art, "w_theta_s", dt_s=60.0)
    assert sum(float(fn(s)) * 60.0 for s in steps) == pytest.approx(-0.5, rel=1e-12)
    # beyond the last output the last interval's mean continues
    assert float(fn(400.0)) == pytest.approx(-2e-3)
    # integer timestamps must not truncate fractional step times
    art_int = _dry_artifact(times_s=jnp.asarray([0, 100, 300]),
                            w_theta_s=jnp.asarray([0.0, -1e-3, -2e-3]),
                            w_theta_s_cum=jnp.asarray([0.0, -0.1, -0.5]),
                            theta=_profile_series(300.0, nt=3),
                            u=_profile_series(1.0, nt=3), v=_profile_series(0.0, nt=3),
                            wtheta_resolved=_profile_series(0.05, nt=3),
                            wtheta_sgs=_profile_series(0.01, nt=3))
    fn_int = surface_flux_fn(art_int, "w_theta_s", dt_s=0.5)
    assert float(fn_int(10.25)) == pytest.approx(-1e-3)
    assert float(fn_int(99.75)) == pytest.approx(0.5 * (-1e-3) + 0.5 * (-2e-3))
    with pytest.raises(BridgeError):
        surface_flux_fn(art, "w_theta_s", dt_s=0.0)
    # an old artifact (no integral): constant flux is fine, a varying one is refused
    assert float(surface_flux_fn(_dry_artifact(), "w_theta_s", dt_s=60.0)(30.0)) == pytest.approx(0.06)
    with pytest.raises(BridgeError):
        surface_flux_fn(_dry_artifact(
            w_theta_s=jnp.asarray([0.0, -1e-3, -2e-3, -3e-3])), "w_theta_s", dt_s=60.0)
    with pytest.raises(BridgeError):                  # an integral without its series
        _dry_artifact(prescribe="none", w_theta_s=None,
                      w_theta_s_cum=jnp.zeros((NT,)))


def test_les_setup_round_trip(tmp_path):
    setup = {"surface_bc": "cooling_K_per_h", "refine": 2}
    save_artifact(_dry_artifact(les_setup=setup), tmp_path / "a.npz")
    assert load_artifact(tmp_path / "a.npz").les_setup == setup
    assert load_artifact_without_setup(tmp_path).les_setup is None
    with pytest.raises(BridgeError):
        _dry_artifact(les_setup="cooling")


def load_artifact_without_setup(tmp_path):
    save_artifact(_dry_artifact(), tmp_path / "b.npz")
    return load_artifact(tmp_path / "b.npz")


def test_surface_roughness_comes_from_the_run_record(tmp_path):
    """z0 / z0h are read from les_setup (z0h=None means one roughness); survives a
    save/load round trip; a record without z0 is refused, never defaulted."""
    art = _dry_artifact(les_setup={"z0_m": 1e-3, "z0h_m": 1e-4})
    path = tmp_path / "a.npz"
    save_artifact(art, path)
    assert surface_roughness(load_artifact(path)) == (1e-3, 1e-4)
    assert surface_roughness(_dry_artifact(les_setup={"z0_m": 0.16, "z0h_m": None})) == (
        0.16, 0.16)
    for setup in (None, {}, {"z0h_m": 1e-4}, {"z0_m": 0.0}, {"z0_m": 0.1, "z0h_m": -1.0},
                  {"z0_m": float("nan")}, {"z0_m": "invalid"}, {"z0_m": 0.1, "z0h_m": []}):
        with pytest.raises(BridgeError, match="roughness"):
            surface_roughness(_dry_artifact(les_setup=setup))


def test_theta_adv_cum_is_one_channel_and_integrates_exactly():
    import numpy as np
    from legoesm.atmosphere.les_suite.bridge import (
        LESReferenceArtifact,
        artifact_to_scm_forcing,
        forcing_profile_fn,
    )
    nt, nz = 4, 5
    z = np.linspace(10.0, 1000.0, nz)
    t = np.array([0.0, 100.0, 250.0, 400.0])
    rate = -1e-5 * (1.0 + np.arange(nt - 1))[:, None] * np.ones((1, nz))
    cum = np.concatenate([np.zeros((1, nz)), np.cumsum(rate * np.diff(t)[:, None], 0)])
    base = dict(case_name="x", sgs="lasd", heights_m=z, times_s=t,
                theta=np.full((nt, nz), 290.0), u=np.zeros((nt, nz)),
                v=np.zeros((nt, nz)), wtheta_resolved=np.zeros((nt, nz)),
                wtheta_sgs=np.zeros((nt, nz)), w_theta_s=np.zeros(nt))
    art = LESReferenceArtifact(**base, theta_adv_cum=cum)
    fn = forcing_profile_fn(art, dt_s=30.0)
    exact = forcing_profile_fn(art)
    np.testing.assert_allclose(np.asarray(exact(120.0)), rate[1])
    # forward-Euler step means telescope to the recorded integral, across frames
    steps = int(390.0 / 30.0)
    np.testing.assert_allclose(sum(np.asarray(fn(k * 30.0)) * 30.0 for k in range(steps)),
                               np.interp(steps * 30.0, t, cum[:, 0]) * np.ones(nz),
                               rtol=1e-12)
    f = artifact_to_scm_forcing(art)
    np.testing.assert_allclose(np.asarray(f.theta_adv(120.0)), rate[1][::-1])
    # with the consumer's step, a step straddling a frame applies the exact amount
    fs = artifact_to_scm_forcing(art, dt_s=40.0)
    np.testing.assert_allclose(np.asarray(fs.theta_adv(80.0)) * 40.0,
                               (20.0 * rate[0] + 20.0 * rate[1])[::-1], rtol=1e-12)
    with pytest.raises(BridgeError):
        LESReferenceArtifact(**base, theta_adv_cum=cum, theta_adv=np.zeros(nz))
    with pytest.raises(BridgeError):
        LESReferenceArtifact(**base, theta_adv_cum=cum[:, :-1])
