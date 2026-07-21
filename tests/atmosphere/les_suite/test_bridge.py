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
