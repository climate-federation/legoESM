"""Equilibrium SCM-RCE realism guardrail.

This complements the tendency-level truth-tier validators by testing the
campaign's equilibrium-profile gate directly.  Slow per-scheme RCE runs are
opt-in via ``LEGOESM_RUN_SLOW_RCE=1``; the synthetic failure test is always
active so the gate cannot become vacuous.
"""

from __future__ import annotations

import csv
import os

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.run import run_scm_rce_campaign as campaign
from legoesm.training.scm_rce_metrics import (
    COLD_POINT_MAX_K,
    COLD_POINT_MIN_K,
    MADIAB_MAX_TOL_K,
    MADIAB_MEAN_TOL_K,
    MIN_FREE_TROP_LEVELS,
    TROP_MAX_Z_KM,
    TROP_MIN_Z_KM,
    moist_adiabat_diagnostics_jax,
    realism_reasons_from_diagnostics,
)


GATED = {
    "convection": {"sbm", "dca", "zhang_mcfarlane"},
    "microphysics": {
        "kessler", "sundqvist", "seifert_beheng", "morrison", "thompson", "p3",
    },
    "turbulence": {
        "louis", "smagorinsky", "tke", "mynn25", "clubb_lite", "clubb",
        "holtslag_boville", "ysu", "edmf",
    },
    "gravity_wave_drag": {"none", "rayleigh", "lindzen", "mcfarlane", "hines", "e3sm_cam"},
}

REALISM_TODO = {
    "convection": {
        "kuo", "mass_flux", "edmf", "kain_fritsch", "emanuel",
        "tiedtke", "bechtold",
    },
    "turbulence": set(),
    "gravity_wave_drag": {"prognostic_spectral"},
}


def test_realism_gate_rejects_synthetic_unphysical_profile():
    nlev = 30
    z_m = np.linspace(32_450.0, 550.0, nlev)
    sigma = np.linspace(0.02, 0.98, nlev)
    p_full = sigma * campaign.WING_P_SFC
    T = np.full(nlev, 280.0)
    qv = np.full(nlev, 1.0e-4)

    diag = moist_adiabat_diagnostics_jax(
        T_profile=jnp.asarray(T),
        qv_profile=jnp.asarray(qv),
        p_full=jnp.asarray(p_full),
        sigma_full=jnp.asarray(sigma),
        z_m=jnp.asarray(z_m),
    )
    scalar_diag = {
        key: float(np.asarray(value))
        for key, value in diag.items()
        if key != "moist_adiabat"
    }
    reasons = realism_reasons_from_diagnostics(scalar_diag)
    assert reasons
    assert any("moist adiabat" in reason or "cold point" in reason for reason in reasons)


def _realistic_scalar_diag() -> dict[str, float]:
    """A diagnostics dict comfortably INSIDE every realism bound (a healthy RCE column).

    Built from the published thresholds (not magic numbers) so a future threshold change keeps
    this fixture centred."""
    return {
        "n_free_trop_levels": float(MIN_FREE_TROP_LEVELS + 2),
        "mean_abs_K": MADIAB_MEAN_TOL_K * 0.25,
        "max_abs_K": MADIAB_MAX_TOL_K * 0.25,
        "cold_point_T_K": 0.5 * (COLD_POINT_MIN_K + COLD_POINT_MAX_K),
        "cold_point_z_km": 0.5 * (TROP_MIN_Z_KM + TROP_MAX_Z_KM),
    }


def test_realism_gate_accepts_a_healthy_profile_nonvacuously():
    """Non-vacuity: a diagnostics dict inside EVERY bound yields ZERO reasons.

    Without this the gate could silently drift to reject-EVERYTHING (a refactor inverting a
    comparison), which would make the correction loop NEVER trust an LES → never correct a
    bias.  This is the lower fence; the rejection tests are the upper one."""
    assert realism_reasons_from_diagnostics(_realistic_scalar_diag()) == []


@pytest.mark.parametrize("mode", ["missing", "nan"])
def test_realism_gate_fails_closed_on_missing_or_nan_diagnostics(mode):
    """Fail-CLOSED safety: a degenerate LES whose diagnostics are MISSING (empty dict) or NaN is
    REJECTED, never silently trusted.

    This is the precise property that throws out a tiny / dead LES — and therefore WHY a
    realistic (HPC-scale) LES is required to demonstrate the empirical clause: a cheap degenerate
    LES cannot sneak a coefficient into the model.  A regression changing the ``diag.get(key,
    nan)`` default to a passing value (e.g. ``0.0``) would let a missing cold point clear its
    lower bound and slip a garbage LES through; here it must produce ≥1 reason.  (NaN comparisons
    are False, so ``not (lo <= nan <= hi)`` is True → a reason is appended.)"""
    if mode == "missing":
        diag: dict[str, float] = {}                       # every key absent → all defaults fire
    else:
        diag = {k: float("nan") for k in _realistic_scalar_diag()}   # present but NaN
    reasons = realism_reasons_from_diagnostics(diag)
    assert reasons, f"the realism gate trusted a degenerate ({mode}) LES diagnostic set"


@pytest.mark.parametrize("field, bad_value, needle", [
    ("n_free_trop_levels", float(MIN_FREE_TROP_LEVELS - 1), "free-troposphere"),
    ("mean_abs_K", MADIAB_MEAN_TOL_K + 1.0, "moist adiabat"),          # space → the MEAN reason
    ("max_abs_K", MADIAB_MAX_TOL_K + 1.0, "moist-adiabat deviation"),  # hyphen → the MAX reason
    ("cold_point_T_K", COLD_POINT_MIN_K - 1.0, "cold point T"),
    ("cold_point_z_km", TROP_MAX_Z_KM + 1.0, "cold point z"),
])
def test_realism_gate_each_dimension_is_load_bearing(field, bad_value, needle):
    """Every gate dimension independently rejects: from a healthy dict, push ONE field past its
    bound and assert the matching reason fires.  Proves no check is dead code — deleting any one
    would drop its row and fail here (the gate is a genuine 5-dimensional fence, not 1 real
    check + 4 decorative)."""
    diag = _realistic_scalar_diag()
    diag[field] = bad_value
    reasons = realism_reasons_from_diagnostics(diag)
    assert any(needle in r for r in reasons), (
        f"pushing {field} to {bad_value} did not raise a '{needle}' realism failure: {reasons}")


def test_crm_clear_sky_subsidence_profile_is_area_weighted_and_closed(tmp_path):
    w = np.zeros((2, 2, 4), dtype=float)
    cond = np.zeros_like(w)
    w[..., 0] = 0.01
    w[..., 1] = -0.02
    cond[0, 0, 1] = 2.0e-6
    cond[0, 1, 1] = 2.0e-6
    w[..., 2] = -0.03
    cond[..., 2] = 2.0e-6
    w[..., 3] = -0.04
    path = tmp_path / "vol_00000001.npz"
    np.savez(path, w=w, cond=cond)

    subsidence, cloud_fraction = campaign._reference_crm_clear_sky_subsidence(
        [path],
        condensate_threshold=1.0e-6,
    )

    np.testing.assert_allclose(subsidence, [0.0, -0.01, 0.0, 0.0])
    np.testing.assert_allclose(cloud_fraction, [0.0, 0.5, 1.0, 0.0])


def test_campaign_rollout_applies_selected_subsidence_forcing():
    nlev = 8
    z_m = np.linspace(8000.0, 550.0, nlev)
    z_half = np.empty(nlev + 1)
    z_half[0] = z_m[0] + 0.5 * (z_m[0] - z_m[1])
    z_half[1:-1] = 0.5 * (z_m[:-1] + z_m[1:])
    z_half[-1] = 0.0
    sigma_half = np.asarray(
        campaign.wing2018_pressure_profile(jnp.asarray(z_half)) / campaign.WING_P_SFC,
        dtype=float,
    )
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    mass_weights = np.diff(sigma_half)
    mass_weights = mass_weights / np.sum(mass_weights)
    zeros = np.zeros(nlev)
    w_sub = np.zeros(nlev)
    w_sub[1:-1] = -0.01
    ref = campaign.ReferenceProfiles(
        z_m=z_m,
        sigma_half=sigma_half,
        sigma_full=sigma_full,
        mass_weights=mass_weights,
        T_ref=zeros,
        qv_ref=zeros,
        qcond_ref=zeros,
        files_used=[],
        sfc_cross_check={},
        crm_clear_sky_subsidence_m_s=w_sub,
    )
    cfg = campaign.PhysicsConfig(
        radiation=campaign.RadiationConfig(scheme="none"),
        convection=campaign.ConvectionConfig(scheme="none"),
        turbulence=campaign.TurbulenceConfig(scheme="none"),
        microphysics=campaign.MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=campaign.GravityWaveDragConfig(scheme="none"),
    )
    common = dict(
        cfg=cfg,
        ref=ref,
        days=60.0 / campaign.SECONDS_PER_DAY,
        dt=60.0,
        analysis_days=60.0 / campaign.SECONDS_PER_DAY,
        require_equilibrium=False,
        require_realism=False,
        equil_T_tol_K=campaign.EQUIL_T_TOL_K,
        equil_qv_tol=campaign.EQUIL_QV_TOL,
        equil_qcond_tol=campaign.EQUIL_QCOND_TOL,
        surface_wind_m_s=0.0,
        coriolis_s_inv=0.0,
    )
    no_forcing = campaign.run_scm_rce(
        label="unit:none",
        large_scale_forcing="none",
        **common,
    )
    with_subsidence = campaign.run_scm_rce(
        label="unit:subsidence",
        large_scale_forcing="crm_clear_sky_subsidence",
        **common,
    )

    diff = np.asarray(with_subsidence.T_profile) - np.asarray(no_forcing.T_profile)
    assert float(np.max(diff[1:-1])) > 1.0e-4


def test_campaign_writes_scheme_profile_artifact(tmp_path):
    nlev = 3
    ref = campaign.ReferenceProfiles(
        z_m=np.array([3000.0, 1500.0, 500.0]),
        sigma_half=np.array([0.0, 0.3, 0.7, 1.0]),
        sigma_full=np.array([0.15, 0.5, 0.85]),
        mass_weights=np.array([0.3, 0.4, 0.3]),
        T_ref=np.array([250.0, 280.0, 300.0]),
        qv_ref=np.array([1.0e-4, 5.0e-3, 1.5e-2]),
        qcond_ref=np.zeros(nlev),
        files_used=[],
        sfc_cross_check={},
    )
    run = campaign.RunDiagnostics(
        label="unit",
        config={
            "radiation": "rrtmgp",
            "convection": "kain_fritsch",
            "turbulence": "louis",
            "microphysics": "kessler",
            "gravity_wave_drag": "none",
        },
        status="ok",
        reason="",
        T_rmse=0.0,
        qv_rmse=0.0,
        cloud_rmse=0.0,
        precip_rmse=0.0,
        score=0.0,
        drift_T_rmse_K=0.0,
        drift_qv_rmse=0.0,
        drift_qcond_rmse=0.0,
        T_profile=[251.0, 279.0, 300.0],
        qv_profile=[2.0e-4, 4.0e-3, 1.4e-2],
        qcond_profile=[0.0, 1.0e-5, 0.0],
    )
    path = tmp_path / "profiles_convection.csv"
    campaign._write_profile_csv(path, ref, [run])

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == nlev
    assert rows[0]["scheme"] == "kain_fritsch"
    assert float(rows[0]["z_km"]) == 3.0
    assert float(rows[0]["T_K"]) == 251.0
    assert float(rows[0]["T_crm_K"]) == 250.0


def test_scheme_realism_partition_is_explicit_and_shrink_only():
    all_classified = {
        category: set(GATED.get(category, set())) | set(REALISM_TODO.get(category, set()))
        for category in campaign.SCHEME_SWEEPS
    }
    for category, schemes in campaign.SCHEME_SWEEPS.items():
        assert set(schemes) <= all_classified[category]
    assert "emanuel" in REALISM_TODO["convection"]
    assert "kain_fritsch" in REALISM_TODO["convection"]
    assert "prognostic_spectral" in REALISM_TODO["gravity_wave_drag"]


@pytest.mark.skipif(
    os.environ.get("LEGOESM_RUN_SLOW_RCE") != "1",
    reason="set LEGOESM_RUN_SLOW_RCE=1 to run the 100-day SCM-RCE guardrail",
)
def test_dca_equilibrium_rce_smoke_passes_reference_gate():
    ref = campaign.build_reference_profiles(
        campaign.DEFAULT_REFERENCE_DIR,
        last_n=2,
        precip_analysis_days=2.0,
    )
    cfg = campaign.make_physics_config(
        radiation="gray",
        radiation_update_interval_steps=1,
        convection="dca",
    )
    run = campaign.run_scm_rce(
        cfg,
        ref,
        label="guardrail:dca",
        days=100.0,
        dt=campaign.DEFAULT_DT_S,
        analysis_days=5.0,
        require_equilibrium=True,
        require_realism=True,
        equil_T_tol_K=campaign.EQUIL_T_TOL_K,
        equil_qv_tol=campaign.EQUIL_QV_TOL,
        equil_qcond_tol=campaign.EQUIL_QCOND_TOL,
    )
    assert run.status == "ok", run.reason
