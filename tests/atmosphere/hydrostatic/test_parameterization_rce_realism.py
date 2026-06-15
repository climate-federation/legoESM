"""Equilibrium SCM-RCE realism guardrail.

This complements the tendency-level truth-tier validators by testing the
campaign's equilibrium-profile gate directly.  Slow per-scheme RCE runs are
opt-in via ``LEGOESM_RUN_SLOW_RCE=1``; the synthetic failure test is always
active so the gate cannot become vacuous.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.run import run_scm_rce_campaign as campaign
from legoesm.training.scm_rce_metrics import (
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
