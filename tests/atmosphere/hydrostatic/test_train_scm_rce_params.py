"""Quick smoke for ``scripts/run/train_scm_rce_params.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legoesm.training.scm_rce_metrics import (
    precip_score_jax,
    score_profiles_jax,
    score_profiles_precip_jax,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run" / "train_scm_rce_params.py"
REFERENCE_DIR = REPO_ROOT / "results" / "rcemip1_n128_ocean"
RECOMMENDED_DEFAULTS = (
    REPO_ROOT / "results" / "scm_rce_campaign" / "recommended_defaults.json"
)
TUNED_PARAMETERS = REPO_ROOT / "results" / "scm_rce_campaign" / "tuned_parameters.json"


def test_scm_rce_metrics_zero_for_matching_profiles():
    class Ref:
        mass_weights = [0.25, 0.75]
        T_ref = [200.0, 300.0]
        qv_ref = [0.001, 0.01]
        qcond_ref = [0.0, 0.001]
        precip_ref_mm_day = 3.5

    T, qv, cloud, profile_combined = score_profiles_jax(
        Ref(),
        Ref.T_ref,
        Ref.qv_ref,
        Ref.qcond_ref,
        profile_floor=1.0e-12,
    )
    assert float(T) == 0.0
    assert float(qv) == 0.0
    assert float(cloud) == 0.0
    assert float(profile_combined) == 0.0

    T, qv, cloud, precip, combined = score_profiles_precip_jax(
        Ref(),
        Ref.T_ref,
        Ref.qv_ref,
        Ref.qcond_ref,
        Ref.precip_ref_mm_day,
        profile_floor=1.0e-12,
    )
    assert float(T) == 0.0
    assert float(qv) == 0.0
    assert float(cloud) == 0.0
    assert float(precip) == 0.0
    assert float(combined) == 0.0


def test_precip_score_units_and_normalization():
    score = precip_score_jax(5.0, 3.5, normalization_mm_day=3.0)
    assert float(score) == 0.5


@pytest.mark.skipif(
    not (REFERENCE_DIR / "snapshots3d").is_dir()
    or not RECOMMENDED_DEFAULTS.exists()
    or not TUNED_PARAMETERS.exists(),
    reason="CRM reference snapshots or SCM campaign tuned parameters are absent",
)
def test_train_scm_rce_params_quick_gradients_loss_and_bounds(tmp_path):
    outdir = tmp_path / "scm_rce_training"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--steps",
            "2",
            "--recommended",
            str(RECOMMENDED_DEFAULTS),
            "--outdir",
            str(outdir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        "SCM RCE gradient quick smoke failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    payload = json.loads((outdir / "trained_parameters.json").read_text())
    winners = payload["campaign_recommendation"]["winners"]
    assert winners == {
        "radiation": "rrtmgp",
        "turbulence": "clubb",
        "microphysics": "sundqvist",
        "convection": "sbm",
        "gravity_wave_drag": "mcfarlane",
    }
    config = payload["campaign_tuned_config"]
    assert config["radiation"]["scheme"] == "rrtmgp"
    assert config["turbulence"]["scheme"] == "clubb"
    assert config["microphysics"]["scheme"] == "sundqvist"
    assert config["convection"]["scheme"] == "sbm"
    assert config["gravity_wave_drag"]["scheme"] == "mcfarlane"
    assert payload["campaign_recommendation"]["trained_scheme_keys"] == [
        "atm.conv.SBMConfig",
        "atm.gwd.McFarlaneConfig",
        "atm.micro.SundqvistConfig",
    ]
    trained = [row for row in payload["parameters"] if row["trained"]]
    assert trained, "quick mode must keep at least one finite nonzero-gradient leaf"
    assert payload["gradient_check"]["all_trained_finite_nonzero"]
    assert payload["loss"]["final_train_loss"] < payload["loss"]["initial_train_loss"]

    stats = payload["gradient_check"]["stats"]
    for row in trained:
        assert stats[row["name"]]["finite"], row
        assert stats[row["name"]]["nonzero"], row
        value = float(row["gradient_trained"])
        assert float(row["lower"]) < value < float(row["upper"]), row

    assert (outdir / "loss_curve.png").exists()
    assert (outdir / "profiles_vs_crm_before_after.png").exists()
    assert (outdir / "summary.md").exists()
