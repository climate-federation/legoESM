"""Quick smoke for ``scripts/run/train_scm_rce_params.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legoesm.training.scm_rce_metrics import score_profiles_jax


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run" / "train_scm_rce_params.py"
REFERENCE_DIR = REPO_ROOT / "results" / "rcemip1_n128_ocean"
TUNED_PARAMETERS = REPO_ROOT / "results" / "scm_rce_campaign" / "tuned_parameters.json"


def test_scm_rce_metrics_zero_for_matching_profiles():
    class Ref:
        mass_weights = [0.25, 0.75]
        T_ref = [200.0, 300.0]
        qv_ref = [0.001, 0.01]
        qcond_ref = [0.0, 0.001]

    T, qv, cloud, combined = score_profiles_jax(
        Ref(),
        Ref.T_ref,
        Ref.qv_ref,
        Ref.qcond_ref,
        profile_floor=1.0e-12,
    )
    assert float(T) == 0.0
    assert float(qv) == 0.0
    assert float(cloud) == 0.0
    assert float(combined) == 0.0


@pytest.mark.skipif(
    not (REFERENCE_DIR / "snapshots3d").is_dir() or not TUNED_PARAMETERS.exists(),
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
