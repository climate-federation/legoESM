"""Quick smoke for ``scripts/run/run_scm_rce_campaign.py``."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py"
REFERENCE_DIR = REPO_ROOT / "results" / "rcemip1_n128_ocean"


@pytest.mark.skipif(
    not (REFERENCE_DIR / "snapshots3d").is_dir(),
    reason="CRM reference snapshots are not present in this checkout",
)
def test_run_scm_rce_campaign_quick_outputs_ranking_and_bounded_tuning(tmp_path):
    outdir = tmp_path / "scm_rce_campaign"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--categories",
            "microphysics",
            "--last-reference-files",
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
        f"campaign quick smoke failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    ranking_path = outdir / "ranking_microphysics.csv"
    assert ranking_path.exists()
    with ranking_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {"kessler", "sundqvist"} <= {row["scheme"] for row in rows}
    assert any(row["status"] == "ok" for row in rows)
    assert "precip_mm_day" in rows[0]
    assert "crm_precip_mm_day" in rows[0]
    assert "realism_status" in rows[0]
    assert any(float(row["precip_mm_day"]) >= 0.0 for row in rows)

    tuned_path = outdir / "tuned_parameters.json"
    tuned = json.loads(tuned_path.read_text())
    assert tuned, "quick tuning must exercise at least one tunable parameter"
    for row in tuned:
        value = float(row["tuned"])
        assert float(row["lower"]) <= value <= float(row["upper"]), row

    recommended = json.loads((outdir / "recommended_defaults.json").read_text())
    assert recommended["winners"]["microphysics"] in {"kessler", "sundqvist"}
