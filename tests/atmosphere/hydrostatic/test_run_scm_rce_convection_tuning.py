"""Quick smoke for ``scripts/run/run_scm_rce_convection_tuning.py``.

Unlike the campaign smoke (which skips when the CRM reference checkout is
absent), this test synthesizes a minimal Wing-2018-consistent CRM reference
from the repo's own RCEMIP analytic profiles, so it always runs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run" / "run_scm_rce_convection_tuning.py"

QUICK_SCHEMES = ("mass_flux", "kuo")


def _write_synthetic_reference(reference_dir: Path) -> None:
    """Minimal CRM RCE reference: Wing-2018 analytic mean state, 2 volumes.

    Layout and keys mirror what ``run_scm_rce_campaign.build_reference_profiles``
    reads: ``snapshots3d/vol_*.npz`` with ``z`` (top->surface), ``T``, ``mse``
    [kJ/kg], ``cond``, ``w`` shaped (nx, ny, nz); ``snapshots/sfc_*.npz`` with
    ``day`` and ``precip``.
    """
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        wing2018_qv_profile,
        wing2018_temperature_profile,
    )

    nx = ny = 4
    nz = 48
    z = np.linspace(20_000.0, 100.0, nz)  # top -> surface, strictly decreasing
    T = np.asarray(wing2018_temperature_profile(jnp.asarray(z)), dtype=float)
    qv = np.asarray(wing2018_qv_profile(jnp.asarray(z)), dtype=float)
    cond = np.where((z > 1_000.0) & (z < 12_000.0), 1.0e-5, 0.0)
    mse_kj = (constants.c_pd * T + constants.g * z + constants.L_v * qv) / 1.0e3

    vol_dir = reference_dir / "snapshots3d"
    sfc_dir = reference_dir / "snapshots"
    vol_dir.mkdir(parents=True)
    sfc_dir.mkdir(parents=True)

    def tile(profile: np.ndarray) -> np.ndarray:
        return np.broadcast_to(profile, (nx, ny, nz)).copy()

    for i, step in enumerate((100, 200)):
        np.savez(
            vol_dir / f"vol_{step:08d}.npz",
            z=z,
            T=tile(T),
            mse=tile(mse_kj),
            cond=tile(cond),
            w=np.zeros((nx, ny, nz)),
        )
        np.savez(
            sfc_dir / f"sfc_{step:08d}.npz",
            day=np.asarray(float(i + 1)),
            precip=np.full((nx, ny), 4.0),
        )


@pytest.mark.slow
def test_quick_convection_tuning_outputs_table_plots_and_bounded_params(tmp_path):
    reference_dir = tmp_path / "reference"
    _write_synthetic_reference(reference_dir)
    outdir = tmp_path / "convection_tuning"

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--reference-dir",
            str(reference_dir),
            "--last-reference-files",
            "2",
            "--outdir",
            str(outdir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"convection tuning quick smoke failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Per-scheme JSON records: both quick schemes, bounded tuned parameters.
    for scheme in QUICK_SCHEMES:
        record = json.loads((outdir / f"scheme_{scheme}.json").read_text())
        assert record["scheme"] == scheme
        assert record["a_priori"]["config"]["convection"] == scheme
        assert record["tuned"]["config"]["convection"] == scheme
        assert math.isfinite(float(record["a_priori"]["score"]))
        assert math.isfinite(float(record["tuned"]["score"]))
        assert (
            float(record["tuned"]["score"])
            <= float(record["a_priori"]["score"]) + 1.0e-12
        )
        assert record["tune_records"], scheme
        for rec in record["tune_records"]:
            assert rec["category"] == "convection"
            assert float(rec["lower"]) <= float(rec["tuned"]) <= float(rec["upper"])
        for key in ("a_priori", "tuned"):
            assert len(record[key]["T_profile"]) == 48
        phys = record["tuned_physical_rmse"]
        assert math.isfinite(phys["T_rmse_K"]) and phys["T_rmse_K"] >= 0.0

    # Summary table covers both schemes with finite a-priori and tuned scores.
    with (outdir / "summary_table.csv").open(newline="") as f:
        rows = {row["scheme"]: row for row in csv.DictReader(f)}
    assert set(QUICK_SCHEMES) <= set(rows)
    for scheme in QUICK_SCHEMES:
        row = rows[scheme]
        assert float(row["apriori_score"]) > 0.0
        assert float(row["tuned_score"]) <= float(row["apriori_score"]) + 1.0e-12
        assert int(row["n_params_tuned"]) > 0
        assert math.isfinite(float(row["tuned_T_rmse_K"]))

    assert "| scheme |" in (outdir / "summary.md").read_text()

    # Plots: per-scheme + combined a-priori/a-posteriori profile figures.
    for scheme in QUICK_SCHEMES:
        assert (outdir / f"profiles_{scheme}.png").stat().st_size > 0
    for name in ("profiles_all_T.png", "profiles_all_qv.png",
                 "profiles_all_qcond.png", "score_apriori_vs_tuned.png"):
        assert (outdir / name).stat().st_size > 0

    # Aggregate-only mode rebuilds the table from the JSON records alone.
    (outdir / "summary_table.csv").unlink()
    result2 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--aggregate-only",
            "--reference-dir",
            str(reference_dir),
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
    assert result2.returncode == 0, result2.stderr
    assert (outdir / "summary_table.csv").exists()

    # A missing scheme record is a hard error without --allow-missing.
    result3 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--aggregate-only",
            "--schemes",
            "mass_flux,kuo,tiedtke",
            "--reference-dir",
            str(reference_dir),
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
    assert result3.returncode != 0
    assert "missing scheme records" in (result3.stderr + result3.stdout)


def test_unknown_scheme_selection_is_a_hard_error(tmp_path):
    reference_dir = tmp_path / "reference"
    _write_synthetic_reference(reference_dir)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--schemes",
            "not_a_scheme",
            "--reference-dir",
            str(reference_dir),
            "--last-reference-files",
            "2",
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode != 0
    assert "Unknown convection schemes" in (result.stderr + result.stdout)
