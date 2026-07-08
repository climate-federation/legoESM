"""Unit test for ``scripts/plot/plot_scm_rce_convection_paper.py``.

Builds a minimal results directory (summary table + two scheme records) against
the synthetic RCEMIP1 reference and asserts the three paper PDFs render.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from scripts.data import build_rcemip1_small_reference as refbuilder
from scripts.plot import plot_scm_rce_convection_paper as paper

_SCHEMES = ("mass_flux", "kuo")
_TABLE_FIELDS = [
    "scheme", "n_params_tuned", "apriori_status", "tuned_status",
    "apriori_score", "tuned_score", "score_improvement_pct",
    "apriori_T_rmse_K", "tuned_T_rmse_K",
    "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
    "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg",
    "apriori_T_rmse_norm", "tuned_T_rmse_norm",
    "apriori_qv_rmse_norm", "tuned_qv_rmse_norm",
    "apriori_cloud_rmse_norm", "tuned_cloud_rmse_norm",
    "apriori_precip_mm_day", "tuned_precip_mm_day", "crm_precip_mm_day",
    "apriori_reason", "tuned_reason",
]


def _write_results(results_dir: Path, nlev: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, scheme in enumerate(_SCHEMES):
        # a slight, distinct a-priori vs tuned pair per scheme
        T_ap = [280.0 + 0.1 * k for k in range(nlev)]
        T_tu = [281.0 + 0.1 * k for k in range(nlev)]
        qv_ap = [0.01 * (1.0 - k / nlev) for k in range(nlev)]
        qv_tu = [0.011 * (1.0 - k / nlev) for k in range(nlev)]
        qc = [1.0e-5 for _ in range(nlev)]
        record = {
            "scheme": scheme,
            "a_priori": {
                "config": {"convection": scheme}, "status": "ok",
                "T_profile": T_ap, "qv_profile": qv_ap, "qcond_profile": qc,
                "score": 2.0 + i, "T_rmse": 1.0, "qv_rmse": 1.0,
                "cloud_rmse": 1.0, "precip_mm_day": 3.0,
                "precip_ref_mm_day": 3.2,
            },
            "tuned": {
                "config": {"convection": scheme}, "status": "ok",
                "T_profile": T_tu, "qv_profile": qv_tu, "qcond_profile": qc,
                "score": 1.5 + i, "T_rmse": 0.9, "qv_rmse": 0.9,
                "cloud_rmse": 0.9, "precip_mm_day": 3.1,
                "precip_ref_mm_day": 3.2,
            },
        }
        (results_dir / f"scheme_{scheme}.json").write_text(json.dumps(record))
        rows.append({
            **{k: "" for k in _TABLE_FIELDS},
            "scheme": scheme, "n_params_tuned": 2 + i,
            "apriori_status": "ok", "tuned_status": "ok",
            "apriori_score": 2.0 + i, "tuned_score": 1.5 + i,
            "score_improvement_pct": 25.0,
            "apriori_T_rmse_K": 3.0 + i, "tuned_T_rmse_K": 2.0 + i,
            "apriori_qv_rmse_g_kg": 1.5, "tuned_qv_rmse_g_kg": 1.2,
            "crm_precip_mm_day": 3.2,
        })
    with (results_dir / "summary_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TABLE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.mark.slow
def test_build_figures_writes_three_pdfs(tmp_path):
    reference_dir = tmp_path / "ref"
    diag = refbuilder.build_reference(reference_dir, n_levels=40, n_volumes=3)
    nlev = int(diag["n_levels"])

    results_dir = tmp_path / "results"
    _write_results(results_dir, nlev)

    outdir = tmp_path / "paper"
    paths = paper.build_figures(
        results_dir, reference_dir, outdir, last_reference_files=3,
    )
    assert len(paths) == 3
    for p in paths:
        assert p.suffix == ".pdf"
        assert p.exists() and p.stat().st_size > 0
        assert p.with_suffix(".png").exists()
    # both the T and qv small-multiples and the summary bars exist
    for name in ("fig_rce_profiles_T", "fig_rce_profiles_qv",
                 "fig_rce_rmse_summary"):
        assert (outdir / f"{name}.pdf").stat().st_size > 0
