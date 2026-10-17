import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_budget_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def write_archive(root: Path, *, drop: str | None = None, non_finite: str | None = None):
    fields = list(PREFLIGHT.LEGO_FULL_FIELDS)
    manifest = {
        "status": "complete",
        "completed_days": 180,
        "per_step": True,
        "full_state_window": [1, 180],
        "full_state_fields": fields,
        "namelist": {"nn_GYRE": 4, "ln_dynldf_blp": True},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for day in (1, 30, 180):
        arrays = {}
        for key in PREFLIGHT.required_legoesm_keys():
            if key != drop:
                arrays[key] = np.array([np.nan if key == non_finite else day])
        np.savez(root / f"day{day:04d}.npz", **arrays)


def test_legoesm_archive_passes_only_with_every_registered_field(tmp_path):
    write_archive(tmp_path)
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "PASS"
    assert report["errors"] == []

    missing = tmp_path / "missing"
    missing.mkdir()
    write_archive(missing, drop="v_m_s_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(missing)
    assert report["status"] == "REFUSE"
    assert any("v_m_s_3d_daily_mean" in error for error in report["errors"])


def test_legoesm_archive_refuses_non_finite_active_provider(tmp_path):
    write_archive(tmp_path, non_finite="temperature_c_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "REFUSE"
    assert any("non-finite temperature_c_3d_daily_mean" in error
               for error in report["errors"])


def test_joint_report_cannot_pass_one_refusing_provider(tmp_path, monkeypatch):
    write_archive(tmp_path)
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "REFUSE", "errors": ["plant"]},
    )
    report = PREFLIGHT.build_report(tmp_path, tmp_path / "nemo")
    assert report["verdict"] == "ACQUISITION_NEEDED"
    assert "refuse term attribution" in report["disposition"]
