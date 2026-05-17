"""Unit tests for scripts/ocean_fidelity/run_comparison.py.

The script lives outside ``src/`` so we add ``scripts/`` to sys.path in the
test (same pattern the script uses internally for ``src/``).
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean.fidelity import artifacts, veros_runner

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def comparison_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import ocean_fidelity.run_comparison as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def _write_synthetic_veros(case_dir: Path) -> None:
    result = veros_runner.VerosResult(
        case_name="acc_channel",
        times_s=np.array([86400.0]),
        variables={
            "temp": np.linspace(-1.0, 15.0, 4 * 4 * 4).reshape((4, 4, 4)),
            "salt": np.full((4, 4, 4), 35.0),
            "u": np.zeros((4, 4, 4)),
        },
        grid_metadata={"nx": 4, "ny": 4, "nz": 4},
        provenance={
            "veros_version": "test",
            "runlen_s": 86400.0,
            "wall_seconds": 0.1,
            "key": "deadbeefdeadbeef",
        },
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "result.pkl", "wb") as f:
        pickle.dump(result, f)


def _write_synthetic_legoesm(root: Path, *, case: str, grid: str, res: str) -> None:
    case_dir = root / case / grid / res
    n = 5
    artifacts.write_synthetic(
        case_dir,
        scalars={"status": "PASS", "days": 1.0, "notes": f"{case}/{grid}"},
        timeseries={
            "step": np.arange(n),
            "time_days": np.linspace(0.0, 1.0, n),
            "mean_T": np.full(n, 10.0),
            "mean_S": np.full(n, 35.0),
        },
    )


def test_render_report_no_data_shows_helpful_messages(comparison_module, tmp_path):
    out = comparison_module.render_report({}, {}, git_sha="abc")
    assert "No Veros results in cache" in out
    assert "No legoESM matrix artifacts" in out
    assert "task #11" in out
    assert "task #12" in out


def test_render_report_lists_pipeline_status(comparison_module, isolated_cache, tmp_path):
    veros_dir = isolated_cache / "veros" / "acc_channel" / "deadbeefdeadbeef"
    _write_synthetic_veros(veros_dir)
    legoesm_root = tmp_path / "results" / "ocean"
    _write_synthetic_legoesm(legoesm_root, case="barotropic_wave",
                              grid="latlon", res="48x72")
    _write_synthetic_legoesm(legoesm_root, case="barotropic_wave",
                              grid="mpas", res="ico4")

    veros = comparison_module._load_veros_results()
    legoesm = comparison_module._load_legoesm_results(legoesm_root)

    out = comparison_module.render_report(veros, legoesm, git_sha="testsha")
    assert "Veros cases cached: ['acc_channel']" in out
    assert "barotropic_wave" in out
    assert "Direct overlap" in out
    assert "cross-grid deltas" in out


def test_legoesm_loader_walks_variant_subdirs(comparison_module, tmp_path):
    """rest_state has an extra variant level — loader must catch that."""
    root = tmp_path / "results" / "ocean"
    _write_synthetic_legoesm(
        root,
        case="rest_state/rest_state_stratified_with_land",
        grid="latlon",
        res="36x72",
    )
    out = comparison_module._load_legoesm_results(root)
    # case name pulled from third-from-last path component
    assert "rest_state_stratified_with_land" in out


def test_field_summary_handles_all_nan(comparison_module):
    arr = np.full((3, 3), np.nan)
    out = comparison_module._field_summary(arr)
    assert "NaN" in out


def test_field_summary_reports_min_max_mean(comparison_module):
    arr = np.array([1.0, 2.0, 3.0])
    out = comparison_module._field_summary(arr)
    assert "min=+1.000e+00" in out
    assert "max=+3.000e+00" in out


def test_veros_scalar_drops_halos_and_picks_last_tau(comparison_module):
    nx_h, ny_h, nz, n_tau = 8, 8, 4, 3
    arr = np.zeros((nx_h, ny_h, nz, n_tau))
    # interior (drop 2-cell halo each side): cells [2..-2]
    arr[2:-2, 2:-2, :, -1] = 1.5  # latest tau, interior
    arr[2:-2, 2:-2, :, 0] = -99.0  # earlier tau (must be ignored)

    class FakeResult:
        variables = {"temp": arr}

    out = comparison_module._veros_scalar(FakeResult(), "temp")
    assert out["min"] == 1.5
    assert out["max"] == 1.5
    assert out["mean"] == 1.5


def test_veros_scalar_missing_variable_returns_empty(comparison_module):
    class FakeResult:
        variables = {}
    assert comparison_module._veros_scalar(FakeResult(), "temp") == {}


def test_legoesm_scalar_handles_missing_and_object_dtype(comparison_module, tmp_path):
    cd = tmp_path / "case" / "latlon" / "36x72"
    cd.mkdir(parents=True)
    (cd / "results.txt").write_text("status: PASS\n")
    (cd / "mean_timeseries.csv").write_text(
        "step,time_days,label\n0,0.0,A\n1,0.5,B\n"
    )
    bundle = artifacts.load(cd)
    assert comparison_module._legoesm_scalar(bundle, "missing_key") is None
    assert comparison_module._legoesm_scalar(bundle, "label") is None


def test_legoesm_scalar_returns_last_sample(comparison_module, tmp_path):
    cd = tmp_path / "case" / "latlon" / "36x72"
    artifacts.write_synthetic(
        cd,
        timeseries={
            "step": np.arange(3),
            "time_days": np.array([0.0, 0.5, 1.0]),
            "mean_T": np.array([10.0, 11.0, 12.5]),
        },
    )
    bundle = artifacts.load(cd)
    assert comparison_module._legoesm_scalar(bundle, "mean_T") == 12.5


def test_format_overlap_section_produces_side_by_side_table(comparison_module, tmp_path):
    veros_dir = tmp_path / "veros" / "lock_exchange" / "deadbeef"
    _write_synthetic_veros(veros_dir)
    veros_result = pickle.load(open(veros_dir / "result.pkl", "rb"))

    cd = tmp_path / "lock_exchange" / "latlon" / "36x72"
    artifacts.write_synthetic(
        cd,
        scalars={"status": "PASS"},
        timeseries={
            "step": np.arange(3),
            "time_days": np.array([0.0, 0.5, 1.0]),
            "mean_T": np.array([10.0, 11.0, 12.5]),
            "max_speed": np.array([0.0, 0.05, 0.1]),
        },
    )
    bundle = artifacts.load(cd)
    lines = comparison_module._format_overlap_section(
        "lock_exchange", veros_result, [bundle]
    )
    text = "\n".join(lines)
    assert "Veros vs legoESM" in text
    assert "lock_exchange" in text
    assert "T mean" in text
    assert "+1.2500e+01" in text  # legoESM mean_T last sample
    assert "max |u|" in text
    assert "B-grid" in text  # caveat note


def test_main_writes_output_file(comparison_module, isolated_cache, tmp_path):
    output = tmp_path / "report.md"
    legoesm_root = tmp_path / "results"
    legoesm_root.mkdir()
    # No data — should still write a report with skeleton sections.
    sys.argv = ["run_comparison.py",
                "--legoesm-root", str(legoesm_root),
                "--output", str(output)]
    rc = comparison_module.main()
    assert rc == 0
    assert output.exists()
    text = output.read_text()
    assert "# Ocean fidelity initial comparison" in text
