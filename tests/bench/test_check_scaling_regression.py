"""check_scaling_regression: pairs rows on the figure's key and fails on a
slowdown, a dropped row, or a changed problem size."""
import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "check_scaling_regression.py"
_spec = importlib.util.spec_from_file_location("check_scaling_regression", _P)
csr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csr)


def _tri(nd, ms, nlev=75, **kw):
    return {"component": "ocean", "grid_type": "tripole", "platform": "gpu",
            "precision": "float32", "mode": "strong", "n_lat": 3072, "n_lon": 4352,
            "nlev": nlev, "n_devices": nd, "steady_median_ms": ms, "valid": True,
            "metadata": {}, **kw}


def _write(d: Path, name: str, rec: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text(json.dumps(rec) + "\n")


def _main(tmp_path, tol="5", nlev="40"):
    return csr.main(["--baseline", str(tmp_path / "b"), "--candidate", str(tmp_path / "c"),
                     "--tol-pct", tol, "--atm-nlev", nlev])


def _pair(tmp_path, base_ms, cand_ms):
    _write(tmp_path / "b", "tri_d8", _tri(8, base_ms))
    _write(tmp_path / "c", "tri_d8", _tri(8, cand_ms))


def test_slowdown_beyond_tolerance_fails(tmp_path):
    _pair(tmp_path, 100.0, 106.0)
    assert _main(tmp_path) == 1


def test_within_tolerance_and_speedup_pass(tmp_path):
    _pair(tmp_path, 100.0, 104.0)
    assert _main(tmp_path) == 0
    _pair(tmp_path, 100.0, 80.0)
    assert _main(tmp_path, tol="0") == 0


def test_no_common_rows_is_not_a_pass(tmp_path):
    _write(tmp_path / "b", "tri_d8", _tri(8, 100.0))
    _write(tmp_path / "c", "tri_d16", _tri(16, 50.0))
    assert _main(tmp_path) == 2


def test_candidate_row_refused_by_figure_filter_is_missing_not_passed(tmp_path):
    """d8 unchanged; d16 regressed AND written at a level count the figure
    refuses, so it drops out of the candidate set -- must not exit 0."""
    _write(tmp_path / "b", "tri_d8", _tri(8, 100.0))
    _write(tmp_path / "b", "tri_d16", _tri(16, 50.0))
    _write(tmp_path / "c", "tri_d8", _tri(8, 100.0))
    _write(tmp_path / "c", "tri_d16", _tri(16, 500.0, nlev=60))
    assert _main(tmp_path) == 3


def test_same_key_different_mesh_is_flagged(tmp_path):
    """Weak ocean-lat-lon label is rows/dev, so 3072 and 3075 rows at 8
    devices share a key; the raw size check must catch it."""
    _write(tmp_path / "b", "tri_weak_d8", _tri(8, 100.0, mode="weak"))
    _write(tmp_path / "c", "tri_weak_d8", _tri(8, 90.0, mode="weak", n_lat=3075))
    assert _main(tmp_path) == 3


def test_atmosphere_rows_follow_atm_nlev(tmp_path):
    atm = {"component": "mpas_atm", "grid_type": "icosahedral", "platform": "gpu",
           "precision": "float32", "subdivision": 9, "nlev": 40, "n_devices": 1,
           "valid": True, "metadata": {}}
    _write(tmp_path / "b", "atm_d1", {**atm, "steady_median_ms": 100.0})
    _write(tmp_path / "c", "atm_d1", {**atm, "steady_median_ms": 200.0})
    assert _main(tmp_path, nlev="40") == 1     # compared, and slower
    assert _main(tmp_path, nlev="26") == 2     # refused on both sides


@pytest.mark.parametrize("tol", ["nan", "inf", "-1"])
def test_tolerance_must_be_finite_nonnegative(tmp_path, tol):
    _pair(tmp_path, 100.0, 500.0)
    with pytest.raises(SystemExit):
        _main(tmp_path, tol=tol)


def test_tolerance_is_required(tmp_path):
    with pytest.raises(SystemExit):
        csr.main(["--baseline", "x", "--candidate", "y", "--atm-nlev", "40"])


def test_step_count_from_metadata_is_compared(tmp_path):
    _write(tmp_path / "b", "tri_d8", _tri(8, 100.0, metadata={"extra": {"steps": 100}}))
    _write(tmp_path / "c", "tri_d8", _tri(8, 100.0, metadata={"extra": {"steps": 200}}))
    assert _main(tmp_path) == 3


def test_tied_timing_with_other_size_in_same_file_is_ambiguous(tmp_path):
    """A receipt the loader REFUSED (valid=false) shares the kept row's timing
    and carries the baseline's mesh; the kept row's mesh changed. Matching on
    timing alone would read the refused row and pass."""
    _write(tmp_path / "b", "tri_weak_d8", _tri(8, 100.0, mode="weak"))
    c = tmp_path / "c"; c.mkdir()
    (c / "tri_weak_d8.jsonl").write_text(
        json.dumps(_tri(8, 100.0, mode="weak", valid=False)) + "\n"
        + json.dumps(_tri(8, 100.0, mode="weak", n_lat=3075)) + "\n")
    assert _main(tmp_path) == 3


def test_refused_row_without_device_count_does_not_crash(tmp_path):
    _write(tmp_path / "b", "tri_d8", _tri(8, 100.0))
    c = tmp_path / "c"; c.mkdir()
    (c / "tri_d8.jsonl").write_text(
        json.dumps(_tri(8, 100.0, n_devices=None)) + "\n"
        + json.dumps(_tri(8, 100.0)) + "\n")
    assert _main(tmp_path) == 0
