"""Unit tests for the MITgcm offline-reference loader."""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_io, mitgcm_runner
from legoesm.ocean.fidelity.mitgcm_runner import (
    MitgcmReferenceError,
    load_mitgcm_reference,
)


def _write_gyre_reference(case_dir, *, iters=(10,), nx=8, ny=6):
    """Write a synthetic barotropic-gyre reference archive (U, V, Eta + grid)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for it in iters:
        for field, val in (("U", 0.1), ("V", -0.2), ("Eta", 0.01 * it)):
            arr = np.full((ny, nx), val, dtype=np.float32)
            mitgcm_io.write_mds(case_dir / field, arr, iteration=it, time_step=it)
    # Grid descriptors (cell centres, depth).
    mitgcm_io.write_mds(case_dir / "XC", np.zeros((ny, nx), np.float32))
    mitgcm_io.write_mds(case_dir / "YC", np.zeros((ny, nx), np.float32))
    mitgcm_io.write_mds(case_dir / "Depth", np.full((ny, nx), 5000.0, np.float32))
    return case_dir


def test_available_cases_includes_gyre():
    assert "barotropic_gyre" in mitgcm_runner.available_cases()


def test_load_latest_iteration(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(10,))
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    assert set(res.variables) == {"U", "V", "Eta"}
    assert res.variables["U"].shape == (6, 8)
    np.testing.assert_array_equal(res.iters, [10])
    # times_s = iter * deltaT (default 1200 s for the gyre).
    np.testing.assert_allclose(res.times_s, [10 * 1200.0])
    assert res.grid_metadata["nx"] == 8
    assert res.grid_metadata["ny"] == 6
    assert res.grid_metadata["nz"] == 1
    assert res.grid_metadata["delta_t_s"] == 1200.0


def test_iteration_discovery_picks_last(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(10, 20, 30))
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    np.testing.assert_array_equal(res.iters, [30])
    # Eta value encodes the iteration (0.01 * it) -> confirms the LAST was read.
    np.testing.assert_allclose(res.variables["Eta"], 0.01 * 30, rtol=1e-6)


def test_explicit_multi_iteration_stacks_time_axis(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(10, 20))
    res = load_mitgcm_reference(
        "barotropic_gyre", ref_dir=case_dir, iterations=(10, 20)
    )
    assert res.variables["U"].shape == (2, 6, 8)  # leading time axis
    np.testing.assert_array_equal(res.iters, [10, 20])
    np.testing.assert_allclose(res.times_s, [10 * 1200.0, 20 * 1200.0])


def test_grid_metadata_loaded(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre")
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    assert "XC" in res.grid_metadata and "Depth" in res.grid_metadata
    np.testing.assert_allclose(res.grid_metadata["Depth"], 5000.0)


def test_delta_t_override(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(5,))
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir, delta_t_s=600.0)
    np.testing.assert_allclose(res.times_s, [5 * 600.0])


def test_unknown_case_raises():
    with pytest.raises(ValueError, match="Unknown MITgcm case"):
        load_mitgcm_reference("not_a_case")


def test_missing_reference_dir_raises(tmp_path):
    with pytest.raises(MitgcmReferenceError, match="not found"):
        load_mitgcm_reference("barotropic_gyre", ref_dir=tmp_path / "nope")


def test_empty_reference_dir_raises(tmp_path):
    empty = tmp_path / "barotropic_gyre"
    empty.mkdir()
    with pytest.raises(MitgcmReferenceError, match="no iteration dumps"):
        load_mitgcm_reference("barotropic_gyre", ref_dir=empty)


def test_provenance_recorded(tmp_path):
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(10,))
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    prov = res.provenance
    assert prov["case_name"] == "barotropic_gyre"
    assert prov["reader"] == "mitgcm_io.read_mds"
    assert prov["iterations"] == [10]
    assert "U.0000000010.meta" in prov["source_mtimes"]


def test_nz_derived_from_deepest_rank_field(tmp_path):
    """A 2-D field listed first must not force nz=1 when a sibling is 3-D."""
    case_dir = tmp_path / "barotropic_gyre"
    case_dir.mkdir()
    # U/V are 3-D (nz=3); Eta is 2-D. Max-rank field carries the true nz.
    for field in ("U", "V"):
        mitgcm_io.write_mds(
            case_dir / field, np.zeros((3, 6, 8), np.float32), iteration=10
        )
    mitgcm_io.write_mds(case_dir / "Eta", np.zeros((6, 8), np.float32), iteration=10)
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    assert res.grid_metadata["nz"] == 3
    assert res.grid_metadata["nx"] == 8 and res.grid_metadata["ny"] == 6


def test_nz_from_rc_descriptor_is_authoritative(tmp_path):
    """When RC (vertical centres) is present it sets nz, even with 2-D fields."""
    case_dir = _write_gyre_reference(tmp_path / "barotropic_gyre", iters=(10,))
    mitgcm_io.write_mds(case_dir / "RC", np.linspace(-2.5, -100.0, 5, dtype=np.float32))
    res = load_mitgcm_reference("barotropic_gyre", ref_dir=case_dir)
    assert res.grid_metadata["nz"] == 5
