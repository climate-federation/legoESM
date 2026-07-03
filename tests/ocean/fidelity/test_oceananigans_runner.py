"""Direct tests for the Oceananigans offline-reference loader.

Writes a synthetic ``<case>/<case>.nc`` archive and loads it through
:func:`load_oceananigans_reference`, plus the dispatch-hardening guards
(unknown case raises; missing dir / missing file raise) mirrored from the
MITgcm runner tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.fidelity.oceananigans_runner import (
    KNOWN_CASES,
    OceananigansReferenceError,
    OceananigansResult,
    available_cases,
    load_oceananigans_reference,
)


def _write_case_archive(root, case_name, *, with_time=True):
    """Write <root>/<case>/<case>.nc with the case's prognostic fields."""
    xr = pytest.importorskip("xarray")
    spec = KNOWN_CASES[case_name]
    nt, nz, ny, nx = 3, 2, 4, 5
    zC = np.linspace(-4000.0, -100.0, nz)
    yC = np.linspace(15.5, 74.5, ny)
    xC = np.linspace(-29.5, 29.5, nx)
    rng = np.random.default_rng(1)
    data_vars = {}
    for f in spec.prognostic_fields:
        if f == "eta":
            data_vars[f] = (("time", "yC", "xC"),
                            rng.standard_normal((nt, ny, nx)))
        else:
            data_vars[f] = (("time", "zC", "yC", "xC"),
                            rng.standard_normal((nt, nz, ny, nx)))
    coords = {"zC": zC, "yC": yC, "xC": xC}
    if with_time:
        coords["time"] = np.array([0.0, 1800.0, 3600.0])
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    case_dir = root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(case_dir / f"{case_name}.nc")
    return dict(nt=nt, nz=nz, ny=ny, nx=nx)


def test_load_round_trip(tmp_path):
    pytest.importorskip("xarray")
    dims = _write_case_archive(tmp_path, "barotropic_gyre")
    out = load_oceananigans_reference("barotropic_gyre", ref_dir=tmp_path / "barotropic_gyre")
    assert isinstance(out, OceananigansResult)
    assert out.case_name == "barotropic_gyre"
    assert set(out.variables) == set(KNOWN_CASES["barotropic_gyre"].prognostic_fields)
    np.testing.assert_allclose(out.times_s, [0.0, 1800.0, 3600.0])
    assert out.grid_metadata["nx"] == dims["nx"]
    assert out.grid_metadata["ny"] == dims["ny"]
    assert out.grid_metadata["nz"] == dims["nz"]
    assert out.grid_metadata["cyclic_x"] is False


def test_synthesised_times_when_no_time_coord(tmp_path):
    pytest.importorskip("xarray")
    _write_case_archive(tmp_path, "bickley_jet", with_time=False)
    out = load_oceananigans_reference(
        "bickley_jet", ref_dir=tmp_path / "bickley_jet", delta_t_s=10.0)
    # 3 snapshots, dt=10 -> [0, 10, 20]
    np.testing.assert_allclose(out.times_s, [0.0, 10.0, 20.0])
    assert out.grid_metadata["cyclic_x"] is True


def test_unknown_case_raises():
    with pytest.raises(ValueError, match="Unknown Oceananigans case"):
        load_oceananigans_reference("not_a_case")


def test_missing_dir_raises(tmp_path):
    with pytest.raises(OceananigansReferenceError, match="not found"):
        load_oceananigans_reference("barotropic_gyre", ref_dir=tmp_path / "absent")


def test_missing_nc_raises(tmp_path):
    case_dir = tmp_path / "barotropic_gyre"
    case_dir.mkdir()
    with pytest.raises(OceananigansReferenceError, match="No NetCDF"):
        load_oceananigans_reference("barotropic_gyre", ref_dir=case_dir)


def test_available_cases_includes_registered():
    cases = available_cases()
    assert "barotropic_gyre" in cases
    assert "bickley_jet" in cases
