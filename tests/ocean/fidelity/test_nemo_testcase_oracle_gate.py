"""Direct controls for the fail-closed NEMO testcase oracle gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4
import pytest

GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_oracle_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_oracle_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("token", ["T", ".T.", "true", ".TRUE."])
def test_logical_parser_accepts_only_true_spellings(token: str) -> None:
    assert gate.parse_logical(token) is True


@pytest.mark.parametrize("token", ["F", ".F.", "false", ".FALSE."])
def test_logical_parser_accepts_only_false_spellings(token: str) -> None:
    assert gate.parse_logical(token) is False


@pytest.mark.parametrize("token", ["", "0", "yes", "truth", ".true.false."])
def test_logical_parser_rejects_everything_else(token: str) -> None:
    with pytest.raises(gate.GateError, match="invalid logical"):
        gate.parse_logical(token)


def test_planted_unaccounted_array_is_file_side_and_missing(tmp_path: Path) -> None:
    with netCDF4.Dataset(tmp_path / "mesh_mask.nc", "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("e1t", "f8", ("x",))[:] = 500.0
    with netCDF4.Dataset(tmp_path / "case_restart.nc", "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("state", "f8", ("x",))[:] = 1.0
    (tmp_path / "output.namelist.dyn").write_text("&namrun\n nn_itend = 1\n/\n")
    manifest = gate.disposition_template(tmp_path)

    with pytest.raises(
        gate.GateError, match=r"missing=\['PLANTED_UNACCOUNTED_FILE_ARRAY'\].*extra=\[\]"
    ):
        gate.check_manifest(tmp_path, manifest, plant_unaccounted=True)


def test_planted_geometry_perturbation_fires(tmp_path: Path) -> None:
    with netCDF4.Dataset(tmp_path / "mesh_mask.nc", "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("x", 130)
        ds.createDimension("y", 3)
        ds.createDimension("nav_lev", 21)
        ds.createVariable("e1t", "f8", ("t", "y", "x"))[:] = 500.0

    with pytest.raises(gate.GateError, match="e1t metric"):
        gate.geometry(tmp_path, "lock_exchange", "zco", plant=True)
