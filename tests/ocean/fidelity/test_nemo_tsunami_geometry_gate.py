"""TSUNAMI geometry-identity gate: passes on NEMO's real mesh_mask, and is shown
to fail when one element of the card or of the mesh is moved."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
         "testcases/nemo_testcase_l1_tsunami_geometry_gate.py")
_spec = importlib.util.spec_from_file_location("tsunami_geometry_gate", _PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

REAL = gate.DEFAULT_MESH


def _copy_mesh(src: Path, dst: Path, perturb: str | None = None):
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w") as b:
        for n, d in a.dimensions.items():
            b.createDimension(n, len(d))
        for n, v in a.variables.items():
            w = b.createVariable(n, v.dtype, v.dimensions)
            data = np.array(v[:])
            if n == perturb:
                data.flat[0] = data.flat[0] + 1
            w[:] = data


@pytest.mark.skipif(not REAL.is_file(), reason="RK3 evidence not mounted")
def test_card_geometry_equals_the_rk3_mesh_mask_bit_for_bit():
    res = gate.compare(REAL)
    assert res["all_bit_identical"], [r for r in res["rows"] if not r["bit_identical"]]
    assert len(res["rows"]) >= 35 and res["executed_levels"] == 1


@pytest.mark.skipif(not REAL.is_file(), reason="RK3 evidence not mounted")
def test_a_planted_card_element_turns_the_gate_red():
    res = gate.compare(REAL, plant="grid dx_u")
    bad = [r["row"] for r in res["rows"] if not r["bit_identical"]]
    assert bad == ["grid dx_u"] and not res["all_bit_identical"]


@pytest.mark.skipif(not REAL.is_file(), reason="RK3 evidence not mounted")
@pytest.mark.parametrize("var", ["e2f", "glamf", "ff_t", "tmask", "gdept_1d"])
def test_a_perturbed_mesh_element_turns_the_gate_red(tmp_path, var):
    mesh = tmp_path / "mesh_mask.nc"
    _copy_mesh(REAL, mesh, perturb=var)
    res = gate.compare(mesh)
    assert not res["all_bit_identical"]
    assert all(r["nemo_variable"] == var
               for r in res["rows"] if not r["bit_identical"])


def test_unknown_plant_is_refused():
    with pytest.raises(Exception):
        gate.compare(REAL, plant="not a row")
