"""Direct tests for the SWG zero-ULP mesh gate and its plants."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg_geometry_gate.py")
_S = importlib.util.spec_from_file_location("swg_geometry_gate", _P)
gate = importlib.util.module_from_spec(_S); _S.loader.exec_module(gate)


@pytest.mark.skipif(not gate.DEFAULT_MESH.is_file(), reason="SWG probe not mounted")
def test_real_probe_geometry_is_bit_identical():
    out = gate.compare(gate.DEFAULT_MESH, allow_dirty=True)
    assert out["all_bit_identical"], [r for r in out["rows"] if not r["bit_identical"]]
    assert len(out["rows"]) == 34 and out["executed_levels"] == 1


@pytest.mark.skipif(not gate.DEFAULT_MESH.is_file(), reason="SWG probe not mounted")
@pytest.mark.parametrize("plant", gate.PLANTS)
def test_each_registered_plant_turns_the_gate_red(plant):
    out = gate.compare(gate.DEFAULT_MESH, plant, allow_dirty=True)
    assert not out["all_bit_identical"]


def test_unknown_plant_is_refused():
    with pytest.raises(Exception):
        gate.compare(gate.DEFAULT_MESH, "unknown", allow_dirty=True)
