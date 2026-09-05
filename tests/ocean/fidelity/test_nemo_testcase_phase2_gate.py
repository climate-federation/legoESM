"""Direct non-vacuity and registry tests for the phase-2 testcase gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase2_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_phase2_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_phase2_registry_file_side_plant_is_missing(tmp_path: Path):
    mesh_path = tmp_path / "mesh.nc"
    with netCDF4.Dataset(mesh_path, "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("e1t", "f8", ("x",))[:] = 1.0
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "format": "nemo-testcase-l1-phase2-registry-v1",
                "mesh": {
                    "e1t": {"status": "VERIFIED", "reason": "unit control"}
                },
            }
        )
    )
    with netCDF4.Dataset(mesh_path) as ds:
        with pytest.raises(gate.GateError, match=r"missing=.*PLANTED_UNACCOUNTED"):
            gate._load_registry(registry, ds, plant=True)


@pytest.mark.parametrize("name", ["geometry.e1t", "step1.before.T"])
def test_phase2_numeric_plants_turn_gate_red(name: str):
    with pytest.raises(gate.GateError, match=name):
        gate._score(
            [],
            {},
            name,
            np.ones((2,), dtype=np.float64),
            np.ones((2,), dtype=np.float64),
            plant=True,
        )


def test_phase2_unmeasured_does_not_override_measured_status():
    assert gate.UNMEASURED
    assert gate.POINTWISE_BAR == 1.0e-15
    assert gate.ACCUMULATING_BAR == 1.0e-12


def test_testcase_kt1_dump_is_registered_before():
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    assert time_level_for_dump("oracle_step_entry_kt00000001.bin") == "before"


@pytest.mark.parametrize("case", sorted(gate.ROOTS))
def test_kt1_rows_follow_case_alignment_contract(case):
    card = gate._card(case)
    rows, _ = gate.ic_step1_gate(card, gate.ROOTS[case])
    by_name = {row["name"].rsplit(".", 1)[-1]: row for row in rows}
    assert by_name["T"]["bar"] == 0.0
    assert by_name["T"]["status"] == "AT-BAR"
    for name in ("S", "u", "v", "ssh"):
        assert by_name[name]["bar"] == 0.0
        if case == "GYRE-zco":
            if name == "S":
                assert by_name[name]["status"] == "AT-BAR"
            else:
                assert by_name[name]["exact_control_status"] == "AT-BAR"
                assert by_name[name]["status"] == "UNINFORMATIVE"
                assert by_name[name]["reason"]
        else:
            assert by_name[name]["alignment_status"] == "AT-BAR"
            assert by_name[name]["status"] == "UNMEASURED"


def test_gyre_rotated_grid_synthetic_violation_fires():
    registry = GATE_PATH.parent / "manifests/gyre_zco_phase2.json"
    with pytest.raises(gate.GateError, match="geometry.glamt"):
        gate.geometry_gate(
            "GYRE-zco",
            gate.ROOTS["GYRE-zco"],
            registry,
            plant_geometry=True,
        )


def test_gyre_seasonal_forcing_synthetic_violation_fires():
    card = gate._card("GYRE-zco")
    with pytest.raises(gate.GateError, match="forcing.kt1.qsr_w_m2"):
        gate.forcing_gate(card, plant_forcing=True)
