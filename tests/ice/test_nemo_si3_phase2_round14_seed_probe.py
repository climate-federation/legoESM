"""Unit controls for the Round-14 carried-seed producer probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROBE_PATH = Path(__file__).parents[2] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_round14_seed_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_round14_seed_probe_test", PROBE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _row(name: str, nonzero: int, status: str) -> dict[str, object]:
    return {
        "name": name,
        "bitwise_nonzero_over_n": f"{nonzero} / 4",
        "status": status,
    }


def test_seed_probe_plant_requires_named_row_transition():
    assert probe._assert_plant(
        _row("entry.step9.a_i", 0, "BIT-EXACT"),
        _row("entry.step9.a_i", 1, "DEBT"),
    )
    with pytest.raises(probe.SeedProbeError, match="numerator"):
        probe._assert_plant(
            _row("entry.step9.a_i", 0, "BIT-EXACT"),
            _row("entry.step9.a_i", 0, "DEBT"),
        )


def test_source_written_stress_arm_changes_nontrivial_state():
    # The full oracle arm is intentionally external-data backed; this unit test
    # only prevents the source-written private discriminator from being a no-op.
    shape = (probe.recipe._ICE_RHEO_ALLOCATED_SIZE,) * 2
    zero = np.zeros(shape, dtype=np.float64)
    state = type("State", (), {})()
    state.dynamics = type("Dynamics", (), {})()
    state.dynamics.u_ice_u = zero
    state.dynamics.v_ice_v = zero
    stress_u, stress_v = probe._source_written_air_stress(state, probe.ENTRY_STEP)
    assert stress_u.dtype == stress_v.dtype == np.float64
    assert float(np.max(np.abs(stress_u))) > 0.0
    assert float(np.max(np.abs(stress_v))) > 0.0
