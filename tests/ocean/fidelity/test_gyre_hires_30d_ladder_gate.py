import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_30d_ladder_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_30d_ladder_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_daily_schedule_covers_initial_and_30_complete_days():
    schedule = GATE.day_boundary_schedule()
    assert len(schedule) == 31
    assert schedule[:3] == (1, 25, 49)
    assert schedule[-1] == 721
    assert np.all(np.diff(schedule) == 24)


def test_daily_schedule_refuses_empty_duration():
    with pytest.raises(ValueError, match="positive"):
        GATE.day_boundary_schedule(0)


def test_daily_writer_registers_sparse_levels_beyond_legacy_range_only():
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    GATE.register_daily_entry_time_levels()
    assert time_level_for_dump("oracle_step_entry_kt00000073.bin") == "before"
    assert time_level_for_dump("oracle_step_entry_kt00000721.bin") == "before"
    # The shared registry inherits the older per-step writer's kt=1..60 names.
    # Beyond that legacy range, this registration must not silently admit a
    # neighboring dump the sparse daily writer never emits.
    with pytest.raises(ValueError, match="no registered NEMO time level"):
        time_level_for_dump("oracle_step_entry_kt00000072.bin")
    with pytest.raises(ValueError, match="no registered NEMO time level"):
        time_level_for_dump("oracle_step_entry_kt00000722.bin")


def test_daily_gate_preregisters_exact_bar_and_acquisition_root():
    assert "round6/oracle_gyre_r4_smag_daily_stage/run" in str(
        GATE.DEFAULT_ORACLE_ROOT)
    assert GATE.STEPS_PER_DAY == 24 and GATE.NDAYS == 30


def test_daily_vector_oracle_admission_checks_all_boundaries():
    admission = GATE.admit_daily_oracle(GATE.DEFAULT_ORACLE_ROOT, "vector")
    entries = [name for name in admission["checked_files"]
               if name.startswith("oracle_step_entry_")]
    assert len(entries) == 31
    assert admission["math_flavour"] == "vector"


def test_daily_scalar_oracle_admission_checks_all_boundaries():
    admission = GATE.admit_daily_oracle(
        GATE.DEFAULT_SCALAR_ORACLE_ROOT, "scalar")
    entries = [name for name in admission["checked_files"]
               if name.startswith("oracle_step_entry_")]
    assert len(entries) == 31
    assert admission["math_flavour"] == "scalar"
    assert admission["producer_commit"] == (
        "63cebeb37ea426cd647688f20e0e0250ae194f3f")
