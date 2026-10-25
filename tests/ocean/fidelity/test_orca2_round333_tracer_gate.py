"""Controls for round 333's kt=2 tracer acquisition admission."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round333_tracer_gate as gate,
)

ROOT = Path(__file__).parents[3]
ACQ = ROOT / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition"


def test_rk3_levels_are_derived_from_nemo_initialization_and_swaps() -> None:
    assert gate.rk3_levels_for_step_stage(1, 1) == (1, 1, 3, 3)
    assert gate.rk3_levels_for_step_stage(1, 2) == (1, 3, 2, 2)
    assert gate.rk3_levels_for_step_stage(1, 3) == (1, 2, 3, 3)
    assert gate.rk3_levels_for_step_stage(2, 1) == (3, 3, 1, 1)
    assert gate.rk3_levels_for_step_stage(2, 2) == (3, 1, 2, 2)
    assert gate.rk3_levels_for_step_stage(2, 3) == (3, 2, 1, 1)
    with pytest.raises(gate.GateError, match="invalid RK3"):
        gate.rk3_levels_for_step_stage(0, 3)


def test_comparison_counts_signed_zero_bits() -> None:
    reference = np.array([0.0, 2.0], dtype=np.float64)
    candidate = np.array([-0.0, np.nextafter(2.0, np.inf)], dtype=np.float64)
    row = gate._comparison(reference, candidate)
    assert row["unequal"] == 2
    assert row["max_abs"] > 0.0


def test_schema_has_post_damping_boundary_and_checksum_plants() -> None:
    assert "after_damping_T" in gate.FIELDS
    assert "after_damping_S" in gate.FIELDS
    assert {"checksum", "truncation", "time-level", "zero", "kmm"} <= set(gate.PLANTS)


def test_stage_patch_has_context_raw_arrays_and_every_rank() -> None:
    patch = (ACQ / "stprk3_stg_round333.patch").read_text()
    assert not re.search(r"^@@ -\d+,0 ", patch, re.MULTILINE)
    assert patch.count("CALL r333_tracer_open") == 1
    assert patch.count("CALL r333_tracer_rhs") == 5
    assert patch.count("CALL r333_tracer_finish") == 1
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+"))
    assert "l4_canon_2d" not in added
    assert "l4_canon_3d" not in added
    assert "lwp .AND. .NOT.ln_tile .AND. kstp == nit000 + 1" not in added


def test_writer_uses_declared_extents_per_rank_and_checksums() -> None:
    patch = (ACQ / "round332_writer_round333.patch").read_text()
    assert "DIMENSION(jpi,jpj,jpk)" in patch
    assert "DIMENSION(jpi,jpj)" in patch
    assert "checksum3(value)" in patch
    assert "checksum2(value)" in patch
    assert '"_r",I4.4' in patch
    assert "narea-1" in patch
    assert "SHAPE(" not in patch


def test_driver_has_placement_plants_fixture_and_pinned_admission() -> None:
    text = (ACQ / "run.sh").read_text()
    assert "check_r333_stage_landing" in text
    for label in ("open", "advection", "sbc", "qsr", "ldf", "damping", "finish"):
        assert f"r333-{label}-misplaced" in text
    assert "-fcheck=bounds" in text
    assert "round333_writer_fixture" in text
    assert "producer_commit.txt" in text
    assert "nemo.sha256" in text
    assert "admit-r333-" in text
    assert "check_tracer_selectors" in text
    assert "r333-${selector}-off" in text
    for selector in ("ln_trabbc", "ln_trabbl", "ln_tradmp"):
        assert selector in text
