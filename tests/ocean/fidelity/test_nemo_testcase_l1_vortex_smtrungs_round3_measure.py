"""SMT-RUNGS round 3: the measurement's statistic and its input-identity gate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "scripts/validate/ocean_fidelity/testcases"))
import nemo_testcase_l1_vortex_smtrungs_round3_smt5_measure as M  # noqa: E402


def _cube():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(4, 5, 3))
    return ref, np.ones_like(ref, dtype=bool)


def test_identical_fields_have_no_unequal_cell():
    ref, mask = _cube()
    s = M.field_stats(ref, ref.copy(), mask)
    assert (s["n_unequal"], s["first_unequal_cell"], s["max_abs"]) == (0, None, 0.0)


def test_one_ulp_plant_is_found_at_its_cell_and_only_there():
    ref, mask = _cube()
    cand = ref.copy()
    cand[2, 3, 1] = np.nextafter(cand[2, 3, 1], np.inf)
    s = M.field_stats(ref, cand, mask)
    assert s["n_unequal"] == 1 and s["first_unequal_cell"] == [2, 3, 1]
    assert s["max_abs"] == abs(np.spacing(ref[2, 3, 1]))


def test_masked_out_difference_is_invisible_and_first_cell_is_c_order():
    ref, mask = _cube()
    cand = ref.copy()
    cand[0, 0, 0] += 1.0
    cand[3, 4, 2] += 1.0
    cand[1, 1, 1] += 1.0
    mask[0, 0, 0] = False
    s = M.field_stats(ref, cand, mask)
    assert s["n_unequal"] == 2 and s["first_unequal_cell"] == [1, 1, 1]
    assert s["rms"] == pytest.approx(np.sqrt(2.0 / (mask.sum())))


def test_nonfinite_candidate_and_shape_mismatch_refuse():
    ref, mask = _cube()
    bad = ref.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(Exception, match="nonfinite"):
        M.field_stats(ref, bad, mask)
    with pytest.raises(Exception, match="shape"):
        M.field_stats(ref, ref[:2], mask)


@pytest.mark.skipif(not (M.ROUND2 / "kt1_10" / "resto.nc").is_file()
                    or not M.SMT4_MESH.is_file(),
                    reason="SMT-5 acquisition evidence not present")
def test_smt5_inputs_are_nemos_own_and_mesh_equals_smt4():
    rows = M.inputs_section()
    assert rows["status"] == "IDENTICAL", rows
    assert rows["mesh_mask_smt5_vs_smt4"]["n_variables"] > 30
