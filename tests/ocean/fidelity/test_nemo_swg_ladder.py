"""SWG restart/stage reader and fail-closed ladder tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg_ladder.py")
_S = importlib.util.spec_from_file_location("swg_ladder", _P)
lad = importlib.util.module_from_spec(_S); _S.loader.exec_module(lad)
ROOT = lad.DEFAULT_ROOT


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_restart_reader_admits_exactly_ten_after_states():
    out = lad.read_restarts(ROOT)
    assert list(out) == list(range(1, 11))
    assert set(out[1]) == set(lad.RESTART_FIELDS)
    assert all(a.shape == (84, 84) for a in out[1].values())


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_reader_self_check_is_exact_and_score_plant_leaves_bar():
    clean = lad.self_check(ROOT, allow_dirty=True)
    plant = lad.self_check(ROOT, plant=True, allow_dirty=True)
    assert clean["all_at_bar"] and clean["all_bit_identical"]
    assert not plant["all_at_bar"] and not plant["all_bit_identical"]


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_real_independent_ladder_refuses_unmeasured_physics():
    with pytest.raises(ValueError, match="ENS total-vorticity"):
        lad.independent(ROOT, allow_dirty=True)


def test_stage_entry_reader_refuses_absent_acquisition(tmp_path):
    with pytest.raises(Exception, match="missing"):
        lad.read_stage_entries(tmp_path)
