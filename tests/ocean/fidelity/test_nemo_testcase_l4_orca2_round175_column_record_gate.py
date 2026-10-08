from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round175_column_record_gate as gate,
)


def _report():
    neighbours = {"west": 0.0, "east": 1.0, "south": 1.0, "north": 1.0}
    rows = [{"boundary": "entry", "status": "AT_BAR_BIT_EXACT",
             "target_available": True, "max_abs": 0.0}]
    rows.extend({"boundary": name,
                 "status": "UNMEASURED_MISSING_RANK1_STREAM",
                 "target_available": False, "max_abs": None}
                for name in gate.FIELD_ORDER[1:-1])
    rows.append({"boundary": "completed_stage", "status": "DEBT",
                 "target_available": True,
                 "max_abs": 3.2847473521544472,
                 "argmax": [86, 159, 3]})
    return {
        "geometry": {
            "target_jik": [86, 159, 3],
            "dtypes": {"nemo": "float64", "legoesm": "float64"},
            "card_matches_nemo": {name: True for name in (
                "e3t_0", "e3w_0", "tmask", "surface_umask", "surface_vmask")},
            "mbkt_fortran": 4, "bottom_zero_based": 3,
            "bottom_is_partial": True,
            "fold_row": False, "cyclic_seam": False,
            "land_adjacent": True, "neighbour_surface_tmask": neighbours,
            "levels_0_5": {"nemo_tmask": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]},
        },
        "record_census": {
            "rank_tagged_files": 0, "rank_complete": False,
            "covered_global_i": [0, 89], "target_i": 159,
            "target_present": False,
        },
        "operator_table": rows,
        "first_unmeasured": "external_mode_qco",
    }


def test_classifier_accepts_frozen_missing_rank1_boundary():
    result = gate.classify(_report())
    assert result["status"] == "STOP_RECORD_R175_STAGE1_RANK_COMPLETE_NEEDED"
    assert result["predictions"] == {f"R175-P{index}": "CONFIRMED"
                                      for index in range(1, 6)}


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)
