from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round80_extremes_gate as gate,
)


def _comparison(tmax: float = 1.2367457128331782,
                smax: float = 0.2871061346986039) -> dict:
    rows = {
        name: {
            "bit_identical": False,
            "unequal": 1,
            "count": 2,
            "max_abs": tmax if name == "T" else smax if name == "S" else 0.1,
            "mean_abs_over_unequal": 0.1,
            "rms": 0.1,
            "first_unequal_index": [0],
        }
        for name in gate.FIELD_ORDER
    }
    return {"rows": rows, "ranked_non_bit_by_max_abs": [],
            "first_non_bit_field": "T"}


def _report() -> dict:
    fixed = _comparison()
    avt = copy.deepcopy(fixed)
    avm = copy.deepcopy(fixed)
    avt["rows"]["T"]["rms"] = 0.09
    avm["rows"]["u"]["rms"] = 0.09
    return {
        "claim_label": "given NEMO's entry",
        "passive_seam_bit_identical": True,
        "comparisons": {
            "fixed_none": fixed,
            "avt": avt,
            "avm": avm,
        },
    }


def test_classifier_accepts_registered_controls() -> None:
    assert gate.classify(_report())["status"] == "PASS_ROUND80_EXTREMES"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_owned_mapping_removes_the_nemo_halo_and_transposes() -> None:
    import numpy as np

    values = np.arange(94 * 152 * 3).reshape(94, 152, 3)
    owned = gate._owned(values)
    assert owned.shape == (148, 90, 3)
    assert owned[0, 0, 0] == values[2, 2, 0]
    assert owned[-1, -1, -1] == values[-3, -3, -1]
