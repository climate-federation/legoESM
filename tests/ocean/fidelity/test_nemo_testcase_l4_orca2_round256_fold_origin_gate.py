from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round256_fold_origin_gate as gate,
)


def _metric(maximum=0.0, unequal=None):
    if unequal is None:
        unequal = int(maximum != 0.0)
    return {
        "support": 1, "unequal": unequal, "max_abs": maximum,
        "rms": maximum, "argmax": [0], "first_nonfinite": None,
        "nonfinite": 0,
    }


def _fold(name, maximum=0.0, unequal=None):
    return {
        "point_type": gate.r249.POINT_TYPES[name],
        "sign": gate.r249.SIGNS[name], "target_row": 147,
        "target_i_start": 0, "source_row": 146,
        **_metric(maximum, unequal),
    }


def _measurement(label):
    boundaries = []
    for kt in range(1, 8):
        for stage in range(4):
            fields = {}
            for name in gate.FIELDS:
                maximum = (1.0 if kt == 1 and stage == 1 and name == "v"
                           else 0.0)
                fields[name] = {
                    "complete": _metric(maximum),
                    "interior": _metric(0.0),
                    "fold_band": _metric(maximum),
                    "fold_identity": _fold(name, maximum),
                    "nemo_fold_identity": _fold(name),
                }
            boundaries.append({"kt": kt, "stage": stage, "fields": fields})
    row = {
        "raw_vs_nemo": _metric(1.0),
        "associated_vs_nemo": _metric(0.0),
        "association_delta": _metric(1.0),
        "raw_fold_band_vs_nemo": _metric(1.0),
        "associated_fold_band_vs_nemo": _metric(0.0),
        "raw_fold_identity": _fold("v", 1.0),
        "associated_fold_identity": _fold("v"),
        "nemo_fold_identity": _fold("v"),
    }
    rows = {name: copy.deepcopy(row) for name in ("u", "v", "T", "S")}
    for name in rows:
        rows[name]["raw_fold_identity"] = _fold(
            name, 0.0 if name == "u" else 1.0)
        rows[name]["associated_fold_identity"] = _fold(name)
        rows[name]["nemo_fold_identity"] = _fold(name)
    rows["u"]["association_delta"] = _metric(0.0)
    return {
        "status": "PASS_R256_MEASUREMENT", "label": label,
        "stage_growth": {
            "status": "PASS_R249_STAGE_TABLE", "unit": True,
            "operand_route": "resolved-candidate",
            "passivity": [
                {"kt": kt, "fields": {name: True for name in gate.FIELDS}}
                for kt in range(1, 8)
            ],
            "boundaries": boundaries,
        },
        "stage_association": {
            "status": "PASS_R256_STAGE_ASSOCIATION_REPLAY",
            "operand_route": "production-d114-resolved",
            "passivity": {name: True for name in gate.FIELDS},
            "source_order": ["raw_stage_update", "post_stage_lbc_lnk"],
            "rows": rows,
        },
        "one_bit_control": _metric(1.0, unequal=1),
    }


def _reports():
    return [_measurement(label) for label in gate.LABELS]


def test_round256_names_post_stage_association() -> None:
    result = gate.classify(_reports())
    assert result["status"] == "HELD_R256_POST_STAGE_T_S_V_ASSOCIATION_NAMED"
    assert all(row["source_owner"]["statement"]
               == "post_stage_lbc_lnk_T_S_v"
               for row in result["labels"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round256_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_reports(), plant=plant)
