import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round22_exact_transition_gate as gate,
)


def _row(value=0.0):
    return {
        "bit_identical": value == 0.0,
        "count": 1,
        "first_unequal_index": None if value == 0.0 else [0],
        "max_abs": value,
        "mean_abs_over_unequal": value,
        "unequal": int(value != 0.0),
    }


def _ladder(stage1_u=0.0):
    checkpoints = []
    for kt in range(1, 11):
        for checkpoint in ("entry", "stage1", "stage2", "stage3"):
            rows = {field: _row() for field in gate.FIELDS}
            if kt == 1 and checkpoint == "stage1":
                rows["u"] = _row(stage1_u)
            checkpoints.append({"kt": kt, "checkpoint": checkpoint, "rows": rows})
    return {"candidate_trajectory": {"checkpoints": checkpoints}}


def _capture(stage1_u=0.0):
    rows = {
        checkpoint: {field: _row() for field in gate.FIELDS}
        for checkpoint in gate.CHECKPOINTS
    }
    rows["stage1"]["u"] = _row(stage1_u)
    return {
        "candidate_rows": rows,
        "oracle_stage3_to_kt2_entry": {
            field: _row() for field in gate.FIELDS},
    }


def _write(path, values):
    np.savez(path, **values)


def test_analyze_recertifies_only_when_third_owner_restores_every_array(tmp_path):
    keys = [
        f"candidate_{checkpoint}_{field}"
        for checkpoint in gate.CHECKPOINTS for field in gate.FIELDS
    ]
    old = {key: np.zeros(1, dtype=np.float64) for key in keys}
    combined = {key: value.copy() for key, value in old.items()}
    combined["candidate_stage1_u"][0] = 1.0
    restored = {key: value.copy() for key, value in old.items()}
    plant = {key: value.copy() for key, value in restored.items()}
    plant["candidate_returned_u"][0] = np.nextafter(0.0, np.inf)
    paths = {name: tmp_path / f"{name}.npz" for name in (
        "old", "combined", "restored", "plant")}
    for name, values in (
        ("old", old), ("combined", combined),
        ("restored", restored), ("plant", plant)):
        _write(paths[name], values)

    result = gate.analyze(
        paths["old"], paths["combined"], paths["restored"], paths["plant"],
        _capture(), _capture(1.0), _capture(),
        _ladder(), _ladder(1.0), _ladder())

    assert result["status"] == "RECERTIFIED"
    assert result["first_exact_combined_difference"]["key"] == (
        "candidate_stage1_u")
    assert result["legacy_ldf_exact_arrays_different_from_old"] == 0
    assert result["plant_cells"] == 1
    assert set(result["predictions"].values()) == {"CONFIRMED"}
