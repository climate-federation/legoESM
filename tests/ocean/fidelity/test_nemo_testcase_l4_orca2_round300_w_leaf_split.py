"""Known-answer controls for round 300's W-leaf classifier."""

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round300_w_leaf_split as gate,
)


def _score(value):
    row = {"max_abs": value, "unequal": int(value != 0.0)}
    return {"complete": dict(row), "interior": dict(row), "fold_band": dict(row)}


def _label(name):
    rows = {leaf: _score(0.0) for leaf in gate.SOURCE_ORDER}
    rows["u_difference"] = _score(1.0)
    rows["v_difference"] = _score(2.0)
    return {
        "status": "PASS_R300_LABEL_MEASUREMENT", "label": name,
        "passivity": {"T": True, "S": True}, "rows": rows,
        "candidate_replay": {"ww": _score(0.0), "trp_w": _score(0.0)},
        "oracle_replay": {"ww": _score(0.0), "trp_w": _score(0.0)},
        "cumulative_arms": {
            "trp_u": _score(1.0),
            "trp_u+trp_v": _score(0.0),
        },
        "one_bit_control": {"unequal": 1},
    }


def _report():
    return {
        "admission": {"status": "PASS_R299_OMT4_KT7_W_ADMISSION"},
        "source_order": list(gate.SOURCE_ORDER),
        "labels": [_label(name) for name in gate.LABELS],
    }


def test_classifier_names_u_difference_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["classification"]["first_nonbit_statement"] == "u_difference"
    assert result["classification"]["prediction_ledger"]["R300-P3"] == "CONFIRMED"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(copy.deepcopy(_report()), plant)


def _record(shape=(4, 5, 31), seed=300):
    """Non-zero NEMO-shaped operands with dry cells and a dry bottom level."""
    rng = np.random.default_rng(seed)
    tmask = (rng.random(shape) > 0.2).astype(np.float64)
    tmask[..., -1] = 0.0
    return {
        "trp_u": rng.normal(0.0, 1e6, shape),
        "trp_v": rng.normal(0.0, 1e6, shape),
        "w_r1area": 1.0 / rng.uniform(1e9, 3e10, shape[:2]),
        "w_area": rng.uniform(1e9, 3e10, shape[:2]),
        "w_e3t0": rng.uniform(10.0, 500.0, shape),
        "w_r3bb": rng.normal(0.0, 1e-3, shape[:2]),
        "w_r3mm": rng.normal(0.0, 1e-3, shape[:2]),
        "w_r3aa": rng.normal(0.0, 1e-3, shape[:2]),
        "w_tmask": tmask,
    }


def _reference_w(x):
    """NumPy NEMO W in written order: divhor.f90:132-139, sshwzv.f90:271-298."""
    pu, pv, tm = x["trp_u"], x["trp_v"], x["w_tmask"]
    west = np.empty_like(pu)
    west[:, 1:], west[:, 0] = pu[:, :-1], pu[:, -1]
    south = np.zeros_like(pv)
    south[1:] = pv[:-1]
    live = x["w_e3t0"] * (1.0 + x["w_r3mm"][..., None] * tm)
    numerator = (pu - west) + (pv - south)
    tdiv = (numerator * x["w_r1area"][..., None]) * tm
    flux_div = (live * (tdiv / np.where(tm > 0.5, live, 1.0))) * tm
    stretch = ((1.0 / x["w_dt"]) * x["w_e3t0"]) * (
        x["w_r3aa"] - x["w_r3bb"])[..., None]
    ww = np.zeros(pu.shape[:2] + (pu.shape[-1] + 1,))
    for k in range(pu.shape[-1] - 1, -1, -1):
        ww[..., k] = ww[..., k + 1] - (flux_div[..., k] + stretch[..., k]) * tm[..., k]
    return ww, stretch


def test_oracle_program_replays_nemo_w_bit_for_bit_at_the_stage_clock():
    dt = 10800.0
    inputs = gate._oracle_inputs(_record(), dt)
    result = gate._program(inputs)
    reference = dict(inputs, w_dt=dt / 3.0)
    ww, stretch = _reference_w(reference)
    assert np.count_nonzero(stretch) > 0 and np.count_nonzero(ww) > 0
    assert np.array_equal(result["stretch_rate"], stretch)
    assert np.array_equal(result["w_recurrence"], ww)
    assert np.array_equal(result["bottom_zero"], np.zeros(ww.shape[:2]))
    assert np.array_equal(result["area_product"],
                          reference["w_area"][..., None] * ww)


def test_ratio_delta_row_compares_matched_time_levels():
    rng = np.random.default_rng(7)
    stage_delta = rng.normal(0.0, 1e-3, (3, 4))
    full_step_delta = stage_delta * 3.0
    matched = gate.time_matched_ratio_delta(stage_delta, 10800.0, 3600.0)
    assert np.array_equal(matched, full_step_delta)
    assert not np.array_equal(stage_delta, full_step_delta)


def test_cumulative_walk_is_monotone_in_recorded_operand_sets():
    previous = set()
    for arm in gate.CUMULATIVE_ARMS:
        current = set(arm)
        assert previous < current
        previous = current
