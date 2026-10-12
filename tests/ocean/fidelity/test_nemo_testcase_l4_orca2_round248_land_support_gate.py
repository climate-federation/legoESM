from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round248_land_support_gate as gate,
)


def _finite():
    return {
        "shape": [1], "nonfinite": 0, "max_abs_finite": 1.0,
        "argmax_abs_finite": [0], "first_nonfinite": None,
    }


def _report():
    entry = {name: _finite() for name in (
        "T", "S", "u", "v", "eta", "uu_b", "vv_b")}
    geometry = {name: _finite() for name in (
        "jacobian", "live_t_thickness", "rho_prime", "pressure_prime",
        "nemo_hpg_rhd")}
    mask = {
        "support": 148 * 180, "unequal": 0, "first_unequal": None,
        "candidate_wet_oracle_dry": 0, "candidate_dry_oracle_wet": 0,
    }
    rows = {
        "u_mask_source": copy.deepcopy(mask),
        "v_mask_source": copy.deepcopy(mask),
    }
    for name in (
        "candidate_r1_hu_0", "candidate_r1_hv_0",
        "candidate_live_r1_hu", "candidate_live_r1_hv",
    ):
        rows[name] = _finite()
    substeps = {
        "r1_face_depth_u_exit": {
            **_finite(), "shape": [65, 148, 181], "nonfinite": 1,
            "first_nonfinite": [1, 7, 114],
        },
    }
    one = {
        "status": "PASS_R248_LAND_SUPPORT_MEASURED",
        "observer": "offline from ordinary completed kt=5 entry",
        "entry_fields": entry,
        "geometry_density": geometry,
        "slow_forcing": {"du_dt": _finite(), "dv_dt": _finite()},
        "barotropic_substeps": substeps,
        "state_after_barotropic": {
            "eta": {**_finite(), "nonfinite": 1, "first_nonfinite": [0, 0]},
        },
        "source_order": [
            "hu_0/hv_0", "ssumask/ssvmask", "r1_hu_0/r1_hv_0",
            "live r1_hu/r1_hv", "zub/zvb", "stage transport",
            "continuity/W", "stage state",
        ],
        "rows": rows,
        "guard_cell": {
            "compact_ji": [1, 49], "hv_0_m": 0.0,
            "resolved_ssvmask": 0.0,
        },
        "first_writer_operands": {
            "substep": 2, "compact_ji": [7, 114], "ssumask": 1.0,
            "face_depth_u_exit_m": 0.0,
            "r1_face_depth_u_exit_per_m": float("inf"),
            "previous_r1_face_depth_u_exit_per_m": 0.0018,
        },
    }
    return {label: copy.deepcopy(one) for label in gate.LABELS}


def test_classifier_names_the_substep_two_exit_reciprocal():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R248_RECIPROCAL_OWNER_REFUTED"
    assert all(
        row["first_nonfinite_writer"]
        == "barotropic_substep_2.r1_face_depth_u_exit"
        for row in result["summary"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round248_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
