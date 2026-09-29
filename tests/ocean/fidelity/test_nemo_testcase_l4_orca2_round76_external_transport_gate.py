from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round76_external_transport_gate as gate,
)


def _row(*, exact: bool, rms: float = 0.0) -> dict[str, object]:
    return {
        "count": 10,
        "max_abs": 0.0 if exact else 1.0,
        "max_ulp": 0 if exact else 1,
        "rms_abs": rms,
        "status": "AT_BAR" if exact else "DEBT",
        "unequal": 0 if exact else 1,
    }


def _report() -> dict[str, object]:
    live_rows = []
    for substep in range(1, gate.N_CYCLE + 1):
        live_rows.append({
            "substep": substep,
            "sum_entry": _row(exact=True),
            "weight": _row(exact=True),
            "metric_transport": _row(exact=substep != 2),
            "r1_e2u": _row(exact=True),
            "sum_exit": _row(exact=substep == 1),
        })
    exact_rows = [_row(exact=True) for _ in range(gate.N_CYCLE)]
    arms = {}
    for name, sources in gate.GEOMETRY_ARM_SOURCES.items():
        pair = name == "oracle_pair"
        rms = {
            "live_pair": gate.EXPECTED_ROUND75["live_pair_rms"],
            "oracle_inverse_only": gate.EXPECTED_ROUND75["oracle_inverse_only_rms"],
            "oracle_thickness_only": 0.8,
            "oracle_pair": 0.0,
        }[name]
        arms[name] = {
            "input_sources": list(sources),
            "rows": {
                row: _row(exact=pair, rms=0.0 if pair else rms)
                for row in gate.round75.ROW_ORDER
            },
        }
    return {
        "claim_label": "independent",
        "execution": {
            "backend": "cpu", "production_jit": True, "dtype": "float64",
            "transcendentals": "libm",
        },
        "card_scope": {
            name: list(values)
            for name, values in gate.round74.EXPECTED_CARD_SCOPE.items()
        },
        "round75_status": "PASS_HYBRID_CORRECTION_PAIR",
        "round75_observed": copy.deepcopy(gate.EXPECTED_ROUND75),
        "support": {"active_u_columns": 8568, "active_u_3d": 226236},
        "zero_seed": _row(exact=True),
        "live_rows": live_rows,
        "first_non_bit_u_accumulator_statement": {
            "substep": 2, "boundary": "metric_transport",
            **_row(exact=False),
        },
        "production_trace_endpoint_identity": _row(exact=True),
        "record_replay": {
            "metric_u": copy.deepcopy(exact_rows),
            "sum_u_exit": copy.deepcopy(exact_rows),
            "pre_lbc_normalized_u": _row(exact=True),
            "all_metric_exact": True,
            "all_exit_exact": True,
        },
        "geometry_arms": arms,
    }


def test_classifier_accepts_external_walk_and_geometry_pair():
    result = gate.classify(_report())
    assert result["status"] == "PASS_EXTERNAL_TRANSPORT_WALK"
    assert all(row["status"] == "CONFIRMED"
               for row in result["prediction_ledger"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_thickness_only_arm_must_worsen_live_pair():
    report = _report()
    report["geometry_arms"]["oracle_thickness_only"]["rows"]["zFu"][
        "rms_abs"] = gate.EXPECTED_ROUND75["live_pair_rms"]
    with pytest.raises(gate.GateError, match="thickness alone"):
        gate.classify(report)
