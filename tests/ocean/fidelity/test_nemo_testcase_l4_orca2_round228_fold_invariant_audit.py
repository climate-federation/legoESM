from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as gate,
)


def _fold(nlon: int = 6):
    index = np.arange(nlon)
    return SimpleNamespace(
        perm_T=(-index) % nlon,
        perm_v=(-index) % nlon,
        perm_u=(-index - 1) % nlon,
    )


def test_fold_residual_scores_t_u_and_v_rules_and_detects_plant():
    fold = _fold()
    t = np.zeros((148, 6), dtype=np.float64)
    t[-1, :3] = (1.0, 2.0, 3.0)
    t[-1, 3:] = t[-1, np.asarray(fold.perm_T)[3:]]
    assert gate.fold_residual(t, "T", fold, 1.0)["unequal"] == 0

    u = np.zeros_like(t)
    u[-1, :3] = (1.0, 2.0, 3.0)
    u[-1, 3:] = -u[-1, np.asarray(fold.perm_u)[3:]]
    assert gate.fold_residual(u, "U", fold, -1.0)["unequal"] == 0

    v = np.zeros_like(t)
    v[-2] = np.arange(6, dtype=np.float64)
    v[-1] = -v[-2, np.asarray(fold.perm_v)]
    assert gate.fold_residual(v, "V", fold, -1.0)["unequal"] == 0
    v[-1, 2] = np.nextafter(v[-1, 2], np.inf)
    row = gate.fold_residual(v, "V", fold, -1.0)
    assert row["unequal"] == 1
    assert row["max_abs"] > 0.0


def _field_row(name: str, *, on: bool, kt: int) -> dict:
    owner = on and name in ("eta", "v")
    maximum = float(kt) if owner else 0.0
    return {
        "fold": {
            "sign": gate.SIGNS[name],
            "unequal": 1 if owner else 0,
            "max_abs": maximum,
            "argmax": [0],
            "first_nonfinite": None,
        },
        "oracle_fold": {"unequal": 0},
        "fold_band_vs_nemo": {"max_abs": maximum},
    }


def _report(card: str, label: str, unit: bool) -> dict:
    rows = []
    for kt in range(1, 9):
        for stage in ((1, 2, 3) if kt < 8 else (1, 2)):
            rows.append({
                "kt": kt,
                "stage": stage,
                "fields": {
                    name: _field_row(name, on=unit, kt=kt)
                    for name in gate.FIELDS
                },
            })
    return {
        "status": "PASS_R228_SCENARIO",
        "card": card,
        "claim_label": label,
        "atomic_unit": unit,
        "passivity": [
            {"kt": kt, "all_state_slots_equal": True} for kt in range(1, 8)
        ],
        "rows": rows,
    }


def _reports() -> list[dict]:
    return [_report(*scenario) for scenario in gate.SCENARIOS]


def _guard() -> dict:
    return {
        "rung0": {"pass": True},
        "omt4": {"pass": True},
    }


def test_classifier_confirms_only_complete_monotonic_four_scenario_owner():
    result = gate.classify(_reports(), _guard())
    assert result["status"] == "CONFIRMED_MISSING_FOLD_ASSOCIATION_OWNER_CANDIDATE"
    assert result["predictions"] == {
        "R228-P1": "CONFIRMED",
        "R228-P2": "CONFIRMED",
        "R228-P3": "CONFIRMED",
        "R228-P4": "CONFIRMED_MISSING_FOLD_ASSOCIATION_OWNER_CANDIDATE",
        "R228-P5": "CONFIRMED",
    }


@pytest.mark.parametrize(
    "plant", ("guard", "fold-sign", "boundary-order", "monotonic", "coverage"))
def test_every_plant_refuses(plant: str):
    with pytest.raises(gate.GateError):
        gate.classify(_reports(), _guard(), plant=plant)
