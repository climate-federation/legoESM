"""Controls for the round-336 ``tra_zdf`` acquisition gate."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_trazdf_gate as gate,
)


def _record():
    nx, ny, nz, nt = 7, 6, 5, 2
    three = {name: np.zeros((nx, ny, nz), dtype=np.float64)
             for name in gate.THREE_D_FIELDS}
    two = {name: np.zeros((nx, ny), dtype=np.float64)
           for name in gate.TWO_D_FIELDS}
    one = {"e3w_1d": np.ones(nz, dtype=np.float64)}
    four = {name: np.full((nx, ny, nz, nt), -0.0, dtype=np.float64)
            for name in gate.FOUR_D_FIELDS}
    fields = {**three, **two, **one, **four}
    for name in ("e3t_3d", "e3w_Kmm", "e3t_Kbb", "e3t_Kmm", "e3t_Kaa", "tmask"):
        fields[name].fill(1.0)
    fields["Kbb_tracer"].fill(2.0)
    owned = np.zeros((nx, ny, nz, nt), dtype=bool)
    owned[2:-2, 2:-2, :nz - 1, :] = True
    fields["effective_K"][2:-2, 2:-2, :, :] = 0.0
    fields["diagonal"][owned] = 1.0
    fields["eliminated"][owned] = 1.0
    fields["content_rhs"][owned] = 2.0
    fields["forward"][owned] = 2.0
    fields["backsub"][owned] = 2.0
    fields["actual_Kaa"][owned] = 2.0
    fields["solve_seen"][owned] = 1.0
    checksums = {
        name: int(np.bitwise_xor.reduce(value.view(np.uint64).reshape(-1), initial=0))
        for name, value in fields.items()
    }
    return {
        "header": {"nx": nx, "ny": ny, "nz": nz},
        "fields": fields,
        "checksums": checksums,
    }


def test_clean_record_replays_and_covers_wet_cells():
    result = gate.verify_record(_record())
    assert result["wet_seen"] > 0
    assert result["signed_zeros"] > 0


@pytest.mark.parametrize("plant", ("nonfinite", "signed-zero", "wet", "replay", "actual"))
def test_value_plants_fire(plant):
    record = copy.deepcopy(_record())
    with pytest.raises(gate.GateError):
        gate.verify_record(record, plant=plant)


def test_structural_plants_are_registered():
    assert set(("header", "field-order", "truncation", "checksum", "stamp",
                "selector", "extent")) < set(gate.PLANTS)
    assert gate.OBSERVER_UNITS[-2:] == ("traldf_iso", "trazdf")
    assert gate.WRITER_UNITS[-1] == "round336_trazdf_writer"
