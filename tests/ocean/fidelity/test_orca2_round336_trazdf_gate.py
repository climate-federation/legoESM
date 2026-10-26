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
    three = {name: np.full((nx, ny, nz), -0.0, dtype=np.float64)
             for name in gate.THREE_D_FIELDS}
    two = {name: np.zeros((nx, ny), dtype=np.float64)
           for name in gate.TWO_D_FIELDS}
    one = {"e3w_1d": np.ones(nz, dtype=np.float64)}
    four = {name: np.full((nx, ny, nz, nt), -0.0, dtype=np.float64)
            for name in gate.FOUR_D_FIELDS}
    fields = {**three, **two, **one, **four}
    fields["tmask"].fill(1.0)
    for name in ("avt", "avs", "akz"):
        fields[name].fill(0.0)
    fields["e3t_3d"].fill(10800.0)
    fields["r3t_Kmm"].fill(-1.0e-5)
    fields["Krhs_tracer"].fill(8.0)
    owned = np.zeros((nx, ny, nz, nt), dtype=bool)
    owned[2:-2, 2:-2, :nz - 1, :] = True
    fields["effective_K"][2:-2, 2:-2, :, :] = 0.0
    fields["e3w_Kmm"][2:-2, 2:-2, :] = np.float64(0.99999)
    metric = np.zeros((nx, ny, nz), dtype=bool)
    metric[2:-2, 2:-2, :nz - 1] = True
    fields["e3t_Kbb"][metric] = 10800.0
    fields["e3t_Kmm"][metric] = np.float64(10800.0) * np.float64(0.99999)
    fields["e3t_Kaa"][metric] = 10800.0
    process = np.zeros((nx, ny, nz), dtype=bool)
    process[2:-2, 2:-2, 1:nz - 1] = True
    for name in ("avt_k_zdf", "avt_after_tke", "avs_after_tke",
                 "avt_after_runoff", "avs_after_runoff", "avt_after_evd",
                 "avs_after_evd", "avt_after_ddm", "avs_after_ddm",
                 "avt_after_iwm", "avs_after_iwm"):
        fields[name][process] = 0.0
    fields["diagonal"][owned] = 10800.0
    fields["eliminated"][owned] = 10800.0
    rhs = (np.float64(10800.0)
           * (np.float64(10800.0) * np.float64(0.99999))
           * np.float64(8.0))
    fields["content_rhs"][owned] = rhs
    fields["forward"][owned] = rhs
    fields["backsub"][owned] = rhs / np.float64(10800.0)
    fields["actual_Kaa"][owned] = rhs / np.float64(10800.0)
    fields["solve_seen"][owned] = 1.0
    checksums = {
        name: int(np.bitwise_xor.reduce(value.view(np.uint64).reshape(-1), initial=0))
        for name, value in fields.items()
    }
    return {
        "header": {"nx": nx, "ny": ny, "nz": nz, "ntsi": 3,
                   "ntei": nx - 2, "ntsj": 3, "ntej": ny - 2},
        "fields": fields,
        "checksums": checksums,
    }


def test_clean_record_replays_and_covers_wet_cells():
    result = gate.verify_record(_record())
    assert result["wet_seen"] > 0
    assert result["signed_zeros"] > 0
    assert result["rhs_association_discriminators"] > 0


@pytest.mark.parametrize("plant", (
    "nonfinite", "signed-zero", "wet", "replay", "association", "actual",
    "filled-range", "zdfphy",
))
def test_value_plants_fire(plant):
    record = copy.deepcopy(_record())
    with pytest.raises(gate.GateError):
        gate.verify_record(record, plant=plant)


def test_structural_plants_are_registered():
    assert set(("header", "field-order", "truncation", "checksum", "stamp",
                "selector", "extent", "time-level")) < set(gate.PLANTS)
    assert gate.OBSERVER_UNITS[-2:] == ("traldf_iso", "trazdf")
    assert gate.WRITER_UNITS[-1] == "round336_trazdf_writer"
