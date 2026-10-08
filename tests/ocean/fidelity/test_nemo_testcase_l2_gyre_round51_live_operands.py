"""Non-vacuity and layout checks for the round-51 live-operand gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l2_gyre_round51_live_operands as gate  # noqa: E402


def test_operand_plant_moves_an_exact_active_cell():
    values = np.arange(6.0).reshape(2, 3)
    mask = np.ones_like(values, dtype=bool)
    clean = gate._masked_row(
        "clean", values, values, mask, stage=1, operator="hpg",
        field="u_Kmm")
    planted = gate._masked_row(
        "planted", values, values, mask, stage=1, operator="hpg",
        field="u_Kmm", plant=True)
    assert clean["n_unequal"] == 0
    assert planted["n_unequal"] == 1
    assert planted["absolute_max"] == 1.0


def test_raw_history_mapping_preserves_b_then_bb_order_and_face_layouts():
    names = ("ub_e", "ubb_e", "vb_e", "vbb_e", "sshb_e", "sshbb_e")
    arrays = {
        name: np.full((6, 7), index, dtype=np.float64)
        for index, name in enumerate(names, start=1)
    }
    ub, ubb, vb, vbb, sshb, sshbb = tuple(
        np.asarray(value) for value in gate._raw_history({"arrays": arrays}))
    assert (ub.shape, ubb.shape) == ((2, 4), (2, 4))
    assert (vb.shape, vbb.shape) == ((3, 3), (3, 3))
    assert (sshb.shape, sshbb.shape) == ((2, 3), (2, 3))
    assert [float(x.max()) for x in (ub, ubb, vb, vbb, sshb, sshbb)] == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.all(vb[0] == 0.0) and np.all(vbb[0] == 0.0)


def test_live_trace_and_raw_history_arms_are_private_and_off_by_default():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    default = _NEMOWSRK3TestHooks()
    assert default.expose_live_stage_operands is False
    assert default.barotropic_raw_history_override is None
    assert default.slow_forcing_incoming_override is None
    assert default.barotropic_slow_forcing_override is None
    def sink(*values):
        return values
    observed = _NEMOWSRK3TestHooks(
        barotropic_slow_forcing_override=sink)
    assert observed.barotropic_slow_forcing_override is sink
    assert "barotropic_slow_forcing_override" not in (
        LatLonCGridOceanConfig._fields)
    assert "slow_forcing_producer" in _NEMOWSLiveOperandTrace._fields
    assert _NEMOWSLiveOperandTrace._fields[-6:] == (
        "stage_rhs", "stage1_full_rhs", "stage1_rhs_walk",
        "stage_raw_velocities",
        "barotropic_correction_geometry", "stage_outputs")
