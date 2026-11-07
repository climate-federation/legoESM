"""Unit cover for the round-14 ORCA2 barotropic-owner gate.

The gate's expensive half is four production ORCA2 steps and is exercised by
running it.  What is tested here is the part that decides what the numbers
MEAN: that the substitution window is exactly rank 0's owned columns and
nothing else, that the two models' opposite drag/wind orders are reconciled by
comparing increments, that the depth-mean replay is the literal left-to-right
Fortran sum, and that the record reader still defaults to GYRE's extent after
being given an ORCA2 one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round14_barotropic_owner_gate.py")
RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5"
    "/acquisition/orca1ice_surface_entry_every_step_a_np2"
    "/oracle_slow_forcing_kt00000001.bin")


def _gate():
    for package in ("packages/core", "packages/ocean"):
        if str(REPO / package) not in sys.path:
            sys.path.insert(0, str(REPO / package))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("_r14_owner_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_injection_window_is_exactly_rank_zeros_owned_columns():
    gate = _gate()
    n = gate.RANK0_COLUMNS
    full_u = np.arange(148 * 181, dtype=np.float64).reshape(148, 181)
    full_v = np.arange(149 * 180, dtype=np.float64).reshape(149, 180)
    rank0_u = np.full((148, n), -1.0)
    rank0_v = np.full((148, n), -2.0)
    out_u, out_v = gate._inject(full_u, full_v, rank0_u, rank0_v)
    assert np.array_equal(out_u[:, 1:1 + n], rank0_u)
    assert np.array_equal(out_v[1:, :n], rank0_v)
    # Everything outside the window is untouched, INCLUDING the wrap column
    # and the V row the native slice drops.
    assert np.array_equal(out_u[:, :1], full_u[:, :1])
    assert np.array_equal(out_u[:, 1 + n:], full_u[:, 1 + n:])
    assert np.array_equal(out_v[:1, :], full_v[:1, :])
    assert np.array_equal(out_v[1:, n:], full_v[1:, n:])


def test_the_injection_refuses_a_window_of_the_wrong_extent():
    gate = _gate()
    with pytest.raises(gate.GateError):
        gate._inject(np.zeros((148, 181)), np.zeros((149, 180)),
                     np.zeros((148, 89)), np.zeros((148, 90)))


def test_the_depth_mean_replay_is_the_literal_left_to_right_sum():
    gate = _gate()
    rng = np.random.default_rng(14)
    e3 = rng.random((3, 4, 6))
    rhs = rng.random((3, 4, 6))
    mask = (rng.random((3, 4, 6)) > 0.3).astype(np.float64)
    recip = rng.random((3, 4))
    literal = np.zeros((3, 4))
    for j in range(3):
        for i in range(4):
            total = e3[j, i, 0] * rhs[j, i, 0] * mask[j, i, 0]
            for k in range(1, 6):
                total = total + e3[j, i, k] * rhs[j, i, k] * mask[j, i, k]
            literal[j, i] = total * recip[j, i]
    assert np.array_equal(gate._source_sum(e3, rhs, mask, recip), literal)


def test_the_two_models_opposite_orders_are_reconciled_as_increments():
    """NEMO does drag then wind; legoESM does wind then drag."""
    gate = _gate()
    shape = (2, 3)
    depth = np.full(shape, 1.0)
    drag, wind = 0.25, -0.5
    oracle = {
        "e3u": np.zeros((2, 3, 1)), "krhs_u": np.zeros((2, 3, 1)),
        "umask": np.ones((2, 3, 1)), "r1_hu0": np.ones(shape),
        "depth_u": depth,
        "post_drag_u": depth + drag,
        "post_wind_u": depth + drag + wind,
        "utau": np.zeros(shape),
    }
    rows = gate._oracle_arrays(oracle, "u")
    assert np.allclose(rows["drag_increment"], drag)
    assert np.allclose(rows["wind_increment"], wind)
    assert np.allclose(rows["final"], depth + drag + wind)

    operands = {
        "h_u": np.zeros((2, 4, 1)), "du_dt": np.zeros((2, 4, 1)),
        "H_u": np.full((2, 4), 2.0),
        "depth_u": np.pad(depth, ((0, 0), (1, 0)), constant_values=9.0),
        "post_wind_u": np.pad(depth + wind, ((0, 0), (1, 0)),
                              constant_values=9.0),
        "post_drag_u": np.pad(depth + wind + drag, ((0, 0), (1, 0)),
                              constant_values=9.0),
        "pre_external_u": np.pad(depth + wind + drag, ((0, 0), (1, 0)),
                                 constant_values=9.0),
        "wind_tau_u": np.zeros((2, 4)),
    }
    saved = gate.RANK0_COLUMNS
    try:
        gate.RANK0_COLUMNS = 3
        live = gate._face_arrays(operands, "u")
    finally:
        gate.RANK0_COLUMNS = saved
    assert np.allclose(live["wind_increment"], wind)
    assert np.allclose(live["drag_increment"], drag)
    assert np.allclose(live["final"], depth + wind + drag)
    assert np.allclose(live["r1_h0"], 0.5)


def test_the_comparison_sees_a_single_representable_move():
    gate = _gate()
    oracle = np.ones((4, 5))
    candidate = np.array(oracle, copy=True)
    candidate[2, 3] = np.nextafter(1.0, np.inf)
    active = np.ones((4, 5), dtype=bool)
    row = gate.compare(candidate, oracle, active)
    assert row["bit_exact"] is False
    assert row["differing_cells"] == 1
    assert row["ulp_max"] == 1
    assert gate.compare(oracle, oracle, active)["bit_exact"] is True


@pytest.mark.skipif(not RECORD.is_file(), reason="ORCA2 record not mounted")
def test_the_record_reader_takes_the_orca2_extent_and_keeps_gyres_default():
    sys.path.insert(0, str(REPO / "scripts/validate/ocean_fidelity/testcases"))
    from nemo_testcase_l2_gyre_round16_slow_forcing import (
        DIMS, read_slow_forcing,
    )
    import inspect

    gate = _gate()
    assert inspect.signature(
        read_slow_forcing).parameters["dims"].default == DIMS
    record = read_slow_forcing(RECORD, dims=gate.RANK0_DIMS)
    nx, ny, nz = gate.RANK0_DIMS
    assert record["header"]["nx"] == nx and record["header"]["nz"] == nz
    assert record["e3u"].shape == (ny - 4, nx - 4, nz - 1)
    assert record["depth_u"].shape == (ny - 4, nx - 4)
    # The ORCA2 extent is NOT GYRE's, so the default would have refused it.
    assert gate.RANK0_DIMS != DIMS
    with pytest.raises(Exception):
        read_slow_forcing(RECORD)
