"""Controls for the round-94 ORCA2 rung-0 slow-boundary walk."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round94_slow_walk as gate,
)


RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round93/"
    "acquisition/orca2_rung0_slow_ranked_10step_np2"
)


def test_first_nonbit_obeys_compiled_source_order() -> None:
    rows = {name: {"bit_exact": True} for name in gate.SOURCE_ORDER}
    rows["depth_v"] = {"bit_exact": False, "absolute_max": 1.0}
    rows["drag_u"] = {"bit_exact": False, "absolute_max": 2.0}
    first = gate.first_nonbit(rows)
    assert first is not None
    assert first["boundary"] == "depth_v"


def test_rank_complete_slow_record_assembles_and_layout_plant_fires() -> None:
    if not RECORD.is_dir():
        pytest.skip("round-93 slow record is unavailable")
    arrays, census = gate.assemble_slow(RECORD)
    assert tuple(arrays) == gate.check_record.NAMES
    assert all(value.shape == (148, 180) for value in arrays.values())
    assert census["coverage"] == "exactly-once"
    with pytest.raises(gate.GateError, match="cover the domain exactly once"):
        gate.assemble_slow(RECORD, plant="layout")


def test_source_order_covers_every_recorded_scientific_boundary() -> None:
    assert set(gate.SOURCE_ORDER) == set(gate.check_record.NAMES) - {"cd_u", "cd_v"}


def test_source_depth_mean_is_literal_left_associated_under_jit() -> None:
    field = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    thickness = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float64)
    mask3 = np.ones_like(field)
    reciprocal = np.array([[0.25]], dtype=np.float64)
    mask2 = np.ones((1, 1), dtype=np.float64)
    expected = np.zeros((1, 1), dtype=np.float64)
    for level in range(3):
        expected = expected + (thickness[..., level] * field[..., level]) \
            * mask3[..., level]
    expected = expected * reciprocal * mask2
    actual = jax.device_get(jax.jit(gate.nemo_literal_depth_mean)(
        jnp.asarray(field), jnp.asarray(thickness), jnp.asarray(mask2),
        jnp.asarray(reciprocal), level_mask=jnp.asarray(mask3)))
    assert np.array_equal(actual, expected)
