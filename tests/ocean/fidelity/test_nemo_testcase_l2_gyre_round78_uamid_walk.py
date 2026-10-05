from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_literal_midpoint_extrapolation,
)

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round78_uamid_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round78_uamid_walk", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _row(exact: bool, cells: int = 0) -> dict:
    return {
        "bit_exact": exact,
        "differing_cells": cells,
        "wet_cells": 4,
        "absolute_max": float(not exact),
        "reference_max_abs": 1.0,
    }


def test_source_order_stops_at_first_midpoint_input() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "coefficient_1": exact,
            "coefficient_2": exact,
            "coefficient_3": exact,
            "un_e": _row(False, 2),
            "ub_e": _row(False, 3),
            "ubb_e": _row(False, 4),
            "ua_e": _row(False, 4),
        }
    ]
    first = GATE.first_live_boundary(rows)
    assert first["substep"] == 1
    assert first["boundary"] == "un_e"
    assert first["differing_cells"] == 2


def test_null_ubb_e_plant_moves_boundary() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "coefficient_1": exact,
            "coefficient_2": exact,
            "coefficient_3": exact,
            "un_e": exact,
            "ub_e": exact,
            "ubb_e": exact,
            "ua_e": _row(False, 4),
        }
    ]
    first = GATE.first_live_boundary(rows)
    assert first["boundary"] == "ua_e"


def test_comparison_distinguishes_signed_zero() -> None:
    row = GATE.comparison(
        np.array([0.0], dtype=np.float64),
        np.array([-0.0], dtype=np.float64),
        np.array([True]),
    )
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1


def test_array_digest_distinguishes_one_ulp() -> None:
    left = np.array([1.0], dtype=np.float64)
    right = np.nextafter(left, np.inf)
    assert GATE._array_digest(left)["sha256"] != GATE._array_digest(right)["sha256"]


def test_shared_midpoint_keeps_left_association_under_jit() -> None:
    coefficients = jnp.ones(3, dtype=jnp.float64)
    got = jax.jit(nemo_literal_midpoint_extrapolation)(
        coefficients,
        jnp.array([1.0e16], dtype=jnp.float64),
        jnp.array([-1.0e16], dtype=jnp.float64),
        jnp.array([1.0], dtype=jnp.float64),
    )
    got_bits = np.asarray(got).view(np.uint64).item()
    expected_bits = np.float64(1.0).view(np.uint64).item()
    assert got_bits == expected_bits
