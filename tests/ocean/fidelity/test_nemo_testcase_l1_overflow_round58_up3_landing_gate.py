"""Controls for the round-58 source-ordered UP3 landing gate."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round58_up3_landing_gate as gate  # noqa: E402


def test_production_source_flux_matches_nemo_association_and_mask():
    transport = (jnp.asarray(2.0), jnp.asarray(3.0))
    velocity = tuple(map(jnp.asarray, (8.0, 1.0, 2.0, 5.0)))
    selector = velocity[1] + velocity[2]
    result = np.float64(4.0) * np.asarray(
        gate._production_source_flux(
            *transport, *velocity, selector, jnp.asarray(0.0), jnp.asarray(1.0)
        ),
        dtype=np.float64,
    )
    assert result == np.float64(15.0)


def test_production_source_flux_refuses_selector_drift():
    with np.testing.assert_raises(gate.GateError):
        gate._production_source_flux(
            *map(jnp.asarray, (2.0, 3.0, 8.0, 1.0, 2.0, 5.0, 4.0, 0.0, 1.0))
        )
