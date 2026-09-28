"""Controls for the round-57 OVERFLOW UP3 statement walk."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round57_up3_statement_gate as gate  # noqa: E402


def test_production_expression_can_differ_from_nemo_source_association():
    velocity = np.array([
        3.31646809321458e-05,
        4.605109786009353e-05,
        1.5578967898351246e-04,
        1.3338847226702986e-04,
    ], dtype=np.float64)
    transport = np.array([
        3.6587957947489358e-03,
        -1.1852888897627461e-02,
    ], dtype=np.float64)
    pair = np.float64(velocity[1] + velocity[2])
    curvature = np.float64(
        (velocity[2] - velocity[1]) + (velocity[0] - velocity[1]))
    nemo = np.float64(
        (transport[0] + transport[1])
        * (pair - np.float64(1.0 / 3.0) * curvature))
    lego = np.float64(4.0) * np.asarray(
        gate._production_t_flux(
            *map(jnp.asarray, (
                transport[0], transport[1], velocity[0], velocity[1],
                velocity[2], velocity[3], pair,
            ))
        ), dtype=np.float64)
    assert nemo.view(np.uint64) == 0xBEB74F4565B8970F
    assert lego.view(np.uint64) == 0xBEB74F4565B8970E


def test_face_flux_plant_adds_exactly_one_refusal():
    oracle = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    candidate = oracle.copy()
    candidate[2] = np.nextafter(candidate[2], np.float64(np.inf))
    baseline = gate._score("flux", oracle, candidate)
    planted = gate._score("flux", oracle, candidate, plant=True)
    assert baseline["baseline_n_unequal"] == 1
    assert baseline["row_scale_ulp_max_nonzero_reference"] == 1.0
    assert planted["n_unequal"] == 2


def test_parent_mask_embedding_obeys_record_origin():
    parent = np.ones((1, 2, 1), dtype=bool)
    full = gate._embed_parent_mask(parent, (6, 5, 1), (3, 3))
    assert np.array_equal(np.argwhere(full), np.array([[2, 2, 0], [3, 2, 0]]))
