"""Bitwise regression pin of the SI3 ORCA2 lane functions on ``main``.

THIS IS A SELF-REFERENCED REGRESSION PIN, NOT AN ORACLE.  The committed
outputs in ``tests/ice/fixtures/si3_orca2_regression/`` are what the lane
computed on deterministic synthetic ORCA2-shaped inputs when the fixture was
generated (``provenance.json``); they say nothing about agreement with NEMO.
The test recomputes every rung from the stored inputs and requires every
output leaf to be bit-identical (floats compared through same-width unsigned views,
so ``-0.0``/``+0.0`` and NaN payloads also count).  A change that moves any
bit fails here; if it was intended, regenerate with ``make_fixture.py`` and
say so in the PR.

Pinned: ``si3_lbc_lnk_orca2`` (T/U/V/F, both signs, both ``ldfull``, 2-D and
4-D), ``si3_apply_state_halos_orca2``, ``si3_o4_step`` (icycle 1, 2 and a
delayed-CFL icycle-3 case), ``si3_orca2_ice_strength`` and
``si3_orca2_aevp_solver``/``_trace`` (both fold programs, both
``differentiable`` arms), ``si3_dom_uniq_t_orca2``, ``si3_o15_step``,
``si3_o16_step`` and ``sea_ice._si3_orca2_o2_step_with_trace``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "si3_orca2_regression"


def _load_generator():
    spec = importlib.util.spec_from_file_location("si3_orca2_make_fixture", FIXTURE / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


PROVENANCE = (FIXTURE / "provenance.json").read_text()


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.view(f"u{a.itemsize}") if a.dtype.kind == "f" else a


def _assert_bitwise(expected: dict, actual: dict) -> None:
    where = f"\nfixture provenance: {PROVENANCE}"
    assert sorted(actual) == sorted(expected), "output leaf set changed" + where
    for key in expected:
        exp, got = np.asarray(expected[key]), np.asarray(actual[key])
        assert got.dtype == exp.dtype and got.shape == exp.shape, key + where
        np.testing.assert_array_equal(_bits(got), _bits(exp), err_msg=key + where)


def _load(rung: str) -> tuple[dict, dict]:
    with np.load(FIXTURE / f"{rung}.npz") as z:
        inputs = {k[3:]: z[k] for k in z.files if k.startswith("in/")}
        outputs = {k[4:]: z[k] for k in z.files if k.startswith("out/")}
    return inputs, outputs


@pytest.fixture
def pin_policy():
    if jax.default_backend() != "cpu":
        pytest.skip("the pin runs under the CPU-only libm transcendental policy")
    with GEN.set_pin_policy():
        yield


@pytest.mark.parametrize("rung", sorted(GEN.RUNGS))
def test_rung_is_bitwise_identical_to_committed_snapshot(rung: str, pin_policy) -> None:
    inputs, expected = _load(rung)
    _assert_bitwise(expected, GEN.RUNGS[rung](inputs))


@pytest.mark.parametrize("rung", sorted(GEN.RUNGS))
def test_one_ulp_change_in_a_stored_output_trips_the_pin(rung: str) -> None:
    """Non-vacuity: the comparison itself can fail on a single-ulp change."""
    _, expected = _load(rung)
    key = next(k for k in sorted(expected) if np.asarray(expected[k]).dtype == np.float64)
    planted = dict(expected)
    arr = np.array(planted[key], copy=True)
    flat = arr.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.inf)
    planted[key] = arr
    with pytest.raises(AssertionError):
        _assert_bitwise(expected, planted)
