"""The round-351 copy counter counts known cases exactly; every arm keeps arm a's bits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts/bench/orca2_round351_eos_store_microbench.py")
_spec = importlib.util.spec_from_file_location("orca2_round351_eos_store_microbench", SCRIPT)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

SMALL = (12, 10, 6)


@pytest.fixture(scope="module")
def args():
    return (*mb.make_inputs(SMALL), jnp.int32(1))


def _summary(fn, *a):
    return mb.copy_summary([jax.jit(fn).lower(*a).compile().as_text()])


def test_density_alone_is_one_complete_copy(args):
    s = _summary(mb.density, *args[:4])
    assert (s["complete"], s["partial"], s["poly_multiplies"]) == (1, 0, mb.POLY_PER_COPY)


def test_two_independent_densities_are_two_copies(args):
    t, s_, d, m = args[:4]
    s = _summary(lambda t, s_, d, m: (mb.density(t, s_, d, m), mb.density(t + 1.0, s_, d, m)),
                 t, s_, d, m)
    assert (s["complete"], s["partial"], s["poly_multiplies"]) == (2, 0, 2 * mb.POLY_PER_COPY)


def test_consumers_alone_hold_no_copy(args):
    t, _, d, _, w, _ = args
    s = _summary(mb.consumers, t, d, w)
    assert (s["complete"], s["partial"], s["poly_multiplies"]) == (0, 0, 0)


def test_baseline_reproduces_duplication(args):
    s = _summary(mb.ARMS["a"], *args)
    assert s["complete"] >= 2 and s["partial"] == 0
    assert s["poly_multiplies"] == s["complete"] * mb.POLY_PER_COPY


@pytest.mark.parametrize("name", [n for n in mb.ARM_ORDER if n != "a"])
def test_arm_keeps_baseline_bits(args, name):
    ref = jax.jit(mb.ARMS["a"])(*args)
    fwd, _ = mb.build_arm(name, args)
    out = fwd(*args)
    assert mb.bits_equal(out[: mb.N_EXACT], ref[: mb.N_EXACT])
    assert np.asarray(out[0]).dtype == np.float64


def test_bit_diff_counts_ulps():
    x = np.linspace(1.0, 2.0, 7)
    y = x.copy()
    y[3] = np.nextafter(y[3], 3.0)
    y[5] = np.nextafter(np.nextafter(y[5], 0.0), 0.0)
    assert mb.bit_diff([x], [y]) == (2, 7, 2)
    assert not mb.bits_equal([x], [y]) and mb.bits_equal([x], [x.copy()])
