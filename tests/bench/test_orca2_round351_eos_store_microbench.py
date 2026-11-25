"""The round-351 copy counter counts known cases exactly; every arm keeps arm a's bits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts/bench/orca2_round351_eos_store_microbench.py")
_spec = importlib.util.spec_from_file_location("orca2_round351_eos_store_microbench", SCRIPT)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

SMALL = (12, 10, 6)
RHO = mb.POLY["rho"][1]


@pytest.fixture(scope="module")
def args():
    return (*mb.make_inputs(SMALL), jnp.int32(1))


def _summary(fn, *a):
    return mb.copy_summary([jax.jit(fn).lower(*a).compile().as_text()], a[0].size)


def _rho(fn, *a):
    s = _summary(fn, *a)["rho"]
    return s["copies"], s["partial_kernels"], s["multiplies"], s["grid_evals"]


def test_density_alone_is_one_copy(args):
    assert _rho(mb.density, *args[:4]) == (1, 0, RHO, 1.0)


def test_rab_alone_is_one_pair(args):
    s = _summary(lambda t, s_, d: mb.nemo_roquet_alpha_beta(
        t, s_, d, eos_form="eos80", source_rounded=True), *args[:3])
    # alpha and beta land in separate kernels even when stored once: count work, not kernels
    assert (s["rab"]["multiplies"], s["rab"]["copy_equivalents"], s["rab"]["grid_evals"]) == (
        mb.POLY["rab"][1], 1.0, 1.0)
    assert s["rho"]["multiplies"] == 0


def test_two_densities_in_one_kernel_are_two_copies(args):
    t, s_, d, m = args[:4]
    s = _summary(lambda t, s_, d, m: mb.density(t, s_, d, m)
                 + mb.density(t + 1.0, s_ + 0.5, d + 1.0, m), t, s_, d, m)
    held = [k for k in s["kernels"] if k[4]]
    assert len(held) == 1 and held[0][4] == 2 * RHO
    assert (s["rho"]["copies"], s["rho"]["grid_evals"]) == (2, 2.0)


def test_copy_split_across_a_scan_counts_each_trip(args):
    def per_level(t, s_, d, m):
        xs = tuple(jnp.moveaxis(x, -1, 0) for x in (t, s_, d, m))
        return lax.scan(lambda c, x: (c, mb.density(*x)), 0.0, xs)[1]
    assert _rho(per_level, *args[:4]) == (SMALL[2], 0, SMALL[2] * RHO, 1.0)


def test_five_trip_loop_is_five_copies(args):
    def five(t, s_, d, m):
        # every input moves each trip, so XLA cannot hoist a loop-invariant part of the copy
        def body(i, c):
            return c[0] + 1e-3, c[1] + 1e-3, c[2] + 1e-3, c[3] + mb.density(c[0], c[1], c[2], m)
        return lax.fori_loop(0, 5, body, (t, s_, d, jnp.zeros_like(t)))[3]
    assert _rho(five, *args[:4]) == (5, 0, 5 * RHO, 5.0)


def test_dead_branch_reported_apart(args):
    t, s_, d, m, _, n = args

    def cnd(t, s_, d, m, n):
        return lax.cond(n > 0, lambda: mb.density(t, s_, d, m),
                        lambda: mb.density(t + 1.0, s_ + 0.5, d + 1.0, m)
                        + mb.density(t + 2.0, s_ + 0.7, d + 2.0, m))
    s = _summary(cnd, t, s_, d, m, n)["rho"]
    assert (s["copies"], s["dead_branch_copies"]) == (1, 2)


def test_slope_stencil_reads_density_twice_per_cell(args):
    def stencil(t, s_, d, m):
        r = mb.density(t, s_, d, m)
        return jnp.roll(r, -1, axis=1) - r
    assert _rho(stencil, *args[:4]) == (1, 0, RHO, 2.0)


def test_consumers_alone_hold_no_copy(args):
    t, s_, d, m, w, _ = args
    fields = mb.eos_fields(t, s_, d, m)
    s = mb.copy_summary([jax.jit(mb.consumers).lower(fields, t, s_, d, w).compile().as_text()],
                        t.size)
    assert s["rho"]["multiplies"] == s["rab"]["multiplies"] == 0


def test_baseline_reproduces_duplication(args):
    s = _summary(mb.ARMS["a"], *args)["rho"]
    assert s["copies"] >= 2 and s["partial_kernels"] == 0


def test_bit_diff_counts_ulps():
    x = np.linspace(1.0, 2.0, 7)
    y = x.copy()
    y[3] = np.nextafter(y[3], 3.0)
    y[5] = np.nextafter(np.nextafter(y[5], 0.0), 0.0)
    assert mb.bit_diff([x], [y]) == (2, 7, 2)
    assert not mb.bits_equal([x], [y]) and mb.bits_equal([x], [x.copy()])


@pytest.mark.parametrize("name", [n for n in mb.ARM_ORDER if n != "a"])
def test_arm_keeps_baseline_bits(args, name):
    ref = jax.jit(mb.ARMS["a"])(*args)
    fwd, _ = mb.build_arm(name, args)
    out = fwd(*args)
    assert mb.bits_equal(out[: mb.N_EXACT], ref[: mb.N_EXACT])
    assert np.asarray(out[0]).dtype == np.float64
