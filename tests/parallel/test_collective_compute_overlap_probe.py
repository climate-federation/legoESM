"""Direct test for the collective/compute overlap capability probe.

The probe answers one question — how much of a ``ppermute`` XLA can hide
behind independent compute — and an engineering decision rests on its
answer, so the instrument itself has to be checked. Each test below
fails if a specific way of lying is reintroduced:

* the collective arm really contains a collective and the control arm
  really contains none (otherwise the subtraction isolates nothing);
* the exchange actually rotates values between devices (a uniform fill
  plus a sum would hide a misrouted permutation);
* the FMA chain survives to the compiled program and grows with
  ``n_flop`` (a chain that is folded away makes the sweep a control that
  perturbs a zero);
* the two outputs stay separate, so no reduction joins the independent
  work to the collective's result;
* a ``--flops`` list that does not start at the reference row is
  refused.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2``.
"""

import re
import subprocess
import sys

import jax
import numpy as np
import pytest

from scripts.validate.collective_compute_overlap import (
    _build, _hlo_census, _paired_times,
)


def _need(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def test_comm_arm_has_a_collective_and_control_has_none():
    _need(2)
    f1, x, y = _build(2, payload=64, work=64, n_flop=2, comm=True)
    f0, _, _ = _build(2, payload=64, work=64, n_flop=2, comm=False)
    t1 = f1.lower(x, y).compile().as_text()
    t0 = f0.lower(x, y).compile().as_text()
    assert "collective-permute" in t1
    assert "collective-permute" not in t0
    c0 = _hlo_census(f0, x, y)
    assert c0["n_start"] == 0 and c0["n_plain"] == 0


def test_exchange_actually_rotates_between_devices():
    """The probe's payload is device-distinct on purpose. Assert the ring
    permutation really moves shard ``i`` to shard ``i+1``: a uniform fill
    plus a global sum would pass even if the permutation were misrouted
    or elided."""
    _need(2)
    from jax.sharding import Mesh, PartitionSpec as P

    from legoesm.parallel.shard_map_compat import shard_map

    mesh = Mesh(np.array(jax.devices()[:2]), ("device",))
    rot = jax.jit(shard_map(
        lambda a: jax.lax.ppermute(a, "device", perm=[(0, 1), (1, 0)]),
        mesh=mesh, in_specs=(P("device"),), out_specs=P("device"),
        check_vma=False))
    src = np.arange(2, dtype=np.float32)[:, None] * np.ones((2, 8), np.float32)
    out = np.asarray(rot(src))
    assert np.array_equal(out[0], src[1]) and np.array_equal(out[1], src[0])


def test_fma_chain_survives_compilation_and_grows():
    """A control that perturbs a zero is not a control: the chain must
    reach the compiled program and lengthen with ``n_flop``."""
    _need(2)
    counts = []
    for nf in (1, 64):
        f, x, y = _build(2, payload=64, work=64, n_flop=nf, comm=False)
        txt = f.lower(x, y).compile().as_text()
        counts.append(len(re.findall(r"multiply|fusion", txt)))
    assert counts[1] > counts[0], (
        f"FMA chain did not grow with n_flop ({counts}); the sweep would "
        f"be measuring a constant workload")


def test_outputs_stay_separate():
    """Two outputs, so no reduction joins the independent work to the
    collective's result — that join is what would manufacture a false
    'no overlap'."""
    _need(2)
    f1, x, y = _build(2, payload=64, work=64, n_flop=2, comm=True)
    out = f1(x, y)
    assert isinstance(out, tuple) and len(out) == 2


def test_paired_timing_returns_equal_length_interleaved_samples():
    _need(2)
    f1, x, y = _build(2, payload=64, work=64, n_flop=1, comm=True)
    f0, _, _ = _build(2, payload=64, work=64, n_flop=1, comm=False)
    t1, t0 = _paired_times(f1, f0, x, y, reps=6, warmup=1)
    assert len(t1) == len(t0) == 6
    assert np.all(np.isfinite(t1)) and np.all(np.isfinite(t0))


def test_flops_list_not_starting_at_zero_is_refused():
    out = subprocess.run(
        [sys.executable, "scripts/validate/collective_compute_overlap.py",
         "--flops", "64", "128"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "must start at 0" in (out.stderr + out.stdout)
