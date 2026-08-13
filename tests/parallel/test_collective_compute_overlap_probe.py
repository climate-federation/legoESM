"""Direct test for the collective/compute overlap capability probe.

The probe answers one question — can XLA hide a ``ppermute`` behind
independent compute — and a whole engineering decision rests on its
answer, so its two arms must actually be what they claim: the ``comm``
arm has to contain a collective and the control arm must not, and the
independent work must actually grow with ``n_flop``. A probe whose
control silently still communicates would report "no overlap" forever.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2``.
"""

import jax
import numpy as np
import pytest

from scripts.validate.collective_compute_overlap import _build, _time


def _need(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def test_comm_arm_has_a_collective_and_control_does_not():
    _need(2)
    f1, x, y = _build(2, payload=64, work=64, n_flop=2, comm=True)
    f0, _, _ = _build(2, payload=64, work=64, n_flop=2, comm=False)
    assert "collective-permute" in f1.lower(x, y).compile().as_text()
    assert "collective-permute" not in f0.lower(x, y).compile().as_text()


def test_independent_work_actually_grows_with_n_flop():
    """A control that perturbs a zero is not a control: the FMA chain
    must lengthen the compiled program, or the sweep measures nothing."""
    _need(2)
    lens = []
    for nf in (1, 64):
        f, x, y = _build(2, payload=64, work=64, n_flop=nf, comm=False)
        lens.append(len(f.lower(x, y).compile().as_text()))
    assert lens[1] > lens[0]


def test_timer_returns_a_positive_median():
    _need(2)
    f, x, y = _build(2, payload=64, work=64, n_flop=1, comm=True)
    t = _time(f, x, y, reps=3)
    assert np.isfinite(t) and t > 0.0
