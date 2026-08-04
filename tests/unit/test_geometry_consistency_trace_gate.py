"""#1405: the per-CALL SPMD agreement gate must not run under a JAX trace.

``multihost_utils.process_allgather`` is an eager host-side utility.  Called
from inside ``jax.jit``/``lax.scan`` it stages its ``device_put``s into the
jaxpr and hands tracers to ``make_array_from_single_device_arrays``, which
raises — that killed every multi-process lat-lon SPMD run (#1405).

These tests are written so they FAIL if the skip is removed (the gate would
call the patched allgather under trace) and also if
``jax._src.core.trace_state_clean`` moves in a future JAX (``in_jax_trace``
would return False under trace).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.parallel import geometry_consistency as gc


def test_in_jax_trace_is_false_eagerly_and_true_under_jit():
    assert gc.in_jax_trace() is False

    seen = []

    @jax.jit
    def f(x):
        seen.append(gc.in_jax_trace())
        return x + 1

    f(np.float32(1.0))
    assert seen == [True], (
        "in_jax_trace() must detect an active trace; if this flipped to "
        "[False], jax._src.core.trace_state_clean moved and the #1405 crash "
        "is back.")


def _patch_multiprocess(monkeypatch, calls):
    """Make the gate believe it runs multi-process, and record allgathers."""
    from jax.experimental import multihost_utils

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        multihost_utils, "process_allgather",
        lambda payload, *a, **k: (calls.append(np.asarray(payload)),
                                  np.stack([np.asarray(payload)] * 2))[1])


def test_flags_gate_runs_eagerly_when_multiprocess(monkeypatch):
    calls = []
    _patch_multiprocess(monkeypatch, calls)
    gc.assert_flags_agree(("a", "b"), (1.0, 2.0), context="eager")
    assert len(calls) == 1


def test_flags_gate_is_skipped_under_trace(monkeypatch):
    calls = []
    _patch_multiprocess(monkeypatch, calls)

    @jax.jit
    def f(x):
        gc.assert_flags_agree(("a", "b"), (1.0, 2.0), context="traced")
        return x + 1

    f(np.float32(1.0))
    assert calls == [], (
        "assert_flags_agree called process_allgather under a trace — this is "
        "the #1405 failure mode (tracers into "
        "make_array_from_single_device_arrays).")


def test_flags_gate_is_skipped_under_scan(monkeypatch):
    """The reported #1405 chain is scan-inside-jit, not a bare jit."""
    calls = []
    _patch_multiprocess(monkeypatch, calls)

    @jax.jit
    def run(x):
        def body(carry, _):
            gc.assert_flags_agree(("a", "b"), (1.0, 2.0), context="scanned")
            return carry + 1, None

        out, _ = jax.lax.scan(body, x, None, length=3)
        return out

    run(np.float32(0.0))
    assert calls == []


def test_flags_gate_still_raises_on_disagreement_eagerly(monkeypatch):
    from jax.experimental import multihost_utils

    monkeypatch.setattr(jax, "process_count", lambda: 2)

    def _disagree(payload, *a, **k):
        p = np.asarray(payload)
        other = p.copy()
        other[-1] += 1.0
        return np.stack([p, other])

    monkeypatch.setattr(multihost_utils, "process_allgather", _disagree)
    with pytest.raises(RuntimeError, match="per-process CONFIG differs"):
        gc.assert_flags_agree(("a",), (1.0,), context="eager-disagree")
