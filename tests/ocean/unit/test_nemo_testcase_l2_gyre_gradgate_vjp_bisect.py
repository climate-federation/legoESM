"""Direct controls for the certified GYRE statement-level VJP bisect."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradgate_vjp_bisect.py"
)
_SPEC = importlib.util.spec_from_file_location("gyre_gradgate_vjp_bisect_tested", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
B = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = B
_SPEC.loader.exec_module(B)


def test_vjp_taps_are_primal_identities_and_localize_poison_boundary():
    events = []

    def recorder(label, shape, bad_count, first_bad, _bad_mask):
        events.append((
            str(label), tuple(np.asarray(shape)), int(np.asarray(bad_count)),
            int(np.asarray(first_bad)),
        ))

    def unsafe(value):
        tapped_input = B.make_vjp_tap("unsafe.input", recorder)(value)
        result = jnp.sqrt(tapped_input)
        return B.make_vjp_tap("unsafe.output", recorder)(result)

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    assert np.asarray(unsafe(zero)).view(np.uint64) == np.asarray(zero).view(np.uint64)
    gradient = jax.grad(lambda value: 0.0 * unsafe(value))(zero)
    jax.block_until_ready(gradient)
    jax.effects_barrier()

    assert np.isnan(np.asarray(gradient))
    by_label = {label: bad for label, _shape, bad, _first in events}
    assert by_label["unsafe.output"] == 0
    assert by_label["unsafe.input"] == 1
