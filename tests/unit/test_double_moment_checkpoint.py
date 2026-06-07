"""Double-moment tracers (q_i/q_s/q_g/N_c/N_r/N_i) must survive checkpoint /
restart on the coupled driver path.

Evolved by the coupled physics loop, these tracers were previously NOT
persisted by ``save_restart`` (only q_v/q_c/q_r were), so a restart silently
reinitialized them to the setup zeros. They now ride the already-round-tripped
``carry_aux`` dict, namespaced ``dmtr_*``. These tests pin the two ModelDriver
helpers that implement that — directly, without standing up a full driver:

* ``_checkpoint_carry_aux`` augments carry_aux with the current DM tracers
  (and is a no-op for warm-rain runs);
* ``_restore_dm_tracers_from_carry_aux`` pops the ``dmtr_*`` entries back into
  the tracer dict on load.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.driver.model_driver import ModelDriver


class _Stub:
    """Minimal object exposing just the attributes the DM checkpoint helpers
    touch, with the helpers bound from ModelDriver."""
    _DOUBLE_MOMENT_TRACERS = ModelDriver._DOUBLE_MOMENT_TRACERS
    _double_moment_step_inputs = ModelDriver._double_moment_step_inputs
    _checkpoint_carry_aux = ModelDriver._checkpoint_carry_aux
    _restore_dm_tracers_from_carry_aux = ModelDriver._restore_dm_tracers_from_carry_aux

    def __init__(self, tracers, carry_aux):
        self.tracers = tracers
        self._carry_aux = carry_aux


_DM = ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")


def _full_tracers(scale=1.0):
    t = {"q_v": jnp.ones((4,)) * 0.01, "q_c": jnp.ones((4,)) * 1e-3,
         "q_r": jnp.zeros((4,))}
    vals = {"q_i": 1e-4, "q_s": 2e-4, "q_g": 3e-4,
            "N_c": 1e8, "N_r": 1e6, "N_i": 5e3}
    for k, v in vals.items():
        t[k] = jnp.full((4,), v * scale)
    return t


def test_checkpoint_carry_aux_includes_dm_tracers():
    src = _Stub(_full_tracers(), {"held_dT_rad": jnp.zeros((4,))})
    aux = src._checkpoint_carry_aux()
    # held-radiation carry_aux preserved AND every DM tracer namespaced dmtr_*.
    assert "held_dT_rad" in aux
    for k in _DM:
        assert f"dmtr_{k}" in aux, k
    # Does NOT mutate the source carry_aux.
    assert "dmtr_q_i" not in src._carry_aux


def test_warm_rain_checkpoint_carry_aux_is_unchanged():
    aux_in = {"held_dT_rad": jnp.zeros((4,))}
    src = _Stub({"q_v": jnp.ones((4,)), "q_c": jnp.zeros((4,)),
                 "q_r": jnp.zeros((4,))}, dict(aux_in))
    aux = src._checkpoint_carry_aux()
    assert set(aux) == set(aux_in)            # no dmtr_* added for warm-rain


def test_restore_round_trip():
    src = _Stub(_full_tracers(scale=2.0), {"held_dT_rad": jnp.zeros((4,))})
    saved = src._checkpoint_carry_aux()       # what gets persisted

    # New driver after restart: DM tracers re-init to setup zeros, carry_aux
    # loaded back (carry_ prefix already stripped by the npz loader → dmtr_*).
    dst = _Stub(_full_tracers(scale=0.0), dict(saved))
    dst._restore_dm_tracers_from_carry_aux()

    for k in _DM:
        np.testing.assert_array_equal(
            np.asarray(dst.tracers[k]), np.asarray(src.tracers[k]))
        # dmtr_* entries popped so they don't pollute held-radiation carry_aux.
        assert f"dmtr_{k}" not in dst._carry_aux
    assert "held_dT_rad" in dst._carry_aux     # genuine aux untouched


def test_restore_noop_without_dm_entries():
    dst = _Stub({"q_v": jnp.ones((4,))}, {"held_dT_rad": jnp.zeros((4,))})
    dst._restore_dm_tracers_from_carry_aux()   # must not raise / change anything
    assert set(dst.tracers) == {"q_v"}
    assert set(dst._carry_aux) == {"held_dT_rad"}
