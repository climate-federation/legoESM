"""``core.state.with_dyn_tendencies``: the dynamics T / q_v tendency every
dycore driver hands to physics (WRF Tiedtke's ``qvften``, IFS-Bechtold)."""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from legoesm.core.field import Field
from legoesm.core.state import with_dyn_tendencies


class PhysicsState(NamedTuple):
    """Stand-in: the helper is duck-typed on ``_replace`` (core imports no
    physics); same two slot names as the real PhysicsState."""
    dyn_tendency_T: object = None
    dyn_tendency_qv: object = None


class _S(NamedTuple):
    T: Field
    tracers: dict | None


def _state(T, q):
    return _S(Field(T), None if q is None else {"q_v": Field(q)})


def _pair(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    sh = (3, 4, 5)
    return (jax.random.normal(k[0], sh) + 280.0,
            jax.random.uniform(k[1], sh) * 1e-2,
            jax.random.normal(k[2], sh) + 281.0,
            jax.random.uniform(k[3], sh) * 1e-2)


def test_writes_state_difference_over_dt():
    T0, q0, T1, q1 = _pair()
    dt = 450.0
    ps = with_dyn_tendencies(PhysicsState(), _state(T0, q0), _state(T1, q1), dt)
    np.testing.assert_allclose(ps.dyn_tendency_T, (T1 - T0) / dt, rtol=1e-14)
    np.testing.assert_allclose(ps.dyn_tendency_qv, (q1 - q0) / dt, rtol=1e-14)


def test_noop_without_phys_state_or_q_v():
    T0, q0, T1, q1 = _pair()
    assert with_dyn_tendencies(None, _state(T0, q0), _state(T1, q1), 1.0) is None
    ps = PhysicsState()
    out = with_dyn_tendencies(ps, _state(T0, None), _state(T1, q1), 1.0)
    assert out.dyn_tendency_qv is None and out.dyn_tendency_T is None


def test_gradients_and_jit_parity():
    T0, q0, T1, q1 = _pair(1)
    dt = 300.0

    def f(T0, q0, T1, q1):
        ps = with_dyn_tendencies(PhysicsState(), _state(T0, q0),
                                 _state(T1, q1), dt)
        return jnp.sum(ps.dyn_tendency_T ** 2) + jnp.sum(
            jnp.sin(ps.dyn_tendency_qv * 1e3))

    check_grads(f, (T0, q0, T1, q1), order=2, modes=("fwd", "rev"))
    np.testing.assert_allclose(jax.jit(f)(T0, q0, T1, q1), f(T0, q0, T1, q1),
                               rtol=1e-12)
