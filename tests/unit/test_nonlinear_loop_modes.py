"""The shared Newton solver's two loop forms must agree BIT FOR BIT (#1736).

``make_implicit_newton_solver(loop="fixed")`` exists because a second-order
operator -- ``curvature.hvp``, a Gauss-Newton inner solve, a Lanczos spectrum --
cannot be compiled through the ``lax.while_loop`` form: forward-over-reverse
produced a program the XLA CPU backend ABORTED on, a core dump rather than an
exception, which is why the whole ``tests/unit`` suite could not be run to
completion in one process.

The fixed form is only useful if it is the SAME solver, so that is what these
tests pin: same roots, same iteration count, same convergence flags, same
gradients -- and, mechanically, that the primitive responsible is gone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.nonlinear import make_implicit_newton_solver

_MAX_ITERS = 12


def _residual(x, parameters):
    """A small nonlinear system with a parameter-dependent root.

    Deliberately not solvable in one Newton step (the cubic and the coupling
    make the damping adapt), so the iteration count is a real number and the
    two loop forms have something to disagree about.
    """
    a, b = parameters
    return jnp.array([
        x[0] ** 3 - 2.0 * x[1] - a,
        jnp.sin(x[0]) + 3.0 * x[1] ** 2 - b,
        x[0] * x[1] - 0.25 * x[2] ** 2 - 0.5,
    ])


def _solver(loop):
    return make_implicit_newton_solver(
        _residual,
        x_scale=jnp.ones(3),
        f_scale=jnp.ones(3),
        max_iters=_MAX_ITERS,
        loop=loop,
    )


_X0 = jnp.array([1.2, 0.7, 1.1])
_P = (jnp.asarray(1.5), jnp.asarray(2.5))


def test_the_two_loop_forms_return_the_same_solve():
    """Every returned field, bit for bit."""
    out_while = _solver("while")(_X0, _P)
    out_fixed = _solver("fixed")(_X0, _P)
    names = ("x", "n_iters", "converged", "n_sq", "n_sq_rel", "damping",
             "hit_cap")
    for name, w, f in zip(names, out_while, out_fixed):
        np.testing.assert_array_equal(
            np.asarray(w), np.asarray(f),
            err_msg=f"loop forms disagree on {name}")
    # The premise: the solve is real work, not a one-step or zero-step case
    # that both forms would pass trivially.
    assert bool(out_while[2]), "fixture did not converge; pick a better system"
    assert 1 < int(out_while[1]) < _MAX_ITERS, (
        f"fixture converged in {int(out_while[1])} iterations -- with an "
        "early exit at the first or last iteration the two forms cannot "
        "disagree, so this test would be vacuous")


def test_the_fixed_form_has_no_while_primitive_and_the_other_one_does():
    """The mechanical link to the defect, not just an equality of numbers."""
    def count_while(jaxpr):
        n = 0
        for eqn in jaxpr.eqns:
            n += eqn.primitive.name == "while"
            for v in eqn.params.values():
                sub = getattr(v, "jaxpr", None)
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    n += count_while(inner)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        s2 = getattr(item, "jaxpr", item)
                        if hasattr(s2, "eqns"):
                            n += count_while(s2)
        return n

    j_while = jax.make_jaxpr(lambda x: _solver("while")(x, _P)[0])(_X0)
    j_fixed = jax.make_jaxpr(lambda x: _solver("fixed")(x, _P)[0])(_X0)
    assert count_while(j_while.jaxpr) >= 1, (
        "the while form lost its while_loop -- this test no longer "
        "discriminates")
    assert count_while(j_fixed.jaxpr) == 0, (
        "the fixed form still contains a while_loop, so it does not fix "
        "what it exists to fix (#1736)")


def test_gradients_agree_between_the_forms():
    """The custom VJP is shared, but the primal it saves comes from the loop."""
    def loss(p0, loop):
        x = _solver(loop)(_X0, (p0, _P[1]))[0]
        return jnp.sum(x ** 2)

    g_while = jax.grad(lambda p: loss(p, "while"))(_P[0])
    g_fixed = jax.grad(lambda p: loss(p, "fixed"))(_P[0])
    np.testing.assert_array_equal(np.asarray(g_while), np.asarray(g_fixed))
    assert np.isfinite(float(g_while)) and abs(float(g_while)) > 0.0


def test_forward_over_reverse_works_on_the_fixed_form():
    """The point of the mode: an hvp that can be built at all.

    The while form is NOT exercised here on purpose -- on this objective it is
    the configuration that aborted the compiler, and an aborting arm cannot be
    a pytest assertion (it takes the interpreter with it).  #1736 records the
    measurement; this pins that the fixed form gives second-order operators a
    path that works.
    """
    def loss(p0):
        x = _solver("fixed")(_X0, (p0, _P[1]))[0]
        return jnp.sum(x ** 2)

    hv = jax.jvp(jax.grad(loss), (_P[0],), (jnp.asarray(1.0),))[1]
    assert np.isfinite(float(hv))

    # against a central finite difference of the gradient
    eps = 1e-4
    g = jax.grad(loss)
    fd = (float(g(_P[0] + eps)) - float(g(_P[0] - eps))) / (2.0 * eps)
    assert abs(float(hv) - fd) <= 1e-3 * max(abs(fd), 1.0), (
        f"hvp {float(hv):.6e} disagrees with the finite difference {fd:.6e}")


def test_unknown_loop_mode_raises():
    with pytest.raises(ValueError, match="loop"):
        make_implicit_newton_solver(
            _residual, x_scale=jnp.ones(3), f_scale=jnp.ones(3),
            max_iters=4, loop="scan")
