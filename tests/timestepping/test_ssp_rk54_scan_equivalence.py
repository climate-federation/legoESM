"""``ssp_rk54_step_scan``: scan-folded SSP-RK(5,4), equivalent to the inlined
``ssp_rk54_step`` but with the tendency compiled ONCE.

Why this variant exists
-----------------------
The inlined ``ssp_rk54_step`` emits FIVE copies of ``tendency_fn`` into the
XLA graph.  For a gather-heavy unstructured (MPAS/TRiSK) tendency the five
copies cross an XLA-CPU op-count threshold that de-vectorizes the indirect-
addressing gathers, so the step runs ~8x slower than its nominal 5-evaluation
cost (measured: 439 ms/step inlined vs 173 ms/step scan-folded vs 11 ms for a
single bare tendency, ico5 / nlev=40 Held-Suarez).  Folding the five stages
into a ``lax.scan`` compiles the stage body once and removes that blowup.

Bit-equivalence vs numerical-equivalence
----------------------------------------
Unlike ``ssp_rk3_step_scan`` (which is pinned BIT-identical to its inline
form), this variant is only equivalent to ~1e-9 relative.  SSP-RK(5,4) is
heterogeneous: stages 1-4 are 2-term Shu-Osher combinations while stage 5 is
a 5-term combination.  A single uniform scan body sums each stage over the
full stage axis (``tensordot``), which re-associates the additions relative
to the inlined form's hand-grouped 2-/5-term sums.  The resulting ~1e-9
relative drift is far below the scheme's O(dt^5) local truncation error, so
the two are the SAME 4th-order method.  ``test_linear_ode_fourth_order`` pins
that the scan variant really is SSP54 (4th-order), independently of the
inline reference.
"""
from __future__ import annotations

import jax
# The equivalence assertions below pin scan-vs-inline agreement to rtol=1e-9 —
# the re-association drift is ~1e-9 only in float64.  Under the default x64-off
# config (and the documented ``pytest tests/`` runner, which does NOT export
# JAX_ENABLE_X64=1) the same state is float32 and the drift is ~2e-7, turning
# these tests red.  Enable x64 at import so the asserted tolerance matches the
# docstring's float64 framing, regardless of the runner.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.timestepping.ssp_rk54 import ssp_rk54_step, ssp_rk54_step_scan


def _make_state(seed: int = 1234):
    """A tiny multi-field pytree shaped like the dycore state (incl. None)."""
    rng = np.random.default_rng(seed=seed)
    return {
        "u": jnp.asarray(rng.standard_normal((8, 16, 4))),
        "v": jnp.asarray(rng.standard_normal((9, 16, 4))),
        "T": jnp.asarray(300.0 + rng.standard_normal((8, 16, 4))),
        "p_s": jnp.asarray(1.0e5 + rng.standard_normal((8, 16))),
        "tracers": {
            "q_v": jnp.asarray(1e-3 * (1.0 + rng.standard_normal((8, 16, 4)))),
            "q_c": jnp.asarray(1e-5 * (1.0 + rng.standard_normal((8, 16, 4)))),
        },
        # None leaf: exercises the flatten/unflatten path (tracers=None etc.).
        "absent": None,
    }


def _polynomial_tendency(state):
    """Deterministic tendency coupling every field (no trivial decoupled pass)."""
    coupling = jnp.mean(state["p_s"]) + jnp.mean(state["T"][:, 0, 0])
    return {
        "u": 1e-3 * state["u"] + 1e-6 * coupling,
        "v": -2e-3 * state["v"] - 1e-6 * coupling,
        "T": 5e-4 * (state["T"] - 300.0) + 1e-6 * state["u"][..., 0:1],
        "p_s": 1e-4 * (state["p_s"] - 1.0e5),
        "tracers": {
            "q_v": -1e-3 * state["tracers"]["q_v"],
            "q_c": -2e-3 * state["tracers"]["q_c"]
                   + 3e-4 * state["tracers"]["q_v"],
        },
        "absent": None,
    }


def test_one_step_matches_inline():
    """One step: scan variant matches the inlined SSP54 to ~1e-9 relative."""
    state = _make_state()
    dt = 100.0
    out_inline = ssp_rk54_step(state, _polynomial_tendency, dt)
    out_scan = ssp_rk54_step_scan(state, _polynomial_tendency, dt)

    leaves_i = jax.tree.leaves(out_inline)
    leaves_s = jax.tree.leaves(out_scan)
    assert len(leaves_i) == len(leaves_s), "pytree structure diverged"
    for li, ls in zip(leaves_i, leaves_s):
        np.testing.assert_allclose(
            np.asarray(ls), np.asarray(li), rtol=1e-9, atol=1e-12,
            err_msg="scan-folded SSP54 drifted >1e-9 from the inline scheme",
        )


def test_multi_step_matches_inline():
    """10 steps: re-associated rounding must not compound beyond ~1e-8."""
    s_i = _make_state(seed=4321)
    s_s = _make_state(seed=4321)
    dt = 50.0
    for _ in range(10):
        s_i = ssp_rk54_step(s_i, _polynomial_tendency, dt)
        s_s = ssp_rk54_step_scan(s_s, _polynomial_tendency, dt)
    for li, ls in zip(jax.tree.leaves(s_i), jax.tree.leaves(s_s)):
        np.testing.assert_allclose(
            np.asarray(ls), np.asarray(li), rtol=1e-8, atol=1e-10,
            err_msg="10-step drift between scan-folded and inline SSP54",
        )


def test_linear_ode_fourth_order():
    """Scan variant is a genuine 4th-order method (independent of inline).

    Integrate y' = lam*y one step from y0=1; local truncation error of a
    4th-order method is O(dt^5), so halving dt cuts the 1-step error by ~2^5.
    A lower (e.g. 3rd-order) scheme would show a ~2^4 ratio — this pins that
    the scan fold did not silently drop the scheme's order.
    """
    lam = -0.7

    def tend(s):
        return {"y": lam * s["y"]}

    def one_step_err(dt):
        out = ssp_rk54_step_scan({"y": jnp.asarray([1.0])}, tend, dt)
        exact = np.exp(lam * dt)
        return abs(float(out["y"][0]) - exact)

    e1 = one_step_err(0.4)
    e2 = one_step_err(0.2)
    rate = np.log(e1 / e2) / np.log(2.0)
    assert rate > 4.5, (
        f"observed local convergence rate {rate:.2f} < 4.5 — scan fold "
        "degraded the SSP-RK(5,4) order"
    )


def test_dispatch_lookup():
    """``ssp_rk54_scan`` is registered and matches the direct call."""
    from legoesm.timestepping.dispatch import dispatch_integrator
    state = _make_state()
    out_d = dispatch_integrator(state, _polynomial_tendency, 100.0, "ssp_rk54_scan")
    out_direct = ssp_rk54_step_scan(state, _polynomial_tendency, 100.0)
    for la, lb in zip(jax.tree.leaves(out_d), jax.tree.leaves(out_direct)):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_homogeneous_precision_matches_inline(dtype):
    """Per-precision: scan preserves the state dtype AND matches the inline
    scheme.

    Pins that the fixed-dtype carry slots neither up- nor down-cast a
    homogeneous-precision state — the precision every production consumer feeds
    in after ``cast_pytree``.  (Earlier this test fed a *mixed* f32/f64 state
    and asserted the scan dtypes in isolation, which pinned the divergent
    behaviour instead of catching it; mixed precision is now rejected outright,
    see ``test_rejects_heterogeneous_precision_state``.)
    """
    rtol = 1e-9 if dtype is np.float64 else 1e-5
    state = {
        "a": jnp.asarray(np.ones((4, 8), dtype=dtype)),
        "b": jnp.asarray(2.0 * np.ones((4, 8), dtype=dtype)),
    }

    def tend(s):
        return {"a": -0.1 * s["a"] + 1e-3 * s["b"], "b": -0.2 * s["b"]}

    out_scan = ssp_rk54_step_scan(state, tend, 1.0)
    out_inline = ssp_rk54_step(state, tend, 1.0)
    for k in state:
        assert out_scan[k].dtype == jnp.dtype(dtype), (
            f"scan SSP54 changed {k} dtype -> {out_scan[k].dtype}")
        np.testing.assert_allclose(
            np.asarray(out_scan[k]), np.asarray(out_inline[k]),
            rtol=rtol, atol=0.0,
            err_msg=f"scan vs inline diverged for {k} at {dtype}")


def test_mixed_precision_matches_inline():
    """A state mixing float32 and float64 leaves matches the inline scheme.

    This is the real production shape: an MPAS state carries f64 prognostics
    alongside f32 tracers.  The inline form promotes each leaf to
    ``result_type(leaf, dt)`` (the f32 tracer -> f64 here, since dt follows the
    f64 first leaf); the scan fold must do the same — NOT silently down-cast the
    f64 stage combination back into an f32 slot (which both diverges and emits a
    deprecated-narrowing-cast FutureWarning).  Pins scan==inline in BOTH value
    and dtype for the mixed state (workflow review MAJOR).
    """
    import warnings
    state = {
        "p_f64": jnp.asarray(np.ones((4,), dtype=np.float64)),   # first leaf
        "q_f32": jnp.asarray(np.ones((4,), dtype=np.float32)),
    }

    def tend(s):
        return {"p_f64": -0.1 * s["p_f64"], "q_f32": -0.2 * s["q_f32"]}

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # no narrowing-cast ride
        out_scan = ssp_rk54_step_scan(state, tend, 1.0)
    out_inline = ssp_rk54_step(state, tend, 1.0)
    for k in state:
        assert out_scan[k].dtype == out_inline[k].dtype, (
            f"{k}: scan dtype {out_scan[k].dtype} != inline {out_inline[k].dtype}")
        np.testing.assert_allclose(
            np.asarray(out_scan[k]), np.asarray(out_inline[k]),
            rtol=1e-9, atol=0.0,
            err_msg=f"scan vs inline diverged for mixed-precision leaf {k}")


def test_rejects_higher_precision_tendency():
    """A tendency leaf wider than its float32 state slot is rejected, not
    silently narrowed (workflow review MAJOR)."""
    state = {"u": jnp.asarray(np.ones((4,), dtype=np.float32))}

    def tend(s):
        # float64 tendency for a float32 state -> would narrow in the carry.
        return {"u": jnp.asarray(np.full((4,), -0.1, dtype=np.float64))}

    with pytest.raises(TypeError, match="precision"):
        ssp_rk54_step_scan(state, tend, 1.0)


def test_scalar_python_tendency_leaf():
    """A weakly-typed Python-scalar tendency leaf is accepted, matching inline.

    The inline form and the carry ``.set`` both coerce Python scalars, so the
    new dtype guard must read the leaf through ``jnp.result_type(fi, ...)``
    rather than ``fi.dtype`` (a Python ``float`` has no ``.dtype``) — codex
    review P2: ``ssp_rk54_step_scan({'y': 1.0}, lambda s: {'y': -0.7}, 0.1)``
    used to raise ``AttributeError``.
    """
    state = {"y": jnp.asarray(np.array([1.0]))}

    def tend(s):
        return {"y": -0.7}  # constant Python float: no .dtype, weakly typed

    out_scan = ssp_rk54_step_scan(state, tend, 0.1)
    out_inline = ssp_rk54_step(state, tend, 0.1)
    assert np.isfinite(float(out_scan["y"][0])), "scalar-tendency scan non-finite"
    np.testing.assert_allclose(
        np.asarray(out_scan["y"]), np.asarray(out_inline["y"]),
        rtol=1e-9, atol=0.0,
        err_msg="scalar-tendency scan diverged from inline")


def _coupled_tend(s):
    """Tendency coupling u and T (module scope so the grad test does not
    rebuild a closure each trace)."""
    c = jnp.mean(s["T"])
    return {"u": 1e-3 * s["u"] + 1e-6 * c,
            "T": 5e-4 * (s["T"] - 300.0) + 1e-6 * s["u"]}


def test_grad_through_scan_matches_inline():
    """Reverse-mode AD through the scan fold matches the inline scheme.

    The scan variant is the production MPAS PE default integrator, so training
    runs ``jax.grad`` through this ``lax.scan`` body.  A future edit that breaks
    reverse-mode AD (a stray ``stop_gradient``, a control-flow change XLA cannot
    reverse-differentiate) would leave every forward-equivalence test green
    while silently zeroing or corrupting the gradient — exactly the 'never break
    autodiff' regression CLAUDE.md forbids.  Pin grad-equivalence + finiteness,
    including under jit (the production path is jitted).
    """
    def loss(scale, stepper):
        s = {"u": scale * jnp.ones((4, 6)),
             "T": 300.0 + scale * jnp.ones((4, 6))}
        for _ in range(3):
            s = stepper(s, _coupled_tend, 100.0)
        return jnp.sum(s["u"] ** 2) + jnp.sum(s["T"] ** 2)

    g_scan = float(jax.grad(loss)(1.3, ssp_rk54_step_scan))
    g_inline = float(jax.grad(loss)(1.3, ssp_rk54_step))
    assert np.isfinite(g_scan), "grad through scan fold is non-finite"
    np.testing.assert_allclose(
        g_scan, g_inline, rtol=1e-9, atol=0.0,
        err_msg="grad through scan fold diverged from the inline scheme")
    # Production differentiates a jitted step; ensure jit(grad(...)) is finite.
    g_jit = float(jax.jit(jax.grad(loss, argnums=0), static_argnums=(1,))(
        1.3, ssp_rk54_step_scan))
    assert np.isfinite(g_jit), "jit(grad) through scan fold is non-finite"


def test_rejects_integer_state_leaf():
    """Integer prognostic leaf is rejected, not silently integer-advanced.

    Fixed-dtype Shu-Osher carry slots cannot reproduce the inlined form's
    per-op promotion, so an int leaf must raise rather than degrade (codex
    review MAJOR).
    """
    state = {"n": jnp.asarray(np.arange(8, dtype=np.int32))}

    def tend(s):
        return {"n": jnp.zeros_like(s["n"])}

    with pytest.raises(TypeError, match="inexact"):
        ssp_rk54_step_scan(state, tend, 1.0)


def test_rejects_mismatched_tendency_structure():
    """Tendency whose pytree structure differs from the state must raise.

    Guards against silently mis-zipping leaves when the tendency drops/adds a
    field (e.g. a None/non-None mismatch) (codex review MAJOR).
    """
    state = {"a": jnp.ones((4,)), "b": jnp.ones((4,))}

    def bad_tend(s):  # drops "b" -> different treedef
        return {"a": -0.1 * s["a"]}

    with pytest.raises(ValueError, match="structure"):
        ssp_rk54_step_scan(state, bad_tend, 1.0)


def test_jaxpr_smaller_than_inline():
    """The fold's whole point: tendency traced once, not five times."""
    state = _make_state()
    n_inline = len(jax.make_jaxpr(ssp_rk54_step, static_argnums=(1,))(
        state, _polynomial_tendency, 100.0).eqns)
    n_scan = len(jax.make_jaxpr(ssp_rk54_step_scan, static_argnums=(1,))(
        state, _polynomial_tendency, 100.0).eqns)
    assert n_scan < n_inline, (
        f"scan jaxpr ({n_scan} eqns) not smaller than inline ({n_inline}) — "
        "the tendency is no longer compiled once")
