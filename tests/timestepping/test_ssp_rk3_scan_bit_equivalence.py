"""Pin ``ssp_rk3_step_scan`` to bit-equivalence with ``ssp_rk3_step``.

Task #25: the scan-folded SSP-RK3 variant is meant to be a JIT-compile-
time optimisation, NOT a numerical change.  Same RK3 coefficients
(α = (0, 0.75, 1/3), β = (1, 0.25, 2/3)), same number of tendency
calls (3), same arithmetic — just expressed as a ``lax.scan`` over
3 iterations instead of three unrolled stages.

The XLA module should be smaller because the tendency function is
traced once (inside the scan body) rather than three times.  But the
floating-point output MUST be unchanged so existing scientific
validation (Williamson, Galewsky, AMIP) keeps holding.

If this test ever fails, treat it as a regression — do NOT relax the
tolerance.  The scan variant should be IEEE-identical to the unrolled
variant; any drift means the two variants made different choices
about associativity / FMA fusion, and the AD signal could differ in
ways the production AMIP run depends on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.timestepping.ssp_rk3 import ssp_rk3_step, ssp_rk3_step_scan


def _make_state(seed: int = 1234):
    """A tiny multi-field pytree shaped like the dycore state."""
    rng = np.random.default_rng(seed=seed)
    return {
        "u": jnp.asarray(rng.standard_normal((8, 16, 4))),
        "v": jnp.asarray(rng.standard_normal((9, 16, 4))),
        "T": jnp.asarray(300.0 + rng.standard_normal((8, 16, 4))),
        "p_s": jnp.asarray(1.0e5 + rng.standard_normal((8, 16))),
        "tracers": {
            "q_v": jnp.asarray(1.0e-3 * (1.0 + rng.standard_normal((8, 16, 4)))),
            "q_c": jnp.asarray(1.0e-5 * (1.0 + rng.standard_normal((8, 16, 4)))),
        },
    }


def _polynomial_tendency(state):
    """A non-trivial but deterministic tendency function.

    Mixes the fields so the tendency is sensitive to every component of
    state — guards against a trivial pass where one or both variants
    only happen to agree because the tendency is decoupled.
    """
    # Compute a scalar coupling factor from p_s + first T column.
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
    }


def test_one_step_bit_equivalent():
    """One SSP-RK3 step: scan variant matches unrolled variant exactly."""
    state = _make_state()
    dt = 100.0

    out_inline = ssp_rk3_step(state, _polynomial_tendency, dt)
    out_scan = ssp_rk3_step_scan(state, _polynomial_tendency, dt)

    leaves_inline = jax.tree.leaves(out_inline)
    leaves_scan = jax.tree.leaves(out_scan)
    assert len(leaves_inline) == len(leaves_scan), (
        f"pytree structure diverged: inline has {len(leaves_inline)} "
        f"leaves, scan has {len(leaves_scan)}"
    )
    for li, ls in zip(leaves_inline, leaves_scan):
        np.testing.assert_array_equal(
            np.asarray(li), np.asarray(ls),
            err_msg=(
                "scan-folded SSP-RK3 produced a different value than the "
                "unrolled SSP-RK3 reference.  The two variants should be "
                "IEEE-identical (same math, same op order); any drift is "
                "a regression that could affect AMIP scientific validation."
            ),
        )


def test_multi_step_bit_equivalent():
    """10 SSP-RK3 steps: drift would compound across steps if any."""
    state_inline = _make_state(seed=4321)
    state_scan = _make_state(seed=4321)
    dt = 50.0

    for _ in range(10):
        state_inline = ssp_rk3_step(state_inline, _polynomial_tendency, dt)
        state_scan = ssp_rk3_step_scan(state_scan, _polynomial_tendency, dt)

    leaves_inline = jax.tree.leaves(state_inline)
    leaves_scan = jax.tree.leaves(state_scan)
    for li, ls in zip(leaves_inline, leaves_scan):
        np.testing.assert_array_equal(
            np.asarray(li), np.asarray(ls),
            err_msg=(
                "10-step drift between scan-folded and unrolled SSP-RK3."
            ),
        )


def test_dispatch_lookup():
    """``ssp_rk3_scan`` is registered in the dispatch table."""
    from legoesm.timestepping.dispatch import dispatch_integrator
    state = _make_state()
    # Should not raise, should produce same output as scan call.
    out_dispatch = dispatch_integrator(
        state, _polynomial_tendency, 100.0, "ssp_rk3_scan",
    )
    out_direct = ssp_rk3_step_scan(state, _polynomial_tendency, 100.0)
    leaves_a = jax.tree.leaves(out_dispatch)
    leaves_b = jax.tree.leaves(out_direct)
    for la, lb in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


def test_scan_preserves_float32_leaf_dtype():
    """Mixed-precision state: scan body must NOT upcast float32 leaves.

    Smoke 8070583 surfaced this: ``jnp.asarray([0.0, 0.75, 1/3])`` in
    the scan body is strongly-typed float64.  Indexing the array gives
    a float64 scalar; multiplying with a float32 state leaf upcasts the
    result to float64.  ``jax.lax.scan`` then refuses to close the
    body because the carry-in (float32) and carry-out (float64) dtypes
    disagree.  This test pins the dtype-preserving fix.
    """
    state = {
        "u_f32": jnp.asarray(np.zeros((4, 8), dtype=np.float32)),
        "u_f64": jnp.asarray(np.zeros((4, 8), dtype=np.float64)),
    }

    def trivial_tendency(s):
        return {k: jnp.zeros_like(v) for k, v in s.items()}

    out = ssp_rk3_step_scan(state, trivial_tendency, 1.0)
    assert out["u_f32"].dtype == jnp.float32, (
        "scan-folded RK3 upcast float32 → "
        f"{out['u_f32'].dtype}.  The dtype-preserving cast in "
        "scan_body has regressed."
    )
    assert out["u_f64"].dtype == jnp.float64, (
        "scan-folded RK3 downcast float64 → "
        f"{out['u_f64'].dtype}; that would change AMIP scientific "
        "validation output."
    )


def test_jaxpr_smaller_than_inline():
    """The scan-folded jaxpr should have FEWER equations than the inline.

    This is the whole point of the optimisation: XLA sees the tendency
    code once (inside the scan body) instead of three times.  If this
    invariant ever flips, the scan variant has stopped delivering the
    JIT-compile speedup and the optimisation should be re-examined.
    """
    state = _make_state()
    inline_jaxpr = jax.make_jaxpr(ssp_rk3_step)(state, _polynomial_tendency, 100.0)
    scan_jaxpr = jax.make_jaxpr(ssp_rk3_step_scan)(state, _polynomial_tendency, 100.0)

    n_inline = len(inline_jaxpr.eqns)
    n_scan = len(scan_jaxpr.eqns)
    assert n_scan < n_inline, (
        f"Scan variant ({n_scan} eqns) is NOT smaller than inline "
        f"({n_inline} eqns) — the JIT-compile optimisation has lost "
        "its bite.  The expected ratio is ~3× smaller (the scan body "
        "is one stage, the inline has three stages unrolled)."
    )
