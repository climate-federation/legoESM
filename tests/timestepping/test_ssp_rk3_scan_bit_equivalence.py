"""Pin ``ssp_rk3_step_scan`` numerically equivalent to ``ssp_rk3_step``.

Task #25: the scan-folded SSP-RK3 variant is a JIT-compile-time
optimisation, NOT a numerical change.  Same RK3 coefficients
(α = (0, 0.75, 1/3), β = (1, 0.25, 2/3)), same number of tendency
calls (3), same arithmetic — just expressed as a ``lax.scan`` over
3 iterations instead of three unrolled stages.

The XLA module should be smaller because the tendency function is
traced once (inside the scan body) rather than three times.  The
floating-point output must stay numerically equivalent so existing
scientific validation (Williamson, Galewsky, AMIP) keeps holding.

Equivalence is NUMERICAL (~1e-9 relative in float64), not bit-exact.
The scan body and the three unrolled stages are the same math, but XLA
is free to fuse/associate the FMAs in the single compiled scan body
differently from the unrolled form, and that choice is JAX/XLA-version
dependent (it changed at JAX 0.10).  A real regression (a coefficient
typo, a dropped stage, an AD-breaking control-flow change) shifts the
result far above 1e-9, so a tight rtol still catches it; demanding
bit-identity across XLA versions does not.  Run under ``x64`` so the
1e-9 tolerance is meaningful (float32's re-association drift is ~2e-7).
"""
from __future__ import annotations

import jax
# Enable x64 at import (sibling test_leapfrog_ab2 convention) so the rtol=1e-9
# equivalence assertions are evaluated in float64; in the default float32 config
# the benign re-association drift is ~2e-7 and the comparison is meaningless.
jax.config.update("jax_enable_x64", True)
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
    """One SSP-RK3 step: scan variant matches unrolled variant to ~1e-9."""
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
        np.testing.assert_allclose(
            np.asarray(ls), np.asarray(li), rtol=1e-9, atol=1e-10,
            err_msg=(
                "scan-folded SSP-RK3 drifted >1e-9 from the unrolled SSP-RK3 "
                "reference.  The two are the same math (XLA may re-associate "
                "the scan body's FMAs); a drift this large is a real "
                "regression that could affect AMIP scientific validation."
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
        np.testing.assert_allclose(
            np.asarray(ls), np.asarray(li), rtol=1e-8, atol=1e-9,
            err_msg=(
                "10-step drift between scan-folded and unrolled SSP-RK3 "
                "exceeded 1e-8 (re-association rounding must not compound "
                "into a scientifically meaningful difference)."
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
    # tendency_fn (arg 1) is a Python callable, not an array — mark it static
    # so make_jaxpr does not try to abstract it (JAX 0.10 raises otherwise).
    inline_jaxpr = jax.make_jaxpr(ssp_rk3_step, static_argnums=(1,))(
        state, _polynomial_tendency, 100.0)
    scan_jaxpr = jax.make_jaxpr(ssp_rk3_step_scan, static_argnums=(1,))(
        state, _polynomial_tendency, 100.0)

    n_inline = len(inline_jaxpr.eqns)
    n_scan = len(scan_jaxpr.eqns)
    assert n_scan < n_inline, (
        f"Scan variant ({n_scan} eqns) is NOT smaller than inline "
        f"({n_inline} eqns) — the JIT-compile optimisation has lost "
        "its bite.  The expected ratio is ~3× smaller (the scan body "
        "is one stage, the inline has three stages unrolled)."
    )


def test_scan_preserves_state_dtype_under_mixed_tendency():
    """#835: a float32 state with a float64 tendency leaf must NOT break the
    scan carry.

    The hydrostatic lat-lon C-grid dycore emits a float64 ``p_s`` tendency
    against float32 storage.  ``ssp_rk3_step_scan`` wraps the RK stages in
    ``lax.scan``, which enforces carry-in == carry-out dtype; before the fix
    ``beta * k_axpy`` upcast the stage to float64 and the scan refused to close:
    ``carry[0].p_s has type float32[...] but the corresponding output carry
    component has type float64[...]``.  The ``_comb`` stage now casts the
    tendency leaf to the STATE leaf dtype, so every output leaf keeps its input
    (storage) dtype and the scan closes.
    """
    rng = np.random.default_rng(7)
    state = {
        "u": jnp.asarray(rng.standard_normal((6, 8, 3)), dtype=jnp.float32),
        "p_s": jnp.asarray(1.0e5 + rng.standard_normal((6, 8)), dtype=jnp.float32),
    }

    def _mixed_dtype_tendency(s):
        # Emit a float64 p_s tendency (the #835 dycore behaviour) against the
        # float32 state; the u tendency stays float32.
        return {
            "u": (1e-3 * s["u"]).astype(jnp.float32),
            "p_s": 1e-4 * (s["p_s"].astype(jnp.float64) - 1.0e5),  # float64
        }

    # Must not raise — the scan carry-type equality is the #835 crash.
    out = ssp_rk3_step_scan(state, _mixed_dtype_tendency, 100.0)
    # Every output leaf keeps its INPUT (storage) dtype.
    assert out["u"].dtype == jnp.float32
    assert out["p_s"].dtype == jnp.float32, (
        f"p_s dtype changed to {out['p_s'].dtype}: the float64 tendency leaked "
        "into the float32 scan carry (#835)"
    )
