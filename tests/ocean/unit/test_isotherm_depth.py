"""RULE 5 tests for ``legoesm.ocean.diagnostics.isotherm_depth`` (smooth D20).

1. width 0 (hard limit) matches an independent numpy first-crossing linear interpolation;
2. width 0.5 K (default) differs from it by the RECORDED per-column gap (measured 2026-11-11);
3. the default is differentiable: finite nonzero derivative, check_grads order 2.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from legoesm.ocean.diagnostics import isotherm_depth

jax.config.update("jax_enable_x64", True)

Z = np.array([5.0, 15.0, 30.0, 50.0, 80.0, 120.0, 170.0, 230.0, 300.0, 400.0])


def _columns():
    c = []
    c.append(27.0 - 12.0 * (1 + np.tanh((Z - 120.0) / 40.0)) / 2)   # smooth thermocline
    c.append(np.where(Z < 60.0, 28.0, 12.0))                          # sharp step
    c.append(np.full(Z.size, 25.0))                                   # all warm
    c.append(np.full(Z.size, 10.0))                                   # all cold
    c.append(np.array([19.0, 23.0, 24.0, 22.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0]))  # inversion
    c.append(np.array([26.0, 25.0, 17.0, 16.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0]))  # warm-cold-warm
    return np.stack(c)


def _reference(T):
    out = []
    for col in T:
        d = None
        for k in range(Z.size - 1):
            if col[k] > 20.0 >= col[k + 1]:
                d = Z[k] + (Z[k + 1] - Z[k]) * (col[k] - 20.0) / (col[k] - col[k + 1])
                break
        out.append(d if d is not None else (Z[-1] if col[0] > 20.0 else 0.0))
    return np.array(out)


def test_hard_limit_matches_reference():
    T = _columns()
    got = np.asarray(isotherm_depth(jnp.asarray(T), jnp.asarray(Z), width_K=0.0))
    np.testing.assert_allclose(got, _reference(T), rtol=1e-12, atol=1e-9)


# measured gap (default 0.5 K - exact) per fixture column [m], job 28916947 (2026-11-12), after
# review round 1 replaced the log-sum-exp max / deepest-wet T with weighted-mean T_hi / T_lo:
# smooth thermocline, sharp step, all warm, all cold, inversion, warm-cold-warm
RECORDED_GAP = [-1.3131472340490404, -0.0035961752220572407, -0.02928093716428748,
                1.0534785127658153e-06, -0.20518495254337665, 0.15934645263353175]


def test_default_gap_recorded_per_column():
    T = _columns()
    gap = (np.asarray(isotherm_depth(jnp.asarray(T), jnp.asarray(Z)))
           - np.asarray(isotherm_depth(jnp.asarray(T), jnp.asarray(Z), width_K=0.0)))
    print("MEASURED_GAP", repr(gap.tolist()))
    np.testing.assert_allclose(gap, RECORDED_GAP, rtol=1e-6, atol=1e-12)


def test_gradient_at_default():
    T = jnp.asarray(_columns()[[0, 1, 4, 5]])
    f = lambda t: jnp.sum(isotherm_depth(t, jnp.asarray(Z)))
    g = np.asarray(jax.grad(f)(T))
    assert np.isfinite(g).all() and (np.abs(g).sum(axis=1) > 0).all()
    check_grads(f, (T,), order=2, modes=["rev"])


def test_dry_nan_levels_do_not_poison_the_column():
    T = _columns()[[0]]
    wet = np.ones_like(T); wet[:, 7:] = 0.0
    T_nan = T.copy(); T_nan[:, 7:] = np.nan
    for w in (0.0, 0.5):
        a = isotherm_depth(jnp.asarray(T_nan), jnp.asarray(Z), wet_mask=jnp.asarray(wet), width_K=w)
        b = isotherm_depth(jnp.asarray(T[:, :7]), jnp.asarray(Z[:7]), width_K=w)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
