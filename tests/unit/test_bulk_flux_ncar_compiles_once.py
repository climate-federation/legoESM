"""An eager ``large_yeager`` MOST call (the hourly 2 m temperature diagnostic)
must not recompile its Obukhov iteration on every call: a per-call loop-body
closure made the eager ``fori_loop`` retrace and recompile each time, 139 ms
per hourly diagnostic on every rank of the MPAS AMIP run."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import monitoring

from legoesm.core.bulk_flux import compute_most_fluxes


def _inputs(seed):
    rng = np.random.default_rng(seed)
    n = 64
    T = 285.0 + 10.0 * rng.random(n)
    return (jnp.asarray(3.0 + 5.0 * rng.random(n)), jnp.asarray(2.0 * rng.random(n) - 1.0),
            jnp.asarray(T), jnp.asarray(0.008 + 0.004 * rng.random(n)),
            jnp.asarray(T + 4.0 * rng.random(n) - 2.0),
            jnp.asarray(0.010 + 0.004 * rng.random(n)), jnp.asarray(1.2 + 0.0 * T))


def test_large_yeager_eager_call_compiles_once():
    first, second = _inputs(0), _inputs(1)
    compiles = []

    def _listen(event, duration, **kw):
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(duration)

    # A persistent cache hit skips the backend compile and would hide the bug.
    was_on = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    monitoring.register_event_duration_secs_listener(_listen)
    try:
        # Positive control: a fresh function must register a compile.
        jax.block_until_ready(jax.jit(lambda x: x * 3.0 + 1.0)(first[2]))
        assert compiles, "listener saw no compile for a fresh function"
        jax.block_until_ready(
            compute_most_fluxes(*first, scheme="large_yeager", return_2m=True))
        compiles.clear()
        out = compute_most_fluxes(*second, scheme="large_yeager", return_2m=True)
        jax.block_until_ready(out)
    finally:
        monitoring.unregister_event_duration_listener(_listen)
        jax.config.update("jax_enable_compilation_cache", was_on)
    assert np.all(np.isfinite(np.asarray(out[-1])))
    assert compiles == [], f"{len(compiles)} backend compiles on a repeated eager call"


@pytest.mark.skipif(not jax.config.jax_enable_x64,
                    reason="np.float64 promotes float32 inputs only with x64 on")
def test_equal_scalars_of_different_type_do_not_share_a_compile():
    """``5.0 == np.float64(5.0)`` but they promote float32 inputs differently,
    so the cached loop must not serve one's compile to the other."""
    a = [x.astype(jnp.float32) for x in _inputs(2)]

    def run(beta):
        return compute_most_fluxes(*a, scheme="large_yeager", return_2m=True,
                                   stable_beta=beta)[-1]

    py = run(5.0)
    np64 = run(np.float64(5.0))
    assert py.dtype != np64.dtype, "control: the two literals must promote differently"
    jax.clear_caches()
    np64_alone = run(np.float64(5.0))
    assert np64.dtype == np64_alone.dtype
    np.testing.assert_array_equal(np.asarray(np64), np.asarray(np64_alone))
