"""One process driving several devices (FV3 duo face-shard / window layouts):
the land columns are split over the devices and solved under a shard_map.

Runs on however many devices jax sees; run ALONE with
``XLA_FLAGS=--xla_force_host_platform_device_count=6`` to cover 2 and 6.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from legoesm.driver.model_driver import (shard_land_columns,
                                         solve_land_on_devices)


class _Sfc(NamedTuple):
    held: jax.Array
    fallback: jax.Array
    fallback_rejected: jax.Array
    n_held: jax.Array
    n_fallback: jax.Array
    n_fallback_rejected: jax.Array


class _Cfg(NamedTuple):
    scale: jax.Array      # per column, as the packed land config carries
    layers: jax.Array     # per LAYER, deliberately as long as the slot vector
    name: str             # non-array config leaf


def _step(st, cfg, d):
    """Column-local stand-in for the land step: (state, response, carbon,
    surface) with per-column masks and whole-run counters."""
    T = st * cfg.scale + d + jnp.sum(cfg.layers) * 1e-3
    # scan whose carry starts replicated, as the canopy stability iteration
    T, _ = jax.lax.scan(lambda c, _: (0.5 * (c + T), None),
                        jnp.zeros(T.shape, T.dtype), None, length=2)
    held = T > 1.0
    fb = T < 0.6
    rej = (T > 0.8) & (T < 1.2)
    return (T, T * 2.0, None,
            _Sfc(held, fb, rej, jnp.sum(held.astype(jnp.int32)),
                 jnp.sum(fb.astype(jnp.int32)), jnp.sum(rej.astype(jnp.int32))))


def test_shares_cover_every_column_once_in_order():
    cols = np.array([3, 4, 8, 9, 11, 15, 16])
    for n in (1, 2, 3, 7):
        slots, keep = shard_land_columns(cols, n)
        assert slots.size % n == 0
        np.testing.assert_array_equal(slots[keep], cols)


@pytest.mark.parametrize("n_dev", [1, 2, 6])
def test_sharded_solve_is_the_one_call_solve(n_dev):
    if jax.device_count() < n_dev:
        pytest.skip(f"need {n_dev} devices (XLA_FLAGS "
                    "--xla_force_host_platform_device_count)")
    rng = np.random.default_rng(1)
    cols = np.sort(rng.choice(100, 37, replace=False))   # uneven shares
    st_full = jnp.asarray(rng.uniform(0.0, 1.0, 100))
    scale_full = jnp.asarray(rng.uniform(0.5, 1.5, 100))
    slots, keep = shard_land_columns(cols, n_dev)
    layers = jnp.asarray(rng.uniform(0.0, 1.0, slots.size))  # decoy size
    # compiled on both sides (eager vs jit differs by FMA fusion alone)
    ref = jax.jit(lambda s, c, d: _step(s, _Cfg(c, layers, "x"), d))(
        st_full[cols], scale_full[cols], jnp.asarray(0.25))
    for nm in ("held", "fallback", "fallback_rejected"):
        assert 0 < int(np.sum(np.asarray(getattr(ref[3], nm)))) < cols.size
    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("land",))
    is_col = (True, _Cfg(True, False, False), False)
    got = jax.jit(lambda s, c, d: solve_land_on_devices(
        _step, (s, _Cfg(c, layers, "x"), d), is_col, mesh, keep))(
        st_full[slots], scale_full[slots], jnp.asarray(0.25))
    for a, b in zip(jax.tree_util.tree_leaves(ref),
                    jax.tree_util.tree_leaves(got)):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("bad", [
    lambda r: jnp.sum(r),                           # column-dependent scalar
    lambda r: jnp.stack([r, r]),                    # columns not leading
])
def test_a_shared_output_is_refused(bad):
    def step(st, cfg, d):
        o = _step(st, cfg, d)
        return (o[0], bad(o[1]), o[2], o[3])
    cols = np.arange(6)
    slots, keep = shard_land_columns(cols, 1)
    mesh = Mesh(np.array(jax.devices()[:1]), ("land",))
    x = jnp.ones(slots.size)
    with pytest.raises(ValueError, match=r"\[.\[1\].\] are not per column"):
        solve_land_on_devices(step, (x, _Cfg(x, jnp.ones(3), "x"),
                                     jnp.asarray(0.0)),
                              (True, _Cfg(True, False, False), False),
                              mesh, keep)


def test_land_split_reuses_the_step_mesh():
    """One process driving the duo's devices: the land split must run on
    the dycore's own step mesh, since the land solve sits inside a jit
    already on that mesh and a second mesh over the same devices fails
    lowering ("mesh should be the same across the entire program").  Runs
    with 2 forced CPU devices in a subprocess; 1-D face mesh and the 3-D
    window-layout shape; result and gradient equal the one-call solve.

    Non-vacuous: make land_device_mesh build Mesh(devs, ("land",)) again and
    the jit raises.

    KNOWN DIFFERENCE: this split-vs-one-call check is exact, but a whole
    1-vs-2-device duo run is not bitwise (summation order in the dynamics,
    fp32 radiation); the measured gaps and their bounds are in
    scripts/validate/validate_duo_device_parity.py.
    """
    body = f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("t", {__file__!r})
        t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
        import jax, jax.numpy as jnp, numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from legoesm.driver.model_driver import (
            land_device_mesh, shard_land_columns, solve_land_on_devices)
        devs = list(jax.devices())
        rng = np.random.default_rng(1)
        cols = np.sort(rng.choice(100, 37, replace=False))
        st = jnp.asarray(rng.uniform(0.0, 1.0, 100))
        sc = jnp.asarray(rng.uniform(0.5, 1.5, 100))
        slots, keep = shard_land_columns(cols, 2)
        layers = jnp.asarray(rng.uniform(0.0, 1.0, slots.size))
        one = lambda s, c: t._step(s, t._Cfg(c, layers, "x"), 0.25)
        ref = jax.jit(one)(st[cols], sc[cols])
        gref = jax.grad(lambda s: jnp.sum(one(s, sc[cols])[0]))(st[cols])
        is_col = (True, t._Cfg(True, False, False), False)
        for step in (Mesh(np.array(devs), ("face",)),
                     Mesh(np.array(devs).reshape(2, 1, 1),
                          ("face", "ti", "tj"))):
            mesh = land_device_mesh(step, devs)
            pin = NamedSharding(step, P(tuple(step.axis_names)))
            def split(s, c):
                s = jax.lax.with_sharding_constraint(s, pin)
                return solve_land_on_devices(
                    t._step, (s, t._Cfg(c, layers, "x"), 0.25), is_col,
                    mesh, keep)
            # committed to the step mesh, as the duo physics inputs are
            s_in = jax.device_put(st[slots], pin)
            got = jax.jit(split)(s_in, sc[slots])
            same = all(np.array_equal(np.asarray(a), np.asarray(b))
                       for a, b in zip(jax.tree_util.tree_leaves(ref),
                                       jax.tree_util.tree_leaves(got)))
            g = jax.grad(lambda s: jnp.sum(jax.jit(split)(s, sc[slots])[0]))(
                s_in)
            print("SAME", mesh is step and same)
            print("GRAD", np.array_equal(np.asarray(g)[keep], np.asarray(gref)))
        print("LAND", land_device_mesh(Mesh(np.array(devs), ("face",)),
                                       devs[:1]).axis_names == ("land",))
    """
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                         env=env, capture_output=True, text=True, timeout=600,
                         cwd=Path(__file__).resolve().parents[2])
    assert out.returncode == 0, out.stderr[-3000:]
    for tag in ("SAME True", "GRAD True"):
        assert out.stdout.count(tag) == 2, out.stdout
    assert "LAND True" in out.stdout, out.stdout
