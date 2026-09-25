"""LEGOESM_SPMD_TREE_PSUM: the butterfly sum in batch_psum_spmd equals psum
to rounding, is bit-identical on every device, differentiates like psum, and
is off by default."""
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent('''
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + os.environ["NDEV"]
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, PartitionSpec as P
    from jax import shard_map
    from legoesm.parallel.reductions import batch_psum_spmd
    mesh = Mesh(np.array(jax.devices()), ("d",))
    def f(x):
        a, b = batch_psum_spmd([x[0], x[1:3]], "d")
        return jnp.concatenate([a[None], b])[None]
    g = jax.jit(shard_map(f, mesh=mesh, in_specs=P("d"), out_specs=P("d")))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(len(jax.devices()), 3)) * 1e3)
    out = np.asarray(g(x))
    ref = x.sum(axis=0)
    loss = lambda y: jnp.sum(g(y) ** 2)
    gr = np.asarray(jax.grad(loss)(x))
    hlo = g.lower(x).compile().as_text()
    print(repr((out.tolist(), np.asarray(ref).tolist(), gr.tolist(), "all-reduce" in hlo, hlo.count("collective-permute"))))
''')


def _run(env_val, ndev=8):
    env = dict(os.environ, JAX_PLATFORMS="cpu", NDEV=str(ndev))
    env.pop("LEGOESM_SPMD_TREE_PSUM", None)
    if env_val is not None:
        env["LEGOESM_SPMD_TREE_PSUM"] = env_val
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    import ast
    import numpy as np
    out, ref, gr, has_ar, n_cp = ast.literal_eval(r.stdout.strip().splitlines()[-1])
    return np.array(out), np.array(ref), np.array(gr), has_ar, n_cp


def test_tree_sum_matches_psum_identical_on_all_devices_and_grad():
    import numpy as np
    o_ps, ref, g_ps, ar_ps, _ = _run(None)             # unset = default psum
    o_tr, _, g_tr, ar_tr, cp_tr = _run("1")
    assert ar_ps and not ar_tr and cp_tr >= 3             # the tree path really ran
    for o in (o_ps, o_tr):
        np.testing.assert_allclose(o, np.broadcast_to(ref, o.shape), rtol=1e-14)
    assert (o_tr == o_tr[0]).all()                       # every device, same bits
    np.testing.assert_allclose(g_tr, g_ps, rtol=1e-13)


def test_bad_value_raises():
    from legoesm.parallel.reductions import _resolve_tree_psum
    assert _resolve_tree_psum("") is False and _resolve_tree_psum("0") is False
    assert _resolve_tree_psum("1") is True
    with pytest.raises(ValueError):
        _resolve_tree_psum("yes")


def test_non_power_of_two_axis_keeps_psum():
    import numpy as np
    out, ref, _, has_ar, _ = _run("1", ndev=6)
    assert has_ar
    np.testing.assert_allclose(out, np.broadcast_to(ref, out.shape), rtol=1e-14)
