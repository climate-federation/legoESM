"""map_column_blocks: an independent-column kernel evaluated in remat'd
column blocks equals the plain call (forward and gradient), passes
non-column leaves through, and stores block inputs, not residuals."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core.column_blocks import map_column_blocks  # noqa: E402

NCOL, NLEV = 11, 6


def kernel(x, w, scale):
    """Per-column: an upward scan with a nonlinear carry (residuals per
    level); ``w`` is per-level (not a column leaf); returns a column field,
    a per-column scalar and a column-independent leaf."""
    def body(c, xk):
        c = jnp.tanh(c + scale * xk)
        return c, c * w[0]
    _, y = jax.lax.scan(body, jnp.zeros(x.shape[0]), x.T)
    return {"y": y.T * w, "top": y[-1], "w2": 2.0 * w}


def _inputs():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((NCOL, NLEV))),
            jnp.asarray(rng.uniform(0.5, 1.5, NLEV)), 0.7)


def test_blocks_equal_the_plain_call_forward_and_gradient():
    x, w, s = _inputs()
    ref = kernel(x, w, s)
    for block in (1, 4, 5, NCOL, 64):     # padded, exact-fit and passthrough
        out = jax.jit(lambda a, b: map_column_blocks(kernel, NCOL, block, a, b, s))(x, w)
        for k in ref:
            np.testing.assert_allclose(np.asarray(out[k]), np.asarray(ref[k]),
                                       rtol=0.0, atol=1e-15, err_msg=f"{block} {k}")

        def loss(a, b, _blk=block):
            o = map_column_blocks(kernel, NCOL, _blk, a, b, s)
            return jnp.sum(jnp.sin(o["y"])) + jnp.sum(o["top"] ** 2) + jnp.sum(o["w2"])
        g = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, w)
        g_ref = jax.grad(lambda a, b: jnp.sum(jnp.sin(kernel(a, b, s)["y"]))
                         + jnp.sum(kernel(a, b, s)["top"] ** 2)
                         + jnp.sum(kernel(a, b, s)["w2"]), argnums=(0, 1))(x, w)
        for gi, ri in zip(g, g_ref):
            np.testing.assert_allclose(np.asarray(gi), np.asarray(ri),
                                       rtol=0.0, atol=1e-13, err_msg=f"{block} grad")


def test_gradient_stacks_block_inputs_not_residuals():
    """With the remat, the block loop stacks its outputs and its inputs only
    (measured 816 + 576 bytes here); without it, also the inner scan's
    per-level residuals (measured 1968 bytes in the outer loop)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "scripts" / "validate"))
    import production_ad_map as pam
    x, w, s = _inputs()
    block, n_block = 4, 3

    def loss(a):
        return jnp.sum(jnp.sin(map_column_blocks(kernel, NCOL, block, a, w, s)["y"]))
    rows = pam.loop_census(jax.grad(loss), x, top=0)
    outer = [r for r in rows if r["length"] == n_block]
    assert outer
    outputs = n_block * (block * NLEV + block + NLEV) * 8
    inputs = n_block * block * NLEV * 8
    assert max(r["stacked_mib"] for r in outer) * 2**20 <= outputs + inputs
