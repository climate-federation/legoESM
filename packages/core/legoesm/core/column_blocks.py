"""Column-block map with per-block rematerialisation.

A column kernel (a function whose columns are independent) evaluated over
blocks of ``block_size`` columns with ``jax.lax.map``, each block wrapped in
``jax.checkpoint``: reverse mode stores each block's INPUTS and recomputes the
block, so a gradient holds one block's residuals instead of every column's.
The forward values are the unblocked call's (columns are independent).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["map_column_blocks"]


def map_column_blocks(fn, ncol: int, block_size: int, *args):
    """``fn(*args)`` over blocks of ``block_size`` columns, each remat'd.

    Inputs: every array leaf of ``args`` whose leading axis is ``ncol`` is
    split into blocks (the last block padded by repeating the last column;
    the padding is discarded, so it carries no cotangent); every other leaf
    is passed whole to each block.  Outputs: a leaf whose leading axis is
    ``ncol`` in ``jax.eval_shape(fn, *args)`` is reassembled; any other
    output leaf is taken from block 0, so it must not depend on the column
    data (a reduction over columns inside ``fn`` would be wrong here and
    cannot be detected).  ``block_size <= 0`` or ``>= ncol``: ``fn(*args)``.
    """
    if block_size <= 0 or block_size >= ncol:
        return fn(*args)
    n_block = -(-ncol // block_size)
    pad = n_block * block_size - ncol

    leaves, tdef = jax.tree_util.tree_flatten(args)
    col = [jnp.ndim(x) >= 1 and jnp.shape(x)[0] == ncol for x in leaves]

    def split(x):
        if pad:
            x = jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)])
        return x.reshape((n_block, block_size) + x.shape[1:])

    xs = [split(x) for x, c in zip(leaves, col) if c]

    @jax.checkpoint
    def one(xs_b):
        it = iter(xs_b)
        return fn(*jax.tree_util.tree_unflatten(
            tdef, [next(it) if c else x for x, c in zip(leaves, col)]))

    out_b = jax.lax.map(one, xs)
    shapes = jax.eval_shape(fn, *args)

    def join(o, s):
        if len(s.shape) >= 1 and s.shape[0] == ncol:
            return o.reshape((n_block * block_size,) + o.shape[2:])[:ncol]
        return o[0]
    return jax.tree_util.tree_map(join, out_b, shapes)
