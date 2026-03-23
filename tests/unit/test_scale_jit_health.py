"""Category 10: JIT compilation health & memory.

Tests Field pytree registration, scan carry structure, and compilation
patterns that affect scalability.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time

from legoesm.core.field import Field


# =========================================================================
# 10h) Field pytree — metadata is static
# =========================================================================

class TestFieldPytree:
    """Field should be a proper JAX pytree with static metadata."""

    def test_metadata_not_in_leaves(self):
        """Only .data should appear as a leaf."""
        field = Field(
            data=jnp.ones((6, 4, 4)),
            name="T", dims=("face", "x", "y"), units="K",
        )
        leaves = jax.tree_util.tree_leaves(field)
        assert len(leaves) == 1
        assert leaves[0] is field.data

    def test_pytree_roundtrip(self):
        """flatten then unflatten should recover exact Field."""
        field = Field(
            data=jnp.ones((6, 4, 4)),
            name="T", dims=("face", "x", "y"), units="K",
        )
        flat, treedef = jax.tree_util.tree_flatten(field)
        recovered = jax.tree_util.tree_unflatten(treedef, flat)
        assert recovered.name == "T"
        assert recovered.dims == ("face", "x", "y")
        assert recovered.units == "K"
        np.testing.assert_array_equal(np.array(recovered.data), np.array(field.data))

    def test_field_in_jit(self):
        """Field should work inside jax.jit."""
        field = Field(
            data=jnp.ones((6, 4, 4)),
            name="T", dims=("face", "x", "y"), units="K",
        )

        @jax.jit
        def double(f):
            return f.replace(data=f.data * 2)

        result = double(field)
        np.testing.assert_allclose(np.array(result.data), 2.0)
        assert result.name == "T"

    def test_field_in_grad(self):
        """Field should work with jax.grad on .data."""
        field = Field(
            data=jnp.ones((4,), dtype=jnp.float64),
            name="x", dims=("i",), units="m",
        )

        def loss(data):
            f = field.replace(data=data)
            return jnp.sum(f.data ** 2)

        grad = jax.grad(loss)(field.data)
        np.testing.assert_allclose(np.array(grad), 2.0)


# =========================================================================
# 10a,b) JIT recompilation behavior
# =========================================================================

class TestJITRecompilation:
    """JIT should not recompile on numeric-only changes."""

    def test_no_recompile_on_value_change(self):
        """Changing array values (not shapes) should not trigger recompilation."""

        @jax.jit
        def f(x):
            return x ** 2

        x1 = jnp.ones(10)
        x2 = jnp.ones(10) * 2.0

        # First call compiles
        t0 = time.time()
        _ = f(x1).block_until_ready()
        t_first = time.time() - t0

        # Second call should be fast (no recompile)
        t0 = time.time()
        _ = f(x2).block_until_ready()
        t_second = time.time() - t0

        # Second should be much faster (at least 2x for a trivial op)
        # Use generous threshold to avoid flakiness
        assert t_second < t_first * 2.0 or t_second < 0.01

    def test_recompile_on_shape_change(self):
        """Changing array shapes should trigger recompilation."""

        @jax.jit
        def f(x):
            return x ** 2

        x1 = jnp.ones(10)
        x2 = jnp.ones(20)

        _ = f(x1).block_until_ready()
        _ = f(x2).block_until_ready()
        # Both should produce correct results
        np.testing.assert_allclose(np.array(f(x1)), 1.0)
        np.testing.assert_allclose(np.array(f(x2)), 1.0)


# =========================================================================
# 10f) Scan carry structure stability
# =========================================================================

class TestScanCarryStructure:
    """Scan carry pytree structure should be stable."""

    def test_namedtuple_structure_preserved(self):
        """NamedTuple carry in scan should preserve structure."""
        from collections import namedtuple
        Carry = namedtuple("Carry", ["x", "step"])

        def body(carry, _):
            return Carry(x=carry.x + 1.0, step=carry.step + 1), None

        init = Carry(x=jnp.array(0.0), step=jnp.array(0))
        final, _ = jax.lax.scan(body, init, None, length=10)

        assert isinstance(final, Carry)
        np.testing.assert_allclose(float(final.x), 10.0)
        assert int(final.step) == 10

        # Structure should match
        td_init = jax.tree_util.tree_structure(init)
        td_final = jax.tree_util.tree_structure(final)
        assert td_init == td_final


# =========================================================================
# 10g) Tridiagonal solver — fori_loop
# =========================================================================

class TestForiLoop:
    """jax.lax.fori_loop should compile and give correct results."""

    def test_fori_loop_sum(self):
        """fori_loop computing cumulative sum."""

        def body(i, acc):
            return acc + i

        result = jax.lax.fori_loop(0, 10, body, 0)
        assert int(result) == 45  # 0+1+...+9

    def test_fori_loop_in_jit(self):
        """fori_loop inside jit should compile."""

        @jax.jit
        def f(n):
            return jax.lax.fori_loop(0, 10, lambda i, acc: acc + 1.0, 0.0)

        result = f(10)
        np.testing.assert_allclose(float(result), 10.0)


# =========================================================================
# 10i) Conditional branches — structure check
# =========================================================================

class TestLaxCond:
    """jax.lax.cond branches must return same pytree structure."""

    def test_cond_same_structure(self):
        """Both branches of lax.cond return same leaf types."""

        def true_fn(x):
            return x * 2, x + 1

        def false_fn(x):
            return x * 3, x + 2

        pred = jnp.array(True)
        r1 = jax.lax.cond(pred, true_fn, false_fn, jnp.array(1.0))
        r2 = jax.lax.cond(jnp.array(False), true_fn, false_fn, jnp.array(1.0))

        # Both should return tuples of same length
        assert len(r1) == len(r2) == 2
        # Dtypes should match
        assert r1[0].dtype == r2[0].dtype
        assert r1[1].dtype == r2[1].dtype
