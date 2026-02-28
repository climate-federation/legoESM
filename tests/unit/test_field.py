"""Unit tests for the Field class."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field, zeros_field, ones_field


class TestField:
    """Tests for Field pytree and arithmetic."""

    def test_creation(self):
        data = jnp.ones((6, 4, 4))
        f = Field(data=data, name="test", dims=("face", "x", "y"), units="m")
        assert f.name == "test"
        assert f.shape == (6, 4, 4)
        assert f.units == "m"
        assert f.staggering == "cell"

    def test_immutability(self):
        f = Field(data=jnp.zeros((2, 2)), name="test")
        with pytest.raises(AttributeError):
            f.data = jnp.ones((2, 2))

    def test_replace(self):
        f = Field(data=jnp.zeros((2, 2)), name="old", units="m")
        f2 = f.replace(name="new", units="km")
        assert f2.name == "new"
        assert f2.units == "km"
        assert f.name == "old"  # Original unchanged

    def test_pytree_flatten_unflatten(self):
        data = jnp.array([1.0, 2.0, 3.0])
        f = Field(data=data, name="test", dims=("x",), units="m")
        children, aux = f.tree_flatten()
        f2 = Field.tree_unflatten(aux, children)
        assert f2.name == "test"
        assert jnp.allclose(f2.data, data)

    def test_jit_compatible(self):
        f = Field(data=jnp.array([1.0, 2.0, 3.0]), name="test")

        @jax.jit
        def double(field):
            return field * 2.0

        result = double(f)
        assert jnp.allclose(result.data, jnp.array([2.0, 4.0, 6.0]))

    def test_grad_compatible(self):
        """jax.grad should work through Field operations."""
        def loss(x):
            f = Field(data=x, name="test")
            return jnp.sum(f.data ** 2)

        grad_fn = jax.grad(loss)
        x = jnp.array([1.0, 2.0, 3.0])
        grads = grad_fn(x)
        assert jnp.allclose(grads, 2 * x)

    def test_arithmetic_add(self):
        a = Field(data=jnp.array([1.0, 2.0]), name="a")
        b = Field(data=jnp.array([3.0, 4.0]), name="b")
        c = a + b
        assert jnp.allclose(c.data, jnp.array([4.0, 6.0]))

    def test_arithmetic_scalar(self):
        f = Field(data=jnp.array([1.0, 2.0]), name="f")
        result = f * 3.0
        assert jnp.allclose(result.data, jnp.array([3.0, 6.0]))

    def test_arithmetic_sub(self):
        a = Field(data=jnp.array([5.0, 3.0]), name="a")
        b = Field(data=jnp.array([1.0, 1.0]), name="b")
        c = a - b
        assert jnp.allclose(c.data, jnp.array([4.0, 2.0]))

    def test_arithmetic_div(self):
        f = Field(data=jnp.array([4.0, 6.0]), name="f")
        result = f / 2.0
        assert jnp.allclose(result.data, jnp.array([2.0, 3.0]))

    def test_neg(self):
        f = Field(data=jnp.array([1.0, -2.0]), name="f")
        result = -f
        assert jnp.allclose(result.data, jnp.array([-1.0, 2.0]))

    def test_zeros_field(self):
        f = zeros_field((3, 4), name="zero", dims=("x", "y"), units="m")
        assert f.shape == (3, 4)
        assert jnp.allclose(f.data, 0.0)

    def test_ones_field(self):
        f = ones_field((2, 3), name="one", units="1")
        assert jnp.allclose(f.data, 1.0)

    def test_astype(self):
        f = Field(data=jnp.ones((2,), dtype=jnp.float32), name="f")
        f16 = f.astype(jnp.bfloat16)
        assert f16.dtype == jnp.bfloat16
