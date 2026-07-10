"""Category 8: TPU compatibility tests.

Tests that key computational patterns are compatible with TPU constraints:
- No Python callbacks inside JIT
- Static shapes (no data-dependent shapes)
- scan length must be static
- dtype compatibility (float32/bfloat16)
- SegmentCarry is a proper pytree
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.driver.compiled_segments import SegmentCarry, compute_segment_length
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# SegmentCarry: pure JAX pytree (no Python side effects)
# ---------------------------------------------------------------------------

class TestSegmentCarryPytree:
    """SegmentCarry is a NamedTuple -> valid JAX pytree for lax.scan carry."""

    def _make_carry(self, n=4, nlev=3):
        shape = (6, n, n, nlev)
        shape2d = (6, n, n)
        return SegmentCarry(
            u=jnp.zeros(shape),
            v=jnp.zeros(shape),
            T=jnp.zeros(shape),
            p_s=jnp.zeros(shape2d),
            phis=jnp.zeros(shape2d),
            q_v=jnp.zeros(shape),
            q_c=jnp.zeros(shape),
            q_r=jnp.zeros(shape),
            conv_prog=jnp.zeros((6 * n * n,)),
            held_dT_rad=jnp.zeros(shape),
            held_sw_net_sfc=jnp.zeros(shape2d),
            held_lw_net_sfc=jnp.zeros(shape2d),
            held_sw_up_toa=jnp.zeros(shape2d),
            held_lw_up_toa=jnp.zeros(shape2d),
            held_sw_up_toa_clr=jnp.zeros(shape2d),
            held_lw_up_toa_clr=jnp.zeros(shape2d),
            held_sw_down_toa=jnp.zeros(shape2d),
            step_index=jnp.int32(0),
            target_moisture=jnp.float32(0.0),
            target_mass=jnp.float32(0.0),
            max_cfl=jnp.float32(0.0),
            precip_accum=jnp.zeros(shape2d),
            shflx_accum=jnp.zeros(shape2d),
            lhflx_accum=jnp.zeros(shape2d),
            sw_up_toa_accum=jnp.zeros(shape2d),
            lw_up_toa_accum=jnp.zeros(shape2d),
            sw_up_toa_clr_accum=jnp.zeros(shape2d),
            lw_up_toa_clr_accum=jnp.zeros(shape2d),
            sw_down_toa_accum=jnp.zeros(shape2d),
            sw_net_sfc_accum=jnp.zeros(shape2d),
            lw_net_sfc_accum=jnp.zeros(shape2d),
            t_low_accum=jnp.zeros(shape2d),
            T_land=jnp.zeros(shape2d),
            # Optional double-moment fields populated here so the no-Python-
            # objects invariant is checked on the fully-populated carry.
            q_i=jnp.zeros(shape), q_s=jnp.zeros(shape), q_g=jnp.zeros(shape),
            N_c=jnp.zeros(shape), N_r=jnp.zeros(shape), N_i=jnp.zeros(shape),
            # Optional stateful-physics carries (issue #413), likewise
            # populated so the fully-populated carry is TPU-clean.
            tke=jnp.zeros((6 * n * n, nlev)),
            qke=jnp.zeros((6 * n * n, nlev)),
            gwd_spectrum=jnp.zeros((6 * n * n, 1, 1)),
            conv_precip_prev=jnp.zeros(shape2d),
            w_land=jnp.zeros(shape2d),
        )

    def test_is_namedtuple(self):
        carry = self._make_carry()
        assert hasattr(carry, '_fields')
        assert 'u' in carry._fields

    def test_pytree_flatten_unflatten(self):
        carry = self._make_carry()
        leaves, treedef = jax.tree.flatten(carry)
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, SegmentCarry)
        for orig, recon in zip(carry, reconstructed):
            np.testing.assert_array_equal(orig, recon)

    def test_all_leaves_are_arrays(self):
        carry = self._make_carry()
        leaves = jax.tree.leaves(carry)
        for leaf in leaves:
            assert isinstance(leaf, jax.Array), f"Non-array leaf: {type(leaf)}"

    def test_no_python_objects_in_carry(self):
        """Carry should have only JAX arrays — no strings, lists, etc.  Optional
        fields may be absent (None ⇒ empty pytree subtree); the optional multilayer
        land state is a pytree whose leaves must themselves all be arrays."""
        carry = self._make_carry()
        for field_name in carry._fields:
            val = getattr(carry, field_name)
            if val is None:                      # optional field absent
                continue
            leaves = jax.tree.leaves(val)
            assert leaves and all(isinstance(leaf, jax.Array) for leaf in leaves), (
                f"Field '{field_name}' has non-array leaves: {type(val)}"
            )


# ---------------------------------------------------------------------------
# Static shapes: no data-dependent shapes
# ---------------------------------------------------------------------------

class TestStaticShapes:
    """Key operations must not produce data-dependent shapes."""

    def test_pad_halo_static_shape(self):
        """pad_halo output shape depends only on input shape, not values."""
        from legoesm.grids.halo import pad_halo
        data1 = jnp.ones((6, 4, 4))
        data2 = jnp.zeros((6, 4, 4))
        p1 = pad_halo(data1)
        p2 = pad_halo(data2)
        assert p1.shape == p2.shape == (6, 6, 6)

    def test_field_replace_preserves_shape(self):
        f = Field(data=jnp.zeros((6, 4, 4)), name="T", units="K")
        new_data = jnp.ones((6, 4, 4))
        f2 = f.replace(data=new_data)
        assert f2.shape == f.shape

    def test_jit_traced_shapes_are_static(self):
        """A simple function should compile without shape errors."""
        @jax.jit
        def step(x):
            return x + 1.0
        x = jnp.zeros((6, 4, 4))
        y = step(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# scan length must be static
# ---------------------------------------------------------------------------

class TestScanLength:
    def test_compute_segment_length_returns_int(self):
        seg = compute_segment_length(10, 50, 5)
        assert isinstance(seg, int)
        assert seg >= 1

    def test_segment_length_gcd(self):
        """Segment length should be GCD of intervals."""
        seg = compute_segment_length(12, 18, 6)
        assert seg == 6

    def test_segment_length_disabled(self):
        """With all zeros, should return 1."""
        seg = compute_segment_length(0, 0, 0)
        assert seg == 1

    def test_scan_with_static_length(self):
        """jax.lax.scan requires static length — test it works."""
        def body(carry, _):
            return carry + 1.0, None
        result, _ = jax.lax.scan(body, 0.0, None, length=5)
        np.testing.assert_allclose(result, 5.0)


# ---------------------------------------------------------------------------
# dtype compatibility
# ---------------------------------------------------------------------------

class TestDtypeCompat:
    def test_float32_operations(self):
        """float32 ops should work in JIT."""
        @jax.jit
        def f(x):
            return x * 2.0 + jnp.sin(x)
        x = jnp.ones(10, dtype=jnp.float32)
        y = f(x)
        assert y.dtype == jnp.float32

    def test_bfloat16_operations(self):
        """bfloat16 should work for ML-style computations."""
        @jax.jit
        def f(x):
            return x * 2.0
        x = jnp.ones(10, dtype=jnp.bfloat16)
        y = f(x)
        assert y.dtype == jnp.bfloat16

    def test_mixed_precision_cast(self):
        """Casting between float32 and float64 should work."""
        x32 = jnp.ones(5, dtype=jnp.float32)
        x64 = x32.astype(jnp.float64)
        assert x64.dtype == jnp.float64
        x32_back = x64.astype(jnp.float32)
        assert x32_back.dtype == jnp.float32

    def test_field_astype(self):
        """Field.astype should correctly cast."""
        f = Field(data=jnp.ones(5, dtype=jnp.float64), name="T")
        f32 = f.astype(jnp.float32)
        assert f32.dtype == jnp.float32
        assert f32.name == "T"


# ---------------------------------------------------------------------------
# No host callbacks in JIT
# ---------------------------------------------------------------------------

class TestNoHostCallbacks:
    def test_pure_jax_in_jit(self):
        """A pure JAX function should not trigger host callbacks."""
        @jax.jit
        def compute(x):
            # This should be entirely device-side, no Python callbacks
            y = jnp.sin(x) + jnp.cos(x)
            z = jnp.where(y > 0, y, -y)
            return jnp.sum(z)
        x = jnp.linspace(0, 2 * jnp.pi, 100)
        result = compute(x)
        assert jnp.isfinite(result)

    def test_lax_cond_no_callback(self):
        """lax.cond should work without host callbacks."""
        @jax.jit
        def f(x, flag):
            return jax.lax.cond(
                flag,
                lambda x: x * 2.0,
                lambda x: x * 3.0,
                x,
            )
        result = f(jnp.array(1.0), True)
        np.testing.assert_allclose(result, 2.0)
