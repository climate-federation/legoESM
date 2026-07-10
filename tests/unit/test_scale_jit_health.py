"""Category 10: JIT compilation health tests.

Tests that:
- No recompilation occurs on numeric-only changes
- donate_argnums works correctly
- Field pytree metadata is static (no spurious recompilations)
- jax.lax.scan compiles once and reuses
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.driver.compiled_segments import SegmentCarry


# ---------------------------------------------------------------------------
# Field pytree: metadata is static
# ---------------------------------------------------------------------------

class TestFieldPytreeStatic:
    """Field aux_data (name, dims, units, etc.) is static => metadata
    changes cause recompilation but value changes do not."""

    def test_same_metadata_same_trace(self):
        """Two Fields with same metadata should compile to same program."""
        call_count = [0]

        @jax.jit
        def f(field):
            call_count[0] += 1  # only counts Python-level calls
            return field.data * 2.0

        f1 = Field(data=jnp.ones(3), name="T", units="K")
        f2 = Field(data=jnp.array([2.0, 3.0, 4.0]), name="T", units="K")

        _ = f(f1)
        c1 = call_count[0]
        _ = f(f2)
        c2 = call_count[0]
        # Second call should NOT trigger recompilation
        assert c2 == c1, "Recompiled when only data changed"

    def test_different_metadata_retrace(self):
        """Different metadata should trigger recompilation."""
        call_count = [0]

        @jax.jit
        def f(field):
            call_count[0] += 1
            return field.data * 2.0

        f1 = Field(data=jnp.ones(3), name="T", units="K")
        f2 = Field(data=jnp.ones(3), name="pressure", units="Pa")

        _ = f(f1)
        c1 = call_count[0]
        _ = f(f2)
        c2 = call_count[0]
        # Different metadata -> should recompile
        assert c2 > c1, "Did not recompile with different metadata"

    def test_field_flatten_unflatten_roundtrip(self):
        f = Field(data=jnp.ones(5), name="T", dims=("x",), units="K",
                  long_name="temperature", staggering="cell")
        leaves, treedef = jax.tree.flatten(f)
        assert len(leaves) == 1  # only data is a leaf
        f2 = treedef.unflatten(leaves)
        assert f2.name == f.name
        assert f2.dims == f.dims
        assert f2.units == f.units
        np.testing.assert_array_equal(f2.data, f.data)


# ---------------------------------------------------------------------------
# No recompilation on numeric changes
# ---------------------------------------------------------------------------

class TestNoRecompilationOnNumeric:
    """JIT should not recompile when only numeric values change
    (same shape and dtype)."""

    def test_no_recompile_on_value_change(self):
        trace_count = [0]

        @jax.jit
        def step(x, dt):
            trace_count[0] += 1
            return x + dt

        x = jnp.ones((6, 4, 4))
        _ = step(x, jnp.float32(1.0))
        c1 = trace_count[0]
        _ = step(x * 2.0, jnp.float32(2.0))
        c2 = trace_count[0]
        assert c2 == c1, "Recompiled when only values changed"

    def test_recompile_on_shape_change(self):
        trace_count = [0]

        @jax.jit
        def step(x):
            trace_count[0] += 1
            return x * 2.0

        _ = step(jnp.ones(5))
        c1 = trace_count[0]
        _ = step(jnp.ones(10))  # different shape
        c2 = trace_count[0]
        assert c2 > c1, "Did not recompile on shape change"


# ---------------------------------------------------------------------------
# donate_argnums
# ---------------------------------------------------------------------------

class TestDonateArgnums:
    def test_donate_works_in_jit(self):
        """donate_argnums should be accepted by jit."""
        @jax.jit
        def step(state, dt):
            return state + dt

        # donate_argnums is specified at jit time in modern JAX
        step_donate = jax.jit(step, donate_argnums=(0,))
        x = jnp.ones(10)
        y = step_donate(x, 1.0)
        np.testing.assert_allclose(y, 2.0)


# ---------------------------------------------------------------------------
# lax.scan compiles once
# ---------------------------------------------------------------------------

class TestScanCompilation:
    def test_scan_single_compilation(self):
        """lax.scan should compile once and run n_steps without retracing.
        Note: scan length must be a static Python int (not traced)."""
        trace_count = [0]

        def body(carry, _):
            trace_count[0] += 1  # counts Python tracing, not execution
            return carry + 1.0, carry

        @jax.jit
        def run(init):
            final, trajectory = jax.lax.scan(body, init, None, length=10)
            return final, trajectory

        init = jnp.array(0.0)
        final, traj = run(init)
        np.testing.assert_allclose(final, 10.0)
        assert traj.shape == (10,)

        # Run again — should NOT retrace body
        c1 = trace_count[0]
        final2, _ = run(init)
        c2 = trace_count[0]
        assert c2 == c1, "scan body was retraced on second call"


# ---------------------------------------------------------------------------
# SegmentCarry: consistent pytree structure
# ---------------------------------------------------------------------------

class TestSegmentCarryJIT:
    def _make_carry(self, n=4, nlev=3):
        shape = (6, n, n, nlev)
        shape2d = (6, n, n)
        return SegmentCarry(
            u=jnp.zeros(shape, dtype=jnp.float32),
            v=jnp.zeros(shape, dtype=jnp.float32),
            T=jnp.zeros(shape, dtype=jnp.float32),
            p_s=jnp.zeros(shape2d, dtype=jnp.float32),
            phis=jnp.zeros(shape2d, dtype=jnp.float32),
            q_v=jnp.zeros(shape, dtype=jnp.float32),
            q_c=jnp.zeros(shape, dtype=jnp.float32),
            q_r=jnp.zeros(shape, dtype=jnp.float32),
            conv_prog=jnp.zeros((6 * n * n,), dtype=jnp.float32),
            held_dT_rad=jnp.zeros(shape, dtype=jnp.float32),
            held_sw_net_sfc=jnp.zeros(shape2d, dtype=jnp.float32),
            held_lw_net_sfc=jnp.zeros(shape2d, dtype=jnp.float32),
            held_sw_up_toa=jnp.zeros(shape2d, dtype=jnp.float32),
            held_lw_up_toa=jnp.zeros(shape2d, dtype=jnp.float32),
            held_sw_up_toa_clr=jnp.zeros(shape2d, dtype=jnp.float32),
            held_lw_up_toa_clr=jnp.zeros(shape2d, dtype=jnp.float32),
            held_sw_down_toa=jnp.zeros(shape2d, dtype=jnp.float32),
            step_index=jnp.int32(0),
            target_moisture=jnp.float32(0.0),
            target_mass=jnp.float32(0.0),
            max_cfl=jnp.float32(0.0),
            precip_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            shflx_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            lhflx_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            sw_up_toa_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            lw_up_toa_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            sw_up_toa_clr_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            lw_up_toa_clr_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            sw_down_toa_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            sw_net_sfc_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            lw_net_sfc_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            t_low_accum=jnp.zeros(shape2d, dtype=jnp.float32),
            T_land=jnp.zeros(shape2d, dtype=jnp.float32),
        )

    def test_carry_through_scan(self):
        """SegmentCarry should pass through lax.scan without issues."""
        carry = self._make_carry()

        def body(c, _):
            new_c = c._replace(
                u=c.u + 1.0,
                step_index=c.step_index + 1,
            )
            return new_c, None

        @jax.jit
        def run(c):
            final, _ = jax.lax.scan(body, c, None, length=3)
            return final

        result = run(carry)
        np.testing.assert_allclose(result.u, 3.0)
        assert int(result.step_index) == 3

    def test_carry_tree_structure_stable(self):
        """Two carries with same shapes should have same tree structure."""
        c1 = self._make_carry()
        c2 = self._make_carry()
        _, td1 = jax.tree.flatten(c1)
        _, td2 = jax.tree.flatten(c2)
        assert td1 == td2
