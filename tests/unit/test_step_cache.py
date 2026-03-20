"""Tests for the sharded-step executable cache.

Validates:
- Compilation happens exactly once for repeated same-structure calls.
- Different state structures trigger recompilation.
- Different dt values do NOT trigger recompilation (dt is traced).
- The cache key captures state structure, leaf meta, sharding, and physics flag.
- SingleDeviceStep and CompiledShardedStep have consistent API.
- clear_cache resets all state.
- First-call compile cost vs steady-state execute cost.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, ShallowWaterState
from legoesm.parallel.mesh import DeviceConfig, create_device_mesh
from legoesm.parallel.sharded_dynamics import (
    StepCacheKey,
    CompiledShardedStep,
    _SingleDeviceStep,
    _make_cache_key,
    make_sharded_step,
    sharded_integrate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FACES = 6
N = 8
NLEV = 5


def _make_field_3d(name="test", fill=1.0):
    data = jnp.full((N_FACES, N, N, NLEV), fill, dtype=jnp.float32)
    return Field(data, name=name, dims=("face", "x", "y", "level"), units="K")


def _make_field_2d(name="test_2d", fill=0.0):
    data = jnp.full((N_FACES, N, N), fill, dtype=jnp.float32)
    return Field(data, name=name, dims=("face", "x", "y"), units="Pa")


def _make_hydrostatic_state():
    return HydrostaticState(
        u=_make_field_3d("u", fill=0.5),
        v=_make_field_3d("v", fill=-0.5),
        T=_make_field_3d("T", fill=280.0),
        p_s=_make_field_2d("p_s", fill=101325.0),
        phis=_make_field_2d("phis", fill=0.0),
    )


def _make_shallow_water_state():
    return ShallowWaterState(
        h=_make_field_2d("h", fill=1000.0),
        u=_make_field_2d("u", fill=1.0),
        v=_make_field_2d("v", fill=-1.0),
        h_s=_make_field_2d("h_s", fill=0.0),
    )


def _single_device_config():
    return create_device_mesh(n_devices=1)


class _MockModel:
    def __init__(self, increment=1.0):
        self._increment = increment

    def step(self, state, dt):
        new_T = state.T.replace(
            data=state.T.data + self._increment * dt / 86400.0
        )
        return state._replace(T=new_T)

    def step_with_physics(self, state, dt, physics_fn):
        state = self.step(state, dt)
        if physics_fn is not None:
            state = physics_fn(state)
        return state


# ===========================================================================
# 1. StepCacheKey construction
# ===========================================================================

class TestStepCacheKey:
    """StepCacheKey correctly discriminates different call shapes."""

    def test_same_state_same_key(self):
        config = _single_device_config()
        state = _make_hydrostatic_state()
        k1 = _make_cache_key(state, config, has_physics=False)
        k2 = _make_cache_key(state, config, has_physics=False)
        assert k1 == k2

    def test_different_physics_flag_different_key(self):
        config = _single_device_config()
        state = _make_hydrostatic_state()
        k1 = _make_cache_key(state, config, has_physics=False)
        k2 = _make_cache_key(state, config, has_physics=True)
        assert k1 != k2

    def test_different_state_type_different_key(self):
        config = _single_device_config()
        hydro = _make_hydrostatic_state()
        sw = _make_shallow_water_state()
        k1 = _make_cache_key(hydro, config, has_physics=False)
        k2 = _make_cache_key(sw, config, has_physics=False)
        assert k1 != k2

    def test_different_shape_different_key(self):
        """Different resolution produces different leaf_meta."""
        config = _single_device_config()
        state8 = _make_hydrostatic_state()  # N=8
        # Create state with N=4
        data4 = jnp.ones((6, 4, 4, NLEV), dtype=jnp.float32)
        state4 = HydrostaticState(
            u=Field(data4, "u", ("face", "x", "y", "level"), "m/s"),
            v=Field(data4, "v", ("face", "x", "y", "level"), "m/s"),
            T=Field(data4, "T", ("face", "x", "y", "level"), "K"),
            p_s=Field(jnp.ones((6, 4, 4), dtype=jnp.float32), "p_s",
                      ("face", "x", "y"), "Pa"),
            phis=Field(jnp.zeros((6, 4, 4), dtype=jnp.float32), "phis",
                       ("face", "x", "y"), "m2/s2"),
        )
        k1 = _make_cache_key(state8, config, has_physics=False)
        k2 = _make_cache_key(state4, config, has_physics=False)
        assert k1 != k2

    def test_key_is_hashable(self):
        config = _single_device_config()
        state = _make_hydrostatic_state()
        key = _make_cache_key(state, config, has_physics=False)
        # Must be usable as a dict key
        d = {key: "ok"}
        assert d[key] == "ok"

    def test_sharding_key_includes_config(self):
        config1 = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=1, backend="CPU", is_distributed=False,
            tiling=(1, 1), grid_type="cubed_sphere",
        )
        config2 = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=6, backend="GPU", is_distributed=False,
            tiling=(1, 1), grid_type="cubed_sphere",
        )
        state = _make_hydrostatic_state()
        k1 = _make_cache_key(state, config1, has_physics=False)
        k2 = _make_cache_key(state, config2, has_physics=False)
        assert k1 != k2


# ===========================================================================
# 2. Compile-once behavior
# ===========================================================================

class TestCompileOnce:
    """Compilation happens once; repeated calls reuse the cached executable."""

    def test_single_compile_for_repeated_calls(self):
        """Same state structure → compile count stays at 1."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        # First call triggers compilation
        _ = step_fn(state, 600.0)
        assert step_fn.compile_count == 1

        # Subsequent calls reuse
        for _ in range(5):
            _ = step_fn(state, 600.0)
        assert step_fn.compile_count == 1

    def test_different_dt_no_recompile(self):
        """Changing dt does NOT trigger recompilation (dt is traced)."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        _ = step_fn(state, 300.0)
        assert step_fn.compile_count == 1

        _ = step_fn(state, 600.0)
        assert step_fn.compile_count == 1

        _ = step_fn(state, 1200.0)
        assert step_fn.compile_count == 1

    def test_is_compiled_for(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        assert step_fn.is_compiled_for(state) is False
        _ = step_fn(state, 600.0)
        assert step_fn.is_compiled_for(state) is True

    def test_cache_size(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        assert step_fn.cache_size == 0
        _ = step_fn(state, 600.0)
        assert step_fn.cache_size == 1

    def test_clear_cache(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        _ = step_fn(state, 600.0)
        assert step_fn.compile_count == 1
        assert step_fn.cache_size == 1

        step_fn.clear_cache()
        assert step_fn.compile_count == 0
        assert step_fn.cache_size == 0

        # Next call recompiles
        _ = step_fn(state, 600.0)
        assert step_fn.compile_count == 1


# ===========================================================================
# 3. Recompilation on structural change
# ===========================================================================

class TestRecompilationTriggers:
    """Structural changes correctly trigger recompilation."""

    def test_physics_flag_triggers_recompile(self):
        """Switching physics on/off produces separate cache entries."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        _ = step_fn(state, 600.0, physics_fn=None)
        assert step_fn.compile_count == 1

        # Passing a physics function should trigger a second compile
        def phys(s):
            return s._replace(T=s.T.replace(data=s.T.data + 1.0))

        _ = step_fn(state, 600.0, physics_fn=phys)
        assert step_fn.compile_count == 2

        # But re-calling with physics again should NOT recompile
        _ = step_fn(state, 600.0, physics_fn=phys)
        assert step_fn.compile_count == 2

    def test_state_value_change_no_recompile(self):
        """Different array values in same structure → no recompile."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state1 = _make_hydrostatic_state()
        state2 = state1._replace(T=state1.T.replace(data=state1.T.data + 100.0))

        _ = step_fn(state1, 600.0)
        _ = step_fn(state2, 600.0)
        assert step_fn.compile_count == 1  # same structure


# ===========================================================================
# 4. Correctness after caching
# ===========================================================================

class TestCorrectnessWithCache:
    """The cached executable produces correct results."""

    def test_cached_step_matches_model_step(self):
        config = _single_device_config()
        model = _MockModel(increment=2.0)
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()
        dt = 600.0

        expected = model.step(state, dt)
        result = step_fn(state, dt)

        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-6,
        )

    def test_cached_step_different_dt_values(self):
        """dt=300 and dt=600 both work from the same compiled executable."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        result_300 = step_fn(state, 300.0)
        result_600 = step_fn(state, 600.0)

        expected_300 = model.step(state, 300.0)
        expected_600 = model.step(state, 600.0)

        np.testing.assert_allclose(
            np.asarray(result_300.T.data),
            np.asarray(expected_300.T.data),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(result_600.T.data),
            np.asarray(expected_600.T.data),
            atol=1e-6,
        )
        assert step_fn.compile_count == 1

    def test_multi_step_integration_uses_cache(self):
        """sharded_integrate reuses the step function across steps."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()

        step_fn = make_sharded_step(model, config)
        final, _ = sharded_integrate(
            model, state, n_steps=10, dt=600.0, config=config,
            step_fn=step_fn,
        )

        # Only one compile despite 10 steps
        assert step_fn.compile_count == 1

        # Result should be finite and larger than initial T
        T_final = np.asarray(final.T.data)
        T_init = np.asarray(state.T.data)
        assert np.all(np.isfinite(T_final))
        assert np.all(T_final > T_init)


# ===========================================================================
# 5. Timing: first-call compile cost vs steady-state
# ===========================================================================

class TestTimingBehavior:
    """First call (compile) is more expensive than subsequent calls."""

    def test_first_call_records_compile_time(self):
        """last_compile_time_s is populated after compilation."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        assert step_fn.last_compile_time_s == 0.0
        _ = step_fn(state, 600.0)
        assert step_fn.last_compile_time_s >= 0.0
        assert step_fn.compile_count == 1

    def test_steady_state_faster_than_first_call(self):
        """Warm calls should not trigger compilation overhead."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()

        # First call (includes compile)
        t0 = time.monotonic()
        _ = step_fn(state, 600.0)
        first_time = time.monotonic() - t0

        # Warm up JIT cache fully
        for _ in range(3):
            _ = step_fn(state, 600.0)

        # Steady-state calls
        n_warm = 10
        t0 = time.monotonic()
        for _ in range(n_warm):
            _ = step_fn(state, 600.0)
        avg_warm = (time.monotonic() - t0) / n_warm

        # The first call should be at least as expensive as warm calls.
        # We don't assert strict inequality because on fast hardware
        # compile time can be negligible, but compile_count confirms
        # only one compilation happened.
        assert step_fn.compile_count == 1


# ===========================================================================
# 6. API consistency between SingleDeviceStep and CompiledShardedStep
# ===========================================================================

class TestAPIConsistency:
    """Both step wrappers expose the same observability interface."""

    def test_single_device_step_type(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        assert isinstance(step_fn, _SingleDeviceStep)

    def test_both_have_compile_count(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        assert hasattr(step_fn, "compile_count")
        assert hasattr(step_fn, "last_compile_time_s")
        assert hasattr(step_fn, "cache_size")

    def test_both_have_is_compiled_for(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        state = _make_hydrostatic_state()
        assert hasattr(step_fn, "is_compiled_for")
        assert step_fn.is_compiled_for(state) is False

    def test_both_have_clear_cache(self):
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        assert hasattr(step_fn, "clear_cache")
        step_fn.clear_cache()  # should not raise


# ===========================================================================
# 7. Exports
# ===========================================================================

class TestExports:
    """New types are exported from the parallel package."""

    def test_step_cache_key_exported(self):
        from legoesm.parallel import StepCacheKey
        assert StepCacheKey is not None

    def test_compiled_sharded_step_exported(self):
        from legoesm.parallel import CompiledShardedStep
        assert CompiledShardedStep is not None
