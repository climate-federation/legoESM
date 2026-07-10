"""Tests for sharded dynamics (single-device, verifying API and correctness).

All tests run on a single CPU device.  They validate that the sharded
dynamics API (shard_state, gather_state, make_sharded_step, etc.) is
correct in single-device mode.  Actual multi-device sharding is tested
in CI with real hardware.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, ShallowWaterState
from legoesm.parallel.mesh import DeviceConfig, create_device_mesh
from legoesm.parallel.sharded_dynamics import (
    ShardingSpec,
    _make_sharding_spec,
    shard_state,
    gather_state,
    create_output_shardings,
    make_sharded_step,
    sharded_step_with_halo,
    sharded_integrate,
    sharded_integrate_scan,
    check_sharding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FACES = 6
N = 8
NLEV = 5


def _make_field_3d(name="test", fill=1.0):
    """Create a 3D field (6, N, N, NLEV)."""
    data = jnp.full((N_FACES, N, N, NLEV), fill, dtype=jnp.float64)
    return Field(data, name=name, dims=("face", "x", "y", "level"), units="K")


def _make_field_2d(name="test_2d", fill=0.0):
    """Create a 2D field (6, N, N)."""
    data = jnp.full((N_FACES, N, N), fill, dtype=jnp.float64)
    return Field(data, name=name, dims=("face", "x", "y"), units="Pa")


def _make_hydrostatic_state():
    """Create a minimal HydrostaticState for testing."""
    return HydrostaticState(
        u=_make_field_3d("u", fill=0.5),
        v=_make_field_3d("v", fill=-0.5),
        T=_make_field_3d("T", fill=280.0),
        p_s=_make_field_2d("p_s", fill=101325.0),
        phis=_make_field_2d("phis", fill=0.0),
    )


def _make_shallow_water_state():
    """Create a minimal ShallowWaterState for testing."""
    return ShallowWaterState(
        h=_make_field_2d("h", fill=1000.0),
        u=_make_field_2d("u", fill=1.0),
        v=_make_field_2d("v", fill=-1.0),
        h_s=_make_field_2d("h_s", fill=0.0),
    )


def _single_device_config():
    """Create a single-device config (mesh=None)."""
    return create_device_mesh(n_devices=1)


class _MockModel:
    """Mock dynamics model with step() and step_with_physics()."""

    def __init__(self, increment=1.0):
        self._increment = increment

    def step(self, state, dt):
        """Advance state by adding increment * dt to temperature."""
        new_T = state.T.replace(
            data=state.T.data + self._increment * dt / 86400.0
        )
        return state._replace(T=new_T)

    def step_with_physics(self, state, dt, physics_fn):
        """Step then apply physics forcing."""
        state = self.step(state, dt)
        if physics_fn is not None:
            state = physics_fn(state)
        return state


# ===========================================================================
# TestShardState
# ===========================================================================

class TestShardState:
    """Tests for shard_state and gather_state."""

    def test_shard_state_returns_correct_structure(self):
        """shard_state preserves the pytree structure."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        sharded = shard_state(state, config)

        assert isinstance(sharded, HydrostaticState)
        assert sharded.T.data.shape == state.T.data.shape
        assert sharded.p_s.data.shape == state.p_s.data.shape

    def test_shard_state_single_device_noop(self):
        """On single device (mesh=None), shard_state returns input unchanged."""
        config = _single_device_config()
        assert config.mesh is None

        state = _make_hydrostatic_state()
        sharded = shard_state(state, config)

        # On single device the state should pass through unchanged
        np.testing.assert_array_equal(
            np.asarray(sharded.T.data), np.asarray(state.T.data)
        )
        np.testing.assert_array_equal(
            np.asarray(sharded.p_s.data), np.asarray(state.p_s.data)
        )

    def test_roundtrip_shard_gather_preserves_values(self):
        """shard -> gather round-trip preserves array values exactly."""
        config = _single_device_config()
        state = _make_hydrostatic_state()

        sharded = shard_state(state, config)
        gathered = gather_state(sharded, config)

        for orig, rec in zip(jax.tree.leaves(state), jax.tree.leaves(gathered)):
            np.testing.assert_array_equal(np.asarray(orig), np.asarray(rec))

    def test_shard_state_shallow_water(self):
        """shard_state works with ShallowWaterState."""
        config = _single_device_config()
        state = _make_shallow_water_state()
        sharded = shard_state(state, config)

        assert isinstance(sharded, ShallowWaterState)
        np.testing.assert_array_equal(
            np.asarray(sharded.h.data), np.asarray(state.h.data)
        )

    def test_shard_state_preserves_metadata(self):
        """Field metadata (name, units, dims) is preserved through sharding."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        sharded = shard_state(state, config)

        assert sharded.T.name == state.T.name
        assert sharded.T.units == state.T.units
        assert sharded.T.dims == state.T.dims

    def test_gather_state_single_device_noop(self):
        """gather_state on single device is a no-op."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        gathered = gather_state(state, config)

        np.testing.assert_array_equal(
            np.asarray(gathered.T.data), np.asarray(state.T.data)
        )


# ===========================================================================
# TestMakeShardedStep
# ===========================================================================

class TestMakeShardedStep:
    """Tests for make_sharded_step."""

    def test_make_sharded_step_returns_callable(self):
        """make_sharded_step returns a callable."""
        config = _single_device_config()
        model = _MockModel()
        step_fn = make_sharded_step(model, config)
        assert callable(step_fn)

    def test_sharded_step_produces_correct_result(self):
        """Sharded step matches regular step on single device."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()
        dt = 600.0

        # Regular step
        expected = model.step(state, dt)

        # Sharded step
        step_fn = make_sharded_step(model, config)
        result = step_fn(state, dt)

        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-12,
        )

    def test_sharded_step_with_physics_none(self):
        """Sharded step with physics_fn=None uses plain model.step."""
        config = _single_device_config()
        model = _MockModel()
        state = _make_hydrostatic_state()
        dt = 600.0

        step_fn = make_sharded_step(model, config)
        result = step_fn(state, dt, physics_fn=None)

        expected = model.step(state, dt)
        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-12,
        )

    def test_sharded_step_with_physics_fn_single_device(self):
        """On single device, physics_fn is captured in the jitted closure.

        The single-device path wraps the step in @jax.jit with the
        physics_fn captured in the closure (not passed as a traced
        argument), so passing a callable works correctly.
        """
        config = _single_device_config()
        model = _MockModel()
        state = _make_hydrostatic_state()
        dt = 600.0

        def physics(s):
            return s._replace(T=s.T.replace(data=s.T.data + 1.0))

        step_fn = make_sharded_step(model, config)
        result = step_fn(state, dt, physics_fn=physics)

        # Verify physics was applied: T should be stepped + 1.0
        expected = model.step_with_physics(state, dt, physics)
        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-6,
        )

    def test_sharded_step_preserves_state_type(self):
        """The returned state has the same type as the input."""
        config = _single_device_config()
        model = _MockModel()
        state = _make_hydrostatic_state()
        dt = 600.0

        step_fn = make_sharded_step(model, config)
        result = step_fn(state, dt)

        assert isinstance(result, HydrostaticState)


# ===========================================================================
# TestShardedStepWithHalo
# ===========================================================================

class TestShardedStepWithHalo:
    """Tests for the functional sharded_step_with_halo."""

    def test_functional_step_matches_model_step(self):
        """sharded_step_with_halo on single device matches model.step."""
        config = _single_device_config()
        model = _MockModel()
        state = _make_hydrostatic_state()
        dt = 600.0

        result = sharded_step_with_halo(model, state, dt, config)
        expected = model.step(state, dt)

        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-12,
        )

    def test_functional_step_with_physics(self):
        """sharded_step_with_halo with physics_fn calls step_with_physics."""
        config = _single_device_config()
        model = _MockModel()
        state = _make_hydrostatic_state()
        dt = 600.0

        def physics(s):
            return s._replace(T=s.T.replace(data=s.T.data + 2.0))

        result = sharded_step_with_halo(
            model, state, dt, config, physics_fn=physics
        )
        expected = model.step_with_physics(state, dt, physics)

        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-12,
        )


# ===========================================================================
# TestShardedIntegrate
# ===========================================================================

class TestShardedIntegrate:
    """Tests for multi-step integration."""

    def test_multi_step_integration(self):
        """sharded_integrate runs N steps and returns final state."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()
        dt = 600.0
        n_steps = 3

        final, trajectory = sharded_integrate(
            model, state, n_steps=n_steps, dt=dt, config=config
        )

        # Verify the temperature increased
        T_init = np.asarray(state.T.data)
        T_final = np.asarray(final.T.data)
        assert np.all(T_final > T_init)

        # Trajectory should be empty (save_every=0)
        assert trajectory == []

    def test_multi_step_with_save_every(self):
        """sharded_integrate with save_every records intermediate states."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()
        dt = 600.0
        n_steps = 6

        final, trajectory = sharded_integrate(
            model, state, n_steps=n_steps, dt=dt,
            config=config, save_every=2,
        )

        # Should have saved at steps 2, 4, 6
        assert len(trajectory) == 3

    def test_sharded_integrate_scan_matches_loop(self):
        """sharded_integrate_scan produces same result as loop integration."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()
        dt = 600.0
        n_steps = 4

        # Loop integration
        final_loop, _ = sharded_integrate(
            model, state, n_steps=n_steps, dt=dt, config=config
        )

        # Scan integration
        final_scan = sharded_integrate_scan(
            model, state, n_steps=n_steps, dt=dt, config=config
        )

        np.testing.assert_allclose(
            np.asarray(final_loop.T.data),
            np.asarray(final_scan.T.data),
            atol=1e-10,
        )

    def test_scan_integration_single_step(self):
        """sharded_integrate_scan with n_steps=1 matches single step."""
        config = _single_device_config()
        model = _MockModel(increment=1.0)
        state = _make_hydrostatic_state()
        dt = 600.0

        expected = model.step(state, dt)
        result = sharded_integrate_scan(
            model, state, n_steps=1, dt=dt, config=config
        )

        np.testing.assert_allclose(
            np.asarray(result.T.data),
            np.asarray(expected.T.data),
            atol=1e-12,
        )


# ===========================================================================
# TestCheckSharding
# ===========================================================================

class TestCheckSharding:
    """Tests for check_sharding diagnostic utility."""

    def test_check_sharding_returns_dict_with_correct_keys(self):
        """check_sharding returns a dict with the expected keys."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        result = check_sharding(state, config)

        assert isinstance(result, dict)
        expected_keys = {"n_leaves", "n_sharded", "n_replicated",
                         "n_unsharded", "healthy"}
        assert set(result.keys()) == expected_keys

    def test_check_sharding_single_device_healthy(self):
        """Single-device states should always be healthy."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        result = check_sharding(state, config)

        assert result["healthy"] is True
        assert result["n_leaves"] > 0

    def test_check_sharding_counts_leaves(self):
        """n_leaves matches the number of JAX tree leaves."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        result = check_sharding(state, config)

        n_expected = len(jax.tree.leaves(state))
        assert result["n_leaves"] == n_expected

    def test_check_sharding_verbose_does_not_crash(self):
        """verbose=True should not raise."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        # Should not raise
        result = check_sharding(state, config, verbose=True)
        assert result["healthy"] is True


# ===========================================================================
# TestShardingSpec
# ===========================================================================

class TestShardingSpec:
    """Tests for _make_sharding_spec."""

    def test_face_only_sharding_spec(self):
        """Face-only config produces correct partition specs."""
        config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=1, backend="CPU", is_distributed=False,
            tiling=(1, 1), grid_type="cubed_sphere",
        )
        spec = _make_sharding_spec(config)
        assert isinstance(spec, ShardingSpec)
        assert spec.tiled_3d is None
        assert spec.tiled_2d is None

    def test_tiled_sharding_spec(self):
        """Tiled config produces non-None tiled specs."""
        config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=24, backend="GPU", is_distributed=False,
            tiling=(2, 2), grid_type="cubed_sphere",
        )
        spec = _make_sharding_spec(config)
        assert spec.tiled_3d is not None
        assert spec.tiled_2d is not None


# ===========================================================================
# TestCreateOutputShardings
# ===========================================================================

class TestCreateOutputShardings:
    """Tests for create_output_shardings."""

    def test_single_device_returns_none_leaves(self):
        """On single device, output shardings are all None."""
        config = _single_device_config()
        state = _make_hydrostatic_state()
        out = create_output_shardings(state, config)

        for leaf in jax.tree.leaves(out):
            assert leaf is None


# ===========================================================================
# TestMakeVoronoiShardedStep
# ===========================================================================

class TestMakeVoronoiShardedStep:
    """Tests for make_voronoi_sharded_step."""

    def test_single_device_returns_model_step(self):
        """On single device, returns model.step directly (not wrapped)."""
        from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
        from legoesm.parallel.mesh import DeviceConfig

        config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=1, backend="CPU", is_distributed=False,
            grid_type="voronoi", voronoi_dims=(100, 200, 50),
        )
        model = _MockModel()
        step_fn = make_voronoi_sharded_step(model, config)
        # Bound methods are not singletons, but the underlying function is.
        assert step_fn.__func__ is model.step.__func__

    def test_raises_without_voronoi_dims(self):
        """Raises ValueError when voronoi_dims is None on multi-device."""
        from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        devices = jax.devices()[:1]
        mesh = Mesh(devices, axis_names=("device",))
        config = DeviceConfig(
            mesh=mesh,
            face_sharding=NamedSharding(mesh, P("device")),
            replicated_sharding=NamedSharding(mesh, P()),
            n_devices=1, backend="CPU", is_distributed=False,
            grid_type="voronoi", voronoi_dims=None,
        )
        # n_devices=1 → returns model.step, so force multi-device config
        config = config._replace(n_devices=2)
        model = _MockModel()
        with pytest.raises(ValueError, match="voronoi_dims"):
            make_voronoi_sharded_step(model, config)


# ---------------------------------------------------------------------------
# #852: global_integral must be shard-count-invariant (conservation mass sum)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    len(jax.devices()) < 3,
    reason="needs >=3 emulated devices "
           "(XLA_FLAGS=--xla_force_host_platform_device_count=6)",
)
def test_global_integral_shard_count_invariant():
    """A global conserved integral must NOT depend on the device count (#852).

    The cube conservation reductions (``global_integral``, ``global_area_sum``,
    ``batch_global_area_sums`` — the mass/energy fixers' global sums) on a
    face-sharded field previously reduced each shard's faces locally then
    all-reduced the partials; float32 addition is non-associative, so the result
    differed between 1 / 2 / 3-faces-per-shard layouts by ~1 ulp — which the mass
    fixer amplified into a shard-count-dependent p_s correction (the symptom
    reported in #852, wrongly attributed to the ppermute halo — the halo is
    bit-exact).  The fixed per-face fixed-order reduction must be BIT-IDENTICAL
    across single-device and every whole-face sharding, even in float32, for ALL
    three entry points.
    """
    from legoesm.core.field import Field
    from legoesm.core.operators import global_integral
    from legoesm.core.conservation import global_area_sum, batch_global_area_sums
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    n_grid = 48
    grid = create_cubed_sphere(n_grid)
    base = np.arange(6 * n_grid * n_grid, dtype=np.float64).reshape(6, n_grid, n_grid)
    # p_s-like extensive field (~1e5 Pa) — the mass fixer's integrand.
    data = jnp.asarray(
        (1.0e5 + 500.0 * np.sin(0.01 * base)).astype(np.float32))
    data2 = jnp.asarray(
        (0.9e5 + 300.0 * np.cos(0.02 * base)).astype(np.float32))

    def _gi(d):
        return global_integral(
            Field(data=d, name="p_s", dims=("face", "x", "y"), units="Pa"), grid)

    def _gas(d):
        return global_area_sum(d, grid)

    def _batch(d, e):
        # The default (non-anchor) mass fixer path — fix_ps_mass.
        return jnp.stack(batch_global_area_sums([d, e], grid))

    reductions = {
        "global_integral": (_gi, (data,)),
        "global_area_sum": (_gas, (data,)),
        "batch_global_area_sums": (_batch, (data, data2)),
    }
    for name, (fn, args) in reductions.items():
        serial = np.asarray(jax.jit(fn)(*args))
        for nd in (2, 3):
            dev = create_device_mesh(n_devices=nd)
            args_sh = tuple(jax.device_put(a, dev.face_sharding) for a in args)
            sharded = np.asarray(jax.jit(fn)(*args_sh))
            # BIT-exact: a global conserved integral is decomposition-independent.
            assert np.array_equal(sharded, serial), (
                f"{name} n_devices={nd} differs from single-device: "
                f"{sharded!r} != {serial!r} "
                f"(maxΔ={float(np.max(np.abs(sharded - serial))):.3e})")
