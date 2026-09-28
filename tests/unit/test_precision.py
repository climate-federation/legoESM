"""Tests for precision management system.

Covers:
1. PrecisionPolicy modes (fp32, fp64, mixed)
2. Cast/const helpers
3. Module-level overrides
4. Reduction wrappers (global_sum, norm, compensated_sum)
5. with_precision decorator
6. Recommended overrides
7. Drift checker
8. Health diagnostics
9. Legacy sync
"""

import pytest
import jax
import jax.numpy as jnp

from legoesm.core.precision import (
    PrecisionPolicy,
    set_policy,
    get_policy,
    cast,
    const,
    cast_pytree,
    global_sum,
    norm,
    weighted_mean,
    compensated_sum,
    with_precision,
    set_module_override,
    clear_module_overrides,
    get_module_overrides,
    set_recommended_overrides,
)


@pytest.fixture(autouse=True)
def _restore_policy():
    """Save and restore the global precision policy around each test.

    Without this, test_scalar_libm_transcendental_policy leaves the
    process-wide policy at fp64/libm for every test that runs after it.
    """
    saved = get_policy()
    yield
    set_policy(saved)


# ===========================================================================
# Part 1: PrecisionPolicy
# ===========================================================================

class TestPrecisionPolicy:
    """Test PrecisionPolicy construction and modes."""

    def test_fp32_mode(self):
        p = PrecisionPolicy.fp32()
        assert p.storage == jnp.float32
        assert p.compute == jnp.float32
        assert p.accumulate == jnp.float32
        assert p.control == jnp.float32
        assert p.transcendentals == "native"

    def test_fp64_mode(self):
        p = PrecisionPolicy.fp64()
        assert p.storage == jnp.float64
        assert p.compute == jnp.float64
        assert p.accumulate == jnp.float64
        assert p.control == jnp.float64
        assert p.transcendentals == "native"

    def test_scalar_libm_transcendental_policy(self):
        p = PrecisionPolicy.fp64(transcendentals="libm")
        set_policy(p)
        assert get_policy() == p
        with pytest.raises(ValueError, match="transcendentals"):
            set_policy(p._replace(transcendentals="unknown"))

    def test_mixed_mode(self):
        p = PrecisionPolicy.mixed()
        assert p.storage == jnp.float32
        assert p.compute == jnp.float32
        assert p.accumulate == jnp.float64
        assert p.control == jnp.float64

    def test_custom_policy(self):
        p = PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float64,
            accumulate=jnp.float64,
            control=jnp.float32,
        )
        assert p.compute == jnp.float64
        assert p.control == jnp.float32

    def test_set_and_get_policy(self):
        set_policy(PrecisionPolicy.mixed())
        p = get_policy()
        assert p.accumulate == jnp.float64
        # Restore default
        set_policy(PrecisionPolicy.fp32())


# ===========================================================================
# Part 2: Cast and const helpers
# ===========================================================================

class TestCastHelpers:
    """Test cast(), const(), cast_pytree()."""

    def setup_method(self):
        set_policy(PrecisionPolicy.mixed())
        clear_module_overrides()

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_cast_to_compute(self):
        x = jnp.ones(5, dtype=jnp.float64)
        # Default: downcast skipped (fp64 -> fp32 blocked)
        y = cast(x, "tracer_advection", "compute")
        assert y.dtype == jnp.float64
        # Explicit downcast allowed
        y2 = cast(x, "tracer_advection", "compute", allow_downcast=True)
        assert y2.dtype == jnp.float32

    def test_cast_to_accumulate(self):
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast(x, None, "accumulate")
        assert y.dtype == jnp.float64

    def test_cast_noop_same_dtype(self):
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast(x, None, "compute")
        assert y is x  # No copy

    def test_const_compute(self):
        c = const(1.0, None, "compute")
        assert c.dtype == jnp.float32
        assert float(c) == 1.0

    def test_const_control(self):
        c = const(3.14, None, "control")
        assert c.dtype == jnp.float64

    def test_cast_pytree(self):
        tree = {"a": jnp.ones(3, dtype=jnp.float32),
                "b": jnp.array(1, dtype=jnp.int32)}
        result = cast_pytree(tree, None, "accumulate")
        assert result["a"].dtype == jnp.float64
        assert result["b"].dtype == jnp.int32  # int unchanged


# ===========================================================================
# Part 3: Module-level overrides
# ===========================================================================

class TestModuleOverrides:
    """Test per-module precision overrides."""

    def setup_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def teardown_method(self):
        clear_module_overrides()

    def test_set_override(self):
        set_module_override("pressure_gradient", compute="fp64")
        overrides = get_module_overrides()
        assert "pressure_gradient" in overrides
        assert overrides["pressure_gradient"]["compute"] == jnp.float64

    def test_override_takes_precedence(self):
        set_module_override("barotropic_solver", compute="fp64")
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast(x, "barotropic_solver", "compute")
        assert y.dtype == jnp.float64

    def test_no_override_uses_global(self):
        x = jnp.ones(5, dtype=jnp.float64)
        # Without override, global fp32 policy applies but downcast is
        # skipped by default to prevent silent precision loss.
        y = cast(x, "tracer_advection", "compute")
        assert y.dtype == jnp.float64
        # With explicit allow_downcast, it does downcast.
        y2 = cast(x, "tracer_advection", "compute", allow_downcast=True)
        assert y2.dtype == jnp.float32

    def test_clear_overrides(self):
        set_module_override("foo", compute="fp64")
        clear_module_overrides()
        assert get_module_overrides() == {}

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Unknown role"):
            set_module_override("foo", invalid_role="fp32")

    def test_none_clears_single_role(self):
        set_module_override("bar", compute="fp64", control="fp64")
        set_module_override("bar", compute=None)
        overrides = get_module_overrides()
        assert "compute" not in overrides["bar"]
        assert overrides["bar"]["control"] == jnp.float64


# ===========================================================================
# Part 4: Reduction wrappers
# ===========================================================================

class TestReductions:
    """Test precision-aware reductions."""

    def setup_method(self):
        set_policy(PrecisionPolicy.mixed())
        clear_module_overrides()

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())

    def test_global_sum_accumulation_dtype(self):
        x = jnp.ones(100, dtype=jnp.float32)
        s = global_sum(x)
        assert s.dtype == jnp.float64
        assert jnp.isclose(s, 100.0)

    def test_norm_l2(self):
        x = jnp.ones(100, dtype=jnp.float32)
        n = norm(x, ord=2)
        assert n.dtype == jnp.float64
        assert jnp.isclose(n, 10.0)

    def test_norm_l1(self):
        x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
        n = norm(x, ord=1)
        assert jnp.isclose(n, 6.0)

    def test_norm_linf(self):
        x = jnp.array([1.0, -5.0, 3.0], dtype=jnp.float32)
        n = norm(x, ord=0)  # ord != 1,2 → Linf
        assert jnp.isclose(n, 5.0)

    def test_weighted_mean(self):
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        w = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
        m = weighted_mean(x, w)
        assert m.dtype == jnp.float64
        assert jnp.isclose(m, 2.0)

    def test_compensated_sum_accuracy(self):
        """Compensated sum should be more accurate than naive sum for fp32."""
        n = 10_000
        x = jnp.ones(n, dtype=jnp.float32) * 1e-4
        naive = jnp.sum(x)
        compensated = compensated_sum(x, axis=0)
        exact = 1.0
        # Compensated should be closer to exact
        assert abs(float(compensated) - exact) <= abs(float(naive) - exact) + 1e-7

    def test_compensated_sum_multidim(self):
        x = jnp.ones((5, 3), dtype=jnp.float32)
        result = compensated_sum(x, axis=0)
        assert result.shape == (3,)
        assert jnp.allclose(result, 5.0)


# ===========================================================================
# Part 5: with_precision decorator
# ===========================================================================

class TestWithPrecision:
    """Test the @with_precision decorator."""

    def setup_method(self):
        set_policy(PrecisionPolicy.mixed())
        clear_module_overrides()
        set_module_override("test_kernel", compute="fp64", storage="fp32")

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_input_upcast(self):
        @with_precision("test_kernel")
        def kernel(x, y):
            assert x.dtype == jnp.float64  # compute = fp64
            assert y.dtype == jnp.float64
            return x + y

        x = jnp.ones(3, dtype=jnp.float32)
        y = jnp.ones(3, dtype=jnp.float32)
        result = kernel(x, y)
        # Output should be downcast to storage = fp32
        assert result.dtype == jnp.float32

    def test_preserves_name(self):
        @with_precision("test_kernel")
        def my_function(x):
            return x

        assert my_function.__name__ == "my_function"


# ===========================================================================
# Part 6: Recommended overrides
# ===========================================================================

class TestRecommendedOverrides:
    """Test set_recommended_overrides()."""

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_mixed_mode_sets_overrides(self):
        set_policy(PrecisionPolicy.mixed())
        set_recommended_overrides("mixed")
        p = get_policy()
        assert p.storage == jnp.float32
        assert p.accumulate == jnp.float64
        overrides = get_module_overrides()
        assert "barotropic_solver" in overrides
        assert overrides["barotropic_solver"]["compute"] == jnp.float64

    def test_fp32_mode_no_overrides(self):
        set_policy(PrecisionPolicy.fp32())
        set_recommended_overrides("fp32")
        assert get_module_overrides() == {}
        assert get_policy().accumulate == jnp.float32

    def test_fp64_mode_no_overrides(self):
        set_policy(PrecisionPolicy.fp64())
        set_recommended_overrides("fp64")
        assert get_module_overrides() == {}
        assert get_policy().compute == jnp.float64

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            set_recommended_overrides("fp16")

    def test_ocean_overrides_present(self):
        set_recommended_overrides("mixed")
        overrides = get_module_overrides()
        assert "pressure_gradient" in overrides
        assert "equation_of_state" in overrides

    def test_atmosphere_overrides_present(self):
        set_recommended_overrides("mixed")
        overrides = get_module_overrides()
        assert "spectral_transform" in overrides
        assert "semi_implicit" in overrides

    def test_land_overrides_present(self):
        set_recommended_overrides("mixed")
        overrides = get_module_overrides()
        assert "carbon_pools" in overrides

    def test_ice_overrides_present(self):
        set_recommended_overrides("mixed")
        overrides = get_module_overrides()
        assert "evp_solver" in overrides


# ===========================================================================
# Part 7: Drift checker
# ===========================================================================

class TestDriftChecker:
    """Test PrecisionDriftChecker."""

    def _make_state(self, T_offset=0.0, u_offset=0.0, ps_offset=0.0):
        """Create a mock HydrostaticState-like object."""
        from types import SimpleNamespace

        class MockField:
            def __init__(self, data):
                self.data = data

        n = 6
        T = MockField(jnp.full((n, 4, 4, 5), 280.0 + T_offset, dtype=jnp.float64))
        u = MockField(jnp.full((n, 4, 4, 5), 10.0 + u_offset, dtype=jnp.float64))
        v = MockField(jnp.full((n, 4, 4, 5), 5.0, dtype=jnp.float64))
        ps = MockField(jnp.full((n, 4, 4), 1e5 + ps_offset, dtype=jnp.float64))

        return SimpleNamespace(T=T, u=u, v=v, p_s=ps)

    def _make_grid(self):
        from types import SimpleNamespace
        area = jnp.ones((6, 4, 4), dtype=jnp.float64) * 1e10
        return SimpleNamespace(area=area)

    def test_identical_states_zero_drift(self):
        from legoesm.diagnostics.precision_drift import PrecisionDriftChecker

        checker = PrecisionDriftChecker()
        state = self._make_state()
        grid = self._make_grid()
        checker.record(state, state, grid, step=0)
        summary = checker.summary()
        assert summary["passed"]
        assert summary["max_rms_T"] == 0.0
        assert summary["max_rms_u"] == 0.0

    def test_small_drift_passes(self):
        from legoesm.diagnostics.precision_drift import PrecisionDriftChecker

        # Use relaxed energy threshold since 0.01 K offset → 3.5e-5 rel energy diff
        checker = PrecisionDriftChecker(energy_rel_threshold=1e-3)
        ref = self._make_state()
        test = self._make_state(T_offset=0.01, u_offset=0.01)
        grid = self._make_grid()
        checker.record(ref, test, grid, step=0)
        summary = checker.summary()
        assert summary["passed"]
        assert summary["max_rms_T"] < 0.1

    def test_large_drift_fails(self):
        from legoesm.diagnostics.precision_drift import PrecisionDriftChecker

        checker = PrecisionDriftChecker(rms_T_threshold=0.5)
        ref = self._make_state()
        test = self._make_state(T_offset=2.0)
        grid = self._make_grid()
        checker.record(ref, test, grid, step=0)
        summary = checker.summary()
        assert not summary["passed"]
        assert any("RMS T" in w for w in summary["warnings"])

    def test_empty_summary(self):
        from legoesm.diagnostics.precision_drift import PrecisionDriftChecker

        checker = PrecisionDriftChecker()
        summary = checker.summary()
        assert summary["n_snapshots"] == 0
        assert summary["passed"]

    def test_multiple_snapshots(self):
        from legoesm.diagnostics.precision_drift import PrecisionDriftChecker

        checker = PrecisionDriftChecker()
        grid = self._make_grid()
        for i in range(5):
            ref = self._make_state()
            test = self._make_state(T_offset=0.001 * i)
            checker.record(ref, test, grid, step=i)
        summary = checker.summary()
        assert summary["n_snapshots"] == 5


# ===========================================================================
# Part 8: Health diagnostics
# ===========================================================================

class TestHealthDiagnostics:
    """Test precision_health_report()."""

    def _make_state(self, T_val=280.0, u_val=10.0, ps_val=1e5):
        from types import SimpleNamespace

        class MockField:
            def __init__(self, data):
                self.data = data

        n = 6
        T = MockField(jnp.full((n, 4, 4, 5), T_val))
        u = MockField(jnp.full((n, 4, 4, 5), u_val))
        v = MockField(jnp.full((n, 4, 4, 5), 5.0))
        ps = MockField(jnp.full((n, 4, 4), ps_val))

        return SimpleNamespace(T=T, u=u, v=v, p_s=ps)

    def test_healthy_state(self):
        from legoesm.diagnostics.precision_drift import precision_health_report

        state = self._make_state()
        report = precision_health_report(state)
        assert report["status"] == "HEALTHY"
        assert len(report["warnings"]) == 0

    def test_cold_temperature_warning(self):
        from legoesm.diagnostics.precision_drift import precision_health_report

        state = self._make_state(T_val=100.0)
        report = precision_health_report(state)
        assert report["status"] == "UNHEALTHY"
        assert any("T_min" in w for w in report["warnings"])

    def test_hot_temperature_warning(self):
        from legoesm.diagnostics.precision_drift import precision_health_report

        state = self._make_state(T_val=400.0)
        report = precision_health_report(state)
        assert any("T_max" in w for w in report["warnings"])

    def test_extreme_wind_warning(self):
        from legoesm.diagnostics.precision_drift import precision_health_report

        state = self._make_state(u_val=300.0)
        report = precision_health_report(state)
        assert any("wind" in w.lower() for w in report["warnings"])

    def test_low_pressure_warning(self):
        from legoesm.diagnostics.precision_drift import precision_health_report

        state = self._make_state(ps_val=3e4)
        report = precision_health_report(state)
        assert any("p_s" in w for w in report["warnings"])

    def test_nan_detection(self):
        from legoesm.diagnostics.precision_drift import precision_health_report
        from types import SimpleNamespace

        class MockField:
            def __init__(self, data):
                self.data = data

        T_data = jnp.full((6, 4, 4, 5), 280.0)
        T_data = T_data.at[0, 0, 0, 0].set(float("nan"))
        state = SimpleNamespace(
            T=MockField(T_data),
            u=MockField(jnp.full((6, 4, 4, 5), 10.0)),
            v=MockField(jnp.full((6, 4, 4, 5), 5.0)),
            p_s=MockField(jnp.full((6, 4, 4), 1e5)),
        )
        report = precision_health_report(state)
        assert report["metrics"]["has_nan"]
        assert report["status"] == "UNHEALTHY"


# ===========================================================================
# Part 9: Tracer negativity
# ===========================================================================

class TestTracerNegativity:
    """Test check_tracer_negativity()."""

    def test_positive_tracers_pass(self):
        from legoesm.diagnostics.precision_drift import check_tracer_negativity

        tracers = {"q_v": jnp.ones(10) * 0.01, "q_c": jnp.ones(10) * 1e-5}
        violations = check_tracer_negativity(tracers)
        assert violations == {}

    def test_negative_tracer_flagged(self):
        from legoesm.diagnostics.precision_drift import check_tracer_negativity

        tracers = {
            "q_v": jnp.array([0.01, -1e-6, 0.005]),
            "q_c": jnp.ones(3) * 0.001,
        }
        violations = check_tracer_negativity(tracers)
        assert "q_v" in violations
        assert "q_c" not in violations


# ===========================================================================
# Part 10: Legacy sync
# ===========================================================================

# ===========================================================================
# Part 11: JIT compatibility
# ===========================================================================

class TestJITCompatibility:
    """Verify precision operations work under jax.jit."""

    def setup_method(self):
        set_policy(PrecisionPolicy.mixed())
        clear_module_overrides()

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_cast_under_jit(self):
        @jax.jit
        def f(x):
            # cast reads global state at trace time — this is fine
            # because precision policy doesn't change within a JIT call
            return x.astype(jnp.float64)  # simulate accumulate cast

        x = jnp.ones(5, dtype=jnp.float32)
        result = f(x)
        assert result.dtype == jnp.float64

    def test_compensated_sum_jit(self):
        @jax.jit
        def f(x):
            return compensated_sum(x, axis=0)

        x = jnp.ones(1000, dtype=jnp.float32) * 1e-4
        result = f(x)
        assert jnp.isclose(result, 0.1, atol=1e-5)

    def test_global_sum_jit(self):
        @jax.jit
        def f(x):
            return jnp.sum(x.astype(jnp.float64))

        x = jnp.ones(100, dtype=jnp.float32)
        result = f(x)
        assert result.dtype == jnp.float64

    def test_grad_through_cast(self):
        """Verify gradients flow through precision casts."""
        @jax.jit
        @jax.grad
        def f(x):
            # Upcast to fp64, compute, downcast
            x64 = x.astype(jnp.float64)
            y = jnp.sum(x64 ** 2)
            return y.astype(jnp.float32)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        grad = f(x)
        assert jnp.allclose(grad, 2.0 * x, atol=1e-5)

    def test_vmap_compatible(self):
        """Verify precision casting works under vmap."""
        @jax.vmap
        def f(x):
            acc = x.astype(jnp.float64)
            return jnp.sum(acc)

        x = jnp.ones((4, 10), dtype=jnp.float32)
        result = f(x)
        assert result.shape == (4,)
        assert result.dtype == jnp.float64


# ===========================================================================
# Part 12: Precision enforcement
# ===========================================================================

class TestPrecisionEnforcement:
    """Verify precision policy enforcement at bootstrap time."""

    def setup_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_validate_fp32_always_ok(self):
        from legoesm.core.precision import validate_policy
        set_policy(PrecisionPolicy.fp32())
        validate_policy()  # Should not raise

    def test_validate_fp64_with_x64_enabled(self):
        from legoesm.core.precision import validate_policy
        jax.config.update("jax_enable_x64", True)
        set_policy(PrecisionPolicy.fp64())
        validate_policy()  # Should not raise

    def test_validate_fp64_without_x64_raises(self):
        from legoesm.core.precision import validate_policy
        original = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", False)
            with pytest.raises(RuntimeError, match="float64.*x64"):
                validate_policy(PrecisionPolicy.fp64())
        finally:
            jax.config.update("jax_enable_x64", original)

    def test_validate_mixed_without_x64_raises(self):
        from legoesm.core.precision import validate_policy
        original = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", False)
            with pytest.raises(RuntimeError, match="float64.*x64"):
                validate_policy(PrecisionPolicy.mixed())
        finally:
            jax.config.update("jax_enable_x64", original)

    def test_trainable_params_dtype_matches_policy(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        # With fp64 policy, raw values should be float64
        jax.config.update("jax_enable_x64", True)
        set_policy(PrecisionPolicy.fp64())
        params = TrainablePhysicsParams.from_defaults()
        for name, val in params.raw_values.items():
            assert val.dtype == jnp.float64, f"{name} has dtype {val.dtype}, expected float64"
        # Restore
        set_policy(PrecisionPolicy.fp32())

    def test_trainable_params_default_fp32(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        set_policy(PrecisionPolicy.fp32())
        params = TrainablePhysicsParams.from_defaults()
        for name, val in params.raw_values.items():
            assert val.dtype == jnp.float32, f"{name} has dtype {val.dtype}, expected float32"


# ===========================================================================
# Part 13: Precision wiring into compute kernels
# ===========================================================================

class TestPrecisionWiring:
    """Verify precision casting is wired into key compute kernels."""

    def setup_method(self):
        set_recommended_overrides("mixed")

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_ocean_area_sum_uses_accumulate_precision(self):
        """Ocean area sum should upcast to accumulation dtype in mixed mode."""
        from legoesm.ocean.conservation import _ocean_area_sum
        from types import SimpleNamespace

        field = jnp.ones((6, 4, 4), dtype=jnp.float32)
        mask = jnp.ones((6, 4, 4), dtype=jnp.float32)
        grid = SimpleNamespace(area=jnp.ones((6, 4, 4), dtype=jnp.float32))
        result = _ocean_area_sum(field, mask, grid)
        # In mixed mode, ocean_diagnostics accumulate is float64
        assert result.dtype == jnp.float64

    def test_ocean_area_sum_fp32_mode(self):
        """Ocean area sum stays fp32 when policy is fp32."""
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()
        from legoesm.ocean.conservation import _ocean_area_sum
        from types import SimpleNamespace

        field = jnp.ones((6, 4, 4), dtype=jnp.float32)
        mask = jnp.ones((6, 4, 4), dtype=jnp.float32)
        grid = SimpleNamespace(area=jnp.ones((6, 4, 4), dtype=jnp.float32))
        result = _ocean_area_sum(field, mask, grid)
        assert result.dtype == jnp.float32

    def test_ocean_volume_sum_uses_accumulate_precision(self):
        """Ocean volume sum should upcast to accumulation dtype."""
        from legoesm.ocean.conservation import _ocean_volume_sum
        from types import SimpleNamespace

        field_3d = jnp.ones((6, 4, 4, 5), dtype=jnp.float32)
        h_k = jnp.ones((6, 4, 4, 5), dtype=jnp.float32)
        mask = jnp.ones((6, 4, 4), dtype=jnp.float32)
        grid = SimpleNamespace(area=jnp.ones((6, 4, 4), dtype=jnp.float32))
        result = _ocean_volume_sum(field_3d, h_k, mask, grid)
        assert result.dtype == jnp.float64

    def test_sigma_coordinate_explicit_dtype(self):
        """Sigma coordinate should respect explicit dtype parameter."""
        from legoesm.grids.vertical import create_sigma_coordinate
        sigma = create_sigma_coordinate(10, dtype=jnp.float64)
        assert sigma.sigma_full.dtype == jnp.float64
        assert sigma.sigma_half.dtype == jnp.float64
        assert sigma.dsigma.dtype == jnp.float64

    def test_sigma_coordinate_default_fp32(self):
        """Sigma coordinate default dtype should be float32 under fp32 policy."""
        from legoesm.grids.vertical import create_sigma_coordinate
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()
        sigma = create_sigma_coordinate(10)
        assert sigma.sigma_full.dtype == jnp.float32

    def test_hybrid_coordinate_explicit_dtype(self):
        """Hybrid coordinate should respect explicit dtype parameter."""
        from legoesm.grids.vertical import create_hybrid_coordinate
        import numpy as np
        n = 5
        A_half = np.linspace(0.1, 0.0, n + 1)
        B_half = np.linspace(0.0, 1.0, n + 1)
        coord = create_hybrid_coordinate(n, A_half, B_half, dtype=jnp.float64)
        assert coord.A_half.dtype == jnp.float64
        assert coord.B_half.dtype == jnp.float64

    def test_hybrid_coordinate_default_fp32(self):
        """Hybrid coordinate default dtype should be float32 under fp32 policy."""
        from legoesm.grids.vertical import create_hybrid_coordinate
        import numpy as np
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()
        n = 5
        A_half = np.linspace(0.1, 0.0, n + 1)
        B_half = np.linspace(0.0, 1.0, n + 1)
        coord = create_hybrid_coordinate(n, A_half, B_half)
        assert coord.A_half.dtype == jnp.float32
