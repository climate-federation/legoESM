"""Regression tests for precision dtype contracts.

Verifies that state constructors, forcing builders, physics helpers,
and checkpoint round-trips produce arrays in the dtype specified by the
active PrecisionPolicy — under fp32, fp64, mixed, and mixed_fp64_storage.

These tests catch the class of bug where bare ``jnp.zeros()`` (no dtype)
silently inherits the JAX default instead of honoring the precision policy.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import (
    PrecisionPolicy, set_policy, get_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_policy():
    """Save and restore the global precision policy around each test."""
    saved = get_policy()
    yield
    set_policy(saved)


MODES = [
    ("fp32", PrecisionPolicy.fp32()),
    ("fp64", PrecisionPolicy.fp64()),
    ("mixed", PrecisionPolicy.mixed()),
    ("mixed_fp64_storage", PrecisionPolicy.mixed_fp64_storage()),
]


def _enable_x64_if_needed(policy):
    """Enable JAX x64 if the policy requires it."""
    needs_x64 = jnp.float64 in (
        policy.storage, policy.compute, policy.accumulate, policy.control,
    )
    if needs_x64 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# PrecisionPolicy modes
# ---------------------------------------------------------------------------

class TestPolicyModes:
    def test_mixed_fp64_storage_roles(self):
        p = PrecisionPolicy.mixed_fp64_storage()
        assert p.storage == jnp.float64
        assert p.compute == jnp.float32
        assert p.accumulate == jnp.float64
        assert p.control == jnp.float64


class TestRuntimeResolve:
    def test_resolve_mixed_fp64_storage(self):
        from legoesm.runtime.precision import resolve_precision
        p = resolve_precision("mixed_fp64_storage")
        assert p.storage == jnp.float64
        assert p.compute == jnp.float32


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

class TestFieldHelpers:
    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_zeros_field_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)
        from legoesm.core.field import zeros_field
        f = zeros_field((4, 8), name="test", dims=("y", "x"), units="m")
        assert f.data.dtype == policy.storage, (
            f"zeros_field under {mode_name}: expected {policy.storage}, got {f.data.dtype}"
        )

    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_ones_field_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)
        from legoesm.core.field import ones_field
        f = ones_field((4, 8), name="test", dims=("y", "x"), units="m")
        assert f.data.dtype == policy.storage, (
            f"ones_field under {mode_name}: expected {policy.storage}, got {f.data.dtype}"
        )


# ---------------------------------------------------------------------------
# Held-Suarez init
# ---------------------------------------------------------------------------

class TestHeldSuarezInit:
    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_held_suarez_init_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(5)
        state = held_suarez_init(grid, sigma)

        expected = policy.storage
        assert state.T.data.dtype == expected, (
            f"held_suarez_init T under {mode_name}: {state.T.data.dtype} != {expected}"
        )
        assert state.u.data.dtype == expected
        assert state.v.data.dtype == expected
        assert state.p_s.data.dtype == expected

    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_held_suarez_latlon_init_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_init_latlon,
        )

        grid = create_latlon_grid(8)
        sigma = create_sigma_coordinate(5)
        state = held_suarez_init_latlon(grid, sigma)

        expected = policy.storage
        assert state.T.data.dtype == expected
        assert state.u.data.dtype == expected
        assert state.p_s.data.dtype == expected


# ---------------------------------------------------------------------------
# Forcing builders
# ---------------------------------------------------------------------------

class TestForcing:
    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_analytical_sst_sic_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.forcing.analytical import analytical_sst_sic

        lat_deg = np.linspace(-90, 90, 32)
        sst, sic = analytical_sst_sic(lat_deg, day=0.0)

        expected = policy.storage
        assert sst.dtype == expected, (
            f"analytical sst under {mode_name}: {sst.dtype} != {expected}"
        )
        assert sic.dtype == expected


# ---------------------------------------------------------------------------
# Physics pipeline zeros
# ---------------------------------------------------------------------------

class TestPhysicsZeros:
    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_noop_convection_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.driver.physics_pipeline import _noop_convection

        T = jnp.ones((10, 5), dtype=policy.storage)
        q_v = jnp.ones_like(T)
        p_full = jnp.ones_like(T) * 5e4
        p_half = jnp.ones((10, 6), dtype=policy.storage) * 5e4

        out = _noop_convection(T, q_v, p_full, p_half, dt=600.0, config=None)
        assert out.dT_dt.dtype == policy.storage
        # Post-Option-C: ``ConvectionOutput.precipitation`` was replaced
        # by the 3D ``dq_c_conv_dt`` field (cloud-water source rate);
        # microphysics owns the surface precipitation diagnostic. The
        # dtype contract still applies — the no-op stub must produce
        # the policy storage dtype.
        assert out.dq_c_conv_dt.dtype == policy.storage

    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_gwd_zero_output_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.atmosphere.physics.gravity_wave_drag.output import (
            make_zero_output,
        )

        out = make_zero_output(10, 5)
        assert out.du_dt.dtype == policy.storage
        assert out.eps_gwd.dtype == policy.storage


# ---------------------------------------------------------------------------
# Tracer init
# ---------------------------------------------------------------------------

class TestTracerInit:
    @pytest.mark.parametrize("mode_name,policy", MODES)
    def test_tracer_zeros_dtype(self, mode_name, policy):
        _enable_x64_if_needed(policy)
        set_policy(policy)

        from legoesm.core.tracers import TracerRegistry, TracerInfo, init_tracers

        reg = TracerRegistry(
            tracers=(TracerInfo(name="q_v", long_name="water vapor", units="kg/kg"),),
        )
        fields = init_tracers(reg, (6, 4, 4, 5))
        assert fields["q_v"].dtype == policy.storage
