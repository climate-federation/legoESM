"""Tests for control vector transforms."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, HydrostaticState
from legoesm.ocean.state import OceanState
from legoesm.da.control_vector import (
    build_control_spec,
    state_to_control,
    control_to_state,
    control_to_increment,
)


def _make_sw_state(shape_2d):
    """Create a ShallowWaterState with given 2D shape."""
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape_2d) * 10000.0, name="h", dims=(), units="m"),
        u=Field(data=jnp.ones(shape_2d) * 5.0, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape_2d) * -3.0, name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape_2d), name="h_s", dims=(), units="m"),
    )


def _make_hydrostatic_state(shape_2d, nlev=5):
    """Create a HydrostaticState."""
    shape_3d = shape_2d + (nlev,)
    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * 10.0, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * -5.0, name="v", dims=(), units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * 280.0, name="T", dims=(), units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=(), units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=(), units="m2/s2"),
        tracers=None,
    )


def _make_ocean_state(shape_2d, nlev=3):
    """Create an OceanState."""
    shape_3d = shape_2d + (nlev,)
    return OceanState(
        u=Field(data=jnp.ones(shape_3d) * 0.1, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * -0.05, name="v", dims=(), units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * 15.0, name="T", dims=(), units="degC"),
        S=Field(data=jnp.ones(shape_3d) * 35.0, name="S", dims=(), units="PSU"),
        eta=Field(data=jnp.zeros(shape_2d), name="eta", dims=(), units="m"),
        H_bathy=Field(data=jnp.ones(shape_2d) * 4000.0, name="H_bathy", dims=(), units="m"),
        land_mask=Field(data=jnp.ones(shape_2d), name="land_mask", dims=(), units=""),
    )


# ---- Round-trip tests ----

class TestRoundTrip:
    """state -> control -> state round-trip recovers original."""

    def test_shallow_water_cubed_sphere(self):
        state = _make_sw_state((6, 4, 4))
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        for name in ("h", "u", "v"):
            orig = getattr(state, name).data
            rec = getattr(recovered, name).data
            assert jnp.allclose(orig, rec, atol=1e-6), f"Round-trip failed for {name}"

    def test_shallow_water_latlon(self):
        state = _make_sw_state((16, 32))
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        assert jnp.allclose(state.h.data, recovered.h.data, atol=1e-6)

    def test_shallow_water_voronoi(self):
        state = _make_sw_state((642,))
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        assert jnp.allclose(state.u.data, recovered.u.data, atol=1e-6)

    def test_hydrostatic(self):
        state = _make_hydrostatic_state((6, 4, 4), nlev=5)
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        for name in ("u", "v", "T", "p_s"):
            orig = getattr(state, name).data
            rec = getattr(recovered, name).data
            assert jnp.allclose(orig, rec, atol=1e-6), f"Round-trip failed for {name}"

    def test_ocean(self):
        state = _make_ocean_state((6, 4, 4), nlev=3)
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        for name in ("u", "v", "T", "S", "eta"):
            orig = getattr(state, name).data
            rec = getattr(recovered, name).data
            assert jnp.allclose(orig, rec, atol=1e-6), f"Round-trip failed for {name}"

    def test_static_fields_preserved(self):
        """Static fields (phis, h_s, H_bathy, land_mask) must be preserved."""
        state = _make_hydrostatic_state((6, 4, 4))
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        # Modify control vector
        x_mod = x + 1.0
        recovered = control_to_state(x_mod, spec, state)
        # phis should be unchanged
        assert jnp.allclose(state.phis.data, recovered.phis.data)


# ---- Spec correctness ----

class TestSpec:
    def test_total_size(self):
        state = _make_sw_state((6, 4, 4))
        spec = build_control_spec(state)
        expected = 3 * 6 * 4 * 4  # h, u, v (not h_s)
        assert spec.total_size == expected

    def test_excludes_static(self):
        state = _make_hydrostatic_state((6, 4, 4), nlev=5)
        spec = build_control_spec(state)
        field_names = [e.field_name for e in spec.entries]
        assert "phis" not in field_names

    def test_custom_fields(self):
        state = _make_hydrostatic_state((6, 4, 4), nlev=5)
        spec = build_control_spec(state, fields=("T", "p_s"))
        field_names = [e.field_name for e in spec.entries]
        assert field_names == ["T", "p_s"]

    def test_ocean_excludes_bathy_mask(self):
        state = _make_ocean_state((6, 4, 4))
        spec = build_control_spec(state)
        field_names = [e.field_name for e in spec.entries]
        assert "H_bathy" not in field_names
        assert "land_mask" not in field_names


# ---- Transforms ----

class TestTransforms:
    def test_log_transform_round_trip(self):
        state = _make_hydrostatic_state((6, 4, 4))
        spec = build_control_spec(state, fields=("p_s",), transforms={"p_s": "log"})
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        assert jnp.allclose(state.p_s.data, recovered.p_s.data, rtol=1e-5)

    def test_softplus_transform_round_trip(self):
        state = _make_hydrostatic_state((6, 4, 4))
        # Use T as a field that should be positive
        spec = build_control_spec(state, fields=("T",), transforms={"T": "softplus"})
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        assert jnp.allclose(state.T.data, recovered.T.data, rtol=1e-4)


# ---- Differentiability ----

class TestDifferentiability:
    def test_grad_through_round_trip(self):
        """jax.grad through state_to_control(control_to_state(x)) should work."""
        state = _make_sw_state((4, 4))
        spec = build_control_spec(state)
        x0 = state_to_control(state, spec)

        def f(x):
            s = control_to_state(x, spec, state)
            return jnp.sum(s.h.data ** 2) + jnp.sum(s.u.data)

        grad = jax.grad(f)(x0)
        assert grad.shape == x0.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_identity_jacobian(self):
        """Round-trip Jacobian should be identity."""
        state = _make_sw_state((2, 2))
        spec = build_control_spec(state)
        x0 = state_to_control(state, spec)

        def round_trip(x):
            return state_to_control(control_to_state(x, spec, state), spec)

        J = jax.jacobian(round_trip)(x0)
        assert jnp.allclose(J, jnp.eye(x0.shape[0]), atol=1e-5)


# ---- Control to increment ----

class TestControlToIncrement:
    def test_zero_increment(self):
        state = _make_sw_state((4, 4))
        spec = build_control_spec(state)
        dx = jnp.zeros(spec.total_size)
        inc = control_to_increment(dx, spec, state)
        assert jnp.allclose(inc.h.data, 0.0)
        assert jnp.allclose(inc.u.data, 0.0)

    def test_nonzero_increment(self):
        state = _make_sw_state((4, 4))
        spec = build_control_spec(state)
        dx = jnp.ones(spec.total_size) * 2.0
        inc = control_to_increment(dx, spec, state)
        assert jnp.allclose(inc.h.data, 2.0)


# ---- Tracers ----

class TestTracers:
    def test_hydrostatic_with_tracers(self):
        shape_2d = (6, 4, 4)
        nlev = 5
        shape_3d = shape_2d + (nlev,)
        state = HydrostaticState(
            u=Field(data=jnp.ones(shape_3d), name="u", dims=(), units="m/s"),
            v=Field(data=jnp.ones(shape_3d), name="v", dims=(), units="m/s"),
            T=Field(data=jnp.ones(shape_3d) * 280.0, name="T", dims=(), units="K"),
            p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=(), units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=(), units="m2/s2"),
            tracers={
                "q_v": Field(data=jnp.ones(shape_3d) * 0.01, name="q_v", dims=(), units="kg/kg"),
                "q_c": Field(data=jnp.ones(shape_3d) * 0.001, name="q_c", dims=(), units="kg/kg"),
            },
        )
        spec = build_control_spec(state)
        x = state_to_control(state, spec)
        recovered = control_to_state(x, spec, state)
        assert jnp.allclose(recovered.tracers["q_v"].data, 0.01, atol=1e-6)
        assert jnp.allclose(recovered.tracers["q_c"].data, 0.001, atol=1e-6)
