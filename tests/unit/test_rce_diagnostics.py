"""RCE diagnostics tests (plane NH CRM)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.rce_diagnostics import (
    cloud_fraction_profile_plane,
    column_moist_static_energy_plane, column_total_water_plane,
    column_water_vapor_plane, domain_mean_profiles_plane,
    precipitation_rate_proxy_plane,
    pseudo_equivalent_potential_temperature,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=4, ny=4, nlev=8, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(8, H=10_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((4, 4, 8, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(0.001)
    tracers = tracers.at[..., 2].set(0.0005)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def test_column_water_vapor_positive_and_finite():
    state, _, hc, _ = _setup()
    cwv = column_water_vapor_plane(state, hc)
    assert cwv.shape == (4, 4)
    assert bool(jnp.all(cwv > 0.0))


def test_column_total_water_includes_all_slots():
    state, _, hc, _ = _setup()
    cwv_only = column_water_vapor_plane(state, hc)
    ctw = column_total_water_plane(state, hc, water_slot_indices=(0, 1, 2))
    assert bool(jnp.all(ctw > cwv_only))


def test_mse_finite_and_per_cell():
    state, _, hc, _ = _setup()
    mse = column_moist_static_energy_plane(state, hc)
    assert mse.shape == (4, 4)
    assert bool(jnp.all(jnp.isfinite(mse)))


def test_mse_increases_with_q_v():
    state, _, hc, _ = _setup()
    mse_initial = float(column_moist_static_energy_plane(state, hc)[0, 0])
    state_wet = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 0].add(0.005),
        ),
    )
    mse_wet = float(column_moist_static_energy_plane(state_wet, hc)[0, 0])
    assert mse_wet > mse_initial


def test_theta_e_proxy_greater_than_theta_when_moist():
    """θ_e_proxy > θ for any positive q_v."""
    state, _, hc, _ = _setup()
    theta_e = pseudo_equivalent_potential_temperature(state, hc)
    theta_total = hc.theta_ref + state.theta_prime.data
    assert bool(jnp.all(theta_e >= theta_total))


def test_theta_e_proxy_returns_nan_for_nonpositive_T():
    """Codex iter-3: T <= 0 → NaN (not silent inf via clipped denom)."""
    state, _, hc, _ = _setup()
    # Drive T to zero by pushing theta_total = 0 via large negative theta_prime.
    theta_kill = -hc.theta_ref[None, None, :] * jnp.ones((4, 4, 8))
    state_bad = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kill),
    )
    theta_e = pseudo_equivalent_potential_temperature(state_bad, hc)
    assert bool(jnp.all(jnp.isnan(theta_e)))


def test_domain_mean_profiles_shape():
    state, _, hc, _ = _setup()
    profiles = domain_mean_profiles_plane(state, hc)
    assert profiles.T.shape == (8,)
    assert profiles.q_v.shape == (8,)
    assert profiles.theta_e_proxy.shape == (8,)
    assert profiles.T_std.shape == (8,)


def test_domain_mean_std_is_zero_for_uniform_state():
    state, _, hc, _ = _setup()
    profiles = domain_mean_profiles_plane(state, hc)
    np.testing.assert_allclose(np.asarray(profiles.T_std), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(profiles.q_v_std), 0.0, atol=1.0e-10)


def test_cloud_fraction_is_one_when_qc_above_threshold():
    state, _, hc, _ = _setup()
    cf = cloud_fraction_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cf), np.ones(8), rtol=0.0, atol=0.0)


def test_cloud_fraction_zero_at_zero_qc():
    state, _, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 1].set(0.0),
        ),
    )
    cf = cloud_fraction_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cf), np.zeros(8), rtol=0.0, atol=0.0)


def test_precipitation_proxy_scales_with_q_r():
    state, _, hc, _ = _setup()
    p_1 = precipitation_rate_proxy_plane(state, hc)
    state2 = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 2].multiply(2.0),
        ),
    )
    p_2 = precipitation_rate_proxy_plane(state2, hc)
    np.testing.assert_allclose(np.asarray(p_2 / p_1), 2.0, rtol=1.0e-12)


def test_precipitation_proxy_zero_at_zero_q_r():
    state, _, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 2].set(0.0),
        ),
    )
    p = precipitation_rate_proxy_plane(state, hc)
    np.testing.assert_array_equal(np.asarray(p), np.zeros((4, 4)))


def test_diagnostics_jit_compilable():
    state, _, hc, _ = _setup()
    fn = jax.jit(
        lambda s: (
            column_water_vapor_plane(s, hc),
            column_moist_static_energy_plane(s, hc),
            pseudo_equivalent_potential_temperature(s, hc),
            domain_mean_profiles_plane(s, hc),
            cloud_fraction_profile_plane(s, hc),
            precipitation_rate_proxy_plane(s, hc),
        )
    )
    out = fn(state)
    for arr in jax.tree_util.tree_leaves(out):
        assert bool(jnp.all(jnp.isfinite(arr)))


def test_diagnostics_support_jax_grad():
    state, _, hc, _ = _setup()

    def loss_fn(qv_data):
        s = state._replace(
            tracers=state.tracers.replace(
                data=state.tracers.data.at[..., 0].set(qv_data),
            ),
        )
        return jnp.sum(column_moist_static_energy_plane(s, hc) ** 2)

    g = jax.grad(loss_fn)(state.tracers.data[..., 0])
    assert g.shape == state.tracers.data[..., 0].shape
    assert bool(jnp.all(jnp.isfinite(g)))


# Codex iter-2: tracer-slot validation
def test_cwv_rejects_negative_qv_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        column_water_vapor_plane(state, hc, qv_slot=-1)


def test_cwv_rejects_oob_qv_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        column_water_vapor_plane(state, hc, qv_slot=99)


def test_total_water_rejects_duplicate_slots():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="duplicates"):
        column_total_water_plane(state, hc, water_slot_indices=(0, 0))


def test_total_water_rejects_empty():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="at least one slot"):
        column_total_water_plane(state, hc, water_slot_indices=())


def test_cloud_fraction_rejects_negative_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        cloud_fraction_profile_plane(state, hc, qc_slot=-1)


def test_precip_rejects_oob_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        precipitation_rate_proxy_plane(state, hc, qr_slot=99)


# Codex iter-2: plane-shape contract
def test_diagnostics_reject_non_plane_layout():
    """4D rho_prime (cubed-sphere shape) must raise."""
    state, _, hc, _ = _setup()
    state_4d = state._replace(
        rho_prime=state.rho_prime.replace(
            data=jnp.zeros((6, 4, 4, 8), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="plane-only"):
        column_water_vapor_plane(state_4d, hc)


def test_diagnostics_reject_mismatched_theta_shape():
    state, _, hc, _ = _setup()
    state_bad = state._replace(
        theta_prime=state.theta_prime.replace(
            data=jnp.zeros((2, 2, 8), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="theta_prime"):
        column_moist_static_energy_plane(state_bad, hc)


def test_diagnostics_reject_mismatched_tracer_shape():
    state, _, hc, _ = _setup()
    state_bad = state._replace(
        tracers=state.tracers.replace(
            data=jnp.zeros((2, 2, 8, 3), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="tracers shape"):
        column_water_vapor_plane(state_bad, hc)
