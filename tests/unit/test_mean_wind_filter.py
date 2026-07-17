"""Domain-mean wind removal tests for periodic CRM."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.mean_wind_filter import (
    compute_horizontal_mean_wind, remove_horizontal_mean_wind,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=8, ny=8, nlev=4, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(4, H=4_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    return make_rest_state(grid, hc, dtype=jnp.float64), grid, hc, tm


def test_remove_mean_yields_zero_mean_u_and_v():
    """Post-removal, horizontal means must be zero per level."""
    state, *_ = _setup()
    rng = np.random.default_rng(0)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape)),
        ),
        v=state.v.replace(
            data=jnp.asarray(rng.standard_normal(state.v.data.shape)),
        ),
    )
    new_state = remove_horizontal_mean_wind(state)
    u_mean = np.asarray(jnp.mean(new_state.u.data, axis=(0, 1)))
    v_mean = np.asarray(jnp.mean(new_state.v.data, axis=(0, 1)))
    np.testing.assert_allclose(u_mean, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(v_mean, 0.0, atol=1.0e-14)


def test_remove_mean_preserves_horizontal_variance():
    """Subtracting a uniform per-level constant doesn't change the
    spatial variance of (u, v)."""
    state, *_ = _setup()
    rng = np.random.default_rng(1)
    u_data = jnp.asarray(rng.standard_normal(state.u.data.shape))
    state = state._replace(u=state.u.replace(data=u_data))
    var_before = float(jnp.var(state.u.data, axis=(0, 1)).sum())
    new_state = remove_horizontal_mean_wind(state)
    var_after = float(jnp.var(new_state.u.data, axis=(0, 1)).sum())
    np.testing.assert_allclose(var_after, var_before, rtol=1.0e-12)


def test_remove_mean_idempotent():
    """Applying twice = applying once."""
    state, *_ = _setup()
    rng = np.random.default_rng(2)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape)),
        ),
    )
    once = remove_horizontal_mean_wind(state)
    twice = remove_horizontal_mean_wind(once)
    np.testing.assert_allclose(
        np.asarray(once.u.data), np.asarray(twice.u.data),
        rtol=0.0, atol=1.0e-14,
    )


def test_remove_mean_at_rest_is_noop():
    """Rest state has u=v=0 → mean=0 → no change."""
    state, *_ = _setup()
    new_state = remove_horizontal_mean_wind(state)
    np.testing.assert_array_equal(
        np.asarray(new_state.u.data), np.asarray(state.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.v.data), np.asarray(state.v.data),
    )


def test_remove_mean_does_not_touch_w_theta_rho_tracers():
    """Only u, v are filtered; w, theta', rho', tracers, phis preserved."""
    state, *_ = _setup()
    rng = np.random.default_rng(3)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape)),
        ),
        theta_prime=state.theta_prime.replace(
            data=jnp.asarray(
                rng.standard_normal(state.theta_prime.data.shape),
            ),
        ),
        w=state.w.replace(
            data=jnp.asarray(rng.standard_normal(state.w.data.shape)),
        ),
    )
    new_state = remove_horizontal_mean_wind(state)
    np.testing.assert_array_equal(
        np.asarray(new_state.w.data), np.asarray(state.w.data),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.theta_prime.data),
        np.asarray(state.theta_prime.data),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.rho_prime.data),
        np.asarray(state.rho_prime.data),
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.tracers.data),
        np.asarray(state.tracers.data),
    )


def test_remove_mean_reduces_total_KE():
    """Post-removal kinetic energy must be ≤ original (zero when the
    original mean was already zero)."""
    state, *_ = _setup()
    rng = np.random.default_rng(4)
    u_data = jnp.asarray(
        rng.standard_normal(state.u.data.shape) + 5.0,
    )
    state = state._replace(u=state.u.replace(data=u_data))
    ke_before = float(jnp.sum(state.u.data ** 2))
    new_state = remove_horizontal_mean_wind(state)
    ke_after = float(jnp.sum(new_state.u.data ** 2))
    assert ke_after < ke_before


def test_compute_mean_diagnostic_matches_jnp_mean():
    state, *_ = _setup()
    rng = np.random.default_rng(5)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape)),
        ),
        v=state.v.replace(
            data=jnp.asarray(rng.standard_normal(state.v.data.shape)),
        ),
    )
    u_mean, v_mean = compute_horizontal_mean_wind(state)
    np.testing.assert_array_equal(
        np.asarray(u_mean),
        np.asarray(jnp.mean(state.u.data, axis=(0, 1))),
    )
    np.testing.assert_array_equal(
        np.asarray(v_mean),
        np.asarray(jnp.mean(state.v.data, axis=(0, 1))),
    )


def test_remove_mean_jit_compilable():
    state, *_ = _setup()
    fn = jax.jit(remove_horizontal_mean_wind)
    out = fn(state)
    assert bool(jnp.all(jnp.isfinite(out.u.data)))


def test_remove_mean_rejects_non_state_protocol():
    """Bare dict / tuple without Field protocol → TypeError."""
    with pytest.raises(TypeError, match="requires a state"):
        remove_horizontal_mean_wind({"u": jnp.zeros(3)})


def test_remove_mean_rejects_non_plane_layout():
    """Codex iter-1: cubed-sphere state with 4D u.data must raise."""
    state, *_ = _setup()
    # Synthesize a 4D u.data shape to mimic cubed-sphere layout.
    state_4d = state._replace(
        u=state.u.replace(
            data=jnp.zeros((6, 4, 4, 4), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="plane-only"):
        remove_horizontal_mean_wind(state_4d)


def test_compute_mean_rejects_non_plane_layout():
    state, *_ = _setup()
    state_4d = state._replace(
        u=state.u.replace(
            data=jnp.zeros((6, 4, 4, 4), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="plane-only"):
        compute_horizontal_mean_wind(state_4d)


def test_compute_mean_rejects_non_state_protocol():
    """Codex iter-1: diagnostic also enforces the Field protocol."""
    with pytest.raises(TypeError, match="requires a state"):
        compute_horizontal_mean_wind({"u": jnp.zeros(3)})


def test_remove_mean_supports_jax_grad():
    state, *_ = _setup()
    rng = np.random.default_rng(6)
    u_init = jnp.asarray(rng.standard_normal(state.u.data.shape))

    def loss_fn(u_data):
        s = state._replace(u=state.u.replace(data=u_data))
        s = remove_horizontal_mean_wind(s)
        return jnp.sum(s.u.data ** 2)

    g = jax.grad(loss_fn)(u_init)
    assert g.shape == u_init.shape
    assert bool(jnp.all(jnp.isfinite(g)))
