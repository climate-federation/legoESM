"""Moist-mass fixer tests for the plane NH CRM."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=4, ny=4, nlev=4, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(4, H=4_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Seed q_v + q_c.
    tracers = jnp.zeros((4, 4, 4, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(0.001)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def test_compute_total_water_mass_finite_and_positive():
    state, grid, hc, _ = _setup()
    m = float(compute_total_water_mass_plane(state, hc, grid))
    assert m > 0.0
    assert np.isfinite(m)


def test_fix_restores_target_mass():
    state, grid, hc, _ = _setup()
    m_initial = float(compute_total_water_mass_plane(state, hc, grid))
    # Drift the q_v by +5% to simulate microphysics round-off.
    drifted_tracers = state.tracers.data.at[..., 0].set(
        state.tracers.data[..., 0] * 1.05,
    )
    drifted_state = state._replace(
        tracers=state.tracers.replace(data=drifted_tracers),
    )
    m_drifted = float(
        compute_total_water_mass_plane(drifted_state, hc, grid),
    )
    assert m_drifted > m_initial
    fixed_state = fix_moist_mass_plane(
        drifted_state, hc, grid, target_total_water=jnp.asarray(m_initial),
    )
    m_fixed = float(compute_total_water_mass_plane(fixed_state, hc, grid))
    np.testing.assert_allclose(m_fixed, m_initial, rtol=1.0e-12)


def test_fix_preserves_spatial_pattern():
    """Multiplicative fixer preserves the spatial DISTRIBUTION; only
    the absolute magnitude changes."""
    state, grid, hc, _ = _setup()
    rng = np.random.default_rng(0)
    rho_tracers = jnp.asarray(
        rng.uniform(1.0e-4, 1.0e-2, state.tracers.data.shape),
    )
    state = state._replace(
        tracers=state.tracers.replace(data=rho_tracers),
    )
    target = jnp.asarray(
        2.0 * float(compute_total_water_mass_plane(state, hc, grid)),
    )
    fixed = fix_moist_mass_plane(state, hc, grid, target_total_water=target)
    # Ratio q_fixed / q_initial should be constant per cell (in selected slots).
    for idx in (0, 1, 2):
        ratio = np.asarray(
            fixed.tracers.data[..., idx] / state.tracers.data[..., idx],
        )
        np.testing.assert_allclose(
            ratio, ratio[0, 0, 0], rtol=1.0e-10,
        )


def test_fix_positivity_preserved():
    """All-positive input → all-positive output."""
    state, grid, hc, _ = _setup()
    target = jnp.asarray(
        float(compute_total_water_mass_plane(state, hc, grid)) * 0.5,
    )
    fixed = fix_moist_mass_plane(state, hc, grid, target_total_water=target)
    assert float(jnp.min(fixed.tracers.data[..., 0])) >= 0.0
    assert float(jnp.min(fixed.tracers.data[..., 1])) >= 0.0


def test_fix_zero_current_zero_target_is_noop():
    """current = target = 0 → factor = 1 → no change (Codex iter-1)."""
    state, grid, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(data=jnp.zeros_like(state.tracers.data)),
    )
    fixed = fix_moist_mass_plane(
        state, hc, grid, target_total_water=jnp.asarray(0.0),
    )
    np.testing.assert_array_equal(
        np.asarray(fixed.tracers.data),
        np.zeros_like(np.asarray(fixed.tracers.data)),
    )


def test_fix_zero_current_positive_target_raises():
    """Codex iter-1: cannot scale zero up to a positive target."""
    state, grid, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(data=jnp.zeros_like(state.tracers.data)),
    )
    with pytest.raises(ValueError, match="cannot scale zero"):
        fix_moist_mass_plane(
            state, hc, grid, target_total_water=jnp.asarray(1.0e10),
        )


def test_fix_negative_current_raises():
    """Codex iter-1: negative current = malformed state → raise."""
    state, grid, hc, _ = _setup()
    # Inject a large negative q_v to drive current < 0.
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 0].set(-1.0),
        ),
    )
    with pytest.raises(ValueError, match="is negative"):
        fix_moist_mass_plane(
            state, hc, grid, target_total_water=jnp.asarray(1.0),
        )


def test_compute_mass_rejects_duplicate_slots():
    state, grid, hc, _ = _setup()
    with pytest.raises(ValueError, match="duplicates"):
        compute_total_water_mass_plane(
            state, hc, grid, water_slot_indices=(0, 0, 1),
        )


def test_compute_mass_rejects_negative_slot():
    state, grid, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        compute_total_water_mass_plane(
            state, hc, grid, water_slot_indices=(-1,),
        )


def test_compute_mass_rejects_oob_slot():
    state, grid, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        compute_total_water_mass_plane(
            state, hc, grid, water_slot_indices=(99,),
        )


def test_compute_mass_rejects_empty_slots():
    state, grid, hc, _ = _setup()
    with pytest.raises(ValueError, match="at least one slot"):
        compute_total_water_mass_plane(
            state, hc, grid, water_slot_indices=(),
        )


def test_compute_mass_rejects_non_plane_shape():
    """Codex iter-1: 4D rho_prime (cubed-sphere shape) raises."""
    state, grid, hc, _ = _setup()
    state_4d = state._replace(
        rho_prime=state.rho_prime.replace(
            data=jnp.zeros((6, 4, 4, 4), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="plane-only"):
        compute_total_water_mass_plane(state_4d, hc, grid)


def test_compute_mass_selects_only_supplied_slots():
    state, grid, hc, _ = _setup()
    m_all = float(compute_total_water_mass_plane(
        state, hc, grid, water_slot_indices=(0, 1, 2),
    ))
    m_qv_only = float(compute_total_water_mass_plane(
        state, hc, grid, water_slot_indices=(0,),
    ))
    assert m_qv_only < m_all


def test_fix_jit_compilable():
    state, grid, hc, _ = _setup()
    target = compute_total_water_mass_plane(state, hc, grid)
    fn = jax.jit(
        lambda s, t: fix_moist_mass_plane(
            s, hc, grid, target_total_water=t,
        ),
    )
    out = fn(state, target)
    assert bool(jnp.all(jnp.isfinite(out.tracers.data)))


def test_fix_supports_jax_grad():
    state, grid, hc, _ = _setup()
    target_init = compute_total_water_mass_plane(state, hc, grid)

    def loss_fn(qv_data):
        s = state._replace(
            tracers=state.tracers.replace(
                data=state.tracers.data.at[..., 0].set(qv_data),
            ),
        )
        s_fixed = fix_moist_mass_plane(
            s, hc, grid, target_total_water=target_init,
        )
        return jnp.sum(s_fixed.tracers.data[..., 0] ** 2)

    qv = state.tracers.data[..., 0]
    g = jax.grad(loss_fn)(qv)
    assert g.shape == qv.shape
    assert bool(jnp.all(jnp.isfinite(g)))
