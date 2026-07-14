"""Single-rank tests for MPI-aware RCE helpers.

Verifies the ``layout.n_ranks == 1`` short-circuit matches the
serial variants exactly. Real multi-rank coverage lives in
``tests/distributed/test_rce_mpi.py`` (requires ``mpirun``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.mean_wind_filter import (
    remove_horizontal_mean_wind,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    compute_total_water_mass_plane_mpi,
    fix_moist_mass_plane_mpi,
    remove_horizontal_mean_wind_plane_mpi,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


def _setup(nx=4, ny=4, nlev=6):
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=2_000.0, dy=2_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(nlev, H=6_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(0.001)
    tracers = tracers.at[..., 2].set(0.0005)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=ny, nx_global=nx,
    )
    return state, grid, hc, layout


def test_mean_wind_singlerank_matches_serial():
    state, _, _, layout = _setup()
    rng = np.random.default_rng(0)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape)),
        ),
    )
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    out_mpi = remove_horizontal_mean_wind_plane_mpi(
        state, layout, owned_mask,
    )
    out_serial = remove_horizontal_mean_wind(state)
    np.testing.assert_array_equal(
        np.asarray(out_mpi.u.data), np.asarray(out_serial.u.data),
    )


def test_total_water_singlerank_matches_serial():
    state, grid, hc, layout = _setup()
    # Owned mask not consulted on single-rank path.
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    m_mpi = float(compute_total_water_mass_plane_mpi(
        state, hc, grid, layout, owned_mask,
    ))
    m_serial = float(compute_total_water_mass_plane(state, hc, grid))
    np.testing.assert_allclose(m_mpi, m_serial, rtol=1.0e-14)


def test_fix_moist_mass_singlerank_matches_serial():
    state, grid, hc, layout = _setup()
    target = compute_total_water_mass_plane(state, hc, grid) * 0.7
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    fixed_mpi = fix_moist_mass_plane_mpi(
        state, hc, grid, layout, owned_mask, target,
    )
    fixed_serial = fix_moist_mass_plane(state, hc, grid, target)
    np.testing.assert_array_equal(
        np.asarray(fixed_mpi.tracers.data),
        np.asarray(fixed_serial.tracers.data),
    )


def test_mean_wind_rejects_non_plane_state_on_multirank_path():
    """Multi-rank validation path rejects non-3D u/v."""
    state, _, _, _ = _setup()
    layout_2 = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=4, nx_global=8,
    )
    state_4d = state._replace(
        u=state.u.replace(
            data=jnp.zeros((6, 4, 4, 6), dtype=jnp.float64),
        ),
    )
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    with pytest.raises(ValueError, match="plane"):
        remove_horizontal_mean_wind_plane_mpi(
            state_4d, layout_2, owned_mask,
        )


def test_layout_validation_missing_attrs():
    """Plain dict / object without n_ranks should raise."""
    state, _, _, _ = _setup()
    bad_layout = object()
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    with pytest.raises(TypeError, match="n_ranks"):
        remove_horizontal_mean_wind_plane_mpi(
            state, bad_layout, owned_mask,
        )


def test_owned_mask_shape_validation():
    """Wrong-shape mask should raise on multi-rank path."""
    state, grid, hc, _ = _setup()
    layout_2 = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=4, nx_global=8,
    )
    bad_mask = jnp.ones((3, 3), dtype=jnp.float64)
    with pytest.raises(ValueError, match="owned_mask"):
        compute_total_water_mass_plane_mpi(
            state, hc, grid, layout_2, bad_mask,
        )


def test_water_mpi_rejects_non_3d_rho_prime():
    """Codex iter-2: full shape contract on multi-rank path."""
    state, grid, hc, _ = _setup()
    layout_2 = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=4, nx_global=8,
    )
    state_4d = state._replace(
        rho_prime=state.rho_prime.replace(
            data=jnp.zeros((6, 4, 4, 6), dtype=jnp.float64),
        ),
    )
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    with pytest.raises(ValueError, match="rho_prime"):
        compute_total_water_mass_plane_mpi(
            state_4d, hc, grid, layout_2, owned_mask,
        )


def test_water_mpi_rejects_3d_tracers():
    """Codex iter-3: 3D tracers without trailing tracer axis must
    raise (otherwise shape[-1] would alias the vertical axis)."""
    state, grid, hc, _ = _setup()
    layout_2 = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=4, nx_global=8,
    )
    state_bad = state._replace(
        tracers=state.tracers.replace(
            data=jnp.zeros((4, 4, 6), dtype=jnp.float64),
        ),
    )
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    with pytest.raises(ValueError, match="tracers must be 4D"):
        compute_total_water_mass_plane_mpi(
            state_bad, hc, grid, layout_2, owned_mask,
        )


def test_water_mpi_rejects_duplicate_slots():
    state, grid, hc, _ = _setup()
    layout_2 = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=4, nx_global=8,
    )
    owned_mask = jnp.ones((4, 4), dtype=jnp.float64)
    with pytest.raises(ValueError, match="duplicates"):
        compute_total_water_mass_plane_mpi(
            state, hc, grid, layout_2, owned_mask,
            water_slot_indices=(0, 0),
        )
