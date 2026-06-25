"""Direct-import sanity tests for ``tests.test_cases.williamson_extended``.

Exercises the W6 (Rossby-Haurwitz wave-4) initial condition on every
grid type for which it is wired in M1.a (latlon, MPAS, spectral).
Cubed-sphere W6 is deferred to M1.b — see
``docs/validation/dycore_validation_catalog.md``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.test_cases.williamson_extended import (
    williamson_test6,
    williamson_test6_latlon,
    williamson_test6_mpas,
    williamson_test6_spectral,
)


def test_w6_latlon_shapes_and_finite():
    grid = create_latlon_grid(36, 72)
    state = williamson_test6_latlon(grid)
    assert state.h.data.shape == (36, 72)
    assert state.u.data.shape == (36, 72)
    assert state.v.data.shape == (36, 72)
    assert bool(jnp.all(jnp.isfinite(state.h.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.v.data)))


def test_w6_latlon_height_in_range():
    """W6 mean depth 8000 m; perturbations should be < ~3000 m."""
    grid = create_latlon_grid(36, 72)
    state = williamson_test6_latlon(grid)
    h = state.h.data
    h_min = float(jnp.min(h))
    h_max = float(jnp.max(h))
    assert 5000.0 < h_min, f"W6 h_min = {h_min} too low"
    assert h_max < 11000.0, f"W6 h_max = {h_max} too high"


def test_w6_mpas_shapes():
    mesh = create_voronoi_mesh(4)
    state = williamson_test6_mpas(mesh)
    assert state.h.data.shape == (mesh.nCells,)
    assert state.u.data.shape == (mesh.nEdges,)
    assert bool(jnp.all(jnp.isfinite(state.h.data)))


def test_w6_spectral_shapes():
    grid = create_gaussian_grid(21)
    state = williamson_test6_spectral(grid)
    assert state.vor_hat.data.shape == (grid.n_sh,)
    assert state.div_hat.data.shape == (grid.n_sh,)
    assert state.phi_hat.data.shape == (grid.n_sh,)
    assert bool(jnp.all(jnp.isfinite(state.vor_hat.data.real)))
    assert bool(jnp.all(jnp.isfinite(state.phi_hat.data.real)))


def test_w6_cubedsphere_deferred():
    """W6 cubed-sphere init is implemented but the matrix-script
    runner is deferred to M1.b — confirm the helper itself produces a
    finite state when given a CubedSphereGrid."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(8)
    state = williamson_test6(grid)
    assert state.h.data.shape == (6, 8, 8)
    assert bool(jnp.all(jnp.isfinite(state.h.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.v.data)))
