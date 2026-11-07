"""A closed-wall v-face of ZERO extent must not poison the barotropic solver.

The ORCA2 tripolar card's southernmost v-row carries ``e1v == e2v == 0`` on
all 180 longitudes.  ``_nemo_ssh_avg_prep`` used to build ``r1_e1e2v`` as a
plain reciprocal, so that row was ``+inf``; ``r3_v`` then multiplied it by the
dry face's exactly-zero ``r1_v0`` (``0 * inf``), the entry inverse face depth
``r1_v_entry`` came out NaN on the whole row, and the barotropic bottom-drag
statement carried the NaN into the sea surface within two substeps.

These are the two claims the fix makes, and each is checked separately:
  1. a zero-area face gets a reciprocal of exactly zero, and the entry inverse
     face depth built from it is finite everywhere (the defect);
  2. wherever the face area is positive the reciprocal is BITWISE the plain
     ``1.0 / area``, so no card with a non-degenerate metric moves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: E402
    _nemo_ssh_avg_apply,
    _nemo_ssh_avg_prep,
    _reciprocal_face_area,
)


def test_zero_area_face_reciprocal_is_zero_and_positive_faces_are_exact():
    area = jnp.asarray([[0.0, 1.0, 4.629921e04], [2.5e09, 0.0, 1.0e-30]])
    got = np.asarray(_reciprocal_face_area(area))
    plain = 1.0 / np.asarray(area, dtype=np.float64)
    positive = np.asarray(area) > 0.0
    assert np.all(np.isfinite(got))
    assert np.all(got[~positive] == 0.0)
    # BITWISE equality on every face that has area.
    assert np.array_equal(
        got[positive].view(np.uint64), plain[positive].view(np.uint64))


def _degenerate_south_row_grid():
    """A tiny lat-lon C-grid whose southernmost v-row has zero extent, the
    way the ORCA2 tripolar card's wall row does."""
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry

    grid = ensure_geometry(create_latlon_grid(n_lat=6, n_lon=8))
    dx_v = np.asarray(grid.dx_v, dtype=np.float64).copy()
    dy_v = np.asarray(grid.dy_v, dtype=np.float64).copy()
    dx_v[0] = 0.0
    dy_v[0] = 0.0
    return grid._replace(dx_v=jnp.asarray(dx_v), dy_v=jnp.asarray(dy_v))


def test_entry_inverse_face_depth_is_finite_on_a_zero_extent_wall_row():
    grid = _degenerate_south_row_grid()
    n_lat, n_lon = grid.n_lat, grid.n_lon
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    # The wall row is dry, exactly as it is on the card.
    v_mask = jnp.ones((n_lat + 1, n_lon)).at[0].set(0.0).at[-1].set(0.0)
    H_bathy = jnp.full((n_lat, n_lon), 4000.0)
    eta = jnp.asarray(
        -0.05 * np.cos(np.linspace(0.0, 3.0, n_lat * n_lon)).reshape(n_lat, n_lon))
    area = jnp.asarray(grid.area_T)

    prep = _nemo_ssh_avg_prep(H_bathy, mask, grid, jnp.float64)
    assert bool(np.all(np.isfinite(np.asarray(prep[3])))), "r1_e1e2v is not finite"

    _, _, r1_u_entry, r1_v_entry = _nemo_ssh_avg_apply(
        eta, u_mask, v_mask, grid, area, prep, return_entry_inverse=True)
    r1_v_entry = np.asarray(r1_v_entry)
    assert np.all(np.isfinite(r1_v_entry)), (
        f"{int((~np.isfinite(r1_v_entry)).sum())} non-finite entry inverse "
        "v-face depths on the zero-extent wall row")
    assert np.all(np.isfinite(np.asarray(r1_u_entry)))
    assert np.all(r1_v_entry[0] == 0.0)
