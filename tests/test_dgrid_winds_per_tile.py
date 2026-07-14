"""Smoke tests for ``plot_dgrid_winds_per_tile`` (issue #274)."""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import pytest


def test_plot_dgrid_winds_per_tile_writes_png():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from legoesm.visualization.maps import plot_dgrid_winds_per_tile

    n = 8
    rng = jnp.linspace(-10.0, 10.0, 6 * (n + 1) * (n + 1)).reshape(
        6, n + 1, n + 1
    )
    u_d = rng
    v_d = jnp.roll(rng, 1, axis=0)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dgrid.png")
        fig, axes = plot_dgrid_winds_per_tile(
            u_d, v_d, title="test", save_path=out,
        )
        assert axes.shape == (6, 2)
        assert os.path.isfile(out) and os.path.getsize(out) > 0
        matplotlib.pyplot.close(fig)


def test_plot_dgrid_winds_per_tile_rejects_shape_mismatch():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from legoesm.visualization.maps import plot_dgrid_winds_per_tile

    u_d = jnp.zeros((6, 5, 5))
    v_d = jnp.zeros((6, 5, 6))
    with pytest.raises(ValueError, match="shape mismatch"):
        plot_dgrid_winds_per_tile(u_d, v_d)


def test_plot_dgrid_winds_per_tile_rejects_non_cubed_sphere():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from legoesm.visualization.maps import plot_dgrid_winds_per_tile

    u_d = jnp.zeros((4, 5, 5))
    v_d = jnp.zeros((4, 5, 5))
    with pytest.raises(ValueError, match="leading axis size 6"):
        plot_dgrid_winds_per_tile(u_d, v_d)


def test_plot_dgrid_winds_per_tile_accepts_fv3_edge_stagger():
    """FV3 native D-grid: u_d on south/north edges (6, n, n+1),
    v_d on west/east edges (6, n+1, n) — also a valid input."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from legoesm.visualization.maps import plot_dgrid_winds_per_tile

    n = 8
    u_d = jnp.ones((6, n, n + 1))
    v_d = jnp.ones((6, n + 1, n))

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dgrid_fv3.png")
        fig, axes = plot_dgrid_winds_per_tile(u_d, v_d, save_path=out)
        assert axes.shape == (6, 2)
        assert os.path.isfile(out)
        matplotlib.pyplot.close(fig)


def test_platecarree_winds_use_geographic_rotation():
    """Williamson-2 alpha=0 is purely zonal. The CLI's PlateCarree
    pipeline therefore must use ``dgrid_to_center_geographic`` so that
    the meridional (v_north) plot is near zero — the previous
    face-local averaging path returned tens of m/s for the same input.
    """
    pytest.importorskip("matplotlib")
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from legoesm.core.operators_cdgrid import (
        center_to_dgrid_vector, dgrid_to_center_geographic,
        dgrid_to_center_vector,
    )
    from tests.test_cases.williamson import williamson_test2

    grid = create_cubed_sphere(16)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Williamson-2 (alpha = 0): purely zonal solid-body rotation in the
    # geographic frame, expressed by ``williamson_test2`` as face-local
    # (u, v) at cell centres.
    sw = williamson_test2(grid)
    u_cc = sw.u.data if hasattr(sw.u, "data") else sw.u
    v_cc = sw.v.data if hasattr(sw.v, "data") else sw.v
    u_d, v_d = center_to_dgrid_vector(u_cc, v_cc, cdgrid)

    _, v_face_local = dgrid_to_center_vector(u_d, v_d)
    _, v_geo = dgrid_to_center_geographic(u_d, v_d, cdgrid)

    v_face_max = float(jnp.max(jnp.abs(v_face_local)))
    v_geo_max = float(jnp.max(jnp.abs(v_geo)))
    # Geographic v must be small (alpha = 0 ⇒ no real meridional wind).
    assert v_geo_max < 5.0, f"v_geo_max={v_geo_max}"
    # Face-local v reflects panel rotation and must be much larger —
    # if these matched, the bug would have gone undetected.
    assert v_face_max > 5.0 * v_geo_max + 1.0
