"""#1455: the vertex-Coriolis placement reaches every C-grid Coriolis path.

The option lives in ONE array, the geometry's ``f_v``, and every path reads
that array through one of exactly two helpers -- ``vertex_coriolis`` (the
barotropic EEN pre-block and the 3-D EEN/ENE vorticity flux) and
``coriolis_at_faces`` (the semi-implicit and ``explicit_ab2`` face-f
Coriolis).  These tests prove that rather than asserting it: both helpers'
output must move with the option, and the barotropic pre-block's stored
``f_vtx`` must BE ``vertex_coriolis``'s output, with no second wiring step.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import vertex_coriolis


def _pair(**kw):
    base = dict(n_lat=12, n_lon=4, omega=1.0e-4, dtype=jnp.float64)
    base.update(kw)
    n_lat = base.pop("n_lat")
    n_lon = base.pop("n_lon")
    return (create_latlon_geometry(n_lat, n_lon, **base),
            create_latlon_geometry(n_lat, n_lon,
                                   coriolis_placement="face_latitude", **base))


def test_vertex_coriolis_carries_the_placement_option():
    """Helper 1: the EEN/ENE vorticity-flux door."""
    avg, face = _pair()
    f_avg = np.asarray(vertex_coriolis(avg))
    f_face = np.asarray(vertex_coriolis(face))
    assert f_avg.shape == f_face.shape == (13, 5)
    assert not np.array_equal(f_avg, f_face)
    # and it is exactly grid.f_v plus the periodic wrap column, both ways
    for geom, f in ((avg, f_avg), (face, f_face)):
        assert np.array_equal(f[:, :4], np.asarray(geom.f_v))
        assert np.array_equal(f[:, 4], np.asarray(geom.f_v)[:, 0])


def test_the_barotropic_EEN_preblock_stores_that_same_array():
    """The barotropic Coriolis path: its f_vtx must be vertex_coriolis's
    output, so the option reaches it with no second wiring step."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs)
    _, face = _pair()
    n_lat, n_lon = 12, 4
    h_k = jnp.ones((n_lat, n_lon, 3), dtype=jnp.float64)
    mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
    u_mask = jnp.ones((n_lat, n_lon + 1), dtype=jnp.float64)
    v_mask = jnp.ones((n_lat + 1, n_lon), dtype=jnp.float64)
    pre = _build_een_barotropic_inputs(h_k, face, mask, u_mask, v_mask,
                                       jnp.float64)
    assert np.array_equal(np.asarray(pre["f_vtx"]),
                          np.asarray(vertex_coriolis(face)))


def test_the_hand_computed_value_survives_all_the_way_to_the_preblock():
    """End to end, with a number a reader can check: at 30 degrees and
    omega = 1e-4 the vertex Coriolis is exactly 1e-4."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs)
    lat_face = jnp.deg2rad(jnp.array([-30.0, 0.0, 30.0, 60.0]))
    lat_c = 0.5 * (lat_face[:-1] + lat_face[1:])
    geom = create_latlon_geometry(
        3, 4, omega=1.0e-4, dtype=jnp.float64, lat_1d=lat_c,
        lat_face_1d=lat_face, coriolis_placement="face_latitude")
    h_k = jnp.ones((3, 4, 2), dtype=jnp.float64)
    pre = _build_een_barotropic_inputs(
        h_k, geom, jnp.ones((3, 4), dtype=jnp.float64),
        jnp.ones((3, 5), dtype=jnp.float64),
        jnp.ones((4, 4), dtype=jnp.float64), jnp.float64)
    assert float(np.asarray(pre["f_vtx"])[2, 0]) == pytest.approx(1.0e-4,
                                                                  rel=1e-12)


def test_coriolis_at_faces_carries_it_too():
    """Helper 2: the semi-implicit / explicit_ab2 face-f door.

    ``f_v`` must move with the option and ``f_u`` must NOT -- on a lat-lon
    grid the u-point shares the tracer row's latitude, so f_u is already f at
    its own point and changing it would be a second, unasked-for change.
    """
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
    avg, face = _pair()
    fu_a, fv_a = coriolis_at_faces(avg, jnp.float64)
    fu_f, fv_f = coriolis_at_faces(face, jnp.float64)
    assert np.array_equal(np.asarray(fu_a), np.asarray(fu_f))
    assert not np.array_equal(np.asarray(fv_a), np.asarray(fv_f))
