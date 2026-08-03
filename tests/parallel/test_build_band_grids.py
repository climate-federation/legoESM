"""Host test for ``build_band_grids`` — the per-device lat-band geometry
construction for the multi-GPU SPMD ocean step (eORCA025 ¼°).

Verifies the SPMD wrapper's band grids (built by reusing the tested MPI slicers
``make_latlon_band_layout`` + ``slice_cgrid_geometry_to_band``) tile the global
grid correctly: uniform bands, cell rows concatenate to the global, v/q rows
share one boundary face, ``total_area`` stays GLOBAL on every band, and the
fold is localized (inactive on a regular grid). No devices needed (pure
host-side construction).

Run: ``JAX_ENABLE_X64=1 pytest tests/parallel/test_build_band_grids.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.sharded_ocean_step import build_band_grids


def _geom(n_lat=48, n_lon=96):
    # the ocean model always converts its grid to a LatLonCGridGeometry
    return ensure_geometry(create_latlon_grid(n_lat=n_lat, n_lon=n_lon))


def test_uniform_bands_reassemble_global():
    n_lat, n_lon, N = 48, 96, 4
    geom = _geom(n_lat, n_lon)
    bands = build_band_grids(geom, N)
    assert len(bands) == N
    # every band is uniform (n_lat/N cell rows)
    for b in bands:
        assert int(b.n_lat) == n_lat // N
        assert int(b.n_lon) == n_lon

    # cell/u-row metrics (leading dim n_lat) concatenate to the global
    for nm in ("lat_T", "area_T", "f_T", "dx_u"):
        cat = np.concatenate([np.asarray(getattr(b, nm)) for b in bands], axis=0)
        np.testing.assert_allclose(
            cat, np.asarray(getattr(geom, nm)), err_msg=f"{nm} reassembly")

    # v/q-row metrics: each band owns n_lat/N+1 rows (its bounding faces); the
    # interior boundary v-face is SHARED (band r's last == band r+1's first)
    for nm in ("dy_v", "f_v", "area_q"):
        for r in range(N - 1):
            last = np.asarray(getattr(bands[r], nm))[-1]
            first = np.asarray(getattr(bands[r + 1], nm))[0]
            np.testing.assert_allclose(
                last, first, err_msg=f"{nm} shared v-face at band {r}/{r+1}")
        assert np.asarray(getattr(bands[0], nm)).shape[0] == n_lat // N + 1

    # total_area is the GLOBAL denominator on every band (NOT band-local)
    g_total = float(np.asarray(geom.total_area))
    for b in bands:
        np.testing.assert_allclose(float(np.asarray(b.total_area)), g_total,
                                   rtol=1e-12)


def test_non_divisible_raises():
    geom = _geom(48, 96)
    with pytest.raises(ValueError, match="divisible"):
        build_band_grids(geom, 5)   # 48 % 5 != 0


def test_regular_grid_fold_inactive_on_all_bands():
    """A regular lat-lon grid has no bipolar fold; every band's fold descriptor
    must be inactive (so no band wrongly takes a north-fold branch)."""
    bands = build_band_grids(_geom(48, 96), 4)
    for b in bands:
        fold = getattr(b, "fold", None)
        if fold is not None:
            assert not bool(getattr(fold, "is_active", False))


def _lead_dims(geom):
    """{field: leading dim} for every array field, with n_lat resolved."""
    import jax.numpy as jnp
    out = {}
    for name in geom._fields:
        v = getattr(geom, name)
        if isinstance(v, (jnp.ndarray, np.ndarray)) and np.ndim(v) >= 1:
            out[name] = int(np.shape(v)[0])
    return out


def test_every_geometry_field_keeps_its_stagger_through_slice_and_widen():
    """Generic stagger audit: EVERY array field is n_lat or n_lat+1 rows.

    Regression for the ``cos_lat_v`` addition, which the band slicer and the
    wide-halo widener both silently passed through at GLOBAL length (they
    enumerate field names by hand, so a new v-face field is missed).  Written
    generically so the NEXT added field cannot slip through either.
    """
    from legoesm.grids.halo_latlon import widen_cgrid_geometry_band
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, slice_cgrid_geometry_to_band,
    )
    n_lat, n_lon = 48, 96
    g = _geom(n_lat, n_lon)
    ref = _lead_dims(g)
    # every field is one of the two staggers on the global geometry
    assert set(ref.values()) <= {n_lat, n_lat + 1, n_lon}, ref

    layout = make_latlon_band_layout(1, 4, n_lat, n_lon)
    band = slice_cgrid_geometry_to_band(g, layout)
    nb = layout.n_lat_local
    for name, lead in _lead_dims(band).items():
        if ref[name] == n_lon:
            continue                       # lon-axis field (lon), untouched
        want = nb if ref[name] == n_lat else nb + 1
        assert lead == want, (
            f"{name}: band leading dim {lead}, expected {want} "
            f"(global {ref[name]}, n_lat={n_lat}) — slicer missed this field")

    halo = 3
    wide = widen_cgrid_geometry_band(band, halo)
    nw = nb + 2 * halo
    for name, lead in _lead_dims(wide).items():
        if ref[name] == n_lon:
            continue
        want = nw if ref[name] == n_lat else nw + 1
        assert lead == want, (
            f"{name}: widened leading dim {lead}, expected {want} "
            f"— widen_cgrid_geometry_band missed this field")
