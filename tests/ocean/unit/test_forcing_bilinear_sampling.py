"""Bilinear CORE-II forcing sampling onto point clouds (tripole T points).

WHY: nearest-neighbour sampling of the ~1.9-degree CORE-II grid onto the
1-degree tripole copied each coarse source ROW onto 1-2 adjacent target rows,
printing zonally aligned bands into every forcing channel; surface mixing
inherited them as the unphysical zonal MLD banding (user report 2026-08-26,
confirmed to survive 5-day averaging).  The fix samples the tripole
bilinearly on BOTH the host path and the lax.scan device path.

Non-vacuity: test_no_row_duplication FAILS under method="nearest" (asserted
here explicitly), so reverting the tripole dispatch to nearest goes red.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.coupler.omip2_applicator import (
    _bilinear_interp_to_points,
    _bilinear_point_maps,
    _nn_interp_to_points,
)


def _t62ish_axes():
    # Gaussian-ish latitude spacing (non-uniform, ascending) + uniform lon.
    lat = np.linspace(-89.5, 89.5, 94) + 0.3 * np.sin(np.arange(94))
    lat = np.sort(lat)
    lon = np.arange(0.0, 360.0, 360.0 / 192.0)
    return lat, lon


def test_linear_field_reproduced_exactly():
    """Bilinear on a field linear in lat and lon is exact (nearest is not)."""
    lat, lon = _t62ish_axes()
    field = 2.0 * lat[:, None] + 0.1 * lon[None, :]
    tgt_lat = np.linspace(-60.0, 60.0, 121)          # 1-deg rows
    tgt_lon = np.full_like(tgt_lat, 200.25)
    out = _bilinear_interp_to_points(field, lat, lon, tgt_lat, tgt_lon)
    np.testing.assert_allclose(out, 2.0 * tgt_lat + 0.1 * 200.25, atol=1e-9)


def test_no_row_duplication_and_nearest_fails_it():
    """Adjacent 1-deg target rows must differ on a lat-varying field.

    Under nearest, ~1.9-deg source rows are copied to 1-2 target rows each,
    so ~half the adjacent-row differences are EXACTLY zero — the banding.
    """
    lat, lon = _t62ish_axes()
    field = np.sin(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    tgt_lat = np.linspace(-60.0, 60.0, 121)
    tgt_lon = np.full_like(tgt_lat, 100.5)
    bil = _bilinear_interp_to_points(field, lat, lon, tgt_lat, tgt_lon)
    nn = _nn_interp_to_points(field, lat, lon, tgt_lat, tgt_lon)
    zero_bil = np.sum(np.abs(np.diff(bil)) < 1e-12)
    zero_nn = np.sum(np.abs(np.diff(nn)) < 1e-12)
    assert zero_bil == 0, "bilinear must vary smoothly row to row"
    assert zero_nn > 20, ("nearest SHOULD duplicate rows on this geometry -- "
                          "if not, this regression test has lost its teeth")


def test_periodic_seam():
    """A target between the last and first source column interpolates across
    the wrap, never extrapolates."""
    lat, lon = _t62ish_axes()
    # field linear in lon with a wrap-consistent sawtooth: use cos(lon)
    field = np.ones_like(lat)[:, None] * np.cos(np.deg2rad(lon))[None, :]
    tgt_lat = np.array([0.0, 10.0])
    tgt_lon = np.array([359.9, 0.7])                 # both inside the seam gap
    out = _bilinear_interp_to_points(field, lat, lon, tgt_lat, tgt_lon)
    np.testing.assert_allclose(out, np.cos(np.deg2rad(tgt_lon)), atol=1e-3)


def test_clamped_beyond_polar_edge():
    lat, lon = _t62ish_axes()
    field = lat[:, None] * np.ones_like(lon)[None, :]
    out = _bilinear_interp_to_points(field, lat, lon,
                                     np.array([-89.9, 89.9]),
                                     np.array([10.0, 10.0]))
    assert out[0] == pytest.approx(lat[0])
    assert out[1] == pytest.approx(lat[-1])


def test_weights_sum_to_one():
    lat, lon = _t62ish_axes()
    rng = np.random.default_rng(0)
    tgt_lat = rng.uniform(-88, 88, 500)
    tgt_lon = rng.uniform(0, 360, 500)
    _, _, w4 = _bilinear_point_maps(lat, lon, tgt_lat, tgt_lon)
    np.testing.assert_allclose(w4.sum(axis=0), 1.0, atol=1e-12)


def test_tripole_dispatch_uses_bilinear():
    """The tripole sample path must produce the bilinear (smooth) result --
    reverting the dispatch to nearest goes red here."""
    import types

    import jax.numpy as jnp

    from legoesm.ocean.coupler.omip2_applicator import sample_omip2_forcing

    lat, lon = _t62ish_axes()
    n_rec = 2
    smooth = np.sin(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    chans = {}
    for name in ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                 "precip"):
        chans[name] = np.repeat(smooth[None], n_rec, axis=0)
    forc = types.SimpleNamespace(lat=lat, lon=lon, **chans)
    glat = np.deg2rad(np.linspace(-60, 60, 121))[:, None] * np.ones((1, 4))
    glon = np.ones((121, 1)) * np.deg2rad(np.array([10., 20., 30., 40.]))[None]
    grid = types.SimpleNamespace(lat_T=jnp.asarray(glat),
                                 lon_T=jnp.asarray(glon))
    out = sample_omip2_forcing(forc, 0, grid, "tripole")
    du = np.abs(np.diff(np.asarray(out["u10"]), axis=0))
    assert np.sum(du < 1e-12) == 0, (
        "tripole forcing sample duplicates rows: nearest-neighbour is back")
