"""Unit tests for the shared point-cloud helpers in
``scripts/plot/plot_mpas_omip_snapshot.py`` (used by both the MPAS restart path
and the FESOM ushow-zarr path)."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from scripts.plot.plot_mpas_omip_snapshot import (  # noqa: E402
    _zonal_mean_T,
    build_ocean_triangulation,
    fesom_wet_columns,
    plot_fesom_store,
    plot_panel,
    raster_mean,
)


def _open_v2_group(path):
    """A zarr FORMAT-2 store, as fesom_jax writes them (zarr 2.x); readable by zarr 2 and 3."""
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        return zarr.open_group(str(path), mode="w", zarr_format=2)
    return zarr.open_group(str(path), mode="w")


def _put(group, name, arr):
    arr = np.asarray(arr)
    if hasattr(group, "create_array"):
        group.create_array(name, shape=arr.shape, dtype=arr.dtype)[...] = arr
    else:
        group.create_dataset(name, data=arr)


def test_delaunay_land_filter():
    # a land point in the middle of an ocean quad: every Delaunay triangle touches it
    lon = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    lat = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    ocean = np.array([True, True, True, True, False])
    assert build_ocean_triangulation(lon, lat, ocean).mask.all()
    assert not build_ocean_triangulation(lon, lat, np.ones(5, bool)).mask.any()


def test_delaunay_seam_filter_independent_of_edge_length():
    # a triangle straddling the dateline; long-edge filter disabled so only the seam rule acts
    lon = np.array([179.5, 179.8, -179.7])
    lat = np.array([0.0, 1.0, 0.5])
    ocean = np.ones(3, bool)
    assert build_ocean_triangulation(lon, lat, ocean, max_edge_deg=400.0).mask.all()
    lon_same_side = np.array([0.5, 0.8, 0.3])
    assert not build_ocean_triangulation(lon_same_side, lat, ocean, max_edge_deg=400.0).mask.any()


def test_delaunay_long_edge_filter():
    lon = np.array([0.0, 6.0, 0.0])
    lat = np.array([0.0, 0.0, 1.0])
    ocean = np.ones(3, bool)
    assert build_ocean_triangulation(lon, lat, ocean).mask.all()          # 6-deg edge > 5
    assert not build_ocean_triangulation(lon, lat, ocean, max_edge_deg=7.0).mask.any()


def test_raster_mean_bins_wraps_and_leaves_holes():
    # two nodes in one 1-deg cell (mean), one node given in 0..360 longitude, one NaN node
    lon = np.array([0.2, 0.7, 359.5, 10.5])
    lat = np.array([0.5, 0.5, 0.5, 10.5])
    f = np.array([1.0, 3.0, 7.0, np.nan])
    lon_e, lat_e, grid = raster_mean(lon, lat, f, res_deg=1.0)
    assert grid.shape == (180, 360) and lon_e[0] == -180.0 and lat_e[-1] == 90.0
    assert grid[90, 180] == 2.0                 # cell [0,1)x[0,1): mean of 1 and 3
    assert grid[90, 179] == 7.0                 # 359.5 -> -0.5 lands in [-1,0)
    assert np.isnan(grid[100, 190])             # the NaN node leaves its cell empty
    assert np.isnan(grid[0, 0])                 # untouched cell stays a hole
    # boundary coordinates and non-finite coordinates
    _, _, g2 = raster_mean(np.array([180.0, 0.0, np.nan]), np.array([90.0, np.nan, 0.0]),
                           np.array([5.0, 6.0, 7.0]), res_deg=1.0)
    assert g2[179, 0] == 5.0                    # lon 180 wraps to -180 (first column); lat 90 -> last row
    assert np.count_nonzero(np.isfinite(g2)) == 1  # NaN-lon / NaN-lat nodes dropped, not binned
    for bad in (7.0, 120.0, 0.0):
        with pytest.raises(ValueError, match="divide 180"):
            raster_mean(lon, lat, f, res_deg=bad)


def test_raster_mean_gap_fill_is_bounded():
    lon = np.array([0.5, 5.5])
    lat = np.array([0.5, 0.5])
    f = np.array([1.0, 2.0])
    _, _, g0 = raster_mean(lon, lat, f, res_deg=1.0)
    assert np.isnan(g0[90, 181:185]).all()
    _, _, g1 = raster_mean(lon, lat, f, res_deg=1.0, fill_gap_cells=1)
    assert g1[90, 181] == 1.0 and g1[90, 184] == 2.0 and np.isnan(g1[90, 182:184]).all()
    _, _, g3 = raster_mean(lon, lat, f, res_deg=1.0, fill_gap_cells=3)
    assert np.isfinite(g3[90, 180:186]).all()
    assert np.isnan(g3[0, 0])                   # far-away cells stay holes


def test_fesom_wet_columns_masks_below_bottom_and_padding():
    # 3 rows in the store (2 layers + padding), 3 nodes with nlevels 3, 2, 3
    temp = np.array([[5.0, 5.0, 5.0],      # layer 0: wet everywhere
                     [6.0, 99.0, 6.0],     # layer 1: node 1 (nlevels=2) is BELOW its bottom
                     [77.0, 77.0, 77.0]])  # padding row: never plotted
    cols = fesom_wet_columns(temp, np.array([3, 2, 3]), n_layers=2)
    assert cols.shape == (3, 2)
    assert np.array_equal(cols[:, 0], [5.0, 5.0, 5.0])
    assert cols[0, 1] == 6.0 and cols[2, 1] == 6.0 and np.isnan(cols[1, 1])
    # and the zonal mean sees only wet values: the 99 and 77 never leak in
    _, zm = _zonal_mean_T(cols, np.zeros(3), np.ones(3), n_bins=2, reduce=np.nanmean)
    assert np.allclose(zm[1], [5.0, 6.0])
    with pytest.raises(ValueError, match="rows <"):
        fesom_wet_columns(temp, np.array([3, 2, 3]), n_layers=4)


def test_plot_panel_colour_limits_ignore_land():
    lon = np.array([0.0, 1.0, 0.0, 1.0, 2.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    ocean = np.array([True, True, True, True, False])
    field = np.array([1.0, 2.0, 3.0, 4.0, 1e9])            # land value must not set the limits
    tri = build_ocean_triangulation(lon, lat, ocean)
    fig, ax = matplotlib.pyplot.subplots()
    tc = plot_panel(ax, tri, field, ocean, "t", "viridis", has_cartopy=False)
    lo, hi = tc.get_clim()
    assert hi < 5.0 and lo >= 1.0
    tc2 = plot_panel(ax, tri, field - 2.5, ocean, "t", "RdBu_r", has_cartopy=False, symmetric=True)
    lo2, hi2 = tc2.get_clim()
    assert lo2 == -hi2 and hi2 < 5.0
    matplotlib.pyplot.close(fig)


def test_zonal_mean_reduce_nanmean_ignores_below_bottom():
    lat = np.deg2rad(np.array([10.0, 12.0]))
    t = np.array([[20.0, 10.0], [22.0, np.nan]])           # 2nd column has 1 wet layer
    _, zm_mean = _zonal_mean_T(t, lat, np.ones(2), n_bins=18)
    _, zm_nan = _zonal_mean_T(t, lat, np.ones(2), n_bins=18, reduce=np.nanmean)
    b = 10  # bin [10, 20)
    assert np.isnan(zm_mean[b, 1]) and zm_nan[b, 1] == 10.0
    assert zm_nan[b, 0] == 21.0


def test_plot_fesom_store_daily_and_monthly(tmp_path):
    pytest.importorskip("zarr")
    # tiny 4-node, 2-element mesh
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    np.save(mesh / "nlevels_nod2D.npy", np.array([3, 3, 2, 3]))
    np.save(mesh / "Z.npy", np.array([-5.0, -15.0]))
    lon = np.array([0.0, 1.0, 0.0, 1.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])

    def store(name, fields):
        g = _open_v2_group(tmp_path / name)
        g.attrs.update({"calendar_month": "1958-01", "n_samples": 3})
        _put(g, "lon", lon)
        _put(g, "lat", lat)
        for k, v in fields.items():
            _put(g, k, np.asarray(v, dtype=np.float32))
        return tmp_path / name

    two = np.ones((1, 4))
    # nz = 3 rows in the store = 2 layers (Z.npy) + 1 padding row, as fesom_jax writes it
    temp = np.ones((1, 3, 4)) * 5
    temp[0, 2, :] = 0.0                                     # padding row must never be plotted
    monthly = store("1958_01", {"ssh": two, "a_ice": two * 0, "temp": temp,
                                "salt": np.ones((1, 3, 4)) * 35, "u": np.full((1, 3, 4), 0.1),
                                "v": np.zeros((1, 3, 4))})
    daily = store("day_1958_001", {"ssh": two, "a_ice": two, "sst": two, "sss": two, "usurf": two,
                                   "vsurf": two, "temp100": two})
    for st in (monthly, daily):
        out = plot_fesom_store(st, mesh, tmp_path / "snap")
        assert out.exists() and out.stat().st_size > 0


def test_plot_fesom_store_rejects_mismatched_mesh(tmp_path):
    pytest.importorskip("zarr")
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    np.save(mesh / "nlevels_nod2D.npy", np.array([3, 3, 3]))                  # 3 nodes
    g = _open_v2_group(tmp_path / "day_1958_001")
    _put(g, "lon", np.zeros(4))
    _put(g, "lat", np.zeros(4))                                 # 4 nodes
    with pytest.raises(ValueError, match="nodes, store has"):
        plot_fesom_store(tmp_path / "day_1958_001", mesh, tmp_path)
