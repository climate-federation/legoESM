"""Tests for the source->target nearest-centre locator and the layer-overlap helper."""
import numpy as np

from legoesm.ocean.forcing.curvilinear_regrid import layer_overlap_thickness, nearest_centre_index


def test_nearest_centre_assigns_each_source_cell_and_wraps_the_dateline():
    tgt_lon, tgt_lat = np.meshgrid([0.0, 90.0, 180.0, 270.0], [-30.0, 30.0])
    src_lon = np.array([[1.0, 359.0, 179.0], [91.0, 268.0, 181.0]])
    src_lat = np.array([[-29.0, 29.0, -31.0], [31.0, 29.0, 28.0]])
    idx = nearest_centre_index(src_lon, src_lat, tgt_lon, tgt_lat)
    assert idx.shape == src_lat.shape
    # flat target index = row * 4 + col
    np.testing.assert_array_equal(idx, [[0, 4, 2], [5, 7, 6]])


def test_layer_overlap_conserves_column_content():
    src_edges = np.array([0.0, 10.0, 25.0, 50.0, 80.0])
    dst_edges = np.array([0.0, 20.0, 60.0])           # stops above the source bottom
    ov = layer_overlap_thickness(src_edges[:-1], src_edges[1:], dst_edges[:-1], dst_edges[1:])
    np.testing.assert_allclose(ov, [[10, 0], [10, 5], [0, 25], [0, 10]])
    np.testing.assert_allclose(ov.sum(axis=0), np.diff(dst_edges))
    T = np.array([20.0, 15.0, 10.0, 5.0])
    remapped = (ov * T[:, None]).sum(0) / ov.sum(0)
    # heat in the shared 0-60 m is identical on both grids
    np.testing.assert_allclose((remapped * np.diff(dst_edges)).sum(), (ov.sum(1) * T).sum())


def test_layer_overlap_broadcasts_over_columns():
    top = np.zeros((3, 2)); bot = np.full((3, 2), 5.0); bot[:, 1] = 10.0; top[:, 1] = 5.0
    ov = layer_overlap_thickness(top, bot, np.array([[0.0]] * 3), np.array([[8.0]] * 3))
    assert ov.shape == (3, 2, 1)
    np.testing.assert_allclose(ov[:, :, 0], [[5.0, 3.0]] * 3)
