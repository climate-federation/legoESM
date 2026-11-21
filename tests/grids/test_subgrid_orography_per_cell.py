"""Per-cell subgrid orography over each Voronoi cell's footprint.

* Against a brute-force reference (nearest-generator owner, spherical-area
  weights, weighted std of max(z, 0) per cell) on a synthetic 0.5 degree field.
* Chunking does not change the result.
* A field constant inside every cell has sgh 0 there.
* The per-cell file is refused on another mesh or with the wrong length; the
  builder script writes what the loader reads.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from legoesm.grids.topography import (  # noqa: E402
    _source_cell_areas, load_subgrid_orography_cells, subgrid_orography_per_cell)
from legoesm.grids.voronoi import create_voronoi_mesh  # noqa: E402


def _source():
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(0.25, 360.0, 0.5)
    lo, la = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    z = 3000.0 * np.sin(3 * lo) * np.cos(la) ** 2 + 800.0 * np.cos(7 * la) - 200.0
    return lat, lon, z


def test_matches_brute_force_and_chunking():
    from scipy.spatial import cKDTree
    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    lat, lon, z = _source()
    clat, clon = np.asarray(mesh.latCell, float), np.asarray(mesh.lonCell, float)
    sgh, mean, cnt = subgrid_orography_per_cell(
        clat, clon, lat, lon, lambda a, b: z[a:b], rows_per_chunk=37)
    sgh2, _, _ = subgrid_orography_per_cell(
        clat, clon, lat, lon, lambda a, b: z[a:b], rows_per_chunk=360)
    np.testing.assert_allclose(sgh, sgh2, rtol=1e-10, atol=1e-8)
    # brute force
    lo, la = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1).reshape(-1, 3)
    cx = np.stack([np.cos(clat) * np.cos(clon), np.cos(clat) * np.sin(clon), np.sin(clat)], -1)
    own = cKDTree(cx).query(xyz)[1]
    w = _source_cell_areas(lat, lon).ravel()
    h = np.maximum(z, 0.0).ravel()
    for c in range(0, mesh.nCells, 17):
        m = own == c
        mu = np.sum(w[m] * h[m]) / np.sum(w[m])
        sd = np.sqrt(np.sum(w[m] * (h[m] - mu) ** 2) / np.sum(w[m]))
        assert cnt[c] == m.sum()
        np.testing.assert_allclose(mean[c], mu, rtol=1e-10)
        np.testing.assert_allclose(sgh[c], sd, rtol=1e-7, atol=1e-6)


def test_constant_inside_cells_gives_zero():
    from scipy.spatial import cKDTree
    mesh = create_voronoi_mesh(1, lloyd_iterations=0)
    lat, lon, _ = _source()
    clat, clon = np.asarray(mesh.latCell, float), np.asarray(mesh.lonCell, float)
    lo, la = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1).reshape(-1, 3)
    cx = np.stack([np.cos(clat) * np.cos(clon), np.cos(clat) * np.sin(clon), np.sin(clat)], -1)
    own = cKDTree(cx).query(xyz)[1].reshape(la.shape)
    z = 100.0 * own.astype(float)
    sgh, mean, _ = subgrid_orography_per_cell(clat, clon, lat, lon, lambda a, b: z[a:b])
    # one-pass variance in float64: cancellation of E[h^2] - mean^2 leaves
    # about 1e-3 m at 4,100 m mean (documented in the function)
    assert np.max(sgh) < 1e-2
    np.testing.assert_allclose(mean, 100.0 * np.arange(mesh.nCells), atol=1e-8)


def test_nonfinite_source_refused():
    mesh = create_voronoi_mesh(1, lloyd_iterations=0)
    lat, lon, z = _source()
    z = z.copy()
    z[10, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        subgrid_orography_per_cell(np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
                                   lat, lon, lambda a, b: z[a:b])


def test_builder_and_loader(tmp_path):
    import xarray as xr
    from scripts.data.build_subgrid_orography_cells import build
    from legoesm.grids.voronoi_mesh_file import save_voronoi_mesh
    mesh = create_voronoi_mesh(1, lloyd_iterations=0)
    mf = tmp_path / "m.npz"
    sha = save_voronoi_mesh(mesh, mf, resolution=1, lloyd_iterations=0)
    lat, lon, z = _source()
    # descending latitude, as ETOPO's GeoTransform suggests some files are
    xr.Dataset({"z": (("lat", "lon"), z[::-1].astype(np.float32))},
               coords={"lat": lat[::-1], "lon": lon}).to_netcdf(tmp_path / "s.nc")
    out = tmp_path / "sgh.npz"
    sgh, _, _ = build(str(mf), sha, str(tmp_path / "s.nc"), str(out), rows_per_chunk=50)
    ref, _, _ = subgrid_orography_per_cell(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell), lat, lon,
        lambda a, b: z.astype(np.float32)[a:b])
    np.testing.assert_allclose(sgh, ref, rtol=1e-10)
    got = load_subgrid_orography_cells(out, mesh.nCells, sha)
    np.testing.assert_array_equal(got, sgh)
    # an MPI rank's owned-then-halo cells, in the partition's order
    from legoesm.parallel.voronoi_partition import partition_voronoi_mesh
    part = partition_voronoi_mesh(mesh, 2, 1, halo_depth=3)
    got_r = load_subgrid_orography_cells(out, mesh.nCells, sha,
                                         cells=part.local_cells)
    np.testing.assert_array_equal(got_r, sgh[np.asarray(part.local_cells)])
    assert got_r.shape[0] < mesh.nCells
    with pytest.raises(ValueError, match="built for mesh"):
        load_subgrid_orography_cells(out, mesh.nCells, "f" * 64)
    with pytest.raises(ValueError, match="shape"):
        load_subgrid_orography_cells(out, mesh.nCells + 1, sha)
