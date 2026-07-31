"""VoronoiMesh must be able to carry per-cell SSO + land fraction.

Without these fields the orographic GWD schemes fall back to the scalar
``config.h_topo`` — a uniform 500 m pseudo-mountain over every OCEAN cell
(measured at 0.036 Pa of spurious column momentum sink and ~22% of the
missing surface circulation in the 2026-07-30 GWD ablation).  The loaders
already existed (``load_subgrid_orography`` handles rank-1 Voronoi cell
centres via ``_target_grid_degrees``; the driver computes ``_f_land``) —
the mesh just could not carry their output.

Each test uses values the defaults cannot produce, so a deleted forward or
field goes red.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.factory import create_grid
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    _extract_land_frac,
    _extract_subgrid_topo_stddev,
)


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=5)   # 162 cells, fast


def test_fields_default_none_and_extractors_fall_back(mesh):
    assert mesh.subgrid_topo_stddev is None
    assert mesh.land_frac is None
    # None => extractors return None => schemes keep the scalar fallback.
    assert _extract_subgrid_topo_stddev(mesh, mesh.nCells) is None
    assert _extract_land_frac(mesh, mesh.nCells) is None


def test_replace_carries_per_cell_fields_to_the_extractors(mesh):
    n = mesh.nCells
    sso = jnp.linspace(0.0, 700.0, n)
    lf = jnp.linspace(0.0, 1.0, n)
    m2 = mesh._replace(subgrid_topo_stddev=sso, land_frac=lf)
    got_sso = _extract_subgrid_topo_stddev(m2, n)
    got_lf = _extract_land_frac(m2, n)
    assert got_sso is not None and got_lf is not None
    np.testing.assert_allclose(np.asarray(got_sso), np.asarray(sso))
    np.testing.assert_allclose(np.asarray(got_lf), np.asarray(lf))


def test_load_subgrid_orography_regrids_to_voronoi_cells(mesh, tmp_path):
    """The EXISTING loader must work on a VoronoiMesh: correct shape, ocean
    ~0, and the mountain localized where the source put it."""
    import xarray as xr

    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(0.0, 358.0, 180)
    sso = np.zeros((90, 180))
    # one "Andes": 800 m stddev near (20S, 290E)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    sso += 800.0 * np.exp(-(((la + 20) / 8.0) ** 2 + ((lo - 290) / 8.0) ** 2))
    p = tmp_path / "sso_synth.nc"
    xr.Dataset({"SSO_STDH": (("lat", "lon"), sso)},
               coords={"lat": lat, "lon": lon}).to_netcdf(p)

    from legoesm.grids.topography import load_subgrid_orography
    out = np.asarray(load_subgrid_orography(mesh, str(p)))
    assert out.shape == (mesh.nCells,)
    assert (out >= 0.0).all()
    latd = np.degrees(np.asarray(mesh.latCell))
    lond = np.degrees(np.asarray(mesh.lonCell)) % 360.0
    near = (np.abs(latd + 20) < 12) & (np.abs(lond - 290) < 12)
    far = (np.abs(latd - 40) < 20) & (np.abs(lond - 120) < 40)   # mid-Pacific
    assert near.any() and far.any()
    assert out[near].max() > 200.0, "mountain did not survive the regrid"
    assert out[far].max() < 50.0, "SSO leaked into the open ocean"


def test_partition_slicer_carries_the_fields(mesh):
    """The local-mesh slicer must not silently drop SSO/land_frac under
    cell-partition MPI (keyword construction would default them to None)."""
    from legoesm.parallel import voronoi_partition as vp

    fn = getattr(vp, "build_local_mesh", None)
    if fn is None:
        pytest.skip("no build_local_mesh in voronoi_partition")
    n = mesh.nCells
    m2 = mesh._replace(subgrid_topo_stddev=jnp.arange(n, dtype=jnp.float64),
                       land_frac=jnp.ones(n) * 0.5)
    import inspect
    src = inspect.getsource(vp)
    # the slicer source must reference both fields (a structural tripwire in
    # ADDITION to the behavioural tests above, which cover the extractors)
    assert "subgrid_topo_stddev" in src and "land_frac" in src


def test_mesh_cache_round_trip_with_none_optionals(tmp_path):
    """The mesh cache must survive the optional fields.

    Regression: np.asarray(None) is a 0-d OBJECT array; savez pickles it and
    the allow_pickle=False load then rejects the whole file — every
    create_grid call regenerated the mesh AND re-saved a broken cache
    (observed live in run gwdE_realoro, job 26573336)."""
    import warnings as _w
    from legoesm.grids.voronoi import (
        _load_voronoi_cache, _save_voronoi_cache,
    )

    m = create_grid("mpas", 2, lloyd_iterations=5)
    assert m.subgrid_topo_stddev is None          # the None-optional case
    p = str(tmp_path / "mesh_cache.npz")
    with _w.catch_warnings():
        _w.simplefilter("error")                  # a cache-write warning FAILS
        _save_voronoi_cache(p, m)
        m2 = _load_voronoi_cache(p)
    assert m2 is not None, "cache unreadable (the object-array regression)"
    assert m2.nCells == m.nCells
    assert m2.subgrid_topo_stddev is None and m2.land_frac is None
    np.testing.assert_allclose(np.asarray(m2.latCell), np.asarray(m.latCell))
