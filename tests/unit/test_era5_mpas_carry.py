"""Mechanical validation of era5_to_mpas_carry on a synthetic ERA5 slice.

The real-data reference path for the MPAS column comparison (iter 75): a
fabricated ERA5Slice is inverse-distance regridded to a level-2 MPAS mesh's cell
centres + vertically interpolated to sigma.  Asserts shape, finiteness, physical
bounds, the specific→mixing-ratio conversion, a constant-field exactness check,
an equator-pole value structure, and the dispatcher wiring.  (Full physical
validation against real ERA5 is deferred to when a zarr path is provided.)
"""

import ast
import pathlib

import numpy as np
import pytest
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.training.era5_to_state import ERA5Slice, era5_to_mpas_carry


def _synthetic_era5(n_lat=73, n_lon=144, n_plev=8):
    lat = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, n_lat)
    lon = np.linspace(0.0, 2 * np.pi * (1 - 1.0 / n_lon), n_lon)
    plev = np.array([1000, 2000, 5000, 10000, 25000, 50000, 85000, 100000.0])
    latg = lat[:, None, None]
    z = (1.0 - plev / 1.0e5) * 12000.0
    T_sfc = 300.0 - 40.0 * np.sin(latg) ** 2
    T = np.broadcast_to(T_sfc - 6.5e-3 * z[None, None, :],
                        (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    u = np.broadcast_to((30.0 * np.cos(latg) * np.sin(2 * latg)),
                        (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    v = np.zeros((n_lat, n_lon, n_plev), np.float32)
    q = np.broadcast_to((0.018 * np.cos(latg) ** 2),
                        (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    p_s = np.full((n_lat, n_lon), 1.0e5, np.float32)
    sst = (300.0 - 40.0 * np.sin(lat[:, None]) ** 2) * np.ones((n_lat, n_lon))
    phis = np.zeros((n_lat, n_lon), np.float32)
    return ERA5Slice(T=T, u=u, v=v, q=q, p_s=p_s, sst=sst.astype(np.float32),
                     phis=phis, lat=lat, lon=lon, plev_Pa=plev)


def _carry_field(carry, name):
    f = getattr(carry, name)
    return np.asarray(f.data if hasattr(f, "data") else f)


def test_mpas_carry_shapes_and_physical():
    # #948: the dead cell-centred SegmentCarry builder was deleted; the sole
    # surviving era5_to_mpas_carry is the edge-normal dycore-IC builder, so u is
    # the edge-normal component on mesh EDGES (nEdges), not cell-centred.
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(30)
    carry = era5_to_mpas_carry(_synthetic_era5(), mesh, sigma)
    T = _carry_field(carry, "T")
    u = _carry_field(carry, "u")
    assert T.shape == (mesh.nCells, 30)          # scalars at cell centres
    assert u.shape == (mesh.nEdges, 30)          # edge-normal wind on edges
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(u))  # incl. dateline cells
    assert 180.0 < T.min() and T.max() < 320.0
    assert np.abs(u).max() > 5.0, "zonal jet did not survive the regrid"
    qv = _carry_field(carry, "q_v")
    assert np.all(qv >= 0.0) and qv.max() < 0.05, "mixing ratio unphysical"


def test_mpas_carry_constant_field_is_exact():
    """IDW of a CONSTANT ERA5 field is exact ⇒ every cell gets that constant
    (pins the regrid weights normalise to 1 + the cell mapping is sane)."""
    base = _synthetic_era5()
    era5 = base._replace(T=np.full_like(base.T, 263.0),
                         q=np.full_like(base.q, 0.0))
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_mpas_carry(era5, mesh, sigma)
    T = _carry_field(carry, "T")
    np.testing.assert_allclose(T, 263.0, atol=1e-2)


def test_mpas_carry_specific_to_mixing_ratio():
    base = _synthetic_era5()
    era5 = base._replace(q=np.full_like(base.q, 0.02))
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_mpas_carry(era5, mesh, sigma)
    qv = _carry_field(carry, "q_v")
    assert 0.0200 < float(qv[..., -1].mean()) < 0.0210   # r = q/(1-q) ≈ 0.0204


def test_mpas_carry_equator_pole_structure():
    """A latitude-only ERA5 T must interpolate to the cells with the right
    equator-pole structure (value-based, not just bounds)."""
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(30)
    carry = era5_to_mpas_carry(_synthetic_era5(), mesh, sigma)
    T = _carry_field(carry, "T")
    lat_cell = np.asarray(mesh.latCell)
    i_eq = int(np.argmin(np.abs(lat_cell)))
    i_pole = int(np.argmax(np.abs(lat_cell)))
    assert float(T[i_eq, -1]) > float(T[i_pole, -1]) + 15.0
    assert 285.0 < float(T[i_eq, -1]) < 305.0


def test_select_era5_regrid_routes_mpas_and_aliases():
    from scripts.validate.compare_amip_era5 import (
        canonical_grid_type,
        select_era5_regrid,
    )
    assert select_era5_regrid("mpas") is era5_to_mpas_carry
    assert select_era5_regrid("voronoi") is era5_to_mpas_carry   # alias
    assert canonical_grid_type("icosahedral") == "mpas"
    with pytest.raises(ValueError, match="Unknown grid_type"):
        select_era5_regrid("octahedral")                         # still raises


def test_mpas_carry_dtype_is_floating():
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_mpas_carry(_synthetic_era5(), mesh, sigma)
    # #948: MPASCarry fields are (u [edge-normal], T, p_s, phis, q_v) — no
    # cell-centred v (that belonged to the deleted SegmentCarry builder).
    for name in ("T", "u", "p_s", "phis", "q_v"):
        arr = _carry_field(carry, name)
        assert np.issubdtype(arr.dtype, np.floating), name   # never int/object


def _flip_lat(s):
    return s._replace(
        T=s.T[::-1].copy(), u=s.u[::-1].copy(), v=s.v[::-1].copy(),
        q=s.q[::-1].copy(), p_s=s.p_s[::-1].copy(), sst=s.sst[::-1].copy(),
        phis=s.phis[::-1].copy(), lat=s.lat[::-1].copy())


def test_mpas_carry_descending_lat_matches_ascending():
    """Native ERA5 is N→S (descending) latitude. Flipping lat + the field rows
    consistently is the SAME physical field, so the cell result is identical (the
    KD-tree finds the same physical neighbours; the cache key separates the two)."""
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(20)
    asc = _synthetic_era5()
    desc = _flip_lat(asc)
    T_asc = _carry_field(era5_to_mpas_carry(asc, mesh, sigma), "T")
    T_desc = _carry_field(era5_to_mpas_carry(desc, mesh, sigma), "T")
    np.testing.assert_allclose(T_asc, T_desc, atol=1e-2)


def test_mpas_carry_longitude_gradient_no_dateline_scramble():
    """A LONGITUDE-varying ERA5 field (T = 280 + 20·cos λ) must land on each cell
    at its OWN longitude — a dateline wrap-scramble (λ→λ+π) would negate the cosine
    (off by up to 40 K), far outside tolerance."""
    base = _synthetic_era5(n_lat=73, n_lon=144)
    lon = base.lon
    t_lon = (280.0 + 20.0 * np.cos(lon))[None, :, None]        # (1, n_lon, 1)
    T = np.broadcast_to(t_lon, base.T.shape).astype(np.float32).copy()
    era5 = base._replace(T=T)
    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(20)
    T_cell = _carry_field(era5_to_mpas_carry(era5, mesh, sigma), "T")
    lon_cell = np.asarray(mesh.lonCell)
    expected = 280.0 + 20.0 * np.cos(lon_cell)
    # Modest tol for IDW smoothing on the coarse mesh; a wrap would be ~40 K off.
    np.testing.assert_allclose(T_cell[:, -1], expected, atol=5.0)


def test_voronoi_weights_cache_separates_ascending_descending():
    from legoesm.training.era5_to_state import _get_voronoi_weights
    mesh = create_voronoi_mesh(2)
    asc = _synthetic_era5()
    w_asc = _get_voronoi_weights(asc.lat, asc.lon, mesh)
    w_desc = _get_voronoi_weights(asc.lat[::-1].copy(), asc.lon, mesh)
    assert w_asc is not w_desc        # same shape, different lat bounds → no reuse


def test_voronoi_weights_cache_hits_on_identical_content():
    from legoesm.training.era5_to_state import _get_voronoi_weights
    mesh = create_voronoi_mesh(2)
    asc = _synthetic_era5()
    # A freshly-built but value-identical source grid still hits the cache
    # (content fingerprint, not object identity).
    w1 = _get_voronoi_weights(asc.lat.copy(), asc.lon.copy(), mesh)
    w2 = _get_voronoi_weights(asc.lat.copy(), asc.lon.copy(), mesh)
    assert w1 is w2


def test_voronoi_weights_cache_separates_same_endpoint_different_interior():
    """The bug class fixed in iter 109's cubed-sphere path, here for MPAS: a source
    grid with the SAME shape + endpoints but different INTERIOR spacing (e.g. uniform
    lat-lon vs Gaussian of the same bounds) must NOT collide on the cache."""
    from legoesm.training.era5_to_state import _get_voronoi_weights
    mesh = create_voronoi_mesh(2)
    asc = _synthetic_era5()
    w_uniform = _get_voronoi_weights(asc.lat, asc.lon, mesh)
    lat_perturbed = asc.lat.copy()
    lat_perturbed[1:-1] *= 0.5            # change interior, keep endpoints
    w_perturbed = _get_voronoi_weights(lat_perturbed, asc.lon, mesh)
    assert w_uniform is not w_perturbed


def test_voronoi_weights_cache_separates_different_meshes():
    """Different MESH cell coordinates → different cached weights (the mesh is
    fingerprinted by content, so two distinct meshes never alias even if a GC'd
    mesh's id were reused)."""
    from legoesm.training.era5_to_state import _get_voronoi_weights
    asc = _synthetic_era5()
    mesh2 = create_voronoi_mesh(2)
    mesh3 = create_voronoi_mesh(3)        # different nCells / cell coords
    w2 = _get_voronoi_weights(asc.lat, asc.lon, mesh2)
    w3 = _get_voronoi_weights(asc.lat, asc.lon, mesh3)
    assert w2 is not w3


def test_era5_to_state_has_no_duplicate_toplevel_defs():
    """#948 guard: era5_to_state.py defined ``era5_to_mpas_carry`` TWICE (the
    cell-centred SegmentCarry builder silently shadowed by the edge-normal
    dycore-IC builder — #565/#797 merge residue).  Any future duplicate
    top-level def/class name in the module now fails here LOUDLY instead of
    binding the name to the last definition and shadowing the rest."""
    import legoesm.training.era5_to_state as mod
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    names = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef))]
    dups = sorted({n for n in names if names.count(n) > 1})
    assert not dups, f"duplicate top-level defs in era5_to_state.py: {dups}"
