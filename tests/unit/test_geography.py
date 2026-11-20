"""Generic geography file (legoesm.grids.geography) + its generator
(scripts/data/make_geography.py) on the MPAS level-3 mesh."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from legoesm.grids import geography as geo
from legoesm.grids.topography import (
    TopographyConfig,
    bin_latlon_to_cells,
    grid_ownership,
    load_real_topography,
)

_GEN = Path(__file__).resolve().parents[2] / "scripts" / "data" / "make_geography.py"
_spec = importlib.util.spec_from_file_location("make_geography", _GEN)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(3)


def _write(ds, tmp_path, name):
    p = str(tmp_path / name)
    ds.to_netcdf(p)
    return p


def _gen(tmp_path, shape, **kw):
    return _write(mg.idealized(shape, **kw), tmp_path, f"{shape}.nc")


def _terrain(mesh, path, mode="binned", width=1):
    phis, f = geo.geography_terrain(mesh, path, mode, smoothing_passes=4,
                                    barrier_width_cells=width)
    return np.asarray(phis).ravel(), np.asarray(f).ravel()


def _ridge_lines():
    return [mg._meridian(0.0, -90.0, 90.0)]



def test_present_day_legacy_is_bitwise_the_topography_plus_mask_path(mesh, tmp_path):
    import xarray as xr
    rng = np.random.default_rng(0)
    # elevation: lon -180..180, lat DESCENDING (both reordered by the loader)
    lat_e = np.arange(89.5, -90.0, -1.0)
    lon_e = np.arange(-179.5, 180.0, 1.0)
    z = rng.normal(0.0, 2000.0, (lat_e.size, lon_e.size))
    e = xr.Dataset({"z": (("latitude", "longitude"), z, {"units": "m"})},
                   coords={"latitude": lat_e, "longitude": lon_e})
    # sftlf in PERCENT on its own 2-degree grid with NaN cells
    lat_m, lon_m = np.arange(-89.0, 90.0, 2.0), np.arange(1.0, 360.0, 2.0)
    m = rng.uniform(0.0, 100.0, (lat_m.size, lon_m.size)).astype(np.float32)
    m[::7, ::5] = np.nan
    s = xr.Dataset({"sftlf": (("lat", "lon"), m)}, coords={"lat": lat_m, "lon": lon_m})
    pe, pm = _write(e, tmp_path, "etopo.nc"), _write(s, tmp_path, "sftlf.nc")
    ref = load_real_topography(mesh, config=TopographyConfig(
        source="file", path=pe, smoothing_passes=4, land_mask_path=pm))
    pg = _write(mg.present_day(pe, pm), tmp_path, "present.nc")
    got = geo.geography_terrain(mesh, pg, "legacy", smoothing_passes=4)
    assert np.array_equal(np.asarray(got[0]), np.asarray(ref[0]))
    assert np.array_equal(np.asarray(got[1]), np.asarray(ref[1]))
    assert 0.0 < float(np.mean(np.asarray(ref[1]))) < 1.0   # a real coastline


def test_aquaplanet_has_no_land_and_no_terrain(mesh, tmp_path):
    phis, f = _terrain(mesh, _gen(tmp_path, "aquaplanet"))
    assert np.all(f == 0.0) and np.all(phis == 0.0)


def test_ridge_is_closed_at_elevation_zero_and_unbroken(mesh, tmp_path):
    phis, f = _terrain(mesh, _gen(tmp_path, "ridge", barrier_lon=0.0))
    closed = f == 1.0
    assert closed.sum() >= 18                       # pole to pole at ~8 deg cells
    assert set(np.unique(f)) == {0.0, 1.0}
    assert np.all(phis == 0.0)                      # land does not protrude
    owners = geo.barrier_owners(mesh, _ridge_lines())
    assert geo.barrier_gaps(mesh, owners) == 0
    assert np.array_equal(closed, geo.closed_barrier_cells(mesh, owners, 1))


def test_barrier_cells_keep_elevation_zero_next_to_mountains(mesh, tmp_path):
    ds = mg.idealized("ridge", barrier_lon=0.0)
    lon = ds["lon"].values
    near = (np.abs(((lon + 180.0) % 360.0) - 180.0) < 20.0)[None, :]
    ds["elevation"] = ds["elevation"].where(~near, 3000.0)   # mountains around it
    phis, f = _terrain(mesh, _write(ds, tmp_path, "ridge_mtn.nc"))
    closed = geo.closed_barrier_cells(mesh, geo.barrier_owners(mesh, _ridge_lines()), 1)
    assert np.all(phis[closed] == 0.0) and np.all(f[closed] == 1.0)
    assert phis[~closed].max() > 0.0


def _closed_lats(mesh, f):
    return np.degrees(np.asarray(mesh.grid_lat)).ravel()[f == 1.0]


def test_drake_is_open_only_south_of_40s(mesh, tmp_path):
    _, f = _terrain(mesh, _gen(tmp_path, "drake", barrier_lon=0.0))
    lats = _closed_lats(mesh, f)
    assert lats.min() > -50.0 and lats.max() > 80.0


def test_eqpas_gap_is_open(mesh, tmp_path):
    _, f = _terrain(mesh, _gen(tmp_path, "eqpas", barrier_lon=0.0))
    lats = _closed_lats(mesh, f)
    assert not np.any(np.abs(lats) < 12.0)
    assert lats.min() < -80.0 and lats.max() > 80.0


@pytest.mark.parametrize("shape,lines", [
    ("ridge", [(-90.0, 90.0)]), ("drake", [(-40.0, 90.0)]),
    ("eqpas", [(-90.0, -20.0), (20.0, 90.0)])])
def test_every_face_the_barrier_crosses_has_a_closed_cell(mesh, tmp_path, monkeypatch,
                                                          shape, lines):
    """Watertightness measured: trace the barrier ~1000x finer than the check
    does; each change of owner is a face crossing, and one of its two cells
    must be closed."""
    _, f = _terrain(mesh, _gen(tmp_path, shape, barrier_lon=0.0))
    polys = [mg._meridian(0.0, a, b) for a, b in lines]
    monkeypatch.setattr(geo, "_DENSIFY_MAX_DEG", 1e-4)
    monkeypatch.setattr(geo, "_DENSIFY_CELL_FRACTION", 1e-4)
    for o in geo.barrier_owners(mesh, polys):
        a, b = o[:-1], o[1:]
        cross = a != b
        assert np.all((f[a[cross]] == 1.0) | (f[b[cross]] == 1.0))
        assert geo.barrier_gaps(mesh, [o]) == 0


def test_a_barrier_that_jumps_cells_is_refused(mesh, tmp_path, monkeypatch):
    ds = mg.idealized("ridge", barrier_lon=0.0).isel(vertex=[0, 90, 180])  # -90, 0, 90
    p = _write(ds, tmp_path, "ridge3.nc")
    _terrain(mesh, p)                                # dense points: accepted
    monkeypatch.setattr(geo, "_DENSIFY_MAX_DEG", 30.0)
    monkeypatch.setattr(geo, "_DENSIFY_CELL_FRACTION", 10.0)
    with pytest.raises(ValueError, match="non-neighbour cells"):
        _terrain(mesh, p)


def test_width_is_total_cells_and_odd(mesh):
    owners = geo.barrier_owners(mesh, _ridge_lines())
    one = geo.closed_barrier_cells(mesh, owners, 1)
    three = geo.closed_barrier_cells(mesh, owners, 3)
    assert np.all(three[one]) and three.sum() > 2 * one.sum()
    with pytest.raises(ValueError, match="odd"):
        geo.closed_barrier_cells(mesh, owners, 2)


def test_binned_refuses_descending_longitude(mesh, tmp_path):
    ds = mg.idealized("box", box=(10.0, 50.0, 20.0, 80.0)).isel(lon=slice(None, None, -1))
    with pytest.raises(ValueError, match="ascending"):
        _terrain(mesh, _write(ds, tmp_path, "box_desc.nc"), "binned")


def test_binned_box_land_fraction_is_the_area_binned_source(mesh, tmp_path):
    ds = mg.idealized("box", box=(10.0, 50.0, 20.0, 80.0))
    _, f = _terrain(mesh, _write(ds, tmp_path, "box.nc"), "binned")
    want = bin_latlon_to_cells(ds["lat"].values, ds["lon"].values,
                               [ds["land_fraction"].values],
                               np.asarray(mesh.grid_lat), np.asarray(mesh.grid_lon),
                               **grid_ownership(mesh))[0]
    assert np.array_equal(f, want) and f.max() == 1.0


def test_bare_soil_where_the_file_has_no_land_map(tmp_path):
    from legoesm.land.clm_surface_map import load_clm_surface
    src = geo.geography_clm_surface(_gen(tmp_path, "aquaplanet"))
    m = load_clm_surface(src, np.array([0.0, 45.0]), np.array([10.0, 200.0]),
                         vg_air_entry_m=0.0)
    assert np.all(np.asarray(m["pft_fractions"])[:, 0] == 1.0)
    assert np.all(np.asarray(m["pct_sand"]) == 43.0)     # VOTED V6: USDA loam centroid
    assert np.all(np.asarray(m["pct_clay"]) == 18.0)
    from legoesm.land.clm_surface_map import soil_color_albedo
    assert np.all(np.asarray(m["soil_albedo"]) == soil_color_albedo(np.array([15]))[0])


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d.attrs.pop("legoesm_geography_version"), "version"),
    (lambda d: d.__setitem__("ice_sheet_thickness", d["elevation"] * 0), "not supported"),
    (lambda d: d.__setitem__("land_fraction", d["elevation"] * 0 + 50.0), r"not %"),
    (lambda d: d["elevation"].attrs.update(units="km"), "units"),
])
def test_open_geography_refuses(tmp_path, edit, msg):
    ds = mg.idealized("aquaplanet")
    edit(ds)
    with pytest.raises(ValueError, match=msg):
        geo.open_geography(_write(ds, tmp_path, "bad.nc"))


def test_generator_cli_writes_a_valid_file(tmp_path):
    out = str(tmp_path / "d.nc")
    assert mg.main(["drake", "--barrier-lon", "30", "-o", out]) == 0
    ds = geo.open_geography(out)
    assert len(geo.barrier_polylines(ds)) == 1
    ds.close()
    with pytest.raises(ValueError, match="barrier-lon"):
        mg.idealized("ridge")


def _mpas_cfg(path, **kw):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0), days=1,
        dataset="analytical", radiation="gray",
        **{"topography": "flat", "geography": path, "geography_regrid": "binned", **kw})


def test_driver_terrain_is_the_geography_terrain(mesh, tmp_path):
    from legoesm.driver.model_driver import ModelDriver
    p = _gen(tmp_path, "ridge", barrier_lon=0.0)
    d = ModelDriver(_mpas_cfg(p), output_dir=str(tmp_path / "run"))
    got = np.asarray(d.static_land_fraction()).ravel()
    _, want = geo.geography_terrain(d.grid, p, "binned", smoothing_passes=4)
    assert np.array_equal(got, np.asarray(want).ravel())
    assert np.asarray(want).max() == 1.0


@pytest.mark.parametrize("shape,cc_kw", [
    ("ridge", dict(land_mode="slab")),
    ("aquaplanet", dict(land_mode="slab")),
    ("ridge", dict(land_mode="multilayer", land_param_source="clm")),
    ("ridge", dict(land_mode="slab", atm_owns_surface=True)),
])
def test_coupled_refuses_geography_with_its_own_land_tile(tmp_path, monkeypatch,
                                                          shape, cc_kw):
    """No coupler land tile on a geography (it would ignore the geography's
    land map, and on an all-ocean mask invent continents) -- refused before
    the coupler reads any reference land data."""
    import legoesm.land.clm_surface_map as csm
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    def no_reference_data(*a, **k):
        raise AssertionError("coupler read reference CLM data")

    monkeypatch.setattr(csm, "download_clm_surfdata", no_reference_data)
    cfg = _mpas_cfg(_gen(tmp_path, shape, barrier_lon=0.0 if shape == "ridge" else None))
    d = CoupledESMDriver(cfg, CoupledConfig(f_land_mode="analytical", **cc_kw),
                         output_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="coupler's own land tile"):
        d.setup()


def test_present_day_refuses_non_metre_elevation(tmp_path):
    import xarray as xr
    lat, lon = np.arange(-89.5, 90.0, 1.0), np.arange(0.5, 360.0, 1.0)
    z = xr.Dataset({"z": (("lat", "lon"), np.zeros((lat.size, lon.size)),
                          {"units": "m**2 s**-2"})}, coords={"lat": lat, "lon": lon})
    m = xr.Dataset({"sftlf": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
                   coords={"lat": lat, "lon": lon})
    with pytest.raises(ValueError, match="not metres"):
        mg.present_day(_write(z, tmp_path, "z.nc"), _write(m, tmp_path, "m.nc"))


def test_duo_column_lane_rebuilds_terrain_from_a_geography():
    from types import SimpleNamespace

    from legoesm.driver.model_driver import ModelDriver
    called = []
    stub = SimpleNamespace(config=_mpas_cfg("g.nc")._replace(ic="era5", ic_path=""),
                           _fv3_duo_column_terrain=lambda: called.append(1))
    with pytest.raises(ValueError, match="needs ic_path"):
        ModelDriver._fv3_duo_column_init_state(stub)
    assert called == [1]


def test_duo_column_factory_refuses_geography_without_era5_ic():
    from types import SimpleNamespace

    from legoesm.driver.component_factory import _create_fv3_duo_column_model
    with pytest.raises(ValueError, match="geography='g.nc'"):
        _create_fv3_duo_column_model(_mpas_cfg("g.nc"), SimpleNamespace(nlev=10),
                                     "hydrostatic")


@pytest.mark.parametrize("kw,msg", [
    (dict(geography_regrid=""), "geography_regrid"),
    (dict(topography="etopo.nc"), "replaces topography"),
    (dict(land_mask_path="m.nc"), "replaces land_mask_path"),
    (dict(surfdata_path="s.nc"), "replaces surfdata_path"),
    (dict(clm_surfdata_path="c.nc"), "replaces clm_surfdata_path"),
    (dict(geography_barrier_width_cells=2), "odd"),
])
def test_validate_refuses_geography_clashes(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _mpas_cfg("g.nc", **kw).validate_strict()


def test_validate_refuses_geography_knobs_without_geography():
    with pytest.raises(ValueError, match="need geography"):
        _mpas_cfg("g.nc")._replace(geography="").validate_strict()


def test_ocean_depth_is_the_binned_sea_depth_clipped(mesh, tmp_path):
    p = _gen(tmp_path, "box", box=(-30.0, 30.0, 60.0, 150.0))
    d = geo.geography_ocean_depth(mesh, p, H_min=10.0, H_max=5500.0)
    _, f = _terrain(mesh, p)
    def near(a, b):   # an area-weighted mean of a constant, to rounding
        return np.allclose(a, b, rtol=1e-12, atol=0.0)
    assert near(d[f == 0.0], 5200.0)              # all-sea cells: the file depth
    assert np.all(d[f == 1.0] == 10.0)            # no sea source point: H_min
    coast = (f > 0.0) & (f < 1.0)
    assert coast.any() and near(d[coast], 5200.0)  # mean over sea points only
    assert np.all(geo.geography_ocean_depth(mesh, p, H_min=10.0, H_max=4000.0)
                  == np.where(f == 1.0, 10.0, 4000.0))


def test_snap_partial_cells_is_the_omip_inline_recipe():
    """The shared snap is the inline OMIP code it replaced, bitwise."""
    import jax.numpy as jnp
    from legoesm.ocean.bathymetry import snap_partial_cells
    from legoesm.ocean.vertical import create_ocean_z_star
    z = create_ocean_z_star(20, H_max=5500.0)
    H = jnp.asarray(np.random.default_rng(0).uniform(-100.0, 5500.0, 4000))
    abs_z_half = jnp.abs(z.z_half_ref)
    n_above = jnp.sum(abs_z_half[None, :] < H[:, None], axis=1)
    bl = jnp.clip(n_above - 1, 0, z.n_levels - 1)
    frac = (H - abs_z_half[bl]) / jnp.maximum(z.dz_ref[bl], 1e-10)
    up, lo = abs_z_half[bl], abs_z_half[jnp.minimum(bl + 1, z.n_levels)]
    want = jnp.where((frac < 0.30) & (frac > 0) & (H > 0),
                     jnp.where(H - up < lo - H, up, lo), H)
    want = jnp.where(want <= 0, 0.0, want)
    got = snap_partial_cells(H, z)
    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert int(np.sum(np.asarray(got) != np.asarray(jnp.maximum(H, 0.0)))) > 100


def test_coupled_mpas_ocean_reads_the_geography(tmp_path):
    import jax.numpy as jnp
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.ocean.bathymetry import snap_partial_cells
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    p = _gen(tmp_path, "box", box=(-30.0, 30.0, 60.0, 150.0))
    d = CoupledESMDriver(_mpas_cfg(p), CoupledConfig(
        f_land_mode="analytical", land_mode="none", ocean_mode="dynamic",
        ocean_H_max_m=5500.0), output_dir=str(tmp_path / "run"))
    d.setup()
    f = np.asarray(d._atm._f_land).ravel()
    wet = np.asarray(d._ocean_land_mask)
    assert np.array_equal(wet, (f < 1.0).astype(wet.dtype)) and 0 < wet.sum() < wet.size
    H = np.asarray(d._ocean_state.H_bathy.data)
    want = np.asarray(snap_partial_cells(jnp.full(H.shape, 5200.0, H.dtype),
                                         d._ocean_z_coord))
    assert np.array_equal(H, np.where(wet > 0.5, want, 0.0))
    assert isinstance(d._ocean_z_coord, OceanPartialCellCoordinate)


def test_coupled_latlon_dynamic_ocean_refuses_a_geography(tmp_path):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=16, nlev=8),
        dycore=DycoreConfig(discretization="latlon_cgrid", dt=300.0),
        days=1, dataset="analytical", radiation="gray", topography="flat",
        geography=_gen(tmp_path, "aquaplanet"), geography_regrid="binned")
    d = CoupledESMDriver(cfg, CoupledConfig(f_land_mode="analytical", land_mode="none",
                                            ocean_mode="dynamic"),
                         output_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="lat-lon/tripole dynamic ocean"):
        d.setup()
