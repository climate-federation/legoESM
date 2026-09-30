"""Unit tests for :meth:`ModelDriver.static_topography_phis`.

The side-effect-free topography probe: it exposes the model's OWN static surface
geopotential ``phis = g·z_s`` (the consistent source for the orographic
LES-forcing term) WITHOUT running the full ``setup()`` — so a launch-time probe
writes no run manifest / output directory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402


def _cfg(topography: str) -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=3),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        topography=topography,
        radiation="gray",
        days=1,
    )


def test_static_topography_phis_flat_is_zeros_and_writes_nothing(tmp_path):
    """A flat model yields an all-zero phis of the grid 2-D shape — and the probe
    performs NO filesystem writes (the output dir / run manifest are NOT created),
    so a launch-time orographic probe leaves no phantom run directory."""
    out = tmp_path / "run"
    driver = ModelDriver(_cfg("flat"), output_dir=str(out))
    phis = driver.static_topography_phis()
    assert tuple(phis.shape) == tuple(driver.grid.grid_shape_2d)
    assert bool(jnp.all(phis == 0))            # flat → identically zero
    assert not out.exists()                    # NO mkdir / manifest / config writes


def test_static_topography_phis_gaussian_is_nonzero(tmp_path):
    """A gaussian-mountain model yields a non-zero phis (real terrain), so the
    caller's flat-detection keeps the orographic term ON."""
    driver = ModelDriver(_cfg("gaussian"), output_dir=str(tmp_path / "run"))
    phis = driver.static_topography_phis()
    assert bool(jnp.any(phis != 0))            # gaussian mountain → real terrain


def test_static_topography_phis_idempotent(tmp_path):
    """Repeated calls return the SAME cached field (the build runs once); a driver
    that already ran setup() would likewise return its built _phis_data."""
    driver = ModelDriver(_cfg("flat"), output_dir=str(tmp_path / "run"))
    first = driver.static_topography_phis()
    second = driver.static_topography_phis()
    assert first is second


def test_static_land_fraction_flat_is_all_ocean(tmp_path):
    """A flat (aquaplanet-like) model yields an all-zero land fraction (all ocean) of
    the grid 2-D shape, built WITHOUT a setup() run (no phantom run directory)."""
    out = tmp_path / "run"
    driver = ModelDriver(_cfg("flat"), output_dir=str(out))
    f_land = driver.static_land_fraction()
    assert tuple(f_land.shape) == tuple(driver.grid.grid_shape_2d)
    assert bool(jnp.all(f_land == 0))          # flat → all ocean
    assert not out.exists()                    # probe writes nothing


def test_static_land_fraction_shares_topography_chain(tmp_path):
    """The land-fraction probe reuses the topography chain — after it runs, phis is
    also populated (one minimal build, both fields)."""
    driver = ModelDriver(_cfg("gaussian"), output_dir=str(tmp_path / "run"))
    f_land = driver.static_land_fraction()
    assert f_land is not None
    assert driver._phis_data is not None       # the same chain set phis too
    assert tuple(f_land.shape) == tuple(driver.grid.grid_shape_2d)


@pytest.mark.parametrize("with_mask", [True, False], ids=["fractional", "flat-ocean"])
@pytest.mark.parametrize("consumer", ["convection", "orographic-gwd"])
def test_gaussian_driver_land_fraction_reaches_both_consumers(tmp_path, with_mask, consumer):
    """Production grid/topography setup preserves fractional column order or zeros."""
    import xarray as xr
    from legoesm.atmosphere.physics.convection.integration import land_fraction_for_columns
    from legoesm.atmosphere.physics.gravity_wave_drag import integration as gwd

    mask_path = tmp_path / "lsm.nc"
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=3, nlev=3),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        topography="flat",
        land_mask_path=str(mask_path) if with_mask else "",
        radiation="gray",
        days=1,
    )
    model = ModelDriver(cfg, output_dir=str(tmp_path / "run"))
    model._bootstrap_runtime()
    model._create_grid()
    n_lat, n_lon = model.grid.n_lat, model.grid.n_lon
    assert n_lat != n_lon
    ncol = n_lat * n_lon
    # Dyadic fractions preserve exact values through NetCDF and dtype casts.
    mask = ((np.arange(ncol).reshape(n_lat, n_lon) * 7) % 17) / 16.0
    assert np.any((mask > 0.0) & (mask < 1.0))
    assert not np.array_equal(mask.reshape(-1), mask.T.reshape(-1))
    if with_mask:
        xr.Dataset(
            {"lsm": (("lat", "lon"), mask)},
            coords={"lat": np.asarray(model.grid.lat) * 180.0 / np.pi,
                    "lon": (np.asarray(model.grid.lon) * 180.0 / np.pi) % 360.0},
        ).to_netcdf(mask_path)
    model._create_topography()

    extract = (land_fraction_for_columns if consumer == "convection"
               else gwd._extract_land_frac)
    actual = extract(model.grid, ncol)
    assert actual is not None, f"{consumer} received no land fraction from model.grid"
    assert actual.shape == (ncol,)
    expected = mask.reshape(-1) if with_mask else np.zeros(ncol)
    np.testing.assert_array_equal(actual, expected)


def test_gaussian_topography_is_ocean_unless_explicit_mask(tmp_path):
    """The mountain changes geopotential; only the named mask supplies land."""
    import xarray as xr

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=3, nlev=3),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        topography="gaussian", radiation="gray", days=1,
    )
    ocean = ModelDriver(cfg, output_dir=str(tmp_path / "ocean"))
    fraction = ocean.static_land_fraction()
    np.testing.assert_array_equal(fraction, np.zeros(ocean.grid.grid_shape_2d))
    assert bool(jnp.any(ocean.static_topography_phis() != 0))

    mask = ((np.arange(fraction.size).reshape(fraction.shape) * 7) % 17) / 16.0
    mask_path = tmp_path / "land_mask.nc"
    xr.Dataset(
        {"lsm": (("lat", "lon"), mask)},
        coords={"lat": np.asarray(ocean.grid.lat) * 180.0 / np.pi,
                "lon": (np.asarray(ocean.grid.lon) * 180.0 / np.pi) % 360.0},
    ).to_netcdf(mask_path)
    masked = ModelDriver(cfg._replace(land_mask_path=str(mask_path)),
                         output_dir=str(tmp_path / "masked"))
    np.testing.assert_array_equal(masked.static_land_fraction(), mask)
    np.testing.assert_array_equal(masked.static_topography_phis(),
                                  ocean.static_topography_phis())


def test_land_mask_to_ocean_only_chain_end_to_end(tmp_path):
    """END-TO-END (iter 454/455): the realistic AMIP config's --land-mask-path flows through
    the WHOLE production chain a `--ocean-only` empirical run depends on — generator ->
    config.land_mask_path -> driver _create_topography (ERA5-lsm auto-detect, latlon grid) ->
    _f_land -> static_land_fraction -> ocean_valid_mask EXCLUDES the land columns. The
    untested pieces this locks: ERA5 `lsm` (fraction) auto-detect AND the latlon grid the
    AMIP config uses (the prior load_land_fraction test was sftlf-percent on cubed-sphere)."""
    import numpy as np
    import xarray as xr
    from legoesm.training.compare_reanalysis import ocean_valid_mask

    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )

    # Synthetic ERA5-lsm-style mask (variable `lsm`, FRACTION 0..1 — exercises auto-detect +
    # the no-percent-rescale branch): Northern hemisphere land, Southern ocean.
    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(0.0, 358.0, 180)
    mask = np.where(lat[:, None] > 0.0, 1.0, 0.0) * np.ones_like(lon)
    mask_path = tmp_path / "era5_lsm.nc"
    xr.Dataset({"lsm": (("lat", "lon"), mask)},
               coords={"lat": lat, "lon": lon}).to_netcdf(mask_path)

    cfg = build_amip_clubb_lite_config(resolution=8, nlev=6, land_mask_path=str(mask_path))
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "run"))
    f_land = driver.static_land_fraction()
    assert tuple(f_land.shape) == tuple(driver.grid.grid_shape_2d)      # latlon (8, 16)
    assert 0.0 <= float(jnp.min(f_land)) and float(jnp.max(f_land)) <= 1.0

    glat_deg = np.asarray(driver.grid.grid_lat) * 180.0 / np.pi
    fl = np.asarray(f_land)
    assert fl[glat_deg > 10.0].mean() > 0.9        # NH resolved as land
    assert fl[glat_deg < -10.0].mean() < 0.1       # SH resolved as ocean

    mask_ocean = ocean_valid_mask(f_land)           # ocean where land_fraction <= 0.5
    # exactly the SH (ocean) half is rankable; the NH (land) columns are EXCLUDED
    assert 0 < int(jnp.sum(mask_ocean)) < int(mask_ocean.size)
    flat_land = fl.reshape(-1)
    np.testing.assert_array_equal(np.asarray(mask_ocean), flat_land <= 0.5)
