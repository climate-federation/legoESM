"""Unit tests for the M5 global surface-data loader (plan §10/M5).

Exercises the pure regrid/remap/renormalize assembly and the per-step time
interpolators on synthetic data — no real surfdata files required.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.land.global_surface_data import (
    GlobalSurfaceDataConfig,
    GlobalSurfaceData,
    _RawSurfaceFields,
    build_global_surface_data,
    get_surfdata_preset,
    interp_annual,
    interp_monthly,
    dominant_pft_collapse,
    _renorm_cover,
    _renorm_pft,
    _remap_soil_layers,
    _MONTH_CENTRES,
)
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
from legoesm.land.surface_params import N_PFT_CLM5


# ---------------------------------------------------------------------------
# Synthetic source + target fixtures
# ---------------------------------------------------------------------------
class _ToyGaussianGrid:
    """Minimal grid exposing lat2d/lon2d (radians) + grid_area, like GaussianGrid."""

    def __init__(self, n_lat=8, n_lon=16):
        lat = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, n_lat)
        lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
        self.lon2d, self.lat2d = np.meshgrid(lon, lat)
        self.grid_area = jnp.asarray(np.full(n_lat * n_lon, 1.0e9), dtype=jnp.float32)
        self.ncol = n_lat * n_lon


def _make_raw(nlat=12, nlon=24, n_src_layer=10, nyear=3, npft=None):
    npft = npft if npft is not None else N_PFT_CLM5
    rng = np.random.default_rng(0)
    src_lat = np.linspace(-89.0, 89.0, nlat)
    src_lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    src_soil_depth = np.cumsum(np.full(n_src_layer, 0.1))  # 0.1..1.0 m

    def soil(scale):
        return rng.uniform(0.0, scale, size=(nlat, nlon, n_src_layer))

    # cover fractions that sum to <= 1 before renorm in some cells, > 1 in others
    f_land = rng.uniform(0.0, 0.9, size=(nyear, nlat, nlon))
    f_lake = rng.uniform(0.0, 0.3, size=(nyear, nlat, nlon))
    f_glacier = rng.uniform(0.0, 0.3, size=(nyear, nlat, nlon))

    pft = rng.uniform(0.0, 1.0, size=(nyear, nlat, nlon, npft))
    veg = lambda: rng.uniform(0.0, 5.0, size=(12, nlat, nlon, npft))

    return _RawSurfaceFields(
        src_lat=src_lat, src_lon=src_lon, src_soil_depth=src_soil_depth,
        sand=soil(0.6), clay=soil(0.4), organic=soil(50.0), bulk_density=soil(1500.0),
        soil_color=rng.integers(1, 20, size=(nlat, nlon)).astype(np.float64),
        years=np.arange(nyear, dtype=np.float64),
        f_land=f_land, f_lake=f_lake, f_glacier=f_glacier, pft_frac=pft,
        lai=veg(), sai=veg(), htop=veg(), hbot=veg(),
        cell_area=None,
    )


# ---------------------------------------------------------------------------
# build_global_surface_data — shapes, ranges, conservation
# ---------------------------------------------------------------------------
def test_build_shapes_and_grid_assembly():
    grid = _ToyGaussianGrid()
    soil_grid = make_soil_grid(SoilGridConfig())  # 8 layers
    raw = _make_raw()
    gsd = build_global_surface_data(raw, grid, GlobalSurfaceDataConfig(), soil_grid=soil_grid)

    ncol = grid.ncol
    nyear = raw.years.shape[0]
    npft = raw.pft_frac.shape[-1]

    assert isinstance(gsd, GlobalSurfaceData)
    assert gsd.sand_frac.shape == (ncol, soil_grid.n_layers)
    assert gsd.clay_frac.shape == (ncol, soil_grid.n_layers)
    assert gsd.soil_color.shape == (ncol,)
    assert gsd.soil_color.dtype == jnp.int32
    assert gsd.cell_area.shape == (ncol,)
    assert gsd.f_land.shape == (nyear, ncol)
    assert gsd.pft_frac.shape == (nyear, ncol, npft)
    assert gsd.lai_monthly.shape == (12, ncol, npft)
    assert gsd.months.shape == (12,)


def test_build_cover_partition_of_unity():
    """land+lake+glacier must be <= 1 everywhere after renorm."""
    grid = _ToyGaussianGrid()
    gsd = build_global_surface_data(_make_raw(), grid, GlobalSurfaceDataConfig())
    total = gsd.f_land + gsd.f_lake + gsd.f_glacier
    assert jnp.all(total <= 1.0 + 1e-5)
    assert jnp.all(gsd.f_land >= 0.0)


def test_build_pft_sums_to_one():
    grid = _ToyGaussianGrid()
    gsd = build_global_surface_data(_make_raw(), grid, GlobalSurfaceDataConfig())
    psum = jnp.sum(gsd.pft_frac, axis=-1)
    assert jnp.allclose(psum, 1.0, atol=1e-5)


def test_build_uses_grid_area_when_no_area_field():
    grid = _ToyGaussianGrid()
    gsd = build_global_surface_data(_make_raw(), grid, GlobalSurfaceDataConfig())
    assert jnp.allclose(gsd.cell_area, jnp.asarray(grid.grid_area))


def test_regrid_constant_field_is_preserved():
    """A spatially constant source must regrid to the same constant (IDW exactness)."""
    grid = _ToyGaussianGrid()
    raw = _make_raw()
    const = np.full_like(raw.sand, 0.314)
    raw = raw._replace(sand=const)
    gsd = build_global_surface_data(raw, grid, GlobalSurfaceDataConfig())
    assert jnp.allclose(gsd.sand_frac, 0.314, atol=1e-4)


# ---------------------------------------------------------------------------
# Leaf helpers
# ---------------------------------------------------------------------------
def test_renorm_cover_scales_only_overshoot():
    f_land = jnp.array([0.5, 0.8])
    f_lake = jnp.array([0.2, 0.6])
    f_glacier = jnp.array([0.1, 0.6])  # cell 0 sums 0.8 (ok), cell 1 sums 2.0
    a, b, c = _renorm_cover(f_land, f_lake, f_glacier)
    assert jnp.allclose(a[0], 0.5) and jnp.allclose(b[0], 0.2) and jnp.allclose(c[0], 0.1)
    assert jnp.allclose(a[1] + b[1] + c[1], 1.0, atol=1e-6)


def test_renorm_pft_keeps_bare_soil_zero():
    pft = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    out = _renorm_pft(pft)
    assert jnp.allclose(out[0], 0.0)
    assert jnp.allclose(jnp.sum(out[1]), 1.0)


def test_remap_soil_layers_linear():
    # constant-in-depth field stays constant after remap
    src_depth = np.array([0.1, 0.5, 1.0])
    field = jnp.array([[2.0, 2.0, 2.0], [1.0, 3.0, 5.0]])  # (ncol=2, 3 layers)
    dst_depth = np.array([0.2, 0.7])
    out = _remap_soil_layers(field, src_depth, dst_depth)
    assert out.shape == (2, 2)
    assert jnp.allclose(out[0], 2.0)            # constant column preserved
    # column 1 is linear in the [0.1,1.0] range -> interpolated monotonic
    assert out[1, 0] < out[1, 1]


# ---------------------------------------------------------------------------
# Time interpolation
# ---------------------------------------------------------------------------
def test_interp_annual_endpoints_and_midpoint():
    years = jnp.array([2000.0, 2001.0, 2002.0])
    field = jnp.array([[10.0], [20.0], [30.0]])  # (nyear, ncol=1)
    assert jnp.allclose(interp_annual(field, years, jnp.array(2000.0)), 10.0)
    assert jnp.allclose(interp_annual(field, years, jnp.array(2002.0)), 30.0)
    assert jnp.allclose(interp_annual(field, years, jnp.array(2000.5)), 15.0)
    # clamp beyond range
    assert jnp.allclose(interp_annual(field, years, jnp.array(1990.0)), 10.0)


def test_interp_annual_single_year():
    field = jnp.array([[42.0]])
    out = interp_annual(field, jnp.array([0.0]), jnp.array(5.0))
    assert jnp.allclose(out, 42.0)


def test_interp_monthly_hits_centres_and_wraps():
    field = jnp.arange(12.0).reshape(12, 1)
    for m in range(12):
        out = interp_monthly(field, jnp.asarray(_MONTH_CENTRES[m]))
        assert jnp.allclose(out, float(m), atol=1e-4)
    # mid-Dec -> mid-Jan wrap stays within [11, 0]-ish blend, finite & bounded
    mid = interp_monthly(field, jnp.asarray(360.0))
    assert jnp.isfinite(mid).all()


# ---------------------------------------------------------------------------
# Dominant-PFT collapse (§10a)
# ---------------------------------------------------------------------------
def test_dominant_pft_collapse_shapes_and_semantics():
    grid = _ToyGaussianGrid()
    gsd = build_global_surface_data(_make_raw(), grid, GlobalSurfaceDataConfig())
    col = dominant_pft_collapse(gsd)
    ncol = grid.ncol
    assert col.pft_frac.shape == (gsd.years.shape[0], ncol, 1)
    assert col.lai_monthly.shape == (12, ncol, 1)
    # collapsed pft slot 0 holds the total vegetated fraction (sum over PFTs)
    assert jnp.allclose(col.pft_frac[..., 0], jnp.sum(gsd.pft_frac, axis=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# Presets / dispatch discipline
# ---------------------------------------------------------------------------
def test_preset_known():
    assert get_surfdata_preset("clm5_surfdata").dataset == "clm5_surfdata"
    assert get_surfdata_preset("modis").dataset == "modis"


def test_preset_unknown_raises():
    with pytest.raises(ValueError):
        get_surfdata_preset("not_a_dataset")
