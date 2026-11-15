"""Direct tests for scripts/data/prep_era5_sso_fields.py (offline, synthetic ERA5)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from legoesm import constants

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "prep_era5_sso_fields.py")
_spec = importlib.util.spec_from_file_location("prep_era5_sso_fields", _SCRIPT)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _raw(sdor=100.0, sdfor=30.0, slor=0.02, isor=0.5, anor=0.3, z=None):
    """Synthetic 1-deg 'ERA5' raw fields on a pole-to-pole point grid."""
    lat = np.linspace(90.0, -90.0, 181)
    lon = np.arange(0.0, 360.0, 1.0)
    shape = (lat.size, lon.size)

    def f(v):
        return np.broadcast_to(np.asarray(v, dtype=np.float64), shape).copy()

    zz = np.zeros(shape) if z is None else z
    data = {"sdor": f(sdor), "sdfor": f(sdfor), "slor": f(slor), "isor": f(isor),
            "anor": f(anor), "z": zz}
    return xr.Dataset({k: (("latitude", "longitude"), v) for k, v in data.items()},
                      coords={"latitude": lat, "longitude": lon})


def test_tensor_round_trip():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(1e-3, 0.5, 200)
    gamma = rng.uniform(0.01, 0.99, 200)
    theta = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 200)
    s2, g2, t2 = prep.sso_from_tensor(*prep.sso_tensor(sigma, gamma, theta))
    np.testing.assert_allclose(s2, sigma, rtol=1e-12)
    np.testing.assert_allclose(g2, gamma, rtol=1e-10)
    np.testing.assert_allclose(t2, theta, atol=1e-10)


def test_flat_cell_is_isotropic_and_slopeless():
    s, g, t = prep.sso_from_tensor(np.zeros(3), np.zeros(3), np.zeros(3))
    assert np.all(s == 0.0) and np.all(g == 1.0) and np.all(t == 0.0)


def test_uniform_fields_survive_aggregation():
    out = prep.build_era5_sso(_raw(), block_deg=2.0)
    np.testing.assert_allclose(out["SSO_STDH"].values, 100.0, rtol=1e-12)
    np.testing.assert_allclose(out["SSO_STDH_FILT"].values, 30.0, rtol=1e-12)
    np.testing.assert_allclose(out["SSO_SIGMA"].values, 0.02, rtol=1e-10)
    np.testing.assert_allclose(out["SSO_GAMMA"].values, 0.5, rtol=1e-10)
    np.testing.assert_allclose(out["SSO_THETA"].values, 0.3, atol=1e-10)
    assert out.attrs["block_deg"] == 2.0 and out.attrs["fine_res_deg"] == 1.0
    assert out["SSO_STDH"].shape == (90, 180)
    s, g, t = 0.02 ** 2 / 2, 0.5 ** 2, 0.6
    np.testing.assert_allclose(out["SSO_K"].values, s * (1 + g), rtol=1e-12)
    np.testing.assert_allclose(out["SSO_L"].values, s * (1 - g) * np.cos(t), rtol=1e-12)
    np.testing.assert_allclose(out["SSO_M"].values, s * (1 - g) * np.sin(t), rtol=1e-12)


def test_orthogonal_ridges_average_to_isotropic_not_mean_angle():
    """Half the block ridged at theta=0, half at pi/2: the tensor mean is isotropic.

    A plain angle mean would give theta=pi/4 and keep gamma=0.2.
    """
    raw = _raw(isor=0.2)
    anor = np.zeros_like(raw["anor"].values)
    anor[:, 1::2] = np.pi / 2
    raw["anor"] = (("latitude", "longitude"), anor)
    out = prep.build_era5_sso(raw, block_deg=2.0)
    np.testing.assert_allclose(out["SSO_GAMMA"].values, 1.0, atol=1e-9)
    np.testing.assert_allclose(out["SSO_SIGMA"].values ** 2,
                               0.02 ** 2 * (1 + 0.2 ** 2) / 2, rtol=1e-9)


def test_resolved_variance_adds_to_subgrid_and_sdfor_does_not():
    """A resolved height step inside a block adds its variance to std, not to sdfor."""
    z = np.zeros((181, 360))
    z[:, 0::2] = 400.0 * constants.g_ifs
    out = prep.build_era5_sso(_raw(z=z), block_deg=2.0)
    # resolved part: values 0/400 m in equal area halves -> variance 200^2
    np.testing.assert_allclose(out["SSO_STDH"].values,
                               np.sqrt(100.0 ** 2 + 200.0 ** 2), rtol=1e-12)
    np.testing.assert_allclose(out["SSO_STDH_FILT"].values, 30.0, rtol=1e-12)


def test_negative_surface_height_is_not_clipped():
    """Vote V4: ERA5 z is surface height, so a depression is real resolved variance."""
    z = np.zeros((181, 360))
    z[:, 0::2] = -100.0 * constants.g_ifs
    out = prep.build_era5_sso(_raw(sdor=0.0, z=z), block_deg=2.0)
    np.testing.assert_allclose(out["SSO_STDH"].values, 50.0, rtol=1e-12)


def test_block_must_tile():
    with pytest.raises(ValueError):
        prep.build_era5_sso(_raw(), block_deg=1.5)


def test_existing_loader_reads_the_file_on_cube_and_unstructured(tmp_path):
    """The runtime loader samples SSO_STDH onto structured and Voronoi-style grids."""
    from types import SimpleNamespace

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.topography import load_subgrid_orography

    path = tmp_path / "era5_sso_2deg.nc"
    prep.build_era5_sso(_raw(), block_deg=2.0).to_netcdf(path)
    cube = load_subgrid_orography(create_cubed_sphere(8), str(path), scale_check="off")
    np.testing.assert_allclose(np.asarray(cube), 100.0, rtol=1e-9)
    rng = np.random.default_rng(1)
    cells = SimpleNamespace(grid_lat=np.arcsin(rng.uniform(-1, 1, 500)),
                            grid_lon=rng.uniform(0, 2 * np.pi, 500))
    vor = load_subgrid_orography(cells, str(path), scale_check="off")
    np.testing.assert_allclose(np.asarray(vor), 100.0, rtol=1e-9)


def test_build_refuses_non_finite_input():
    raw = _raw()
    sd = raw["sdor"].values.copy()
    sd[10, 10] = np.nan
    raw["sdor"] = (("latitude", "longitude"), sd)
    with pytest.raises(ValueError, match="non-finite"):
        prep.build_era5_sso(raw, block_deg=2.0)


def test_blocks_are_cos_lat_area_weighted():
    """sdor differs between the two rows of each block: the mean of sdor^2 is cos-weighted."""
    raw = _raw()
    lat = raw["latitude"].values
    sd = np.where((np.round(lat) % 2 == 0)[:, None], 100.0, 300.0) * np.ones((1, 360))
    raw["sdor"] = (("latitude", "longitude"), sd)
    out = prep.build_era5_sso(raw, block_deg=2.0)
    rows = np.sort(lat)[1:].reshape(-1, 2)          # cell rows after dropping -90
    w = np.cos(np.deg2rad(rows))
    v = np.where(np.round(rows) % 2 == 0, 100.0, 300.0) ** 2
    expect = np.sqrt((w * v).sum(1) / w.sum(1))
    np.testing.assert_allclose(out["SSO_STDH"].values[:, 0], expect, rtol=1e-12)
    assert not np.allclose(expect[-10:], np.sqrt((100.0 ** 2 + 300.0 ** 2) / 2), rtol=1e-4)


def test_rejects_partial_globe_and_out_of_range_inputs():
    with pytest.raises(ValueError, match="global"):
        prep.build_era5_sso(_raw().isel(longitude=slice(0, 180)), block_deg=2.0)
    raw = _raw()
    iso = raw["isor"].values.copy()
    iso[5, 5] = 1.2
    raw["isor"] = (("latitude", "longitude"), iso)
    with pytest.raises(ValueError, match="isor"):
        prep.build_era5_sso(raw, block_deg=2.0)


def _resolved(z_m, block_deg=2.0):
    """Build with no ERA5 subgrid slope, so K/L/M are the resolved terrain's alone."""
    out = prep.build_era5_sso(_raw(slor=0.0, z=z_m * constants.g_ifs), block_deg=block_deg)
    return out["SSO_K"].values, out["SSO_L"].values, out["SSO_M"].values


def _row_cos(block):
    """cos(lat) of the cell rows of each block of the 1-deg fixture."""
    return np.cos(np.deg2rad(np.arange(-89.0, 90.5, 1.0).reshape(-1, block)))


_D = np.deg2rad(1.0) * constants.R_earth


def test_resolved_slope_zonal_ramp_is_exact():
    """h = c * lon index: every in-block square has hx = c / (D cos), hy = 0."""
    c = 10.0
    z = np.broadcast_to(c * np.arange(360.0), (181, 360)).copy()
    K, L, M = _resolved(z)
    w = _row_cos(2)
    expect = (w * 0.5 * (c / (_D * w)) ** 2).sum(1) / w.sum(1)
    np.testing.assert_allclose(K[:, 0], expect, rtol=1e-12)
    np.testing.assert_allclose(L[:, 0], expect, rtol=1e-12)
    np.testing.assert_allclose(M, 0.0, atol=1e-20)


def test_resolved_slope_meridional_and_oblique_are_exact():
    c = 10.0
    i = np.arange(181.0)[::-1, None]          # raw lat runs 90 -> -90; i grows northward
    K, L, M = _resolved(np.broadcast_to(c * i, (181, 360)).copy())
    hy2 = (c / _D) ** 2
    np.testing.assert_allclose(K, 0.5 * hy2, rtol=1e-12)
    np.testing.assert_allclose(L, -0.5 * hy2, rtol=1e-12)
    np.testing.assert_allclose(M, 0.0, atol=1e-20)
    j = np.arange(360.0)[None, :]
    K2, L2, M2 = _resolved(c * i + c * j)
    wr = _row_cos(2)
    hx = 0.5 * (c / (_D * wr[:, 0]) + c / (_D * wr[:, 1]))     # mean of south/north edges
    np.testing.assert_allclose(M2[:, 0], hx * (c / _D), rtol=1e-12)


def test_resolved_slope_keeps_two_point_variation():
    """Alternating 0/400 m columns: a centred stencil would see no slope at all."""
    z = np.zeros((181, 360))
    z[:, 1::2] = 400.0
    K, L, M = _resolved(z)
    w = _row_cos(2)
    expect = (w * 0.5 * (400.0 / (_D * w)) ** 2).sum(1) / w.sum(1)
    np.testing.assert_allclose(K[:, 0], expect, rtol=1e-12)
    np.testing.assert_allclose(M, 0.0, atol=1e-20)


def test_resolved_slope_sees_a_checkerboard():
    """Checkerboard 0/400 m: averaging gradients before squaring would cancel to zero."""
    z = 400.0 * ((np.arange(181)[:, None] + np.arange(360)[None, :]) % 2)
    K, L, M = _resolved(z.astype(float))
    wr = _row_cos(2)
    hx2 = (wr * (400.0 / (_D * wr)) ** 2).sum(1) / wr.sum(1)
    hy2 = (400.0 / _D) ** 2
    np.testing.assert_allclose(K[:, 0], 0.5 * (hx2 + hy2), rtol=1e-12)
    np.testing.assert_allclose(L[:, 0], 0.5 * (hx2 - hy2), rtol=1e-12)
    np.testing.assert_allclose(M, 0.0, atol=1e-20)


def test_jump_between_blocks_is_resolved_not_subgrid():
    """Blocks flat inside, different from their neighbours: no subgrid slope or variance."""
    z = np.zeros((181, 360))
    z[:, 2::4] = 500.0
    z[:, 3::4] = 500.0
    K, L, M = _resolved(z)
    np.testing.assert_allclose(K, 0.0, atol=1e-20)
    out = prep.build_era5_sso(_raw(sdor=0.0, slor=0.0, z=z * constants.g_ifs), block_deg=2.0)
    np.testing.assert_allclose(out["SSO_STDH"].values, 0.0, atol=1e-9)
