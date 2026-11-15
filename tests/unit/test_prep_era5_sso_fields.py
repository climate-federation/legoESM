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
