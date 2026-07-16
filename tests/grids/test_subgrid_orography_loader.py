"""load_subgrid_orography + grid.subgrid_topo_stddev wiring (offline).

The orographic GWD launch reads a per-column ``h_topo`` stddev from the
canonical grid attribute ``subgrid_topo_stddev`` (see
``gravity_wave_drag/integration._extract_subgrid_topo_stddev``); the driver
sets it via ``grid._replace(subgrid_topo_stddev=load_subgrid_orography(...))``
from ``ExperimentConfig.subgrid_orography_path``. These tests lock the loader
output layout and the grid-field plumbing that wiring depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from legoesm.grids.cubed_sphere import CubedSphereGrid, create_cubed_sphere
from legoesm.grids.gaussian import GaussianGrid
from legoesm.grids.topography import load_subgrid_orography


@pytest.fixture(scope="module")
def sso_file(tmp_path_factory):
    """Synthetic 2-deg SSO_STDH: 800 m in a NH mountain box, 0 elsewhere."""
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    box = ((lat[:, None] >= 25) & (lat[:, None] <= 45)
           & (lon[None, :] >= 70) & (lon[None, :] <= 100))
    sso = np.where(box, 800.0, 0.0)
    ds = xr.Dataset({"SSO_STDH": (("lat", "lon"), sso)},
                    coords={"lat": lat, "lon": lon})
    path = tmp_path_factory.mktemp("sso") / "sso_stdh_2deg.nc"
    ds.to_netcdf(path)
    return str(path)


def test_loader_regrids_to_cube_layout_and_range(sso_file):
    grid = create_cubed_sphere(8)
    sso = load_subgrid_orography(grid, sso_file)
    assert sso.shape == (6, 8, 8)
    assert float(sso.min()) >= 0.0
    # peak preserved to within bilinear smoothing of the box edge
    assert 400.0 < float(sso.max()) <= 800.0 + 1e-6
    # mountains cover ~ (20/180)*(30/360) of the sphere -> most columns zero
    assert float((np.asarray(sso) < 1.0).mean()) > 0.8


def test_grid_field_replace_flows_to_per_column_view(sso_file):
    grid = create_cubed_sphere(8)
    assert grid.subgrid_topo_stddev is None          # default: scalar fallback
    sso = load_subgrid_orography(grid, sso_file)
    grid2 = grid._replace(subgrid_topo_stddev=sso)
    # the physics integration reads getattr(grid, 'subgrid_topo_stddev') and
    # flattens to (ncol,); lock that contract here
    raw = getattr(grid2, "subgrid_topo_stddev", None)
    assert raw is not None
    ncol = 6 * 8 * 8
    col = np.asarray(raw).reshape(-1)[:ncol]
    assert col.shape == (ncol,)
    assert np.array_equal(col, np.asarray(sso).reshape(-1))


def test_both_grid_classes_declare_the_field():
    assert "subgrid_topo_stddev" in CubedSphereGrid._fields
    assert "subgrid_topo_stddev" in GaussianGrid._fields
    # field is defaulted so existing full-arity constructions stay valid
    assert CubedSphereGrid._field_defaults["subgrid_topo_stddev"] is None
    assert GaussianGrid._field_defaults["subgrid_topo_stddev"] is None


def test_loader_missing_variable_raises(tmp_path):
    ds = xr.Dataset({"junk": (("lat", "lon"), np.zeros((4, 8)))},
                    coords={"lat": np.linspace(-60, 60, 4),
                            "lon": np.linspace(0, 315, 8)})
    path = tmp_path / "bad.nc"
    ds.to_netcdf(path)
    grid = create_cubed_sphere(4)
    with pytest.raises(KeyError, match="subgrid orography variable"):
        load_subgrid_orography(grid, str(path))
