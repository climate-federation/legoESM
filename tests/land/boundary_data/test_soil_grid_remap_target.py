"""The surfdata soil remap must target the CONFIG's soil grid.

``build_soil_hydraulics`` derives per-(col, layer) Cosby hydraulics from
``gsd.sand_frac``/``clay_frac``, which the loader has already remapped onto a
:class:`SoilGrid`.  That target used to be hardcoded to ``SoilGridConfig()``
(8 layers), so a ``MultiLayerLandConfig`` carrying any other discretisation got
(ncol, 8) hydraulics against an (ncol, n_layers) soil state — a silent desync.
These tests pin the two together.
"""
import numpy as np
import pytest
import xarray as xr

from legoesm.grids.latlon import create_latlon_grid
from legoesm.land.boundary_data import init_land_surface_data
from legoesm.land.canopy import CanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid

_SURFDATA = "data/legoesm_surfdata_c260716.nc"


def _have_surfdata():
    import os
    return os.path.exists(_SURFDATA)


pytestmark = pytest.mark.skipif(
    not _have_surfdata(), reason=f"{_SURFDATA} not staged")


def _hydraulics_layer_count(cfg):
    grid = create_latlon_grid(8)                     # 8 x 16, cheap
    out_cfg, _params, gsd = init_land_surface_data(_SURFDATA, grid, cfg, 0.0)
    # theta_sat is one of the per-(col, layer) Cosby fields
    sat = np.asarray(out_cfg.hydraulics.theta_sat)
    return sat.shape, np.asarray(gsd.sand_frac).shape


@pytest.mark.parametrize("n_layers,total_depth", [
    (8, 0.0),        # the default — must stay unchanged
    (10, 3.0),       # AMIP parity
    (6, 2.0),        # an arbitrary third grid, so the test is not 2-point
])
def test_hydraulics_layers_follow_the_config_soil_grid(n_layers, total_depth):
    cfg = MultiLayerLandConfig(
        surface_scheme=CanopyConfig(),
        soil_grid=SoilGridConfig(n_layers=n_layers, total_depth=total_depth))
    sat_shape, sand_shape = _hydraulics_layer_count(cfg)
    assert sat_shape[1] == n_layers, (
        f"hydraulics has {sat_shape[1]} layers for a {n_layers}-layer config — "
        "the loader remap target desynced from MultiLayerLandConfig.soil_grid")
    assert sand_shape[1] == n_layers


def test_configured_depth_is_honoured():
    """10 layers / 3.0 m must really be 3.0 m deep, not the 6.375 m default."""
    sg = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0))
    assert float(sg.z_interface[-1]) == pytest.approx(3.0, rel=1e-9)
    assert sg.n_layers == 10
    default = make_soil_grid(SoilGridConfig())
    assert float(default.z_interface[-1]) == pytest.approx(6.375, rel=1e-6)


def test_soil_state_and_hydraulics_shapes_agree():
    """The Richards solver indexes hydraulics against the soil state; a mismatch
    here is the bug this plumbing prevents."""
    from legoesm.land.multilayer_land import init_multilayer_land_state

    cfg = MultiLayerLandConfig(
        surface_scheme=CanopyConfig(),
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0))
    grid = create_latlon_grid(8)
    out_cfg, _params, _gsd = init_land_surface_data(_SURFDATA, grid, cfg, 0.0)
    ncol = int(np.asarray(out_cfg.hydraulics.theta_sat).shape[0])
    state = init_multilayer_land_state(ncol, out_cfg)
    assert np.asarray(state.theta_soil).shape == \
        np.asarray(out_cfg.hydraulics.theta_sat).shape
    assert np.asarray(state.T_soil).shape[1] == 10
