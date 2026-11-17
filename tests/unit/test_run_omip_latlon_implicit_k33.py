"""The lat-lon bathymetry OMIP config applies the Redi K33 diagonal implicitly.

Explicit K33 on the ~1 m eORCA1 top layers at dt=1800 s overshoots surface T
by ~100 degC in one step.  implicit_K33 only takes effect inside the implicit
vertical solve, so the driver forces that solve on and the model refuses the
pair that would silently drop K33.
"""

from __future__ import annotations

import pytest

from scripts.run.run_omip import _create_setup


def _bathy_config(**kw):
    _, _, config, _, _ = _create_setup(
        "latlon", "18x36", 4, 5000.0,
        physics_preset="full", water_type="II", use_bathymetry=True, **kw)
    return config


def test_bathy_default_is_implicit_k33_with_implicit_vertical_mixing():
    cfg = _bathy_config(implicit_vertical_mixing=False)
    assert cfg.gm_redi.implicit_K33 is True
    assert cfg.implicit_vertical_mixing is True


def test_no_gm_redi_does_not_force_implicit_vertical_mixing():
    cfg = _bathy_config(implicit_vertical_mixing=False, no_gm_redi=True)
    assert cfg.gm_redi is None
    assert cfg.implicit_vertical_mixing is False


def test_model_refuses_implicit_k33_on_explicit_vertical_mixing():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=5, H_max=3000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        gm_redi=GMRediConfig(implicit_K33=True), implicit_vertical_mixing=False)
    with pytest.raises(ValueError, match="implicit_K33=True requires"):
        LatLonCGridOceanModel(grid, z, cfg)
