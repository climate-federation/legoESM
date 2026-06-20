"""Phase-1 coupled 3D-ocean wiring (docs/coupled_3d_ocean_plan.md): the coupled
driver can step the prognostic LatLonCGridOceanModel (ocean_mode='dynamic') on a
SHARED lat-lon grid with the atmosphere, instead of only a thermodynamic slab.

These tests pin the config dispatch + the build path (a real coupled segment is
exercised by the integration smoke `scripts/tmp/_ocean3d_smoke.sbatch`, which is
too heavy/JIT-bound for unit CI).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from legoesm.driver.coupled_config import CoupledConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.simple_ocean import SimpleOceanConfig


def test_config_holds_dynamic_ocean():
    """CoupledConfig accepts ocean_mode='dynamic' + a LatLonCGridOceanConfig +
    the 3D-ocean knobs (the union widening); slab default is unchanged."""
    slab = CoupledConfig()
    assert slab.ocean_mode == "slab"
    assert isinstance(slab.ocean_config, SimpleOceanConfig)
    # 3D-ocean knobs default but present
    assert slab.ocean_nlev == 20 and slab.ocean_dt_s == 300.0
    dyn = CoupledConfig(
        ocean_mode="dynamic", ocean_config=LatLonCGridOceanConfig(),
        ocean_nlev=8, ocean_dt_s=300.0,
    )
    assert dyn.ocean_mode == "dynamic"
    assert isinstance(dyn.ocean_config, LatLonCGridOceanConfig)


def test_init_ocean_rejects_unknown_mode():
    """_init_ocean dispatch raises on an unknown ocean_mode (no silent
    else->slab; CLAUDE.md dispatch-hardening)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    # Build a driver far enough to call _init_ocean with a bogus mode.  Use a
    # tiny lat-lon atm; _init_ocean runs after the atm grid exists.
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="garbage_mode"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="unknown ocean_mode"):
        drv._init_ocean()


def test_dynamic_ocean_requires_latlon_grid():
    """ocean_mode='dynamic' on a non-lat-lon ocean grid raises (same-grid Phase
    1 only; cube-atm + tripole-ocean needs the deferred cross-grid remap)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig(), ocean_nlev=4),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="requires a regular lat-lon ocean grid"):
        drv._init_ocean()


# ---------------------------------------------------------------------------
# WOA realistic initial condition (ocean_ic="woa")
# ---------------------------------------------------------------------------

import os

_WOA_T = "data/woa18/woa18_decav_t00_01.nc"
_WOA_S = "data/woa18/woa18_decav_s00_01.nc"


def test_config_holds_woa_ic_fields():
    """CoupledConfig exposes the WOA-IC knobs; defaults keep the rest state."""
    c = CoupledConfig()
    assert c.ocean_ic == "rest"
    assert c.woa_t_path is None and c.woa_s_path is None
    w = CoupledConfig(ocean_ic="woa", woa_t_path="t.nc", woa_s_path="s.nc")
    assert w.ocean_ic == "woa" and w.woa_t_path == "t.nc"


def test_woa_ic_requires_paths():
    """ocean_ic='woa' without WOA paths raises (no silent fallback to rest)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig(),
                           ocean_nlev=4, ocean_ic="woa"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="requires woa_t_path"):
        drv._init_ocean()


def test_init_dynamic_ocean_rejects_unknown_ic():
    """_init_dynamic_ocean raises on an unknown ocean_ic (dispatch-hardening)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig(),
                           ocean_nlev=4, ocean_ic="garbage_ic"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="ocean_ic must be"):
        drv._init_ocean()


@pytest.mark.skipif(
    not (os.path.exists(_WOA_T) and os.path.exists(_WOA_S)),
    reason="WOA18 files not present (data/woa18/)",
)
def test_woa_ocean_mask_realistic():
    """woa_ocean_mask on a 2deg lat-lon grid yields a physical land/ocean split
    (both land and ocean present; global ocean fraction ~0.6-0.75)."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import woa_ocean_mask
    grid = create_latlon_grid(n_lat=90, n_lon=180)  # 2 deg
    mask = woa_ocean_mask(grid, _WOA_T)
    assert mask.shape == (90, 180)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    ocean_frac = float(mask.mean())
    assert 0.55 < ocean_frac < 0.80, f"ocean_frac={ocean_frac} unphysical"
    # Land must exist (continents) and ocean must exist.
    assert mask.min() == 0.0 and mask.max() == 1.0
