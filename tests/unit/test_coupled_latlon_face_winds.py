"""A coupled run on the lat-lon C-grid atmosphere carries its face winds.

The canonical CMIP decks (config/cmip/cmip_ocean_slab.yaml, cmip_ocean_3D.yaml)
are coupled lat-lon C-grid runs.  The coupled driver reads the lowest-level
winds for the surface fluxes through the atmosphere's cell-centre view, so the
atmosphere can keep its native face winds between steps instead of being
refused (or re-projecting every step).
"""
from __future__ import annotations

import jax
import numpy as np

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState, hydrostatic_to_facewind)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.coupled_config import PRESETS
from legoesm.driver.coupled_esm_driver import CoupledESMDriver

jax.config.update("jax_enable_x64", True)


def test_coupled_latlon_run_carries_face_winds(tmp_path):
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=600.0),
        output=OutputConfig(diag_days=1),
        radiation="gray", days=1)
    drv = CoupledESMDriver(atm, PRESETS["aquaplanet"](),
                           output_dir=str(tmp_path))
    drv.setup()
    assert drv.run() == "COMPLETED"
    st = drv._atm.state
    assert isinstance(st, LatLonFaceWindState)
    cc = drv._atm.cell_centre_state()
    u = np.asarray(cc.u.data)
    assert np.isfinite(u).all() and np.abs(u).max() > 0.0
    # the faces are the native shapes, not the cell-centre ones
    assert st.u_f.data.shape[1] == u.shape[1] + 1
    assert st.v_f.data.shape[0] == u.shape[0] + 1
    _assert_carried(drv._atm)


def _assert_carried(atm):
    """The faces are carried, not re-derived from the cell-centre average."""
    st = atm.state
    relift = hydrostatic_to_facewind(atm.cell_centre_state(), atm.grid)
    for f in ("u_f", "v_f"):
        assert np.abs(np.asarray(getattr(st, f).data)
                      - np.asarray(getattr(relift, f).data)).max() > 0.0, f


def test_earth_system_latlon_run_carries_face_winds(tmp_path):
    from legoesm.driver.earth_system_driver import EarthSystemDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=600.0),
        output=OutputConfig(diag_days=1),
        radiation="gray", days=1, dataset="analytical", precision="fp64")
    drv = EarthSystemDriver(cfg, output_dir=tmp_path)
    drv.setup()
    assert drv.run() == "COMPLETED"
    assert isinstance(drv.state, LatLonFaceWindState)
    _assert_carried(drv._atm)
