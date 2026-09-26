"""CAM6's 32-level hybrid table, as a legoESM vertical coordinate.

The numbers are CESM2.1's ``cam_vcoords_L32_c180105.nc`` (hyai/hybi, P0 = 1e5
Pa).  These tests pin the properties the AMIP lane relies on: the published
top pressure, positive layer mass over the highest real terrain (the reason
the analytic hybrid builder was unusable there), and the config/driver
dispatch that selects the table.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.grids.vertical import (
    CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0, make_cam6_l32_levels,
)

jax.config.update("jax_enable_x64", True)


def test_table_reproduces_the_cam6_file():
    # Spot values read from the file (top, the A maximum at interface 14, the
    # first nonzero B, the surface) -- a mistyped digit anywhere shifts these.
    assert len(CAM6_L32_HYAI) == 33 and len(CAM6_L32_HYBI) == 33
    assert CAM6_L32_HYAI[0] == pytest.approx(0.00225523952394724, rel=0, abs=1e-17)
    assert CAM6_L32_HYAI[14] == pytest.approx(0.181863352656364, rel=0, abs=1e-15)
    assert CAM6_L32_HYBI[15] == pytest.approx(0.0393548272550106, rel=0, abs=1e-16)
    assert CAM6_L32_HYAI[-1] == 0.0 and CAM6_L32_HYBI[-1] == 1.0
    assert all(b == 0.0 for b in CAM6_L32_HYBI[:15])


def test_top_pressure_and_positive_layer_mass_over_terrain():
    c = make_cam6_l32_levels()
    assert c.n_levels == 32
    p_half = np.asarray(c.pressure_at_half(np.asarray([1.0e5])))
    assert p_half[0, 0] == pytest.approx(225.52395, abs=1e-3)      # 2.26 hPa top
    # Tibet reaches ~543 hPa; the analytic hybrid inverted below ~656 hPa.
    for ps in (1.05e5, 1.0e5, 8.0e4, 6.0e4, 5.0e4):
        dp = np.diff(np.asarray(c.pressure_at_half(np.asarray([ps])))[0])
        assert dp.min() > 250.0, (ps, dp.min())
    assert np.all(np.diff(CAM6_L32_HYBI) >= 0.0)


def test_wrong_reference_pressure_is_refused():
    with pytest.raises(ValueError, match="P0"):
        make_cam6_l32_levels(p_ref=CAM6_L32_P0 * 1.01)


def test_config_dispatch_requires_32_levels_and_default_knobs():
    from legoesm.driver.config import ExperimentConfig, GridConfig
    ExperimentConfig(grid=GridConfig(vertical_coord="cam_l32", nlev=32)).validate_strict()
    bad = ExperimentConfig(grid=GridConfig(vertical_coord="cam_l32", nlev=36))
    with pytest.raises(ValueError, match="nlev must be 32"):
        bad.validate_strict()
    inert = ExperimentConfig(grid=GridConfig(vertical_coord="cam_l32", nlev=32, stretching=3.0))
    with pytest.raises(ValueError, match="inert"):
        inert.validate_strict()
    with pytest.raises(ValueError, match="vertical_coord must be one of"):
        ExperimentConfig(grid=GridConfig(vertical_coord="bogus")).validate_strict()
