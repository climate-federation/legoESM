"""The Pacific zoom builder selects every ported zoom option, and a model accepts it."""
import jax

jax.config.update("jax_enable_x64", True)

from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.experiments.pacific_zoom import (  # noqa: E402
    OM4P5, pacific_zoom_config, pacific_zoom_open_boundary)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import create_z_star_from_thicknesses  # noqa: E402

import numpy as np  # noqa: E402


def test_builder_selects_every_zoom_option():
    c = pacific_zoom_config(LatLonCGridOceanConfig())
    assert (c.momentum_face_mask, c.coriolis_time_scheme, c.coriolis_baroclinic_operator) == (
        "3d", "trapezoidal_step", "energy_conserving")
    assert c.lateral_friction_scheme == "om4p25" and c.omp25 is OM4P5
    assert OM4P5.resolution_function == "mom6_step_wkb" and OM4P5.bound_coef == 0.8
    assert c.open_boundary_smooth_width == 0.01 and OM4P5.smooth_width == 0.01
    ob = pacific_zoom_open_boundary()
    assert ob.velocity_rx_max == 1.0 and ob.tangential_nudge_only is True
    grid, _ = create_regional_grid("latlon", n_lat=6, n_lon=8, lat_south=-3.0, lat_north=3.0,
                                   lon_west=200.0, lon_east=208.0)
    zc = create_z_star_from_thicknesses(np.array([20.0, 40.0, 80.0]))
    model = LatLonCGridOceanModel(grid, zc, c)          # the model validates the selection
    assert model.config.coriolis_time_scheme == "trapezoidal_step"
