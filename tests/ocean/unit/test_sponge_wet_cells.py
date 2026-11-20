"""The tracer sponge acts on wet cells only, with partial bottom cells.

A closed box over a sloping bottom (partial cells, several bottom levels) with a 2-D sponge rate toward fixed
targets for T and S. Cells below the seafloor must stay at zero: with the 2-D land mask alone
the explicit sponge filled them with target values (S 0.40, 0.81, 1.20 per step on grid E's band, ENSO-project notes (iadicarlo/legoESM-enso) 86),
which other operators could read as water. Wet cells still relax toward the targets.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    make_partial_cell_latlon, rest_state_latlon_cgrid_ocean)
from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig  # noqa: E402
from legoesm.ocean.sponge import SpongeForcing  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0
NY, NX = 8, 12


def test_sponge_leaves_cells_below_the_seafloor_empty():
    grid, wall = create_regional_grid("latlon", n_lat=NY, n_lon=NX, lat_south=-4.0,
                                      lat_north=4.0, lon_west=200.0, lon_east=200.0 + NX)
    ocean = np.asarray(wall) > 0.5
    lon = np.arange(NX + 2)[None, :] * np.ones((NY + 2, 1))
    H = np.where(ocean, 150.0 + 90.0 * lon, 0.0)                        # 150 to 1230 m: many bottom levels
    zc = create_z_star_from_thicknesses(np.array([20.0, 40.0, 80.0, 160.0, 300.0, 600.0]))
    zc_pc, H2, lm = make_partial_cell_latlon(zc, H, ocean.astype(float), smoothing_passes=0)
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H2), H_max=float(zc.H_max))
    wet = np.asarray(compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc)) > 0
    assert (~wet & (np.asarray(lm) > 0.5)[..., None]).any()             # dry levels under wet columns exist
    st = st._replace(T=st.T.replace(data=jnp.asarray(15.0 * wet)), S=st.S.replace(data=jnp.asarray(35.0 * wet)))
    gam = jnp.asarray(np.where(np.asarray(lm) > 0.5, 1.0 / 86400.0, 0.0))  # 2-D: every wet column, all levels
    shape = st.T.data.shape
    sponge = SpongeForcing(gamma=gam, T_ref=jnp.full(shape, 10.0), S_ref=jnp.full(shape, 34.0))
    cfg = LatLonCGridOceanConfig.from_flat(physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"), shortwave_penetration=None))
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    z = jnp.zeros_like(st.eta.data)
    sf = OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z, sw_down=None)
    for _ in range(4):
        st = model.step(st, DT, surface_forcing=sf, sponge=sponge)
    T, S = np.asarray(st.T.data), np.asarray(st.S.data)
    dry = ~wet
    assert np.abs(T[dry]).max() == 0.0 and np.abs(S[dry]).max() == 0.0, (np.abs(T[dry]).max(), np.abs(S[dry]).max())
    assert S[wet].max() < 35.0                                            # the wet cells do relax toward 34
