"""The split-explicit sea level damping must reach the tracer transports.

With the default barotropic eta damping on, the sea level change of a step must
equal minus dt times the column divergence of the tracer mass fluxes in every
column, and a passive tracer of one must stay one.  Before the damping was
booked as a transport the two differed by up to 1.7 cm per step in a wind
driven box and the tracer of one drifted by 1e-3 within a few days.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.grids.operators_latlon_cgrid import divergence_cgrid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    make_partial_cell_latlon, rest_state_latlon_cgrid_ocean)
from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0
N_STEPS = 40


def test_damping_booked_in_tracer_transports():
    grid, wall = create_regional_grid("latlon", n_lat=12, n_lon=16, lat_south=-6.0,
                                      lat_north=6.0, lon_west=190.0, lon_east=206.0)
    lat = np.broadcast_to(np.linspace(-6.5, 6.5, 14)[:, None], (14, 18))
    lon = np.broadcast_to(np.linspace(189.5, 206.5, 18)[None, :], (14, 18))
    ocean = np.asarray(wall) > 0.5
    H = np.where(ocean, 3000.0 - 1500.0 * np.exp(-((lon - 198.0) / 2.0) ** 2), 0.0)   # a ridge
    dz = np.geomspace(10.0, 500.0, 8)
    dz *= 3000.0 / dz.sum()
    zc = create_z_star_from_thicknesses(dz)
    zc_pc, H2, lm = make_partial_cell_latlon(zc, H, ocean.astype(float))
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H2), H_max=float(zc.H_max))
    h0 = np.asarray(compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc))
    mask3 = (h0 > 0) & ocean[..., None] & (np.asarray(lm) > 0.5)[..., None]
    mask2 = mask3[..., 0].astype(float)
    zmid = np.cumsum(dz) - dz / 2
    T0 = (4.0 + 22.0 * np.exp(-zmid / 200.0))[None, None, :] * mask3
    st = st._replace(T=st.T.replace(data=jnp.asarray(T0)),
                     passive=jnp.asarray(mask3, float)[..., None])
    cfg = LatLonCGridOceanConfig.from_flat(store_mass_flux=True, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"), lateral_mixing=LateralMixingConfig(scheme="none")))
    assert cfg.barotropic.barotropic_diffusion_alpha > 0.0          # the damping under test is on
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    taux = 0.08 * np.cos(np.deg2rad(lat * 8)) ** 2 * mask2
    sf = OceanSurfaceForcing(tau_x=jnp.asarray(-taux), tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(50.0 * mask2), sw_down=None)
    worst = 0.0
    for _ in range(N_STEPS):
        new = model.step(st, DT, surface_forcing=sf)
        div = np.asarray(divergence_cgrid(new.mass_flux_u.data, new.mass_flux_v.data, grid).sum(-1))
        worst = max(worst, float(np.abs((np.asarray(new.eta.data - st.eta.data) + DT * div) * mask2).max()))
        st = new
    one = np.asarray(st.passive[..., 0])
    assert worst < 1e-12, worst
    assert np.abs(np.where(mask3, one - 1.0, 0.0)).max() < 1e-11
