"""Open-boundary radiation does not see dry neighbours along the ring.

A box with west, south and north open, salinity fresher at the boundary (data 34.5) than inside (35.0), and a
wind that drives flow through the open sides. Temperature is 15 everywhere, data included, so it must stay 15 in
every wet cell; dry cells (the land corners of the ring) stay zero. Before the tangential gradients were masked,
the land corners fed zeros into the ring and salinity reached 12 in a corner.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.dynamics.open_boundary_latlon import OpenBoundaryData  # noqa: E402
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
N_STEPS = 8
NY, NX = 10, 14


def _run():
    grid, wall = create_regional_grid("latlon", n_lat=NY, n_lon=NX, lat_south=-5.0,
                                      lat_north=5.0, lon_west=200.0, lon_east=200.0 + NX)
    ocean = np.asarray(wall) > 0.5
    ocean[1:-1, 0] = True
    ocean[0, 1:-1] = True
    ocean[-1, 1:-1] = True
    zc = create_z_star_from_thicknesses(np.array([20.0, 40.0, 80.0, 160.0, 300.0, 600.0]))
    zc_pc, H, lm = make_partial_cell_latlon(zc, np.where(ocean, 1200.0, 0.0), ocean.astype(float),
                                            smoothing_passes=0)
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H), H_max=float(zc.H_max))
    lmj = jnp.asarray(lm)
    nl = st.T.data.shape[-1]
    S = jnp.full(st.S.data.shape, 35.0) * lmj[..., None]
    st = st._replace(T=st.T.replace(data=jnp.full(st.T.data.shape, 15.0) * lmj[..., None]),
                     S=st.S.replace(data=S))
    kw = {}
    for side, n in (("west", NY + 2), ("south", NX + 2), ("north", NX + 2)):
        Sd = jnp.full((nl, n), 34.5)
        bar = "ubar" if side == "west" else "vbar"
        kw.update({f"{side}_{bar}": jnp.zeros(n), f"{side}_zeta": jnp.zeros(n),
                   f"{side}_T": jnp.full((nl, n), 15.0), f"{side}_S": Sd,
                   f"{side}_u3": jnp.zeros((nl, n)), f"{side}_t3": jnp.zeros((nl, n + 1))})
    ob = OpenBoundaryData(tau_in=jnp.asarray(86400.0), tau_out=jnp.asarray(360 * 86400.0), **kw)
    cfg = LatLonCGridOceanConfig.from_flat(fix_eta_drift=False, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"), shortwave_penetration=None))
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    lat = np.degrees(np.asarray(grid.lat))[:, None] * np.ones((1, NX + 2))
    taux = jnp.asarray(0.1 * np.cos(np.deg2rad(lat * 18)) * lm)        # wind that drives flow across the open sides
    z = jnp.zeros_like(st.eta.data)
    sf = OceanSurfaceForcing(tau_x=taux, tau_y=z, q_net=z, sw_down=None)
    sponge = SpongeForcing(gamma=z, T_ref=st.T.data, S_ref=st.S.data, open_boundary=ob)
    uniform_T, ring_change, dry = 0.0, 0.0, 0.0
    for _ in range(N_STEPS):
        st = model.step(st, DT, surface_forcing=sf, sponge=sponge)
        h = np.asarray(compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc))
        wet = h > 0
        Sn, Tn = np.asarray(st.S.data), np.asarray(st.T.data)
        uniform_T = max(uniform_T, float(np.abs(Tn - 15.0)[wet].max()))
        ring_change = max(ring_change, float(np.abs(Sn[:, 0] - 35.0)[wet[:, 0]].max()))
        dry = max(dry, float(np.abs(Sn[~wet]).max()), float(np.abs(Tn[~wet]).max()))
    return uniform_T, ring_change, dry


def test_ring_radiation_ignores_dry_neighbours():
    uniform_T, ring_change, dry = _run()
    assert ring_change > 1e-4, ring_change          # the boundary data reach the ring: the test is not vacuous
    assert uniform_T < 1e-12, uniform_T             # land corners no longer feed zeros into the ring
    assert dry == 0.0, dry                          # nothing is written into dry ring cells or corners
