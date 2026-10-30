"""One owner for the shortwave deposit on the lat-lon ocean with external forcing.

With ``sw_down`` given, the external surface-forcing stage removes the shortwave
from ``q_net`` and deposits it down its own profile; the shared physics
pipeline (``physics.shortwave_penetration``) would deposit 100% of the same
``sw_down`` again.  The step must refuse that combination, and with the
pipeline off the column must gain exactly ``q_net``.
"""
import numpy as np
import pytest

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
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import (  # noqa: E402
    compute_layer_thickness, create_z_star_from_thicknesses)

set_policy(PrecisionPolicy.fp64())
DT = 1800.0
Q = 200.0


def _box():
    grid, wall = create_regional_grid("latlon", n_lat=6, n_lon=8, lat_south=-3.0,
                                      lat_north=3.0, lon_west=200.0, lon_east=208.0)
    ocean = np.asarray(wall) > 0.5
    dz = np.geomspace(10.0, 400.0, 6)
    dz *= 2000.0 / dz.sum()
    zc = create_z_star_from_thicknesses(dz)
    zc_pc, H, lm = make_partial_cell_latlon(zc, np.where(ocean, 2000.0, 0.0), ocean.astype(float))
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H), H_max=float(zc.H_max))
    zmid = np.cumsum(dz) - dz / 2
    T0 = (4.0 + 20.0 * np.exp(-zmid / 200.0))[None, None, :] * np.asarray(lm)[..., None]
    st = st._replace(T=st.T.replace(data=jnp.asarray(np.broadcast_to(T0, st.T.data.shape).copy())))
    return grid, zc_pc, st, np.asarray(lm)


def _cfg(pipeline_off, **sf):
    cfg = LatLonCGridOceanConfig.from_flat(physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none")))
    ph = cfg.physics._replace(surface_forcing=cfg.physics.surface_forcing._replace(**sf))
    if pipeline_off:
        ph = ph._replace(shortwave_penetration=None)
    return cfg._replace(physics=ph)


def _heat_gain(grid, zc_pc, st, lm, cfg, sw):
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    sf = OceanSurfaceForcing(tau_x=jnp.zeros_like(st.eta.data), tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(Q * lm), sw_down=None if sw is None else jnp.asarray(sw * lm))
    new = model.step(st, DT, surface_forcing=sf)
    area = np.asarray(grid.area)[..., None]

    def heat(s):
        h = np.asarray(compute_layer_thickness(s.eta.data, s.H_bathy.data, zc_pc))
        return float(np.sum(h * np.asarray(s.T.data) * area))
    return heat(new) - heat(st)


def test_default_pipeline_with_sw_down_is_refused():
    grid, zc_pc, st, lm = _box()
    with pytest.raises(ValueError, match="would both deposit the shortwave"):
        _heat_gain(grid, zc_pc, st, lm, _cfg(False), Q)


@pytest.mark.parametrize("sf", [{}, dict(shortwave_scheme="jerlov_2band", shortwave_water_type="I",
                                       shortwave_skin_fraction=0.0)])
def test_single_owner_gains_q_net(sf):
    grid, zc_pc, st, lm = _box()
    ref = _heat_gain(grid, zc_pc, st, lm, _cfg(True, **sf), None)       # all of q_net non solar
    got = _heat_gain(grid, zc_pc, st, lm, _cfg(True, **sf), Q)          # q_net with sw_down = q_net
    assert got == pytest.approx(ref, rel=1e-9)


def test_pipeline_alone_owns_sw_down_without_q_net():
    # No q_net: the external stage deposits nothing, so the pipeline is the only owner and the step runs.
    grid, zc_pc, st, lm = _box()
    model = LatLonCGridOceanModel(grid, zc_pc, _cfg(False))
    sf = OceanSurfaceForcing(tau_x=jnp.zeros_like(st.eta.data), tau_y=jnp.zeros_like(st.eta.data),
                             sw_down=jnp.asarray(Q * lm))
    new = model.step(st, DT, surface_forcing=sf)
    assert np.all(np.isfinite(np.asarray(new.T.data)))
