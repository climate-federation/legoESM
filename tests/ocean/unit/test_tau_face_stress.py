"""tau_u_face/tau_v_face: on-ocean stress prescribed at the C-grid faces is the momentum stress, uninterpolated.

A resting box forced only by a face stress on one u face must accelerate that face and no T-point stress is
needed; supplying the face stress together with the native stress is refused.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import surface_stress_faces  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean  # noqa: E402
from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
NY, NX = 8, 10


def _setup():
    grid, wall = create_regional_grid("latlon", n_lat=NY, n_lon=NX, lat_south=-4.0,
                                      lat_north=4.0, lon_west=200.0, lon_east=200.0 + NX)
    zc = create_z_star_from_thicknesses(np.array([20.0, 40.0, 80.0, 160.0]))
    st = rest_state_latlon_cgrid_ocean(grid, zc, land_mask_override=jnp.asarray(wall))
    cfg = LatLonCGridOceanConfig.from_flat(fix_eta_drift=False, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"), shortwave_penetration=None))
    return grid, zc, st, LatLonCGridOceanModel(grid, zc, cfg)


def test_face_stress_drives_its_own_face():
    grid, zc, st, model = _setup()
    nu, nv = st.u.data.shape[:2], st.v.data.shape[:2]
    j, i = NY // 2 + 1, NX // 2 + 1
    tu = jnp.zeros(nu).at[j, i].set(0.1)
    z = jnp.zeros_like(st.eta.data)
    sf = OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z, sw_down=None,
                             tau_u_face=tu, tau_v_face=jnp.zeros(nv))
    u = np.asarray(model.step(st, 1800.0, surface_forcing=sf).u.data)[..., 0]
    assert u[j, i] > 1e-6, u[j, i]                       # the forced face accelerates eastward
    u0 = np.asarray(model.step(st, 1800.0, surface_forcing=OceanSurfaceForcing(
        tau_x=z, tau_y=z, q_net=z, sw_down=None)).u.data)[..., 0]
    assert np.abs(u0).max() < 1e-12                       # control: no stress, no flow


def test_face_stress_refuses_the_native_stress_alongside():
    z = jnp.zeros((2, 2))
    forcing = SimpleNamespace(tau_x=z, tau_y=z, tau_u_face=z, tau_v_face=z,
                              tau_i_native=z, tau_j_native=z)
    with pytest.raises(ValueError, match="tau_u_face"):
        surface_stress_faces(forcing, jnp.float64, None, None, None)
