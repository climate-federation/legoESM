"""A resting ocean over a partial-cell ridge must stay at rest.

Horizontally uniform T(z), S(z), no forcing, Wright in-situ EOS.  Neighbouring
partial bottom cells have centres at different depths; the default
pgf_scheme="adcroft" shifts their pressures to a common depth linearly, which is
not exact for an in-situ density that is curved in z, and the error spins up
bottom currents of metres per second in two days (measured 4.0 m/s in this
ridge box).  The density Jacobian of Shchepetkin and McWilliams (2003),
pgf_scheme="smc03" (CROCO's pressure gradient), leaves 3.2 cm/s.  The bounds
(1 m/s, 5 cm/s) only separate the two schemes; they are not an accuracy claim.
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
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0
N_STEPS = 96          # 2 days


def _rest_max_speed(pgf_scheme):
    grid, wall = create_regional_grid("latlon", n_lat=12, n_lon=16, lat_south=-6.0,
                                      lat_north=6.0, lon_west=190.0, lon_east=206.0)
    lon = np.broadcast_to(np.linspace(189.5, 206.5, 18)[None, :], (14, 18))
    ocean = np.asarray(wall) > 0.5
    H = np.where(ocean, 4000.0 - 2500.0 * np.exp(-((lon - 198.0) / 2.0) ** 2), 0.0)   # a steep ridge
    dz = np.geomspace(10.0, 700.0, 10)
    dz *= 4000.0 / dz.sum()
    zc = create_z_star_from_thicknesses(dz)
    zc_pc, H2, lm = make_partial_cell_latlon(zc, H, ocean.astype(float))
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H2), H_max=float(zc.H_max))
    h0 = np.asarray(compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc))
    mask3 = (h0 > 0) & ocean[..., None] & (np.asarray(lm) > 0.5)[..., None]
    zmid = np.cumsum(dz) - dz / 2
    T0 = (4.0 + 24.0 * np.exp(-zmid / 200.0))[None, None, :] * mask3
    S0 = (34.6 + 0.6 * np.exp(-zmid / 300.0))[None, None, :] * mask3
    st = st._replace(T=st.T.replace(data=jnp.asarray(T0)), S=st.S.replace(data=jnp.asarray(S0)))
    z = jnp.zeros_like(st.eta.data)
    sf = OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z, sw_down=None)
    cfg = LatLonCGridOceanConfig.from_flat(pgf_scheme=pgf_scheme, eos="wright", physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"), lateral_mixing=LateralMixingConfig(scheme="none"),
        shortwave_penetration=None))
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    for _ in range(N_STEPS):
        st = model.step(st, DT, surface_forcing=sf)
    return float(max(jnp.abs(st.u.data).max(), jnp.abs(st.v.data).max()))


def test_adcroft_pgf_spins_up_a_resting_ocean():
    assert _rest_max_speed("adcroft") > 1.0


def test_smc03_pgf_keeps_a_resting_ocean_at_rest():
    assert _rest_max_speed("smc03") < 0.05
