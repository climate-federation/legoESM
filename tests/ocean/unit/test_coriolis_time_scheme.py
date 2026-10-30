"""Kinetic energy of one baroclinic Coriolis rotation of a zonal flow.

The forward-backward rotation (u' with the old v', then v' with the new u')
adds energy of order (f dt)^2 to a zonal flow every step (Sielecki 1968),
1.2 percent per step here (20N to 32N, dt = 1800 s), which feeds zonal
geostrophic jets.
``coriolis_time_scheme="trapezoidal"`` (Crank-Nicolson rotation) leaves it
unchanged to the fixed-point and metric residuals, and still rotates the flow.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as oml  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    min_cell_to_uface, min_cell_to_vface)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0


def _ke_change(scheme):
    grid, wall = create_regional_grid("latlon", n_lat=12, n_lon=16, lat_south=20.0,
                                      lat_north=32.0, lon_west=140.0, lon_east=156.0)
    zc = create_z_star_from_thicknesses(np.full(6, 500.0))
    cfg = LatLonCGridOceanConfig.from_flat(coriolis_time_scheme=scheme)
    model = LatLonCGridOceanModel(grid, zc, cfg)
    st = rest_state_latlon_cgrid_ocean(grid, zc, land_mask_override=jnp.asarray(np.asarray(wall) > 0.5,
                                                                                    dtype=jnp.float64))
    h = compute_layer_thickness(st.eta.data, st.H_bathy.data, zc)
    hu = np.asarray(min_cell_to_uface(h)) * np.asarray(st.u_mask.data)[..., None]
    hv = np.asarray(min_cell_to_vface(h, model.grid)) * np.asarray(st.v_mask.data)[..., None]
    zmid = np.arange(6) * 500.0 + 250.0
    u0 = 0.5 * np.cos(np.pi * zmid / 3000.0)[None, None, :] * (hu > 0)     # baroclinic, zero depth mean
    u0 = jnp.asarray(u0)
    v0 = jnp.zeros(hv.shape)
    u1, v1 = oml._forward_backward_coriolis_3d(u0, v0, DT, model.grid, model.z_coord, model.config, st.u_mask.data,
                                               st.v_mask.data, st.land_mask.data, st.eta.data,
                                               st.H_bathy.data)
    Au = (np.asarray(model.grid.dx_u) * np.asarray(model.grid.dy_u))[..., None]
    Av = (np.asarray(model.grid.dx_v) * np.asarray(model.grid.dy_v))[..., None]

    def ke(u, v):
        return float(np.sum(hu * Au * np.asarray(u) ** 2) + np.sum(hv * Av * np.asarray(v) ** 2))

    return (ke(u1, v1) - ke(u0, v0)) / ke(u0, v0), float(np.abs(np.asarray(v1)).max())


def test_forward_backward_adds_energy_to_zonal_flow():
    dke, vmax = _ke_change("forward_backward")
    assert dke > 4e-3, dke                    # measured 1.2e-2
    assert vmax > 0.03, vmax


def test_trapezoidal_conserves_energy_and_rotates():
    dke, vmax = _ke_change("trapezoidal")
    assert abs(dke) < 1e-5, dke              # measured 1.3e-6
    assert vmax > 0.03, vmax                  # dt f u is about 0.06 m/s: the flow still turns


def test_unknown_coriolis_time_scheme_raises():
    grid, _ = create_regional_grid("latlon", n_lat=4, n_lon=4, lat_south=20.0, lat_north=24.0,
                                   lon_west=140.0, lon_east=144.0)
    zc = create_z_star_from_thicknesses(np.full(2, 500.0))
    with pytest.raises(ValueError):
        LatLonCGridOceanModel(grid, zc, LatLonCGridOceanConfig.from_flat(coriolis_time_scheme="leapfrog"))
