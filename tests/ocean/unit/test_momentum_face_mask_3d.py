"""Velocity at faces below the shallower seafloor of a partial-cell ridge.

Such a face has zero thickness at that level and is closed for the tracers,
but with the 2-D face masks the step leaves a velocity there (it receives the
depth-uniform increments and nothing damps it), which the horizontal momentum
stencils then read.  ``momentum_face_mask="3d"`` keeps it at zero, as NEMO's
per-level umask/vmask do.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402
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
N_STEPS = 20


def _ridge_box():
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
    st = st._replace(T=st.T.replace(data=jnp.asarray(T0)))
    taux = 0.08 * np.cos(np.deg2rad(lat * 8)) ** 2 * mask2
    sf = OceanSurfaceForcing(tau_x=jnp.asarray(-taux), tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(50.0 * mask2), sw_down=None)
    um3, vm3 = (np.asarray(m, dtype=float) for m in compute_face_masks_3d(zc_pc.is_active, grid))
    buried_u = (np.asarray(st.u_mask.data)[..., None] > 0) & (um3 == 0)
    buried_v = (np.asarray(st.v_mask.data)[..., None] > 0) & (vm3 == 0)
    return grid, zc_pc, st, sf, buried_u, buried_v


def _run(face_mask):
    grid, zc_pc, st, sf, bu, bv = _ridge_box()
    assert bu.sum() > 0 and bv.sum() > 0                          # the ridge makes buried faces
    cfg = LatLonCGridOceanConfig.from_flat(momentum_face_mask=face_mask, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"), lateral_mixing=LateralMixingConfig(scheme="none")))
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    for _ in range(N_STEPS):
        st = model.step(st, DT, surface_forcing=sf)
    u, v = np.asarray(st.u.data), np.asarray(st.v.data)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    return np.abs(u[bu]).max(), np.abs(v[bv]).max()


def test_buried_faces_hold_velocity_with_2d_mask():
    bu, bv = _run("2d")
    assert max(bu, bv) > 1e-4, (bu, bv)


def test_buried_faces_stay_zero_with_3d_mask():
    bu, bv = _run("3d")
    assert bu == 0.0 and bv == 0.0, (bu, bv)


def test_unknown_face_mask_raises():
    grid, zc_pc, *_ = _ridge_box()
    with pytest.raises(ValueError, match="momentum_face_mask"):
        LatLonCGridOceanModel(grid, zc_pc, LatLonCGridOceanConfig.from_flat(momentum_face_mask="3D"))
