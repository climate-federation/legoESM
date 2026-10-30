"""The open-boundary volume correction can hold a prescribed net inflow.

A nested child must follow its parent's mass changes, so the correction that
removes the net flux through the open faces (CROCO obc_volcons) takes an
optional target instead of zero.  Checked twice: the corrected faces carry
exactly the target, and the interior volume of a resting box with three open
sides grows by the target times the elapsed time, while without a target it
stays put.  The second check is repeated with the baroclinic and tracer
condition on and an uneven boundary sea level (data +1 m on the west side,
-1 m on the north, interior 0.5 m above the data), where large inflows and
outflows nearly cancel: the correction must then measure the face depth as
H + eta, as the model's transports do.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.dynamics.open_boundary_latlon import (  # noqa: E402
    OpenBoundaryData, apply_barotropic_open_boundary)
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    make_partial_cell_latlon, rest_state_latlon_cgrid_ocean)
from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig  # noqa: E402
from legoesm.ocean.sponge import SpongeForcing  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0
N_STEPS = 12
NY, NX = 10, 14
Q = 1.0e6                       # 1 Sv into the domain


def _box():
    grid, wall = create_regional_grid("latlon", n_lat=NY, n_lon=NX, lat_south=-5.0,
                                      lat_north=5.0, lon_west=200.0, lon_east=200.0 + NX)
    ocean = np.asarray(wall) > 0.5
    ocean[1:-1, 0] = True       # west, south and north open; east a wall
    ocean[0, 1:-1] = True
    ocean[-1, 1:-1] = True
    zc = create_z_star_from_thicknesses(np.full(4, 1000.0))
    zc_pc, H, lm = make_partial_cell_latlon(zc, np.where(ocean, 4000.0, 0.0), ocean.astype(float),
                                            smoothing_passes=0)
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H), H_max=float(zc.H_max))
    lm = jnp.asarray(lm)
    st = st._replace(T=st.T.replace(data=jnp.full(st.T.data.shape, 10.0) * lm[..., None]),
                     S=st.S.replace(data=jnp.full(st.S.data.shape, 35.0) * lm[..., None]))
    return grid, zc_pc, st


def _data(target):
    return OpenBoundaryData(west_ubar=jnp.zeros(NY + 2), west_zeta=jnp.zeros(NY + 2),
                            south_vbar=jnp.zeros(NX + 2), south_zeta=jnp.zeros(NX + 2),
                            north_vbar=jnp.zeros(NX + 2), north_zeta=jnp.zeros(NX + 2),
                            target_net_flux=None if target is None else jnp.asarray(target))


def _net_inflow(U, V, H, eta, grid, u_mask, v_mask):
    """Net volume flux into the domain through the open faces, total depth H + eta by the min rule."""
    H = H + eta
    dy_cell = 0.5 * jnp.asarray(grid.dy)
    R, dlon = float(grid.radius), float(grid.dlon)
    lat_v = jnp.asarray(grid.lat_v)
    west = jnp.sum(U[:, 1] * jnp.minimum(H[:, 0], H[:, 1]) * u_mask[:, 1] * dy_cell)
    south = jnp.sum(V[1] * jnp.minimum(H[0], H[1]) * v_mask[1]) * R * jnp.cos(lat_v[1]) * dlon
    nv = V.shape[0] - 2
    north = jnp.sum(V[-2] * jnp.minimum(H[-1], H[-2]) * v_mask[-2]) * R * jnp.cos(lat_v[nv]) * dlon
    return float(west + south - north)


def test_corrected_faces_carry_the_target():
    grid, _, st = _box()
    rng = np.random.default_rng(0)
    eta = jnp.asarray(0.05 * rng.standard_normal(st.eta.data.shape)) * st.land_mask.data
    U = jnp.asarray(0.1 * rng.standard_normal(st.u_mask.data.shape)) * st.u_mask.data
    V = jnp.asarray(0.1 * rng.standard_normal(st.v_mask.data.shape)) * st.v_mask.data
    H = st.H_bathy.data
    for target in (None, Q, -2.5 * Q):
        U2, V2, eta2 = apply_barotropic_open_boundary(
            eta, U, V, eta, U, V, H, grid, 60.0, 9.81, _data(target),
            st.u_mask.data, st.v_mask.data, st.land_mask.data)
        net = _net_inflow(U2, V2, H, eta2, grid, st.u_mask.data, st.v_mask.data)
        want = 0.0 if target is None else target
        assert abs(net - want) <= 1e-10 * Q, (target, net)


def test_interior_volume_follows_the_target():
    """After the first step (which starts from rest), every step adds the
    target times dt to the interior, at the default sea level damping."""
    grid, zc_pc, st0 = _box()
    cfg = LatLonCGridOceanConfig.from_flat(fix_eta_drift=False, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"), shortwave_penetration=None))
    assert cfg.barotropic.barotropic_diffusion_alpha > 0.0     # the damping must not leak through open faces
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    z = jnp.zeros_like(st0.eta.data)
    sf = OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z, sw_down=None)
    area = np.asarray(grid.area)[1:-1, 1:-1]
    nl = st0.T.data.shape[-1]
    baroclinic = {}
    for side, n in (("west", NY + 2), ("south", NX + 2), ("north", NX + 2)):
        baroclinic.update({f"{side}_T": jnp.full((nl, n), 10.0), f"{side}_S": jnp.full((nl, n), 35.0),
                           f"{side}_u3": jnp.zeros((nl, n)), f"{side}_t3": jnp.zeros((nl, n + 1))})
    uneven = _data(Q)._replace(west_zeta=jnp.full(NY + 2, 1.0), north_zeta=jnp.full(NX + 2, -1.0),
                               tau_in=jnp.asarray(86400.0), tau_out=jnp.asarray(360 * 86400.0), **baroclinic)
    st_up = st0._replace(eta=st0.eta.replace(data=st0.eta.data + 0.5 * st0.land_mask.data))
    for target, ob, start in ((None, _data(None), st0), (Q, _data(Q), st0), (Q, uneven, st_up)):
        sponge = SpongeForcing(gamma=z, T_ref=st0.T.data, S_ref=st0.S.data, open_boundary=ob)
        st = start
        vols = [float(np.sum(np.asarray(st.eta.data)[1:-1, 1:-1] * area))]
        for _ in range(N_STEPS):
            st = model.step(st, DT, surface_forcing=sf, sponge=sponge)
            vols.append(float(np.sum(np.asarray(st.eta.data)[1:-1, 1:-1] * area)))
        rate = np.diff(vols)[3:] / DT                 # steps 4 to N_STEPS
        want = 0.0 if target is None else Q
        assert np.abs(rate - want).max() < 1e-4 * Q, (target, rate / Q)


def test_baroclinic_part_keeps_the_depth_mean_over_a_shelf():
    """The baroclinic condition must keep the depth mean of the open-face
    velocity over the wet part of each face.  A 50 m shelf on the open west
    side (a partial second level) with data of 0.3 m/s in the top two levels
    and none below: weighted over all reference levels, the face carried 1.2
    to 1.3 times the target after a step."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface, min_cell_to_vface
    from legoesm.ocean.vertical import compute_layer_thickness

    grid, wall = create_regional_grid("latlon", n_lat=NY, n_lon=NX, lat_south=-5.0,
                                      lat_north=5.0, lon_west=200.0, lon_east=200.0 + NX)
    ocean = np.asarray(wall) > 0.5
    ocean[1:-1, 0] = True
    ocean[0, 1:-1] = True
    ocean[-1, 1:-1] = True
    H = np.where(ocean, 4000.0, 0.0)
    H[:, :4] = np.where(ocean[:, :4], 50.0, 0.0)
    zc = create_z_star_from_thicknesses(np.array([20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1200.0, 1600.0]))
    zc_pc, H2, lm = make_partial_cell_latlon(zc, H, ocean.astype(float), smoothing_passes=0)
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H2), H_max=float(zc.H_max))
    lmj = jnp.asarray(lm)
    st = st._replace(T=st.T.replace(data=jnp.full(st.T.data.shape, 10.0) * lmj[..., None]),
                     S=st.S.replace(data=jnp.full(st.S.data.shape, 35.0) * lmj[..., None]))
    nl = st.T.data.shape[-1]
    kw = {}
    for side, n in (("west", NY + 2), ("south", NX + 2), ("north", NX + 2)):
        u3 = jnp.zeros((nl, n)).at[:2].set(0.3) if side == "west" else jnp.zeros((nl, n))
        kw.update({f"{side}_T": jnp.full((nl, n), 10.0), f"{side}_S": jnp.full((nl, n), 35.0),
                   f"{side}_u3": u3, f"{side}_t3": jnp.zeros((nl, n + 1))})
    ob = _data(Q)._replace(tau_in=jnp.asarray(86400.0), tau_out=jnp.asarray(360 * 86400.0), **kw)
    cfg = LatLonCGridOceanConfig.from_flat(fix_eta_drift=False, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"), shortwave_penetration=None))
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    z = jnp.zeros_like(st.eta.data)
    sf = OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z, sw_down=None)
    sponge = SpongeForcing(gamma=z, T_ref=st.T.data, S_ref=st.S.data, open_boundary=ob)
    dy_cell = 0.5 * jnp.asarray(grid.dy)
    lat_v = jnp.asarray(grid.lat_v)
    L = float(grid.radius) * float(grid.dlon)
    for _ in range(4):
        st = model.step(st, DT, surface_forcing=sf, sponge=sponge)
        h = compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc)
        U = (st.u.data * min_cell_to_uface(h)).sum(-1)
        V = (st.v.data * min_cell_to_vface(h, grid)).sum(-1)
        net = float(jnp.sum(U[:, 1] * dy_cell) + jnp.sum(V[1]) * L * jnp.cos(lat_v[1])
                    - jnp.sum(V[-2]) * L * jnp.cos(lat_v[-2]))
        assert abs(net - Q) < 1e-3 * Q, net / Q
