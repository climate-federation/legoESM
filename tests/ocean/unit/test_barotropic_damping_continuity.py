"""The split-explicit sea level damping must reach the tracer transports.

On the lat-lon and MPAS lanes, with the default barotropic eta damping on, the sea level change of a step must
equal minus dt times the column divergence of the tracer mass fluxes in every
column, and a uniform salinity with no surface salt flux must stay uniform.
Before the damping was booked as a transport the two differed by centimetres
per step in a wind driven box and the uniform tracer drifted.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
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

DT = 1800.0
N_STEPS = 40


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _latlon_box(book):
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
                     S=st.S.replace(data=jnp.asarray(35.0 * mask3)))
    cfg = LatLonCGridOceanConfig.from_flat(store_mass_flux=True, physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"), lateral_mixing=LateralMixingConfig(scheme="none")),
        **({} if book is None else {"barotropic_diffusion_book_transport": book}))
    assert cfg.barotropic.barotropic_diffusion_alpha > 0.0          # the damping under test is on
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    taux = 0.08 * np.cos(np.deg2rad(lat * 8)) ** 2 * mask2
    sf = OceanSurfaceForcing(tau_x=jnp.asarray(-taux), tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(50.0 * mask2), sw_down=None)
    worst = 0.0
    for _ in range(N_STEPS):
        new = model.step(st, DT, surface_forcing=sf)
        div = np.asarray(divergence_cgrid(new.mass_flux_u.data, new.mass_flux_v.data, grid).sum(-1))
        resid = float(np.abs((np.asarray(new.eta.data - st.eta.data) + DT * div) * mask2).max())
        assert np.isfinite(resid), resid
        worst = max(worst, resid)
        st = new
    salt = np.asarray(st.S.data)
    return worst, float(np.abs(np.where(mask3, salt - 35.0, 0.0)).max())


def test_damping_booked_in_tracer_transports():
    worst, drift = _latlon_box(book=None)       # config default
    assert worst < 1e-12, worst
    assert drift < 35.0e-11, drift


def test_unbooked_switch_restores_old_behaviour():
    worst, _ = _latlon_box(book=False)
    assert worst > 1e-6, worst


# --- MPAS (explicit_substep) ---------------------------------------------

def _mpas_setup(book):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(subdivision_level=2)
    zc = create_ocean_z_star(n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0)
    st = rest_state_mpas_ocean(mesh, zc, T_water_init_C=20.0, T_deep=2.0,
                               S_uniform=35.0, H_max=500.0, land_lat_threshold=85.0)
    mask = np.asarray(st.land_mask.data)
    lat, lon = np.asarray(mesh.latCell), np.asarray(mesh.lonCell)
    bump = 0.5 * np.exp(-((lat - 0.3) ** 2 + (lon - 1.0) ** 2) / 0.2) * mask
    st = st._replace(eta=st.eta.replace(data=jnp.asarray(bump)))
    kw = {} if book is None else {"barotropic_diffusion_book_transport": book}
    cfg = MPASOceanConfig(A_h=1.0e3, K_h=1.0e2, A_v=1.0e-3, K_v=1.0e-4,
                          n_barotropic_substeps=5, **kw)
    assert cfg.barotropic_solver == "explicit_substep"
    assert cfg.barotropic_diffusion_alpha > 0.0
    return mesh, zc, st, cfg, mask


def _mpas_closure(book):
    from legoesm.core.operators_voronoi import divergence_cell
    from legoesm.ocean.dynamics.barotropic_mpas import barotropic_substeps_mpas

    mesh, zc, st, cfg, mask = _mpas_setup(book)
    dt, n_sub = 600.0, cfg.n_barotropic_substeps
    eta_avg, _, Hu_avg = barotropic_substeps_mpas(st, mesh, zc, cfg, dt / n_sub, n_sub)
    resid = (np.asarray(st.eta.data) - np.asarray(eta_avg)
             - dt * np.asarray(divergence_cell(Hu_avg, mesh))) * mask
    return float(np.abs(resid).max())


def test_mpas_damping_booked_in_barotropic_transport():
    assert _mpas_closure(book=None) < 1e-12


def test_mpas_unbooked_switch_restores_old_behaviour():
    assert _mpas_closure(book=False) > 1e-6


def test_mpas_uniform_salinity_stays_uniform():
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    mesh, zc, st, cfg, mask = _mpas_setup(book=None)
    model = MPASOceanModel(mesh, zc, cfg)
    for _ in range(10):
        st = model.step(st, 600.0)
    salt = np.asarray(st.S.data)
    assert np.all(np.isfinite(salt))
    assert np.abs(np.where(mask[:, None] > 0.5, salt - 35.0, 0.0)).max() < 35.0e-11
