"""The learned arms consume the prescribed planes on the driver path.

Pins the gridded column-MLP step (cubed sphere AND MPAS adapters) and the
lat-lon SFNO coupling: a flag-on network refuses a missing plane by name,
its output responds to the sensible-flux plane and to the land fraction,
a flag-off network is byte-identical to the keyword-free call; plus the
shared plane builder and the lat-lon classical model's policy-freeze names.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics import neural_physics as nph
from legoesm.core.grid_adapters import make_adapter
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.training import inert_params
from legoesm.training.model_registry import sfno_extra_input_channels
from legoesm.training.sfno_dycore_coupling import (
    SFNOPhysics,
    make_sfno_step_unified_latlon,
)
from legoesm.training.trainable_params import TrainablePhysicsParams

from tests.unit.test_sfno_latlon_coupling import N_LAT_LL, N_LON_LL, _weights

NLEV = 3


# ---------------------------------------------------------------- column MLP

@pytest.fixture(scope="module", params=["cubed_sphere", "mpas"])
def adapter(request):
    grid = (create_cubed_sphere(4) if request.param == "cubed_sphere"
            else create_voronoi_mesh(3, lloyd_iterations=8))
    return make_adapter(grid)


def _net(adapter, **flags):
    net = nph.build_column_physics(
        NLEV, hidden_dim=8, n_layers=2, key=jax.random.PRNGKey(0),
        n_columns=adapter.ncol, **flags)
    # the last layer is zero-initialised (exactly-zero output): give it
    # small random weights so the inputs can reach the output.
    w = 0.05 * jax.random.normal(jax.random.PRNGKey(1), net.layers[-1].weight.shape)
    return eqx.tree_at(lambda m: m.layers[-1].weight, net, w)


def _positional(adapter):
    s2 = adapter.shape_2d
    s3 = (*s2, NLEV)
    z2 = jnp.zeros(s2)
    return (jnp.full(s3, 280.0), jnp.full(s2, 1.0e5), jnp.full(s3, 5e-3),
            jnp.zeros(s3), jnp.zeros(s3),
            jnp.zeros(s3), jnp.zeros(s3),                 # u, v
            jnp.full(s2, 290.0), z2,                      # sst, sic
            jnp.full(s2, 0.3), z2,                        # lat, lon
            jnp.asarray(100.0), jnp.asarray(43200.0), jnp.asarray(600.0),
            jnp.ones(s2), jnp.asarray(1361.0), None, None,
            jnp.zeros(s3), z2, z2, z2, z2, z2)


def _planes(adapter, shf=20.0, land=0.3):
    s2 = adapter.shape_2d
    return dict(land_frac=jnp.full(s2, land), phis=jnp.full(s2, 500.0 * 9.8),
                sfc_taux_override=jnp.full(s2, -0.05),
                sfc_tauy_override=jnp.full(s2, 0.02),
                sfc_shflx_override=jnp.full(s2, shf),
                sfc_lhflx_override=jnp.full(s2, 80.0),
                sfc_sw_up=jnp.full(s2, 40.0), sfc_lw_up=jnp.full(s2, 390.0))


def test_flag_off_network_ignores_the_planes(adapter):
    step = nph.make_neural_step_unified(_net(adapter), adapter)
    a, _ = step(True, *_positional(adapter))
    b, _ = step(True, *_positional(adapter), **_planes(adapter))
    assert bool(jnp.array_equal(a.dT_dt, b.dT_dt))


def test_flag_on_network_names_the_missing_plane(adapter):
    step = nph.make_neural_step_unified(
        _net(adapter, spatial_embedding=True, era5_surface_fluxes=True), adapter)
    planes = _planes(adapter)
    for key in ("land_frac", "phis") + nph.SFC_FLUX_STEP_UNIFIED_KEYS:
        short = {k: v for k, v in planes.items() if k != key}
        with pytest.raises(ValueError, match=key):
            step(True, *_positional(adapter), **short)


def test_planes_reach_the_network_output(adapter):
    step = nph.make_neural_step_unified(
        _net(adapter, spatial_embedding=True, era5_surface_fluxes=True), adapter)
    base, _ = step(True, *_positional(adapter), **_planes(adapter))
    assert base.dT_dt.shape == (*adapter.shape_2d, NLEV)
    assert bool(jnp.all(jnp.isfinite(base.dT_dt)))
    shf, _ = step(True, *_positional(adapter), **_planes(adapter, shf=120.0))
    land, _ = step(True, *_positional(adapter), **_planes(adapter, land=1.0))
    assert float(jnp.max(jnp.abs(shf.dT_dt - base.dT_dt))) > 0.0
    assert float(jnp.max(jnp.abs(land.dT_dt - base.dT_dt))) > 0.0


# ---------------------------------------------------------------- SFNO lat-lon

@pytest.fixture(scope="module")
def sfno_setup():
    gauss = create_gaussian_grid(10)
    n_ch = 4 * NLEV + 2
    extra = sfno_extra_input_channels(spatial_embedding=True, era5_surface_fluxes=True)
    assert extra == 7
    sfno = SFNO(SFNOConfig(in_channels=n_ch + extra, out_channels=n_ch,
                           embed_dim=8, n_blocks=1, residual_prediction=False),
                gauss, key=jax.random.PRNGKey(0))
    ph = SFNOPhysics(sfno=sfno, grid=gauss, nlev=NLEV,
                     spatial_embedding=True, era5_surface_fluxes=True)
    ll_lat = jnp.linspace(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, N_LAT_LL)
    ll_lon = jnp.linspace(0.0, 2 * np.pi, N_LON_LL, endpoint=False)
    w_ll2g = _weights(ll_lat, ll_lon, gauss.lat, gauss.lon,
                      (int(gauss.n_lat), int(gauss.n_lon)))
    w_g2ll = _weights(gauss.lat, gauss.lon, ll_lat, ll_lon, (N_LAT_LL, N_LON_LL))
    return ph, w_ll2g, w_g2ll


def _sfno_call(step, **kwargs):
    s2 = (N_LAT_LL, N_LON_LL)
    s3 = (*s2, NLEV)
    z2 = jnp.zeros(s2)
    return step(
        True, jnp.full(s3, 280.0), jnp.full(s2, 1.0e5), jnp.full(s3, 5e-3),
        jnp.zeros(s3), jnp.zeros(s3), jnp.zeros(s3), jnp.zeros(s3),
        z2 + 290.0, z2, z2, z2, jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(300.0), jnp.ones(s2), jnp.asarray(1361.0), None, None,
        jnp.zeros(s3), z2, z2, z2, z2, z2, **kwargs)


def _sfno_planes(shf=20.0):
    s2 = (N_LAT_LL, N_LON_LL)
    return dict(land_frac=jnp.full(s2, 0.4), phis=jnp.full(s2, 300.0 * 9.8),
                sfc_taux_override=jnp.full(s2, -0.05),
                sfc_tauy_override=jnp.full(s2, 0.02),
                sfc_shflx_override=jnp.full(s2, shf),
                sfc_lhflx_override=jnp.full(s2, 80.0),
                sfc_sw_up=jnp.full(s2, 40.0), sfc_lw_up=jnp.full(s2, 390.0))


def test_sfno_latlon_consumes_the_planes(sfno_setup):
    ph, w_ll2g, w_g2ll = sfno_setup
    step = make_sfno_step_unified_latlon(ph, w_ll2g, w_g2ll)
    out, held = _sfno_call(step, **_sfno_planes())
    assert out.dT_dt.shape == (N_LAT_LL, N_LON_LL, NLEV)
    assert bool(jnp.all(jnp.isfinite(out.dT_dt)))
    assert len(held) == 6
    with pytest.raises(ValueError, match="sfc_shflx_override"):
        planes = _sfno_planes(); planes.pop("sfc_shflx_override")
        _sfno_call(step, **planes)
    with pytest.raises(ValueError, match="land_frac"):
        planes = _sfno_planes(); planes.pop("land_frac")
        _sfno_call(step, **planes)
    # the decoder is random here (not the zero-init the trainer ships), so
    # the flux plane must move the tendency
    other, _ = _sfno_call(step, **_sfno_planes(shf=200.0))
    assert float(jnp.max(jnp.abs(other.dT_dt - out.dT_dt))) > 0.0


def test_sfno_channel_count_mismatch_is_refused():
    gauss = create_gaussian_grid(10)
    n_ch = 4 * NLEV + 2
    sfno = SFNO(SFNOConfig(in_channels=n_ch, out_channels=n_ch, embed_dim=8,
                           n_blocks=1, residual_prediction=False),
                gauss, key=jax.random.PRNGKey(0))
    ph = SFNOPhysics(sfno=sfno, grid=gauss, nlev=NLEV, spatial_embedding=True)
    s2 = (int(gauss.n_lat), int(gauss.n_lon))
    with pytest.raises(Exception):  # the encoder's width no longer matches
        ph(jnp.full((*s2, NLEV), 280.0), jnp.zeros((*s2, NLEV)),
           jnp.zeros((*s2, NLEV)), jnp.full((*s2, NLEV), 5e-3),
           jnp.full(s2, 1e5), jnp.zeros(s2), jnp.asarray(300.0),
           land_frac=jnp.zeros(s2))


# ---------------------------------------------------------------- shared planes

def test_prescribed_surface_planes_follow_the_slice():
    from legoesm.training.era5_to_state import ERA5Slice
    from legoesm.training.scale_build import prescribed_surface_planes

    grid = create_gaussian_grid(8)
    n_lat, n_lon = 9, 12
    lat = np.deg2rad(np.linspace(90.0, -90.0, n_lat))
    lon = np.deg2rad(np.linspace(0.0, 330.0, n_lon))
    z3 = np.zeros((n_lat, n_lon, 3), np.float32)
    two = np.full((n_lat, n_lon), 2.0, np.float32)
    base = dict(T=z3, u=z3, v=z3, q=z3, p_s=two, sst=two, phis=two, lat=lat,
                lon=lon, plev_Pa=np.array([300.0, 500.0, 850.0]))
    assert prescribed_surface_planes(ERA5Slice(**base), grid) == {}
    planes = prescribed_surface_planes(
        ERA5Slice(**base, sfc_shf=two, sfc_lhf=two, sfc_tau_x=two,
                  sfc_tau_y=-two, sfc_sw_up=two, sfc_sw_down=two,
                  sfc_lw_up=two, land_frac=1.5 * two), grid)
    assert set(planes) == {"sfc_shf", "sfc_lhf", "sfc_tau_x", "sfc_tau_y",
                           "sfc_sw_up", "sfc_sw_down", "sfc_lw_up", "land_frac"}
    assert planes["sfc_tau_y"].shape == (int(grid.n_lat), int(grid.n_lon))
    assert float(jnp.max(planes["sfc_tau_y"])) == -2.0
    assert float(jnp.max(planes["land_frac"])) == 1.0


# ---------------------------------------------------------------- policy freeze

def test_latlon_classical_surface_leaves_are_frozen_by_policy():
    names = inert_params.prescribed_surface_frozen_names(
        TrainablePhysicsParams.from_defaults())
    got = {n.split("['")[1].rstrip("']") for n in names}
    assert got == {"C_H", "C_E", "albedo_ice", "albedo_ocean"}
