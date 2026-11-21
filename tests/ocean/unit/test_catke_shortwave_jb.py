"""core-0027: CATKE's surface buoyancy flux leaves out the penetrating shortwave, as Oceananigans' CATKE does when
the shortwave is a body forcing (NumericalEarth TwoColorRadiation: shortwave_radiative_forcing returns zero for the
top flux). One column, Wright EOS, 50 levels of 5 m."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig, VerticalMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.k_profiles import compute_vertical_K_profiles  # noqa: E402
from legoesm.ocean.scm import make_ocean_column_state  # noqa: E402
from legoesm.ocean.state import OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import create_z_star_from_thicknesses  # noqa: E402

NLEV = 50
SHP = (1, 1, 1)


def _column():
    dz = jnp.full(NLEV, 5.0)
    zc = jnp.cumsum(dz) - 2.5
    T = 26.0 - 6.0 * jnp.clip((zc - 40.0) / 100.0, 0.0, 1.0)          # 40 m mixed layer over a thermocline
    S = jnp.full(NLEV, 35.0)
    zcol = create_z_star_from_thicknesses(dz)
    st = make_ocean_column_state(NLEV, T_profile=T, S_profile=S, z_coord=zcol)
    u = 0.3 * jnp.exp(-zc / 30.0)
    st = st._replace(u=st.u.replace(data=u.reshape(1, 1, 1, NLEV)))
    return st, zcol


def _K(mode, q_net, sw_down, e0=None):
    st, zcol = _column()
    cfg = OceanPhysicsConfig(vertical_mixing=VerticalMixingConfig(
        scheme="catke", catke=CATKEConfig(surface_buoyancy_flux_shortwave=mode)))
    sfc = OceanSurfaceForcing(q_net=jnp.full(SHP, q_net), sw_down=None if sw_down is None else jnp.full(SHP, sw_down),
                              tau_x=jnp.full(SHP, 0.05), tau_y=jnp.zeros(SHP), freshwater=jnp.zeros(SHP))
    e = jnp.full((1, 1, 1, NLEV - 1), 1e-5) if e0 is None else e0
    return compute_vertical_K_profiles(st, zcol, sfc, cfg, tke_old=e, dt_tke=1800.0, return_tke=True)


def test_default_is_exclude():
    assert CATKEConfig().surface_buoyancy_flux_shortwave == "exclude"


def test_exclude_equals_include_with_non_solar_q_net():
    q, sw = 300.0, 500.0
    a = _K("exclude", q, sw)
    b = _K("include", q - sw, sw)
    for x, y in zip(a, b):
        assert jnp.array_equal(x, y)


def test_no_sw_down_leaves_jb_unchanged():
    a = _K("exclude", -150.0, None)
    b = _K("include", -150.0, None)
    for x, y in zip(a, b):
        assert jnp.array_equal(x, y)


def test_daytime_non_solar_cooling_drives_more_mixing():
    # midday: q_net = +300 W m-2 with 500 W m-2 of shortwave, so the non-solar flux is -200 (cooling)
    K_ex, _, e_ex = _K("exclude", 300.0, 500.0)
    K_in, _, e_in = _K("include", 300.0, 500.0)
    upper = slice(0, 6)                                                  # interfaces above 30 m
    assert float(jnp.mean(K_ex[0, 0, 0, upper])) > float(jnp.mean(K_in[0, 0, 0, upper]))
    assert float(jnp.mean(e_ex[0, 0, 0, upper])) > float(jnp.mean(e_in[0, 0, 0, upper]))


def test_sw_down_without_q_net_is_skipped():
    st, zcol = _column()
    for mode in ("exclude", "include"):
        cfg = OceanPhysicsConfig(vertical_mixing=VerticalMixingConfig(
            scheme="catke", catke=CATKEConfig(surface_buoyancy_flux_shortwave=mode)))
        sfc = OceanSurfaceForcing(sw_down=jnp.full(SHP, 400.0), tau_x=jnp.full(SHP, 0.05), tau_y=jnp.zeros(SHP))
        K, A, e = compute_vertical_K_profiles(st, zcol, sfc, cfg, tke_old=jnp.full((1, 1, 1, NLEV - 1), 1e-5),
                                              dt_tke=1800.0, return_tke=True)
        assert bool(jnp.all(jnp.isfinite(K)))


def test_jit_and_gradient_through_sw_down():
    st, zcol = _column()
    cfg = OceanPhysicsConfig(vertical_mixing=VerticalMixingConfig(scheme="catke", catke=CATKEConfig()))

    def f(sw):
        sfc = OceanSurfaceForcing(q_net=jnp.full(SHP, 300.0), sw_down=jnp.full(SHP, sw), tau_x=jnp.full(SHP, 0.05),
                                  tau_y=jnp.zeros(SHP), freshwater=jnp.zeros(SHP))
        K, _, _ = compute_vertical_K_profiles(st, zcol, sfc, cfg, tke_old=jnp.full((1, 1, 1, NLEV - 1), 1e-5),
                                              dt_tke=1800.0, return_tke=True)
        return jnp.sum(K[0, 0, 0, :6])
    g = jax.jit(jax.grad(f))(500.0)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0          # more shortwave left out of Jb: more convective mixing
    assert float(jax.jit(f)(500.0)) == pytest.approx(float(f(500.0)), rel=1e-12)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        _K("half", 0.0, 100.0)
