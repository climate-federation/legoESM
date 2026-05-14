"""Unit tests for DINO initial conditions (Phase 2D, Appendix D).

Cross-checks against:
- Paper text Appendix D: T(z) ranges from ~24°C surface to ~4°C abyss;
  S(z) ranges from ~37 surface to ~35 abyss.
- Zenodo source (vopikamm/DINO@v0.2.0 MY_SRC/usrdef_istate.F90):
  meridional gradient interpolates between equatorial profile and
  *bottom* values at the poles (NOT surface values; paper eq D4 has
  a typo).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_initial_T_S,
    dino_S_profile_1d,
    dino_T_profile_1d,
)


# -------------------- 1D temperature profile -------------------------

def test_T_profile_surface_value_is_warm():
    T0 = float(dino_T_profile_1d(jnp.array(0.0)))
    # Paper IC has equatorial surface T ~24°C (matches restoring target
    # of T*_eq=27°C with finite restoring rate)
    assert 22.0 < T0 < 26.0


def test_T_profile_abyss_value_is_cold():
    T_deep = float(dino_T_profile_1d(jnp.array(4000.0)))
    # Should be ~4°C at the abyss (deep-water mass)
    assert 3.0 < T_deep < 5.0


def test_T_profile_thermocline_decrease():
    """Temperature must decrease monotonically with depth in the
    thermocline (top 1500 m)."""
    z = jnp.linspace(10.0, 1500.0, 50)
    T = dino_T_profile_1d(z)
    diffs = T[1:] - T[:-1]
    # All differences should be negative (T decreases with depth)
    assert bool(jnp.all(diffs < 0))


def test_T_profile_jit_compatible():
    import jax
    fn = jax.jit(dino_T_profile_1d)
    out = fn(jnp.linspace(0.0, 4000.0, 36))
    assert out.shape == (36,)
    assert bool(jnp.all(jnp.isfinite(out)))


# -------------------- 1D salinity profile ----------------------------

def test_S_profile_surface_value_is_high():
    S0 = float(dino_S_profile_1d(jnp.array(0.0)))
    # Paper IC has equatorial surface S ~36-37 g/kg (subtropical surface)
    assert 35.5 < S0 < 37.5


def test_S_profile_abyss_value():
    S_deep = float(dino_S_profile_1d(jnp.array(4000.0)))
    # Should be ~35.1 g/kg at the abyss (NADW/AABW value)
    assert 34.8 < S_deep < 35.5


def test_S_profile_finite_everywhere():
    z = jnp.linspace(0.0, 4000.0, 1000)
    S = dino_S_profile_1d(z)
    assert bool(jnp.all(jnp.isfinite(S)))


# -------------------- meridional gradient ----------------------------

def test_initial_T_at_equator_equals_1d_profile():
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.array([0.0])  # equator
    T, _ = dino_initial_T_S(lat, z.z_full_ref, cfg)
    T_1d = dino_T_profile_1d(-z.z_full_ref)
    assert bool(jnp.allclose(T[0], T_1d, atol=1e-9))


def test_initial_T_at_pole_is_isothermal():
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.array([cfg.lat_max_deg])
    T, _ = dino_initial_T_S(lat, z.z_full_ref, cfg)
    # Should be a CONSTANT column at the bottom value
    T_pole = np.asarray(T[0])
    assert T_pole.std() < 1e-9
    # And should equal T_1d(z_bottom)
    T_bot = float(dino_T_profile_1d(jnp.array(-float(z.z_full_ref[-1]))))
    assert T_pole[0] == pytest.approx(T_bot, abs=1e-9)


def test_initial_S_at_pole_is_isohaline():
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.array([-cfg.lat_max_deg])  # southern pole too
    _, S = dino_initial_T_S(lat, z.z_full_ref, cfg)
    S_pole = np.asarray(S[0])
    assert S_pole.std() < 1e-9


def test_initial_T_north_south_symmetric():
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.array([-30.0, 30.0])  # symmetric pair
    T, _ = dino_initial_T_S(lat, z.z_full_ref, cfg)
    assert bool(jnp.allclose(T[0], T[1], atol=1e-9))


def test_initial_T_decreases_from_equator_to_pole():
    """At any depth above the abyss, T should decrease as |φ| increases
    (more equatorial profile = warmer surface, polar profile = uniform 4°C).
    """
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lats = jnp.array([0.0, 20.0, 40.0, 60.0])
    T, _ = dino_initial_T_S(lats, z.z_full_ref, cfg)
    # Surface temperature must decrease equator → pole
    surface_T = T[:, 0]
    diffs = surface_T[1:] - surface_T[:-1]
    assert bool(jnp.all(diffs < 0))


def test_initial_T_S_shape_broadcasts_2d_lat():
    """Function should accept a 2D lat array (e.g., lat2d from a grid)."""
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat2d = jnp.array([[-30.0, 0.0, 30.0],
                       [-50.0, 10.0, 50.0]])
    T, S = dino_initial_T_S(lat2d, z.z_full_ref, cfg)
    assert T.shape == lat2d.shape + (cfg.n_levels,)
    assert S.shape == lat2d.shape + (cfg.n_levels,)


def test_initial_T_finite_no_NaN():
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.linspace(-cfg.lat_max_deg, cfg.lat_max_deg, 31)
    T, S = dino_initial_T_S(lat, z.z_full_ref, cfg)
    assert bool(jnp.all(jnp.isfinite(T)))
    assert bool(jnp.all(jnp.isfinite(S)))


def test_initial_T_S_jit_compatible():
    import jax
    cfg = DINOConfig()
    z = create_dino_z_star(cfg)
    lat = jnp.linspace(-cfg.lat_max_deg, cfg.lat_max_deg, 11)
    fn = jax.jit(lambda lat_in: dino_initial_T_S(lat_in, z.z_full_ref, cfg))
    T, S = fn(lat)
    assert T.shape == (11, cfg.n_levels)


def test_initial_T_at_pole_matches_T_bot_independent_of_n_levels():
    """The polar columns should equal the bottom-level value of the 1D
    profile, regardless of how many vertical levels are used."""
    cfg_a = DINOConfig(n_levels=20, k_th=19, a_cr=8.0)
    cfg_b = DINOConfig(n_levels=36)  # default
    z_a = create_dino_z_star(cfg_a)
    z_b = create_dino_z_star(cfg_b)
    lat = jnp.array([cfg_a.lat_max_deg])
    T_a, _ = dino_initial_T_S(lat, z_a.z_full_ref, cfg_a)
    T_b, _ = dino_initial_T_S(lat, z_b.z_full_ref, cfg_b)
    # Different n_levels means different deepest center; both bottom
    # values should be similar (~ 4°C) but not exactly equal
    assert abs(float(T_a[0, 0]) - float(T_b[0, 0])) < 1.0
