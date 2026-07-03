"""USDA texture triangle + van-Genuchten class params (soil_texture)."""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.soil_texture import (
    USDA_TEXTURES, SOIL_TEXTURE_VG, usda_texture_index, vg_params_from_index,
    wilting_field_capacity)


def test_triangle_corner_classes():
    sand = jnp.array([95.0, 40.0, 20.0, 10.0, 60.0])
    clay = jnp.array([2.0, 20.0, 60.0, 10.0, 30.0])
    names = [USDA_TEXTURES[int(i)] for i in usda_texture_index(sand, clay)]
    assert names[0] == "sand"             # 95/2
    assert names[1] == "loam"             # 40/20 (silt 40)
    assert names[2] == "clay"             # 20/60
    assert names[3] == "silt_loam"        # 10/10 (silt 80 -> folded)
    assert names[4] == "sandy_clay_loam"  # 60/30


def test_vg_params_monotone_with_clay():
    # finer texture -> lower K_sat, higher wilting point
    idx = usda_texture_index(jnp.array([95.0, 20.0]), jnp.array([2.0, 60.0]))
    vg = vg_params_from_index(idx)
    assert float(vg["K_sat"][0]) > float(vg["K_sat"][1])      # sand drains faster
    wp, fc = wilting_field_capacity(vg)
    assert float(wp[1]) > float(wp[0])                        # clay holds more water
    assert bool(jnp.all(fc > wp))                             # FC above WP everywhere
    assert bool(jnp.all((wp > 0) & (fc < 0.5)))               # physical range


def test_all_classes_have_vg():
    assert set(SOIL_TEXTURE_VG) == set(USDA_TEXTURES)
    for t in USDA_TEXTURES:
        assert SOIL_TEXTURE_VG[t]["theta_sat"] > SOIL_TEXTURE_VG[t]["theta_r"]


def test_index_shape_preserved():
    s = np.random.default_rng(0).uniform(0, 100, (5, 7))
    c = np.minimum(100 - s, np.random.default_rng(1).uniform(0, 60, (5, 7)))
    assert usda_texture_index(jnp.asarray(s), jnp.asarray(c)).shape == (5, 7)


def test_solid_thermal_from_texture():
    """Sand-rich solids conduct heat ~3x better than clay (Oleson 2013); heat
    capacity stays in the soil-solids band, clay slightly higher; bounded by the
    end members."""
    from legoesm.land.soil_texture import (
        soil_solid_conductivity, soil_solid_heat_capacity)
    k_sand = float(soil_solid_conductivity(80.0, 5.0))
    k_clay = float(soil_solid_conductivity(5.0, 80.0))
    assert 2.9 <= k_clay < k_sand <= 8.8          # sand conducts more, bounded
    c_sand = float(soil_solid_heat_capacity(80.0, 5.0))
    c_clay = float(soil_solid_heat_capacity(5.0, 80.0))
    assert 2.0e6 < c_sand < c_clay < 2.5e6        # capacity varies little, clay higher
    s = np.array([90.0, 10.0]); c = np.array([5.0, 85.0])
    k = np.asarray(soil_solid_conductivity(jnp.asarray(s), jnp.asarray(c)))
    assert k.shape == (2,) and k[0] > k[1]         # per-element ordering
    # no-mineral-data cell (ice/organic, %sand=%clay=0) -> loam fallback, never zero
    k0 = float(soil_solid_conductivity(0.0, 0.0))
    assert k0 > 2.9 and k0 == float(soil_solid_conductivity(40.0, 30.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
