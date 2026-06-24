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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
