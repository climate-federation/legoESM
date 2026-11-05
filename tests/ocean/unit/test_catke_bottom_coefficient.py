"""core-0026: CATKE bottom-distance coefficient as in Oceananigans (docs/92).

1. The default CATKEConfig.c_bottom_shear is 0.967, Oceananigans CATKEMixingLength.Cᵇ (main, 8e4f98d).
2. It gives the near-bottom similarity constant C_b (c_lo_u^3 / c_lo_diss)^(1/4) = 0.40 to 3 decimals, the value
   the Oceananigans docstring states, and the surface coefficient gives 0.47 (Wagner et al. 2025, Table 4).
3. In the stable length, the bottom limit acts as c_bottom_shear times the height above bottom: where that is the
   smallest of the three limits, _stable_length returns exactly 0.967 * hab.
4. Far from the bottom (hab large) the stable length does not depend on c_bottom_shear (bit for bit).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import catke
from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_default_matches_oceananigans():
    assert CATKEConfig().c_bottom_shear == 0.967


def test_similarity_constants():
    c = CATKEConfig()
    base = (c.c_lo_u ** 3 / c.c_lo_diss) ** 0.25
    assert round(c.c_bottom_shear * base, 3) == 0.400
    assert round(c.c_surface_shear * base, 2) == 0.47


def test_bottom_limit_value():
    c = CATKEConfig()
    w_star = jnp.asarray([0.05])
    N2 = jnp.asarray([1e-8])            # w*/N = 500 m, not limiting
    depth = jnp.asarray([400.0])         # surface limit 452 m
    hab = jnp.asarray([10.0])            # bottom limit 9.67 m
    ell = catke._stable_length(w_star, N2, depth, hab, c.c_surface_shear, c.c_bottom_shear)
    np.testing.assert_allclose(np.asarray(ell), 0.967 * 10.0, rtol=1e-15)


def test_diffusivities_pick_up_the_default():
    """catke_diffusivities and catke_dissipation_rate use cfg.c_bottom_shear near the bottom (bottom limit binding):
    K and the dissipation length scale with it."""
    c = CATKEConfig()
    old = c._replace(c_bottom_shear=0.28)
    args = (jnp.asarray([1e-4]), jnp.asarray([1e-8]), jnp.asarray([1e-8]), jnp.asarray([1e-10]),
            jnp.asarray([400.0]), jnp.asarray([10.0]), jnp.asarray(1.0e4), jnp.asarray(0.0))
    Ku_new, Kc_new, Ke_new = catke.catke_diffusivities(*args, c)
    Ku_old, Kc_old, Ke_old = catke.catke_diffusivities(*args, old)
    for a, b in ((Ku_new, Ku_old), (Kc_new, Kc_old), (Ke_new, Ke_old)):
        np.testing.assert_allclose(np.asarray(a) / np.asarray(b), 0.967 / 0.28, rtol=1e-12)
    om_new = catke.catke_dissipation_rate(*args, c)
    om_old = catke.catke_dissipation_rate(*args, old)
    np.testing.assert_allclose(np.asarray(om_old) / np.asarray(om_new), 0.967 / 0.28, rtol=1e-12)


def test_far_from_bottom_unchanged():
    c = CATKEConfig()
    w_star = jnp.asarray([0.02, 0.05, 0.001])
    N2 = jnp.asarray([1e-5, 1e-4, 1e-6])
    depth = jnp.asarray([20.0, 60.0, 5.0])
    hab = jnp.asarray([4000.0, 4000.0, 4000.0])
    a = catke._stable_length(w_star, N2, depth, hab, c.c_surface_shear, 0.28)
    b = catke._stable_length(w_star, N2, depth, hab, c.c_surface_shear, c.c_bottom_shear)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
