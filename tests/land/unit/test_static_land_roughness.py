"""static_land_roughness: the roughness the land-stress seed uses before any
land solve must be the one the column's own surface scheme would use."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.stability import compute_aerodynamics
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import static_land_roughness
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig


def test_two_leaf_uses_the_per_column_canopy_geometry():
    lp = SimpleNamespace(hc=jnp.array([20.0, 0.5]), LAI=jnp.array([5.0, 0.3]),
                         rz0m=jnp.array([0.055, 0.12]),
                         rd=jnp.array([0.67, 0.67]))
    z0, d = static_land_roughness(
        lp, MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig()), 2)
    z0e, de = compute_aerodynamics(lp.hc, lp.LAI, lp.rz0m, lp.rd)
    np.testing.assert_allclose(z0, z0e, rtol=1e-12)
    np.testing.assert_allclose(d, de, rtol=1e-12)
    assert z0[0] > 5.0 * z0[1] and d[0] > d[1]   # forest rougher than grass


def test_simple_seb_uses_its_z0_and_no_displacement():
    cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig())
    z0, d = static_land_roughness(SimpleNamespace(z0=jnp.array([0.3, 0.01])),
                                  cfg, 2)
    np.testing.assert_array_equal(z0, [0.3, 0.01])
    np.testing.assert_array_equal(d, [0.0, 0.0])
    z0, _ = static_land_roughness(None, cfg, 3)
    np.testing.assert_array_equal(z0, [cfg.z0_land] * 3)


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="no static roughness"):
        static_land_roughness(
            None, MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig()), 2)
