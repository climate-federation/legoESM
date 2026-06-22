"""Lat-structured warm-start soil IC (``--warm-start-soil``).

The coupled driver can initialise the land soil temperature at the atmosphere's
lat-structured near-surface air temperature (t=0) instead of the uniform 280 K
default, which starts tropical land soil ~18 K too cold and cold-spins the slow
multilayer soil for months.  ``init_multilayer_land_state`` /
``init_surface_state`` accept a scalar (legacy, uniform) OR a per-column /
spatial array; the scalar path stays byte-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from legoesm.coupler.coupler import init_surface_state
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.multilayer_land import init_multilayer_land_state


def test_multilayer_scalar_default_uniform() -> None:
    st = init_multilayer_land_state(6, MultiLayerLandConfig(), T_init=280.0)
    assert jnp.allclose(st.T_soil, 280.0)


def test_multilayer_array_warmstart_broadcasts_vertically() -> None:
    ncol = 6
    T0 = jnp.linspace(250.0, 300.0, ncol)
    st = init_multilayer_land_state(ncol, MultiLayerLandConfig(), T_init=T0)
    assert st.T_soil.shape[0] == ncol
    # each column's whole soil column equals that column's warm-start T
    assert jnp.allclose(st.T_soil[:, 0], T0)
    assert jnp.allclose(st.T_soil[:, -1], T0)


def test_surface_state_multilayer_array() -> None:
    shape = (6, 4, 4)
    ncol = 6 * 4 * 4
    T0 = jnp.reshape(jnp.linspace(250.0, 300.0, ncol), shape)
    sfc = init_surface_state(shape, T_soil_init=T0,
                             land_config=MultiLayerLandConfig())
    assert jnp.allclose(sfc.land.T_soil[:, 0], T0.reshape(ncol))


def test_surface_state_slab_array() -> None:
    shape = (6, 4, 4)
    T0 = jnp.full(shape, 290.0)
    sfc = init_surface_state(shape, T_soil_init=T0, land_config=LandConfig())
    assert jnp.allclose(sfc.land.T_soil.data, 290.0)


def test_surface_state_scalar_default_unchanged() -> None:
    # No T_soil_init => the legacy uniform 280 K default (byte-identical).
    shape = (6, 4, 4)
    sfc = init_surface_state(shape, land_config=LandConfig())
    assert jnp.allclose(sfc.land.T_soil.data, 280.0)
