"""Melt-pond albedo must DECREASE with depth (Ebert & Curry 1993; Briegleb &
Light 2007).  Regression guard for the inverted-ramp bug where a zero-depth
pond went perfectly black and a deep pond was the brightest state."""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ice.shortwave import _band_albedo_pond, delta_eddington_albedo


def test_band_albedo_pond_darkens_with_depth():
    ice_vis, ice_nir = jnp.asarray(0.68), jnp.asarray(0.40)
    kw = dict(alpha_deep_vis=constants.alpha_pond_max_vis,
              alpha_deep_nir=constants.alpha_pond_max_nir)
    shallow = _band_albedo_pond(jnp.asarray(0.0), ice_vis, ice_nir, **kw)
    deep = _band_albedo_pond(jnp.asarray(0.5), ice_vis, ice_nir, **kw)
    # h=0 -> the wet ice it sits on (NOT black); deep -> darker deep-pond floor
    assert float(shallow[0]) == float(ice_vis)
    assert float(deep[0]) < float(shallow[0])
    assert abs(float(deep[0]) - constants.alpha_pond_max_vis) < 1e-6


def test_band_albedo_pond_never_brighter_than_thin_dark_ice():
    # Thin ice darker than the deep-pond floor: deepening must NOT brighten it.
    ice_vis, ice_nir = jnp.asarray(0.10), jnp.asarray(0.05)
    kw = dict(alpha_deep_vis=constants.alpha_pond_max_vis,
              alpha_deep_nir=constants.alpha_pond_max_nir)
    shallow = _band_albedo_pond(jnp.asarray(0.0), ice_vis, ice_nir, **kw)
    deep = _band_albedo_pond(jnp.asarray(0.5), ice_vis, ice_nir, **kw)
    assert float(deep[0]) <= float(shallow[0])
    assert float(deep[1]) <= float(shallow[1])


def test_delta_eddington_deeper_pond_lowers_tile_albedo():
    # (T_sfc, h_ice, h_snow, pond_area); vary only pond_depth
    args = (jnp.asarray(constants.T_freeze), jnp.asarray(1.0),
            jnp.asarray(0.0), jnp.asarray(0.5))
    shallow, _ = delta_eddington_albedo(*args, jnp.asarray(0.02))
    deep, _ = delta_eddington_albedo(*args, jnp.asarray(0.4))
    assert float(deep) < float(shallow)
