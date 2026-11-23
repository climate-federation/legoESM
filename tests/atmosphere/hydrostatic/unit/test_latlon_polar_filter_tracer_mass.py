"""The lat-lon C-grid dycore's polar filter conserves tracer mass.

The filter acts on every tendency near the poles.  Filtering a tracer's
mixing-ratio tendency and the surface-pressure tendency separately keeps
neither term of d/dt sum(q dp); filtering the tracer's layer-mass tendency
does.  Measured on the coupled slab deck: the separate form lost ~1e-3 of a
day's precipitation from the atmosphere at fp64.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
    hydrostatic_to_cgrid)
from legoesm.grids.vertical import dp_from_hybrid

RES, NLEV, DT = 12, 8, 600.0


def _coords():
    from legoesm.grids.vertical import (create_sigma_coordinate,
                                        make_hybrid_levels)
    return {"sigma": create_sigma_coordinate(NLEV),
            "hybrid": make_hybrid_levels(NLEV)}


def _tracer_mass(nat, s, area):
    if hasattr(s, "dB"):
        dp = dp_from_hybrid(s, nat.p_s)
    else:
        dp = nat.p_s[..., None] * s.dsigma
    return {k: float(jnp.sum(area[..., None] * q * dp) / constants.g)
            for k, q in nat.tracers.items()}


@pytest.mark.parametrize("coord", ["sigma", "hybrid"])
def test_polar_filtered_transport_conserves_tracer_mass(coord):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.factory import create_grid
    g = create_grid("latlon", RES)
    s = _coords()[coord]
    nat = hydrostatic_to_cgrid(held_suarez_init_latlon(g, s, T_init=280.0), g)
    lat, lon = jnp.asarray(g.lat2d), jnp.asarray(g.lon2d)
    # zonal structure in p_s and in the tracers where the filter acts
    wave = (jnp.sin(lat) ** 4 * jnp.cos(3.0 * lon))
    nat = nat._replace(
        p_s=nat.p_s + 800.0 * wave,
        u=nat.u + 5.0,
        tracers={"q_v": 5e-3 * (1.0 + 0.5 * wave)[..., None]
                 * jnp.ones_like(nat.T),
                 "q_c": 1e-4 * (1.0 + jnp.sin(5.0 * lon))[..., None]
                 * jnp.ones_like(nat.T)})
    cfg = CGridLatLonPrimitiveEquationConfig(use_polar_filter=True)
    model = CGridLatLonPrimitiveEquationModel(g, s, cfg, dt=DT)
    area = jnp.asarray(g.area)
    m0 = _tracer_mass(nat, s, area)
    for _ in range(4):
        nat = model.step(nat, DT)
    m1 = _tracer_mass(nat, s, area)
    for k in m0:
        assert abs(m1[k] - m0[k]) <= 1e-12 * m0[k], (k, m0[k], m1[k])
