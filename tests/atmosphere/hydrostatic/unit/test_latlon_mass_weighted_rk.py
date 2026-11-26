"""The lat-lon C-grid dycore advances tracers through the RK step as layer
mass Q = dp*q, so the step conserves sum(q dp) (``tracer_mass_weighted_rk``,
default True).  The legacy mixing-ratio update (False) combines q and p_s
separately across the SSP-RK3 stages and creates or destroys tracer mass at
second order in dt; on the coupled slab deck at fp64 it created 2.5e-6 of a
day's precipitation (the whole atmosphere residual).

KNOWN DIFFERENCE vs the legacy form, measured on the fixture below (12x24,
8 levels, dt 600 s, polar filter on, 4 steps), pinned per case.  Re-pinned
after the pole-row v / polar-cap vorticity fix (#2639 follow-up): every pin
moved by <= 1.2%; with grid.global_poles=False the old pins return to
every digit, so the shift is the pole fix alone.  The mass-weighted step
still conserves ~50x better than the legacy update.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
    hydrostatic_to_cgrid)

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_pf_tracer_mass",
    Path(__file__).with_name("test_latlon_polar_filter_tracer_mass.py"))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)
DT, RES, _coords, _tracer_mass = _pf.DT, _pf.RES, _pf._coords, _pf._tracer_mass
fp64 = _pf.fp64

# relative change of each tracer's mass over the 4 steps
_MASS = {
    ("sigma", True): {"q_v": -4.079417973597815e-10, "q_c": 9.849726936235415e-11},
    ("sigma", False): {"q_v": -2.1684814028531837e-08, "q_c": -1.0698386758825048e-09},
    ("hybrid", False): {"q_v": -2.1255574859590977e-08, "q_c": -1.1744393796154918e-09},
}
# max |q_mass_weighted - q_legacy| / max |q_legacy| after the 4 steps
_GAP = {
    "sigma": {"q_v": 4.7626983119430156e-07, "q_c": 6.966465708577796e-07},
    "hybrid": {"q_v": 1.2063852513064226e-06, "q_c": 1.763639178611702e-06},
}


def _run(coord, mass_weighted):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.factory import create_grid
    g = create_grid("latlon", RES)
    s = _coords()[coord]
    nat = hydrostatic_to_cgrid(held_suarez_init_latlon(g, s, T_init=280.0), g)
    lat, lon = jnp.asarray(g.lat2d), jnp.asarray(g.lon2d)
    wave = jnp.sin(lat) ** 4 * jnp.cos(9.0 * lon)
    nat = nat._replace(
        p_s=nat.p_s + 800.0 * wave, u=nat.u + 5.0,
        tracers={"q_v": 5e-3 * (1.0 + 0.5 * wave)[..., None]
                 * jnp.ones_like(nat.T),
                 "q_c": 1e-4 * (1.0 + 0.5 * jnp.sin(11.0 * lon))[..., None]
                 * jnp.ones_like(nat.T)})
    cfg = CGridLatLonPrimitiveEquationConfig(
        use_polar_filter=True, tracer_mass_weighted_rk=mass_weighted)
    model = CGridLatLonPrimitiveEquationModel(g, s, cfg, dt=DT)
    area = jnp.asarray(g.area)
    m0 = _tracer_mass(nat, s, area)
    for _ in range(4):
        nat = model.step(nat, DT)
    m1 = _tracer_mass(nat, s, area)
    return nat, {k: (m1[k] - m0[k]) / m0[k] for k in m0}


def test_mass_weighted_rk_conserves_tracer_mass_on_hybrid_levels(fp64):
    _, rel = _run("hybrid", True)
    for k, v in rel.items():
        assert abs(v) <= 1e-12, (k, v)          # measured 3.3e-14


@pytest.mark.parametrize("coord, mw", sorted(_MASS))
def test_tracer_mass_change_is_the_recorded_one(coord, mw, fp64):
    _, rel = _run(coord, mw)
    for k, v in rel.items():
        np.testing.assert_allclose(v, _MASS[(coord, mw)][k], rtol=1e-6,
                                   atol=1e-12, err_msg=k)


@pytest.mark.parametrize("coord", sorted(_GAP))
def test_gap_to_the_legacy_update_is_the_recorded_one(coord, fp64):
    a, _ = _run(coord, True)
    b, _ = _run(coord, False)
    for k in a.tracers:
        gap = float(jnp.max(jnp.abs(a.tracers[k] - b.tracers[k]))
                    / jnp.max(jnp.abs(b.tracers[k])))
        np.testing.assert_allclose(gap, _GAP[coord][k], rtol=1e-6,
                                   atol=1e-12, err_msg=k)


def test_gradient_through_the_mass_weighted_step_is_finite(fp64):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.factory import create_grid
    g = create_grid("latlon", RES)
    s = _coords()["hybrid"]
    nat = hydrostatic_to_cgrid(held_suarez_init_latlon(g, s, T_init=280.0), g)
    nat = nat._replace(tracers={"q_v": 5e-3 * jnp.ones_like(nat.T)})
    model = CGridLatLonPrimitiveEquationModel(
        g, s, CGridLatLonPrimitiveEquationConfig(use_polar_filter=True), dt=DT)

    def f(a):
        out = model.step(nat._replace(
            tracers={"q_v": nat.tracers["q_v"] * a}), DT)
        return jnp.sum(out.tracers["q_v"] * out.T)
    gr = jax.grad(f)(jnp.asarray(1.0))
    assert np.isfinite(float(gr)) and float(gr) != 0.0
