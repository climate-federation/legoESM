"""Lat-lon ``vert_advection_scheme`` of the shared-core ``sb81_theta`` form.

The lat-lon plug-ins used to hard-code the upwind theta transport; the field
threads the conservative Simmons-Burridge form ("sb", hybrid only), as
``MPASPrimitiveEquationConfig.vert_advection_scheme`` does on the ico lane.
Budget: global-mean temperature per model level.  At linear order in a wave
amplitude over a horizontally uniform background, every vertical-transport
term is linear in the mass flux, whose global level mean is zero, so the
level means cancel at that order; the first-order upwind form instead drifts
them linearly in the amplitude (|F|-weighted mean).  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

_x64 = pytest.mark.skipif(not jax.config.jax_enable_x64,
                          reason="fp64 tolerances: needs JAX_ENABLE_X64=1")


def _gw31_level_mean_drift(scheme, monkeypatch, n_steps=300):
    """max_k |global-mean T_k(t) - T_k(0)| [K] of DCMIP 2012 3-1 on an 18x36
    small-planet grid with the sb81_theta form, ``n_steps`` of 1.6 s."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
        hydrostatic_to_cgrid)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import make_dcmip2012_gw31_levels
    from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw

    # small bell (0.01 K): near the linear regime, where the conservative
    # form's level means cancel and the upwind |F| bias dominates
    monkeypatch.setattr(gw, "DELTA_THETA", 0.01)
    grid = create_latlon_grid(18, 36, radius=gw.small_planet_radius(),
                              omega=0.0)
    sigma = make_dcmip2012_gw31_levels()
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=True, vertical_form="sb81_theta",
        vert_advection_scheme=scheme)
    dt = 1.6
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    s = hydrostatic_to_cgrid(gw.gravity_wave_init_latlon(grid, sigma), grid)
    w = np.asarray(grid.area) / np.asarray(grid.area).sum()
    mean0 = np.einsum("ij,ijk->k", w, np.asarray(s.T))
    for _ in range(n_steps):
        s = model.step(s, dt)
    assert np.all(np.isfinite(np.asarray(s.T)))
    return float(np.abs(np.einsum("ij,ijk->k", w, np.asarray(s.T))
                        - mean0).max())


@_x64
def test_sb_holds_the_level_means_where_upwind_drifts(monkeypatch):
    """0.01 K bell, 480 s.  Measured (fp64 CPU): "sb" 9.6e-9 K (the finite-
    amplitude residual), "upwind" (the old hard-coded lat-lon transport)
    7.3e-6 K.  Fails if the field is not threaded (both arms run upwind)."""
    up = _gw31_level_mean_drift("upwind", monkeypatch)
    sb = _gw31_level_mean_drift("sb", monkeypatch)
    print(f"level-mean drift: upwind {up:.3e} K, sb {sb:.3e} K")
    assert sb < 1e-7
    assert up > 2e-6
    assert up > 100.0 * sb


def test_scheme_reaches_the_operators():
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg)
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(8, 16)
    assert Cfg().vert_advection_scheme == "upwind"
    for scheme in ("upwind", "sb"):
        ops = g.latlon_pe_operators(
            grid, Cfg(vertical_form="sb81_theta", vert_advection_scheme=scheme),
            {}, 600.0)
        assert ops.vert_scheme == scheme


@pytest.mark.parametrize("scheme, form, hybrid, match", [
    ("van_leer", "sb81_theta", True, "unknown"),
    ("ppm", "sb81_theta", True, "unknown"),
    ("sb", "legacy", True, "acts only"),
    ("sb", "as83", True, "acts only"),
    ("sb", "as83_logmean", True, "acts only"),
    ("sb", "sb81_theta", False, "acts only"),
])
def test_guard_refuses_inert_or_unknown(scheme, form, hybrid, match):
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    with pytest.raises(ValueError, match=match):
        g._check_latlon_vert_scheme(scheme, form, hybrid)


@pytest.mark.parametrize("scheme, form, hybrid", [
    ("upwind", "legacy", True), ("upwind", "legacy", False),
    ("upwind", "sb81_theta", False), ("upwind", "as83", True),
    ("sb", "sb81_theta", True),
])
def test_guard_accepts(scheme, form, hybrid):
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    g._check_latlon_vert_scheme(scheme, form, hybrid)


def test_tendency_entry_runs_the_guard():
    """The model's tendency entry refuses the legacy form with "sb" (it would
    run the old upwind transport silently)."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg,
        CGridLatLonHydrostaticState)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import make_dcmip2012_gw31_levels
    import jax.numpy as jnp
    grid = create_latlon_grid(8, 16)
    sigma = make_dcmip2012_gw31_levels()
    st = CGridLatLonHydrostaticState(
        u=jnp.zeros((8, 17, 10)), v=jnp.zeros((9, 16, 10)),
        T=jnp.full((8, 16, 10), 250.0), p_s=jnp.full((8, 16), 1e5),
        phis=jnp.zeros((8, 16)))
    with pytest.raises(ValueError, match="acts only"):
        g.cgrid_latlon_pe_tendencies(
            st, grid, sigma, Cfg(vert_advection_scheme="sb"))


def test_operator_factory_refuses_inert_sb():
    """The factory path (no tendency entry) refuses "sb" on the forms that
    ignore it."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg)
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(8, 16)
    for form in ("legacy", "as83", "as83_logmean"):
        with pytest.raises(ValueError, match="acts only"):
            g.latlon_pe_operators(
                grid, Cfg(vertical_form=form, vert_advection_scheme="sb"),
                {}, 600.0)


def test_shared_core_refuses_sb_on_sigma():
    """sigma: the factory cannot see the coordinate; the shared core's
    vertical operator refuses "sb" itself."""
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_latlon_grid(8, 16)
    sigma = create_sigma_coordinate(5)
    ops = g.latlon_pe_operators(
        grid, Cfg(vertical_form="sb81_theta", vert_advection_scheme="sb"),
        {}, 600.0)
    vel = (jnp.zeros((8, 17, 5)), jnp.zeros((9, 16, 5)))
    with pytest.raises(ValueError, match="sb"):
        g.primitive_equation_tendencies(
            jnp.full((8, 16, 5), 250.0), vel, jnp.full((8, 16), 1e5),
            jnp.zeros((8, 16)), sigma, ops)


@_x64
def test_sb_changes_theta_transport_only():
    """Hybrid lat-lon: "sb" changes the vertical temperature term and leaves
    the vertical momentum advection (upwind on hybrid, as on the ico lane)
    bit-identical, on a sheared, divergent, horizontally varying state."""
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import make_dcmip2012_gw31_levels
    grid = create_latlon_grid(8, 16)
    sigma = make_dcmip2012_gw31_levels()
    rng = np.random.default_rng(0)
    k = jnp.arange(10.0)
    u = 10.0 + 2.0 * k + jnp.asarray(rng.normal(size=(8, 17, 10)))
    v = jnp.asarray(rng.normal(size=(9, 16, 10))).at[0].set(0.0).at[-1].set(0.0)
    T = 230.0 + 6.0 * k + jnp.asarray(rng.normal(size=(8, 16, 10)))
    ps = 1e5 + 300.0 * jnp.asarray(rng.normal(size=(8, 16)))
    out = {}
    for scheme in ("upwind", "sb"):
        ops = g.latlon_pe_operators(
            grid, Cfg(vertical_form="sb81_theta", vert_advection_scheme=scheme),
            {}, 600.0)
        *_, ctx = g.primitive_equation_tendencies(
            T, (u, v), ps, jnp.zeros((8, 16)), sigma, ops)
        out[scheme] = ctx["terms"]
    for a, b in zip(out["upwind"]["vert_mom"], out["sb"]["vert_mom"]):
        assert float(jnp.abs(a).max()) > 0.0
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for s in ("upwind", "sb"):
        assert np.all(np.isfinite(np.asarray(out[s]["vert_T"])))
    assert not np.allclose(np.asarray(out["upwind"]["vert_T"]),
                           np.asarray(out["sb"]["vert_T"]))
