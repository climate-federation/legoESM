"""Direct unit tests for the Kessler -> lat-lon C-grid operator-split forcing.

Exercises ``make_kessler_forcing_latlon`` (the moist-baroclinic-wave
``physics_fn`` for :class:`CGridLatLonPrimitiveEquationModel`) on a hand-built
lat-lon column state.  No grid metric and no dycore are needed: the adapter is
column-local (it ignores ``grid``), so these tests isolate the microphysics
wiring, sign conventions, water conservation, AD-safety, and the guard rails.
Stepping the full dycore (with mass-consistent tracer advection) is covered
separately by the moist smoke run on a compute node.

Mirrors ``tests/atmosphere/test_kessler_forcing_mpas.py`` for the lat-lon
representation (3-D ``(n_lat, n_lon, nlev)`` tracer fields, returns the unified
:class:`HydrostaticTendencies` with ``dv_dt`` populated).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_latlon


def _make_state(n_lat=2, n_lon=2, nlev=12):
    """Lat-lon state: 4 columns — supersaturated / bone-dry / half-RH / sat."""
    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    T_col = jnp.linspace(300.0, 230.0, nlev)
    T = jnp.broadcast_to(
        T_col[None, None, :], (n_lat, n_lon, nlev)
    ).astype(jnp.float64)
    u = jnp.zeros((n_lat, n_lon, nlev))
    v = jnp.zeros((n_lat, n_lon, nlev))
    phis = jnp.zeros((n_lat, n_lon))

    p_full = p_s[..., None] * sigma.sigma_full[None, None, :]
    q_sat = saturation_mixing_ratio(T, p_full)
    # rh per column, laid out (n_lat, n_lon): [[2.0, 1e-4], [0.5, 1.0]].
    rh = jnp.array([[2.0, 1.0e-4], [0.5, 1.0]])[:n_lat, :n_lon, None]
    q_v = rh * q_sat
    q_zero = jnp.zeros((n_lat, n_lon, nlev))

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    state = HydrostaticState(
        u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers={
            "q_v": Field(data=q_v, name="q_v", dims=dims_3d, units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c", dims=dims_3d, units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r", dims=dims_3d, units="kg/kg"),
        },
    )
    return state, sigma


def test_shapes_types_and_no_momentum_source():
    state, sigma = _make_state()
    out = make_kessler_forcing_latlon(dt=300.0)(state, None, sigma)
    assert isinstance(out, HydrostaticTendencies)
    assert out.du_dt.data.shape == state.u.data.shape
    assert out.dv_dt.data.shape == state.v.data.shape
    assert out.dT_dt.data.shape == state.T.data.shape
    assert set(out.tracer_tendencies) == {"q_v", "q_c", "q_r"}
    # Warm-rain microphysics has no direct momentum / surface-pressure source.
    assert jnp.all(out.du_dt.data == 0.0)
    assert jnp.all(out.dv_dt.data == 0.0)
    assert jnp.all(out.dp_s_dt.data == 0.0)
    assert jnp.all(out.dphis_dt.data == 0.0)
    assert jnp.all(jnp.isfinite(out.dT_dt.data))
    for k in ("q_v", "q_c", "q_r"):
        assert jnp.all(jnp.isfinite(out.tracer_tendencies[k].data))


def test_supersaturated_condenses_and_heats():
    state, sigma = _make_state()
    out = make_kessler_forcing_latlon(dt=300.0)(state, None, sigma)
    dqv = out.tracer_tendencies["q_v"].data
    dqc = out.tracer_tendencies["q_c"].data
    dT = out.dT_dt.data
    # Supersaturated column (0,0): vapor down, cloud up, latent heating up.
    assert jnp.any(dqv[0, 0] < 0.0)
    assert jnp.all(dqv[0, 0] <= 1e-12)
    assert jnp.all(dqc[0, 0] >= -1e-12)
    cond_mask = dqv[0, 0] < -1e-12
    assert jnp.all(dT[0, 0][cond_mask] > 0.0)
    # Bone-dry column (0,1): essentially no condensation.
    assert jnp.allclose(dqv[0, 1], 0.0, atol=1e-12)


def test_conserves_total_water_at_zero_hydrometeors():
    # q_c = q_r = 0 => only condensation active => dq_v = -dq_c, dq_r = 0,
    # so the total-water tendency must be (numerically) exactly zero.
    state, sigma = _make_state()
    out = make_kessler_forcing_latlon(dt=300.0)(state, None, sigma)
    total = (
        out.tracer_tendencies["q_v"].data
        + out.tracer_tendencies["q_c"].data
        + out.tracer_tendencies["q_r"].data
    )
    assert jnp.max(jnp.abs(total)) < 1e-9


def test_differentiable_wrt_temperature():
    state, sigma = _make_state()
    fn = make_kessler_forcing_latlon(dt=300.0)

    def loss(T):
        s = state._replace(T=state.T.replace(data=T))
        return jnp.sum(fn(s, None, sigma).dT_dt.data ** 2)

    g = jax.grad(loss)(state.T.data)
    assert g.shape == state.T.data.shape
    assert jnp.all(jnp.isfinite(g))


def test_requires_tracers():
    state, sigma = _make_state()
    fn = make_kessler_forcing_latlon(dt=300.0)
    with pytest.raises(ValueError):
        fn(state._replace(tracers=None), None, sigma)


def test_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        make_kessler_forcing_latlon(dt=0.0)


def _make_cube_state(n_face=6, n=2, nlev=12):
    """Cubed-sphere cell-centred state: same 4 RH columns tiled over faces."""
    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    p_s = jnp.full((n_face, n, n), 1.0e5)
    T_col = jnp.linspace(300.0, 230.0, nlev)
    T = jnp.broadcast_to(
        T_col[None, None, None, :], (n_face, n, n, nlev)
    ).astype(jnp.float64)
    u = jnp.zeros((n_face, n, n, nlev))
    v = jnp.zeros((n_face, n, n, nlev))
    phis = jnp.zeros((n_face, n, n))

    p_full = p_s[..., None] * sigma.sigma_full[None, None, None, :]
    q_sat = saturation_mixing_ratio(T, p_full)
    rh = jnp.array([2.0, 1.0e-4, 0.5, 1.0]).reshape(1, n, n, 1)
    q_v = rh * q_sat
    q_zero = jnp.zeros((n_face, n, n, nlev))

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    state = HydrostaticState(
        u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers={
            "q_v": Field(data=q_v, name="q_v", dims=dims_3d, units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c", dims=dims_3d, units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r", dims=dims_3d, units="kg/kg"),
        },
    )
    return state, sigma


def test_cube_wrapper_shapes_dims_and_conservation():
    """make_kessler_forcing_cube (thin wrapper over the shared grid-space
    adapter) must handle the 4-D cube layout: cube dim labels, correct shapes,
    no momentum source, and total-water conservation at zero hydrometeors."""
    from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_cube

    state, sigma = _make_cube_state()
    out = make_kessler_forcing_cube(dt=300.0)(state, None, sigma)
    assert isinstance(out, HydrostaticTendencies)
    assert out.dT_dt.data.shape == state.T.data.shape
    assert out.dT_dt.dims == ("face", "x", "y", "level")
    assert out.dp_s_dt.dims == ("face", "x", "y")
    assert set(out.tracer_tendencies) == {"q_v", "q_c", "q_r"}
    assert jnp.all(out.du_dt.data == 0.0)
    assert jnp.all(out.dv_dt.data == 0.0)
    assert jnp.all(out.dp_s_dt.data == 0.0)
    total = (
        out.tracer_tendencies["q_v"].data
        + out.tracer_tendencies["q_c"].data
        + out.tracer_tendencies["q_r"].data
    )
    assert jnp.max(jnp.abs(total)) < 1e-9


def test_moist_ic_attaches_tracers():
    # baroclinic_wave_init_latlon(moist=True) must attach q_v/q_c/q_r so the
    # forcing has something to condense; dry default keeps tracers=None.
    from legoesm.grids.latlon import create_latlon_grid
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    grid = create_latlon_grid(n_lat=16, n_lon=32, dtype=jnp.float64)
    sigma = create_sigma_coordinate(10, dtype=jnp.float64)
    dry = baroclinic_wave_init_latlon(grid, sigma, perturbed=False)
    assert dry.tracers is None
    wet = baroclinic_wave_init_latlon(grid, sigma, perturbed=False, moist=True)
    assert wet.tracers is not None
    assert set(wet.tracers) == {"q_v", "q_c", "q_r"}
    assert jnp.all(wet.tracers["q_v"].data >= 0.0)
    assert jnp.any(wet.tracers["q_v"].data > 0.0)
    # q_v never exceeds saturation (capped) and q_c/q_r start at zero.
    assert jnp.all(wet.tracers["q_c"].data == 0.0)
    assert jnp.all(wet.tracers["q_r"].data == 0.0)
