"""B4: a live dynamical core wraps as a differentiable AbstractComponent brick."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.components import AbstractComponent
from legoesm.components.dycore_component import DycoreComponent

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="cubed-sphere shallow water needs JAX_ENABLE_X64=1",
)


def _sw_model_and_state():
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
        CDGridShallowWaterState,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    n = 8
    grid = create_cubed_sphere(n)
    model = CDGridShallowWaterModel(grid, CDGridShallowWaterConfig())
    # A spatially-varying height (NOT a rest fixed point) so the dynamical RHS is
    # non-zero and genuinely depends on h (-g*grad(h) drives the momentum tendency).
    ramp = jnp.sin(2.0 * jnp.pi * jnp.linspace(0.0, 1.0, n))
    h = jnp.full((6, n, n), 1.0e4) + 200.0 * ramp[None, :, None]
    state = CDGridShallowWaterState(
        h=h,
        u_d=jnp.zeros((6, n + 1, n + 1)),
        v_d=jnp.zeros((6, n + 1, n + 1)),
        h_s=jnp.zeros((6, n, n)),
    )
    return grid, model, state


def test_wrap_dycore_as_abstract_component() -> None:
    """A real dycore, wrapped, IS an AbstractComponent with its metadata declared."""
    grid, model, _state = _sw_model_and_state()
    comp = DycoreComponent(model, prognostic_variables=("h", "u_d", "v_d"))

    assert isinstance(comp, AbstractComponent)
    assert comp.prognostic_variables == ("h", "u_d", "v_d")
    assert comp.required_forcing == ()   # a dry dynamical core needs no partner forcing
    assert comp.provided_fluxes == ()    # the pure dynamics provides no surface flux
    assert comp.model is model


def test_wrap_rejects_a_model_without_tendencies() -> None:
    with pytest.raises(TypeError, match="tendencies"):
        DycoreComponent(object(), prognostic_variables=("h",))


def test_dry_wrapper_rejects_forcing_rather_than_dropping_it() -> None:
    """A non-None forcing/params payload raises — never silently ignored."""
    grid, model, state = _sw_model_and_state()
    comp = DycoreComponent(model, prognostic_variables=("h", "u_d", "v_d"))
    with pytest.raises(ValueError, match="DRY dynamical core"):
        comp.tendency(grid, state, {"sw_down": 1.0}, None)
    with pytest.raises(ValueError, match="DRY dynamical core"):
        comp.tendency(grid, state, None, {"some_param": 1.0})


@_needs_x64
def test_component_tendency_runs_and_is_differentiable() -> None:
    """The brick's tendency seam is a pure, jax.grad-differentiable RHS (D1)."""
    grid, model, state = _sw_model_and_state()
    comp = DycoreComponent(model, prognostic_variables=("h", "u_d", "v_d"))

    tend = comp.tendency(grid, state, None, None)
    leaves = jax.tree.leaves(tend)
    assert leaves and all(jnp.all(jnp.isfinite(x)) for x in leaves)

    # d(sum of squared tendency leaves)/d(state.h): finite AND non-zero -> the
    # dynamical RHS genuinely depends on the height (the gradient flows through
    # the brick's tendency seam, exercising D1 on the real dycore).
    def loss(h0):
        t = comp.tendency(grid, state._replace(h=h0), None, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    g = jax.grad(loss)(state.h)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.max(jnp.abs(g)) > 0.0


# --- the SAME generic wrapper across other grid families (B4 replication) ---

def _latlon_sw_model_and_state():
    from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel,
        CGridLatLonShallowWaterState,
    )
    from legoesm.grids.latlon import create_latlon_grid

    n_lat = 16
    grid = create_latlon_grid(n_lat=n_lat)
    model = CGridLatLonShallowWaterModel(grid)
    n_lon = grid.n_lon
    ramp = jnp.sin(2.0 * jnp.pi * jnp.linspace(0.0, 1.0, n_lat))
    h = jnp.full((n_lat, n_lon), 1.0e4) + 200.0 * ramp[:, None]
    state = CGridLatLonShallowWaterState(
        h=h,
        # Non-zero zonal flow so the continuity tendency dh/dt = -div(h*u) is also
        # live (not just the pressure-gradient momentum tendency).
        u=jnp.full((n_lat, n_lon + 1), 5.0),
        v=jnp.zeros((n_lat + 1, n_lon)),
        h_s=jnp.zeros((n_lat, n_lon)),
    )
    return grid, model, state


def _mpas_sw_model_and_state():
    from legoesm.atmosphere.dynamics.shallow_water_mpas import (
        MPASShallowWaterConfig,
        MPASShallowWaterModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import MPASShallowWaterState
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    model = MPASShallowWaterModel(mesh, MPASShallowWaterConfig())
    # spatially-varying height so -g*grad(h) drives a non-zero RHS; non-zero
    # edge-normal velocity so continuity dh/dt = -div(h*u) is live too.
    h_data = 1.0e4 + 200.0 * jnp.sin(mesh.latCell)
    state = MPASShallowWaterState(
        h=Field(h_data, name="h", dims=("nCells",), units="m"),
        u=Field(5.0 * jnp.cos(mesh.lonEdge), name="u", dims=("nEdges",),
                units="m/s", staggering="edge"),
        h_s=Field(jnp.zeros(mesh.nCells), name="h_s", dims=("nCells",), units="m"),
    )
    return mesh, model, state


@_needs_x64
def test_wrap_latlon_sw_as_differentiable_component() -> None:
    """The generic wrapper turns the lat-lon C-grid SW core into a brick too."""
    grid, model, state = _latlon_sw_model_and_state()
    comp = DycoreComponent(model, prognostic_variables=("h", "u", "v"))

    assert isinstance(comp, AbstractComponent)
    leaves = jax.tree.leaves(comp.tendency(grid, state, None, None))
    assert leaves and all(jnp.all(jnp.isfinite(x)) for x in leaves)

    def loss_h(h0):
        t = comp.tendency(grid, state._replace(h=h0), None, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    def loss_u(u0):
        t = comp.tendency(grid, state._replace(u=u0), None, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    for loss, x in ((loss_h, state.h), (loss_u, state.u)):
        g = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(g)) and jnp.max(jnp.abs(g)) > 0.0


@_needs_x64
def test_wrap_mpas_sw_as_differentiable_component() -> None:
    """...and the MPAS/Voronoi edge-normal SW core — one wrapper, every grid."""
    mesh, model, state = _mpas_sw_model_and_state()
    comp = DycoreComponent(model, prognostic_variables=("h", "u"))

    assert isinstance(comp, AbstractComponent)
    leaves = jax.tree.leaves(comp.tendency(mesh, state, None, None))
    assert leaves and all(jnp.all(jnp.isfinite(x)) for x in leaves)

    def loss_h(h0):
        s = state._replace(h=state.h.replace(data=h0))
        t = comp.tendency(mesh, s, None, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    def loss_u(u0):
        s = state._replace(u=state.u.replace(data=u0))
        t = comp.tendency(mesh, s, None, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    for loss, x in ((loss_h, state.h.data), (loss_u, state.u.data)):
        g = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(g)) and jnp.max(jnp.abs(g)) > 0.0
