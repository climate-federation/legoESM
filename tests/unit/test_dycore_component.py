"""B4: a live dynamical core wraps as a differentiable AbstractComponent brick."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.components import AbstractComponent
from legoesm.components.dycore_component import (
    DycoreComponent,
    ForcedDycoreComponent,
)

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="cubed-sphere shallow water needs JAX_ENABLE_X64=1",
)


def _sw_model_and_state():
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
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


# --- ForcedDycoreComponent: physics-tendency-coupled cores (B4 forcing variant) ---

def _cdgrid_pe_model_and_state():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import FV3HydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

    # n=8: at n=4 create_cubed_sphere_cdgrid's gnomonic="auto" inference is
    # ambiguous (cell aspect ~1.16 in the [1.15,1.25] band); n=8 equiangular
    # is unambiguous.
    n, nlev = 8, 5
    base = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    model = CDGridPrimitiveEquationModel(base, sigma)
    t_data = 250.0 * jnp.ones((6, n, n, nlev)) + jax.random.normal(
        jax.random.PRNGKey(10), (6, n, n, nlev))
    state = FV3HydrostaticState(
        u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
        v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
        T=Field(t_data, name="T"),
        p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
    )
    return base, model, state


def _mpas_pe_model_and_state():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    nlev = 5
    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    sigma = create_sigma_coordinate(nlev)
    model = MPASPrimitiveEquationModel(mesh, sigma)
    t_data = 250.0 * jnp.ones((mesh.nCells, nlev)) + jax.random.normal(
        jax.random.PRNGKey(11), (mesh.nCells, nlev))
    state = MPASHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, nlev)), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(t_data, name="T", dims=("nCells", "nlev"), units="K"),
        p_s=Field(1e5 * jnp.ones((mesh.nCells,)), name="p_s",
                  dims=("nCells",), units="Pa"),
        phis=Field(jnp.zeros((mesh.nCells,)), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
    )
    return mesh, model, state


def test_forced_wrapper_rejects_a_dry_core() -> None:
    """A DRY core (tendencies without physics_tendency) is rejected with a pointer
    back to DycoreComponent — the forcing-coupled wrapper is not a silent fallback."""
    _grid, model, _state = _sw_model_and_state()  # cube SW: tendencies(state) only
    with pytest.raises(TypeError, match="physics_tendency"):
        ForcedDycoreComponent(model, prognostic_variables=("h",))


def test_forced_wrapper_metadata_and_param_rejection() -> None:
    grid, model, state = _cdgrid_pe_model_and_state()
    comp = ForcedDycoreComponent(model, prognostic_variables=("u_d", "v_d", "T", "p_s"))

    assert isinstance(comp, AbstractComponent)
    assert comp.required_forcing == ("physics_tendency",)
    assert comp.provided_fluxes == ()
    assert comp.model is model
    # params has no seam → rejected, never silently dropped
    with pytest.raises(ValueError, match="no tunable-parameter seam"):
        comp.tendency(grid, state, None, {"k": 1.0})


def _tree_allclose(a, b) -> bool:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        jnp.allclose(x, y) for x, y in zip(la, lb))


def _assert_forced_core(grid, model, state, prognostic_variables) -> None:
    """Zero forcing == pure dynamics; non-zero forcing changes the RHS and is
    differentiable through the forcing seam (D1)."""
    comp = ForcedDycoreComponent(
        model, prognostic_variables=prognostic_variables)
    assert comp.prognostic_variables == prognostic_variables

    dyn = model.tendencies(state)                       # pure dynamical RHS
    # forcing=None routes physics_tendency=None → identical to pure dynamics
    assert _tree_allclose(comp.tendency(grid, state, None, None), dyn)
    # an explicit ZERO physics tendency is a no-op (added leaf-wise)
    zero_phys = jax.tree.map(jnp.zeros_like, dyn)
    assert _tree_allclose(comp.tendency(grid, state, zero_phys, None), dyn)
    # The core ADDS physics_tendency leaf-wise, so a NON-zero forcing reaches the
    # RHS EXACTLY: every consumed leaf gains the forcing identically (forced - dyn
    # == pert), every unconsumed leaf is unchanged.  This is stronger than an
    # aggregate "RHS changed" check — it pins the per-leaf identity-add and proves
    # at least one leaf is genuinely consumed.
    pert = jax.tree.map(lambda x: 1e-2 * jnp.ones_like(x), dyn)
    forced = comp.tendency(grid, state, pert, None)
    consumed = 0
    for fl, dl, pl in zip(jax.tree.leaves(forced), jax.tree.leaves(dyn),
                          jax.tree.leaves(pert)):
        assert jnp.all(jnp.isfinite(fl))
        diff = fl - dl
        dropped = bool(jnp.allclose(diff, 0.0))
        added = bool(jnp.allclose(diff, pl))
        assert dropped or added, (
            "a forcing leaf was neither dropped nor added identically — the "
            "wrapper must not scale/transform the supplied physics tendency")
        consumed += int(added and not dropped)
    assert consumed > 0, "no forcing leaf was consumed by the core"

    # Differentiate wrt the FULL forcing pytree (not an aggregate scalar): the
    # gradient is finite and at least one consumed leaf carries a non-zero
    # gradient — the physics tendency flows differentiably through the seam (D1).
    def loss(phys):
        t = comp.tendency(grid, state, phys, None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    g_leaves = jax.tree.leaves(jax.grad(loss)(pert))
    assert g_leaves and all(jnp.all(jnp.isfinite(x)) for x in g_leaves)
    assert any(float(jnp.max(jnp.abs(x))) > 0.0 for x in g_leaves)


@_needs_x64
def test_forced_cdgrid_pe_couples_and_differentiates() -> None:
    grid, model, state = _cdgrid_pe_model_and_state()
    # cdgrid PE (FV3HydrostaticState) evolves D-grid winds u_d/v_d, not 'u'.
    _assert_forced_core(grid, model, state, ("u_d", "v_d", "T", "p_s"))


@_needs_x64
def test_forced_mpas_pe_couples_and_differentiates() -> None:
    mesh, model, state = _mpas_pe_model_and_state()
    # MPAS PE evolves a single edge-normal wind 'u'.
    _assert_forced_core(mesh, model, state, ("u", "T", "p_s"))


@_needs_x64
def test_forced_wrapper_validates_the_configured_forcing_name() -> None:
    """forcing_name is functional: the construction gate validates THAT name (not
    a hardcoded 'physics_tendency'), so a name the core's tendencies does not
    accept is rejected — and required_forcing advertises the configured name."""
    _grid, model, _state = _cdgrid_pe_model_and_state()
    comp = ForcedDycoreComponent(model, prognostic_variables=("u_d",))
    assert comp.required_forcing == ("physics_tendency",)
    with pytest.raises(TypeError, match="nonexistent_forcing"):
        ForcedDycoreComponent(model, prognostic_variables=("u_d",),
                              forcing_name="nonexistent_forcing")


@_needs_x64
def test_prescribed_brick_drives_forced_dycore() -> None:
    """The independent-OR-coupled component seam, end-to-end: a PrescribedComponent
    PRODUCES the physics tendency a ForcedDycoreComponent CONSUMES — the producer's
    provided_fluxes matches the consumer's required_forcing, the coupling is the
    exact additive RHS, and a gradient flows producer -> consumer (D1)."""
    from legoesm.components.prescribed_component import PrescribedComponent

    grid, model, state = _cdgrid_pe_model_and_state()
    forced = ForcedDycoreComponent(
        model, prognostic_variables=("u_d", "v_d", "T", "p_s"))
    dyn = model.tendencies(state)
    phys = jax.tree.map(lambda x: 1e-2 * jnp.ones_like(x), dyn)
    name = forced.required_forcing[0]                       # "physics_tendency"
    prescribed = PrescribedComponent({name: phys}, provided_fluxes=(name,))

    # the producer provides (at least) what the consumer requires
    assert set(forced.required_forcing) <= set(prescribed.provided_fluxes)

    # couple by ROUTING the named flux (provided_fluxes is load-bearing): the
    # consumer's RHS = dynamics + the prescribed forcing (exact additive add).
    forcing = prescribed.provide(grid, state)[name]
    coupled = forced.tendency(grid, state, forcing, None)
    consumed = 0
    for cl, dl, pl in zip(jax.tree.leaves(coupled), jax.tree.leaves(dyn),
                          jax.tree.leaves(phys)):
        diff = cl - dl
        assert bool(jnp.allclose(diff, 0.0)) or bool(jnp.allclose(diff, pl))
        consumed += int(bool(jnp.allclose(diff, pl)) and not bool(jnp.allclose(pl, 0.0)))
    assert consumed > 0, "the consumer dropped the entire prescribed forcing"

    # differentiable producer -> consumer: grad of the consumer loss wrt the
    # prescribed payload (routed by name) is finite and non-zero somewhere.
    def loss(payload):
        pc = PrescribedComponent({name: payload}, provided_fluxes=(name,))
        t = forced.tendency(grid, state, pc.provide(grid, state)[name], None)
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(t))

    g_leaves = jax.tree.leaves(jax.grad(loss)(phys))
    assert g_leaves and all(jnp.all(jnp.isfinite(x)) for x in g_leaves)
    assert any(float(jnp.max(jnp.abs(x))) > 0.0 for x in g_leaves)
