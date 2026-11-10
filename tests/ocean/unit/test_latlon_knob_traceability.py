"""Ocean-WB arm (c): the lat-lon C-grid ocean's scalar knobs are trainable.

Each knob is passed as a TRACED value (under ``jax.jit``, as in training: eager ``jax.grad`` in
JAX 0.10 linearises around a concrete primal, so a Python ``x > 0`` silently passes there) through ``_step_impl(config=...)`` and must give a finite,
nonzero gradient of the two-step state.  Before ``is_active`` the step gated each term with a
Python ``if cfg.X > 0`` (TracerBoolConversionError) or ``float(cfg.X)``
(ConcretizationTypeError).  Static Python floats must keep the historical branch choice.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.param_overrides import is_active, is_traced

KNOBS = ("A_h", "B_h", "A_h_floor", "C_smag_lap", "B_h_barotropic", "K_h", "A_v", "K_v",
         "bottom_drag_r", "bottom_drag_bg_velocity")


def test_is_active_static_semantics():
    assert is_active(1e-3) and not is_active(0.0) and not is_active(-1.0)
    assert is_active(np.float64(2.0)) and not is_active(np.float32(0.0))
    assert not is_traced(1.0)
    seen = []
    jax.make_jaxpr(lambda x: seen.append((is_traced(x), is_active(x))) or x)(0.0)
    assert seen == [(True, True)]


@pytest.fixture(scope="module")
def problem():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(3, H_max=500.0)
    config = LatLonCGridOceanConfig.from_flat(
        use_conservation_fixer=False, enable_runtime_checks=False, n_barotropic_substeps=2,
        differentiable_barotropic=True, implicit_vertical_mixing=True,
        A_h=2e5, B_h=5e9, A_h_lat_scaling=True, A_h_floor=1e5, C_smag_lap=0.15,  # floor/A_h=0.5 binds poleward of 60deg
        B_h_barotropic=1e14, K_h=1e3, A_v=1e-3, K_v=1e-4,
        bottom_drag_r=2.5e-3, bottom_drag_bg_velocity=0.1)
    model = LatLonCGridOceanModel(grid, z_coord, config)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord, land_lat_threshold=90.0)
    rng = np.random.default_rng(0)
    state = state._replace(
        T=state.T.replace(data=state.T.data + jnp.asarray(rng.normal(0, 1, state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(rng.normal(0, .1, state.S.data.shape))),
        u=state.u.replace(data=jnp.asarray(rng.normal(0, .2, state.u.data.shape))),
        v=state.v.replace(data=jnp.asarray(rng.normal(0, .2, state.v.data.shape))))
    model.prime_step_caches(state)
    return model, state, config


@pytest.mark.parametrize("name", KNOBS)
def test_knob_gradient_finite_nonzero(problem, name):
    model, state, config = problem
    v0 = float(config.flat_get(name))

    def loss(x):
        cfg = config.replace_flat(**{name: x})
        s = state
        for _ in range(2):
            s = model._step_impl(s, 600.0, config=cfg)
        return (jnp.sum(s.u.data ** 2) + jnp.sum(s.v.data ** 2) + jnp.sum(s.T.data ** 2)
                + jnp.sum(s.S.data ** 2))

    g = float(jax.jit(jax.grad(loss))(jnp.asarray(v0)))   # jit: eager grad lets `x > 0` pass
    assert np.isfinite(g) and g != 0.0, (name, v0, g)


def test_arm_c_trainable_set_excludes_stabilisers():
    from legoesm.training.trainable_ocean_params import (
        LATLON_DYNAMICS_TRAINABLE, LATLON_FIXED_STABILISERS, OceanParamSpec,
        TrainableOceanParams, names)
    from legoesm.training.trainable_params import ParamConstraint

    trained = set(names(LATLON_DYNAMICS_TRAINABLE))
    assert trained == set(KNOBS) - set(LATLON_FIXED_STABILISERS) and len(trained) == 8
    bad = LATLON_DYNAMICS_TRAINABLE + (
        OceanParamSpec(ParamConstraint("A_h_floor", 1e2, 1e4, "sigmoid"), True, 2e3),)
    with pytest.raises(ValueError, match="stabilisers"):
        TrainableOceanParams.from_defaults(bad).to_latlon_config(None)


def test_arm_c_gate_every_raw_leaf_has_gradient(problem):
    """The gradient-access gate on the real trainable object, seeded at the deck values."""
    from legoesm.training.trainable_ocean_params import (
        LATLON_DYNAMICS_TRAINABLE, TrainableOceanParams)

    model, state, config = problem
    p0 = TrainableOceanParams.from_defaults(LATLON_DYNAMICS_TRAINABLE)
    for k, v in p0.as_dict().items():
        assert np.isclose(float(v), float(config.flat_get(k)), rtol=1e-9), k

    def loss(raw):
        cfg = p0.__class__(raw=raw, specs=p0.specs).to_latlon_config(config)
        s = state
        for _ in range(2):
            s = model._step_impl(s, 600.0, config=cfg)
        return (jnp.sum(s.u.data ** 2) + jnp.sum(s.v.data ** 2) + jnp.sum(s.T.data ** 2)
                + jnp.sum(s.S.data ** 2))

    g = np.asarray(jax.jit(jax.grad(loss))(p0.raw))
    dead = [s.constraint.name for s, gi in zip(p0.specs, g) if not (np.isfinite(gi) and gi != 0)]
    assert not dead, f"no gradient for {dead}: {dict(zip([s.constraint.name for s in p0.specs], g))}"
