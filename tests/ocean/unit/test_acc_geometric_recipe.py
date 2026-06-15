"""ACC recipe — GEOMETRIC closure override (Stage-0 calibration twin).

Covers ``build_acc_recipe(eke_override=...)`` + ``geometric_eke_config``: the
default Eden-Greatbatch recipe stays bit-identical, the geometric override
threads ``closure="geometric"`` into ``model_config.gm_redi.eke`` AND seeds a
2-D depth-integrated eke field, and a forward step preserves the scan-carry
pytree structure (the regression for the eke-units metadata mismatch — the
geometric step writes ``units="m^3/s^2"`` and the seed must match or a
``jax.lax.scan`` free run breaks on step 1).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
from legoesm.ocean.fidelity.veros_acc_recipe import (
    build_acc_recipe,
    geometric_eke_config,
)
from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_geometric_eke_config_is_clean_geometric():
    cfg = geometric_eke_config(GeometricConfig(alpha=0.05))
    assert cfg.closure == "geometric"
    assert cfg.geometric is not None and cfg.geometric.alpha == 0.05
    # The EG-only flags the geometric cross-validation rejects must be at default.
    assert cfg.eke_3d is False
    assert cfg.isopycnal_diffusion is False
    assert cfg.gm_source_mode == "parameterized"
    assert cfg.source_kdiss_h is False


def test_default_recipe_unchanged_eden_greatbatch():
    """eke_override=None keeps the Eden-Greatbatch ACC default (3-D eke)."""
    r = build_acc_recipe(with_surface_forcing=True)
    eke = r.model_config.gm_redi.eke
    assert eke.closure == "eden_greatbatch"
    assert eke.eke_3d is True
    assert r.initial_state.eke.dims == ("lat", "lon", "level")
    assert r.initial_state.eke.units == "m^2/s^2"


def test_geometric_override_threads_config_and_seeds_2d():
    geom = GeometricConfig()
    r = build_acc_recipe(with_surface_forcing=True,
                         eke_override=geometric_eke_config(geom))
    eke = r.model_config.gm_redi.eke
    assert eke.closure == "geometric"
    assert eke.geometric.alpha == geom.alpha
    # Depth-integrated 2-D field, units matching the geometric step's output.
    assert r.initial_state.eke.dims == ("lat", "lon")
    assert r.initial_state.eke.units == "m^3/s^2"
    assert r.initial_state.eke.data.shape == (r.grid.n_lat, r.grid.n_lon)
    # The faithful free-run stepping bundle still rides along.
    assert r.model_config.outer_integrator == "ab2"
    assert r.model_config.barotropic_solver == "rigid_lid"


def test_forward_step_preserves_eke_carry_pytree():
    """One geometric step must NOT change the eke Field's pytree metadata
    (dims + units) — the regression for the m^2/s^2 vs m^3/s^2 seed bug."""
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from jax.tree_util import tree_structure

    r = build_acc_recipe(with_surface_forcing=True,
                         eke_override=geometric_eke_config(GeometricConfig()))
    cfg = r.model_config
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    state = r.initial_state

    # Seed the AB2 + rigid-lid carries (constant-pytree scan requirement).
    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    state = state._replace(
        T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
        u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    rl = model._ensure_rigid_lid_data(state)
    _zV = jnp.zeros((r.grid.n_lat + 1, r.grid.n_lon + 1), dtype=state.u.data.dtype)
    _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
    state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                           dpsin=_zI, dpsin_prev=_zI)

    nxt = model.step(state, 43200.0, surface_forcing=r.wind_forcing)

    # The eke Field metadata is identical before/after the step.
    assert nxt.eke.dims == state.eke.dims
    assert nxt.eke.units == state.eke.units == "m^3/s^2"
    # Whole-state pytree structure is stable (what jax.lax.scan enforces).
    assert tree_structure(nxt) == tree_structure(state)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))
    assert bool(jnp.all(jnp.isfinite(nxt.eke.data)))
