"""Decision-logic of the ModelDriver lat-lon SPMD dispatch (A1) — the physics_fn
selection + the unsupported-config rejections. Tested via a lightweight stub so
no full (data-loading) ModelDriver construction is needed: the methods read only
``self.config``. The numerics of the run itself are covered by the gated
run_atm_latlon_spmd equivalence tests."""
from __future__ import annotations

import pytest

from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver


class _Stub:
    """Minimal stand-in exposing only what the dispatch helpers read."""
    def __init__(self, cfg):
        self.config = cfg


def _latlon_cfg(**over):
    cfg = ExperimentConfig()
    fields = dict(
        enable_latlon_spmd=True,
        grid=cfg.grid._replace(grid_type="latlon"),
        radiation="none", convection="none", turbulence="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
    )
    fields.update(over)  # caller overrides specific schemes
    return cfg._replace(**fields)


def test_physics_fn_held_suarez():
    """held_suarez_forcing=True -> the column-local Held-Suarez forcing."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
    fn = ModelDriver._latlon_spmd_physics_fn(
        _Stub(_latlon_cfg(held_suarez_forcing=True)))
    assert fn is held_suarez_forcing_latlon


def test_physics_fn_dynamics_only():
    """All parameterizations 'none', no Held-Suarez -> dynamics-only (None)."""
    fn = ModelDriver._latlon_spmd_physics_fn(_Stub(_latlon_cfg()))
    assert fn is None


def test_physics_fn_rejects_stateful_unified_physics():
    """A stateful scheme (radiation/convection/...) -> loud NotImplementedError,
    never a silent per-step reseed of the un-sharded PhysicsState carry."""
    cfg = _latlon_cfg(radiation="gray", convection="sbm")
    with pytest.raises(NotImplementedError, match="not yet SPMD-routed"):
        ModelDriver._latlon_spmd_physics_fn(_Stub(cfg))
