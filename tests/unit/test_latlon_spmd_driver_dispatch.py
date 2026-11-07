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


@pytest.mark.parametrize("hs", [False, True])
def test_physics_fn_refuses_active_schemes(hs):
    """Active schemes, with or without Held-Suarez, must never reach the
    stateless closure: it would silently drop them."""
    cfg = _latlon_cfg(radiation="gray", convection="sbm", held_suarez_forcing=hs)
    with pytest.raises(RuntimeError, match="operator-split"):
        ModelDriver._latlon_spmd_physics_fn(_Stub(cfg))


@pytest.mark.parametrize("hs,schemes,expected", [
    (True, {}, False),                       # pure HS -> stateless lane
    (False, {}, False),                      # dynamics-only -> stateless lane
    (False, {"turbulence": "louis"}, True),
    (True, {"turbulence": "louis"}, True),   # HS + scheme -> operator split
    (True, {"radiation": "gray", "convection": "sbm"}, True),
])
def test_operator_split_routing(hs, schemes, expected):
    """HS is additive: any active scheme selects the operator-split lane
    (which applies the HS relaxation on top), as on every other lane."""
    cfg = _latlon_cfg(held_suarez_forcing=hs, **schemes)
    assert ModelDriver._operator_split_spmd_active(_Stub(cfg)) is expected
