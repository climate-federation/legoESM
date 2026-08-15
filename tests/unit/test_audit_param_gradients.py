"""Direct tests for the parameter reachability/differentiability audit.

The audit's job is to publish DEAD verdicts, so the tests that matter are the
NON-VACUITY ones: a harness that silently returned zero for everything would
report the whole registry as dead and look like a triumphant finding.  So we
assert both directions on real schemes — at least one parameter is detected
LIVE, and a parameter known to be gated off by a default flag is detected
DEAD — before any of the audit's output may be quoted.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.audit_param_gradients import (
    ATM_CATEGORIES,
    _numeric_leaves,
    _projection_weights,
    _single_scheme_config,
    _subconfig_of,
    _tunable_object,
    build_states,
    finite_difference_response,
    scheme_gradients,
    scheme_key_map,
)

_NLEV = 12
_DT = 600.0


def test_exactly_one_category_is_active():
    cfg = _single_scheme_config("convection", "emanuel")
    assert cfg.convection.scheme == "emanuel"
    for other in ("turbulence", "microphysics", "radiation",
                  "gravity_wave_drag"):
        assert getattr(cfg, other).scheme == "none", (
            f"{other} is live; a gradient could arrive through it and a dead "
            "parameter would read live")


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown category"):
        _single_scheme_config("cloud_fraction", "sundqvist")


def test_scheme_key_map_resolves_real_registry_keys():
    mapping = scheme_key_map(("convection",))
    assert "atm.conv.EmanuelConfig" in mapping
    category, scheme = mapping["atm.conv.EmanuelConfig"]
    assert (category, scheme) == ("convection", "emanuel")
    assert set(ATM_CATEGORIES) >= {c for c, _s in mapping.values()}


def test_projection_weights_match_the_tree_and_are_not_all_zero():
    import jax

    leaves = [jnp.zeros((2, 3)), jnp.zeros((4,))]
    w = _projection_weights(leaves, jax.random.PRNGKey(0))
    assert [jnp.shape(x) for x in w] == [(2, 3), (4,)]
    assert float(sum(jnp.sum(jnp.abs(x)) for x in w)) > 0.0


def test_numeric_leaves_drops_non_inexact():
    leaves = _numeric_leaves((jnp.ones(3), jnp.arange(3), None))
    assert len(leaves) == 1


def test_states_span_several_columns_with_seeded_condensate():
    states = build_states(_NLEV)
    assert len(states) >= 4, "one column cannot support a DEAD verdict"
    names = {n for n, _s, _sig, _g in states}
    assert len(names) == len(states)
    for _name, state, _sigma, _grid in states:
        for species in ("q_c", "q_r", "q_i", "q_s"):
            assert species in state.tracers
            # Seeded non-zero: a rate proportional to a species that is exactly
            # zero has an exactly zero derivative, which the audit would
            # publish as a dead parameter.
            assert float(jnp.min(state.tracers[species].data)) > 0.0


@pytest.mark.slow
def test_a_live_scheme_has_at_least_one_nonzero_gradient():
    """NON-VACUITY. If this fails, every DEAD verdict the audit prints is an
    artifact of the harness and none of them may be quoted."""
    grads, note = scheme_gradients(
        "convection", "emanuel", "atm.conv.EmanuelConfig",
        nlev=_NLEV, dt=_DT, states=build_states(_NLEV))
    assert note == "", f"tracing failed: {note}"
    assert grads, "no parameters resolved for emanuel"
    finite = [g for g in grads.values() if np.isfinite(g)]
    assert len(finite) == len(grads), "a non-finite gradient escaped"
    assert max(finite) > 0.0, (
        "no emanuel parameter has a non-zero gradient — the harness, not the "
        "scheme, is broken")


@pytest.mark.slow
def test_a_flag_gated_parameter_is_detected_dead():
    """The other direction.  ``EmanuelConfig.downdraft_efficiency`` is read
    ONLY inside ``if config.enable_unsaturated_downdraft:``, which defaults to
    False, so it can move nothing — both by gradient and by sweep."""
    states = build_states(_NLEV)
    grads, note = scheme_gradients(
        "convection", "emanuel", "atm.conv.EmanuelConfig",
        nlev=_NLEV, dt=_DT, states=states)
    assert note == ""
    name = "atm.conv.EmanuelConfig.downdraft_efficiency"
    assert name in grads, "the parameter left the registry; update this test"
    assert grads[name] == 0.0

    from legoesm.training.param_collector import build_trainable_params

    params = build_trainable_params(
        active_scheme_keys={"atm.conv.EmanuelConfig"}, tier="aggressive",
        dtype=jnp.float64)
    constraint = next(c for c in params.constraints if c.name == name)
    default = float(params.as_dict()[name])
    fd = finite_difference_response(
        "convection", "emanuel", constraint, default, dt=_DT, states=states)
    assert fd == 0.0, (
        "the sweep moved the tendency, so the parameter is BLOCKED (read, "
        "derivative severed) rather than DEAD — the two need different fixes")


def test_tunable_object_unwraps_clubb_but_passes_others_through():
    conv = _subconfig_of(_single_scheme_config("convection", "emanuel"),
                         "convection")
    assert _tunable_object(conv) is conv
    turb = _subconfig_of(_single_scheme_config("turbulence", "clubb"),
                         "turbulence")
    assert _tunable_object(turb) is turb.params


def test_every_state_carries_a_grid():
    """Gravity-wave drag and radiation read ``grid_lat`` off it.  Passing None
    made all 36 GWD parameters raise during tracing and read as
    non-differentiable — a statement about the probe, not the model."""
    for _name, _state, _sigma, grid in build_states(_NLEV):
        assert hasattr(grid, "grid_lat")
        assert hasattr(grid, "grid_lon")


@pytest.mark.slow
def test_gravity_wave_drag_traces_now():
    """The regression for the 36 false verdicts: GWD must produce a verdict,
    not a trace failure."""
    grads, note = scheme_gradients(
        "gravity_wave_drag", "rayleigh", "atm.gwd.RayleighConfig",
        nlev=_NLEV, dt=_DT, states=build_states(_NLEV))
    assert note == "", f"gravity-wave drag still fails to trace: {note}"
    assert grads
