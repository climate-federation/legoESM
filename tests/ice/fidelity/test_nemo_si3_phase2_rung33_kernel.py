"""Unit and oracle-bound controls for the SI3 rung-3.3 C-grid aEVP arm."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest
from legoesm.core.precision import get_policy
from legoesm.ice.dynamics import (
    SI3CGridAEVPState,
    si3_cgrid_aevp_solver,
    si3_cgrid_deformation,
)
from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
    _forcing_for_state,
    build_ice_adv2d_rhg_card,
    step_ice_adv2d_rhg_card,
    validate_ice_adv2d_rhg_card,
)
from legoesm.ice.transport import (
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
)

_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final")
_ORACLE_GATE = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py"
)
_POINTWISE_BAR = 1.0e-15
_PLANTED_WEIGHT = 0.5001
_TEST_PERTURBATION = 1.0e-6


def _oracle_gate_module():
    spec = importlib.util.spec_from_file_location("rung33_oracle_gate", _ORACLE_GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card():
    with netCDF4.Dataset(_ROOT / "output.init_ice.nc") as dataset:
        sst = np.asarray(dataset["sst"][0]).T
    oracle_gate = _oracle_gate_module()
    _, entry = oracle_gate.read_frame(
        _ROOT / "oracle_ice_step_entry_kt00000001.bin"
    )
    return build_ice_adv2d_rhg_card(sst, entry)


def _normalized(oracle: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(oracle - candidate))) / max(
        1.0, float(np.max(np.abs(oracle)))
    )


def test_rung33_card_is_fp64_and_fail_closed():
    card = _card()
    assert get_policy().transcendentals == "libm"
    assert card.dynamics_scheme == "si3_aevp"
    assert card.rheology_staggering == "si3_c_grid"
    assert card.dynamics_config.n_subcycles == 100
    assert card.base.jpl == 1
    assert card.base.landfast is False
    assert all(value.dtype == jnp.float64 for value in card.initial_state.dynamics)
    assert all(value.dtype == jnp.float64 for value in card.metrics)
    with pytest.raises(ValueError, match="selector composition"):
        validate_ice_adv2d_rhg_card(
            card._replace(rheology_staggering="a_grid")
        )
    with pytest.raises(ValueError, match="zero Coriolis"):
        validate_ice_adv2d_rhg_card(
            card._replace(
                forcing_template=card.forcing_template._replace(
                    coriolis_t=card.forcing_template.coriolis_t.at[4, 4].set(
                        _TEST_PERTURBATION
                    )
                )
            )
        )


def test_prather_bridge_is_the_jitted_nemo_pack_then_reciprocal_products():
    card = _card()
    area = card.metrics.area_t
    wet = card.forcing_template.tmask_t.astype(bool)

    @jax.jit
    def bridge(value):
        packed = si3_prather_pack_intensives(value, area)
        return packed, si3_prather_unpack_intensives(packed, area, wet)

    packed, recovered = bridge(card.initial_state.contents)
    expected_packed = np.asarray(card.initial_state.contents) * np.asarray(area)[..., None]
    reciprocal = np.reciprocal(np.asarray(area))
    expected_recovered = expected_packed * reciprocal[..., None]
    expected_recovered *= np.asarray(wet)[..., None]
    np.testing.assert_array_equal(packed, expected_packed)
    np.testing.assert_array_equal(recovered, expected_recovered)


def test_stress_divergence_weight_plant_reaches_scored_card_trajectory():
    card = _card()
    baseline = step_ice_adv2d_rhg_card(card, completed_steps=0)
    planted = step_ice_adv2d_rhg_card(
        card,
        completed_steps=0,
        stress_divergence_outer_weight=_PLANTED_WEIGHT,
    )
    difference = float(
        jnp.max(jnp.abs(baseline.dynamics.u_ice_u - planted.dynamics.u_ice_u))
    )
    scale = float(jnp.max(jnp.abs(baseline.dynamics.u_ice_u)))
    assert scale > 0.0
    assert difference / max(1.0, scale) > _POINTWISE_BAR


def test_rn_ishlat_outside_resolved_orca1_arm_is_rejected_before_tracing():
    card = _card()
    assert card.dynamics_config.rn_ishlat == 2.0
    with pytest.raises(ValueError, match="no Frankenstein fallback"):
        si3_cgrid_aevp_solver(
            card.initial_state.dynamics,
            _forcing_for_state(card.forcing_template, card.initial_state, card.base),
            card.metrics,
            card.dynamics_config._replace(rn_ishlat=0.0),
        )


def test_drag_io_t_is_directionally_averaged_like_nemo():
    card = _card()
    forcing = _forcing_for_state(card.forcing_template, card.initial_state, card.base)
    ramp = jnp.arange(forcing.drag_io_t.shape[0], dtype=jnp.float64)[:, None]
    ramp = jnp.broadcast_to(ramp, forcing.drag_io_t.shape) * _TEST_PERTURBATION
    changed = forcing._replace(drag_io_t=forcing.drag_io_t + ramp)
    moving = card.initial_state.dynamics._replace(
        u_ice_u=jnp.full_like(card.initial_state.dynamics.u_ice_u, 0.1)
    )
    result = si3_cgrid_aevp_solver(
        moving,
        changed,
        card.metrics,
        card.dynamics_config._replace(n_subcycles=1),
    )
    assert np.all(np.isfinite(np.asarray(result.u_ice_u)))
    assert not np.array_equal(
        np.asarray(result.u_ice_u),
        np.asarray(
            si3_cgrid_aevp_solver(
                moving,
                forcing,
                card.metrics,
                card.dynamics_config._replace(n_subcycles=1),
            ).u_ice_u
        ),
    )


def test_f_stress_deliberately_has_no_t_presence_mask():
    card = _card()
    one = jnp.ones_like(card.initial_state.dynamics.stress1_t)
    zero = jnp.zeros_like(one)
    forcing = card.forcing_template._replace(
        concentration_t=zero,
        ice_volume_t=zero,
        snow_volume_t=zero,
        pond_volume_t=zero,
        lid_volume_t=zero,
    )
    state = SI3CGridAEVPState(zero, zero, one, one, one)
    config = card.dynamics_config._replace(n_subcycles=1)
    result = si3_cgrid_aevp_solver(state, forcing, card.metrics, config)
    assert float(jnp.max(jnp.abs(result.stress1_t))) == 0.0
    assert float(jnp.max(jnp.abs(result.stress2_t))) == 0.0
    # icedyn_rhg_evp.F90:488-489 has no zmsk on stress12.
    assert float(jnp.max(jnp.abs(result.stress12_f))) > 0.0


def test_final_deformation_diagnostic_binds_to_velocity_gradients():
    card = _card()
    forcing = _forcing_for_state(
        card.forcing_template, card.initial_state, card.base
    )._replace(
        concentration_t=jnp.ones_like(card.forcing_template.concentration_t)
    )
    zero_divergence, zero_deformation = si3_cgrid_deformation(
        card.initial_state.dynamics, forcing, card.metrics, card.dynamics_config
    )
    np.testing.assert_array_equal(zero_divergence, 0.0)
    np.testing.assert_array_equal(zero_deformation, 0.0)

    impulse = card.initial_state.dynamics._replace(
        u_ice_u=card.initial_state.dynamics.u_ice_u.at[20, 20].set(
            _TEST_PERTURBATION
        )
    )
    divergence, deformation = si3_cgrid_deformation(
        impulse, forcing, card.metrics, card.dynamics_config
    )
    assert float(jnp.max(jnp.abs(divergence))) > 0.0
    assert float(jnp.max(deformation)) > 0.0


def test_full_hundred_subcycle_kernel_jits_and_differentiates():
    card = _card()
    forcing = _forcing_for_state(card.forcing_template, card.initial_state, card.base)
    axis = jnp.arange(forcing.air_stress_u_t.shape[0], dtype=jnp.float64)
    perturbation = _TEST_PERTURBATION * (
        axis[:, None] + axis[None, :]
    )
    initial = card.initial_state.dynamics._replace(
        u_ice_u=card.initial_state.dynamics.u_ice_u + perturbation,
        v_ice_v=card.initial_state.dynamics.v_ice_v - perturbation,
    )

    def objective(air_stress):
        changed = forcing._replace(air_stress_u_t=air_stress)
        result = si3_cgrid_aevp_solver(
            initial,
            changed,
            card.metrics,
            card.dynamics_config,
            differentiable=True,
        )
        return jnp.sum(result.u_ice_u[2:-2, 2:-2])

    compiled = jax.jit(objective)
    value = compiled(forcing.air_stress_u_t)
    gradient = jax.grad(compiled)(forcing.air_stress_u_t)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_kt1_full_dynamics_is_bit_exact_from_pinned_intensive_entry():
    """Pin the Round-11 bridge result for all five aEVP carries."""

    card = _card()
    candidate = step_ice_adv2d_rhg_card(card, completed_steps=0).dynamics
    gate = _oracle_gate_module()
    _, frame = gate.read_frame(
        _ROOT / "oracle_ice_step_entry_kt00000002.bin"
    )
    rows = {}
    for name, value in (
        ("u_ice", candidate.u_ice_u),
        ("v_ice", candidate.v_ice_v),
        ("stress1_i", candidate.stress1_t),
        ("stress2_i", candidate.stress2_t),
        ("stress12_i", candidate.stress12_f),
    ):
        rows[name] = _normalized(
            gate._interior(frame[name]),
            np.asarray(value)[2:-2, 2:-2],
        )
    assert rows == {
        "u_ice": 0.0,
        "v_ice": 0.0,
        "stress1_i": 0.0,
        "stress2_i": 0.0,
        "stress12_i": 0.0,
    }
