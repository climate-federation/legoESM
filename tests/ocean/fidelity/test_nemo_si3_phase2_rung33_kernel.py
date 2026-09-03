"""Unit and oracle-bound controls for the SI3 rung-3.3 C-grid aEVP arm."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest
from legoesm.ice.dynamics import (
    SI3CGridAEVPState,
    _si3_stress_divergence,
    si3_cgrid_aevp_solver,
)
from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
    _forcing_for_state,
    build_ice_adv2d_rhg_card,
    step_ice_adv2d_rhg_card,
    validate_ice_adv2d_rhg_card,
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
    return build_ice_adv2d_rhg_card(sst)


def _normalized(oracle: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(oracle - candidate))) / max(
        1.0, float(np.max(np.abs(oracle)))
    )


def test_rung33_card_is_fp64_and_fail_closed():
    card = _card()
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


def test_stress_divergence_weight_plant_binds_on_the_card():
    card = _card()
    dynamics = step_ice_adv2d_rhg_card(card, completed_steps=0).dynamics
    force = _si3_stress_divergence(
        dynamics.stress1_t,
        dynamics.stress2_t,
        dynamics.stress12_f,
        card.metrics,
    )
    planted = _si3_stress_divergence(
        dynamics.stress1_t,
        dynamics.stress2_t,
        dynamics.stress12_f,
        card.metrics,
        outer_weight=_PLANTED_WEIGHT,
    )
    difference = max(
        float(jnp.max(jnp.abs(actual - changed)))
        for actual, changed in zip(force, planted)
    )
    scale = max(float(jnp.max(jnp.abs(value))) for value in force)
    assert scale > 0.0
    assert difference / max(1.0, scale) > _POINTWISE_BAR


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


def test_kt1_full_dynamics_measurement_is_loud_about_stress_debt():
    """Pin the current measured boundary; this is not an exactness claim."""

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
    assert rows["u_ice"] <= _POINTWISE_BAR
    assert rows["v_ice"] <= _POINTWISE_BAR
    assert rows["stress1_i"] > _POINTWISE_BAR
    assert rows["stress2_i"] > _POINTWISE_BAR
    assert rows["stress12_i"] > _POINTWISE_BAR
