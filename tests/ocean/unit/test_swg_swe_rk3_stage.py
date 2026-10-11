"""Direct tests for the dispatched NEMO SWE RK3 stage program."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry
from legoesm.ocean.eos import nemo_qco_r3t_ratio
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def card_and_model():
    card = build_swg_card()
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    return card, model


def _entry(fill=0.0):
    arrays = {
        name: jnp.full((84, 84), fill, dtype=jnp.float64)
        for name in SWERK3StageEntry._fields
    }
    return SWERK3StageEntry(**arrays)


def test_stage1_zero_state_executes_jit_and_preserves_record_layout(card_and_model):
    card, model = card_and_model
    entry = _entry()
    out = model.replay_swe_rk3_stage(entry, card.dt_s, 1)
    assert out._fields == SWERK3StageEntry._fields
    for value in out:
        assert value.shape == (84, 84)
        assert value.dtype == jnp.float64
        assert np.array_equal(np.asarray(value), np.zeros((84, 84)))


def test_stage1_one_ulp_record_input_reaches_production_output(card_and_model):
    card, model = card_and_model
    entry = _entry()
    wet = np.asarray(card.recipe.land_mask, dtype=bool)
    at = tuple(np.argwhere(wet & np.roll(wet, -1, axis=1))[0])
    before = np.zeros((84, 84), dtype=np.float64)
    before[at] = np.nextafter(np.float64(0.0), np.float64(np.inf))
    moved = entry._replace(utau_before=jnp.asarray(before))
    out = model.replay_swe_rk3_stage(moved, card.dt_s, 1)
    assert np.asarray(out.utau_before)[at] == before[at]


def test_stage1_nonzero_pressure_is_source_literal_and_differentiable(
        card_and_model):
    card, model = card_and_model
    wet = jnp.asarray(card.recipe.land_mask)
    jj, ii = np.argwhere(np.asarray(wet, dtype=bool))[20]
    pattern = jnp.zeros((84, 84), dtype=jnp.float64).at[jj, ii].set(1.0)

    def one_output(amplitude):
        entry = _entry()._replace(ssh_k=amplitude * pattern)
        return model.replay_swe_rk3_stage(
            entry, card.dt_s, 1).u_k[jj, ii]

    amplitude = jnp.asarray(1.0e-3, dtype=jnp.float64)
    result = model.replay_swe_rk3_stage(
        _entry()._replace(ssh_k=amplitude * pattern), card.dt_s, 1)
    # stprk3.f90:172 and :189, retaining each fp64 source boundary.
    r1_e1u = np.float64(1.0) / np.float64(25_000.0)
    zrhs = np.float64(
        np.float64(-np.float64(0.02) * np.float64(0.0 - 1.0e-3))
        * r1_e1u)
    expected_u = np.float64(
        np.float64(3600.0 / 3.0) * zrhs)
    assert np.asarray(result.u_k)[jj, ii] == expected_u
    assert np.asarray(result.r3t_k)[jj, ii] == np.float64(1.0e-3 / 500.0)

    derivative = jax.grad(one_output)(amplitude)
    assert np.isfinite(float(derivative))
    assert float(derivative) != 0.0


def test_unfloored_qco_ratio_is_available_only_to_stage_boundary(card_and_model):
    card, _ = card_and_model
    eta = -500.0 * jnp.asarray(card.recipe.land_mask)
    ratio = nemo_qco_r3t_ratio(
        card.recipe.z_coord, eta, 500.0 * card.recipe.land_mask,
        evaluation="nemo_reciprocal")
    assert np.all(np.asarray(ratio)[np.asarray(card.recipe.land_mask, bool)] == -1.0)


def test_later_stage_and_full_step_dispatches_fail_closed(card_and_model):
    card, model = card_and_model
    with pytest.raises(ValueError, match="only stage 1 is landed"):
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 2)
    with pytest.raises(ValueError, match="not full-step ready"):
        model.step(card.recipe.initial_state, card.dt_s)


def test_stage_replay_refuses_a_non_swe_selector(card_and_model):
    card, _ = card_and_model
    bad = card.recipe.model_config._replace(
        outer_integrator="forward_euler", vorticity_scheme="ene_total")
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, bad)
    with pytest.raises(ValueError, match="requires outer_integrator"):
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 1)
