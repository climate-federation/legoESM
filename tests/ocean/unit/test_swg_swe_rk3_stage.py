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
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_swg_card, swg_horizontal_coordinates)
from legoesm.ocean.state import OceanSurfaceForcing


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


@pytest.mark.parametrize("stage", (1, 2, 3))
def test_zero_state_executes_jit_and_preserves_record_layout(
        card_and_model, stage):
    card, model = card_and_model
    entry = _entry()
    out = model.replay_swe_rk3_stage(
        entry, card.dt_s, stage, entry if stage != 1 else None)
    assert out._fields == SWERK3StageEntry._fields
    for value in out:
        assert value.shape == (84, 84)
        assert value.dtype == jnp.float64
        assert np.array_equal(np.asarray(value), np.zeros((84, 84)))


def test_stage1_one_ulp_consumed_input_reaches_production_output(card_and_model):
    card, model = card_and_model
    entry = _entry()
    wet = np.asarray(card.recipe.land_mask, dtype=bool)
    at = tuple(np.argwhere(wet & np.roll(wet, -1, axis=1))[0])
    base = np.ones((84, 84), dtype=np.float64)
    moved_u = base.copy()
    moved_u[at] = np.nextafter(np.float64(1.0), np.float64(np.inf))
    baseline = model.replay_swe_rk3_stage(
        entry._replace(u_k=jnp.asarray(base)), card.dt_s, 1)
    moved = model.replay_swe_rk3_stage(
        entry._replace(u_k=jnp.asarray(moved_u)), card.dt_s, 1)
    assert np.any(np.asarray(moved.ssh_k) != np.asarray(baseline.ssh_k))


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
    with pytest.raises(ValueError, match="require the whole-step before"):
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 2)
    with pytest.raises(ValueError, match="surface stress"):
        model.step(card.recipe.initial_state, card.dt_s)


def test_normal_step_uses_the_jitted_three_stage_production_program(card_and_model):
    card, model = card_and_model
    src = swg_horizontal_coordinates()
    tau_u = -0.2 * np.cos(np.pi * src["gphit"] / 2_000_000.0)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u),
        tau_j_native=jnp.zeros_like(jnp.asarray(tau_u)))
    out = model.step(card.recipe.initial_state, card.dt_s,
                     surface_forcing=forcing)
    assert np.any(np.asarray(out.u.data) != 0.0)
    assert np.array_equal(np.asarray(out.uu_b.data),
                          np.asarray(card.recipe.initial_state.uu_b.data))
    assert np.array_equal(np.asarray(out.vv_b.data),
                          np.asarray(card.recipe.initial_state.vv_b.data))
    assert out.eta.data.dtype == jnp.float64
    # The first whole step is the shipped non-degenerate forcing transition;
    # reverse-mode must cross the same ordinary step() dispatch.
    def objective(scale):
        forced = forcing._replace(tau_i_native=scale * forcing.tau_i_native)
        return jnp.sum(model.step(
            card.recipe.initial_state, card.dt_s,
            surface_forcing=forced).u.data)
    assert np.isfinite(float(jax.grad(objective)(jnp.asarray(1.0))))


def test_nonzero_manufactured_stage_matches_literal_nemo_statements(
        card_and_model):
    """Exercise C2, ENS, pressure, continuity, QCO, and face masking together."""
    card, model = card_and_model
    masks = np.asarray(card.recipe.land_mask, dtype=bool)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    uf, vf = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    um = np.asarray(uf[:, 1:], dtype=bool)
    vm = np.asarray(vf[1:, :], dtype=bool)
    jj, ii = np.indices((84, 84), dtype=np.float64)
    u = um * (1.0e-3 + 2.0e-6 * ii + 3.0e-6 * jj)
    v = vm * (-7.0e-4 + 1.0e-6 * ii - 2.0e-6 * jj)
    ssh = masks * (1.0e-2 + 4.0e-5 * ii + 5.0e-5 * jj)
    entry = _entry()._replace(
        ssh_k=jnp.asarray(ssh), u_k=jnp.asarray(u), v_k=jnp.asarray(v),
        u_before=jnp.asarray(u), v_before=jnp.asarray(v))
    out = model.replay_swe_rk3_stage(entry, card.dt_s, 1)

    j0, i0 = 40, 40
    dx = np.float64(25_000.0)
    height = np.float64(500.0)
    dt = np.float64(1_200.0)
    ke = np.float64(0.25) * (
        (np.roll(u, 1, axis=1) ** 2 + u ** 2)
        + (np.roll(v, 1, axis=0) ** 2 + v ** 2))
    zeta = (((np.roll(v, -1, axis=1) - v) * dx
             - (np.roll(u, -1, axis=0) - u) * dx) / (dx * dx))
    fmask = np.asarray(
        card.recipe.z_coord.nemo_een_barotropic.fe3mask[..., 0])
    q = ((swg_horizontal_coordinates()["ff_f"] + zeta * fmask) / height)
    fu = dx * height * u
    fv = dx * height * v
    keg_u = -(ke[j0, i0 + 1] - ke[j0, i0]) / dx
    keg_v = -(ke[j0 + 1, i0] - ke[j0, i0]) / dx
    ens_u = (np.float64(0.125) / dx
             * ((fv[j0 - 1, i0] + fv[j0 - 1, i0 + 1])
                + (fv[j0, i0] + fv[j0, i0 + 1]))
             * (q[j0 - 1, i0] + q[j0, i0]))
    ens_v = (-np.float64(0.125) / dx
             * ((fu[j0, i0 - 1] + fu[j0 + 1, i0 - 1])
                + (fu[j0, i0] + fu[j0 + 1, i0]))
             * (q[j0, i0 - 1] + q[j0, i0]))
    hpg_u = -np.float64(0.02) * (ssh[j0, i0 + 1] - ssh[j0, i0]) / dx
    hpg_v = -np.float64(0.02) * (ssh[j0 + 1, i0] - ssh[j0, i0]) / dx
    expected_u = u[j0, i0] + dt * (keg_u + ens_u + hpg_u)
    expected_v = v[j0, i0] + dt * (keg_v + ens_v + hpg_v)
    fu0 = np.float64(np.float64(dx * height) * u[j0, i0])
    fuw = np.float64(np.float64(dx * height) * u[j0, i0 - 1])
    fv0 = np.float64(np.float64(dx * height) * v[j0, i0])
    fvs = np.float64(np.float64(dx * height) * v[j0 - 1, i0])
    numerator = np.float64(
        np.float64(fu0 - fuw) + np.float64(fv0 - fvs))
    transport_div = np.float64(
        numerator * np.float64(1.0 / (dx * dx)))
    hdiv = np.float64(transport_div / height)
    div = np.float64(height * hdiv)
    expected_ssh = np.float64(ssh[j0, i0] - np.float64(dt * div))
    assert np.asarray(out.u_k)[j0, i0] == expected_u
    assert np.asarray(out.v_k)[j0, i0] == expected_v
    assert np.asarray(out.ssh_k)[j0, i0] == expected_ssh
    assert np.asarray(out.r3t_k)[j0, i0] == np.float64(
        expected_ssh * np.float64(1.0 / height))
    assert all(value != 0.0 for value in (
        keg_u, keg_v, ens_u, ens_v, hpg_u, hpg_v, div))
    dry_u = tuple(np.argwhere(~um)[0])
    dry_v = tuple(np.argwhere(~vm)[0])
    assert np.asarray(out.u_k)[dry_u] == 0.0
    assert np.asarray(out.v_k)[dry_v] == 0.0


def test_stage3_wind_reads_distinct_before_and_now_stresses_once(card_and_model):
    card, model = card_and_model
    tau_before = np.float64(0.25)
    tau_now = np.float64(0.5)
    entry = _entry()._replace(
        utau_before=jnp.full((84, 84), tau_before, dtype=jnp.float64),
        utau_now=jnp.full((84, 84), tau_now, dtype=jnp.float64))
    out = model.replay_swe_rk3_stage(entry, card.dt_s, 3, _entry())
    at = (40, 40)
    rho0 = np.float64(card.recipe.model_config.constants.rho_0)
    height = np.float64(np.asarray(card.recipe.z_coord.dz_ref)[0])
    r1_rho0 = np.float64(np.float64(1.0) / rho0)
    half_r1_rho0 = np.float64(np.float64(0.5) * r1_rho0)
    tau_sum = np.float64(tau_before + tau_now)
    wind = np.float64(np.float64(half_r1_rho0 * tau_sum) / height)
    expected = np.float64(np.float64(card.dt_s) * wind)
    assert np.asarray(out.u_k)[at] == expected


def test_stage3_drag_uses_whole_step_before_velocity(card_and_model):
    card, model = card_and_model
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

    u_mask, _ = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    before_u = np.asarray(u_mask[:, 1:], dtype=np.float64) * np.float64(0.25)
    before = _entry()._replace(u_k=jnp.asarray(before_u))
    out = model.replay_swe_rk3_stage(_entry(), card.dt_s, 3, before)
    at = (40, 40)
    drag = np.float64(card.recipe.model_config.bottom_drag.bottom_drag_r)
    drag_rhs = np.float64(-np.float64(drag * before_u[at]))
    expected = np.float64(before_u[at] + np.float64(card.dt_s) * drag_rhs)
    assert np.asarray(out.u_k)[at] == expected


def test_stage_replay_refuses_a_non_swe_selector(card_and_model):
    card, _ = card_and_model
    bad = card.recipe.model_config._replace(
        outer_integrator="forward_euler", vorticity_scheme="ene_total")
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, bad)
    with pytest.raises(ValueError, match="requires outer_integrator"):
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 1)


@pytest.mark.parametrize("field,value", (
    ("lateral_viscosity_operator", "vector_laplacian"),
    ("lateral_viscosity_e3_weighting", "off"),
    ("lateral_viscosity_coefficient_source", "typo"),
))
def test_stage_replay_refuses_wrong_or_unknown_ldf_dispatch(
        card_and_model, field, value):
    card, _ = card_and_model
    bad = card.recipe.model_config._replace(**{field: value})
    with pytest.raises(ValueError, match="viscosity|coefficient_source"):
        model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, bad)
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 3, _entry())
