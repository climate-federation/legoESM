"""Direct tests for the dispatched NEMO SWE RK3 stage program."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.swe_rk3 import (
    SWERK3StageEntry,
    swe_rk3_stage,
    swe_rotated_potential_forcing,
)
from legoesm.ocean.eos import nemo_qco_r3t_ratio
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_NAMELIST,
    build_swg_card,
    swg_horizontal_coordinates,
)
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


def test_rotated_potential_force_is_closed_wall_nondivergent_and_zero_integral():
    rng = np.random.default_rng(7281)
    potential = rng.standard_normal((12, 13))
    vertex = np.ones_like(potential, dtype=bool)
    vertex[[0, -1], :] = False
    vertex[:, [0, -1]] = False
    e2u = 40_000.0 + rng.random(potential.shape) * 20_000.0
    e1v = 40_000.0 + rng.random(potential.shape) * 20_000.0
    e3u = 400.0 + rng.random(potential.shape) * 200.0
    e3v = 400.0 + rng.random(potential.shape) * 200.0
    mask = np.ones_like(potential)
    force_u, force_v = swe_rotated_potential_forcing(
        jnp.asarray(potential), jnp.asarray(e2u), jnp.asarray(e1v),
        jnp.asarray(e3u), jnp.asarray(e3v), jnp.asarray(mask),
        jnp.asarray(mask), jnp.asarray(vertex))
    transport_u = np.asarray(force_u) * e2u * e3u
    transport_v = np.asarray(force_v) * e1v * e3v
    divergence = (transport_u - np.roll(transport_u, 1, axis=1)
                  + transport_v
                  - np.concatenate([np.zeros_like(transport_v[:1]),
                                    transport_v[:-1]], axis=0))
    assert np.max(np.abs(divergence)) < 2.0e-15
    assert np.max(np.abs(transport_u[:, [0, -1]])) == 0.0
    assert np.max(np.abs(transport_v[[0, -1], :])) == 0.0
    assert abs(np.sum(transport_u)) < 2.0e-14
    assert abs(np.sum(transport_v)) < 2.0e-14


def test_rotated_potential_orientation_signs_and_all_four_corners():
    potential = np.zeros((7, 8), dtype=np.float64)
    potential[3, 4] = 2.0
    vertex = np.ones_like(potential, dtype=bool)
    vertex[[0, -1], :] = False
    vertex[:, [0, -1]] = False
    mask = np.ones_like(potential)
    one = np.ones_like(potential)
    force_u, force_v = swe_rotated_potential_forcing(
        potential, one, one, one, one, mask, mask, vertex)
    force_u, force_v = np.asarray(force_u), np.asarray(force_v)
    assert force_u[3, 4] == 2.0 and force_u[4, 4] == -2.0
    assert force_v[3, 4] == -2.0 and force_v[3, 5] == 2.0
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert force_u[corner] == force_v[corner] == 0.0
    # Flipping only V is a plausible rotated-gradient sign error; it must
    # destroy the matched divergence cancellation by a large margin.
    wrong_divergence = (force_u - np.roll(force_u, 1, axis=1)
                        - force_v
                        + np.concatenate([np.zeros_like(force_v[:1]),
                                          force_v[:-1]], axis=0))
    assert np.max(np.abs(wrong_divergence)) >= 4.0


def test_absent_potential_is_bit_identical_to_certified_stage(card_and_model):
    card, model = card_and_model
    entry = _entry()
    direct = model.replay_swe_rk3_stage(entry, card.dt_s, 1)
    from legoesm.ocean.dynamics.swe_rk3 import swe_rk3_stage
    absent = jax.jit(lambda value: swe_rk3_stage(
        value, value, card.dt_s, 1, card.recipe.grid, card.recipe.z_coord,
        card.recipe.land_mask, card.recipe.model_config,
        potential_f=None))(entry)
    for left, right in zip(direct, absent, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))
def test_swe_face_masks_follow_the_config_not_the_caller_scope(
        card_and_model, monkeypatch):
    import legoesm.ocean.dynamics.swe_rk3 as swe
    from legoesm.grids.halo_latlon import (
        get_meridionally_periodic, meridional_periodicity)

    card, _ = card_and_model
    assert card.recipe.model_config.meridionally_periodic is False
    seen = []
    original = swe.compute_face_masks_3d

    def recording(*args, **kwargs):
        seen.append(get_meridionally_periodic())
        return original(*args, **kwargs)

    monkeypatch.setattr(swe, "compute_face_masks_3d", recording)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.zeros((84, 84), dtype=jnp.float64),
        tau_j_native=jnp.zeros((84, 84), dtype=jnp.float64))
    with meridional_periodicity(True):
        swe.swe_rk3_stage(
            _entry(), _entry(), card.dt_s, 1, card.recipe.grid,
            card.recipe.z_coord, card.recipe.land_mask,
            card.recipe.model_config)
        swe.swe_rk3_step(
            card.recipe.initial_state, card.dt_s, forcing, card.recipe.grid,
            card.recipe.z_coord, card.recipe.model_config)
        stress = (forcing.tau_i_native, forcing.tau_j_native)
        swe._entry_from_state(
            card.recipe.initial_state, stress, stress, card.recipe.grid,
            card.recipe.z_coord, card.recipe.model_config)
        fresh = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        n_eager = len(seen)
        fresh.replay_swe_rk3_stage(_entry(), card.dt_s, 1)
        fresh.step(card.recipe.initial_state, card.dt_s,
                   surface_forcing=forcing)
    assert len(seen) > n_eager > 0 and not any(seen)


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


@pytest.mark.parametrize("override", ("grid", "vertex_mask"))
def test_swe_step_refuses_grid_overrides(card_and_model, override):
    card, model = card_and_model
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.zeros((84, 84), dtype=jnp.float64),
        tau_j_native=jnp.zeros((84, 84), dtype=jnp.float64))
    kwargs = {override: (card.recipe.grid if override == "grid"
                         else jnp.ones((84, 84), dtype=jnp.float64))}
    with pytest.raises(ValueError, match="does not accept grid overrides"):
        model.step(card.recipe.initial_state, card.dt_s,
                   surface_forcing=forcing, **kwargs)


def test_swe_step_refuses_time_argument(card_and_model):
    card, model = card_and_model
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.zeros((84, 84), dtype=jnp.float64),
        tau_j_native=jnp.zeros((84, 84), dtype=jnp.float64))
    with pytest.raises(ValueError, match="surface stress input"):
        model.step(card.recipe.initial_state, card.dt_s,
                   surface_forcing=forcing, t_seconds=0.0)
    with pytest.raises(ValueError, match="surface stress input"):
        model._step_jitted(card.recipe.initial_state, card.dt_s, None,
                           forcing, None, t_seconds=0.0)


def test_swe_jitted_step_refuses_grid_override(card_and_model):
    card, model = card_and_model
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.zeros((84, 84), dtype=jnp.float64),
        tau_j_native=jnp.zeros((84, 84), dtype=jnp.float64))
    with pytest.raises(ValueError, match="does not accept grid overrides"):
        model._step_jitted(card.recipe.initial_state, card.dt_s, None,
                           forcing, None, grid=card.recipe.grid)


def test_normal_step_uses_the_jitted_three_stage_production_program(card_and_model):
    card, model = card_and_model
    src = swg_horizontal_coordinates()
    tau_u = (-SWG_NAMELIST.rn_tau_pa
             * np.cos(np.pi * src["gphit"] / SWG_NAMELIST.rn_domsiz_m))
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


def test_normal_step_materializes_each_jitted_stage(card_and_model):
    """The old monolithic JIT is a binding one-bit step-six plant."""
    card, model = card_and_model
    src = swg_horizontal_coordinates()
    tau_u = (-SWG_NAMELIST.rn_tau_pa
             * np.cos(np.pi * src["gphit"] / SWG_NAMELIST.rn_domsiz_m))
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.zeros_like(jnp.asarray(tau_u), dtype=jnp.float64))
    state = card.recipe.initial_state
    for _ in range(5):
        state = model.step(state, card.dt_s, surface_forcing=forcing)
    materialized = model.step(state, card.dt_s, surface_forcing=forcing)
    fused = model._step_jitted(state, card.dt_s, None, forcing, None)
    unequal = np.asarray(materialized.u.data) != np.asarray(fused.u.data)
    assert np.count_nonzero(unequal) == 1
    assert np.max(np.abs(
        np.asarray(materialized.u.data) - np.asarray(fused.u.data))) == np.float64(
            4.336808689942018e-19)


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


def test_stage3_viscosity_reads_current_unindexed_r3f(card_and_model):
    card, model = card_and_model
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

    u_mask, v_mask = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    jj, ii = np.indices((84, 84), dtype=np.float64)
    before_u = np.asarray(u_mask[:, 1:]) * (1.0e-3 + 2.0e-6 * jj)
    before_v = np.asarray(v_mask[1:, :]) * (-7.0e-4 + 3.0e-6 * ii)
    base_entry = _entry()
    base_before = _entry()._replace(
        u_k=jnp.asarray(before_u), v_k=jnp.asarray(before_v))
    r3f = np.zeros((84, 84), dtype=np.float64)
    r3f[40, 40] = np.float64(0.125)

    base = model.replay_swe_rk3_stage(
        base_entry, card.dt_s, 3, base_before)
    current = model.replay_swe_rk3_stage(
        base_entry._replace(r3f=jnp.asarray(r3f)), card.dt_s, 3, base_before)
    stale_before = model.replay_swe_rk3_stage(
        base_entry, card.dt_s, 3,
        base_before._replace(r3f=jnp.asarray(r3f)))

    assert (np.any(np.asarray(current.u_k) != np.asarray(base.u_k))
            or np.any(np.asarray(current.v_k) != np.asarray(base.v_k)))
    np.testing.assert_array_equal(np.asarray(stale_before.u_k), np.asarray(base.u_k))
    np.testing.assert_array_equal(np.asarray(stale_before.v_k), np.asarray(base.v_k))


def test_stage_replay_refuses_a_non_swe_selector(card_and_model):
    card, _ = card_and_model
    bad = card.recipe.model_config._replace(
        outer_integrator="forward_euler", vorticity_scheme="ene_total")
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, bad)
    with pytest.raises(ValueError, match="requires outer_integrator"):
        model.replay_swe_rk3_stage(_entry(), card.dt_s, 1)


@pytest.mark.parametrize("field,value", (
    ("coriolis_scheme", "matsuno_split"),
    ("een_e3f_scheme", "min"),
    ("een_metric_weighting", "off"),
    ("een_q_boundary", "neumann_fill"),
    ("tracer_advection", "tvd"),
))
def test_stage_replay_refuses_ignored_program_selector(
        card_and_model, field, value):
    card, _ = card_and_model
    bad = card.recipe.model_config._replace(**{field: value})
    with pytest.raises(ValueError, match="SWE RK3 requires"):
        swe_rk3_stage(
            _entry(), _entry(), card.dt_s, 1, card.recipe.grid,
            card.recipe.z_coord, card.recipe.land_mask, bad)


def test_stage_replay_refuses_bypassed_time_and_vertical_physics_selectors(
        card_and_model):
    card, _ = card_and_model
    base = card.recipe.model_config
    variants = (
        base._replace(tracer_time_integrator="euler",
                      momentum_time_integrator="euler"),
        base._replace(implicit_vertical_mixing=True),
        base._replace(physics=base.physics._replace(
            vertical_mixing=base.physics.vertical_mixing._replace(
                scheme="constant"))),
        base._replace(physics=base.physics._replace(
            convection=base.physics.convection._replace(
                scheme="enhanced_diffusion"))),
    )
    for bad in variants:
        with pytest.raises(ValueError, match="SWE RK3 requires"):
            swe_rk3_stage(
                _entry(), _entry(), card.dt_s, 1, card.recipe.grid,
                card.recipe.z_coord, card.recipe.land_mask, bad)


@pytest.mark.parametrize(
    "field", [name for name in OceanSurfaceForcing._fields
              if name not in ("tau_i_native", "tau_j_native")])
def test_swe_step_refuses_each_non_native_surface_field(
        card_and_model, field):
    card, model = card_and_model
    value = jnp.zeros((84, 84), dtype=jnp.float64)
    forcing = OceanSurfaceForcing(
        tau_i_native=value, tau_j_native=value)._replace(**{field: value})
    with pytest.raises(ValueError, match=f"does not accept.*{field}"):
        model.step(card.recipe.initial_state, card.dt_s,
                   surface_forcing=forcing)


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
