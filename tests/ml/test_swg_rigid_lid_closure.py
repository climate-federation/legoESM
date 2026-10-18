"""Rigid-lid closure input (phi/g), offline potential, and the RL decks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics import swe_rk3 as rk3
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_DG50_RL_NAMELIST,
    SWG_MADDISON_NAMELIST,
    SWG_MADDISON_RL_NAMELIST,
    _swg_rescaled_clock,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.fidelity.swg_online_decks import (
    RIGID_LID_ONLY_DECKS,
    require_surface_for_deck,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import (
    SWGNetworkSpec,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    coarse_state_from_native,
    eq6_scale_report,
    initial_closure_acceleration_ratio,
    maddison_eq6_network_scales,
    require_rigid_lid_zero_ssh,
    rigid_lid_start_state,
    start_state_from_native,
)

PHI_TOL = 1.0e-14   # relative to max|phi/g|; same arithmetic, same operands


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def lid():
    return build_swg_card(SWG_DG50_RL_NAMELIST, surface="rigid_lid")


@pytest.fixture(scope="module")
def explicit():
    return build_swg_card(SWG_DG50_NAMELIST)


@pytest.fixture(scope="module")
def forcing():
    tau_u, tau_v = swg_surface_wind(SWG_DG50_RL_NAMELIST)
    return OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u), tau_j_native=jnp.asarray(tau_v))


@pytest.fixture(scope="module")
def stepped(lid, forcing):
    """A state the stepper produced: three hourly lid steps from rest."""
    state = lid.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    for _ in range(3):
        state = rk3.swe_rk3_step(
            state, lid.dt_s, forcing, lid.recipe.grid, lid.recipe.z_coord,
            lid.recipe.model_config)
    return state


def _step_with_recorder(card, state, forcing, closure=None):
    seen = {}

    def potential(entry, stage):
        seen[stage] = entry
        return None if closure is None else closure(entry, stage)

    out = rk3.swe_rk3_step(
        state, card.dt_s, forcing, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config, stage_potential_fn=potential)
    return out, seen


def _manual_stage_phis(card, state, forcing):
    """phi of stages 1-3 from the public stage functions (no closure)."""
    config, grid, z = (card.recipe.model_config, card.recipe.grid,
                       card.recipe.z_coord)
    state, boundary, _, geometry = rk3._step_boundary(
        state, forcing, grid, z, config)
    entry, phis = boundary, []
    for stage in (1, 2, 3):
        out = rk3.swe_rk3_stage(
            entry, boundary, card.dt_s, stage, grid, z,
            state.land_mask.data, config)
        entry, phi = rk3.swe_rigid_lid_stage_output(
            out, entry, stage, card.dt_s / (3.0, 2.0, 1.0)[stage - 1],
            geometry)
        phis.append(phi)
    return phis


def test_closure_sees_phi_over_g_under_the_lid(lid, stepped, forcing):
    g = lid.recipe.model_config.constants.g
    assert g == lid.recipe.model_config.g == 9.80665
    _, seen = _step_with_recorder(lid, stepped, forcing)
    phi1, phi2, phi3 = _manual_stage_phis(lid, stepped, forcing)
    scale = float(jnp.abs(stepped.rigid_lid_phi).max() / g)
    assert scale > 0.0
    # Stage 1: the previous step's stage-3 pressure; stages 2-3: the
    # previous stage's.  All non-zero (the network input is not ssh = 0).
    for stage, expected in ((1, stepped.rigid_lid_phi), (2, phi1), (3, phi2)):
        shown = seen[stage].ssh_k
        assert float(jnp.abs(shown).max()) > 0.0
        assert float(jnp.abs(shown - expected / g).max()) <= PHI_TOL * scale
    # The other entry fields are the certified stage entry, unchanged.
    out = rk3.swe_rk3_step(
        stepped, lid.dt_s, forcing, lid.recipe.grid, lid.recipe.z_coord,
        lid.recipe.model_config)
    assert float(jnp.abs(out.rigid_lid_phi - phi3).max()) <= PHI_TOL * scale * g


def test_certified_stage_still_runs_on_the_zero_ssh_entry(lid, stepped, forcing):
    # A zero closure potential must leave the step bit-identical to no
    # closure: the phi/g view reaches only the closure, never the stage.
    plain = rk3.swe_rk3_step(
        stepped, lid.dt_s, forcing, lid.recipe.grid, lid.recipe.z_coord,
        lid.recipe.model_config)
    shape = stepped.eta.data.shape
    zero, _ = _step_with_recorder(
        lid, stepped, forcing, lambda entry, stage: jnp.zeros(shape))
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(zero)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert float(jnp.abs(zero.eta.data).max()) == 0.0


def test_closure_without_stored_phi_is_refused(lid, stepped, forcing):
    bare = stepped._replace(rigid_lid_phi=None)
    with pytest.raises(ValueError, match="needs state.rigid_lid_phi"):
        _step_with_recorder(lid, bare, forcing)
    # No closure -> no phi needed.
    rk3.swe_rk3_step(bare, lid.dt_s, forcing, lid.recipe.grid,
                     lid.recipe.z_coord, lid.recipe.model_config)


def test_explicit_closure_input_is_ssh_unchanged(explicit):
    tau_u, tau_v = swg_surface_wind(SWG_DG50_NAMELIST)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u), tau_j_native=jnp.asarray(tau_v))
    state = explicit.recipe.initial_state
    bump = 0.01 * jnp.asarray(np.random.default_rng(3).standard_normal(
        state.eta.data.shape)) * state.land_mask.data
    state = state._replace(eta=state.eta.replace(data=bump))
    out, seen = _step_with_recorder(explicit, state, forcing)
    assert np.array_equal(np.asarray(seen[1].ssh_k), np.asarray(bump))
    assert out.rigid_lid_phi is None
    ref = rk3.swe_rk3_step(
        state, explicit.dt_s, forcing, explicit.recipe.grid,
        explicit.recipe.z_coord, explicit.recipe.model_config)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(out)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_offline_phi_is_the_steppers_stage_one_phi(lid, stepped, forcing):
    g = lid.recipe.model_config.constants.g
    native = (stepped.eta.data, stepped.u.data[:, 1:, 0],
              stepped.v.data[1:, :, 0])
    stored = start_state_from_native(lid, *native)
    assert float(jnp.abs(stored.u.data).max()) > 0.0
    # What the stepper shows the closure at stage 2 is its stage-1 phi/g.
    _, seen = _step_with_recorder(lid, stored, forcing)
    phi = stored.rigid_lid_phi
    scale = float(jnp.abs(phi).max())
    assert scale > 0.0
    assert float(jnp.abs(seen[2].ssh_k * g - phi).max()) <= PHI_TOL * scale
    # And it is the potential of the velocities, not the stored carry.
    assert float(jnp.abs(phi - stepped.rigid_lid_phi).max()) > 0.0
    again = rigid_lid_start_state(lid, coarse_state_from_native(
        lid.recipe.initial_state, *native))
    assert np.array_equal(np.asarray(again.rigid_lid_phi), np.asarray(phi))


def test_stored_lid_state_with_sea_level_is_refused(lid, explicit, stepped):
    u, v = stepped.u.data[:, 1:, 0], stepped.v.data[1:, :, 0]
    bumped = jnp.zeros_like(stepped.eta.data).at[10, 10].set(1.0e-3)
    with pytest.raises(ValueError, match="identically zero ssh"):
        start_state_from_native(lid, bumped, u, v)
    with pytest.raises(ValueError, match="identically zero ssh"):
        require_rigid_lid_zero_ssh(lid, bumped)
    # Explicit stored states keep their sea level and gain no potential.
    state = start_state_from_native(explicit, bumped, u, v)
    assert state.rigid_lid_phi is None
    assert float(state.eta.data[10, 10]) == 1.0e-3
    assert rigid_lid_start_state(explicit, state) is state


def test_offline_network_inputs_read_phi_over_g(lid, stepped):
    geometry = build_closure_geometry(lid)
    scales = maddison_eq6_network_scales(lid, geometry)
    spec = SWGNetworkSpec(8, 2, 3, scales.ssh_scale_m,
                          scales.velocity_scale_m_s,
                          scales.potential_scale_m3_s2, 0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(3))
    state = start_state_from_native(
        lid, stepped.eta.data, stepped.u.data[:, 1:, 0],
        stepped.v.data[1:, :, 0])
    wet = np.asarray(geometry.tmask, dtype=bool)
    expected = (np.asarray(state.rigid_lid_phi)[wet]
                / lid.recipe.model_config.constants.g / scales.ssh_scale_m)
    report = eq6_scale_report(
        network, (state,), lid, geometry, scales, key=jax.random.PRNGKey(0))
    assert report["ssh"]["max_abs"] > 0.0
    assert report["ssh"]["max_abs"] == pytest.approx(
        float(np.abs(expected).max()), rel=1.0e-13)
    ratio = initial_closure_acceleration_ratio(
        network, (state,), lid, geometry, key=jax.random.PRNGKey(0))
    assert np.isfinite(ratio["maximum_ratio"])
    with pytest.raises(ValueError, match="needs rigid_lid_phi"):
        eq6_scale_report(network, (state._replace(rigid_lid_phi=None),),
                         lid, geometry, scales, key=jax.random.PRNGKey(0))


def test_rigid_lid_namelists_keep_every_physical_duration():
    for parent, rl, dt in ((SWG_MADDISON_NAMELIST, SWG_MADDISON_RL_NAMELIST, 1800.0),
                           (SWG_DG50_NAMELIST, SWG_DG50_RL_NAMELIST, 3600.0)):
        assert rl.rn_Dt_s == dt
        for name in ("nn_itend", "nn_write", "nn_stock"):
            assert getattr(rl, name) * rl.rn_Dt_s \
                == getattr(parent, name) * parent.rn_Dt_s
        changed = {name for name in parent._fields
                   if getattr(parent, name) != getattr(rl, name)}
        assert changed == {"rn_Dt_s", "nn_itend", "nn_write", "nn_stock"}
    assert (SWG_MADDISON_RL_NAMELIST.nn_itend, SWG_MADDISON_RL_NAMELIST.nn_write,
            SWG_MADDISON_RL_NAMELIST.nn_stock) == (51_840, 48, 51_840)
    assert (SWG_DG50_RL_NAMELIST.nn_itend, SWG_DG50_RL_NAMELIST.nn_write,
            SWG_DG50_RL_NAMELIST.nn_stock) == (25_920, 24, 25_920)
    with pytest.raises(ValueError, match="not a whole number"):
        _swg_rescaled_clock(SWG_DG50_NAMELIST, 7000.0)


def test_rigid_lid_only_decks_refuse_the_explicit_surface():
    assert RIGID_LID_ONLY_DECKS == ("maddison-rl", "dg50-rl")
    for deck in RIGID_LID_ONLY_DECKS:
        require_surface_for_deck(deck, "rigid_lid")
        for surface in ("explicit", None):
            with pytest.raises(ValueError, match="requires surface='rigid_lid'"):
                require_surface_for_deck(deck, surface)
    # An explicit deck under the lid is legal physics; nothing refuses it.
    require_surface_for_deck("dg50", "rigid_lid")
    require_surface_for_deck("dg50", "explicit")
