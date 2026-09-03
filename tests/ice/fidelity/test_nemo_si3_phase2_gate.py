"""Controls for the SI3 rung-3.1 legoESM fidelity gate."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import legoesm.ice.transport as ice_transport
import numpy as np
import pytest
from legoesm.ice.fidelity.nemo_testcase_recipe import (
    ICE_ADV1D_TRACERS,
    apply_ice_adv1d_zapsmall,
    build_ice_adv1d_card,
    load_ice_adv1d_restart,
    save_ice_adv1d_restart,
    step_ice_adv1d_card,
    validate_ice_adv1d_card,
)
from legoesm.ice.transport import advect_si3_prather_1d

GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_phase2_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _build_card():
    return build_ice_adv1d_card(gate._oracle_surface_temperature_c(gate.ROOT))


def test_card_is_fp64_and_default_ice_state_layout_is_unchanged():
    from legoesm.ice.state import DynamicSeaIceState

    card = _build_card()
    assert card.initial_state.contents.dtype == jnp.float64
    assert all(moment.dtype == jnp.float64 for moment in card.initial_state.moments)
    assert len(DynamicSeaIceState._fields) == 12


def test_prather_rung_rejects_a_nonzero_y_velocity():
    card = _build_card()
    with pytest.raises(ValueError, match="v_ice == 0 exactly"):
        validate_ice_adv1d_card(
            card,
            card.initial_state._replace(
                v_ice=jnp.ones_like(card.initial_state.v_ice)
            ),
        )


def test_complete_card_step_jits_and_has_nonzero_finite_reverse_mode_gradient():
    card = _build_card()
    state = card.initial_state
    compiled = jax.jit(lambda value: step_ice_adv1d_card(card, value))
    advanced = compiled(state)
    gradient = jax.grad(
        lambda packed: jnp.sum(
            compiled(state._replace(contents=packed)).contents
        )
    )(state.contents)
    assert advanced.contents.dtype == jnp.float64
    assert all(value.dtype == jnp.float64 for value in advanced.moments)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_zero_content_zeroes_all_five_prather_moments():
    shape = (7, 7, 1)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    seeded = (zero, jnp.ones_like(zero), zero, jnp.ones_like(zero), zero)
    _, moments, _ = advect_si3_prather_1d(
        zero,
        seeded,
        jnp.zeros(shape[:2], dtype=jnp.float64),
        jnp.zeros(shape[:2], dtype=jnp.float64),
        jnp.full(shape[:2], 16.0, dtype=jnp.float64),
        jnp.ones(shape[:2], dtype=bool),
        2.0,
        dx=4.0,
        dy=4.0,
        halo_width=2,
        ice_volume_index=0,
        concentration_index=0,
        subcycles=1,
    )
    assert all(
        np.array_equal(np.asarray(value)[2:-2, 2:-2], np.zeros((3, 3, 1)))
        for value in moments
    )


def test_kt1_and_first_completed_step_clear_the_pointwise_bar():
    card = _build_card()
    _, first = gate.oracle_gate.read_frame(
        gate.ROOT / "oracle_ice_step_entry_kt00000001.bin"
    )
    for name in ICE_ADV1D_TRACERS:
        oracle = gate._entry_field(first, name)
        lego = gate.state_field(card, card.initial_state, name)
        scale = max(float(np.max(np.abs(oracle))), 1.0)
        assert float(np.max(np.abs(lego - oracle))) / scale <= gate.POINTWISE_BAR

    state = step_ice_adv1d_card(card)
    _, second = gate.oracle_gate.read_frame(
        gate.ROOT / "oracle_ice_step_entry_kt00000002.bin"
    )
    for name in tuple(n for n in ICE_ADV1D_TRACERS if n != "sv_i"):
        oracle = gate._entry_field(second, name)
        lego = gate.state_field(card, state, name)
        scale = max(float(np.max(np.abs(oracle))), 1.0)
        assert float(np.max(np.abs(lego - oracle))) / scale <= gate.POINTWISE_BAR
    oracle_t = gate._entry_field(second, "t_su")
    lego_t = np.asarray(state.t_surface)[2:-2, 2:-2]
    assert float(np.max(np.abs(lego_t - oracle_t))) <= gate.POINTWISE_BAR


def test_prather_requires_hbig_indices():
    shape = (7, 7, 1)
    contents = jnp.ones(shape, dtype=jnp.float64)
    zero = jnp.zeros_like(contents)
    with pytest.raises(ValueError, match="requires volume and concentration indices"):
        advect_si3_prather_1d(
            contents,
            (zero, zero, zero, zero, zero),
            jnp.zeros(shape[:2], dtype=jnp.float64),
            jnp.zeros(shape[:2], dtype=jnp.float64),
            jnp.ones(shape[:2], dtype=jnp.float64),
            jnp.ones(shape[:2], dtype=bool),
            1.0,
            dx=1.0,
            dy=1.0,
            subcycles=1,
        )


def test_prather_dual_outflow_uses_nemo_sequential_donor_residual():
    shape = (7, 5, 2)
    contents = jnp.zeros(shape, dtype=jnp.float64).at[3, 2, :].set(1.0)
    sx = jnp.zeros_like(contents).at[3, 2, :].set(0.4)
    sxx = jnp.zeros_like(contents).at[3, 2, :].set(0.2)
    zero = jnp.zeros_like(contents)
    u_ice = jnp.zeros(shape[:2], dtype=jnp.float64)
    u_ice = u_ice.at[2, 2].set(-0.3).at[3, 2].set(0.2)
    result, result_moments, _ = advect_si3_prather_1d(
        contents,
        (sx, zero, sxx, zero, zero),
        u_ice,
        jnp.zeros(shape[:2], dtype=jnp.float64),
        jnp.ones(shape[:2], dtype=jnp.float64),
        jnp.ones(shape[:2], dtype=bool),
        1.0,
        dx=1.0,
        dy=1.0,
        ice_volume_index=0,
        concentration_index=1,
        subcycles=1,
    )
    # icedyn_adv_pra.F90:590-597 first leaves the right-going residual;
    # :618-664 then extracts the left-going slab from that residual.
    right_fraction = 0.2
    right_one = 1.0 - right_fraction
    content_after_right = 1.0 - right_fraction * (
        1.0 + right_one * (0.4 + (right_one - right_fraction) * 0.2)
    )
    sx_after_right = right_one**2 * (0.4 - 3.0 * right_fraction * 0.2)
    sxx_after_right = right_one**3 * 0.2
    left_fraction = 0.3 / right_one
    left_one = 1.0 - left_fraction
    left_flux = left_fraction * (
        content_after_right
        - left_one
        * (sx_after_right - (left_one - left_fraction) * sxx_after_right)
    )
    expected_content = content_after_right - left_flux
    expected_sx = left_one**2 * (
        sx_after_right + 3.0 * left_fraction * sxx_after_right
    )
    expected_sxx = left_one**3 * sxx_after_right
    assert float(result[3, 2, 0]) == pytest.approx(expected_content)
    assert float(result_moments[0][3, 2, 0]) == pytest.approx(expected_sx)
    assert float(result_moments[2][3, 2, 0]) == pytest.approx(expected_sxx)


def test_hbig_concentration_correction_is_non_vacuous(monkeypatch):
    card = _build_card()
    with_hbig = card.initial_state
    for _ in range(15):
        with_hbig = step_ice_adv1d_card(card, with_hbig)

    monkeypatch.setattr(
        ice_transport, "_SI3_PRA_HBIG_CONCENTRATION_THRESHOLD", -np.inf
    )
    without_hbig = card.initial_state
    for _ in range(15):
        area = card.dx_m * card.dy_m
        contents, moments, _ = advect_si3_prather_1d(
            without_hbig.contents,
            without_hbig.moments,
            card.prescribed_u_ice,
            without_hbig.v_ice,
            jnp.full(without_hbig.u_ice.shape, area),
            jnp.pad(card.wet_global_xy, card.halo_width),
            card.dt_s,
            dx=card.dx_m,
            dy=card.dy_m,
            halo_width=card.halo_width,
            ice_volume_index=ICE_ADV1D_TRACERS.index("v_i"),
            concentration_index=ICE_ADV1D_TRACERS.index("a_i"),
            subcycles=2,
        )
        without_hbig = apply_ice_adv1d_zapsmall(
            card, without_hbig, contents, moments
        )
    _, frame = gate.oracle_gate.read_frame(
        gate.ROOT / "oracle_ice_step_entry_kt00000016.bin"
    )
    oracle = gate._entry_field(frame, "a_i")
    good = gate.state_field(card, with_hbig, "a_i")
    bad = gate.state_field(card, without_hbig, "a_i")
    scale = max(float(np.max(np.abs(oracle))), 1.0)
    assert float(np.max(np.abs(good - oracle))) / scale <= gate.POINTWISE_BAR
    assert float(np.max(np.abs(bad - oracle))) / scale > 1.0e-11


def test_moment_restart_carry_and_perturbation_control():
    card = _build_card()
    assert gate.restart_carry_control(card)["status"] == "VERIFIED"
    with pytest.raises(gate.GateError, match="moment carry changed"):
        gate.restart_carry_control(card, plant_moment=True)
    with pytest.raises(gate.GateError, match="missing.*moment_4"):
        gate.restart_carry_control(card, drop_moment=True)
    with pytest.raises(gate.GateError, match="moment_0 shape/dtype mismatch"):
        gate.restart_carry_control(card, retype_moment=True)


def test_complete_restart_refuses_a_retyped_moment():
    card = _build_card()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv1d_restart(path, card, card.initial_state, completed_steps=0)
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload["moment_0"] = payload["moment_0"].astype(np.float32)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="moment_0 shape/dtype mismatch"):
            load_ice_adv1d_restart(path, card)


def test_complete_restart_refuses_selector_forcing_and_clock_changes():
    card = _build_card()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv1d_restart(path, card, card.initial_state, completed_steps=7)
        changed_cards = (
            card._replace(dt_s=1.0),
            card._replace(n_steps=99),
            card._replace(landfast=True),
            card._replace(
                prescribed_u_ice=card.prescribed_u_ice.at[20, 20].add(0.1)
            ),
        )
        for changed in changed_cards:
            with pytest.raises(ValueError, match="selector composition|card_contract"):
                load_ice_adv1d_restart(path, changed)

        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload["completed_steps"] = np.asarray(41, dtype=np.int64)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="completed_steps 41 is out of range"):
            load_ice_adv1d_restart(path, card)

        save_ice_adv1d_restart(path, card, card.initial_state, completed_steps=7)
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload["v_ice"][20, 20] = 1.0
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="v_ice == 0 exactly"):
            load_ice_adv1d_restart(path, card)


def test_geometry_and_state_plants_go_red():
    card = _build_card()
    rows: list[dict] = []
    gate.geometry_gate(gate.ROOT, card, rows, {}, plant_geometry=True)
    assert next(row for row in rows if row["name"] == "geometry.e1t")["status"] == "DEBT"

    rows = []
    gate._score(rows, {}, "plant.state", np.zeros((2,)), np.zeros((2,)), plant=True)
    assert rows[0]["status"] == "DEBT"


def test_unaccounted_mesh_array_control_goes_red():
    with pytest.raises(gate.oracle_gate.GateError, match="PLANTED_UNACCOUNTED"):
        gate._mesh_coverage(gate.ROOT, plant_unaccounted=True)
    with pytest.raises(gate.GateError, match="PLANTED_UNACCOUNTED_FRAME_ARRAY"):
        gate.candidate_frame_contract(plant_unaccounted=True)


def test_real_gate_reports_core_at_bar_and_loud_restart_moment_debt():
    report, code = gate.run_gate(gate.ROOT)
    assert code == 1
    core = [row for row in report["rows"] if row["name"].startswith("trajectory.")]
    assert core and all(row["status"] == "AT-BAR" for row in core)
    first = report["trajectory"]["first_divergence"]
    assert first is not None
    assert first["name"].startswith("restart_moment.")
    assert first["normalized_max_abs"] > gate.POINTWISE_BAR
    assert set(report["candidate_frame_contract"]) == {
        item[0] for item in gate.oracle_gate.FRAME_REGISTRY
    }
    trajectory = [row["name"] for row in report["rows"] if row["name"].startswith("trajectory.")]
    for field in report["trajectory"]["compared_core_fields"]:
        assert sum(name.endswith(f".{field}") for name in trajectory) == 40
