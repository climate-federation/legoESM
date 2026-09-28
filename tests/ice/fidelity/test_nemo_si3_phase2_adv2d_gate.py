"""Controls for the SI3 rung-3.2 legoESM fidelity gate."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    ICE_ADV2D_TRACERS,
    apply_ice_adv2d_source_corrections,
    build_ice_adv2d_card,
    load_ice_adv2d_restart,
    save_ice_adv2d_restart,
    step_ice_adv2d_card,
)

GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_adv2d_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_phase2_adv2d_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

# A deliberately ill-conditioned fp64 triplet for the source-order check at
# icedyn_adv_pra.F90:570-582.  These are numerics operands, not card physics.
_ORDER_TEST_AREA = 3743035203.7156296
_ORDER_TEST_TRANSPORT = 378.6794450766214
_ORDER_TEST_DT = 44097.24518081014
_SOURCE_EXACT_SHAPE = (11, 11, 3)
_SOURCE_EXACT_AREA_M2 = 3.7e9
_SOURCE_EXACT_AREA_INCREMENT_M2 = 12345.6789
_SOURCE_EXACT_U_SCALE_M_S = 317.123456
_SOURCE_EXACT_V_SCALE_M_S = 173.98765
_SOURCE_EXACT_DT_S = 1800.0
_SOURCE_EXACT_SPACING_M = 3000.0


def _build_card():
    return build_ice_adv2d_card(gate.oracle_surface_temperature_c(gate.ROOT))


def _normalized_error(oracle: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(oracle - candidate))) / max(1.0, float(np.max(np.abs(oracle))))


def test_card_is_fp64_and_resolves_the_pinned_oracle_selectors():
    card = _build_card()
    assert card.jpl == 1
    assert card.nn_icesal == 4
    assert card.ponds is True
    assert card.thermodynamics is False
    assert card.landfast is False
    assert card.initial_state.contents.dtype == jnp.float64
    assert all(value.dtype == jnp.float64 for value in card.initial_state.moments)
    assert len(ICE_ADV2D_TRACERS) == 16


def test_complete_card_step_jits_and_has_a_finite_nonzero_gradient():
    card = _build_card()
    state = card.initial_state
    compiled = jax.jit(lambda packed: step_ice_adv2d_card(card, packed, completed_steps=0))
    advanced = compiled(state)
    gradient = jax.grad(lambda packed: jnp.sum(compiled(state._replace(contents=packed)).contents))(
        state.contents
    )
    assert advanced.contents.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_kt1_and_first_completed_step_clear_the_pointwise_bar_for_all_tracers():
    card = _build_card()
    _, first = gate.oracle_gate.read_frame(gate.ROOT / "oracle_ice_step_entry_kt00000001.bin")
    for name in ICE_ADV2D_TRACERS:
        assert (
            _normalized_error(
                gate._entry_tracer(first, name),
                gate.state_field(card, card.initial_state, name),
            )
            <= gate.POINTWISE_BAR
        )
    assert (
        _normalized_error(
            gate._entry_field(first, "sv_i"),
            gate._bulk_salinity(card, card.initial_state),
        )
        <= gate.POINTWISE_BAR
    )

    state = step_ice_adv2d_card(card, completed_steps=0)
    _, second = gate.oracle_gate.read_frame(gate.ROOT / "oracle_ice_step_entry_kt00000002.bin")
    for name in ICE_ADV2D_TRACERS:
        assert (
            _normalized_error(gate._entry_tracer(second, name), gate.state_field(card, state, name))
            <= gate.POINTWISE_BAR
        )


def test_even_step_uses_y_then_x_and_wrong_parity_goes_red():
    card = _build_card()
    after_one = step_ice_adv2d_card(card, completed_steps=0)
    correct = step_ice_adv2d_card(card, after_one, completed_steps=1)
    _, third = gate.oracle_gate.read_frame(gate.ROOT / "oracle_ice_step_entry_kt00000003.bin")
    oracle = gate._entry_tracer(third, "v_i")
    assert _normalized_error(oracle, gate.state_field(card, correct, "v_i")) <= gate.POINTWISE_BAR

    # The documented card is symmetric in x/y, so a wrong order is nearly
    # self-cancelling. An asymmetric planted tracer proves the parity switch
    # at icedyn_adv_pra.F90:253-351 is live rather than vacuous.
    tracer = ICE_ADV2D_TRACERS.index("v_i")
    moments = list(after_one.moments)
    moments[4] = moments[4].at[51, 54, tracer].add(1.0e6)
    planted = after_one._replace(moments=tuple(moments))
    x_first = step_ice_adv2d_card(card, planted, completed_steps=0)
    y_first = step_ice_adv2d_card(card, planted, completed_steps=1)
    difference = float(
        np.max(
            np.abs(np.asarray(x_first.moments[4][..., tracer] - y_first.moments[4][..., tracer]))
        )
    )
    scale = float(np.max(np.abs(np.asarray(planted.moments[4][..., tracer]))))
    assert difference / scale > gate.POINTWISE_BAR


def test_y_sweep_limiter_limits_y_moments_not_x_moments():
    import legoesm.ice.transport as transport

    shape = (7, 7, 1)
    content = jnp.ones(shape, dtype=jnp.float64)
    sx = jnp.full(shape, 0.25, dtype=jnp.float64)
    sy = jnp.full(shape, 4.0, dtype=jnp.float64)
    sxx = jnp.full(shape, 0.125, dtype=jnp.float64)
    syy = jnp.zeros(shape, dtype=jnp.float64)
    sxy = jnp.zeros(shape, dtype=jnp.float64)
    _, moments, _ = transport._si3_prather_y_substep(
        content,
        (sx, sy, sxx, syy, sxy),
        jnp.zeros(shape[:2], dtype=jnp.float64),
        jnp.ones(shape[:2], dtype=jnp.float64),
        jnp.ones(shape[:2], dtype=bool),
        1.0,
        initial_area=jnp.ones(shape[:2], dtype=jnp.float64),
        first_sweep=True,
        halo_width=2,
        subcycle_index=1,
        subcycles=1,
    )
    assert float(moments[0][3, 3, 0]) == pytest.approx(0.25)
    assert float(moments[1][3, 3, 0]) == pytest.approx(1.5)
    assert float(moments[2][3, 3, 0]) == pytest.approx(0.125)


def test_flux_area_preserves_nemo_courant_then_area_operation_order():
    import legoesm.ice.transport as transport

    shape = (7, 7, 1)
    area = jnp.full(shape[:2], _ORDER_TEST_AREA, dtype=jnp.float64)
    u_transport = jnp.zeros(shape[:2], dtype=jnp.float64)
    u_transport = u_transport.at[3, 3].set(_ORDER_TEST_TRANSPORT)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    _, _, area_after = transport._si3_prather_x_substep(
        jnp.ones(shape, dtype=jnp.float64),
        (zero, zero, zero, zero, zero),
        u_transport,
        area,
        jnp.ones(shape[:2], dtype=bool),
        _ORDER_TEST_DT,
        initial_area=area,
        first_sweep=True,
        halo_width=2,
        subcycle_index=1,
        subcycles=1,
    )
    alpha = _ORDER_TEST_TRANSPORT * _ORDER_TEST_DT / _ORDER_TEST_AREA
    written_flux_area = alpha * _ORDER_TEST_AREA
    simplified_flux_area = _ORDER_TEST_TRANSPORT * _ORDER_TEST_DT
    assert written_flux_area != simplified_flux_area
    assert float(area_after[3, 3]) == _ORDER_TEST_AREA - written_flux_area


def test_source_rounding_changes_compiled_prather_association() -> None:
    """The private identity ablation must change a scored Prather leaf."""

    import legoesm.ice.transport as transport

    shape = _SOURCE_EXACT_SHAPE
    index = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    content = jnp.asarray(1.0 + (index % 37) / 53.0)
    area = jnp.asarray(
        _SOURCE_EXACT_AREA_M2
        + (index[..., 0] % 13) * _SOURCE_EXACT_AREA_INCREMENT_M2
    )
    u_ice = jnp.asarray(
        ((index[..., 0] % 7) - 3) * _SOURCE_EXACT_U_SCALE_M_S
    )
    v_ice = jnp.asarray(
        ((index[..., 0] % 5) - 2) * _SOURCE_EXACT_V_SCALE_M_S
    )
    wet = jnp.ones(shape[:2], dtype=bool)
    moments = tuple(
        jnp.asarray(content * (0.01 * (moment + 1)) + index % (11 + moment) / 1.0e6)
        for moment in range(5)
    )

    def make_stepper():
        def step(packed, carried_moments):
            return transport.advect_si3_prather_2d(
                packed,
                carried_moments,
                u_ice,
                v_ice,
                area,
                wet,
                _SOURCE_EXACT_DT_S,
                dx=_SOURCE_EXACT_SPACING_M,
                dy=_SOURCE_EXACT_SPACING_M,
                ice_step_index=8,
                halo_width=2,
                ice_volume_index=0,
                concentration_index=1,
                subcycles=1,
            )

        return jax.jit(step)

    guarded = make_stepper()(content, moments)
    jax.block_until_ready(guarded)
    source_round = transport.nemo_source_round
    try:
        transport.nemo_source_round = lambda value: value
        unrounded = make_stepper()(content, moments)
        jax.block_until_ready(unrounded)
    finally:
        transport.nemo_source_round = source_round

    guarded_leaves = jax.tree.leaves((guarded[0], guarded[1]))
    unrounded_leaves = jax.tree.leaves((unrounded[0], unrounded[1]))
    assert any(
        np.asarray(left).tobytes() != np.asarray(right).tobytes()
        for left, right in zip(guarded_leaves, unrounded_leaves, strict=True)
    )


def test_nemo_intensive_correction_order_is_default_and_old_order_is_private() -> None:
    """Decision 13 keeps the former association only as an ablation."""

    card = _build_card()
    default = step_ice_adv2d_card(card, completed_steps=0)
    legacy = step_ice_adv2d_card(
        card,
        completed_steps=0,
        _use_legacy_extensive_correction_order=True,
    )
    assert np.asarray(default.contents).tobytes() != np.asarray(legacy.contents).tobytes()

    # The default is exactly one source-ordered recovery/correction/repack;
    # calling the public wrapper without its private hook must select it too.
    corrected_default = apply_ice_adv2d_source_corrections(
        card,
        card.initial_state.contents,
        legacy.contents,
    )
    corrected_legacy = apply_ice_adv2d_source_corrections(
        card,
        card.initial_state.contents,
        legacy.contents,
        _use_legacy_extensive_order=True,
    )
    assert np.asarray(corrected_default).tobytes() != np.asarray(corrected_legacy).tobytes()


def test_hsnow_and_pond_caps_are_non_vacuous():
    card = _build_card()
    area = card.dx_m * card.dy_m
    contents = card.initial_state.contents
    center = (51, 51)
    v_i_index = ICE_ADV2D_TRACERS.index("v_i")
    v_s_index = ICE_ADV2D_TRACERS.index("v_s")
    a_i_index = ICE_ADV2D_TRACERS.index("a_i")
    a_ip_index = ICE_ADV2D_TRACERS.index("a_ip")
    e_s_index = ICE_ADV2D_TRACERS.index("e_s_l01")
    contents = contents.at[center + (v_i_index,)].set(0.3 * area)
    contents = contents.at[center + (v_s_index,)].set(0.3 * area)
    contents = contents.at[center + (a_i_index,)].set(0.2 * area)
    contents = contents.at[center + (a_ip_index,)].set(0.4 * area)
    before_energy = float(contents[center + (e_s_index,)])
    corrected = apply_ice_adv2d_source_corrections(card, card.initial_state.contents, contents)
    assert float(corrected[center + (a_ip_index,)]) <= float(corrected[center + (a_i_index,)])
    assert float(corrected[center + (v_s_index,)]) < 0.3 * area
    assert float(corrected[center + (e_s_index,)]) < before_energy


def test_maximum_floor_has_finite_gradient_and_zapneg_uses_updated_concentration():
    card = _build_card()
    area = card.dx_m * card.dy_m
    center = (51, 51)
    index = ICE_ADV2D_TRACERS.index
    empty = jnp.zeros_like(card.initial_state.contents)

    # An isolated pond has a zero entry-neighborhood depth.  SI3's epsi20
    # floor (`icedyn_adv_pra.F90:1519-1534`) keeps Hbig finite and differentiable.
    def isolated_pond(pond_volume):
        transported = empty.at[center + (index("v_i"),)].set(0.1 * area)
        transported = transported.at[center + (index("a_i"),)].set(0.1 * area)
        transported = transported.at[center + (index("a_ip"),)].set(0.01 * area)
        transported = transported.at[center + (index("v_ip"),)].set(pond_volume)
        return jnp.sum(apply_ice_adv2d_source_corrections(card, empty, transported))

    derivative = jax.grad(isolated_pond)(jnp.asarray(0.001 * area, dtype=jnp.float64))
    assert np.isfinite(float(derivative))

    # The source divides by max(epsi20,a_ip), so a transported pond volume
    # with zero pond area is rescued when ice survives (:986-995).
    transported = empty.at[center + (index("v_i"),)].set(0.1 * area)
    transported = transported.at[center + (index("a_i"),)].set(0.1 * area)
    transported = transported.at[center + (index("v_ip"),)].set(0.001 * area)
    corrected = apply_ice_adv2d_source_corrections(card, empty, transported)
    assert float(corrected[center + (index("a_ip"),)]) > 0.0
    assert float(corrected[center + (index("v_ip"),)]) > 0.0

    # The same floor gives isolated snow the source's nonzero a_i*epsi20 cap.
    transported = empty.at[center + (index("v_i"),)].set(0.1 * area)
    transported = transported.at[center + (index("a_i"),)].set(0.1 * area)
    transported = transported.at[center + (index("v_s"),)].set(0.01 * area)
    corrected = apply_ice_adv2d_source_corrections(card, empty, transported)
    assert float(corrected[center + (index("v_s"),)] / area) == pytest.approx(1.0e-21)

    # pa_i is zeroed first when pv_i<=0; the later snow test must see it
    # (`icevar.F90:759-760,807-816`).
    transported = empty.at[center + (index("v_i"),)].set(-0.1 * area)
    transported = transported.at[center + (index("a_i"),)].set(0.1 * area)
    transported = transported.at[center + (index("v_s"),)].set(0.1 * area)
    corrected = apply_ice_adv2d_source_corrections(card, empty, transported)
    assert float(corrected[center + (index("a_i"),)]) == 0.0
    assert float(corrected[center + (index("v_s"),)]) == 0.0


def test_restart_carries_all_eighty_moments_and_controls_go_red():
    card = _build_card()
    assert gate.restart_control(card) == {
        "status": "VERIFIED",
        "state_arrays": 10,
        "moment_leaves": 80,
    }
    with pytest.raises((gate.GateError, ValueError)):
        gate.restart_control(card, plant=True)
    with pytest.raises(ValueError, match="missing=.*moment_4"):
        gate.restart_control(card, drop=True)
    with pytest.raises(ValueError, match="moment_4 shape/dtype mismatch"):
        gate.restart_control(card, retype=True)


def test_restart_loader_rejects_a_retyped_moment():
    card = _build_card()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv2d_restart(path, card, card.initial_state, completed_steps=0)
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload["moment_4"] = payload["moment_4"].astype(np.float32)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="moment_4 shape/dtype mismatch"):
            load_ice_adv2d_restart(path, card)


def test_restart_loader_rejects_extra_key_and_selector_clock_velocity_changes():
    card = _build_card()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv2d_restart(path, card, card.initial_state, completed_steps=0)
        with np.load(path, allow_pickle=False) as archive:
            original = {name: archive[name].copy() for name in archive.files}

        np.savez(path, **original, planted_extra=np.asarray(1))
        with pytest.raises(ValueError, match="extra=.*planted_extra"):
            load_ice_adv2d_restart(path, card)

        np.savez(path, **original)
        with pytest.raises(ValueError, match="selector composition"):
            load_ice_adv2d_restart(path, card._replace(dt_s=card.dt_s + 1.0))

        changed_velocity = card._replace(
            prescribed_u_ice=card.prescribed_u_ice.at[20, 20].add(1.0)
        )
        with pytest.raises(ValueError, match="card contract mismatch"):
            load_ice_adv2d_restart(path, changed_velocity)

        bad_clock = dict(original)
        bad_clock["completed_steps"] = np.asarray(card.n_steps + 1, dtype=np.int64)
        np.savez(path, **bad_clock)
        with pytest.raises(ValueError, match="clock out of range"):
            load_ice_adv2d_restart(path, card)


def test_geometry_state_and_coverage_plants_go_red():
    card = _build_card()
    rows: list[dict] = []
    gate.geometry_gate(gate.ROOT, card, rows, {}, plant=True)
    assert next(row for row in rows if row["name"] == "geometry.e1t")["status"] == "DEBT"
    state_rows: list[dict] = []
    gate._score(
        state_rows,
        {},
        "planted.ordinary_state",
        np.ones((2, 2), dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        plant=True,
    )
    assert state_rows[0]["status"] == "DEBT"
    with pytest.raises(gate.oracle_gate.GateError, match="PLANTED_UNACCOUNTED"):
        gate.mesh_coverage(gate.ROOT, plant=True)
    with pytest.raises(gate.GateError, match="PLANTED_UNACCOUNTED_FRAME_ARRAY"):
        gate.candidate_frame_contract(plant=True)


def test_frame_header_and_row_roster_controls_go_red():
    card = _build_card()
    provenance = gate.frame_provenance(gate.ROOT, card)
    assert provenance["frame_count"] == card.n_steps
    assert len(provenance["ordered_frame_hash_aggregate"]) == 64
    bad_header = dict(provenance["header"])
    bad_header["kt"] = 2
    with pytest.raises(gate.GateError, match="oracle frame header"):
        gate._validate_frame_header(bad_header, kt=1, card=card)

    rows = [{"name": "duplicate"}, {"name": "duplicate"}]
    with pytest.raises(gate.GateError, match="duplicate registered"):
        gate._validate_row_registry(gate.ROOT, card, rows)


def test_real_gate_has_the_corrected_row_count_and_loud_first_divergence():
    report, code = gate.run_gate(gate.ROOT)
    assert code == 1
    assert report["status"] == "UNMEASURED"
    assert report["numeric_status"] == "DEBT"
    assert len(report["rows"]) == 9822
    assert report["restart_carry_control"]["moment_leaves"] == 80
    first = report["trajectory"]["first_divergence"]
    assert first is not None
    assert first["normalized_max_abs"] > gate.POINTWISE_BAR
    assert set(report["candidate_frame_contract"]) == {
        item[0] for item in gate.oracle_gate.FRAME_REGISTRY
    }
    assert "candidate_frame.snwice_mass" in report["unmeasured"]
