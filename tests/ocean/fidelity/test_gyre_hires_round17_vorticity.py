"""Round-17 common-state ENE-q discriminator controls."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/validate/ocean_fidelity/testcases"
TOOL = SCRIPTS / "nemo_testcase_l2_gyre_hires_round17_vorticity.py"
RUN = (SCRIPTS / "nemo_testcase_l2_gyre_hires_round17_vorticity" / "run.sh")


def _load_tool():
    name = "_test_gyre_hires_round17_vorticity"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_and_q_readers_refuse_layout_and_preserve_fortran_order(
    tmp_path, monkeypatch,
):
    tool = _load_tool()
    monkeypatch.setattr(tool, "NATIVE_DIMS", (8, 7, 3))
    monkeypatch.setattr(tool, "ACTIVE_LEVELS", 2)
    nx, ny, nz = tool.NATIVE_DIMS
    n3 = nx * ny * nz
    arrays = [np.arange(n3, dtype=np.float64) + 1000 * i for i in range(4)]
    stage = tmp_path / tool.STAGE_NAME
    stage.write_bytes(
        b"NEMO_L2_R17VOR1 "
        + struct.pack("=11i", 1, 4321, 3, 1, 2, 3, 3, nx, ny, nz, 64)
        + b"".join(array.tobytes() for array in arrays)
    )
    got = tool.read_stage_vorticity(stage)
    expected = arrays[1].reshape((nx, ny, nz), order="F").transpose(1, 0, 2)
    np.testing.assert_array_equal(got["before_v"], expected)

    qpath = tmp_path / "q.bin"
    slabs = []
    for level in range(tool.ACTIVE_LEVELS):
        for field in range(3):
            slabs.append(np.arange(nx * ny, dtype=np.float64)
                         + 100 * level + 10 * field)
    qpath.write_bytes(
        b"NEMO_L2_R17Q__1 "
        + struct.pack("=8i", 1, 4321, 2, 4, nx, ny,
                      tool.ACTIVE_LEVELS, 64)
        + b"".join(array.tobytes() for array in slabs)
    )
    operands = tool.read_ene_operands(qpath)
    expected_q0 = slabs[0].reshape((nx, ny), order="F").T
    np.testing.assert_array_equal(operands["q"][..., 0], expected_q0)
    fu, fv = tool._transports_to_legoesm(operands)
    np.testing.assert_array_equal(fu, operands["Fu"][2:-2, 1:-2])
    np.testing.assert_array_equal(fv, operands["Fv"][1:-2, 2:-2])

    planted = bytearray(qpath.read_bytes())
    planted[:16] = b"WRONG_MAGIC____1"
    qpath.write_bytes(planted)
    with pytest.raises(Exception, match="bad magic"):
        tool.read_ene_operands(qpath)


def test_stage2_reader_names_the_stage3_kmm_provider(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(tool, "NATIVE_DIMS", (8, 7, 3))
    monkeypatch.setattr(tool, "ACTIVE_LEVELS", 2)
    nx, ny, nz = tool.NATIVE_DIMS
    n3 = nx * ny * nz
    n2 = nx * ny
    arrays = [np.arange(n3, dtype=np.float64) + 1000 * i for i in range(4)]
    ssh = np.arange(n2, dtype=np.float64) + 9000
    path = tmp_path / "oracle_stage_kt00004321_s2.bin"
    path.write_bytes(
        b"NEMO_L1_STAGE_1 "
        + struct.pack("=9i", 1, 4321, 2, 2, nx, ny, nz, 2, 64)
        + b"".join(array.tobytes() for array in arrays)
        + ssh.tobytes()
    )
    got = tool.read_stage2_state(path)
    expected = arrays[2].reshape((nx, ny, nz), order="F").transpose(
        1, 0, 2)[2:-2, 2:-2, :2]
    np.testing.assert_array_equal(got["u"], expected)
    assert got["header"]["Kaa"] == 2

    moved = bytearray(path.read_bytes())
    moved[16 + 3 * 4:16 + 4 * 4] = struct.pack("=i", 3)
    path.write_bytes(moved)
    with pytest.raises(Exception, match="bad header"):
        tool.read_stage2_state(path)


def test_q_reader_ignores_unread_halo_but_refuses_consumed_nonfinite(
    tmp_path, monkeypatch,
):
    """The guard follows dynvor's loop bounds, not uninitialised work memory."""
    tool = _load_tool()
    monkeypatch.setattr(tool, "NATIVE_DIMS", (8, 7, 3))
    monkeypatch.setattr(tool, "ACTIVE_LEVELS", 2)
    nx, ny, _ = tool.NATIVE_DIMS
    slabs = [np.ones(nx * ny, dtype=np.float64) for _ in range(6)]
    slabs[0][0] = np.nan  # q(1,1): outside every executed/consumed loop bound.

    def write(path):
        path.write_bytes(
            b"NEMO_L2_R17Q__1 "
            + struct.pack("=8i", 1, 4321, 2, 4, nx, ny,
                          tool.ACTIVE_LEVELS, 64)
            + b"".join(array.tobytes() for array in slabs)
        )

    qpath = tmp_path / "q.bin"
    write(qpath)
    tool.read_ene_operands(qpath)
    slabs[0][nx + 1] = np.nan  # q(j=2,i=2): consumed by the live stencil.
    write(qpath)
    with pytest.raises(Exception, match="non-finite consumed q word"):
        tool.read_ene_operands(qpath)


def test_common_card_keeps_registered_deck_run_controls():
    """The developed-step number is forcing time, not a card namelist edit."""
    tool = _load_tool()
    card = tool._common_card()
    namelist = card.resolved_namelist
    assert namelist.nn_it000 == 1
    assert namelist.nn_itend == 26280
    assert namelist.nn_GYRE == 4
    assert namelist.rn_Dt_s == 3600.0
    assert namelist.nn_ahm_ijk_t == 32


def test_substitution_verdict_requires_twin_envelope_and_untouched_rows():
    tool = _load_tool()
    selection = np.ones((2, 3, 1), dtype=bool)
    vector = np.zeros(selection.shape)
    scalar = np.full(selection.shape, 2.0)
    baseline = np.full(selection.shape, 20.0)
    substituted = np.full(selection.shape, 2.5)
    report = tool.substitution_verdict(
        baseline, substituted, vector, scalar, selection, True)
    assert report["verdict"] == "CONFIRM_Q_FORMATION"
    assert report["baseline_candidate_to_twin_ratio"] == 10.0

    outside = tool.substitution_verdict(
        baseline, np.full(selection.shape, 4.01), vector, scalar,
        selection, True)
    assert outside["verdict"] == "REFUTE_Q_FORMATION"
    moved = tool.substitution_verdict(
        baseline, substituted, vector, scalar, selection, False)
    assert moved["verdict"] == "REFUTE_Q_FORMATION"


def test_source_accumulator_and_boundary_use_exact_executed_addition(
    monkeypatch,
):
    tool = _load_tool()
    before = np.ones((2, 3, 1), dtype=np.float64)
    term = np.full_like(before, 0.5)
    after = before + term
    monkeypatch.setattr(
        tool, "_native_v_field",
        lambda record, name: before if name == "before_v" else after,
    )
    selection = np.ones_like(before, dtype=bool)
    exact = tool.source_accumulator_result(term, {}, selection)
    assert exact["bit_exact"] is True
    assert exact["registered_unequal_words"] == 0
    moved = tool.source_accumulator_result(
        term + 1.0e-6, {}, selection)
    assert moved["bit_exact"] is False
    assert moved["registered_unequal_words"] == term.size

    arms = {
        "q_only": {"source_accumulator": {"bit_exact": False}},
        "Fu_only": {"source_accumulator": {"bit_exact": False}},
        "q_plus_Fu": {"source_accumulator": {"bit_exact": True}},
    }
    assert tool.classify_operand_boundary(arms) == (
        ["q_plus_Fu"], "TRANSPORT_FORMATION")
    arms["q_plus_Fu"]["source_accumulator"]["bit_exact"] = False
    assert tool.classify_operand_boundary(arms) == (
        [], "COMPLETED_STENCIL")


def test_wordwise_and_factor_classification_require_both_exact_boundaries():
    tool = _load_tool()
    oracle = np.array([1.0, 2.0], dtype=np.float64)
    assert tool.wordwise_result(oracle, oracle) == {
        "words": 2,
        "unequal_words": 0,
        "max_abs": 0.0,
        "max_ulp": 0,
        "bit_exact": True,
    }
    moved = oracle.copy()
    moved[1] = np.nextafter(moved[1], np.inf)
    row = tool.wordwise_result(moved, oracle)
    assert row["unequal_words"] == 1
    assert row["max_ulp"] == 1

    exact = {"bit_exact": True}
    debt = {"bit_exact": False}
    arms = {
        "formed_only": {
            "formed_operand": exact,
            "source_accumulator": debt,
            "untouched_rows_bit_exact": True,
        },
        "source_only": {
            "formed_operand": debt,
            "source_accumulator": exact,
            "untouched_rows_bit_exact": True,
        },
    }
    assert tool.classify_factor_walk(arms) == (
        [], "COUPLED_FORMATION_COMPOSITION")
    arms["single"] = {
        "formed_operand": exact,
        "source_accumulator": exact,
        "untouched_rows_bit_exact": True,
    }
    assert tool.classify_factor_walk(arms) == (
        ["single"], "SINGLE_FACTOR_SUFFICIENT")


def test_active_plants_select_consumed_nonzero_stencil_words():
    tool = _load_tool()
    western = np.zeros((3, 4), dtype=bool)
    western[1, 2] = True
    q = np.ones((3, 5, 1), dtype=np.float64)
    fu = np.zeros((2, 5, 1), dtype=np.float64)
    fu[1, 2, 0] = 3.0
    fu[1, 3, 0] = 1.0
    q_index, fu_index = tool.active_plant_locations(western, q, fu)
    assert q_index == (1, 2, 0)
    assert fu_index == (1, 2, 0)

    fu[:] = 0.0
    with pytest.raises(Exception, match="coefficient is zero"):
        tool.active_plant_locations(western, q, fu)


def test_model_tendency_operand_overrides_move_only_vorticity_component():
    """The shared path consumes recorded operands without rebuilding them."""
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("GYRE-zco")
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    jj_u, ii_u, kk_u = np.indices(state.u.data.shape)
    jj_v, ii_v, kk_v = np.indices(state.v.data.shape)
    u = (1.0e-3 * (1 + jj_u + 2 * ii_u + kk_u)
         * np.asarray(state.u_mask.data)[..., None])
    v = (-7.0e-4 * (2 + 2 * jj_v + ii_v + kk_v)
         * np.asarray(state.v_mask.data)[..., None])
    u[:, -1, :] = u[:, 0, :]
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
    )
    baseline = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        return_nemo_operator_components=True)
    q_shape = (state.T.data.shape[0] + 1,
               state.T.data.shape[1] + 1, state.T.data.shape[2])
    overridden = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        nemo_vorticity_q_override=jnp.zeros(q_shape, dtype=jnp.float64),
        return_nemo_operator_components=True)
    transport_overridden = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        nemo_vorticity_transport_override=(
            jnp.zeros_like(state.u.data), None),
        return_nemo_operator_components=True)
    baseline_components = baseline[2]
    override_components = overridden[2]
    transport_components = transport_overridden[2]
    for name, shape in (
        ("operand_vorticity_q", q_shape),
        ("operand_vorticity_Fu", state.u.data.shape),
        ("operand_vorticity_Fv", state.v.data.shape),
    ):
        assert name in baseline_components
        assert baseline_components[name].shape == shape
    replayed = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        nemo_vorticity_q_override=baseline_components["operand_vorticity_q"],
        nemo_vorticity_transport_override=(
            baseline_components["operand_vorticity_Fu"],
            baseline_components["operand_vorticity_Fv"],
        ),
        return_nemo_operator_components=True,
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_components["vorticity_u"].data),
        np.asarray(replayed[2]["vorticity_u"].data),
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_components["vorticity_v"].data),
        np.asarray(replayed[2]["vorticity_v"].data),
    )
    assert not np.array_equal(
        np.asarray(baseline_components["vorticity_v"].data),
        np.asarray(override_components["vorticity_v"].data),
    )
    assert not np.array_equal(
        np.asarray(baseline_components["vorticity_v"].data),
        np.asarray(transport_components["vorticity_v"].data),
    )
    for name in ("hpg_v", "advection_v", "operand_velocity_v",
                 "operand_momentum_h_v"):
        left = getattr(baseline_components[name], "data",
                       baseline_components[name])
        right = getattr(override_components[name], "data",
                        override_components[name])
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        transport_right = getattr(transport_components[name], "data",
                                  transport_components[name])
        np.testing.assert_array_equal(
            np.asarray(left), np.asarray(transport_right))


def test_operator_script_pins_common_restart_and_same_binary_control():
    source = RUN.read_text(encoding="utf-8")
    for needle in (
        "--preflight-vector", "--preflight-scalar", "--run-vector",
        "--run-scalar", "-d 'OCE'", "RANKS must be 1",
        "COMMON_RESTART_SHA256=a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12",
        "run_nemo control 0", "run_nemo instrumented 1",
        "REFUSE_WRITER_MOVED_TRAJECTORY", "GYRE_R17_WRITE",
        "PREFLIGHT_OK",
    ):
        assert needle in source
    assert "rg " not in source
    assert "--plant-fu" in TOOL.read_text(encoding="utf-8")
    model_source = (ROOT / "packages/ocean/legoesm/ocean/dynamics/"
                    "ocean_model_latlon_cgrid.py").read_text(encoding="utf-8")
    assert "if stage_index == 3 else None" in model_source
