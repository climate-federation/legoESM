#!/usr/bin/env python3
"""Walk OMT-0's kt=1 split-explicit solve from admitted passive records."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as r15,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round172_passive_rhs_replay as r172,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from nemo_testcase_l2_gyre_round14_advmean import read_ordered

FLOOR = np.float64(2.0e-10)
PLANTS = (
    "none", "record-header", "twin-ulp", "source-order", "passivity",
    "terminal-ulp",
)
STREAMS = (
    ("substeps", "oracle_bt_substeps_kt00000001.bin"),
    ("ordered", "oracle_bt_ordered_operands_kt00000001.bin"),
)
ENTRY_ORDER = (
    "continuity_forcing", "slow_u", "slow_v", "eta_entry", "u_entry",
    "v_entry", "eta_history_b", "eta_history_bb", "u_history_b",
    "u_history_bb", "v_history_b", "v_history_bb",
)
SUBSTEP_ORDER = (
    "eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
    "transport_metric_u", "transport_metric_v", "eta_exit", "eta_pgf",
    "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v", "slow_u",
    "slow_v", "u_exit", "v_exit",
)


class GateError(RuntimeError):
    """The admitted OMT-0 record or source-ordered replay moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate, oracle, mask) -> dict[str, object]:
    # Fold and cyclic halo points are inputs to neighbouring active-cell
    # stencils.  They therefore belong to the statement domain even when the
    # prognostic active mask excludes them (round 178's established rule).
    return r178._score(candidate, oracle, mask, complete_domain=True)


def _first(rows: list[dict[str, object]], key: str) -> dict[str, object] | None:
    return next((row for row in rows if not bool(row[key])), None)


def _admit_streams(twin_a: Path, twin_b: Path) -> dict[str, object]:
    masks = o1_gate._defined_masks(twin_a)
    rows = []
    for family, name in STREAMS:
        left, right = twin_a / name, twin_b / name
        require(left.is_file() and right.is_file(), f"missing inherited {name}")
        parsed_left = phase1._parse_bt(left, family)
        parsed_right = phase1._parse_bt(right, family)
        exact = o1_gate._compare_hygiene_record(left, right, masks)
        require(exact["status"] == "EXACT_DEFINED_BYTES",
                f"{name}: twin defined payload differs")
        rows.append({
            "family": family, "name": name, "header_valid": True,
            "left": parsed_left, "right": parsed_right,
            "defined_twin_status": exact["status"],
            "defined_f64": exact["defined_f64"],
            "canonical_zero_f64": exact["canonical_zero_f64"],
        })
    return {"stream_count": len(rows), "streams": rows}


def _native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:91]


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:149, :90]


def _native_t(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, :90]


def _slow_forcing(model, card, state, surface):
    """Replay the live stage-1 RHS and NEMO stp2d depth average offline."""

    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_coriolis_een_pre_step,
        nemo_carried_barotropic_depth_mean,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_face_masks_3d
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
    )

    tendency = jax.device_get(jax.jit(
        lambda source, forcing: r172._stage1_tendency(
            model, source, forcing, card.dt_s, components=False)
    )(state, surface))
    du_dt = jnp.asarray(tendency.du_dt.data)
    dv_dt = jnp.asarray(tendency.dv_dt.data)
    cfg, grid, z_coord = card.recipe.model_config, card.recipe.grid, card.recipe.z_coord
    dtype = du_dt.dtype
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=cfg.min_water_column_m).astype(dtype)
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_card_mesh_operands(
        h_ref, umask3.astype(dtype), vmask3.astype(dtype), grid, dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    wet_u = (ops.hu_0 > 0.0).astype(dtype)
    wet_v = (ops.hv_0 > 0.0).astype(dtype)
    r1_hu = wet_u / (ops.hu_0 + one - wet_u)
    r1_hv = wet_v / (ops.hv_0 + one - wet_v)
    slow_u_native = jnp.sum(
        ops.e3u_0 * du_dt[:, 1:, :] * ops.umask3, axis=-1) * r1_hu
    slow_v_native = jnp.sum(
        ops.e3v_0 * dv_dt[1:, :, :] * ops.vmask3, axis=-1) * r1_hv

    raw, fold = z_coord.nemo_een_barotropic, grid.fold
    require(raw is not None and fold is not None and bool(fold.is_active),
            "OMT-0 atomic HPG fold operands are absent")
    raw_e3v = jnp.asarray(raw.e3v_0, dtype=dtype)[..., :dv_dt.shape[-1]]
    raw_vmask = jnp.asarray(raw.vmask, dtype=dtype)[..., :dv_dt.shape[-1]]
    raw_hv = jnp.asarray(raw.hv_0, dtype=dtype)
    raw_wet_v = (raw_hv > 0.0).astype(dtype)
    fold_unit_v = (raw_wet_v > 0.0) & (wet_v == 0.0)
    raw_r1_hv = raw_wet_v / (raw_hv + one - raw_wet_v)
    raw_slow_v = jnp.sum(
        raw_e3v * dv_dt[1:, :, :] * raw_vmask, axis=-1) * raw_r1_hv
    slow_v_native = jnp.where(fold_unit_v, raw_slow_v, slow_v_native)

    slow_u = jnp.zeros_like(state.u_mask.data).at[:, 1:].set(slow_u_native)
    slow_v = jnp.zeros_like(state.v_mask.data).at[1:, :].set(slow_v_native)
    slow_u = slow_u * state.u_mask.data
    slow_v_masked = slow_v * state.v_mask.data
    slow_v = slow_v_masked.at[1:, :].set(jnp.where(
        fold_unit_v, slow_v[1:, :], slow_v_masked[1:, :]))

    carried = nemo_carried_barotropic_depth_mean(state, dtype, cfg)
    min_wc = jnp.asarray(cfg.min_water_column_m, dtype=dtype)
    cor_u, cor_v, _ = barotropic_coriolis_een_pre_step(
        state.u.data, state.v.data, h_ref, grid, state.land_mask.data,
        state.u_mask.data, state.v_mask.data, min_wc, dtype,
        metric_complete=True,
        een_q_boundary=cfg.een_q_boundary,
        een_e3f_scheme=cfg.een_e3f_scheme,
        dz_ref=z_coord.dz_ref,
        coefficient_evaluation="nemo_literal",
        eta=state.eta.data, z_coord=z_coord, return_pre=True,
        scheme="een", entry_barotropic_velocity=carried,
    )
    slow_u = (slow_u - cor_u) * state.u_mask.data
    slow_v = (slow_v - cor_v) * state.v_mask.data
    return tuple(np.asarray(value) for value in jax.device_get((slow_u, slow_v)))


def _terminal_rows(left, right) -> dict[str, bool]:
    left_state, left_transport = left[:2]
    right_state, right_transport = right[:2]
    return {
        "ssh": bool(np.array_equal(left_state.eta.data, right_state.eta.data)),
        "u_barotropic": bool(np.array_equal(left_state.uu_b.data, right_state.uu_b.data)),
        "v_barotropic": bool(np.array_equal(left_state.vv_b.data, right_state.vv_b.data)),
        "transport_u": bool(np.array_equal(left_transport[0], right_transport[0])),
        "transport_v": bool(np.array_equal(left_transport[1], right_transport[1])),
    }


def _summary_table(trace, oracle, masks) -> tuple[list[dict], list[dict]]:
    registry = (
        ("eta_entry", "eta_entry", "t"),
        ("u_entry", "u_entry", "u"),
        ("v_entry", "v_entry", "v"),
        ("eta_mid", "eta_mid", "t"),
        ("u_mid", "u_mid", "u"),
        ("v_mid", "v_mid", "v"),
        ("transport_metric_u", "transport_metric_u", "u"),
        ("transport_metric_v", "transport_metric_v", "v"),
        ("eta_exit", "eta_exit", "t"),
        ("eta_pgf", "eta_pgf", "t"),
        ("pgf_u", "pgf_u", "u"),
        ("pgf_v", "pgf_v", "v"),
        ("cor_u", "cor_u", "u"),
        ("cor_v", "cor_v", "v"),
        ("trd_u", "trd_u", "u"),
        ("trd_v", "trd_v", "v"),
        ("slow_u", "slow_u", "u"),
        ("slow_v", "slow_v", "v"),
        ("u_exit", "u_exit", "u"),
        ("v_exit", "v_exit", "v"),
    )
    rows, table = [], []
    converters = {"t": _native_t, "u": _native_u, "v": _native_v}
    for substep in range(65):
        step_rows = []
        for name, trace_name, face in registry:
            row = {"substep": substep + 1, "name": name, **_score(
                converters[face](trace[trace_name][substep]),
                oracle[name][substep], masks[face])}
            rows.append(row)
            step_rows.append(row)
        by_name = {row["name"]: row for row in step_rows}
        first_nonbit = _first(step_rows, "comparison_bit_exact")
        first_over = _first(step_rows, "at_floor")
        table.append({
            "substep": substep + 1,
            "first_nonbit": None if first_nonbit is None else first_nonbit["name"],
            "first_over_floor": None if first_over is None else first_over["name"],
            "ssh": by_name["eta_exit"],
            "ua_b": by_name["u_exit"],
            "va_b": by_name["v_exit"],
        })
    return rows, table


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "record-header":
        report["record_admission"]["streams"][0]["header_valid"] = False
    elif plant == "twin-ulp":
        report["record_admission"]["streams"][0]["defined_twin_status"] = "DIFF"
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "passivity":
        report["offline_replay_passivity"]["ssh"] = False
    elif plant == "terminal-ulp":
        report["terminal_ulp_control"]["bit_exact"] = True

    require(report["claim_label"] == "independent OMT-0", "claim label moved")
    require(report["record_admission"]["stream_count"] == 2, "stream census moved")
    require(all(row["header_valid"] for row in report["record_admission"]["streams"]),
            "record header plant or malformed record")
    require(all(row["defined_twin_status"] == "EXACT_DEFINED_BYTES"
                for row in report["record_admission"]["streams"]),
            "barotropic twin payload differs")
    require(all(report["offline_replay_passivity"].values()),
            "offline trace does not reproduce the untraced pure solver")
    require(tuple(report["source_order"]) == ENTRY_ORDER,
            "entry source order moved")
    require(report["terminal_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "terminal one-ULP control did not fire")
    require(report["slow_v_arm"]["input"]["comparison_bit_exact"],
            "recorded slow-V substitution did not install exactly")
    require(len(report["slow_v_arm"]["substep_table"]) == 65,
            "slow-V arm is incomplete")
    require(len(report["slow_v_association_unit"]["substep_table"]) == 65,
            "slow-V plus association unit is incomplete")
    require(len(report["substep_table"]) == 65, "substep table is incomplete")
    require(report["first_nonbit"] == _first(
        report["source_rows"], "comparison_bit_exact"),
            "first non-bit selector moved")
    require(report["first_over_floor"] == _first(
        report["source_rows"], "at_floor"),
            "first-over-floor selector moved")
    report["status"] = "PASS_R205_OMT0_SUBSTEP_WALK"
    return report


def measure(deck_root: Path, twin_a: Path, twin_b: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower().startswith(expect_commit.lower()),
            "round-205 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-205 replay requires production JIT on CPU")

    admission = _admit_streams(twin_a, twin_b)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    card = omt0.build_omt0_card(deck_root)
    omt0.validate_omt0_card(card)
    cfg = card.recipe.model_config
    resolved = {
        "dtype": str(card.recipe.initial_state.eta.data.dtype),
        "outer_integrator": cfg.outer_integrator,
        "momentum_time_integrator": cfg.momentum_time_integrator,
        "momentum_advection": cfg.momentum_advection,
        "momentum_flux_scheme": cfg.momentum_flux_scheme,
        "vertical_momentum_scheme": cfg.vertical_momentum_scheme,
        "barotropic_solver": cfg.barotropic.barotropic_solver,
        "n_substeps": cfg.barotropic.n_barotropic_substeps,
        "time_filter": cfg.barotropic.barotropic_time_filter,
        "continuity": cfg.barotropic.barotropic_continuity_evaluation,
        "transport": cfg.barotropic.barotropic_transport_accumulation_evaluation,
        "face_depth": cfg.barotropic.barotropic_face_depth,
        "pgf": cfg.barotropic.barotropic_pgf_evaluation,
        "coriolis": cfg.barotropic.barotropic_coriolis,
        "drag": bool(cfg.barotropic_drag_substep),
    }
    require(resolved == {
        "dtype": "float64", "outer_integrator": "forward_euler",
        "momentum_time_integrator": "rk3_ws", "momentum_advection": "flux_form",
        "momentum_flux_scheme": "none", "vertical_momentum_scheme": "none",
        "barotropic_solver": "explicit_substep", "n_substeps": 65,
        "time_filter": "nemo_ab3am4", "continuity": "nemo_literal",
        "transport": "nemo_literal", "face_depth": "nemo_ssh_avg",
        "pgf": "nemo_literal", "coriolis": "een_metric", "drag": False,
    }, f"resolved OMT-0 identity moved: {resolved}")

    state = card.recipe.initial_state
    freshwater, surface = rung0_ladder._zero_forcing(state.eta.data.shape)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    model.prime_step_caches(state)
    slow_u, slow_v = _slow_forcing(model, card, state, surface)
    zero_eta = np.zeros(state.eta.data.shape, dtype=np.float64)
    dt = np.float64(card.dt_s / 65)
    require(np.array_equal(np.asarray([dt]), np.asarray([ordered["dt"]])),
            f"external substep dt moved: {dt} != {ordered['dt']}")

    def solve(trace: bool, association: bool = False, slow_v_override=None):
        selected_v = f_v if slow_v_override is None else slow_v_override
        return jax.device_get(jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, dt, 65, card.recipe.grid, card.recipe.z_coord, cfg,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
                _nemo_external_mode_association_test_override=association,
            )
        ))(state, jnp.asarray(f_eta), jnp.asarray(f_u), jnp.asarray(selected_v)))

    f_eta, f_u, f_v = zero_eta, slow_u, slow_v
    live = solve(False)
    traced = solve(True)
    passivity = _terminal_rows(live, traced)
    require(all(passivity.values()), f"offline trace is not passive: {passivity}")
    trace = traced[2]

    # Use the already-gated rank-0 staggering convention.  In particular the
    # U native payload starts at halo column 1 and the V payload at halo row 1.
    masks = r15._masks(card)
    candidate_ordered = r15._candidate_ordered(card, trace)
    entry_candidates = {
        "continuity_forcing": candidate_ordered["continuity_forcing"][0],
        "slow_u": _native_u(slow_u), "slow_v": _native_v(slow_v),
        "eta_entry": candidate_ordered["eta_entry"][0],
        "u_entry": candidate_ordered["u_entry"][0],
        "v_entry": candidate_ordered["v_entry"][0],
        "eta_history_b": candidate_ordered["eta_history_b"][0],
        "eta_history_bb": candidate_ordered["eta_history_bb"][0],
        "u_history_b": candidate_ordered["u_history_b"][0],
        "u_history_bb": candidate_ordered["u_history_bb"][0],
        "v_history_b": candidate_ordered["v_history_b"][0],
        "v_history_bb": candidate_ordered["v_history_bb"][0],
    }
    face = {name: ("u" if "_u" in name or name.startswith("u_") else
                   "v" if "_v" in name or name.startswith("v_") else "t")
            for name in ENTRY_ORDER}
    entry_rows = [{"name": name, **_score(
        entry_candidates[name],
        ordered[name][0] if name not in ("continuity_forcing", "slow_u", "slow_v")
        else (ordered["continuity_forcing"][0] if name == "continuity_forcing"
              else ordered[name][0]),
        masks[face[name]])} for name in ENTRY_ORDER]

    substep_rows, substep_table = _summary_table(trace, substeps, masks)
    source_rows = entry_rows + substep_rows
    first_nonbit = _first(source_rows, "comparison_bit_exact")
    first_over = _first(source_rows, "at_floor")

    association = solve(True, association=True)
    association_trace = association[2]
    association_rows, association_table = _summary_table(
        association_trace, substeps, masks)
    recorded_slow_v = np.asarray(slow_v).copy()
    recorded_slow_v[1:149, :90] = ordered["slow_v"][0]
    slow_v_arm = solve(True, slow_v_override=recorded_slow_v)
    slow_v_rows, slow_v_table = _summary_table(slow_v_arm[2], substeps, masks)
    unit_arm = solve(
        True, association=True, slow_v_override=recorded_slow_v)
    unit_rows, unit_table = _summary_table(unit_arm[2], substeps, masks)
    one = np.asarray([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    raw = {
        "format": "nemo-testcase-l4-orca2-round205-omt0-substep-v1",
        "claim_label": "independent OMT-0",
        "execution": "offline-production-pure-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR), "worktree": stamp,
        "record_admission": admission, "resolved": resolved,
        "source_order": list(ENTRY_ORDER), "entry_rows": entry_rows,
        "entry_first_nonbit": _first(entry_rows, "comparison_bit_exact"),
        "offline_replay_passivity": passivity,
        "substep_source_order": list(SUBSTEP_ORDER),
        "source_rows": source_rows,
        "substep_rows": substep_rows, "substep_table": substep_table,
        "first_nonbit": first_nonbit, "first_over_floor": first_over,
        "association_arm": {
            "source_statement": "dynspg_ts.f90:712-741",
            "terminal": _terminal_rows(live, association),
            "first_nonbit": _first(
                association_rows, "comparison_bit_exact"),
            "first_over_floor": _first(association_rows, "at_floor"),
            "substep_table": association_table,
        },
        "slow_v_arm": {
            "source_statement": "dynspg_ts.f90:289,320-324",
            "input": _score(_native_v(recorded_slow_v),
                            ordered["slow_v"][0], masks["v"]),
            "terminal": _terminal_rows(live, slow_v_arm),
            "first_nonbit": _first(
                slow_v_rows, "comparison_bit_exact"),
            "first_over_floor": _first(slow_v_rows, "at_floor"),
            "substep_table": slow_v_table,
        },
        "slow_v_association_unit": {
            "source_statements": [
                "dynspg_ts.f90:289,320-324", "dynspg_ts.f90:712-741"],
            "terminal": _terminal_rows(live, unit_arm),
            "first_nonbit": _first(unit_rows, "comparison_bit_exact"),
            "first_over_floor": _first(unit_rows, "at_floor"),
            "substep_table": unit_table,
        },
        "terminal_ulp_control": {
            "bit_exact": bool(np.array_equal(one, next_one)),
            "differing_cells": int(np.count_nonzero(one != next_one)),
        },
        "compiled_source": {
            "forcing": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:286-324",
            "entry": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:339-381",
            "substeps": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:445-813",
            "association": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741",
        },
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.twin_a, args.twin_b,
                         args.expect_commit)), "measurement arguments missing")
            result = measure(args.deck_root, args.twin_a, args.twin_b,
                             args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            phase1.GateError, o1_gate.GateError, omt0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R205_OMT0_SUBSTEP_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
