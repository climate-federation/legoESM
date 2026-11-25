#!/usr/bin/env python3
"""Replay ORCA2 rung-0 kt=1 stage 1 from the admitted passive record.

The executable is never instrumented here.  Each calculation calls a shared
pure implementation on either the independent card entry or a recorded NEMO
operand, then scores the result against the next recorded boundary.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round175_stage1_acquisition import (
    check_record,
)

FLOOR = np.float64(2.0e-10)
CELLS = ((147, 49, 0), (86, 159, 3))
SOURCE_ORDER = (
    "entry_T", "entry_S", "entry_u", "entry_v", "entry_ssh",
    "external_stage_ssh", "r3t_stage1",
    "momentum_update_u", "momentum_update_v",
    "barotropic_correction_u", "barotropic_correction_v",
    "metric_transport_u", "metric_transport_v",
    "centered_advection_T", "centered_advection_S",
    "zero_surface_source_T", "zero_surface_source_S",
    "qco_update_T", "qco_update_S",
)
PLANTS = (
    "none", "admission", "rank-placement", "source-order", "cell",
    "active-mask", "first-debt", "ulp",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _payload_values(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read values using the record's self-described headers."""

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    offset = 92
    values: dict[str, np.ndarray] = {}
    for _ in check_record.NAMES:
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        count = n1 * n2 * n3
        local = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        offset += count * 8
        local = local.reshape((n1, n2, n3), order="F")
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        owned = np.array(local[ntsi - 1:ntei, ntsj - 1:ntej, :], copy=True)
        owned = owned.transpose(1, 0, 2)
        values[name] = owned[..., 0] if ndim == 2 else owned
    require(offset == len(raw), f"{path.name}: value parser did not consume record")
    require(tuple(values) == check_record.NAMES, "record field order moved")
    return metadata, values


def assemble_record(root: Path, baseline: Path) -> tuple[dict[str, np.ndarray], dict]:
    admission = check_record.run(root, baseline)
    arrays: dict[str, np.ndarray] = {}
    coverage = np.zeros((148, 180), dtype=np.int8)
    for rank in (0, 1):
        path = root / f"oracle_r175_stage1_rank{rank:04d}_kt00000001.bin"
        metadata, values = _payload_values(path)
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"rank {rank}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, block in values.items():
            shape = (148, 180) if block.ndim == 2 else (148, 180, block.shape[-1])
            arrays.setdefault(name, np.empty(shape, dtype=np.float64))[j0:j1, i0:i1, ...] = block
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")
    return arrays, admission


def _with_jpk_zero(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 3 and value.shape[-1] == 30, "physical level count moved")
    return np.concatenate((value, np.zeros_like(value[..., :1])), axis=-1)


def _score(candidate, oracle, active) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape,
            f"score shape mismatch {candidate.shape} {oracle.shape} {active.shape}")
    c, o = candidate[active], oracle[active]
    require(c.size > 0 and np.isfinite(c).all() and np.isfinite(o).all(),
            "score is empty or non-finite")
    delta = np.abs(c - o)
    differing = c.view(np.uint64) != o.view(np.uint64)
    flat_active = np.flatnonzero(active)
    max_pos = int(np.argmax(delta))
    argmax = list(np.unravel_index(int(flat_active[max_pos]), candidate.shape))
    return {
        "bit_exact": bool(np.array_equal(c, o)),
        "differing_cells": int(np.count_nonzero(differing)),
        "active_cells": int(c.size),
        "max_abs": float(delta[max_pos]),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "at_floor": bool(float(delta[max_pos]) <= FLOOR),
        "argmax": [int(v) for v in argmax],
    }


def _cell_rows(candidate, oracle) -> list[dict[str, object]]:
    rows = []
    for j, i, k in CELLS:
        if candidate.ndim == 2:
            cv, ov = candidate[j, i], oracle[j, i]
        else:
            cv, ov = candidate[j, i, k], oracle[j, i, k]
        rows.append({"cell": [j, i, k], "candidate": float(cv), "oracle": float(ov),
                     "abs_error": float(abs(cv - ov)),
                     "bit_exact": bool(np.asarray(cv).view(np.uint64) == np.asarray(ov).view(np.uint64))})
    return rows


def _geometry(card, h_ref, umask, vmask) -> list[dict[str, object]]:
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    h_ref = np.asarray(h_ref)
    rows = []
    for j, i, _ in CELLS:
        neighbours = {
            "west": bool(active[j, (i - 1) % 180, 0]),
            "east": bool(active[j, (i + 1) % 180, 0]),
            "south": bool(active[max(j - 1, 0), i, 0]),
            "north": bool(active[min(j + 1, 147), i, 0]),
        }
        levels = int(np.count_nonzero(active[j, i]))
        rows.append({
            "cell": [j, i], "nemo_mesh_input": "ORCA_R2_zps_domcfg.nc",
            "wet_levels": levels, "mbkt_1_based": levels,
            "partial_bottom_thickness_m": float(h_ref[j, i, max(levels - 1, 0)]),
            "e3t_levels_0_5_m": [float(v) for v in h_ref[j, i, :6]],
            "tmask_levels_0_5": [int(v) for v in active[j, i, :6]],
            "umask_levels_0_5": [int(v) for v in np.asarray(umask)[j, i, :6]],
            "vmask_levels_0_5": [int(v) for v in np.asarray(vmask)[j, i, :6]],
            "surface_neighbours_wet": neighbours,
            "land_adjacent": not all(neighbours.values()),
            "fold_row": bool(j == 147), "cyclic_seam": bool(i in (0, 179)),
        })
    return rows


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["status"] = "BROKEN"
    elif plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = report["source_order"][1], report["source_order"][0]
    elif plant == "cell":
        report["cells"][0] = [86, 159, 3]
    elif plant == "active-mask":
        report["active_counts"]["T"] -= 1
    elif plant == "first-debt":
        report["first_replayed_debt"] = {"name": "planted"}
    elif plant == "ulp":
        report["one_ulp_control"]["bit_exact"] = True
    require(report["claim_label"] == "independent", "claim label moved")
    require(report["admission"]["status"] == "PASS_R175_STAGE1_ADMISSION", "record admission moved")
    require(report["admission"]["rank_coverage"] == "exactly-once", "rank placement moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER, "source order moved")
    require(tuple(tuple(v) for v in report["cells"]) == CELLS, "registered cells moved")
    require(report["active_counts"] == {
        "T": report["rows"][0]["active_cells"],
        "S": report["rows"][1]["active_cells"],
        "u": report["rows"][2]["active_cells"],
        "v": report["rows"][3]["active_cells"],
        "ssh": report["rows"][4]["active_cells"],
    }, "active-mask registry moved")
    require(report["one_ulp_control"] == {"bit_exact": False, "differing_cells": 1},
            "one-ULP plant control did not fire")
    first = next((row for row in report["rows"] if not row["at_floor"]), None)
    require(first == report["first_replayed_debt"], "first replayed debt selector moved")
    stopped_at_entry = report.get("terminal_boundary") == "independent_initial_state"
    report["prediction_ledger"] = {
        "R176-P1": "CONFIRMED", "R176-P2": "CONFIRMED",
        "R176-P3": ("UNMEASURED" if stopped_at_entry else
                     ("CONFIRMED" if first is None else "REFUTED")),
        "R176-P4": "CONFIRMED" if report["recorded_adv_equals_sbc"] else "REFUTED",
        "R176-P5": ("UNMEASURED" if stopped_at_entry else
                     ("CONFIRMED" if first is None else "REFUTED")),
        "R176-P6": "CONFIRMED",
    }
    report["status"] = (
        "HELD_FIRST_NONBIT_INDEPENDENT_INITIAL_STATE" if stopped_at_entry else
        ("HELD_FIRST_UNAVAILABLE_CANDIDATE_EXTERNAL_MODE_AND_RHS"
         if first is None else "PASS_FIRST_REPLAYED_DEBT"))
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_common import rk3_stage_barotropic_correction
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        _nemo_metric_stage_transport, _nemo_ws_qco_stage_faces,
        _nemo_ws_rk3_tracer_pair_step, rk3_stage_velocity_update,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_face_masks_3d
    from legoesm.ocean.eos import nemo_r3t_rk3_stage1_stretch
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import compute_layer_thickness

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-176 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64), "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu", "round-176 replay is CPU-only")

    oracle, admission = assemble_record(record_root, baseline_root)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    entry = rung0.assemble_frame(frame_root, 1, 0)
    candidate_entry = rung0.candidate_fields(state)
    cfg = card.recipe.model_config
    grid = card.recipe.grid
    zc = card.recipe.z_coord
    h_ref = compute_layer_thickness(jnp.zeros_like(state.eta.data), state.H_bathy.data, zc,
                                    min_water_column_m=cfg.min_water_column_m).astype(jnp.float64)
    um3, vm3 = compute_face_masks_3d(zc.is_active, grid)
    um3, vm3 = um3.astype(jnp.float64), vm3.astype(jnp.float64)
    ops = r170._candidate_reference_operands(card, state)
    active_t = np.asarray(zc.is_active, dtype=bool)
    active_u = np.asarray(ops["mask_u"] != 0.0)
    active_v = np.asarray(ops["mask_v"] != 0.0)
    rows = []

    def add(name, candidate, expected, active):
        c, e = np.asarray(candidate, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        rows.append({"name": name, **_score(c, e, active), "registered_cells": _cell_rows(c, e)})

    for name, active in (("T", active_t), ("S", active_t), ("u", active_u[..., :-1]),
                         ("v", active_v[..., :-1]), ("ssh", np.asarray(state.H_bathy.data) > 0)):
        add(f"entry_{name}", candidate_entry[name], entry[name], active)

    active_counts = {
        "T": int(np.count_nonzero(active_t)), "S": int(np.count_nonzero(active_t)),
        "u": int(np.count_nonzero(active_u[..., :-1])),
        "v": int(np.count_nonzero(active_v[..., :-1])),
        "ssh": int(np.count_nonzero(np.asarray(state.H_bathy.data) > 0)),
    }
    entry_exact = all(row["bit_exact"] for row in rows)
    if not entry_exact:
        one = np.array([1.0], dtype=np.float64)
        next_one = np.nextafter(one, np.inf)
        all_cell_entry = {}
        for name in ("T", "S", "u", "v", "ssh"):
            left, right = np.asarray(candidate_entry[name]), np.asarray(entry[name])
            delta = np.abs(left - right)
            all_cell_entry[name] = {
                "bit_exact": bool(np.array_equal(left, right)),
                "differing_cells": int(np.count_nonzero(
                    left.view(np.uint64) != right.view(np.uint64))),
                "max_abs": float(np.max(delta)),
                "argmax": [int(v) for v in np.unravel_index(np.argmax(delta), delta.shape)],
            }
        return classify({
            "format": "nemo-testcase-l4-orca2-round176-stage1-offline-v1",
            "claim_label": "independent", "floor": float(FLOOR),
            "admission": admission, "source_order": list(SOURCE_ORDER),
            "cells": [list(v) for v in CELLS],
            "geometry": _geometry(card, h_ref, um3, vm3),
            "active_counts": active_counts,
            "rows": rows,
            "first_replayed_debt": next(row for row in rows if not row["at_floor"]),
            "recorded_adv_equals_sbc": bool(
                np.array_equal(oracle["adv_t"], oracle["sbc_t"])
                and np.array_equal(oracle["adv_s"], oracle["sbc_s"])),
            "terminal_boundary": "independent_initial_state",
            "all_cell_entry": all_cell_entry,
            "first_statement": {
                "name": "rung-0 initial T/S hand-alteration guard",
                "candidate": "build_orca2_initial_ts default applies ORCA_R2 alterations",
                "oracle": "alterations execute only under ln_tsd_dmp; rung 0 sets it false",
                "nemo_source": "ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-255",
                "candidate_source": "nemo_testcase_recipe.py:1413-1453,1650-1651",
            },
            "skipped_source_rows": list(SOURCE_ORDER[5:]),
            "first_unavailable_candidate_operand": None,
            "one_ulp_control": {"bit_exact": bool(np.array_equal(one, next_one)),
                                "differing_cells": 1},
            "worktree": stamp,
        })

    zero = jnp.zeros_like(state.eta.data, dtype=jnp.float64)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    surface = OceanSurfaceForcing(
        sw_down=zero, q_net=zero, tau_x=zero, tau_y=zero,
        freshwater=zero, salt_flux=zero, taum=zero,
        tau_i_native=zero, tau_j_native=zero,
    )
    stage_model = LatLonCGridOceanModel(
        grid, zc, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_stage=1),
    )
    final_model = LatLonCGridOceanModel(grid, zc, cfg)
    stage_state = jax.device_get(stage_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    final_state = jax.device_get(final_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    add("external_stage_ssh", stage_state.eta.data, oracle["ext_ssh"],
        np.asarray(state.H_bathy.data) > 0)
    q_after = nemo_r3t_rk3_stage1_stretch(
        zc, state.eta.data, final_state.eta.data, state.H_bathy.data)
    add("r3t_stage1", np.asarray(q_after) - 1.0, oracle["ext_r3t"], np.asarray(state.H_bathy.data) > 0)

    u0, v0 = state.u.data[:, 1:, :], state.v.data[1:, :, :]
    pre_u = rk3_stage_velocity_update(u0, jnp.asarray(oracle["rhs_u"][..., :-1]), card.dt_s / 3.0,
                                      jnp.asarray(active_u[..., :-1]), vector_form=True)
    pre_v = rk3_stage_velocity_update(v0, jnp.asarray(oracle["rhs_v"][..., :-1]), card.dt_s / 3.0,
                                      jnp.asarray(active_v[..., :-1]), vector_form=True)
    add("momentum_update_u", _with_jpk_zero(pre_u), oracle["pre_u"], active_u)
    add("momentum_update_v", _with_jpk_zero(pre_v), oracle["pre_v"], active_v)

    cor_u = rk3_stage_barotropic_correction(pre_u, jnp.asarray(oracle["ext_ub"]),
                                             jnp.asarray(ops["e3_u"][..., :-1]),
                                             jnp.asarray(ops["r1_h0_u"]),
                                             jnp.asarray(ops["mask_u"][..., :-1]))
    cor_v = rk3_stage_barotropic_correction(pre_v, jnp.asarray(oracle["ext_vb"]),
                                             jnp.asarray(ops["e3_v"][..., :-1]),
                                             jnp.asarray(ops["r1_h0_v"]),
                                             jnp.asarray(ops["mask_v"][..., :-1]))
    add("barotropic_correction_u", _with_jpk_zero(cor_u), oracle["cor_u"], active_u)
    add("barotropic_correction_v", _with_jpk_zero(cor_v), oracle["cor_v"], active_v)

    hu, hv, _, _ = _nemo_ws_qco_stage_faces(state.eta.data, h_ref, um3, vm3, grid)
    trp_u = _nemo_metric_stage_transport(jnp.asarray(grid.dy_u)[:, 1:], hu[:, 1:], cor_u)
    trp_v = _nemo_metric_stage_transport(jnp.asarray(grid.dx_v)[1:], hv[1:], cor_v)
    add("metric_transport_u", _with_jpk_zero(trp_u), oracle["trp_u"], active_u)
    add("metric_transport_v", _with_jpk_zero(trp_v), oracle["trp_v"], active_v)

    zfu_owned = jnp.asarray(oracle["trp_u"][..., :-1])
    zfv_owned = jnp.asarray(oracle["trp_v"][..., :-1])
    # zFw is a W-interface transport with jpk=31 levels for 30 T cells.
    # _nemo_cen2_tracer_rhs consumes its 29 interior interfaces via 1:-1;
    # dropping the terminal level here makes that slice one level too short.
    zfw = jnp.asarray(oracle["trp_w"])
    require(zfw.shape[-1] == state.T.data.shape[-1] + 1,
            "recorded W transport does not carry all T-cell interfaces")
    zfu = jnp.concatenate((zfu_owned[:, -1:, :], zfu_owned), axis=1)
    zfv = jnp.concatenate((zfv_owned[-1:, :, :], zfv_owned), axis=0)
    geom = (zfu / jnp.asarray(grid.dy_u)[..., None],
            zfv / jnp.asarray(grid.dx_v)[..., None],
            zfw / jnp.asarray(grid.area_T)[..., None],
            h_ref, hu, hv, None, zfu, zfv)
    zero_t = jnp.zeros_like(state.T.data)
    trace = _nemo_ws_rk3_tracer_pair_step(
        state.T.data, state.S.data, "fct2", geom[0], geom[1], geom[2],
        h_ref, h_ref * q_after[..., None], hu, hv, grid, card.dt_s, jnp.asarray(active_t),
        stage_transport_geometry=(geom, geom, geom),
        stage_source_rates=((zero_t, zero_t),) * 3,
        stage_qco_weights=((jnp.ones_like(q_after), jnp.ones_like(q_after), q_after),) * 3,
        stop_after_stage=1, return_stage1_trace=True)
    qco_t, qco_s, adv_t, adv_s, sbc_t, sbc_s = trace
    for name, value in (("centered_advection_T", adv_t), ("centered_advection_S", adv_s),
                        ("zero_surface_source_T", sbc_t), ("zero_surface_source_S", sbc_s),
                        ("qco_update_T", qco_t), ("qco_update_S", qco_s)):
        target = {"centered_advection_T": "adv_t", "centered_advection_S": "adv_s",
                  "zero_surface_source_T": "sbc_t", "zero_surface_source_S": "sbc_s",
                  "qco_update_T": "qco_t", "qco_update_S": "qco_s"}[name]
        add(name, _with_jpk_zero(value), oracle[target], np.concatenate((active_t, np.zeros_like(active_t[..., :1])), axis=-1))

    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    raw = {
        "format": "nemo-testcase-l4-orca2-round176-stage1-offline-v1",
        "claim_label": "independent", "floor": float(FLOOR),
        "admission": admission, "source_order": list(SOURCE_ORDER),
        "cells": [list(v) for v in CELLS], "geometry": _geometry(card, h_ref, um3, vm3),
        "active_counts": active_counts,
        "rows": rows, "first_replayed_debt": next((row for row in rows if not row["at_floor"]), None),
        "recorded_adv_equals_sbc": bool(np.array_equal(oracle["adv_t"], oracle["sbc_t"])
                                             and np.array_equal(oracle["adv_s"], oracle["sbc_s"])),
        "first_unavailable_candidate_operand": "external_mode_and_completed_momentum_rhs",
        "one_ulp_control": {"bit_exact": bool(np.array_equal(one, next_one)), "differing_cells": 1},
        "worktree": stamp,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.baseline_root, args.expect_commit)), "measurement arguments missing")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, GateError, check_record.Refusal,
            rung0.GateError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
