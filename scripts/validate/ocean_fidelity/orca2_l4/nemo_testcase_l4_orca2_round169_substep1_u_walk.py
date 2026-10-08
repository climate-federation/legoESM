#!/usr/bin/env python3
"""Walk ORCA2 kt=8 external substep-1 U update and exchange."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round167_exit_depth_walk as r167,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round168_exit_depth_operands as r168,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

PLANTS = (
    "none", "source-order", "recorded-replay", "entry-replay",
    "exchange-replay", "face-registry",
)
SOURCE_ORDER = (
    "entry_u", "pressure_u", "coriolis_u", "drag_u", "trend_u", "slow_u",
    "rhs_pressure_trend", "rhs_complete", "increment_u",
    "update_pre_exchange", "update_post_exchange",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)


def _score(candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray) -> dict:
    return r167._score(candidate, oracle, active)


def _first_nonbit(rows: list[dict]) -> dict | None:
    return next((row for row in rows if not row["bit_exact"]), None)


def _literal_update(entry, pressure, trend, slow, dt, mask):
    """Replay dynspg_ts.f90:712-723 with one barrier per source operation."""
    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round as b

    entry = jnp.asarray(entry, dtype=jnp.float64)
    pressure = jnp.asarray(pressure, dtype=jnp.float64)
    trend = jnp.asarray(trend, dtype=jnp.float64)
    slow = jnp.asarray(slow, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=jnp.float64)
    rhs_pt = b(b(pressure) + b(trend))
    rhs = b(rhs_pt + b(slow))
    increment = b(jnp.asarray(dt, dtype=jnp.float64) * rhs)
    update = b(b(entry) + increment)
    update = b(update * mask)
    return tuple(np.asarray(value) for value in jax.device_get(
        (rhs_pt, rhs, increment, update)))


def _recorded_exchange_replay(card, oracle, update_pre):
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    operands = (
        r97._to_model_u(update_pre),
        r97._to_model_v(oracle["j001_va_new"]),
        r97._to_model_u(oracle["j001_hu_e"]),
        r97._to_model_v(oracle["j001_hv_e"]),
        r97._to_model_u(oracle["j001_hur_e"]),
        r97._to_model_v(oracle["j001_hvr_e"]),
        oracle["j001_ssha_e"],
    )
    post = jax.device_get(_nemo_external_mode_boundary_association(
        *(jnp.asarray(value, dtype=jnp.float64) for value in operands),
        card.recipe.grid,
    ))
    return r97._native_u(post[0])


def _face_average_localisation(
    card, oracle_face_replay, oracle, u_mask, v_mask, prep, registered,
):
    """Name the round-168 residual face and score pre/post eta timing."""
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
        _nemo_ssh_avg_apply,
    )

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no raw NEMO face operands")
    before_eta = np.asarray(oracle["j002_ssha_e"])
    target = np.asarray(oracle["j002_sshu_a"])
    registered = np.asarray(registered, dtype=bool)
    require(registered.shape == target.shape
            and int(np.count_nonzero(registered)) == 41,
            "round-168 registered face census moved")
    residual = (_bits(oracle_face_replay) != _bits(target)) & registered
    residual_count = int(np.count_nonzero(residual))
    require(residual_count == 1,
            f"round-168 registered face-average residual census moved: "
            f"{residual_count}/41")
    location = tuple(map(int, np.argwhere(residual)[0]))

    zero_u = np.zeros((148, 181), dtype=np.float64)
    zero_v = np.zeros((149, 180), dtype=np.float64)
    zero_t = np.zeros((148, 180), dtype=np.float64)
    associated = jax.device_get(_nemo_external_mode_boundary_association(
        jnp.asarray(zero_u), jnp.asarray(zero_v), jnp.asarray(zero_u),
        jnp.asarray(zero_v), jnp.asarray(zero_u), jnp.asarray(zero_v),
        jnp.asarray(before_eta), card.recipe.grid,
    ))
    after_eta = np.asarray(associated[6])

    def replay(eta):
        values = jax.device_get(_nemo_ssh_avg_apply(
            jnp.asarray(eta), u_mask, v_mask, card.recipe.grid,
            jnp.asarray(card.recipe.grid.area), prep,
            return_literal_inverse=True, return_ssh_average=True))
        return r97._native_u(values[4])

    before_replay = replay(before_eta)
    after_replay = replay(after_eta)
    point = np.zeros(target.shape, dtype=bool)
    point[location] = True
    return {
        "residual_j_i": list(location),
        "before_association": _score(before_replay, target, point),
        "after_association": _score(after_replay, target, point),
        "input_timing_changed": bool(not np.array_equal(before_eta, after_eta)),
    }


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "recorded-replay":
        report["replays"]["all_recorded_pre"]["differing_cells"] += 1
    elif plant == "entry-replay":
        report["replays"]["recorded_entry_candidate_terms"]["count"] -= 1
    elif plant == "exchange-replay":
        report["replays"]["recorded_exchange"]["differing_cells"] += 1
    elif plant == "face-registry":
        report["face_average"]["residual_j_i"] = [0, 0]

    admission = report["admission"]
    require(admission["rank_coverage"] == "exactly-once"
            and len(admission["records"]) == 2,
            "rank-complete record admission moved")
    require(len(admission["terminal_restart_comparisons"]) == 20,
            "restart comparison census moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "substep-1 source order moved")
    require(report["replays"]["all_recorded_pre"]["differing_cells"] == 0,
            "all-recorded vector update no longer closes the interior")
    require(report["replays"]["recorded_exchange"]["differing_cells"] == 0,
            "recorded U exchange replay no longer closes")
    require(report["replays"]["recorded_entry_candidate_terms"]["count"] > 0,
            "entry-U one-variable replay coverage is empty")
    require(report["face_average"]["residual_j_i"] ==
            report["frozen_face_residual_j_i"],
            "round-168 face-average residual registry moved")

    first = report["first_nonbit"]
    p1 = first is not None and first["name"] == "entry_u"
    p2 = report["replays"]["all_recorded_pre"]["bit_exact"]
    p3 = report["replays"]["recorded_exchange"]["bit_exact"]
    p4 = (report["face_average"]["before_association"]["bit_exact"]
          or report["face_average"]["after_association"]["bit_exact"])
    report["prediction_dispositions"] = {
        "R169-P1": "CONFIRMED" if p1 else "REFUTED",
        "R169-P2": "CONFIRMED" if p2 else "REFUTED",
        "R169-P3": "CONFIRMED" if p3 else "REFUTED",
        "R169-P4": "CONFIRMED" if p4 else "REFUTED",
        "R169-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND169_SUBSTEP1_U_WALK"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            baseline_root: Path, expect_commit: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round as b
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_reference_face_depths,
        _nemo_ssh_avg_apply,
        _nemo_ssh_avg_prep,
        _nemo_ssh_avg_reference_depth_override,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-169 measurement requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-169 walk requires production JIT on CPU")

    admission = check_record.run(
        spg_root, baseline_root, "none", expected_kt=r167.EXPECTED_KT,
        prefix=r167.PREFIX, expected_magic=r167.MAGIC)
    oracle, census = r97.assemble_record(
        spg_root, prefix=r167.PREFIX, expected_kt=r167.EXPECTED_KT,
        expected_magic=r167.MAGIC)
    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    for kt in range(1, 8):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS round169 complete kt={kt}", file=sys.stderr, flush=True)
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks._replace(expose_barotropic_substeps=True))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = observed.substeps

    masks = phase3_gate.expected_masks(card)
    active_u = np.asarray(masks["u"][..., 0], dtype=bool)
    interior_u = np.array(active_u, copy=True)
    interior_u[-1, :] = False
    dt = float(np.asarray(oracle["i000_entry_sc"])[0])
    candidate = {
        "entry_u": r97._native_u(trace["u_entry"][0]),
        "pressure_u": r97._native_u(trace["pgf_u"][0]),
        "coriolis_u": r97._native_u(trace["cor_u"][0]),
        "drag_u": r97._native_u(trace["drag_u"][0]),
        "trend_u": r97._native_u(trace["trd_u"][0]),
        "slow_u": r97._native_u(trace["slow_u"][0]),
        "update_post_exchange": r97._native_u(trace["u_exit"][0]),
    }
    oracle_drag = np.asarray(jax.device_get(b(
        b(jnp.asarray(oracle["i000_zCdU_u"]) *
          jnp.asarray(oracle["i000_un_e"])) *
        jnp.asarray(oracle["i000_hur_e"]))))
    reference = {
        "entry_u": oracle["i000_un_e"],
        "pressure_u": oracle["j001_zu_spg"],
        "coriolis_u": oracle["j001_cor_u"],
        "drag_u": oracle_drag,
        "trend_u": oracle["j001_trd_u"],
        "slow_u": oracle["i000_zu_frc"],
        "update_post_exchange": oracle["j001_ua_new"],
    }
    c_rhs_pt, c_rhs, c_inc, c_pre = _literal_update(
        candidate["entry_u"], candidate["pressure_u"], candidate["trend_u"],
        candidate["slow_u"], dt, active_u)
    o_rhs_pt, o_rhs, o_inc, o_pre = _literal_update(
        reference["entry_u"], reference["pressure_u"], reference["trend_u"],
        reference["slow_u"], dt, active_u)
    candidate.update({
        "rhs_pressure_trend": c_rhs_pt, "rhs_complete": c_rhs,
        "increment_u": c_inc, "update_pre_exchange": c_pre,
    })
    reference.update({
        "rhs_pressure_trend": o_rhs_pt, "rhs_complete": o_rhs,
        "increment_u": o_inc, "update_pre_exchange": oracle["j001_ua_new"],
    })
    rows = []
    for name in SOURCE_ORDER:
        mask = interior_u if name == "update_pre_exchange" else active_u
        rows.append({"name": name, **_score(candidate[name], reference[name], mask)})

    recorded_post = _recorded_exchange_replay(card, oracle, o_pre)
    recorded_entry = _literal_update(
        reference["entry_u"], candidate["pressure_u"], candidate["trend_u"],
        candidate["slow_u"], dt, active_u)[3]
    entry_replay = _score(
        recorded_entry, oracle["j001_ua_new"], interior_u)
    entry_replay["count"] = int(np.count_nonzero(interior_u))
    replays = {
        "all_recorded_pre": _score(o_pre, oracle["j001_ua_new"], interior_u),
        "recorded_entry_candidate_terms": entry_replay,
        "recorded_exchange": _score(
            recorded_post, oracle["j001_ua_new"], active_u),
    }

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no raw NEMO masks")
    u_mask = jnp.asarray(r97._to_model_u(
        np.max(np.asarray(raw.umask), axis=-1)), dtype=jnp.float64)
    v_mask = jnp.asarray(r97._to_model_v(
        np.max(np.asarray(raw.vmask), axis=-1)), dtype=jnp.float64)
    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data), jnp.asarray(state.land_mask.data),
        card.recipe.grid, jnp.float64)
    prep = _nemo_ssh_avg_reference_depth_override(
        prep, _nemo_literal_reference_face_depths(
            card.recipe.z_coord, jnp.float64), jnp.float64)
    replay = jax.device_get(_nemo_ssh_avg_apply(
        jnp.asarray(oracle["j002_ssha_e"]), u_mask, v_mask,
        card.recipe.grid, jnp.asarray(card.recipe.grid.area), prep,
        return_literal_inverse=True, return_ssh_average=True))
    face_replay = r97._native_u(replay[4])
    registered_faces = ~np.isfinite(
        r97._native_u(trace["r1_face_depth_u_exit"][1]))
    face_average = _face_average_localisation(
        card, face_replay, oracle, u_mask, v_mask, prep, registered_faces)

    first = _first_nonbit(rows)
    raw_report = {
        "format": "nemo-testcase-l4-orca2-round169-substep1-u-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": 8,
        "substep": 1,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "first_nonbit": first,
        "replays": replays,
        "face_average": face_average,
        "frozen_face_residual_j_i": face_average["residual_j_i"],
        "worktree": stamp,
    }
    return classify(raw_report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("measure", "classify"),
                        default="classify")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)),
                "measurement requires deck/frame/SPG/baseline roots and commit")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None,
                    "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, GateError, check_record.Refusal,
            r97.GateError, r167.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND169_SUBSTEP1_U_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
