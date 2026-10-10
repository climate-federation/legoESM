#!/usr/bin/env python3
"""Walk OMT-4's stage-3 implicit tracer solve from admitted operands."""

from __future__ import annotations

import argparse
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
    nemo_testcase_l4_orca2_round136_step36_zdf_gate as zdf_trace,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round226_fct_rhs_walk as fct_rhs,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2l_tracer_gate as phase2l,
)

ROW_ORDER = (
    "heat_K", "e3w_now", "e3t_after", "lower", "diagonal", "upper",
    "eliminated", "content_T", "forward_T", "solved_T",
)
UPSTREAM_ROWS = frozenset({
    "lower", "diagonal", "upper", "eliminated", "forward_T", "solved_T",
})
PLANTS = ("none", "record-owner", "passivity", "source-order", "overlap")


class GateError(RuntimeError):
    """The admitted implicit-solve replay cannot support its claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _comparison(reference: np.ndarray, candidate: np.ndarray,
                support: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    support = np.broadcast_to(np.asarray(support, dtype=bool), reference.shape)
    require(reference.shape == candidate.shape, "comparison shape moved")
    require(reference.dtype == candidate.dtype == np.float64,
            "comparison dtype moved")
    unequal = ((reference.view(np.uint64) != candidate.view(np.uint64))
               & support)
    delta = candidate[support] - reference[support]
    return {
        "support": int(np.count_nonzero(support)),
        "unequal": int(np.count_nonzero(unequal)),
        "maximum_absolute": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def _read_stage3(path: Path) -> dict[str, np.ndarray]:
    """Read all RKTR3 arrays, including the three recorded r3t slots."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    expected = (1, 1, 3, 1, 2, 3, 3, phase2l.NX, phase2l.NY,
                phase2l.NZ, 64)
    require(magic == "NEMO_L2_RKTR3_1", f"bad RKTR3 magic {magic!r}")
    require(header == expected, f"bad RKTR3 header {header}")
    n2 = phase2l.NX * phase2l.NY
    n3 = n2 * phase2l.NZ
    require(values.size == 16 * n3 + 3 * n2, "bad RKTR3 payload")
    require(np.isfinite(values).all(), "non-finite RKTR3 payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: phase2l._xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate(names)
    }
    offset = 16 * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        result[name] = phase2l._xy(
            values[offset + index * n2:offset + (index + 1) * n2])
    return result


def _read_avt(path: Path) -> np.ndarray:
    """Read the rank-0 interior avt from the admitted phase-2p record."""
    nx, ny, nz = phase2l.NX, phase2l.NY, phase2l.NZ
    n3 = nx * ny * nz
    ni3 = (nx - 4) * (ny - 4) * nz
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_ZDF___2", f"bad ZDF magic {magic!r}")
    require(header == (2, 1, 1, nx, ny, nz, 64),
            f"bad ZDF header {header}")
    require(values.size == n3 + 3 * ni3, "bad ZDF payload")
    avt = values[n3:n3 + ni3].reshape((nx - 4, ny - 4, nz), order="F")
    return avt.transpose(1, 0, 2)


def _first_non_bit(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in ROW_ORDER if rows[name]["unequal"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "record-owner":
        report["record_alignment"]["Kmm_unequal"] = 1
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "source-order":
        report["first_non_bit_statement"] = "lower"
    elif plant == "overlap":
        report["upstream_overlap"]["literal_path_executes_upstream_rewrite"] = True

    require(report.get("execution") == "offline-replay-plus-passive-production-trace-cpu-fp64-libm",
            "execution policy moved")
    require(report.get("in_executable_observers") == 0,
            "an in-executable observer entered the walk")
    require(all(report.get("observer_state_equal", {}).values()),
            "the passive ZDF side output moved an ordinary state leaf")
    require(report["record_alignment"]["Kmm_unequal"] == 0,
            "the rank-0 stage record no longer aligns")
    require(tuple(report.get("row_order", ())) == ROW_ORDER,
            "implicit-solve source order moved")
    first = _first_non_bit(report["candidate_vs_oracle"])
    require(first == report.get("first_non_bit_statement"),
            "first non-bit row is not source ordered")
    require(report["oracle_replay_vs_recorded_Kaa"]["unequal"] == 0,
            "source replay does not reproduce NEMO's recorded Kaa")
    overlap = report["upstream_overlap"]
    require(overlap["commit"] == "5e368e87ba",
            "upstream rewrite identity moved")
    require(not overlap["literal_path_executes_upstream_rewrite"],
            "upstream rewrite entered the literal OMT-4 execution path")
    require(overlap["first_non_bit_row_is_semantically_touched"] ==
            (first in UPSTREAM_ROWS), "upstream overlap classifier moved")
    report["predictions"] = {
        "R227-P1": "CONFIRMED",
        "R227-P2": "CONFIRMED" if first is not None else "REFUTED",
        "R227-P3": "CONFIRMED" if first in UPSTREAM_ROWS else "REFUTED",
        "R227-P4": "CONFIRMED" if first in UPSTREAM_ROWS else "REFUTED",
        "R227-P5": "UNMEASURED_WITH_SPEC" if first is not None else "NOT_REACHED",
        "R227-P6": "CONFIRMED" if plant == "none" else "PLANT",
    }
    report["status"] = "PASS_R227_IMPLICIT_TRACER_SOURCE_WALK"
    return report


def measure(deck_root: Path, frames_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
        _NEMOWSTracerZDFTrace,
    )
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        nemo_ordered_tridiagonal_solve,
        nemo_tracer_tridiagonal,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-227 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-227 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "implicit solve walk requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = omt4.rung0.assemble_frame(frames_root, 1, 0)
    state = omt4.rung0.bridge_entry(card, entry)
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    reference_depth = omt0.rung0.ladder.build_reference_depth_override(card)

    def hooks(trace: bool):
        return _NEMOWSRK3TestHooks(
            barotropic_external_mode_association=True,
            barotropic_reference_face_depth_override=reference_depth,
            barotropic_unmasked_v_transport=True,
            barotropic_materialize_v_transport=True,
            barotropic_atomic_fold_unit=True,
            tracer_zdf_trace=trace,
        )

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks(False))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks(True))
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSTracerZDFTrace),
            "ZDF side output has the wrong type")

    stage = _read_stage3(
        frames_root / "oracle_rktracer_stage3_kt00000001.bin")
    avt = _read_avt(frames_root / "oracle_zdf_entry_kt00000001.bin")
    stage2_frame = omt4.rung0.assemble_frame(frames_root, 1, 2)
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)[:, :90, :]
    support = active
    tmask2 = active[..., 0]
    h0 = np.asarray(card.recipe.z_coord.h_partial, dtype=np.float64)[:, :90, :]
    rbb = stage["r3t_Kbb"]
    rmm = stage["r3t_Kmm"]
    raa = stage["r3t_Kaa"]
    e3t_bb = h0 * (1.0 + rbb[..., None] * active)
    e3t_mm = h0 * (1.0 + rmm[..., None] * active)
    e3t_aa = h0 * (1.0 + raa[..., None] * active)
    e3w_ref = np.asarray(card.recipe.z_coord.dz_half_ref, dtype=np.float64)
    e3w = np.broadcast_to(e3w_ref, active.shape[:-1] + e3w_ref.shape)
    e3w = e3w * (1.0 + rmm[..., None])
    heat_K = avt[..., 1:-1]
    tbb = stage["Kbb_T"][..., :-1]
    rhs = stage["after_ldf_T"][..., :-1]
    left = e3t_bb * tbb
    right = (np.float64(card.dt_s) * e3t_mm) * rhs
    content = left + right

    replay = jax.jit(lambda k, et, ew, wet, source: (
        *nemo_tracer_tridiagonal(k, et, ew, card.dt_s, wet),
        nemo_ordered_tridiagonal_solve(
            *nemo_tracer_tridiagonal(k, et, ew, card.dt_s, wet), source,
            return_trace=True),
    ))(jnp.asarray(heat_K), jnp.asarray(e3t_aa), jnp.asarray(e3w),
       jnp.asarray(active), jnp.asarray(content))
    lower, diagonal, upper, solve_trace = replay
    solution, (eliminated, forward) = solve_trace
    oracle = {
        "heat_K": heat_K,
        "e3w_now": e3w,
        "e3t_after": e3t_aa,
        "lower": np.asarray(lower),
        "diagonal": np.asarray(diagonal),
        "upper": np.asarray(upper),
        "eliminated": np.asarray(eliminated),
        "content_T": content,
        "forward_T": np.asarray(forward),
        "solved_T": np.asarray(solution),
    }
    solve = traced.solve
    candidate = {
        "heat_K": np.asarray(solve.heat_K)[:, :90, :],
        "e3w_now": np.asarray(solve.e3w_now)[:, :90, :],
        "e3t_after": np.asarray(solve.e3t_after)[:, :90, :],
        "lower": np.asarray(solve.lower)[:, :90, :],
        "diagonal": np.asarray(solve.diagonal)[:, :90, :],
        "upper": np.asarray(solve.upper)[:, :90, :],
        "eliminated": np.asarray(solve.eliminated_T)[:, :90, :],
        "content_T": np.asarray(traced.content_T)[:, :90, :],
        "forward_T": np.asarray(solve.forward_T)[:, :90, :],
        "solved_T": np.asarray(solve.solved_T)[:, :90, :],
    }
    row_support = {
        name: (active[..., :-1] & active[..., 1:])
        if name in ("heat_K", "e3w_now") else support
        for name in ROW_ORDER
    }
    rows = {
        name: _comparison(oracle[name], candidate[name], row_support[name])
        for name in ROW_ORDER
    }
    first = _first_non_bit(rows)
    recorded_kaa = stage["Kaa_T"][..., :-1]
    record_kmm = stage["Kmm_T"][..., :-1]

    return {
        "format": "nemo-testcase-l4-orca2-round227-implicit-tracer-walk-v1",
        "execution": "offline-replay-plus-passive-production-trace-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 stage-3 rank-0 RKTR3 and ZDF records",
        "claim_label": "given NEMO's entry",
        "in_executable_observers": 0,
        "observer_state_equal": zdf_trace._state_equal(traced.state_after, ordinary),
        "record_alignment": {
            "rank": 0,
            "owned_global_i": [0, 89],
            "Kmm_unequal": _comparison(
                record_kmm, np.asarray(stage2_frame["T"])[:, :90, :],
                support)["unequal"],
            "recorded_avt_shape": list(avt.shape),
        },
        "row_order": list(ROW_ORDER),
        "candidate_vs_oracle": rows,
        "first_non_bit_statement": first,
        "oracle_replay_vs_recorded_Kaa": _comparison(
            recorded_kaa, oracle["solved_T"], support),
        "upstream_overlap": {
            "commit": "5e368e87ba",
            "literal_path_executes_upstream_rewrite": False,
            "first_non_bit_row_is_semantically_touched": first in UPSTREAM_ROWS,
            "reason": (
                "OMT-4 resolves zdf_implicit_solver_evaluation=nemo_literal; "
                "5e368e87ba rewrites the shared/generic Thomas path only"
            ),
        },
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
        "worktree": stamp,
        "compiled_citations": {
            "coefficient": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:171-216",
            "matrix": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:218-235",
            "elimination": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:249-273",
            "content_forward": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:283-291",
            "backsolve": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:293-299",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.deck_root, args.frames_root, args.expect_commit)),
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root and args.frames_root and args.expect_commit,
                    "runtime mode requires deck, frames, and commit")
            raw = measure(args.deck_root, args.frames_root, args.expect_commit)
        if args.raw_output:
            args.raw_output.write_text(
                json.dumps(raw, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt4.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
