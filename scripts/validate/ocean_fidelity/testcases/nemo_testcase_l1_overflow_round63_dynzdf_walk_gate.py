#!/usr/bin/env python3
"""Walk OVERFLOW's four stage-3 dyn_zdf boundaries under the held UP3 arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
import nemo_testcase_l1_overflow_round51_pair_gate as R51
import nemo_testcase_l1_overflow_round60_downstream_walk_gate as R60
import nemo_testcase_l1_overflow_round62_dynzdf_gate as R62


FORMAT = "nemo-testcase-l1-overflow-round63-dynzdf-walk-v1"
GATE_REVISION = "rhs-distinct-v2"
BOUNDARIES = ("explicit", "baro_subtract", "baro_drag", "implicit_solve")
COMPONENTS = ("u", "v")
SOURCE_ORDER = tuple(
    f"s3.zdf.{boundary}.{component}"
    for boundary in BOUNDARIES for component in COMPONENTS
)
RECORD_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round62/acquisition/oracle_overflow_dynzdf_internals"
)
RECORD = RECORD_ROOT / "oracle_r62_dynzdf_kt00000003_s3.bin"
PARENT = RECORD_ROOT / "oracle_r50_momentum_kt00000003_s3.bin"
PRODUCER_FILE = RECORD_ROOT / "producer_commit.txt"
PRODUCER_COMMIT = "6ac4ecd1b8d76eb35b48888b28892fdec78aa4e8"


def _physical(value: np.ndarray, component: str, mask: np.ndarray) -> np.ndarray:
    mapped = R51._u(value) if component == "u" else R51._v(value)
    return R51._physical_levels(mapped, mask)


def _state_arrays(state) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value.data)
        for name, value in zip(state._fields, state, strict=True)
        if value is not None and hasattr(value, "data")
    }


def _observer(card, state) -> tuple[object, dict[str, np.ndarray]]:
    calls: list[tuple[np.ndarray, ...]] = []

    def receive(*values) -> None:
        calls.append(tuple(np.asarray(value) for value in values))

    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            zdf_momentum_observer=receive),
    )
    observed = model.step(state, dt=card.dt_s)
    _state_arrays(observed)
    jax.effects_barrier()
    R60.require(len(calls) == 1,
                f"dyn_zdf observer call count {len(calls)} != 1")
    values = calls[0]
    R60.require(len(values) == len(R62.FIELDS),
                f"dyn_zdf observer field count {len(values)} != 8")
    trace = dict(zip(R62.FIELDS, values, strict=True))
    return observed, trace


def _noninterference(before: dict[str, np.ndarray], observed) -> list[dict]:
    after = _state_arrays(observed)
    R60.require(before.keys() == after.keys(), "observer state-field drift")
    rows = []
    for name in before:
        equal = np.array_equal(before[name], after[name])
        rows.append({
            "name": name,
            "shape": list(before[name].shape),
            "dtype": str(before[name].dtype),
            "exact": bool(equal),
        })
        R60.require(equal, f"dyn_zdf observer perturbed state field {name}")
    return rows


def _write_sidecar(output: Path, arrays: dict[str, np.ndarray]) -> dict:
    return R60._write_sidecar(output, arrays)


def _read_sidecar(report: dict) -> dict[str, np.ndarray]:
    return R60._read_sidecar(report)


def compare(reference_path: Path, candidate_report: dict) -> dict:
    reference_report = json.loads(reference_path.read_text())
    R60.require(reference_report["format"] == FORMAT,
                "round-63 reference format drift")
    base = _read_sidecar(reference_report)
    candidate = _read_sidecar(candidate_report)
    R60.require(base.keys() == candidate.keys(), "round-63 sidecar key drift")
    rows = []
    for boundary in BOUNDARIES:
        name = f"s3.zdf.{boundary}.u"
        rows.append({
            "name": name,
            **R60.classify_direction(
                base[name], candidate[name], base[f"oracle::{name}"]),
        })
    first_moved = next((row for row in rows if row["n_moved"]), None)
    initial = first_moved["direction"] if first_moved is not None else "UNCHANGED"
    change = next((
        row for row in rows
        if row["n_moved"] and row["direction"] != initial
    ), None)
    opposite = {"TOWARD": "AWAY", "AWAY": "TOWARD"}.get(initial)
    reversal = next((
        row for row in rows
        if row["n_moved"] and row["direction"] == opposite
    ), None)
    return {
        "status": "MEASURED",
        "reference": str(reference_path),
        "reference_commit": reference_report["worktree"]["commit"],
        "candidate_commit": candidate_report["worktree"]["commit"],
        "first_moved_boundary": first_moved,
        "initial_direction": initial,
        "first_direction_change": change,
        "first_direction_reversal": reversal,
        "compensating_owner": None if reversal is None else reversal["name"],
        "rows": rows,
    }


def run(output: Path, expect_commit: str, entry_input: Path,
        reference: Path | None, plant: bool) -> dict:
    print(f"ROUND63_GATE_REVISION {GATE_REVISION}", file=sys.stderr,
          flush=True)
    stamp = worktree_stamp()
    R60.require(stamp["clean"], "producer worktree is dirty")
    R60.require(stamp["commit"] == expect_commit,
                f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    R60.require(jax.default_backend() == "cpu", "round-63 gate is CPU-only")
    R60.require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    R60.require(not bool(jax.config.jax_disable_jit),
                "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    R60.require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
                "precision policy is not fp64/libm")

    R60.require(PRODUCER_FILE.read_text().strip() == PRODUCER_COMMIT,
                "round-62 producer commit drift")
    admission = R62.admit(RECORD, PARENT, PRODUCER_COMMIT, None)
    R60.require(admission["status"] == "AT_BAR",
                "round-62 dyn_zdf record is not admitted")
    oracle_record = R62.read_record(RECORD)
    acquired_parent = R62.r50_gate.read_record(PARENT, "momentum", 3)
    historical_parent = R51._record_pair(R60.ORACLE_ROOT)[0][3]

    card = build_nemo_testcase_card("OVERFLOW-zps")
    cfg = card.recipe.model_config
    R60.require(
        (cfg.outer_integrator, cfg.momentum_time_integrator,
         cfg.momentum_advection, cfg.momentum_flux_scheme, cfg.pgf_scheme)
        == ("forward_euler", "rk3_ws", "flux_form", "nemo_up3", "nemo_sco"),
        "resolved OVERFLOW program drift")
    R60.require(str(card.recipe.initial_state.u.data.dtype) == "float64",
                "state dtype is not float64")
    state = R60._read_entry_state(entry_input, card.recipe.initial_state)
    masks = R60.expected_masks(card)

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    ordinary = ordinary_model.step(state, dt=card.dt_s)
    ordinary_arrays = _state_arrays(ordinary)
    del ordinary, ordinary_model
    jax.clear_caches()
    observed, trace = _observer(card, state)
    observer_rows = _noninterference(ordinary_arrays, observed)

    arrays: dict[str, np.ndarray] = {}
    rows = []
    for boundary in BOUNDARIES:
        for component in COMPONENTS:
            name = f"s3.zdf.{boundary}.{component}"
            mask = np.asarray(masks[component], dtype=bool)
            oracle = R51._physical_levels(
                R51._nemo_owned(oracle_record["fields"][f"{boundary}_{component}"]),
                mask,
            )
            candidate = _physical(trace[f"{boundary}_{component}"], component, mask)
            row = R51._score(name, oracle, candidate, mask, plant=False)
            rows.append(row)
            arrays[f"oracle::{name}"] = R60._active(oracle, mask)
            arrays[name] = R60._active(candidate, mask)

    oracle_explicit_u = oracle_record["fields"]["explicit_u"]
    oracle_implicit_u = oracle_record["fields"]["implicit_solve_u"]
    R60.require(
        np.array_equal(oracle_implicit_u,
                       acquired_parent["fields"]["raw_kaa_u"]),
        "oracle implicit U differs from same-build parent raw-Kaa U")
    parent_pre_zdf_u = acquired_parent["fields"]["pre_zdf_u"]
    cross_build = {
        "same_build_rhs_pre_zdf_vs_explicit_kaa": {
            "n": int(oracle_explicit_u.size),
            "n_unequal": int(np.count_nonzero(
                oracle_explicit_u.view(np.uint64)
                != parent_pre_zdf_u.view(np.uint64))),
            "absolute_max": float(np.max(np.abs(
                oracle_explicit_u - parent_pre_zdf_u))),
            "disposition": "distinct compiled statements, not endpoints",
        },
    }
    for name in ("pre_zdf_u", "raw_kaa_u"):
        current = acquired_parent["fields"][name]
        historical = historical_parent[name]
        R60.require(current.shape == historical.shape,
                    f"cross-build {name} shape drift")
        cross_build[name] = {
            "n": int(current.size),
            "n_unequal": int(np.count_nonzero(
                current.view(np.uint64) != historical.view(np.uint64))),
            "absolute_max": float(np.max(np.abs(current - historical))),
        }

    planted = None
    if plant:
        name = "s3.zdf.explicit.u"
        candidate = arrays[name].copy()
        oracle = arrays[f"oracle::{name}"]
        clean = candidate.view(np.uint64) == oracle.view(np.uint64)
        R60.require(bool(np.any(clean)), "no exact active U cell for plant")
        index = int(np.flatnonzero(clean)[0])
        before = int(np.count_nonzero(~clean))
        candidate[index] = np.nextafter(candidate[index], np.float64(np.inf))
        after = int(np.count_nonzero(
            candidate.view(np.uint64) != oracle.view(np.uint64)))
        R60.require(after == before + 1, "plant did not add one refusal")
        arrays[name] = candidate
        planted = {
            "name": name, "active_flat_index": index,
            "before_unequal": before, "after_unequal": after,
        }

    report = {
        "format": FORMAT,
        "gate_revision": GATE_REVISION,
        "status": "PLANTED_REFUSAL" if plant else "WALK_MEASURED",
        "case": "OVERFLOW-zps",
        "kt": 3,
        "stage": 3,
        "claim_label": "independent",
        "precision": "cpu-fp64-libm-production-jit",
        "worktree": stamp,
        "record_admission": {
            "status": admission["status"],
            "producer_commit": admission["producer_commit"],
            "endpoint_identity": admission["endpoint_identity"],
        },
        "cross_build_information": cross_build,
        "compiled_source_order": list(SOURCE_ORDER),
        "controlled_entry": {
            "path": str(entry_input), "sha256": R60._sha256(entry_input)},
        "rows": rows,
        "observer_noninterference": observer_rows,
        "plant": planted,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report["sidecar"] = _write_sidecar(output, arrays)
    if reference is not None:
        report["comparison"] = compare(reference, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.output, args.expect_commit, args.entry_input,
            args.reference, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant else 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        rendered = json.dumps(
            {"status": "REFUSE", "reason": str(error)}, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
