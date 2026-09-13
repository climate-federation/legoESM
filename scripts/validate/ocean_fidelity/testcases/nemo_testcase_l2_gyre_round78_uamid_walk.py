#!/usr/bin/env python3
"""Walk the admitted GYRE kt=2 U midpoint in compiled source order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round76_uamid_gate as record_gate  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
RECORD_SHA256 = "efff2a6ab74770890221790ac7b3c7d01d07bbf9ee3a520fdad1d99041be8a0b"
PREDICTED_MAX = 2.117582368135751e-22
SOURCE_ORDER = (
    "coefficient_1",
    "coefficient_2",
    "coefficient_3",
    "un_e",
    "ub_e",
    "ubb_e",
    "ua_e",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def comparison(candidate, oracle, active) -> dict:
    """Return a strict float64 bit comparison on registered wet cells."""
    candidate = np.asarray(candidate)
    oracle = np.asarray(oracle)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape, "comparison shape mismatch")
    require(candidate.dtype == oracle.dtype == np.float64, "comparison is not float64")
    require(np.any(active), "comparison mask is empty")
    require(np.all(np.isfinite(candidate[active])), "candidate has non-finite wet cells")
    require(np.all(np.isfinite(oracle[active])), "oracle has non-finite wet cells")
    candidate_bits = candidate[active].view(np.uint64)
    oracle_bits = oracle[active].view(np.uint64)
    delta = candidate[active] - oracle[active]
    return {
        "bit_exact": bool(np.array_equal(candidate_bits, oracle_bits)),
        "differing_cells": int(np.count_nonzero(candidate_bits != oracle_bits)),
        "wet_cells": int(candidate_bits.size),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "reference_max_abs": float(np.max(np.abs(oracle[active]), initial=0.0)),
    }


def first_live_boundary(rows: list[dict]) -> dict | None:
    """Return the first non-bit U row in compiled substep/source order."""
    for row in rows:
        for boundary in SOURCE_ORDER:
            result = row[boundary]
            if not result["bit_exact"]:
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value) -> dict:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def _trace_digests(trace) -> list[dict]:
    return [_array_digest(value) for value in jax.tree_util.tree_leaves(trace)]


def _context(args):
    base, context = round72._capture_seeded_context(args)
    card, seeded, freshwater, surface, _ = context
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True
        ),
    )
    model.prime_step_caches(seeded)
    trace = jax.device_get(
        model.step(seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    )
    return base, card, seeded, trace


def _setup(args):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64 libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-78 measurement worktree is dirty")
    require(
        stamp["commit"].lower() == args.expect_commit.lower(),
        "round-78 measurement commit mismatch",
    )
    return stamp


def capture(args) -> dict:
    stamp = _setup(args)
    base, _card, _seeded, trace = _context(args)
    return {
        "format": "nemo-testcase-l2-gyre-round78-trace-digest-v1",
        "worktree": stamp,
        "base_round66_status": base["status"],
        "leaf_digests": _trace_digests(trace),
    }


def measure(args) -> dict:
    stamp = _setup(args)
    baseline = json.loads(args.baseline.read_text())
    require(
        baseline["format"] == "nemo-testcase-l2-gyre-round78-trace-digest-v1",
        "wrong pre-refactor baseline format",
    )

    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "round-77 inherited admission admission failed")
    require(
        (
            admission["byte_identical_records"],
            len(admission["classified_changed_records"]),
            admission["admitted_difference_count"],
        )
        == (45, 24, 281),
        "round-77 inherited-record census changed",
    )
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(
        producer.lower() == args.expect_record_commit.lower(),
        "round-77 producer commit mismatch",
    )
    record_path = args.record_root / record_gate.RECORD
    require(_sha256(record_path) == RECORD_SHA256, "round-77 U-midpoint digest changed")
    require(
        record_path.with_name(record_gate.RECORD + ".stamp").read_text().split()
        == [RECORD_SHA256, producer, record_gate.RECORD],
        "round-77 U-midpoint stamp mismatch",
    )
    fields = record_gate.read_record(record_path)
    replay = record_gate.validate_fields(fields)

    base, card, _seeded, trace = _context(args)
    candidate_digests = _trace_digests(trace)
    extraction_identity = candidate_digests == baseline["leaf_digests"]
    require(extraction_identity, "shared-helper extraction moved a production trace bit")

    masks = gate.expected_masks(card)
    active_u = masks["u"][..., 0]
    substeps = trace.substeps
    live = {
        "un_e": gate._trace_native(substeps["u_entry"], "u_entry"),
        "ub_e": gate._trace_native(substeps["u_history_b"], "u_history_b"),
        "ubb_e": gate._trace_native(substeps["u_history_bb"], "u_history_bb"),
        "ua_e": gate._trace_native(substeps["u_mid"], "u_mid"),
    }
    live_coefficients = np.stack(
        [
            np.asarray(substeps[f"mid_weight_{index}"], dtype=np.float64)
            for index in (1, 2, 3)
        ],
        axis=-1,
    )
    oracle_live = {name: np.array(fields[name], copy=True) for name in record_gate.FIELDS}
    if args.plant == "null-live-un-e":
        un_e_baseline = comparison(live["un_e"][0], fields["un_e"][0], active_u)
        require(
            not un_e_baseline["bit_exact"],
            "null-live-un-e plant target is already exact",
        )
        oracle_live["un_e"][0] = live["un_e"][0]
    elif args.plant == "null-live-ubb-e":
        ubb_e_baseline = comparison(live["ubb_e"][0], fields["ubb_e"][0], active_u)
        require(
            not ubb_e_baseline["bit_exact"],
            "null-live-ubb-e plant target is already exact",
        )
        oracle_live["ubb_e"][0] = live["ubb_e"][0]

    scalar_mask = np.ones((), dtype=bool)
    rows = []
    for index in range(record_gate.N_CYCLE):
        rows.append(
            {
                "substep": index + 1,
                "coefficient_1": comparison(
                    live_coefficients[index, 0], fields["coefficients"][index, 0], scalar_mask
                ),
                "coefficient_2": comparison(
                    live_coefficients[index, 1], fields["coefficients"][index, 1], scalar_mask
                ),
                "coefficient_3": comparison(
                    live_coefficients[index, 2], fields["coefficients"][index, 2], scalar_mask
                ),
                **{
                    name: comparison(live[name][index], oracle_live[name][index], active_u)
                    for name in record_gate.FIELDS
                },
            }
        )
    first = first_live_boundary(rows)

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: PLC0415
        nemo_literal_midpoint_extrapolation,
    )

    shared = jax.jit(
        jax.vmap(nemo_literal_midpoint_extrapolation, in_axes=(0, 0, 0, 0))
    )(
        jnp.asarray(fields["coefficients"]),
        jnp.asarray(fields["un_e"]),
        jnp.asarray(fields["ub_e"]),
        jnp.asarray(fields["ubb_e"]),
    )
    shared = np.asarray(jax.device_get(shared))
    shared_target = np.array(fields["ua_e"], copy=True)
    if args.plant == "shared-result-ulp":
        active_target = np.broadcast_to(active_u, shared_target.shape)
        locations = np.argwhere(active_target & (shared_target != 0.0))
        require(locations.size > 0, "shared-result plant requires a nonzero wet target")
        location = tuple(locations[0])
        shared_target[location] = np.nextafter(shared_target[location], np.inf)
    shared_rows = [
        comparison(shared[index], shared_target[index], active_u)
        for index in range(record_gate.N_CYCLE)
    ]
    shared_exact = all(row["bit_exact"] for row in shared_rows)

    dtype = {
        "record": str(fields["ua_e"].dtype),
        "live": str(live["ua_e"].dtype),
        "shared": str(shared.dtype),
    }
    dtype_exact = all(value == "float64" for value in dtype.values())
    prediction_confirmed = bool(
        args.plant == "none"
        and first is not None
        and first["substep"] == 1
        and first["boundary"] == "un_e"
        and first["differing_cells"] == 2
        and first["wet_cells"] == 580
        and first["absolute_max"] == PREDICTED_MAX
        and shared_exact
        and all(row["bit_exact"] for row in replay)
        and dtype_exact
        and base["status"] == "CONFIRMED"
    )
    null_plant_fires = bool(
        args.plant == "null-live-un-e"
        and (first is None or first["substep"] != 1 or first["boundary"] != "un_e")
    )
    ubb_plant_fires = bool(
        args.plant == "null-live-ubb-e"
        and (first is None or first["substep"] != 1 or first["boundary"] != "ubb_e")
    )
    shared_plant_fires = bool(args.plant == "shared-result-ulp" and not shared_exact)
    return {
        "format": "nemo-testcase-l2-gyre-round78-uamid-walk-v1",
        "status": "CONFIRMED" if prediction_confirmed else "REFUTED",
        "worktree": stamp,
        "record_producer": producer,
        "record_sha256": RECORD_SHA256,
        "record_replay_exact": all(row["bit_exact"] for row in replay),
        "admission_counts": {"exact": 45, "total": 69, "changed": 24, "admitted": 281},
        "base_round66_status": base["status"],
        "pre_refactor_commit": baseline["worktree"]["commit"],
        "shared_helper_extraction_bit_exact": extraction_identity,
        "dtype": dtype,
        "first_live_u_non_bit_statement": first,
        "live_u_rows": rows,
        "shared_statement_on_oracle_inputs": {
            "all_bit_exact": shared_exact,
            "rows": shared_rows,
        },
        "plant": args.plant,
        "null_live_un_e_plant_fires": null_plant_fires,
        "null_live_ubb_e_plant_fires": ubb_plant_fires,
        "shared_result_ulp_plant_fires": shared_plant_fires,
        "v_live_walk": "WITHHELD_UNTIL_U_OWNER",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path, default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument(
        "--admission",
        type=Path,
        default=ROOT / "round77/oracle_uamid_kt2/round77_admission.json",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--plant",
        choices=("none", "null-live-un-e", "null-live-ubb-e", "shared-result-ulp"),
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = capture(args) if args.capture else measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.capture:
        print("ROUND78 PRE-REFACTOR TRACE CAPTURED")
        return 0
    if args.plant != "none":
        fires = {
            "null-live-un-e": report["null_live_un_e_plant_fires"],
            "null-live-ubb-e": report["null_live_ubb_e_plant_fires"],
            "shared-result-ulp": report["shared_result_ulp_plant_fires"],
        }[args.plant]
        print(f"ROUND78 {args.plant.upper()} PLANT {'FIRED' if fires else 'STAYED_GREEN'}")
        return 1
    print(
        f"ROUND78 UAMID {report['status']}: "
        f"first={report['first_live_u_non_bit_statement']} "
        f"shared_exact={report['shared_statement_on_oracle_inputs']['all_bit_exact']}"
    )
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
