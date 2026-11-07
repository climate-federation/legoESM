#!/usr/bin/env python3
"""Walk the admitted GYRE kt=2 transport mean against the production JIT."""

from __future__ import annotations

import argparse
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
import nemo_testcase_l2_gyre_round14_advmean as round14  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round73_advmean_gate as record_gate  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: E402
    nemo_literal_accumulate_transport,
    nemo_literal_metric_transports,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
RECORD_NAME = "oracle_bt_advmean_operands_kt00000002.bin"
RECORD_SHA256 = "47ba91919df3a9a1c55be7fd886d620c721b2f175a737f32a05d1be2498ddf4c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def comparison(candidate, oracle, active) -> dict:
    """Return a strict float64 bit comparison on the registered wet cells."""
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


def first_live_u_boundary(seed: dict, rows: list[dict]) -> dict | None:
    """Return the first non-bit U statement in compiled source order."""
    if not seed["bit_exact"]:
        return {"substep": 0, "boundary": "zero_seed", **seed}
    order = ("sum_entry", "weight", "metric_transport", "r1_e2u", "sum_exit")
    for row in rows:
        for boundary in order:
            result = row[boundary]
            if not result["bit_exact"]:
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _full_u(values: np.ndarray) -> np.ndarray:
    """Restore legoESM's duplicated western U face to an owned NEMO field."""
    return np.concatenate((values[..., -1:], values), axis=-1)


def _full_v(values: np.ndarray) -> np.ndarray:
    """Restore legoESM's closed southern V face to an owned NEMO field."""
    return np.concatenate((np.zeros_like(values[..., :1, :]), values), axis=-2)


def _native(values, face: str) -> np.ndarray:
    return gate._trace_native(np.asarray(values), f"field_{face}")


def _shared_oracle_input_rows(fields: dict, card, seeded, active_u) -> dict:
    """Run the one shared transport statement on NEMO's recorded operands."""
    sum_u = _full_u(fields["sum_u_entry"])
    sum_v = _full_v(fields["sum_v_entry"])
    depth_u = _full_u(fields["face_depth_u"])
    depth_v = _full_v(fields["face_depth_v"])
    velocity_u = _full_u(fields["velocity_u"])
    velocity_v = _full_v(fields["velocity_v"])
    weights = np.asarray(fields["weight"], dtype=np.float64)
    u_mask = jnp.asarray(seeded.u_mask.data, dtype=jnp.float64)
    v_mask = jnp.asarray(seeded.v_mask.data, dtype=jnp.float64)
    grid = card.recipe.grid

    # Keep the actual production helper as the only numerical implementation.
    # The small wrapper only returns its already-materialized U intermediates.
    def one(su, sv, weight, hu, hv, u, v):
        metric_u, _ = nemo_literal_metric_transports(hu, hv, u, v, u_mask, v_mask, grid)
        exit_u, _ = nemo_literal_accumulate_transport(
            su, sv, weight, hu, hv, u, v, u_mask, v_mask, grid
        )
        return metric_u, exit_u

    metric_u, exit_u = jax.jit(jax.vmap(one))(
        jnp.asarray(sum_u),
        jnp.asarray(sum_v),
        jnp.asarray(weights),
        jnp.asarray(depth_u),
        jnp.asarray(depth_v),
        jnp.asarray(velocity_u),
        jnp.asarray(velocity_v),
    )
    metric_u = _native(jax.device_get(metric_u), "u")
    exit_u = _native(jax.device_get(exit_u), "u")
    metric_rows = [
        comparison(metric_u[index], fields["metric_u"][index], active_u)
        for index in range(record_gate.N_CYCLE)
    ]
    exit_rows = [
        comparison(exit_u[index], fields["sum_u_exit"][index], active_u)
        for index in range(record_gate.N_CYCLE)
    ]
    return {
        "metric_u": metric_rows,
        "sum_u_exit": exit_rows,
        "all_metric_u_bit_exact": all(row["bit_exact"] for row in metric_rows),
        "all_sum_u_exit_bit_exact": all(row["bit_exact"] for row in exit_rows),
    }


def measure(args) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64 libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-76 measurement worktree is dirty")
    require(
        stamp["commit"].lower() == args.expect_commit.lower(),
        "round-76 measurement commit mismatch",
    )

    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "round-75 admission failed")
    require(
        (
            admission["byte_identical_records"],
            len(admission["classified_changed_records"]),
            admission["admitted_difference_count"],
        )
        == (45, 23, 167),
        "round-75 admission census changed",
    )
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(
        producer.lower() == args.expect_record_commit.lower(), "round-75 producer commit mismatch"
    )
    record_path = args.record_root / RECORD_NAME
    require(record_gate.sha256(record_path) == RECORD_SHA256, "round-75 record digest changed")
    stamp_parts = record_path.with_name(RECORD_NAME + ".stamp").read_text().split()
    require(stamp_parts == [RECORD_SHA256, producer, RECORD_NAME], "round-75 record stamp mismatch")
    fields = round14.read_advmean(record_path, expected_kt=2)
    if args.plant == "exit-ulp":
        record_gate.validate_fields(fields, replay_ulp=True)
        raise AssertionError("exit ULP plant stayed green")
    replay = record_gate.validate_fields(fields)

    base, context = round72._capture_seeded_context(args)
    card, seeded, freshwater, surface, _ = context
    ordinary_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            expose_live_stage_operands=True
        ),
    )
    ordinary_model.prime_step_caches(seeded)
    ordinary_trace = jax.device_get(
        ordinary_model.step(
            seeded,
            card.dt_s,
            freshwater=freshwater,
            surface_forcing=surface,
        )
    )
    baro_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    baro_model.prime_step_caches(seeded)
    baro_trace = jax.device_get(
        baro_model.step(seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    )

    masks = gate.expected_masks(card)
    active_u = masks["u"][..., 0]
    live = {
        name: _native(baro_trace.substeps[name], "u")
        for name in (
            "transport_sum_u_entry",
            "transport_metric_u",
            "transport_velocity_u",
            "transport_face_depth_u",
            "transport_sum_u_exit",
        )
    }
    live_weight = np.asarray(baro_trace.substeps["transport_weight"])
    live_r1_e2u_full = jax.jit(lambda metric, wet: jnp.where(wet != 0, 1.0 / metric, 0.0))(
        jnp.asarray(card.recipe.grid.dy_u), jnp.asarray(seeded.u_mask.data)
    )
    live_r1_e2u = _native(jax.device_get(live_r1_e2u_full)[None, ...], "u")[0]

    oracle_metric_u = np.array(fields["metric_u"], copy=True)
    if args.plant == "null-live-zh-u":
        oracle_metric_u[0] = live["transport_metric_u"][0]
    seed_row = comparison(
        np.zeros_like(fields["sum_u_entry"][0]), fields["sum_u_entry"][0], active_u
    )
    live_rows = []
    scalar_mask = np.ones((), dtype=bool)
    for index in range(record_gate.N_CYCLE):
        live_rows.append(
            {
                "substep": index + 1,
                "sum_entry": comparison(
                    live["transport_sum_u_entry"][index], fields["sum_u_entry"][index], active_u
                ),
                "weight": comparison(
                    np.asarray(live_weight[index], dtype=np.float64),
                    np.asarray(fields["weight"][index], dtype=np.float64),
                    scalar_mask,
                ),
                "metric_transport": comparison(
                    live["transport_metric_u"][index], oracle_metric_u[index], active_u
                ),
                "r1_e2u": comparison(live_r1_e2u, fields["r1_e2u"], active_u),
                "sum_exit": comparison(
                    live["transport_sum_u_exit"][index], fields["sum_u_exit"][index], active_u
                ),
            }
        )
    first = first_live_u_boundary(seed_row, live_rows)
    operand_rows = "UNREACHED"
    if first and first["boundary"] == "metric_transport":
        index = first["substep"] - 1
        operand_rows = {
            "ua_e": comparison(
                live["transport_velocity_u"][index], fields["velocity_u"][index], active_u
            ),
            "zhup2_e": comparison(
                live["transport_face_depth_u"][index], fields["face_depth_u"][index], active_u
            ),
        }

    ordinary_u = np.asarray(ordinary_trace.barotropic_targets[2])[:, 1:]
    ordinary_v = np.asarray(ordinary_trace.barotropic_targets[3])[1:, :]
    traced_u = np.asarray(baro_trace.transport_average[0])[:, 1:]
    traced_v = np.asarray(baro_trace.transport_average[1])[1:, :]
    ordinary_identity = {
        "u": comparison(traced_u, ordinary_u, active_u),
        "v": comparison(traced_v, ordinary_v, masks["v"][..., 0]),
    }
    shared = _shared_oracle_input_rows(fields, card, seeded, active_u)
    dtype = {
        "record": str(fields["metric_u"].dtype),
        "live_metric_transport": str(live["transport_metric_u"].dtype),
        "live_accumulator": str(live["transport_sum_u_exit"].dtype),
        "grid_metric": str(np.asarray(card.recipe.grid.dy_u).dtype),
    }
    dtype_exact = all(value == "float64" for value in dtype.values())
    replay_exact = all(row["bit_exact"] for face in ("u", "v") for row in replay[face])
    replay_exact = (
        replay_exact and replay["u_normalized"]["bit_exact"] and replay["v_normalized"]["bit_exact"]
    )
    weights_exact = all(row["weight"]["bit_exact"] for row in live_rows)
    prediction_confirmed = bool(
        args.plant == "none"
        and seed_row["bit_exact"]
        and weights_exact
        and first is not None
        and first["substep"] == 1
        and first["boundary"] == "metric_transport"
        and first["differing_cells"] == first["wet_cells"] == 580
        and isinstance(operand_rows, dict)
        and not operand_rows["ua_e"]["bit_exact"]
        and replay_exact
        and shared["all_metric_u_bit_exact"]
        and shared["all_sum_u_exit_bit_exact"]
        and all(row["bit_exact"] for row in ordinary_identity.values())
        and dtype_exact
    )
    null_plant_fires = bool(
        args.plant == "null-live-zh-u"
        and (first is None or first["boundary"] != "metric_transport")
    )
    return {
        "format": "nemo-testcase-l2-gyre-round76-advmean-walk-v1",
        "status": "CONFIRMED" if prediction_confirmed else "REFUTED",
        "worktree": stamp,
        "record_producer": producer,
        "record_sha256": RECORD_SHA256,
        "admission_counts": {"exact": 45, "total": 68, "changed": 23, "admitted": 167},
        "base_round66_status": base["status"],
        "record_replay_exact": replay_exact,
        "dtype": dtype,
        "ordinary_trace_identity": ordinary_identity,
        "zero_seed": seed_row,
        "weights_all_bit_exact": weights_exact,
        "first_live_u_non_bit_statement": first,
        "first_statement_operands": operand_rows,
        "live_u_rows": live_rows,
        "shared_statement_on_oracle_inputs": shared,
        "plant": args.plant,
        "null_live_zh_u_plant_fires": null_plant_fires,
        "v_live_walk": "WITHHELD_UNTIL_U_OWNER",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path, default=ROOT / "round75/oracle_advmean_kt2")
    parser.add_argument(
        "--admission",
        type=Path,
        default=(ROOT / "round75/oracle_advmean_kt2/round75_admission.json"),
    )
    parser.add_argument("--plant", choices=("none", "exit-ulp", "null-live-zh-u"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant == "null-live-zh-u":
        verdict = "FIRED" if report["null_live_zh_u_plant_fires"] else "STAYED_GREEN"
        print(f"ROUND76 NULL_LIVE_ZHU_PLANT {verdict}")
        return 1
    first = report["first_live_u_non_bit_statement"]
    print(
        f"ROUND76 ADVMEAN {report['status']}: first={first} "
        f"shared_exact={report['shared_statement_on_oracle_inputs']['all_sum_u_exit_bit_exact']}"
    )
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
