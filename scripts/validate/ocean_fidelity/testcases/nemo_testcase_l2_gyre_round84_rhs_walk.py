#!/usr/bin/env python3
"""Walk GYRE kt=2 stage-1 momentum RHS boundaries in compiled order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
BOUNDARIES = ("after_hpg", "after_ldf", "after_vor", "after_keg",
              "after_zad", "after_adv")
INHERITED_FINAL_MAX = {
    "u": np.float64(1.9220276136603893e-09),
    "v": np.float64(1.966059508480186e-09),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_order_accumulators(
    hpg_u, hpg_v, ldf_u, ldf_v, vor_u, vor_v,
    keg_u, keg_v, zad_u, zad_v,
):
    """Accumulate already-computed terms in compiled ``stp2d`` order."""
    hpg_u = jax.lax.optimization_barrier(hpg_u)
    hpg_v = jax.lax.optimization_barrier(hpg_v)
    ldf_u = jax.lax.optimization_barrier(hpg_u + ldf_u)
    ldf_v = jax.lax.optimization_barrier(hpg_v + ldf_v)
    vor_u = jax.lax.optimization_barrier(ldf_u + vor_u)
    vor_v = jax.lax.optimization_barrier(ldf_v + vor_v)
    keg_u = jax.lax.optimization_barrier(vor_u + keg_u)
    keg_v = jax.lax.optimization_barrier(vor_v + keg_v)
    zad_u = jax.lax.optimization_barrier(keg_u + zad_u)
    zad_v = jax.lax.optimization_barrier(keg_v + zad_v)
    return {
        "after_hpg_u": hpg_u,
        "after_hpg_v": hpg_v,
        "after_ldf_u": ldf_u,
        "after_ldf_v": ldf_v,
        "after_vor_u": vor_u,
        "after_vor_v": vor_v,
        "after_keg_u": keg_u,
        "after_keg_v": keg_v,
        "after_zad_u": zad_u,
        "after_zad_v": zad_v,
        # The compiled write-only after_adv snapshot immediately follows ZAD.
        "after_adv_u": zad_u,
        "after_adv_v": zad_v,
    }


def first_nonbit(rows: dict[str, dict[str, dict]]) -> dict | None:
    """Return the first non-bit accumulator in compiled boundary order."""
    for boundary in BOUNDARIES:
        for face in ("u", "v"):
            if not rows[face][boundary]["bit_exact"]:
                return {"boundary": boundary, "face": face,
                        **rows[face][boundary]}
    return None


def _away_one_ulp(reference: np.ndarray, candidate: np.ndarray,
                  active: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    delta = np.where(active, np.abs(candidate - reference), -np.inf)
    at = np.unravel_index(np.argmax(delta), delta.shape)
    changed = np.array(reference, copy=True)
    direction = -np.inf if candidate[at] >= reference[at] else np.inf
    changed[at] = np.nextafter(changed[at], direction)
    return changed, tuple(map(int, at))


def _native(value, face: str) -> np.ndarray:
    return round83.native_u(value) if face == "u" else round83.native_v(value)


def _capture_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output,
    )


def _row(candidate, oracle, active) -> dict[str, object]:
    row = round83.comparison(candidate, oracle, active)
    denominator = row["reference_max_abs"]
    row["normalized_max"] = (
        row["absolute_max"] / denominator if denominator else
        (0.0 if row["absolute_max"] == 0.0 else float("inf")))
    return row


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-84 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-84 commit stamp mismatch")

    stage, _ = round83._admit_round64(args)
    arrays = stage["arrays"]
    base, context = round72._capture_seeded_context(_capture_args(args))
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    card, seeded, freshwater, surface, trace = context
    require(card.recipe.model_config.momentum_advection == "vector_invariant",
            "GYRE no longer runs vector-invariant momentum")
    parts = trace.operator_operands[0]

    accumulated = jax.device_get(jax.jit(source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))

    baro_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True),
    )
    baro_model.prime_step_caches(seeded)
    baro_trace = jax.device_get(baro_model.step(
        seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    ordinary_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary_model.prime_step_caches(seeded)
    ordinary = jax.device_get(ordinary_model.step(
        seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    masks = gate.expected_masks(card)
    active = {
        "u": np.asarray(masks["u"], dtype=bool),
        "v": np.asarray(masks["v"], dtype=bool),
        "t": np.asarray(masks["T"], dtype=bool),
    }
    oracle = {
        face: {
            boundary: round83.owned3(arrays[f"{boundary}_{face}"])
            for boundary in BOUNDARIES
        }
        for face in ("u", "v")
    }
    live = {
        face: {
            boundary: _native(accumulated[f"{boundary}_{face}"], face)
            for boundary in BOUNDARIES
        }
        for face in ("u", "v")
    }
    live_total = {
        "u": round83.native_u(baro_trace.slow_forcing_operands["du_dt"]),
        "v": round83.native_v(baro_trace.slow_forcing_operands["dv_dt"]),
    }

    density_oracle = round83.owned3(arrays["rhd_in"])
    density_live = np.asarray(parts["operand_rho_prime"], dtype=np.float64) / np.float64(
        card.recipe.model_config.rho_0)
    ordinary_controls = {
        "hpg-ulp": _row(live["u"]["after_hpg"],
                        oracle["u"]["after_hpg"], active["u"]),
        "density-ulp": _row(density_live, density_oracle, active["t"]),
        "closure-ulp": _row(live["u"]["after_adv"], live_total["u"], active["u"]),
    }
    plant_detail = None
    if args.plant == "hpg-ulp":
        oracle["u"]["after_hpg"], at = _away_one_ulp(
            oracle["u"]["after_hpg"], live["u"]["after_hpg"], active["u"])
        plant_detail = {"field": "after_hpg_u", "location": list(at)}
    elif args.plant == "density-ulp":
        density_oracle, at = _away_one_ulp(
            density_oracle, density_live, active["t"])
        plant_detail = {"field": "rhd_in", "location": list(at)}
    elif args.plant == "closure-ulp":
        live_total["u"], at = _away_one_ulp(
            live_total["u"], live["u"]["after_adv"], active["u"])
        plant_detail = {"field": "live_total_u", "location": list(at)}

    rows = {
        face: {
            boundary: _row(live[face][boundary], oracle[face][boundary], active[face])
            for boundary in BOUNDARIES
        }
        for face in ("u", "v")
    }
    closure = {
        face: _row(live[face]["after_adv"], live_total[face], active[face])
        for face in ("u", "v")
    }
    after_adv_identity = {
        face: _row(oracle[face]["after_adv"], oracle[face]["after_zad"], active[face])
        for face in ("u", "v")
    }

    increments = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        previous = np.zeros_like(live[face]["after_hpg"])
        for boundary in BOUNDARIES:
            residual = live[face][boundary] - oracle[face][boundary]
            delta = residual - previous
            increments[face][boundary] = float(
                np.max(np.abs(delta[active[face]]), initial=0.0))
            previous = residual

    state_u, state_v, state_T, state_S, state_ssh = trace.stage_states[0]
    r3t, _, _ = trace.stage_qco[0]
    h_ref = np.asarray(compute_layer_thickness(
        jnp.zeros_like(seeded.eta.data), seeded.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m),
        dtype=np.float64)
    hpg_inputs = {
        "T": _row(state_T, round83.owned3(arrays["T_Kmm"]), active["t"]),
        "S": _row(state_S, round83.owned3(arrays["S_Kmm"]), active["t"]),
        "ssh": _row(state_ssh, round83.owned2(arrays["ssh_Kmm"]), active["t"][..., 0]),
        "r3t": _row(r3t, round83.owned2(arrays["r3t_Kmm"]), active["t"][..., 0]),
        "density": _row(density_live, density_oracle, active["t"]),
        "live_thickness": _row(parts["operand_h_k"],
                               round83.owned3(arrays["e3t_Kmm"]), active["t"]),
        "reference_thickness": _row(h_ref, round83.owned3(arrays["e3t_0"]), active["t"]),
        "reciprocal_metric_u": _row(
            round83.native_u(np.float64(1.0) / np.asarray(card.recipe.grid.dx_u)),
            round83.owned2(arrays["r1_e1u"]), active["u"][..., 0]),
        "reciprocal_metric_v": _row(
            round83.native_v(np.float64(1.0) / np.asarray(card.recipe.grid.dy_v)),
            round83.owned2(arrays["r1_e2v"]), active["v"][..., 0]),
    }
    handed_velocity = {
        "u": _row(round83.native_u(state_u), round83.owned3(arrays["u_Kmm"]), active["u"]),
        "v": _row(round83.native_v(state_v), round83.owned3(arrays["v_Kmm"]), active["v"]),
        "classification": "handed stage member; not consumed by dyn_hpg",
    }

    ordinary_fields = gate.lego_fields(ordinary)
    traced_fields = gate.lego_fields(trace.state_after)
    trace_noninterference = {
        name: _row(traced_fields[name], ordinary_fields[name], masks[name])
        for name in ("T", "S", "u", "v", "ssh")
    }

    first = first_nonbit(rows)
    magnitude = {
        face: {
            "threshold": float(np.float64(0.1) * INHERITED_FINAL_MAX[face]),
            "after_hpg_reaches_threshold": bool(
                rows[face]["after_hpg"]["absolute_max"] >=
                np.float64(0.1) * INHERITED_FINAL_MAX[face]),
            "largest_increment_boundary": max(
                BOUNDARIES, key=lambda name: increments[face][name]),
            "largest_increment_absolute_max": max(increments[face].values()),
        }
        for face in ("u", "v")
    }
    density_nonbit = not hpg_inputs["density"]["bit_exact"]
    confirmed = bool(
        args.plant is None
        and all(row["bit_exact"] for row in closure.values())
        and all(row["bit_exact"] for row in trace_noninterference.values())
        and all(row["bit_exact"] for row in after_adv_identity.values())
        and first is not None and first["boundary"] == "after_hpg"
        and any(row["after_hpg_reaches_threshold"] for row in magnitude.values())
        and all(row["largest_increment_boundary"] == "after_hpg"
                for row in magnitude.values())
        and density_nonbit
    )

    plant_fired = None
    if args.plant is not None:
        target = {
            "hpg-ulp": rows["u"]["after_hpg"],
            "density-ulp": hpg_inputs["density"],
            "closure-ulp": closure["u"],
        }[args.plant]
        before = ordinary_controls[args.plant]
        plant_fired = bool(
            target["absolute_max"] != before["absolute_max"]
            and target["differing_cells"] >= before["differing_cells"])
        require(plant_fired, f"{args.plant} differential control was invisible")

    return {
        "format": "nemo-testcase-l2-gyre-round84-rhs-walk-v1",
        "status": "PLANT_FIRED" if args.plant else (
            "CONFIRMED" if confirmed else "REFUTED"),
        "worktree": stamp,
        "record": {
            "path": str(args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
            "producer_commit": round83.ROUND64_PRODUCER,
            "admission": str(args.round64_admission),
        },
        "source_order": list(BOUNDARIES),
        "cumulative_rows": rows,
        "incremental_residual_maxima": increments,
        "first_nonbit": first,
        "magnitude_test": magnitude,
        "hpg_inputs": hpg_inputs,
        "hpg_handed_velocity": handed_velocity,
        "live_total_closure": closure,
        "oracle_after_adv_equals_after_zad": after_adv_identity,
        "trace_noninterference": trace_noninterference,
        "prediction": {
            "density_nonbit": density_nonbit,
            "confirmed": confirmed,
        },
        "plant": args.plant,
        "plant_detail": plant_detail,
        "plant_fired": plant_fired,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--round64-root", type=Path,
                        default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--prerequisite-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("hpg-ulp", "density-ulp", "closure-ulp"))
    args = parser.parse_args(argv)
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND84_RHS_WALK", report["status"])
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
