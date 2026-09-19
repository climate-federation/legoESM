#!/usr/bin/env python3
"""Walk the admitted GYRE kt=2 ``stp2d`` slow-forcing producer."""

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
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round16_slow_forcing as round16  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
import nemo_testcase_l2_gyre_round81_btstep_gate as round81  # noqa: E402
import nemo_testcase_l2_gyre_round82_btstep_walk as round82  # noqa: E402
import nemo_testcase_l2_gyre_round117_preloop_gate as round117_record  # noqa: E402
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
ROUND64_PRODUCER = "3b3b045bd9e03b60330204e7590e4c4470b7a0ca"
ROUND117_PRODUCER = "c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd"
ROUND117_BOUNDARIES = (
    "after_hpg", "after_ldf", "after_vor", "after_keg",
    "after_zad", "after_adv",
)
ROUND117_FINAL_MAX = {
    "u": np.float64(1.0529650291768787e-11),
    "v": np.float64(1.0765559917925099e-11),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def owned3(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :30]


def owned3_with_bottom(value) -> np.ndarray:
    """Keep NEMO's non-contributing ``jpk`` slot for literal ``SUM`` replay."""
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :]


def owned2(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2]


def native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:, ...]


def native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :, ...]


def bottom_value(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select NEMO's deepest wet level from an already face-staggered field."""
    require(values.shape == mask.shape and values.ndim == 3, "bad bottom-field shape")
    count = np.sum(mask, axis=-1, dtype=np.int64)
    require(np.all(count[mask.any(axis=-1)] > 0), "wet face has no wet level")
    index = np.maximum(count - 1, 0)[..., None]
    return np.take_along_axis(values, index, axis=-1)[..., 0]


def source_chain(
    rhs: np.ndarray,
    e3: np.ndarray,
    mask3: np.ndarray,
    reciprocal_ref: np.ndarray,
    inverse_depth: np.ndarray,
    drag_coefficient: np.ndarray,
    bottom_velocity: np.ndarray,
    barotropic_velocity: np.ndarray,
    rho_reciprocal: np.float64,
    stress: np.ndarray,
    coriolis: np.ndarray,
    mask2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay compiled stp2d/dynspg statements without reassociation."""
    depth = round16._source_sum(e3, rhs, mask3, reciprocal_ref)
    residual = bottom_velocity - barotropic_velocity
    drag_increment = (inverse_depth * drag_coefficient) * residual
    post_drag = depth + drag_increment
    wind_increment = (rho_reciprocal * stress) * inverse_depth
    post_wind = post_drag + wind_increment
    final = post_wind - coriolis * mask2
    return {
        "depth_mean": depth,
        "drag_increment": drag_increment,
        "post_drag": post_drag,
        "wind_increment": wind_increment,
        "post_wind": post_wind,
        "final": final,
    }


def comparison(candidate, oracle, active) -> dict[str, object]:
    return round78.comparison(
        np.asarray(candidate, dtype=np.float64),
        np.asarray(oracle, dtype=np.float64),
        np.asarray(active, dtype=bool),
    )


def round117_source_order_accumulators(
    hpg_u, hpg_v, ldf_u, ldf_v, vor_u, vor_v,
    keg_u, keg_v, zad_u, zad_v,
):
    """Materialize the compiled stage-1 accumulator order under JIT."""
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
        "after_adv_u": zad_u,
        "after_adv_v": zad_v,
    }


def round117_first_nonbit(rows: dict[str, dict[str, dict]]) -> dict | None:
    """Return the first cumulative boundary, with U ordered before V."""
    for boundary in ROUND117_BOUNDARIES:
        for face in ("u", "v"):
            row = rows[face][boundary]
            if not row["bit_exact"]:
                return {"boundary": boundary, "face": face, **row}
    return None


def _full_from_native(full, native, face: str) -> np.ndarray:
    """Replace only the owned native face extent, retaining excluded halos."""
    result = np.array(full, dtype=np.float64, copy=True)
    native = np.asarray(native, dtype=np.float64)
    if face == "u":
        require(result[:, 1:].shape == native.shape,
                "U native/full extents disagree")
        result[:, 1:] = native
    elif face == "v":
        require(result[1:, :].shape == native.shape,
                "V native/full extents disagree")
        result[1:, :] = native
    else:
        raise ValueError(f"unknown face {face!r}")
    return result


def _round117_live_trace(
    card, seeded, freshwater, surface, execution_mode: str, *,
    incoming_override=None, final_override=None, association_arm=False,
):
    hooks = model_module._NEMOWSRK3TestHooks(
        expose_live_stage_operands=True,
        slow_forcing_incoming_override=incoming_override,
        barotropic_slow_forcing_override=final_override,
        nemo_stage_rhs_accumulation_order_arm=association_arm,
    )
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    model.prime_step_caches(seeded)
    return jax.device_get(round82._execute_step(
        model, seeded, card.dt_s, freshwater, surface, execution_mode))


@jax.jit
def _round117_subtract(incoming_u, incoming_v, coriolis_u, coriolis_v,
                       mask_u, mask_v):
    """Isolated JIT replay; deliberately not labelled production."""
    return (
        (incoming_u - coriolis_u) * mask_u,
        (incoming_v - coriolis_v) * mask_v,
    )


def _admit_round64(args) -> tuple[dict, dict]:
    admission = json.loads(args.round64_admission.read_text())
    require(admission.get("verdict") == "PASS", "Round-64 admission failed")
    require(
        (
            admission.get("byte_identical_records"),
            len(admission.get("classified_changed_records", [])),
            admission.get("admitted_difference_count"),
        ) == (43, 20, 132),
        "Round-64 inherited-record census changed",
    )
    producer = (args.round64_root / "producer_commit.txt").read_text().strip()
    require(producer == ROUND64_PRODUCER, "Round-64 producer commit changed")
    stage_path = args.round64_root / "oracle_momstage_kt00000002_s1.bin"
    original_stage = args.round46_root / stage_path.name
    require(stage_path.is_file() and original_stage.is_file(), "missing kt=2 stage record")
    classified = [
        row for row in admission["classified_changed_records"]
        if row["record"] == stage_path.name
    ]
    require(len(classified) == 1, "kt=2 stage record lacks one admission classification")
    stage_admission = classified[0]
    require(stage_admission["consumed_equal"], "kt=2 stage consumed projection differs")
    require(stage_admission["owned_field_differences"] == [],
            "kt=2 stage has an owned-field difference")
    require(
        stage_admission["reason_counts"] == {
            "halo": 7,
            "owned_defined_violation": 0,
            "owned_undefined_region": 0,
            "owned_undefined_slot": 0,
        },
        "kt=2 stage admission reasons changed",
    )
    used_fields = {
        "after_adv_u", "after_adv_v", "e3u_0", "e3v_0", "umask", "vmask",
        "u_Kmm", "v_Kmm", "uu_b_Kmm", "vv_b_Kmm", "has_ldf",
    }
    require(used_fields <= set(stage_admission["compared_fields"]),
            "a consumed kt=2 stage field is outside admission coverage")
    slow_path = args.round64_root / "oracle_slow_forcing_kt00000001.bin"
    stage = round46.read_stage(stage_path)
    slow = round16.read_slow_forcing(slow_path)
    require(stage["header"]["kt"] == 2 and stage["header"]["stage"] == 1,
            "wrong kt/stage record")
    require(stage["arrays"]["has_ldf"] == 1.0, "kt=2 RHS lacks final LDF row")
    return stage, slow


def _admit_round117_direct(args, external: dict) -> dict[str, object]:
    """Admit and parse the same-run kt=2 slow/pre-loop record pair."""
    admission = json.loads(args.preloop_admission.read_text())
    require(admission.get("verdict") == "PASS",
            "Round-117 direct-record admission failed")
    require(
        (
            admission.get("byte_identical_records"),
            len(admission.get("classified_changed_records", [])),
            admission.get("admitted_difference_count"),
        ) == (46, 20, 132),
        "Round-117 inherited-record census changed",
    )
    producer = (args.preloop_root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_preloop_record_commit == ROUND117_PRODUCER,
            "Round-117 direct-record producer changed")
    validation = json.loads(
        (args.preloop_root / "round117_preloop_validation.json").read_text())
    require(validation.get("status") == "PASS",
            "Round-117 direct-record gate did not pass")
    require(validation.get("producer_commit") == producer,
            "Round-117 validation producer changed")

    slow_path = args.preloop_root / round117_record.SLOW_RECORD
    preloop_path = args.preloop_root / round117_record.PRELOOP_RECORD
    for path in (slow_path, preloop_path):
        parts = path.with_name(path.name + ".stamp").read_text().split()
        require(len(parts) == 3, f"malformed direct-record stamp for {path.name}")
        require(parts == [sha256(path), producer, path.name],
                f"direct-record stamp moved for {path.name}")
        registered = validation["records"][path.name]
        require(registered["sha256"] == sha256(path),
                f"direct-record validation hash moved for {path.name}")

    slow = round117_record.read_slow_record(slow_path)
    preloop = round117_record.read_preloop_record(preloop_path)
    fields = preloop["fields"]
    for face in ("u", "v"):
        incoming = fields[f"incoming_{face}"]
        coriolis = fields[f"cor_{face}"][2:-2, 2:-2]
        mask = fields[f"{face}_mask"][2:-2, 2:-2]
        final = fields[f"final_{face}"]
        require(np.array_equal(slow["fields"][f"post_wind_{face}"], incoming),
                f"same-run slow/pre-loop {face.upper()} boundary moved")
        require(np.array_equal(incoming - coriolis * mask, final),
                f"compiled pre-loop {face.upper()} subtract did not replay")
        require(np.array_equal(final, external[f"slow_{face}"][0]),
                f"pre-loop/Round-81 {face.upper()} final boundary moved")
    return {
        "admission": admission,
        "validation": validation,
        "slow": slow,
        "preloop": preloop,
        "producer_commit": producer,
    }


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-83 measurement worktree is dirty")
    require(stamp["commit"] == args.expect_commit, "Round-83 commit stamp mismatch")

    stage, static = _admit_round64(args)
    bt = round82._admit(args)
    base, card, seeded, captured = round78._context(args)
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    trace = captured.slow_forcing_operands
    substeps = captured.substeps
    masks = gate.expected_masks(card)
    active = {
        "u3": np.asarray(masks["u"], dtype=bool),
        "v3": np.asarray(masks["v"], dtype=bool),
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]

    stage_arrays = stage["arrays"]
    oracle = {
        "u": {
            "rhs": owned3(stage_arrays["after_adv_u"]),
            "e3": owned3(stage_arrays["e3u_0"]),
            "mask3": owned3(stage_arrays["umask"]),
            "reciprocal_ref": np.asarray(static["r1_hu0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_u"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_u"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["u_Kmm"]), owned3(stage_arrays["umask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["uu_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_u"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_u"][0], dtype=np.float64),
        },
        "v": {
            "rhs": owned3(stage_arrays["after_adv_v"]),
            "e3": owned3(stage_arrays["e3v_0"]),
            "mask3": owned3(stage_arrays["vmask"]),
            "reciprocal_ref": np.asarray(static["r1_hv0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_v"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_v"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["v_Kmm"]), owned3(stage_arrays["vmask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["vv_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_v"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_v"][0], dtype=np.float64),
        },
    }
    live = {
        "u": {
            "rhs": native_u(trace["du_dt"]),
            "e3": native_u(trace["h_u"]),
            "mask3": active["u3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_u(trace["H_u"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_u"], "inverse_depth_u")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_u"], "drag_coefficient_u")[0],
            "bottom_velocity": bottom_value(native_u(seeded.u.data), active["u3"]),
            "barotropic_velocity": gate._trace_native(substeps["u_entry"], "u_entry")[0],
            "stress": native_u(trace["wind_tau_u"]),
            "coriolis": gate._trace_native(substeps["cor_u"], "cor_u")[0],
        },
        "v": {
            "rhs": native_v(trace["dv_dt"]),
            "e3": native_v(trace["h_v"]),
            "mask3": active["v3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_v(trace["H_v"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_v"], "inverse_depth_v")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_v"], "drag_coefficient_v")[0],
            "bottom_velocity": bottom_value(native_v(seeded.v.data), active["v3"]),
            "barotropic_velocity": gate._trace_native(substeps["v_entry"], "v_entry")[0],
            "stress": native_v(trace["wind_tau_v"]),
            "coriolis": gate._trace_native(substeps["cor_v"], "cor_v")[0],
        },
    }
    # The inherited slow-forcing stream is kt=1; its stress is not silently
    # reused at kt=2.  No direct kt=2 stress record exists.  Use the live kt=2
    # value only as a cross-record calibration operand, and withhold a direct
    # wind-stress identity claim.  The replay must still recover NEMO's final
    # recorded forcing exactly or the joined record is refused.
    oracle["u"]["stress"] = live["u"]["stress"]
    oracle["v"]["stress"] = live["v"]["stress"]
    rho_reciprocal = np.float64(static["r1_rho0"])
    require(float(trace["wind_r1_rho0"]) == float(rho_reciprocal),
            "live/oracle density reciprocal differs")

    ordinary_plant_target = {
        "e3-ulp": comparison(live["u"]["e3"], oracle["u"]["e3"], active["u3"]),
        "rhs-ulp": comparison(live["u"]["rhs"], oracle["u"]["rhs"], active["u3"]),
        "final-ulp": comparison(
            source_chain(
                **{key: value for key, value in oracle["u"].items() if key != "final"},
                rho_reciprocal=rho_reciprocal,
                mask2=active["u2"].astype(np.float64),
            )["final"],
            oracle["u"]["final"],
            active["u2"],
        ),
    }

    plant_detail = None
    if args.plant == "e3-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["e3"] - oracle["u"]["e3"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["e3"] = np.array(oracle["u"]["e3"], copy=True)
        direction = np.inf if oracle["u"]["e3"][at] >= live["u"]["e3"][at] else -np.inf
        oracle["u"]["e3"][at] = np.nextafter(oracle["u"]["e3"][at], direction)
        plant_detail = {"field": "e3_u", "location": list(map(int, at))}
    elif args.plant == "rhs-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["rhs"] - oracle["u"]["rhs"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["rhs"] = np.array(oracle["u"]["rhs"], copy=True)
        direction = np.inf if oracle["u"]["rhs"][at] >= live["u"]["rhs"][at] else -np.inf
        oracle["u"]["rhs"][at] = np.nextafter(oracle["u"]["rhs"][at], direction)
        plant_detail = {"field": "rhs_u", "location": list(map(int, at))}
    elif args.plant == "final-ulp":
        replay = source_chain(
            **{key: value for key, value in oracle["u"].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active["u2"].astype(np.float64),
        )["final"]
        delta = np.where(
            active["u2"], np.abs(replay - oracle["u"]["final"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["final"] = np.array(oracle["u"]["final"], copy=True)
        direction = np.inf if oracle["u"]["final"][at] >= replay[at] else -np.inf
        oracle["u"]["final"][at] = np.nextafter(oracle["u"]["final"][at], direction)
        plant_detail = {"field": "slow_u", "location": list(map(int, at))}

    rows = {}
    cross_record = {}
    reference_geometry_arms = {}
    for face in ("u", "v"):
        active3 = active[f"{face}3"]
        active2 = active[f"{face}2"]
        for name in oracle[face]:
            if name == "final":
                continue
            require(np.asarray(oracle[face][name]).shape == np.asarray(live[face][name]).shape,
                    f"{face}.{name} live/oracle shape mismatch")
        oracle_chain = source_chain(
            **{key: value for key, value in oracle[face].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        live_chain = source_chain(
            **live[face], rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        reference_geometry_inputs = dict(live[face])
        for name in ("e3", "mask3", "reciprocal_ref"):
            reference_geometry_inputs[name] = oracle[face][name]
        reference_geometry_chain = source_chain(
            **reference_geometry_inputs,
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        cross_record[face] = comparison(oracle_chain["final"], oracle[face]["final"], active2)
        reference_geometry_arms[face] = {
            "depth_mean": comparison(
                reference_geometry_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "final_slow_forcing": comparison(
                reference_geometry_chain["final"], oracle[face]["final"], active2),
            "movement_from_current_depth_mean": comparison(
                reference_geometry_chain["depth_mean"], live_chain["depth_mean"], active2),
        }
        rows[face] = {
            "e3": comparison(live[face]["e3"], oracle[face]["e3"], active3),
            "mask": comparison(live[face]["mask3"], oracle[face]["mask3"], active3),
            "three_dimensional_rhs": comparison(live[face]["rhs"], oracle[face]["rhs"], active3),
            "reference_depth_reciprocal": comparison(
                live[face]["reciprocal_ref"], oracle[face]["reciprocal_ref"], active2),
            "depth_mean": comparison(live_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "drag_coefficient": comparison(
                live[face]["drag_coefficient"], oracle[face]["drag_coefficient"], active2),
            "inverse_depth": comparison(
                live[face]["inverse_depth"], oracle[face]["inverse_depth"], active2),
            "bottom_velocity": comparison(
                live[face]["bottom_velocity"], oracle[face]["bottom_velocity"], active2),
            "barotropic_velocity": comparison(
                live[face]["barotropic_velocity"], oracle[face]["barotropic_velocity"], active2),
            "post_drag": comparison(live_chain["post_drag"], oracle_chain["post_drag"], active2),
            "post_wind": comparison(live_chain["post_wind"], oracle_chain["post_wind"], active2),
            "coriolis": comparison(live[face]["coriolis"], oracle[face]["coriolis"], active2),
            "final_slow_forcing": comparison(live_chain["final"], oracle[face]["final"], active2),
            "oracle_rhs_substitution_final": comparison(
                oracle_chain["final"], oracle[face]["final"], active2),
        }

    cross_exact = all(row["bit_exact"] for row in cross_record.values())
    first = None
    source_order = (
        "e3", "mask", "three_dimensional_rhs", "reference_depth_reciprocal",
        "depth_mean", "drag_coefficient", "inverse_depth", "bottom_velocity",
        "barotropic_velocity", "post_drag", "post_wind",
        "coriolis", "final_slow_forcing",
    )
    for boundary in source_order:
        for face in ("u", "v"):
            if first is None and not rows[face][boundary]["bit_exact"]:
                first = {"boundary": boundary, "face": face, **rows[face][boundary]}

    confirmed = bool(
        args.plant == "none"
        and cross_exact
        and first is not None
        and first["boundary"] == "three_dimensional_rhs"
        and all(rows[face]["oracle_rhs_substitution_final"]["bit_exact"] for face in ("u", "v"))
    )
    planted_row = {
        "e3-ulp": rows["u"]["e3"],
        "rhs-ulp": rows["u"]["three_dimensional_rhs"],
        "final-ulp": cross_record["u"],
    }.get(args.plant)
    plant_fires = bool(
        args.plant != "none"
        and planted_row != ordinary_plant_target[args.plant]
    )
    status = "CONFIRMED" if confirmed else ("PLANT_FIRED" if plant_fires else "REFUTED")
    return {
        "format": "nemo-testcase-l2-gyre-round83-slow-forcing-walk-v1",
        "status": status,
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "round64_stage_sha256": sha256(args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
        "round81_btstep_sha256": round81.sha256(args.record_root / round81.RECORD),
        "cross_record_replay": cross_record,
        "kt2_wind_stress_identity": "WITHHELD_NO_DIRECT_RECORD",
        "reference_geometry_arm": reference_geometry_arms,
        "first_non_bit_statement": first,
        "rows": rows,
        "plant": args.plant,
        "plant_detail": plant_detail,
        "plant_fires": plant_fires,
    }


def _round117_propagating_ulp(incoming, coriolis, mask):
    """Find one incoming ULP that survives the written subtract-and-mask."""
    incoming = np.asarray(incoming, dtype=np.float64)
    coriolis = np.asarray(coriolis, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    ordinary = (incoming - coriolis) * mask.astype(np.float64)
    for location in map(tuple, np.argwhere(mask)):
        for direction in (np.float64(np.inf), np.float64(-np.inf)):
            planted = np.array(incoming, copy=True)
            planted[location] = np.nextafter(planted[location], direction)
            changed = (planted - coriolis) * mask.astype(np.float64)
            if changed[location].view(np.uint64) != ordinary[location].view(np.uint64):
                return planted, tuple(int(index) for index in location), direction
    raise RuntimeError("no active one-ULP incoming plant survives subtraction")


def _round119_propagating_hpg_ulp(hpg, ldf, vor, keg, zad, mask):
    """Find one HPG ULP that survives every compiled accumulator boundary."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (hpg, ldf, vor, keg, zad)
    ]
    mask = np.asarray(mask, dtype=bool)
    require(all(value.shape == arrays[0].shape for value in arrays),
            "Round-119 term extents differ")
    require(mask.shape == arrays[0].shape, "Round-119 plant mask differs")
    for location in map(tuple, np.argwhere(mask)):
        ordinary = []
        value = arrays[0][location]
        ordinary.append(value)
        for term in arrays[1:]:
            value = value + term[location]
            ordinary.append(value)
        for direction in (np.float64(np.inf), np.float64(-np.inf)):
            planted = np.array(arrays[0], copy=True)
            value = np.nextafter(planted[location], direction)
            planted[location] = value
            changed = [value.view(np.uint64) != ordinary[0].view(np.uint64)]
            for index, term in enumerate(arrays[1:], start=1):
                value = value + term[location]
                changed.append(
                    value.view(np.uint64) != ordinary[index].view(np.uint64))
            if all(changed):
                return planted, tuple(int(index) for index in location), direction
    raise RuntimeError(
        "no active one-ULP HPG plant survives all source-order boundaries")


def measure_round117(args) -> dict[str, object]:
    """Split the current-tip producer in the existing admitted Round-83 gate."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-117 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-117 commit stamp mismatch")

    stage, static = _admit_round64(args)
    external = round82._admit(args)
    direct = _admit_round117_direct(
        args, external) if (args.round118 or args.round119) else None
    (base, card, seeded, freshwater, surface,
     captured) = round82._context(args)
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    ordinary = _round117_live_trace(
        card, seeded, freshwater, surface, args.execution_mode)
    source_order = (
        _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            association_arm=("stage1-source-order", None))
        if args.round119 else None)
    trace_identity = round82._pytree_identity(
        ordinary.state_after, captured.plain_state)
    require(trace_identity["bit_exact"],
            "producer trace moved the returned production state")

    producer = ordinary.slow_forcing_producer
    masks = gate.expected_masks(card)
    active = {
        "u3": np.asarray(masks["u"], dtype=bool),
        "v3": np.asarray(masks["v"], dtype=bool),
        "t3": np.asarray(masks["T"], dtype=bool),
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]
    active["t2"] = active["t3"][..., 0]
    require(np.array_equal(external["u_mask"] != 0.0, active["u2"]),
            "Round-81 U mask differs from the live native mask")
    require(np.array_equal(external["v_mask"] != 0.0, active["v2"]),
            "Round-81 V mask differs from the live native mask")

    live_split = {
        "u": {
            "incoming": native_u(producer["incoming_u"]),
            "coriolis": native_u(producer["coriolis_u"]),
            "final": native_u(producer["final_u"]),
        },
        "v": {
            "incoming": native_v(producer["incoming_v"]),
            "coriolis": native_v(producer["coriolis_v"]),
            "final": native_v(producer["final_v"]),
        },
    }
    if direct is not None:
        direct_fields = direct["preloop"]["fields"]
        oracle_split = {
            face: {
                "direct_incoming": np.asarray(
                    direct_fields[f"incoming_{face}"], dtype=np.float64),
                "direct_coriolis": np.asarray(
                    direct_fields[f"cor_{face}"][2:-2, 2:-2],
                    dtype=np.float64),
                "final": np.asarray(
                    direct_fields[f"final_{face}"], dtype=np.float64),
            }
            for face in ("u", "v")
        }
    else:
        oracle_split = {
            "u": {
                "substep1_coriolis_proxy": np.asarray(
                    external["cor_u"][0], dtype=np.float64),
                "final": np.asarray(external["slow_u"][0], dtype=np.float64),
            },
            "v": {
                "substep1_coriolis_proxy": np.asarray(
                    external["cor_v"][0], dtype=np.float64),
                "final": np.asarray(external["slow_v"][0], dtype=np.float64),
            },
        }
        for face in ("u", "v"):
            mask = active[f"{face}2"].astype(np.float64)
            oracle_split[face]["proxy_incoming_preimage"] = (
                oracle_split[face]["final"]
                + oracle_split[face]["substep1_coriolis_proxy"] * mask)

    actual_external_final = {
        "u": gate._trace_native(
            captured.substeps["slow_u"], "slow_u")[0],
        "v": gate._trace_native(
            captured.substeps["slow_v"], "slow_v")[0],
    }
    trace_final_identity = {
        face: comparison(
            live_split[face]["final"], actual_external_final[face],
            active[f"{face}2"])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in trace_final_identity.values()),
            "WRITE-only producer trace changed the actual external forcing")

    isolated_final_u, isolated_final_v = jax.device_get(_round117_subtract(
        jnp.asarray(live_split["u"]["incoming"]),
        jnp.asarray(live_split["v"]["incoming"]),
        jnp.asarray(live_split["u"]["coriolis"]),
        jnp.asarray(live_split["v"]["coriolis"]),
        jnp.asarray(active["u2"], dtype=np.float64),
        jnp.asarray(active["v2"], dtype=np.float64),
    ))
    if direct is not None:
        split_rows = {
            face: {
                "incoming_vs_direct_record": comparison(
                    live_split[face]["incoming"],
                    oracle_split[face]["direct_incoming"],
                    active[f"{face}2"]),
                "preloop_coriolis_vs_direct_record": comparison(
                    live_split[face]["coriolis"],
                    oracle_split[face]["direct_coriolis"],
                    active[f"{face}2"]),
                "final_vs_direct_record": comparison(
                    live_split[face]["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
                "isolated_subtract_vs_production_final": comparison(
                    isolated_final_u if face == "u" else isolated_final_v,
                    live_split[face]["final"], active[f"{face}2"]),
            }
            for face in ("u", "v")
        }
        split_order = (
            "incoming_vs_direct_record",
            "preloop_coriolis_vs_direct_record",
            "final_vs_direct_record",
        )
    else:
        split_rows = {
            face: {
                "incoming_vs_substep_proxy_preimage": comparison(
                    live_split[face]["incoming"],
                    oracle_split[face]["proxy_incoming_preimage"],
                    active[f"{face}2"]),
                "preloop_coriolis_vs_substep1_proxy": comparison(
                    live_split[face]["coriolis"],
                    oracle_split[face]["substep1_coriolis_proxy"],
                    active[f"{face}2"]),
                "final_vs_direct_record": comparison(
                    live_split[face]["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
                "isolated_subtract_vs_production_final": comparison(
                    isolated_final_u if face == "u" else isolated_final_v,
                    live_split[face]["final"], active[f"{face}2"]),
            }
            for face in ("u", "v")
        }
        split_order = (
            "incoming_vs_substep_proxy_preimage",
            "preloop_coriolis_vs_substep1_proxy",
            "final_vs_direct_record",
        )
    proxy_first = None
    for boundary in split_order:
        for face in ("u", "v"):
            row = split_rows[face][boundary]
            if proxy_first is None and not row["bit_exact"]:
                proxy_first = {"boundary": boundary, "face": face, **row}
    direct_first = next((
        {"boundary": boundary, "face": face, **split_rows[face][boundary]}
        for boundary in split_order
        for face in ("u", "v")
        if not split_rows[face][boundary]["bit_exact"]
    ), None) if direct is not None else next((
        {"boundary": "final_vs_direct_record", "face": face,
         **split_rows[face]["final_vs_direct_record"]}
        for face in ("u", "v")
        if not split_rows[face]["final_vs_direct_record"]["bit_exact"]
    ), None)
    cancellation = None
    if direct is not None:
        cancellation = {}
        for face in ("u", "v"):
            mask = active[f"{face}2"]
            incoming_error = (
                live_split[face]["incoming"]
                - oracle_split[face]["direct_incoming"])[mask]
            coriolis_error = (
                live_split[face]["coriolis"]
                - oracle_split[face]["direct_coriolis"])[mask]
            final_error = (
                live_split[face]["final"]
                - oracle_split[face]["final"])[mask]
            component_l1 = float(
                np.sum(np.abs(incoming_error), dtype=np.float64)
                + np.sum(np.abs(coriolis_error), dtype=np.float64))
            final_l1 = float(np.sum(np.abs(final_error), dtype=np.float64))
            cancellation[face] = {
                "incoming_error_max": float(
                    np.max(np.abs(incoming_error), initial=0.0)),
                "coriolis_error_max": float(
                    np.max(np.abs(coriolis_error), initial=0.0)),
                "final_error_max": float(
                    np.max(np.abs(final_error), initial=0.0)),
                "component_l1": component_l1,
                "final_l1": final_l1,
                "cancellation_fraction": (
                    0.0 if component_l1 == 0.0 else
                    float(1.0 - final_l1 / component_l1)),
                "subtract_cancelling_cells": int(np.count_nonzero(
                    (incoming_error != 0.0) & (coriolis_error != 0.0)
                    & (np.signbit(incoming_error)
                       == np.signbit(coriolis_error)))),
            }

    if args.plant == "incoming-ulp":
        planted_native, location, direction = _round117_propagating_ulp(
            live_split["u"]["incoming"], live_split["u"]["coriolis"],
            active["u2"])
        incoming_override = (
            jnp.asarray(_full_from_native(
                producer["incoming_u"], planted_native, "u")),
            jnp.asarray(producer["incoming_v"]),
        )
        planted = _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            incoming_override=incoming_override)
        planted_producer = planted.slow_forcing_producer
        plant_rows = {
            "incoming_u": comparison(
                native_u(planted_producer["incoming_u"]),
                live_split["u"]["incoming"], active["u2"]),
            "coriolis_u": comparison(
                native_u(planted_producer["coriolis_u"]),
                live_split["u"]["coriolis"], active["u2"]),
            "final_u": comparison(
                native_u(planted_producer["final_u"]),
                live_split["u"]["final"], active["u2"]),
        }
        plant_fires = bool(
            plant_rows["incoming_u"]["differing_cells"] == 1
            and plant_rows["coriolis_u"]["bit_exact"]
            and plant_rows["final_u"]["differing_cells"] > 0)
        require(plant_fires, "incoming ULP plant did not cross production subtract")
        return {
            "format": (
                "nemo-testcase-l2-gyre-round119-association-walk-v1"
                if args.round119 else
                ("nemo-testcase-l2-gyre-round118-producer-walk-v1"
                 if args.round118 else
                 "nemo-testcase-l2-gyre-round117-producer-walk-v1")),
            "status": "PLANT_FIRED",
            "worktree": stamp,
            "execution_regime": args.execution_mode,
            "trace_noninterference": trace_identity,
            "plant": args.plant,
            "plant_location": list(location),
            "plant_direction": float(direction),
            "plant_rows": plant_rows,
            "plant_fires": plant_fires,
        }

    stage_arrays = stage["arrays"]
    parts = ordinary.operator_operands[0]
    cumulative = jax.device_get(jax.jit(round117_source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    cumulative_rows = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        for boundary in ROUND117_BOUNDARIES:
            candidate = (
                native_u(cumulative[f"{boundary}_{face}"])
                if face == "u" else
                native_v(cumulative[f"{boundary}_{face}"]))
            reference = owned3(stage_arrays[f"{boundary}_{face}"])
            cumulative_rows[face][boundary] = comparison(
                candidate, reference, active[f"{face}3"])
    cumulative_first = round117_first_nonbit(cumulative_rows)
    cumulative_closure = {
        "u": comparison(
            native_u(cumulative["after_adv_u"]),
            native_u(producer["rhs_u"]), active["u3"]),
        "v": comparison(
            native_v(cumulative["after_adv_v"]),
            native_v(producer["rhs_v"]), active["v3"]),
    }
    oracle_after_adv_identity = {
        face: comparison(
            owned3(stage_arrays[f"after_adv_{face}"]),
            owned3(stage_arrays[f"after_zad_{face}"]), active[f"{face}3"])
        for face in ("u", "v")
    }
    incremental_residual = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        previous = np.zeros_like(
            native_u(cumulative["after_hpg_u"])
            if face == "u" else native_v(cumulative["after_hpg_v"]))
        mask = active[f"{face}3"]
        for boundary in ROUND117_BOUNDARIES:
            candidate = (
                native_u(cumulative[f"{boundary}_{face}"])
                if face == "u" else
                native_v(cumulative[f"{boundary}_{face}"]))
            residual = candidate - owned3(stage_arrays[f"{boundary}_{face}"])
            delta = residual - previous
            incremental_residual[face][boundary] = float(
                np.max(np.abs(delta[mask]), initial=0.0))
            previous = residual

    association_walk = None
    zad_operand_walk = None
    if args.round119:
        source_parts = source_order.operator_operands[0]

        def _native_field(value, face):
            data = value.data if hasattr(value, "data") else value
            return native_u(data) if face == "u" else native_v(data)

        raw_names = ("hpg", "ldf", "vorticity", "keg", "zad")
        raw_identity = {
            face: {
                name: comparison(
                    _native_field(source_parts[f"{name}_{face}"], face),
                    _native_field(parts[f"{name}_{face}"], face),
                    active[f"{face}3"],
                )
                for name in raw_names
            }
            for face in ("u", "v")
        }
        production_vs_isolated = {face: {} for face in ("u", "v")}
        production_vs_oracle = {face: {} for face in ("u", "v")}
        for face in ("u", "v"):
            for boundary in ROUND117_BOUNDARIES:
                production_key = (
                    "after_adv" if boundary in ("after_zad", "after_adv")
                    else boundary)
                production_value = _native_field(
                    source_parts[f"{production_key}_{face}"], face)
                isolated_value = (
                    native_u(cumulative[f"{boundary}_{face}"])
                    if face == "u" else
                    native_v(cumulative[f"{boundary}_{face}"]))
                production_vs_isolated[face][boundary] = comparison(
                    production_value, isolated_value, active[f"{face}3"])
                production_vs_oracle[face][boundary] = comparison(
                    production_value,
                    owned3(stage_arrays[f"{boundary}_{face}"]),
                    active[f"{face}3"])

        ordinary_final = {
            face: _native_field(parts[f"after_ldf_{face}"], face)
            for face in ("u", "v")
        }
        source_final = {
            face: _native_field(source_parts[f"after_adv_{face}"], face)
            for face in ("u", "v")
        }
        ordinary_live_closure = {
            "u": comparison(
                ordinary_final["u"], native_u(producer["rhs_u"]), active["u3"]),
            "v": comparison(
                ordinary_final["v"], native_v(producer["rhs_v"]), active["v3"]),
        }
        source_producer = source_order.slow_forcing_producer
        source_live_closure = {
            "u": comparison(
                source_final["u"], native_u(source_producer["rhs_u"]),
                active["u3"]),
            "v": comparison(
                source_final["v"], native_v(source_producer["rhs_v"]),
                active["v3"]),
        }
        source_to_ordinary = {
            face: comparison(
                source_final[face], ordinary_final[face], active[f"{face}3"])
            for face in ("u", "v")
        }
        association_confirmed = bool(
            all(row["bit_exact"] for rows in raw_identity.values()
                for row in rows.values())
            and all(row["bit_exact"] for rows in production_vs_isolated.values()
                    for row in rows.values())
            and all(row["bit_exact"] for row in ordinary_live_closure.values())
            and all(row["bit_exact"] for row in source_live_closure.values())
            and source_to_ordinary["u"]["differing_cells"] == 6882
            and source_to_ordinary["v"]["differing_cells"] == 6566
            and all(row["absolute_max"]
                    == np.float64(8.470329472543003e-22)
                    for row in source_to_ordinary.values()))
        association_walk = {
            "ordinary_label": (
                "full production step; actual combined (HPG+KEG) -> VOR -> "
                "ZAD -> LDF accumulator"),
            "source_order_label": (
                "full production step; private HPG -> LDF -> VOR -> KEG -> "
                "ZAD association arm"),
            "isolated_label": "isolated-closure JIT; not production",
            "raw_operand_identity": raw_identity,
            "source_order_production_vs_isolated": production_vs_isolated,
            "source_order_production_vs_oracle": production_vs_oracle,
            "ordinary_final_vs_ordinary_live_total": ordinary_live_closure,
            "source_final_vs_source_live_total": source_live_closure,
            "source_order_vs_ordinary_final": source_to_ordinary,
            "state_after_arm_vs_ordinary": round82._pytree_identity(
                source_order.state_after, ordinary.state_after),
            "prediction_confirmed": association_confirmed,
        }

        _, live_r3u, live_r3v = ordinary.stage_qco[0]
        zad_live = {
            "velocity_u": native_u(parts["operand_velocity_u"]),
            "velocity_v": native_v(parts["operand_velocity_v"]),
            "ww": np.asarray(parts["operand_zad_w"], dtype=np.float64)[..., :30],
            "r3u": native_u(live_r3u)[..., 0],
            "r3v": native_v(live_r3v)[..., 0],
            "thickness_u": native_u(parts["operand_zad_h_u"]),
            "thickness_v": native_v(parts["operand_zad_h_v"]),
            "area_t": np.asarray(card.recipe.grid.area_T, dtype=np.float64),
            "reciprocal_area_u": native_u(
                np.float64(1.0) / (
                    np.asarray(card.recipe.grid.dx_u, dtype=np.float64)
                    * np.asarray(card.recipe.grid.dy_u, dtype=np.float64))),
            "reciprocal_area_v": native_v(
                np.float64(1.0) / (
                    np.asarray(card.recipe.grid.dx_v, dtype=np.float64)
                    * np.asarray(card.recipe.grid.dy_v, dtype=np.float64))),
        }
        zad_oracle = {
            "velocity_u": owned3(stage_arrays["u_Kmm"]),
            "velocity_v": owned3(stage_arrays["v_Kmm"]),
            "ww": owned3(stage_arrays["ww"]),
            "r3u": owned2(stage_arrays["r3u_Kmm"]),
            "r3v": owned2(stage_arrays["r3v_Kmm"]),
            "thickness_u": owned3(stage_arrays["e3u_Kmm"]),
            "thickness_v": owned3(stage_arrays["e3v_Kmm"]),
            "area_t": owned2(stage_arrays["e1e2t"]),
            "reciprocal_area_u": owned2(stage_arrays["r1_e1e2u"]),
            "reciprocal_area_v": owned2(stage_arrays["r1_e1e2v"]),
        }
        zad_active = {
            "velocity_u": active["u3"], "velocity_v": active["v3"],
            "ww": active["t3"], "r3u": active["u2"],
            "r3v": active["v2"], "thickness_u": active["u3"],
            "thickness_v": active["v3"], "area_t": active["t2"],
            "reciprocal_area_u": active["u2"],
            "reciprocal_area_v": active["v2"],
        }
        zad_order = tuple(zad_live)
        zad_rows = {
            name: comparison(zad_live[name], zad_oracle[name], zad_active[name])
            for name in zad_order
        }
        zad_first = next(
            (name for name in zad_order if not zad_rows[name]["bit_exact"]), None)
        zad_operand_walk = {
            "order": list(zad_order),
            "rows": zad_rows,
            "first_non_bit": zad_first,
            "largest_absolute_max": max(
                zad_order, key=lambda name: zad_rows[name]["absolute_max"]),
            "prediction_confirmed": bool(
                zad_first == "ww"
                and max(zad_order,
                        key=lambda name: zad_rows[name]["absolute_max"]) == "ww"
                and all(zad_rows[name]["bit_exact"] for name in (
                    "velocity_u", "velocity_v", "thickness_u", "thickness_v",
                    "area_t", "reciprocal_area_u", "reciprocal_area_v"))),
        }

        if args.plant == "association-hpg-ulp":
            planted_native, location, direction = _round119_propagating_hpg_ulp(
                _native_field(parts["hpg_u"], "u"),
                _native_field(parts["ldf_u"], "u"),
                _native_field(parts["vorticity_u"], "u"),
                _native_field(parts["keg_u"], "u"),
                _native_field(parts["zad_u"], "u"), active["u3"])
            hpg_override = (
                jnp.asarray(_full_from_native(
                    parts["hpg_u"].data, planted_native, "u")),
                jnp.asarray(parts["hpg_v"].data),
            )
            planted = _round117_live_trace(
                card, seeded, freshwater, surface, args.execution_mode,
                association_arm=("stage1-source-order", hpg_override))
            planted_parts = planted.operator_operands[0]
            planted_boundaries = {}
            for boundary in ROUND117_BOUNDARIES:
                production_key = (
                    "after_adv" if boundary in ("after_zad", "after_adv")
                    else boundary)
                planted_boundaries[boundary] = comparison(
                    _native_field(
                        planted_parts[f"{production_key}_u"], "u"),
                    _native_field(source_parts[f"{production_key}_u"], "u"),
                    active["u3"])
            planted_raw_identity = {
                name: comparison(
                    _native_field(planted_parts[f"{name}_u"], "u"),
                    _native_field(source_parts[f"{name}_u"], "u"),
                    active["u3"])
                for name in raw_names
            }
            plant_fires = bool(
                planted_raw_identity["hpg"]["differing_cells"] == 1
                and all(planted_raw_identity[name]["bit_exact"]
                        for name in raw_names if name != "hpg")
                and all(row["differing_cells"] > 0
                        for row in planted_boundaries.values()))
            require(plant_fires,
                    "Round-119 HPG ULP did not propagate through every boundary")
            return {
                "format": "nemo-testcase-l2-gyre-round119-association-walk-v1",
                "status": "PLANT_FIRED",
                "worktree": stamp,
                "execution_regime": args.execution_mode,
                "trace_noninterference": trace_identity,
                "plant": args.plant,
                "plant_location": list(location),
                "plant_direction": float(direction),
                "raw_operand_rows": planted_raw_identity,
                "boundary_rows": planted_boundaries,
                "plant_fires": plant_fires,
            }

    substeps = captured.substeps
    source_replay = {}
    if direct is not None:
        slow_fields = direct["slow"]["fields"]
        rho_reciprocal = np.float64(slow_fields["r1_rho0"])
        require(float(producer["wind_r1_rho0"]) == float(rho_reciprocal),
                "live/direct density reciprocal differs")
        for face in ("u", "v"):
            # round16._source_sum implements NEMO's 1:jpkm1 reduction by
            # deliberately omitting the final, non-contributing jpk slot.
            # Preserve that slot here: trimming first would omit physical
            # level jpkm1 as well (the Round-28 instrument defect).
            e3 = owned3_with_bottom(slow_fields[f"e3{face}"])
            mask3 = owned3_with_bottom(slow_fields[f"{face}mask"])
            direct_rhs = owned3_with_bottom(slow_fields[f"krhs_{face}"])
            inherited_rhs = owned3(stage_arrays[f"after_adv_{face}"])
            reciprocal_ref = owned2(slow_fields[f"r1_h{face}0"])
            inverse_depth = owned2(slow_fields[f"r1_h{face}"])
            drag = owned2(slow_fields[f"cd_{face}"])
            bottom = bottom_value(
                owned3(stage_arrays[f"{face}_Kmm"]), mask3[..., :30] != 0.0)
            barotropic = owned2(
                direct["preloop"]["fields"][f"{face}_kmm"])
            stress = owned2(slow_fields[f"{face}tau"])
            coriolis = oracle_split[face]["direct_coriolis"]
            chain = source_chain(
                rhs=direct_rhs, e3=e3, mask3=mask3,
                reciprocal_ref=reciprocal_ref,
                inverse_depth=inverse_depth, drag_coefficient=drag,
                bottom_velocity=bottom, barotropic_velocity=barotropic,
                rho_reciprocal=rho_reciprocal, stress=stress,
                coriolis=coriolis,
                mask2=active[f"{face}2"].astype(np.float64),
            )
            source_replay[face] = {
                "inherited_after_adv_to_same_run_krhs": comparison(
                    inherited_rhs, direct_rhs[..., :30], active[f"{face}3"]),
                "depth_replay_to_direct_record": comparison(
                    chain["depth_mean"], slow_fields[f"depth_{face}"],
                    active[f"{face}2"]),
                "post_drag_replay_to_direct_record": comparison(
                    chain["post_drag"], slow_fields[f"post_drag_{face}"],
                    active[f"{face}2"]),
                "post_wind_replay_to_direct_record": comparison(
                    chain["post_wind"], slow_fields[f"post_wind_{face}"],
                    active[f"{face}2"]),
                "forward_final_vs_direct_record": comparison(
                    chain["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
            }
    else:
        # The Round-64 stream has no direct kt=2 stress snapshot.  Keep the
        # proxy join explicit and withhold input certification.
        rho_reciprocal = np.float64(static["r1_rho0"])
        require(float(producer["wind_r1_rho0"]) == float(rho_reciprocal),
                "live/oracle density reciprocal differs")
        for face in ("u", "v"):
            if face == "u":
                e3 = owned3(stage_arrays["e3u_0"])
                mask3 = owned3(stage_arrays["umask"])
                reciprocal_ref = np.asarray(static["r1_hu0"], dtype=np.float64)
                inverse_depth = gate._trace_native(
                    substeps["inverse_depth_u"], "inverse_depth_u")[0]
                drag = gate._trace_native(
                    substeps["drag_coefficient_u"], "drag_coefficient_u")[0]
                bottom = bottom_value(
                    owned3(stage_arrays["u_Kmm"]), mask3 != 0.0)
                barotropic = owned2(stage_arrays["uu_b_Kmm"])
                stress = native_u(producer["wind_tau_u"])
                coriolis = oracle_split[face]["substep1_coriolis_proxy"]
            else:
                e3 = owned3(stage_arrays["e3v_0"])
                mask3 = owned3(stage_arrays["vmask"])
                reciprocal_ref = np.asarray(static["r1_hv0"], dtype=np.float64)
                inverse_depth = gate._trace_native(
                    substeps["inverse_depth_v"], "inverse_depth_v")[0]
                drag = gate._trace_native(
                    substeps["drag_coefficient_v"], "drag_coefficient_v")[0]
                bottom = bottom_value(
                    owned3(stage_arrays["v_Kmm"]), mask3 != 0.0)
                barotropic = owned2(stage_arrays["vv_b_Kmm"])
                stress = native_v(producer["wind_tau_v"])
                coriolis = oracle_split[face]["substep1_coriolis_proxy"]
            chain = source_chain(
                rhs=owned3(stage_arrays[f"after_adv_{face}"]),
                e3=e3, mask3=mask3, reciprocal_ref=reciprocal_ref,
                inverse_depth=inverse_depth, drag_coefficient=drag,
                bottom_velocity=bottom, barotropic_velocity=barotropic,
                rho_reciprocal=rho_reciprocal, stress=stress,
                coriolis=coriolis,
                mask2=active[f"{face}2"].astype(np.float64),
            )
            source_replay[face] = {
                "post_wind_vs_substep_proxy_preimage": comparison(
                    chain["post_wind"],
                    oracle_split[face]["proxy_incoming_preimage"],
                    active[f"{face}2"]),
                "forward_final_vs_direct_record": comparison(
                    chain["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
            }

    if direct is not None:
        p1_confirmed = bool(
            direct_first is not None
            and direct_first["boundary"] == "incoming_vs_direct_record"
            and direct_first["face"] == "u"
            and all(
                split_rows[face]["incoming_vs_direct_record"]
                ["differing_cells"] == (580 if face == "u" else 570)
                and abs(
                    split_rows[face]["incoming_vs_direct_record"]
                    ["absolute_max"] - ROUND117_FINAL_MAX[face])
                <= np.float64(1.0e-22)
                and split_rows[face]["preloop_coriolis_vs_direct_record"]
                ["bit_exact"]
                and split_rows[face]["final_vs_direct_record"]
                ["differing_cells"] == (580 if face == "u" else 570)
                and abs(
                    split_rows[face]["final_vs_direct_record"]
                    ["absolute_max"] - ROUND117_FINAL_MAX[face])
                <= np.float64(1.0e-22)
                for face in ("u", "v"))
            and all(split_rows[face]["isolated_subtract_vs_production_final"]
                    ["bit_exact"] for face in ("u", "v")))
    else:
        p1_confirmed = False
    ldf_ranges = {
        "u": (np.float64(6.3e-15), np.float64(1.02e-13)),
        "v": (np.float64(8.7e-15), np.float64(1.41e-13)),
    }
    common_p2 = bool(
        cumulative_first is not None
        and cumulative_first["boundary"] == "after_ldf"
        and all(cumulative_rows[face]["after_hpg"]["bit_exact"]
                for face in ("u", "v"))
        and all(
            ldf_ranges[face][0]
            <= cumulative_rows[face]["after_ldf"]["absolute_max"]
            <= ldf_ranges[face][1]
            for face in ("u", "v"))
        and all(max(incremental_residual[face],
                    key=incremental_residual[face].get) == "after_zad"
                for face in ("u", "v"))
        and all(row["bit_exact"] for row in oracle_after_adv_identity.values())
    )
    if direct is not None:
        p2_confirmed = bool(
            common_p2
            and all(
                abs(cumulative_rows[face]["after_ldf"]["absolute_max"]
                    - value) <= np.float64(1.0e-26)
                for face, value in {
                    "u": np.float64(2.5292467120726215e-14),
                    "v": np.float64(3.502735092670824e-14),
                }.items())
            and cumulative_closure["u"]["differing_cells"] == 6882
            and cumulative_closure["v"]["differing_cells"] == 6566
            and all(
                row["absolute_max"] == np.float64(8.470329472543003e-22)
                for row in cumulative_closure.values())
            and all(
                row["bit_exact"]
                for face_rows in source_replay.values()
                for row in face_rows.values()))
    else:
        p2_confirmed = bool(
            common_p2
            and all(row["bit_exact"] for row in cumulative_closure.values()))

    magnitude = None
    if args.execution_mode == "production-jit" and direct is None:
        directed_final = (
            jnp.asarray(_full_from_native(
                producer["final_u"], oracle_split["u"]["final"], "u")),
            jnp.asarray(_full_from_native(
                producer["final_v"], oracle_split["v"]["final"], "v")),
        )
        directed = _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            final_override=directed_final)
        directed_producer = directed.slow_forcing_producer
        override_identity = {
            "incoming_u": comparison(
                native_u(directed_producer["incoming_u"]),
                live_split["u"]["incoming"], active["u2"]),
            "incoming_v": comparison(
                native_v(directed_producer["incoming_v"]),
                live_split["v"]["incoming"], active["v2"]),
            "coriolis_u": comparison(
                native_u(directed_producer["coriolis_u"]),
                live_split["u"]["coriolis"], active["u2"]),
            "coriolis_v": comparison(
                native_v(directed_producer["coriolis_v"]),
                live_split["v"]["coriolis"], active["v2"]),
            "final_u": comparison(
                native_u(directed_producer["final_u"]),
                oracle_split["u"]["final"], active["u2"]),
            "final_v": comparison(
                native_v(directed_producer["final_v"]),
                oracle_split["v"]["final"], active["v2"]),
        }
        require(all(row["bit_exact"] for row in override_identity.values()),
                "directed arm changed or missed a registered producer input")
        next_entry = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")
        references = {
            "ssh": np.asarray(next_entry["ssh"], dtype=np.float64),
            "T": np.asarray(next_entry["T"], dtype=np.float64)[..., :30],
            "S": np.asarray(next_entry["S"], dtype=np.float64)[..., :30],
        }
        ordinary_values = {
            "ssh": np.asarray(ordinary.barotropic_targets[4]),
            "T": np.asarray(ordinary.stage_outputs[2][2]),
            "S": np.asarray(ordinary.stage_outputs[2][3]),
        }
        directed_values = {
            "ssh": np.asarray(directed.barotropic_targets[4]),
            "T": np.asarray(directed.stage_outputs[2][2]),
            "S": np.asarray(directed.stage_outputs[2][3]),
        }
        field_masks = {"ssh": active["t2"], "T": active["t3"],
                       "S": active["t3"]}
        rows = {
            name: {
                "ordinary": comparison(
                    ordinary_values[name], references[name], field_masks[name]),
                "record_directed": comparison(
                    directed_values[name], references[name], field_masks[name]),
                "directed_minus_ordinary": comparison(
                    directed_values[name], ordinary_values[name],
                    field_masks[name]),
            }
            for name in ("ssh", "T", "S")
        }
        bands = {"ssh": 0.10, "T": 0.10, "S": 0.50}
        band_ok = all(
            abs(rows[name]["record_directed"]["absolute_max"]
                / rows[name]["ordinary"]["absolute_max"] - 1.0)
            < bands[name]
            for name in rows
        )
        predicted_direction = all(
            rows[name]["record_directed"]["absolute_max"]
            < rows[name]["ordinary"]["absolute_max"]
            for name in ("ssh", "T")
        )
        magnitude_confirmed = bool(
            all(rows[name]["directed_minus_ordinary"]["differing_cells"] > 0
                for name in rows)
            and all(not rows[name]["record_directed"]["bit_exact"]
                    for name in rows)
            and band_ok and predicted_direction)
        magnitude = {
            "intervention": (
                "replace only the final frozen slow-U/slow-V pair at the "
                "external-solver call"),
            "changed_operand_registry": ["final_slow_forcing_pair"],
            "producer_override_identity": override_identity,
            "rows": rows,
            "band_ok": band_ok,
            "predicted_ssh_and_T_improve": predicted_direction,
            "prediction_confirmed": magnitude_confirmed,
        }

    input_certified = bool(
        direct is not None
        and all(
            row["bit_exact"]
            for face_rows in source_replay.values()
            for row in face_rows.values()))
    prediction_confirmed = bool(
        p1_confirmed and p2_confirmed
        and (magnitude is None or magnitude["prediction_confirmed"])
        and (not args.round119 or (
            association_walk is not None
            and association_walk["prediction_confirmed"]
            and zad_operand_walk is not None
            and zad_operand_walk["prediction_confirmed"])))
    status = (
        "MEASURED" if args.execution_mode == "production-eager" else
        ("CONFIRMED" if prediction_confirmed else "REFUTED"))
    return {
        "format": (
            "nemo-testcase-l2-gyre-round119-association-walk-v1"
            if args.round119 else
            ("nemo-testcase-l2-gyre-round118-producer-walk-v1"
             if direct is not None else
             "nemo-testcase-l2-gyre-round117-producer-walk-v1")),
        "status": status,
        "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "round64_stage_sha256": sha256(
            args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
        "round81_btstep_sha256": round81.sha256(
            args.record_root / round81.RECORD),
        "trace_noninterference": trace_identity,
        "trace_final_vs_actual_external_call": trace_final_identity,
        "producer_split": {
            "record_limitation": (
                None if direct is not None else
                "Round-81 cor_u/cor_v is the substep-1 dyn_cor_2D result at "
                "compiled lines 683-686, not the pre-loop Kmm result at line "
                "322; direct pre-loop incoming and Coriolis rows are absent"),
            "incoming_reference": (
                "admitted direct same-run NEMO pre-loop field"
                if direct is not None else
                "PROXY algebraic preimage from final plus substep-1 Coriolis; "
                "not a direct NEMO pre-loop field"),
            "source_order": ["incoming", "coriolis", "final"],
            "rows": split_rows,
            "first_proxy_non_bit": proxy_first,
            "first_direct_non_bit": direct_first,
            "incoming_coriolis_cancellation": cancellation,
            "prediction_confirmed": p1_confirmed,
        },
        "isolated_subtract": {
            "label": "isolated-closure JIT; not production",
            "rows": {
                face: split_rows[face][
                    "isolated_subtract_vs_production_final"]
                for face in ("u", "v")},
        },
        ("round64_round117_join" if direct is not None
         else "round64_round81_join"): {
            "kt2_wind_stress_identity": (
                "DIRECT_SAME_RUN" if direct is not None
                else "WITHHELD_NO_DIRECT_RECORD"),
            "rows": source_replay,
            "direct_input_certified": input_certified,
        },
        "current_tip_cumulative_rhs": {
            "label": (
                "production-step operands; isolated source-order JIT "
                "accumulator"),
            "source_order": list(ROUND117_BOUNDARIES),
            "rows": cumulative_rows,
            "incremental_residual_maxima": incremental_residual,
            "first_non_bit": cumulative_first,
            "live_total_closure": cumulative_closure,
            "oracle_after_adv_equals_after_zad": oracle_after_adv_identity,
            "prediction_confirmed": p2_confirmed,
        },
        "record_directed_magnitude": magnitude,
        "production_association_walk": association_walk,
        "current_tip_zad_operand_walk": zad_operand_walk,
        "candidate_eligible": False,
        "candidate_reason": (
            "the production association is attributed, but it is a last-bit "
            "effect below the measured W-owned ZAD magnitude; the next "
            "candidate must start at W's current-tip producer"
            if args.round119 and association_walk is not None
            and association_walk["prediction_confirmed"] else
            ("direct producer inputs are certified, but the isolated cumulative "
             "association does not close bit-for-bit to the live production RHS"
            if direct is not None else
            "diagnostic only; incoming kt2 Ue_rhs/Ve_rhs is not directly "
            "recorded and the cumulative LDF output lacks a complete direct-"
            "input source-exactness proof")),
        "prediction_confirmed": prediction_confirmed,
        "plant": args.plant,
        "plant_fires": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    round_group = parser.add_mutually_exclusive_group()
    round_group.add_argument(
        "--round117", action="store_true",
        help="run the proxy-era current-tip production producer split")
    round_group.add_argument(
        "--round118", action="store_true",
        help="run the admitted direct pre-loop producer split")
    round_group.add_argument(
        "--round119", action="store_true",
        help="run the full-step stage-1 accumulator-association discriminator")
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--round64-root", type=Path, default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path, default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--record-root", type=Path, default=ROOT / "round81/oracle_btstep_kt2")
    parser.add_argument("--uamid-root", type=Path, default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument("--admission", type=Path,
                        default=ROOT / "round81/oracle_btstep_kt2/round81_admission.json")
    parser.add_argument("--entry-root", type=Path,
                        default=ROOT / "round75/oracle_advmean_kt2")
    parser.add_argument(
        "--preloop-root", type=Path,
        default=ROOT / "round117/oracle_preloop_forcing")
    parser.add_argument(
        "--preloop-admission", type=Path,
        default=(ROOT / "round117/oracle_preloop_forcing"
                 / "round117_admission.json"))
    parser.add_argument(
        "--expect-preloop-record-commit", default=ROUND117_PRODUCER)
    parser.add_argument(
        "--execution-mode",
        choices=("production-jit", "production-eager"),
        default="production-jit")
    parser.add_argument("--plant", choices=(
                            "none", "e3-ulp", "rhs-ulp", "final-ulp",
                            "incoming-ulp", "association-hpg-ulp"),
                        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = (
            measure_round117(args)
            if args.round117 or args.round118 or args.round119 else measure(args))
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError, KeyError, ValueError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        prefix = (
            "ROUND119" if args.round119 else
            ("ROUND118" if args.round118 else
             ("ROUND117" if args.round117 else "ROUND83")))
        state = "STATUS PLANT-FIRED" if report["plant_fires"] else "STATUS PLANT-INERT"
        print(f"{prefix} {args.plant.upper()} {state}")
        return 1
    if args.round117 or args.round118 or args.round119:
        prefix = (
            "ROUND119" if args.round119 else
            ("ROUND118" if args.round118 else "ROUND117"))
        print(
            prefix + " PRODUCER " + report["status"] + ": first="
            + repr(report["producer_split"]["first_direct_non_bit"])
        )
        return 0 if report["status"] in ("CONFIRMED", "MEASURED") else 1
    print(f"ROUND83 SLOW FORCING {report['status']}: first={report['first_non_bit_statement']}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
