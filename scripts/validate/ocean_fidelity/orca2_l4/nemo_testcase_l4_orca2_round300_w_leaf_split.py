#!/usr/bin/env python3
"""Split OMT-4's kt=7 stage-1 W builder in compiled source order."""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round298_stage7_writer as r298,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round299_w_acquisition import (  # noqa: E402,E501
    check_record as admission_gate,
)

LABELS = r298.LABELS
FLOOR = r298.FLOOR
SOURCE_ORDER = (
    "bottom_zero", "u_difference", "v_difference", "horizontal_sum",
    "transport_divergence", "live_thickness", "flux_divergence",
    "ratio_delta", "stretch_rate", "w_recurrence", "area_product",
)
CUMULATIVE_ARMS = (
    ("trp_u",),
    ("trp_u", "trp_v"),
    ("trp_u", "trp_v", "w_r1area"),
    ("trp_u", "trp_v", "w_r1area", "w_e3t0", "w_r3mm", "w_tmask"),
    ("trp_u", "trp_v", "w_r1area", "w_e3t0", "w_r3mm", "w_tmask",
     "w_r3bb", "w_r3aa", "w_dt"),
    ("trp_u", "trp_v", "w_r1area", "w_e3t0", "w_r3mm", "w_tmask",
     "w_r3bb", "w_r3aa", "w_dt", "w_area"),
)
PLANTS = (
    "none", "admission", "label-coverage", "source-order", "passivity",
    "candidate-replay", "oracle-replay", "first-leaf", "one-bit",
)
_PROGRAM_JIT = None


class GateError(RuntimeError):
    """The admitted record, offline replay, or frozen classifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read_record(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read values only after the admission parser has validated the header."""
    meta = admission_gate.read_record(path)
    raw = path.read_bytes()
    offset = 92
    values = {}
    for expected in admission_gate.NAMES:
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        require(name == expected, f"{path.name}: field order moved")
        count = n1 * n2 * n3
        local = np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset).reshape(
                (n1, n2, n3), order="F")
        offset += count * 8
        ntsi, ntsj, ntei, ntej = meta["owned"]
        owned = np.array(
            local[ntsi - 1:ntei, ntsj - 1:ntej], copy=True).transpose(1, 0, 2)
        values[name] = owned[..., 0] if ndim == 2 else owned
    require(offset == len(raw), f"{path.name}: payload was not consumed")
    return meta, values


def assemble_record(root: Path, baseline: Path) -> tuple[dict[str, np.ndarray], dict]:
    admission = admission_gate.run(root, baseline)
    arrays: dict[str, np.ndarray] = {}
    coverage = np.zeros((148, 180), dtype=np.int8)
    for rank in (0, 1):
        path = root / f"oracle_r299_w_rank{rank:04d}_kt00000007.bin"
        meta, values = _read_record(path)
        nimpp, njmpp = meta["origin"]
        ntsi, ntsj, ntei, ntej = meta["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"rank {rank}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, block in values.items():
            shape = ((148, 180) if block.ndim == 2
                     else (148, 180, block.shape[-1]))
            arrays.setdefault(name, np.empty(shape, np.float64))[
                j0:j1, i0:i1, ...] = block
    require(bool(np.all(coverage == 1)), "rank coverage is not exactly once")
    require(tuple(arrays) == admission_gate.NAMES,
            "assembled field order moved")
    return arrays, admission


def _program(inputs: dict[str, object]) -> dict[str, np.ndarray]:
    """Run the already-shared literal divergence and QCO recurrence helpers."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_qco_wzv_recurrence,
        nemo_transport_wzv_divergence_level,
    )

    global _PROGRAM_JIT
    names = ("trp_u", "trp_v", "w_r1area", "w_area", "w_e3t0",
             "w_r3bb", "w_r3mm", "w_r3aa", "w_tmask", "w_dt")
    values = tuple(jnp.asarray(inputs[name], dtype=jnp.float64) for name in names)

    def execute(pu, pv, r1area, area, e3t0, r3bb, r3mm, r3aa, tmask, dt):
        west = jnp.roll(pu, 1, axis=1)
        south = jnp.concatenate([jnp.zeros_like(pv[:1]), pv[:-1]], axis=0)
        u_difference = sr(pu - west)
        v_difference = sr(pv - south)
        horizontal_sum = sr(u_difference + v_difference)
        transport_divergence = sr(horizontal_sum * r1area[..., None]) * tmask
        live_thickness = sr(e3t0 * sr(
            jnp.asarray(1.0, e3t0.dtype) + sr(r3mm[..., None] * tmask)))
        levels = []
        for level in range(e3t0.shape[-1]):
            levels.append(nemo_transport_wzv_divergence_level(
                pu[..., level], west[..., level],
                pv[..., level], south[..., level], r1area,
                live_thickness[..., level], tmask[..., level]))
        flux_divergence = jnp.stack(levels, axis=-1)
        ratio_delta = sr(r3aa - r3bb)
        r1_dt = sr(jnp.asarray(1.0, e3t0.dtype) / dt)
        stretch_rate = sr(sr(r1_dt * e3t0) * ratio_delta[..., None])
        ww = nemo_qco_wzv_recurrence(
            flux_divergence, e3t0, r3bb, r3aa, tmask,
            dt)
        area_product = sr(area[..., None] * ww)
        bottom_zero = ww[..., -1]
        return (bottom_zero, u_difference, v_difference, horizontal_sum,
                transport_divergence, live_thickness, flux_divergence,
                ratio_delta, stretch_rate, ww, area_product)

    if _PROGRAM_JIT is None:
        _PROGRAM_JIT = jax.jit(execute)
    result = jax.device_get(_PROGRAM_JIT(*values))
    return {name: np.asarray(value, np.float64)
            for name, value in zip(SOURCE_ORDER, result)}


def _candidate_context(deck_root: Path, frame_root: Path, label: str):
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_qco_resolved_mesh_operands,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d

    require(label in LABELS, f"unknown label {label}")
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = omt4.rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else omt4.rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True))
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True))
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        for _ in range(6):
            state = jax.device_get(ordinary_model.step(
                state, card.dt_s, freshwater=forcing[0],
                surface_forcing=forcing[1]))
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0],
            surface_forcing=forcing[1]))
        trace = jax.device_get(traced_model.step(
            state, card.dt_s, freshwater=forcing[0],
            surface_forcing=forcing[1]))
    require(isinstance(trace, _NEMOWSLiveOperandTrace),
            "live trace return type moved")
    passivity = passive._ordinary_state_equal(trace.state_after, ordinary)
    geometry = trace.stage_geometry[0]
    u_mask, v_mask = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    ops = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid, u_mask, v_mask,
        jnp.float64, card.recipe.z_coord.n_levels)
    def ratio_from_eta(eta, e3t0, tmask):
        h0 = jnp.zeros_like(eta)
        for level in range(e3t0.shape[-1]):
            h0 = jax.lax.optimization_barrier(
                h0 + e3t0[..., level] * tmask[..., level])
        reciprocal = jax.lax.optimization_barrier(
            1.0 / jnp.where(h0 > 0.0, h0, 1.0))
        return jax.lax.optimization_barrier(eta * reciprocal)

    tmask = jnp.asarray(card.recipe.z_coord.is_active, jnp.float64)
    e3t0 = jnp.asarray(ops.e3t_0, jnp.float64)
    r3bb = ratio_from_eta(jnp.asarray(state.eta.data), e3t0, tmask)
    r3aa = ratio_from_eta(
        jnp.asarray(trace.barotropic_targets[4]), e3t0, tmask)
    candidate = {
        "trp_u": np.asarray(geometry[7][:, 1:], np.float64),
        "trp_v": np.asarray(geometry[8][1:], np.float64),
        "w_r1area": np.asarray(sr(1.0 / ops.area_t), np.float64),
        "w_area": np.asarray(ops.area_t, np.float64),
        "w_e3t0": np.asarray(ops.e3t_0, np.float64),
        "w_r3bb": np.asarray(r3bb, np.float64),
        "w_r3mm": np.asarray(r3bb, np.float64),
        "w_r3aa": np.asarray(r3aa, np.float64),
        "w_tmask": np.asarray(tmask, np.float64),
        "w_dt": np.asarray(card.dt_s, np.float64),
    }
    native = {
        "ww": np.asarray(geometry[2], np.float64),
        "trp_w": np.asarray(sr(
            jnp.asarray(ops.area_t)[..., None] * geometry[2]), np.float64),
    }
    return card, passivity, candidate, native


def _oracle_inputs(oracle: dict[str, np.ndarray], dt_s: float) -> dict[str, np.ndarray]:
    """NEMO's stage-1 W operands on the candidate's 30 active layers."""
    inputs = {
        name: (oracle[name][..., :30]
               if name in {"trp_u", "trp_v", "w_e3t0", "w_tmask"}
               else oracle[name])
        for name in ("trp_u", "trp_v", "w_r1area", "w_area", "w_e3t0",
                     "w_r3bb", "w_r3mm", "w_r3aa", "w_tmask")
    }
    # Stage-1 clock rn_Dt/3 (stprk3_stg.f90:136-137).
    inputs["w_dt"] = np.asarray(dt_s / 3.0, np.float64)
    return inputs


def time_matched_ratio_delta(oracle_delta, candidate_dt, oracle_dt):
    """Put NEMO's stage-1 Kaa-Kbb ratio delta on the candidate's step.

    NEMO's stage-1 r3t(Kaa) is 2/3 r3t(Kbb) + 1/3 r3ta (np_HYB,
    stprk3_stg.f90:51,179), so its delta spans rn_Dt/3; the candidate's
    delta spans the full step to ssh(N+1).  Comparing the raw deltas
    compares different time levels.
    """
    return np.asarray(oracle_delta) * (float(candidate_dt) / float(oracle_dt))


def _support(name: str, oracle: dict[str, np.ndarray]) -> np.ndarray:
    tmask = np.asarray(oracle["w_tmask"], bool)
    if name in {"bottom_zero", "ratio_delta"}:
        return np.asarray(oracle["w_tmask"][..., 0], bool)
    if name in {"w_recurrence", "area_product"}:
        return tmask
    return tmask[..., :30]


def _measure_label(deck_root: Path, frame_root: Path,
                   oracle: dict[str, np.ndarray], label: str) -> dict[str, object]:
    card, passivity, candidate, native = _candidate_context(
        deck_root, frame_root, label)
    candidate_program = _program(candidate)
    oracle_inputs = _oracle_inputs(oracle, card.dt_s)
    oracle_program = _program(oracle_inputs)
    oracle_program["ratio_delta"] = time_matched_ratio_delta(
        oracle_program["ratio_delta"], candidate["w_dt"], oracle_inputs["w_dt"])
    rows = {
        name: r298._regions(candidate_program[name], oracle_program[name],
                            _support(name, oracle))
        for name in SOURCE_ORDER
    }
    candidate_replay = {
        "ww": r298._regions(candidate_program["w_recurrence"], native["ww"],
                            np.asarray(oracle["w_tmask"], bool)),
        # ``trp_w`` is an area-scaled representation of the same W field.
        # Score its replay in W-equivalent units so a sub-floor W rounding
        # difference is not amplified by the O(1e10 m2) ORCA2 cell area.
        # The raw area-product statement remains separately reported in
        # ``rows`` below and is never hidden by this calibration check.
        "trp_w_w_equivalent": r298._regions(
            candidate_program["area_product"] / candidate["w_area"][..., None],
            native["trp_w"] / candidate["w_area"][..., None],
            np.asarray(oracle["w_tmask"], bool)),
    }
    oracle_replay = {
        "ww": r298._regions(oracle_program["w_recurrence"], oracle["w_ww"],
                            np.asarray(oracle["w_tmask"], bool)),
        "trp_w_w_equivalent": r298._regions(
            oracle_program["area_product"] / oracle_inputs["w_area"][..., None],
            oracle["trp_w"] / oracle_inputs["w_area"][..., None],
            np.asarray(oracle["w_tmask"], bool)),
    }
    arms = {}
    for names in CUMULATIVE_ARMS:
        inputs = dict(candidate)
        inputs.update({name: oracle_inputs[name] for name in names})
        program = _program(inputs)
        arms["+".join(names)] = r298._regions(
            program["w_recurrence"], oracle["w_ww"],
            np.asarray(oracle["w_tmask"], bool))
    return {
        "status": "PASS_R300_LABEL_MEASUREMENT", "label": label,
        "passivity": passivity, "rows": rows,
        "candidate_replay": candidate_replay,
        "oracle_replay": oracle_replay, "cumulative_arms": arms,
        "subnormal_flush_control": r298._score(
            np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
            np.asarray([True])),
        "one_bit_control": r298._score(
            np.asarray([1.0]), np.asarray([np.nextafter(1.0, 2.0)]),
            np.asarray([True])),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    for row in report.get("labels", ()):
        row["subnormal_flush_control"] = r298._score(
            np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
            np.asarray([True]))
        row["one_bit_control"] = r298._score(
            np.asarray([1.0]), np.asarray([np.nextafter(1.0, 2.0)]),
            np.asarray([True]))
    if plant == "admission":
        report["admission"]["status"] = "BROKEN"
    elif plant == "label-coverage":
        report["labels"].pop()
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "passivity":
        report["labels"][0]["passivity"]["S"] = False
    elif plant == "candidate-replay":
        report["labels"][0]["candidate_replay"]["ww"]["complete"]["max_abs"] = 1.0
    elif plant == "oracle-replay":
        report["labels"][0]["oracle_replay"]["ww"]["complete"]["max_abs"] = 1.0
    elif plant == "first-leaf":
        for item in report["labels"][0]["rows"].values():
            item["complete"]["max_abs"] = 0.0
    elif plant == "one-bit":
        report["labels"][0]["one_bit_control"]["unequal"] = 0

    require(report["admission"]["status"] ==
            "PASS_R299_OMT4_KT7_W_ADMISSION", "admission moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "source order moved")
    by_label = {row["label"]: row for row in report["labels"]}
    require(set(by_label) == set(LABELS), "both labels are required")
    first_by_label = {}
    summaries = {}
    for label in LABELS:
        row = by_label[label]
        require(row["status"] == "PASS_R300_LABEL_MEASUREMENT",
                f"{label}: measurement status moved")
        require(all(row["passivity"].values()), f"{label}: trace moved state")
        require(all(item["complete"]["max_abs"] <= FLOOR
                    for item in row["candidate_replay"].values()),
                f"{label}: candidate replay missed production")
        require(all(item["complete"]["max_abs"] <= FLOOR
                    for item in row["oracle_replay"].values()),
                f"{label}: oracle replay missed its record")
        require(row["one_bit_control"]["unequal"] == 1,
                f"{label}: normal one-bit control did not fire")
        sub = row["subnormal_flush_control"]
        require(sub["raw_unequal"] == 1 and sub["subnormal_matches"] == 1
                and sub["unequal"] == 0,
                f"{label}: Decision-127 control moved")
        nonbit = [name for name in SOURCE_ORDER
                  if row["rows"][name]["complete"]["max_abs"] > FLOOR]
        require(bool(nonbit), f"{label}: first-leaf plant erased every boundary")
        first = nonbit[0]
        first_by_label[label] = first
        closing = [name for name, item in row["cumulative_arms"].items()
                   if item["complete"]["max_abs"] <= FLOOR]
        summaries[label] = {
            "first_nonbit_statement": first,
            "first_row": row["rows"][first]["complete"],
            "closing_arms": closing,
            "first_closing_arm": closing[0] if closing else "none",
        }
    require(len(set(first_by_label.values())) == 1,
            "labels disagree on the first W leaf")
    first = next(iter(first_by_label.values()))
    report["classification"] = {
        "first_nonbit_statement": first, "labels": summaries,
        "prediction_ledger": {
            "R300-P1": "CONFIRMED", "R300-P2": "CONFIRMED",
            "R300-P3": ("CONFIRMED" if first == "u_difference" else "REFUTED"),
            "R300-P4": ("CONFIRMED" if all(
                any("w_r3bb" not in arm and "w_r3aa" not in arm
                    for arm in summary["closing_arms"])
                for summary in summaries.values()) else "REFUTED"),
            "R300-P5": "CONFIRMED", "R300-P6": "CONFIRMED_MEASUREMENT_ONLY",
        },
    }
    report["status"] = f"HELD_R300_FIRST_W_LEAF_{first.upper()}"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str,
            labels: tuple[str, ...] = LABELS) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-300 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 300 requires production JIT on CPU")
    oracle, admission = assemble_record(record_root, baseline_root)
    labels = [_measure_label(deck_root, frame_root, oracle, label)
              for label in labels]
    return {
        "format": "nemo-testcase-l4-orca2-round300-w-leaf-v1",
        "status": "PASS_R300_MEASUREMENT", "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm plus offline W replay",
        "admission": admission, "source_order": list(SOURCE_ORDER),
        "labels": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    # ponytail: one label per call fits the 10-minute synchronous limit;
    # classification still requires both labels' rows in one report.
    parser.add_argument("--label", choices=LABELS, action="append")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.record_root,
                args.baseline_root, args.expect_commit)),
                "measurement paths and --expect-commit are required")
            report = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit,
                             tuple(args.label or LABELS))
            if args.raw_output:
                args.raw_output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
        else:
            require(args.report_in is not None, "--report-in is required")
            report = json.loads(args.report_in.read_text(encoding="utf-8"))
        result = classify(report, args.plant)
    except (GateError, OSError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
