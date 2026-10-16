#!/usr/bin/env python3
"""Name OMT-4's kt=7 stage-1 growth carrier in compiled source order."""

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
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round297_stage7_acquisition import (  # noqa: E402,E501
    check_record as admission_gate,
)

LABELS = ("independent", "given_nemo_entry")
SOURCE_ORDER = ("rhs", "pre", "cor", "trp", "adv", "sbc", "qco", "final")
FLOOR = np.float64(2.0e-10)
PLANTS = (
    "none", "admission", "label-coverage", "source-order", "passivity",
    "replay", "unforced", "first-carrier", "one-bit",
)


class GateError(RuntimeError):
    """The admitted record, replay, or frozen classifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read_record(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    schema = admission_gate.schema
    meta = schema.read_record(
        path, expected_kt=7, expected_magic=admission_gate.MAGIC)
    raw = path.read_bytes()
    offset = 92
    values = {}
    for expected in schema.NAMES:
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
        path = root / f"oracle_r297_stage1_rank{rank:04d}_kt00000007.bin"
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
    require(tuple(arrays) == admission_gate.schema.NAMES,
            "assembled field order moved")
    return arrays, admission


def _native(name: str, values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if name.endswith("_u"):
        return values[:, 1:]
    if name.endswith("_v"):
        return values[1:]
    return values


def _score(candidate, oracle, support) -> dict[str, object]:
    candidate = np.ascontiguousarray(candidate, dtype=np.float64)
    oracle = np.ascontiguousarray(oracle, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    require(candidate.shape == oracle.shape == support.shape,
            f"score shape moved: {candidate.shape}, {oracle.shape}, {support.shape}")
    require(bool(np.any(support)), "score support is empty")
    c, o = candidate[support], oracle[support]
    require(bool(np.isfinite(c).all() and np.isfinite(o).all()),
            "score contains non-finite values")
    raw_unequal = c.view(np.uint64) != o.view(np.uint64)
    subnormal = ((c == 0.0) & (o != 0.0)
                 & (np.abs(o) < np.finfo(np.float64).tiny))
    unequal = raw_unequal & ~subnormal
    delta = np.abs(c - o)
    scored_delta = np.where(subnormal, 0.0, delta)
    pos = int(np.argmax(scored_delta))
    flat = np.flatnonzero(support)
    argmax = np.unravel_index(int(flat[pos]), candidate.shape)
    return {
        "support": int(c.size),
        "raw_unequal": int(np.count_nonzero(raw_unequal)),
        "subnormal_matches": int(np.count_nonzero(subnormal)),
        "unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(scored_delta[pos]),
        "rms": float(np.sqrt(np.mean(scored_delta * scored_delta))),
        "argmax": [int(value) for value in argmax],
        "at_floor": bool(float(scored_delta[pos]) <= FLOOR),
    }


def _regions(candidate, oracle, support) -> dict[str, object]:
    return {
        "complete": _score(candidate, oracle, support),
        "interior": _score(candidate[:-3], oracle[:-3], support[:-3]),
        "fold_band": _score(candidate[-3:], oracle[-3:], support[-3:]),
    }


def _oracle_geometry(geometry, oracle, grid):
    p_u = np.concatenate(
        [oracle["trp_u"][..., :30][:, -1:], oracle["trp_u"][..., :30]], axis=1)
    p_v = np.concatenate(
        [oracle["trp_v"][..., :30][-1:], oracle["trp_v"][..., :30]], axis=0)
    p_w = oracle["trp_w"]
    dy = np.asarray(grid.dy_u, np.float64)[..., None]
    dx = np.asarray(grid.dx_v, np.float64)[..., None]
    area = np.asarray(grid.area_T, np.float64)[..., None]
    return (
        p_u / dy, p_v / dx, p_w / area,
        *geometry[3:7], p_u, p_v, *geometry[9:],
    )


def _tracer_replay(card, state, trace, geometry):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_rk3_tracer_pair_step,
    )

    rates, _ = trace.stage_tracer_sources
    ratios = trace.stage_qco
    one = np.float64(1.0)
    q0 = one + ratios[0][0]
    q1 = one + ratios[1][0]
    q2 = one + ratios[2][0]
    weights = ((q0, q0, q1), (q0, q1, q2), (q0, q2, q2))
    result = _nemo_ws_rk3_tracer_pair_step(
        state.T.data, state.S.data, card.recipe.model_config.tracer_advection,
        geometry[0], geometry[1], geometry[2], geometry[3], geometry[3],
        geometry[4], geometry[5], card.recipe.grid, card.dt_s,
        card.recipe.z_coord.is_active,
        stage_transport_geometry=(geometry, geometry, geometry),
        stage_source_rates=rates,
        stage_qco_weights=weights,
        stage_tracer_written_order=(
            card.recipe.model_config.rk3_stage_tracer_written_order),
        stop_after_stage=1, return_stage1_trace=True,
    )
    return tuple(np.asarray(value, np.float64) for value in jax.device_get(result))


def _adv_qco(card, state, trace, adv_t, adv_s):
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as r

    q0 = jnp.asarray(1.0 + trace.stage_qco[0][0])
    q1 = jnp.asarray(1.0 + trace.stage_qco[1][0])
    dt = jnp.asarray(card.dt_s / 3.0, dtype=q0.dtype)
    active = jnp.asarray(card.recipe.z_coord.is_active) > 0.5

    def update(base, rhs):
        if card.recipe.model_config.rk3_stage_tracer_written_order:
            out = r(r(r(q0[..., None] * base)
                      + r(r(dt * q0[..., None]) * rhs)) / q1[..., None])
        else:
            out = (q0[..., None] * base + dt * q0[..., None] * rhs) / q1[..., None]
        return jnp.where(active, out, base)

    return tuple(np.asarray(value, np.float64) for value in jax.device_get(
        jax.jit(lambda t, s: (update(state.T.data, t), update(state.S.data, s)))(
            jnp.asarray(adv_t[..., :30]), jnp.asarray(adv_s[..., :30]))))


def _measure_label(deck_root: Path, frame_root: Path, oracle: dict,
                   label: str, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

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

    active_t = np.asarray(card.recipe.z_coord.is_active, bool)
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 raw mesh operands are absent")
    supports = {
        "u": np.asarray(raw.umask, bool),
        "v": np.asarray(raw.vmask, bool),
        "t": active_t,
        "w": np.ones((148, 180, 31), dtype=bool),
    }
    g0 = trace.stage_geometry[0]
    replay = _tracer_replay(card, state, trace, g0)
    oracle_geometry = _oracle_geometry(g0, oracle, card.recipe.grid)
    transport_replay = _tracer_replay(card, state, trace, oracle_geometry)
    adv_qco = _adv_qco(card, state, trace, oracle["adv_t"], oracle["adv_s"])

    candidate = {
        "rhs_u": _native("rhs_u", trace.stage_rhs[0][0]),
        "rhs_v": _native("rhs_v", trace.stage_rhs[0][1]),
        "pre_u": _native("pre_u", trace.stage_raw_velocities[0][0]),
        "pre_v": _native("pre_v", trace.stage_raw_velocities[0][1]),
        "cor_u": _native("cor_u", trace.stage_outputs[0][0]),
        "cor_v": _native("cor_v", trace.stage_outputs[0][1]),
        "trp_u": _native("trp_u", g0[7]),
        "trp_v": _native("trp_v", g0[8]),
        "trp_w": np.asarray(g0[2], np.float64)
                 * np.asarray(card.recipe.grid.area_T, np.float64)[..., None],
        "adv_t": replay[2], "adv_s": replay[3],
        "sbc_t": replay[4], "sbc_s": replay[5],
        "qco_t": replay[0], "qco_s": replay[1],
        "final_u": _native("final_u", trace.stage_outputs[0][0]),
        "final_v": _native("final_v", trace.stage_outputs[0][1]),
        "final_t": replay[0], "final_s": replay[1],
    }
    field_support = {
        **{name: supports["u"] for name in ("rhs_u", "pre_u", "cor_u", "trp_u", "final_u")},
        **{name: supports["v"] for name in ("rhs_v", "pre_v", "cor_v", "trp_v", "final_v")},
        **{name: supports["t"] for name in ("adv_t", "adv_s", "sbc_t", "sbc_s", "qco_t", "qco_s", "final_t", "final_s")},
        "trp_w": supports["w"],
    }
    rows = {
        name: _regions(value, oracle[name][..., :value.shape[-1]],
                       field_support[name][..., :value.shape[-1]])
        for name, value in candidate.items()
    }
    frame0 = omt4.rung0.assemble_frame(frame_root, 7, 0)
    entry_rows = {
        name: _regions(np.asarray(getattr(state, name).data, np.float64),
                       np.asarray(frame0[name], np.float64), supports["t"])
        for name in ("T", "S")
    }
    replay_checks = {
        "T": _regions(replay[0], np.asarray(trace.stage_outputs[0][2]), supports["t"]),
        "S": _regions(replay[1], np.asarray(trace.stage_outputs[0][3]), supports["t"]),
    }
    transport_arm = {
        "T": _regions(transport_replay[0], oracle["qco_t"][..., :30], supports["t"]),
        "S": _regions(transport_replay[1], oracle["qco_s"][..., :30], supports["t"]),
    }
    adv_arm = {
        "T": _regions(adv_qco[0], oracle["qco_t"][..., :30], supports["t"]),
        "S": _regions(adv_qco[1], oracle["qco_s"][..., :30], supports["t"]),
    }
    one = _score(np.asarray([0.0]),
                 np.asarray([np.nextafter(0.0, 1.0)]), np.asarray([True]))
    return {
        "status": "PASS_R298_LABEL_MEASUREMENT", "label": label,
        "passivity": passivity, "rows": rows, "entry": entry_rows,
        "candidate_replay": replay_checks,
        "oracle_transport_qco_arm": transport_arm,
        "oracle_adv_qco_arm": adv_arm,
        "unforced_checks": {
            "oracle_adv_t_equals_sbc_t": bool(np.array_equal(oracle["adv_t"], oracle["sbc_t"])),
            "oracle_adv_s_equals_sbc_s": bool(np.array_equal(oracle["adv_s"], oracle["sbc_s"])),
            "candidate_adv_t_equals_sbc_t": bool(np.array_equal(replay[2], replay[4])),
            "candidate_adv_s_equals_sbc_s": bool(np.array_equal(replay[3], replay[5])),
        },
        "one_bit_control": one,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["status"] = "BROKEN"
    elif plant == "label-coverage":
        report["labels"].pop()
    elif plant == "source-order":
        report["source_order"][3], report["source_order"][4] = (
            report["source_order"][4], report["source_order"][3])
    elif plant == "passivity":
        report["labels"][0]["passivity"]["T"] = False
    elif plant == "replay":
        report["labels"][0]["candidate_replay"]["S"]["complete"]["max_abs"] = 1.0
    elif plant == "unforced":
        report["labels"][0]["unforced_checks"]["oracle_adv_s_equals_sbc_s"] = False
    elif plant == "first-carrier":
        report["labels"][0]["oracle_transport_qco_arm"]["S"]["fold_band"]["max_abs"] = 0.0
    elif plant == "one-bit":
        report["labels"][0]["one_bit_control"]["unequal"] = 0

    require(report["admission"]["status"] ==
            "PASS_R297_OMT4_KT7_STAGE1_ADMISSION", "admission moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER, "source order moved")
    by_label = {row["label"]: row for row in report["labels"]}
    require(set(by_label) == set(LABELS), "both labels are required")
    carriers = {}
    summaries = {}
    for label in LABELS:
        row = by_label[label]
        require(row["status"] == "PASS_R298_LABEL_MEASUREMENT",
                f"{label}: measurement status moved")
        require(all(row["passivity"].values()), f"{label}: trace moved state")
        require(all(item["complete"]["max_abs"] <= FLOOR
                    for item in row["candidate_replay"].values()),
                f"{label}: offline tracer replay missed production")
        require(all(row["unforced_checks"].values()),
                f"{label}: OMT-4 adv/sbc equality moved")
        require(row["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        prior = row["entry"]["S"]["fold_band"]["max_abs"]
        actual = row["rows"]["qco_s"]["fold_band"]["max_abs"]
        trp = row["oracle_transport_qco_arm"]["S"]["fold_band"]["max_abs"]
        adv = row["oracle_adv_qco_arm"]["S"]["fold_band"]["max_abs"]
        require(prior > FLOOR and actual > 10.0 * prior,
                f"{label}: frozen kt=7 growth did not reproduce")
        if trp <= 10.0 * prior:
            carrier = "trp"
        elif adv <= 10.0 * prior:
            carrier = "adv"
        else:
            carrier = "qco"
        carriers[label] = carrier
        summaries[label] = {
            "entry_fold_S_max": prior,
            "qco_fold_S_max": actual,
            "growth_ratio": actual / prior,
            "oracle_transport_arm_fold_S_max": trp,
            "oracle_adv_arm_fold_S_max": adv,
            "carrier": carrier,
            "rows": row["rows"],
            "entry": row["entry"],
            "candidate_replay": row["candidate_replay"],
            "unforced_checks": row["unforced_checks"],
        }
    require(len(set(carriers.values())) == 1,
            "labels disagree on the first growth carrier")
    first = next(iter(carriers.values()))
    report["classification"] = {
        "first_growth_carrier": first, "labels": summaries,
        "prediction_ledger": {
            "R298-P1": "CONFIRMED", "R298-P2": "CONFIRMED",
            "R298-P3": "CONFIRMED" if first == "adv" else "REFUTED",
            "R298-P4": "CONFIRMED", "R298-P5": "CONFIRMED_MEASUREMENT_ONLY",
        },
    }
    report["status"] = f"HELD_R298_FIRST_GROWTH_CARRIER_{first.upper()}"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-298 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 298 requires production JIT on CPU")
    oracle, admission = assemble_record(record_root, baseline_root)
    labels = [_measure_label(
        deck_root, frame_root, oracle, label, expect_commit) for label in LABELS]
    return {
        "format": "nemo-testcase-l4-orca2-round298-stage7-writer-v1",
        "status": "PASS_R298_MEASUREMENT", "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm plus offline pure tracer replay",
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
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.baseline_root, args.expect_commit)),
                    "measurement arguments are incomplete")
            report = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            report = json.loads(args.report_in.read_text(encoding="utf-8"))
        result = classify(report, args.plant)
    except (GateError, admission_gate.Refusal, admission_gate.schema.Refusal,
            omt4.GateError, OSError, KeyError, TypeError, ValueError,
            IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
