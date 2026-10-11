#!/usr/bin/env python3
"""Split OMT-4's substep-3 midpoint V depth on admitted operands."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round244_inloop_fold_audit as r244,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

SUBSTEPS = (3, 4, 5, 6)
OPERAND_ORDER = (
    "hv_0", "r1_e1e2v", "e1e2t", "eta_mid_owned",
    "north_e1e2t_halo", "north_eta_halo", "ssvmask",
)
CLASSIFY_PLANTS = (
    "none", "label-coverage", "source-order", "one-bit",
    "replay-closure", "production-route", "passivity",
)


class GateError(RuntimeError):
    """The admitted record cannot support the frozen round-245 split."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _difference(left, right) -> dict[str, object]:
    return r228._difference(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _native_v(values) -> np.ndarray:
    return r235._native(values, "v")


def _compact_v(native) -> np.ndarray:
    native = np.asarray(native, np.float64)
    return np.concatenate([np.zeros_like(native[:1]), native], axis=0)


def _raw_surface_vmask(card) -> np.ndarray:
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "ORCA2 card has no raw NEMO geometry bundle")
    mask = np.asarray(raw.vmask, np.float64)
    require(mask.ndim == 3, f"raw vmask rank moved: {mask.shape}")
    return np.max(mask, axis=-1)


def _raw_compact_face_masks(card):
    """Map NEMO native east/north surface masks to compact west/south faces."""

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "ORCA2 card has no raw NEMO geometry bundle")
    native_u = np.max(np.asarray(raw.umask, np.float64), axis=-1)
    native_v = np.max(np.asarray(raw.vmask, np.float64), axis=-1)
    compact_u = np.concatenate([native_u[:, -1:], native_u], axis=1)
    compact_v = _compact_v(native_v)
    return compact_u, compact_v


@contextlib.contextmanager
def _barotropic_surface_mask_arm(card, enabled: bool):
    """Private causal arm: hand NEMO's raw ssumask/ssvmask to dynspg only."""

    if not enabled:
        yield
        return
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    original = model_module.barotropic_substeps_latlon_cgrid
    raw_u, raw_v = _raw_compact_face_masks(card)

    def wrapped(state, *args, **kwargs):
        dtype = state.eta.data.dtype
        masked = state._replace(
            u_mask=state.u_mask.replace(data=jnp.asarray(raw_u, dtype=dtype)),
            v_mask=state.v_mask.replace(data=jnp.asarray(raw_v, dtype=dtype)),
        )
        return original(masked, *args, **kwargs)

    model_module.barotropic_substeps_latlon_cgrid = wrapped
    try:
        yield
    finally:
        model_module.barotropic_substeps_latlon_cgrid = original


def _v_depth_replay(
    eta, hv_0, r1_e1e2v, area_t, ssvmask, grid, *,
    north_area_row: int = -1, north_eta_row: int = -1,
):
    """Replay dynspg_ts.f90:487-519 with independently selectable operands."""

    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.grids.operators_latlon_cgrid import apply_north_fold
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _zero_polar_lat_ends,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        fold_is_local,
        north_fold_mask,
        pad_ns_zero,
    )

    eta = jnp.asarray(eta)
    hv_0 = jnp.asarray(hv_0, dtype=eta.dtype)
    r1_e1e2v = jnp.asarray(r1_e1e2v, dtype=eta.dtype)
    area_t = jnp.asarray(area_t, dtype=eta.dtype)
    ssvmask = jnp.asarray(ssvmask, dtype=eta.dtype)
    south = sr(pad_ns_zero(area_t)[:-1] * pad_ns_zero(eta)[:-1])
    north = sr(pad_ns_zero(area_t)[1:] * pad_ns_zero(eta)[1:])
    summed = sr(south + north)
    average = sr(sr(0.5 * r1_e1e2v) * summed)
    average = _zero_polar_lat_ends(sr(average * ssvmask))

    fold = getattr(grid, "fold", None)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        area_fold = area_t[north_area_row][jnp.newaxis, fold.perm_T]
        eta_fold = eta[north_eta_row][jnp.newaxis, fold.perm_T]
        local = sr(area_t[-1:] * eta[-1:])
        remote = sr(area_fold * eta_fold)
        fold_average = sr(sr(0.5 * r1_e1e2v[-1:]) * sr(local + remote))
        fold_average = sr(fold_average * ssvmask[-1:])
        average = apply_north_fold(average, fold_average, grid, north_mask=nmask)
    return sr(hv_0 + average)


def _one_bit_control() -> dict[str, object]:
    value = np.asarray([1.0], np.float64)
    return _difference(value, np.nextafter(value, np.inf))


def measure(
    deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
    label: str, expect_commit: str, *, raw_surface_masks: bool = False,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_reference_face_depths,
        _nemo_ssh_avg_prep,
        _nemo_ssh_avg_reference_depth_override,
    )

    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-245 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-245 split requires production JIT on CPU")

    admission = r235._admit_streams(twin_a, twin_b)
    oracle_substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    oracle_advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    card = r244.omt4.build_omt4_card(deck_root)
    with _barotropic_surface_mask_arm(card, raw_surface_masks):
        card, trace, passivity = r244._trace_run(deck_root, frame_root, label)

    state = card.recipe.initial_state
    dtype = jnp.asarray(state.eta.data).dtype
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    area = jnp.asarray(grid.area, dtype=dtype)
    land = jnp.asarray(state.land_mask.data, dtype=dtype)
    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data, dtype=dtype), land, grid, dtype)
    prep = _nemo_ssh_avg_reference_depth_override(
        prep, _nemo_literal_reference_face_depths(z_coord, dtype), dtype)
    hv_candidate = prep[1]
    r1_area_candidate = prep[3]
    raw_compact_u, raw_compact_v = _raw_compact_face_masks(card)
    vmask_candidate = jnp.asarray(
        raw_compact_v if raw_surface_masks else state.v_mask.data, dtype=dtype)

    raw_hv = np.asarray(z_coord.nemo_hv_0, np.float64)
    raw_area_t = np.asarray(z_coord.nemo_e1e2t, np.float64)
    raw_area_v = np.asarray(z_coord.nemo_e1e2v, np.float64)
    raw_r1_area_v = np.divide(
        1.0, raw_area_v, out=np.zeros_like(raw_area_v), where=raw_area_v > 0.0)
    raw_vmask = _raw_surface_vmask(card)
    r1_area_raw_compact = _compact_v(raw_r1_area_v)
    vmask_raw_compact = _compact_v(raw_vmask)

    operand_rows = {
        "hv_0": _difference(_native_v(hv_candidate), raw_hv[:, :90]),
        "r1_e1e2v": _difference(
            _native_v(r1_area_candidate), raw_r1_area_v[:, :90]),
        "e1e2t": _difference(np.asarray(area)[:, :90], raw_area_t[:, :90]),
        "ssvmask": _difference(_native_v(vmask_candidate), raw_vmask[:, :90]),
    }

    rows = []
    for substep in SUBSTEPS:
        index = substep - 1
        eta = jnp.asarray(trace["eta_mid"][index], dtype=dtype)
        oracle_eta = np.asarray(oracle_substeps["eta_mid"][index], np.float64)
        oracle_depth = np.asarray(oracle_advmean["face_depth_v"][index], np.float64)
        production_depth = _native_v(trace["transport_face_depth_v"][index])

        values = dict(
            eta=eta, hv_0=hv_candidate, r1_e1e2v=r1_area_candidate,
            area_t=area, ssvmask=vmask_candidate, grid=grid,
            north_area_row=-1, north_eta_row=-1)
        baseline = _v_depth_replay(**values)
        variants = {}
        cumulative = dict(values)
        changes = (
            ("hv_0", {"hv_0": _compact_v(raw_hv)}),
            ("r1_e1e2v", {"r1_e1e2v": r1_area_raw_compact}),
            ("e1e2t", {"area_t": raw_area_t}),
            ("eta_mid_owned", {}),
            ("north_e1e2t_halo", {"north_area_row": -2}),
            ("north_eta_halo", {"north_eta_row": -2}),
            ("ssvmask", {"ssvmask": vmask_raw_compact}),
        )
        for name, update in changes:
            cumulative.update(update)
            variants[name] = _difference(
                _native_v(_v_depth_replay(**cumulative)), oracle_depth)
        rows.append({
            "substep": substep,
            "eta_mid_owned": _difference(np.asarray(eta)[:, :90], oracle_eta),
            "baseline_replay_vs_production": _difference(
                _native_v(baseline), production_depth),
            "baseline_vs_nemo": _difference(production_depth, oracle_depth),
            "pair": {
                "v_mid": r244._regional(
                    r244._native_full(trace["transport_velocity_v"][index], "V")[:, :90],
                    np.asarray(oracle_advmean["velocity_v"][index], np.float64),
                    support=np.asarray(oracle_advmean["r1_e1v"] != 0.0)),
                "face_depth_v_mid": r244._regional(
                    production_depth, oracle_depth,
                    support=np.asarray(oracle_advmean["r1_e1v"] != 0.0)),
                "metric_transport_v": r244._regional(
                    r244._native_full(trace["transport_metric_v"][index], "V")[:, :90],
                    np.asarray(oracle_advmean["metric_v"][index], np.float64),
                    support=np.asarray(oracle_advmean["r1_e1v"] != 0.0)),
            },
            "cumulative_substitution": variants,
        })

    return {
        "format": "nemo-testcase-l4-orca2-round245-v-depth-split-v1",
        "status": "PASS_R245_V_DEPTH_OPERAND_SPLIT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "source": {
            "compiled_file": (
                "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90"),
            "lines": "487-519",
            "statement": (
                "zhvp2_e=hv_0+0.5*r1_e1e2v*"
                "(e1e2t(j)*zsshp2_e(j)+e1e2t(j+1)*zsshp2_e(j+1))*ssvmask"),
        },
        "production_operand_route": (
            "D114 raw hv_0; raw NEMO ssumask/ssvmask causal arm"
            if raw_surface_masks
            else "D114 raw hv_0; current compact area/mask"),
        "raw_surface_masks": raw_surface_masks,
        "record_admission": admission,
        "passivity": passivity,
        "substeps": list(SUBSTEPS),
        "operand_order": list(OPERAND_ORDER),
        "operand_rows": operand_rows,
        "rows": rows,
        "one_bit_control": _one_bit_control(),
    }


def classify(reports: list[dict], *, plant: str = "none") -> dict[str, object]:
    require(plant in CLASSIFY_PLANTS, f"unknown classify plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "both claim labels are required")
    independent = by_label["independent"]
    if plant == "source-order":
        independent["operand_order"][0], independent["operand_order"][1] = (
            independent["operand_order"][1], independent["operand_order"][0])
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0
    elif plant == "replay-closure":
        independent["rows"][0]["baseline_replay_vs_production"]["unequal"] = 1
    elif plant == "production-route":
        independent["production_operand_route"] = "reconstructed"
    elif plant == "passivity":
        independent["passivity"]["eta"] = False

    summaries = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R245_V_DEPTH_OPERAND_SPLIT",
                f"{label}: measurement status moved")
        require(tuple(report["substeps"]) == SUBSTEPS,
                f"{label}: substep order moved")
        require(tuple(report["operand_order"]) == OPERAND_ORDER,
                f"{label}: operand order moved")
        require(report["production_operand_route"].startswith("D114 raw"),
                f"{label}: stale/non-production mesh route")
        require(all(report["passivity"].values()),
                f"{label}: observer moved completed state")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        for row in report["rows"]:
            require(row["baseline_replay_vs_production"]["unequal"] == 0,
                    f"{label}: offline depth replay missed production")
        summaries[label] = {
            "operand_rows": report["operand_rows"],
            "rows": report["rows"],
        }

    def signature(summary):
        return [
            (row["substep"], row["baseline_vs_nemo"]["unequal"],
             tuple((name, result["unequal"])
                   for name, result in row["cumulative_substitution"].items()))
            for row in summary["rows"]
        ]

    require(signature(summaries["independent"])
            == signature(summaries["given_nemo_entry"]),
            "claim labels disagree on V-depth attribution")
    first = summaries["independent"]
    first_operand = next(
        (name for name in OPERAND_ORDER
         if (first["operand_rows"].get(name, {}).get("unequal", 0) > 0
             or (name in ("north_e1e2t_halo", "north_eta_halo")
                 and first["rows"][0]["cumulative_substitution"][name]["unequal"]
                 < first["rows"][0]["baseline_vs_nemo"]["unequal"]))),
        None)
    return {
        "format": "nemo-testcase-l4-orca2-round245-classification-v1",
        "status": "HELD_R245_V_DEPTH_OPERAND_NAMED",
        "source": independent["source"],
        "first_effective_operand": first_operand,
        "rows": summaries,
        "predictions": {
            "R245-P1": "CONFIRMED_PASSIVE_ADMITTED_FP64_BOTH_LABELS",
            "R245-P2": (
                "CONFIRMED_HV_AND_SSVMASK_EXACT"
                if (first["operand_rows"]["hv_0"]["unequal"] == 0
                    and first["operand_rows"]["ssvmask"]["unequal"] == 0)
                else "REFUTED_HV_OR_SSVMASK_NONBIT"),
            "R245-P3": f"MEASURED_FIRST_{first_operand}",
            "R245-P4": "MEASURED_CUMULATIVE_DEPTH_REPLAY",
            "R245-P5": "MEASUREMENT_ONLY_PENDING_ATOMIC_PAIR_SCORE",
            "R245-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--raw-surface-masks", action="store_true")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--classify-plant", choices=CLASSIFY_PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify],
                plant=args.classify_plant)
        else:
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)),
                "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit,
                raw_surface_masks=args.raw_surface_masks)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        planted = args.classify is not None and args.classify_plant != "none"
        print(f"STATUS {'PLANT-FIRED' if planted else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
