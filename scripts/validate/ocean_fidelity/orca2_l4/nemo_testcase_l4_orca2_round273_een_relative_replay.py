#!/usr/bin/env python3
"""Calibrate OMT-4's fused stage-2 EEN replay before a leaf split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round266_rhs_operator_gate as r266,
    nemo_testcase_l4_orca2_round272_een_leaf_split as r272,
)
from nemo_testcase_l2_gyre_phase3_gate import read_stage2_terms  # noqa: E402

FLOOR = np.float64(2.0e-10)
LABELS = r249.LABELS
PLANTS = ("none", "replay-bit", "relative-bypass")


class GateError(RuntimeError):
    """The frozen round-273 replay contract moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str, npz_out: Path,
            *, plant: str = "none") -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_een_ene_vertex_coriolis,
        nemo_vor_ene_vorticity_cgrid,
    )

    require(label in LABELS, f"bad label {label!r}")
    require(plant in PLANTS, f"bad plant {plant!r}")
    terms = read_stage2_terms(
        record_root / "oracle_rkstage2_terms_kt00000001.bin",
        expected_dims=(94, 152, 31))
    oracle_after = np.asarray(terms["after_vorticity_u"][..., :30], np.float64)

    captured: dict[str, object] = {}

    def capture(card, parts, state, passivity, vertex_mask):
        captured.update(
            card=card, parts=parts, state=state, passivity=passivity,
            vertex_mask=vertex_mask)

    baseline = r266.measure(
        deck_root, frame_root, record_root, label, expect_commit,
        passive_trace_callback=capture)
    require(set(captured) == {
        "card", "parts", "state", "passivity", "vertex_mask"},
        "passive stage-2 handoff is incomplete")
    require(all(captured["passivity"].values()),
            "certified passive trace moved the completed state")
    card = captured["card"]
    parts = captured["parts"]
    grid = card.recipe.grid
    u_mask, v_mask = pe.compute_face_masks_3d(
        card.recipe.z_coord.is_active, grid)
    operands = {
        "u": np.asarray(parts["operand_velocity_u"]),
        "v": np.asarray(parts["operand_velocity_v"]),
        "h_u": np.asarray(parts["operand_momentum_h_u"]),
        "h_v": np.asarray(parts["operand_momentum_h_v"]),
        "h_vtx": np.asarray(parts["operand_vorticity_h_vtx"]),
        "u_mask": np.asarray(u_mask, bool),
        "v_mask": np.asarray(v_mask, bool),
        "vtx_mask": np.asarray(captured["vertex_mask"], bool),
        "f_vtx": np.asarray(nemo_een_ene_vertex_coriolis(grid)),
    }
    live_u = np.asarray(parts["vorticity_u"].data)
    hpg_u = r266._native_u(parts["after_hpg_u"].data)
    after_u = r266._native_u(parts["after_vor_u"].data)

    def fused(u, v, h_u, h_v, h_vtx, um, vm, fm, *, relative=True):
        zeta = (nemo_vor_ene_vorticity_cgrid(u, v, grid)
                if relative else jnp.zeros_like(h_vtx))
        h_v_f, v_f, h_u_f, u_f, um_f, vm_f = pe._een_fold_image_v_row(
            card.recipe.model_config.een_fold_image_v_row, grid,
            h_v, v, h_u, u, um, vm)
        return pe.pv_flux_al81_partial_cell(
            zeta, h_vtx, h_v_f, v_f, h_u_f, u_f, um_f, vm_f,
            jnp.asarray(operands["vtx_mask"]), f_vtx=fm,
            q_boundary="nemo_live",
            metric_widths=(grid.dx_u, grid.dx_v, grid.dy_u, grid.dy_v))

    fused_replay = jax.jit(fused, static_argnames=("relative",))
    replay = jax.device_get(fused_replay(
        *map(jnp.asarray, (
            operands["u"], operands["v"], operands["h_u"], operands["h_v"],
            operands["h_vtx"], operands["u_mask"], operands["v_mask"],
            operands["f_vtx"])), relative=(plant != "relative-bypass")))
    replay_u = np.asarray(replay[0]) * operands["u_mask"]
    replay_reference = live_u.copy()
    if plant == "replay-bit":
        wet = np.flatnonzero(operands["u_mask"].ravel())
        require(wet.size > 0, "one-bit plant has no active U face")
        replay_reference.view(np.uint64).ravel()[int(wet[0])] ^= np.uint64(1)

    replay_row = _score(replay_u, replay_reference)
    accumulator_row = _score(hpg_u + r266._native_u(replay_u), after_u)
    boundary_row = _score(after_u, oracle_after)
    require(boundary_row["argmax"] == [19, 33, 3],
            f"landed after-VOR argmax moved: {boundary_row['argmax']}")
    require(abs(boundary_row["max_abs"] - 1.2863307536456779e-8) <= 1.0e-20,
            f"landed after-VOR maximum moved: {boundary_row['max_abs']}")
    require(replay_row["max_abs"] <= FLOOR,
            f"fused candidate replay is above the floor: {replay_row}")
    require(accumulator_row["max_abs"] <= FLOOR,
            f"fused accumulator replay is above the floor: {accumulator_row}")
    if plant != "none":
        raise GateError(f"{plant} plant stayed green")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out, live_u=live_u, replay_u=replay_u, after_u=after_u,
        oracle_after_u=oracle_after, **operands)
    return {
        "format": "nemo-testcase-l4-orca2-round273-een-replay-v1",
        "status": "PASS_R273_FUSED_REPLAY_CALIBRATED",
        "label": label,
        "passivity": captured["passivity"],
        "baseline_after_vorticity_u": boundary_row,
        "candidate_replay": replay_row,
        "accumulator_replay": accumulator_row,
        "operand_digests": {name: _digest(value) for name, value in operands.items()},
        "npz": str(npz_out),
        "npz_sha256": hashlib.sha256(npz_out.read_bytes()).hexdigest(),
        "compiled_source": {
            "relative_vorticity": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770",
            "transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:782-786",
            "triads_and_trend": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:788-803",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: bool = False):
    reports = copy.deepcopy(reports)
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both labels are required")
    if plant:
        by_label["independent"]["candidate_replay"]["max_abs"] = 1.0
    rows = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R273_FUSED_REPLAY_CALIBRATED",
                f"{label}: status moved")
        require(report["candidate_replay"]["max_abs"] <= FLOOR,
                f"{label}: fused candidate replay moved above the floor")
        require(report["accumulator_replay"]["max_abs"] <= FLOOR,
                f"{label}: fused accumulator replay moved above the floor")
        require(all(report["passivity"].values()), f"{label}: trace moved state")
        rows[label] = {
            "baseline_after_vorticity_u": report["baseline_after_vorticity_u"],
            "candidate_replay": report["candidate_replay"],
            "accumulator_replay": report["accumulator_replay"],
        }
    require(rows["independent"] == rows["given_nemo_entry"],
            "labels disagree on fused EEN calibration")
    require(not plant, "classification plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round273-classification-v1",
        "status": "PASS_R273_FUSED_REPLAY_CALIBRATED",
        "labels": rows,
        "first_non_floor_leaf": None,
        "leaf_verdict": "UNMEASURED_REQUIRES_NEMO_LEAF_STREAM",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--npz-out", type=Path)
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--classification-plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit, args.npz_out)),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.record_root, args.label,
                args.expect_commit, args.npz_out, plant=args.plant)
        else:
            require(args.classify is not None, "classification reports required")
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify], plant=args.classification_plant)
    except (GateError, r249.GateError, r266.GateError, r272.GateError,
            OSError, KeyError, TypeError, ValueError, IndexError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
