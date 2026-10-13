#!/usr/bin/env python3
"""Capture and split OMT-4's matched stage-2 EEN statement."""

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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round266_rhs_operator_gate as r266,
)
from nemo_testcase_l2_gyre_phase3_gate import read_stage2_terms  # noqa: E402

FLOOR = np.float64(2.0e-10)
LABELS = r249.LABELS
PLANTS = ("none", "topology", "replay-bit", "inactive-mask")


class GateError(RuntimeError):
    """The frozen round-272 measurement contract moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value, np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _topology_rows(residual, u_mask, v_mask, *, count=100):
    """Classify the largest native-U residuals by the live EEN stencil mask."""

    residual = np.asarray(residual, np.float64)
    u_mask = np.asarray(u_mask, bool)
    v_mask = np.asarray(v_mask, bool)
    require(residual.shape == (148, 90, 30), f"residual shape moved: {residual.shape}")
    require(u_mask.shape == (148, 181, 30), f"U mask shape moved: {u_mask.shape}")
    require(v_mask.shape == (149, 180, 30), f"V mask shape moved: {v_mask.shape}")
    flat_order = np.argsort(np.abs(residual).ravel())[::-1][:count]
    rows = []
    coastal_count = 0
    for flat in flat_order:
        j, local_i, k = map(int, np.unravel_index(int(flat), residual.shape))
        # Native rank-0 U column local_i maps to redundant lego column local_i+1.
        i = local_i + 1
        iw = (i - 1) % 180
        ie = i % 180
        # These are the four V transports read directly by NEMO's U-trend
        # triads.  The neighbouring U faces cover the circulation bracket at
        # the two adjacent F points.  Any closed face makes this a coastal
        # stencil; no land-mask inference is used.
        v_open = (
            v_mask[j, iw, k], v_mask[j, ie, k],
            v_mask[j + 1, iw, k], v_mask[j + 1, ie, k],
        )
        u_open = (u_mask[j, i - 1, k], u_mask[j, i, k], u_mask[j, i + 1, k])
        coastal = not (all(v_open) and all(u_open))
        coastal_count += int(coastal)
        rows.append({
            "j_i_k": [j, local_i, k],
            "abs_residual": float(abs(residual[j, local_i, k])),
            "coastal": coastal,
            "v_open": list(map(bool, v_open)),
            "u_open": list(map(bool, u_open)),
        })
    return {
        "definition": "any closed face in four V-triad transports or three adjacent U circulation faces",
        "count": count,
        "coastal": coastal_count,
        "open": count - coastal_count,
        "rows": rows,
    }


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str, npz_out: Path,
            *, plant: str = "none") -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_vor_ene_vorticity_cgrid,
    )

    require(label in LABELS, f"bad label {label!r}")
    require(plant in PLANTS, f"bad plant {plant!r}")
    terms_path = record_root / "oracle_rkstage2_terms_kt00000001.bin"
    terms = read_stage2_terms(terms_path, expected_dims=(94, 152, 31))
    oracle_rows = {
        name: np.asarray(terms[name][..., :30], np.float64)
        for name in r266.BOUNDARY_ORDER
    }

    calls: list[dict[str, np.ndarray]] = []
    boundary_arrays: dict[str, np.ndarray] = {}
    original_flux = pe.pv_flux_al81_partial_cell
    original_score = r266._score

    def save_call(*values):
        names = (
            "zeta", "h_vtx", "h_v", "v", "h_u", "u", "u_mask",
            "v_mask", "vtx_mask", "f_vtx", "output_u", "output_v",
        )
        calls.append({
            name: np.asarray(value, np.float64).copy()
            for name, value in zip(names, values, strict=True)
        })

    def observed(zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d,
                 vtx_mask, f_vtx=None, eps_h=1.0e-10,
                 q_boundary="neumann_fill", metric_widths=None):
        output = original_flux(
            zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d,
            vtx_mask, f_vtx=f_vtx, eps_h=eps_h,
            q_boundary=q_boundary, metric_widths=metric_widths)
        require(f_vtx is not None, "executing EEN call lost ff_f")
        jax.debug.callback(
            save_call, zeta, h_vtx, h_v, v, h_u, u, u_mask_3d,
            v_mask_3d, vtx_mask, f_vtx, output[0], output[1], ordered=True)
        return output

    def scored(left, right):
        left_array = np.asarray(left, np.float64)
        right_array = np.asarray(right, np.float64)
        if left_array.shape == (148, 90, 30):
            for name, oracle in oracle_rows.items():
                if np.array_equal(right_array, oracle):
                    boundary_arrays[name] = left_array.copy()
        return original_score(left, right)

    pe.pv_flux_al81_partial_cell = observed
    r266._score = scored
    try:
        baseline = r266.measure(
            deck_root, frame_root, record_root, label, expect_commit)
    finally:
        pe.pv_flux_al81_partial_cell = original_flux
        r266._score = original_score

    require(set(boundary_arrays) == set(r266.BOUNDARY_ORDER),
            f"boundary capture incomplete: {sorted(boundary_arrays)}")
    require(calls, "EEN observer captured no calls")
    hpg = boundary_arrays["after_hpg_u"]
    after = boundary_arrays["after_vorticity_u"]

    # Locate the exact stage-2 call by replaying each captured EEN output into
    # the captured HPG accumulator.  This does not trust callback order.
    ranked = []
    for index, call in enumerate(calls):
        native = r266._native_u(call["output_u"])
        row = _score(hpg + native, after)
        ranked.append((row["max_abs"], row["unequal"], index, row))
    ranked.sort()
    _, _, target_index, target_replay = ranked[0]
    require(target_replay["max_abs"] <= FLOOR,
            "no captured EEN call reproduces the live stage-2 accumulator")
    call = calls[target_index]

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    current_replay = jax.device_get(jax.jit(original_flux)(
        *map(jnp.asarray, (
            call["zeta"], call["h_vtx"], call["h_v"], call["v"],
            call["h_u"], call["u"], call["u_mask"], call["v_mask"],
            call["vtx_mask"])),
        f_vtx=jnp.asarray(call["f_vtx"]), q_boundary="nemo_live",
        metric_widths=(
            jnp.asarray(card.recipe.grid.dx_u),
            jnp.asarray(card.recipe.grid.dx_v),
            jnp.asarray(card.recipe.grid.dy_u),
            jnp.asarray(card.recipe.grid.dy_v),
        )))
    current_replay_row = _score(current_replay[0], call["output_u"])
    require(current_replay_row["max_abs"] <= FLOOR,
            "candidate-side source replay does not reproduce the live output")

    literal_zeta = np.asarray(jax.device_get(jax.jit(
        lambda uu, vv: nemo_vor_ene_vorticity_cgrid(
            uu, vv, card.recipe.grid))(
                jnp.asarray(call["u"]), jnp.asarray(call["v"]))))
    if plant == "replay-bit":
        literal_zeta = literal_zeta.copy()
        literal_zeta.view(np.uint64).flat[0] ^= np.uint64(1)
    literal_output = jax.device_get(jax.jit(original_flux)(
        jnp.asarray(literal_zeta),
        *map(jnp.asarray, (
            call["h_vtx"], call["h_v"], call["v"], call["h_u"],
            call["u"], call["u_mask"], call["v_mask"], call["vtx_mask"])),
        f_vtx=jnp.asarray(call["f_vtx"]), q_boundary="nemo_live",
        metric_widths=(
            jnp.asarray(card.recipe.grid.dx_u),
            jnp.asarray(card.recipe.grid.dx_v),
            jnp.asarray(card.recipe.grid.dy_u),
            jnp.asarray(card.recipe.grid.dy_v),
        )))

    zeta_row = _score(call["zeta"], literal_zeta)
    literal_output_row = _score(literal_output[0], call["output_u"])
    literal_boundary_row = _score(
        hpg + r266._native_u(literal_output[0]),
        oracle_rows["after_vorticity_u"])
    residual = after - oracle_rows["after_vorticity_u"]
    topology = _topology_rows(
        residual, call["u_mask"], call["v_mask"])
    if plant == "topology":
        topology["coastal"] = -1
    if plant == "inactive-mask":
        baseline = copy.deepcopy(baseline)
        baseline["compiled_source"]["ln_dynvor_msk"] = True

    require(baseline["boundary_rows"]["after_vorticity_u"]["complete"][
                "argmax"] == [19, 33, 3], "round-271 argmax moved")
    require(abs(baseline["boundary_rows"]["after_vorticity_u"]["complete"][
                    "max_abs"] - 1.2863307536453391e-8) <= 1.0e-20,
            "round-271 maximum moved")
    require(0 <= topology["coastal"] <= topology["count"],
            "topology plant fired")
    require(not baseline.get("compiled_source", {}).get("ln_dynvor_msk", False),
            "inactive-mask plant fired")
    require(current_replay_row["unequal"] == 0,
            "candidate replay is not bit-exact")
    if plant != "none":
        raise GateError(f"{plant} plant fired")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out,
        residual=residual,
        candidate_zeta=call["zeta"],
        literal_zeta=literal_zeta,
        candidate_vorticity_u=call["output_u"],
        literal_vorticity_u=np.asarray(literal_output[0]),
        u_mask=call["u_mask"],
        v_mask=call["v_mask"],
    )
    return {
        "format": "nemo-testcase-l4-orca2-round272-een-leaf-v1",
        "status": "PASS_R272_EEN_LEAF_MEASUREMENT",
        "label": label,
        "baseline": baseline,
        "capture": {
            "call_count": len(calls),
            "target_index": target_index,
            "target_accumulator_replay": target_replay,
            "candidate_source_replay": current_replay_row,
        },
        "topology": topology,
        "leaf_order": ["ff_f", "relative_circulation"],
        "ff_f": {"status": "same production operand"},
        "relative_circulation": {
            "candidate_vs_nemo_literal": zeta_row,
            "output_vs_candidate_output": literal_output_row,
            "after_vorticity_vs_nemo": literal_boundary_row,
        },
        "first_non_floor_leaf": (
            "relative_circulation" if zeta_row["max_abs"] > FLOOR else None),
        "npz": str(npz_out),
        "npz_sha256": hashlib.sha256(npz_out.read_bytes()).hexdigest(),
        "compiled_source": {
            "combined_vorticity": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770",
            "transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:782-786",
            "triads_and_trend": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:788-803",
            "mask_branch": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:768-771",
            "ln_dynvor_msk": False,
        },
    }


def classify(reports: list[dict[str, object]], *, plant: bool = False):
    reports = copy.deepcopy(reports)
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both labels are required")
    if plant:
        by_label["independent"]["capture"]["candidate_source_replay"]["unequal"] = 1
    rows = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R272_EEN_LEAF_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["capture"]["candidate_source_replay"]["unequal"] == 0,
                f"{label}: candidate replay moved")
        require(report["compiled_source"]["ln_dynvor_msk"] is False,
                f"{label}: dead mask branch became live")
        rows[label] = {
            "topology": report["topology"],
            "relative_circulation": report["relative_circulation"],
            "first_non_floor_leaf": report["first_non_floor_leaf"],
        }
    require(rows["independent"] == rows["given_nemo_entry"],
            "labels disagree on the matched EEN split")
    leaf = rows["independent"]["first_non_floor_leaf"]
    return {
        "format": "nemo-testcase-l4-orca2-round272-classification-v1",
        "status": "PASS_R272_EEN_LEAF_CLASSIFIED",
        "labels": rows,
        "first_non_floor_leaf": leaf,
        "mask_branch_disposition": "DEAD_ln_dynvor_msk_FALSE",
        "predictions": {
            "R272-P1": "CONFIRMED",
            "R272-P2": "MEASURED",
            "R272-P3": "CONFIRMED",
            "R272-P4": "MEASURED" if leaf else "NOT_REACHED",
            "R272-P5": "CONFIRMED",
        },
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
                 for path in args.classify],
                plant=args.classification_plant)
            require(not args.classification_plant,
                    "classification plant stayed green")
    except (GateError, r249.GateError, r266.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError) as error:
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
