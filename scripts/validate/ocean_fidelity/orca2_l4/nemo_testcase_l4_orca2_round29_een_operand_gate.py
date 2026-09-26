#!/usr/bin/env python3
"""Fail-closed gate for round 29's production EEN operand census."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round27_consumer_gate as r27,
)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _digest(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON capture: {path}")
    return json.loads(path.read_text())


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ capture: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def capture(deck_root: Path, record_root: Path, helper_variant: str,
            npz_out: Path) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import vertical
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(helper_variant in ("parent", "raw_f"),
            "helper variant must be parent or raw_f")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 29 is CPU-only")
    require(not bool(jax.config.jax_disable_jit),
            "production stage exposure requires JIT")

    has_raw_seam = "use_bridge_raw" in inspect.signature(
        vertical.nemo_qco_live_vorticity_e3f_cgrid).parameters
    require(has_raw_seam == (helper_variant == "raw_f"),
            "helper label does not match the checked-out source arm")

    calls: list[dict[str, np.ndarray]] = []
    original = pe.pv_flux_al81_partial_cell

    def observer(zeta, h_vtx, h_v, v, h_u, u, u_mask, v_mask, f_vtx,
                 out_u, out_v):
        calls.append({
            "zeta": np.asarray(zeta, dtype=np.float64).copy(),
            "denominator": np.asarray(h_vtx, dtype=np.float64).copy(),
            "h_v": np.asarray(h_v, dtype=np.float64).copy(),
            "v": np.asarray(v, dtype=np.float64).copy(),
            "h_u": np.asarray(h_u, dtype=np.float64).copy(),
            "u": np.asarray(u, dtype=np.float64).copy(),
            "u_mask": np.asarray(u_mask, dtype=np.float64).copy(),
            "v_mask": np.asarray(v_mask, dtype=np.float64).copy(),
            "f_vtx": np.asarray(f_vtx, dtype=np.float64).copy(),
            "output_u": np.asarray(out_u, dtype=np.float64).copy(),
            "output_v": np.asarray(out_v, dtype=np.float64).copy(),
        })

    def observed(
        zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask,
        f_vtx=None, eps_h=1.0e-10, q_boundary="neumann_fill",
        metric_widths=None,
    ):
        result = original(
            zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask,
            f_vtx=f_vtx, eps_h=eps_h, q_boundary=q_boundary,
            metric_widths=metric_widths)
        require(f_vtx is not None, "ORCA2 EEN call has no vertex Coriolis")
        jax.debug.callback(
            observer, zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d,
            f_vtx, result[0], result[1], ordered=True)
        return result

    pe.pv_flux_al81_partial_cell = observed
    try:
        exposed_u, exposed_v = r27._stage2_vorticity(deck_root, record_root)
    finally:
        pe.pv_flux_al81_partial_cell = original

    require(calls, "production EEN observer captured no calls")
    exposed_digest = _digest(exposed_u) + ":" + _digest(exposed_v)
    call_rows = []
    arrays: dict[str, np.ndarray] = {
        "exposed_u": exposed_u,
        "exposed_v": exposed_v,
    }
    for index, call in enumerate(calls):
        fields = {name: _digest(value) for name, value in call.items()}
        output_digest = fields["output_u"] + ":" + fields["output_v"]
        call_rows.append({
            "index": index,
            "digests": fields,
            "is_exposed_stage2": output_digest == exposed_digest,
        })
        for name in ("denominator", "output_u", "output_v"):
            arrays[f"call{index}_{name}"] = call[name]
    require(any(row["is_exposed_stage2"] for row in call_rows),
            "exposed stage-2 result does not identify an EEN call")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_out, **arrays)
    return {
        "status": "CAPTURED",
        "claim_label": "independent with Decision-52 SSH",
        "helper_variant": helper_variant,
        "backend": jax.default_backend(),
        "dtype": "float64",
        "record_root": str(record_root),
        "worktree": worktree_stamp(),
        "call_count": len(call_rows),
        "calls": call_rows,
        "npz": str(npz_out),
        "npz_sha256": hashlib.sha256(npz_out.read_bytes()).hexdigest(),
        "exposed_digest": exposed_digest,
        "citations": {
            "denominator": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:733-738",
            "numerator": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:741-780",
            "transport": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:782-786",
            "tendency": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:788-803",
            "reference": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:907-937",
        },
    }


def evaluate(parent: dict, raw: dict, parent_arrays: dict,
             raw_arrays: dict, prior_parent: dict, prior_raw: dict,
             *, plant: bool = False) -> dict:
    require(parent["helper_variant"] == "parent", "first capture is not parent")
    require(raw["helper_variant"] == "raw_f", "second capture is not raw-F")
    require(parent["backend"] == raw["backend"] == "cpu",
            "capture backend changed")
    require(parent["dtype"] == raw["dtype"] == "float64",
            "capture dtype changed")
    require(parent["record_root"] == raw["record_root"],
            "capture record roots differ")
    require(parent["call_count"] == raw["call_count"],
            "EEN call counts differ")
    require(parent["worktree"]["clean"] and raw["worktree"]["clean"],
            "a capture worktree is dirty")

    stable_inputs = ("zeta", "h_v", "v", "h_u", "u", "u_mask",
                     "v_mask", "f_vtx")
    changed_outputs = []
    for pcall, rcall in zip(parent["calls"], raw["calls"], strict=True):
        if (pcall["digests"]["output_u"] != rcall["digests"]["output_u"]
                or pcall["digests"]["output_v"] !=
                rcall["digests"]["output_v"]):
            changed_outputs.append(pcall["index"])
    require(changed_outputs, "raw-F arm changes no EEN output")
    first = changed_outputs[0]
    pcall = parent["calls"][first]
    rcall = raw["calls"][first]
    if plant:
        rcall = json.loads(json.dumps(rcall))
        rcall["digests"]["u"] = "PLANTED"
    require(all(pcall["digests"][name] == rcall["digests"][name]
                for name in stable_inputs),
            "first changed EEN call also changes numerator or transport input")
    require(pcall["digests"]["denominator"] !=
            rcall["digests"]["denominator"],
            "first changed EEN call does not change the denominator")

    stage_p = [row["index"] for row in parent["calls"]
               if row["is_exposed_stage2"]]
    stage_r = [row["index"] for row in raw["calls"]
               if row["is_exposed_stage2"]]
    require(stage_p == stage_r, "exposed stage-2 EEN call indices moved")
    p_u = r27.array_score(parent_arrays["exposed_u"],
                          prior_parent["stage2_vorticity_u"])
    p_v = r27.array_score(parent_arrays["exposed_v"],
                          prior_parent["stage2_vorticity_v"])
    r_u = r27.array_score(raw_arrays["exposed_u"],
                          prior_raw["stage2_vorticity_u"])
    r_v = r27.array_score(raw_arrays["exposed_v"],
                          prior_raw["stage2_vorticity_v"])
    require(p_u["bit_identical"] and p_v["bit_identical"],
            "observer changed the parent stage-2 EEN result")
    require(r_u["bit_identical"] and r_v["bit_identical"],
            "denominator arm does not reproduce round 27/28 raw-F EEN")
    moved_u = r27.array_score(raw_arrays["exposed_u"],
                              parent_arrays["exposed_u"])
    moved_v = r27.array_score(raw_arrays["exposed_v"],
                              parent_arrays["exposed_v"])
    require(moved_u["unequal"] == 413554 and moved_v["unequal"] == 412558,
            "stage-2 unequal-cell counts do not reproduce round 27/28")
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "first_changed_call": first,
        "exposed_stage2_calls": stage_p,
        "changed_output_calls": changed_outputs,
        "first_changed_call_inputs": {
            "bit_identical": list(stable_inputs),
            "different": ["denominator"],
        },
        "stage2_parent_vs_prior": {"u": p_u, "v": p_v},
        "stage2_raw_vs_prior": {"u": r_u, "v": r_v},
        "stage2_raw_vs_parent": {"u": moved_u, "v": moved_v},
        "predictions": {
            "R29-P1": "CONFIRMED",
            "R29-P2": "CONFIRMED",
            "R29-P3": "CONFIRMED",
            "R29-P4": "REFUTED" if plant else "CONFIRMED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--deck-root", type=Path, required=True)
    cap.add_argument("--record-root", type=Path, required=True)
    cap.add_argument("--helper-variant", choices=("parent", "raw_f"),
                     required=True)
    cap.add_argument("--npz-out", type=Path, required=True)
    cap.add_argument("--json-out", type=Path, required=True)
    outcome = sub.add_parser("outcome")
    for name in ("parent-json", "parent-npz", "raw-json", "raw-npz",
                 "prior-parent-npz", "prior-raw-npz"):
        outcome.add_argument(f"--{name}", type=Path, required=True)
    outcome.add_argument("--json-out", type=Path)
    outcome.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args.deck_root, args.record_root,
                             args.helper_variant, args.npz_out)
        else:
            result = evaluate(
                _read_json(args.parent_json), _read_json(args.raw_json),
                _read_npz(args.parent_npz), _read_npz(args.raw_npz),
                _read_npz(args.prior_parent_npz),
                _read_npz(args.prior_raw_npz), plant=args.plant)
    except (GateError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 2 if result["status"] == "HELD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
