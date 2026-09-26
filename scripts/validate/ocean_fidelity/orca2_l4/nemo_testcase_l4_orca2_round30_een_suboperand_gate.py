#!/usr/bin/env python3
"""Fail-closed gate for round 30's EEN denominator sub-operand split."""

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


VARIANTS = {
    "parent": (),
    "e3f0vor": ("e3f0vor",),
    "r3f": ("r3f",),
    "fe3mask": ("fe3mask",),
    "all": ("e3f0vor", "r3f", "fe3mask"),
}


def capture(deck_root: Path, record_root: Path, variant: str,
            npz_out: Path) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import vertical
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    require(variant in VARIANTS, f"unknown variant {variant!r}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 30 is CPU-only")
    require(not bool(jax.config.jax_disable_jit),
            "production stage exposure requires JIT")
    require("bridge_operands" in inspect.signature(
        vertical.nemo_qco_live_vorticity_e3f_cgrid).parameters,
        "checked-out source has no round-30 operand seam")

    _, card = ladder.card_fields(deck_root)
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "ORCA2 card carries no EEN operands")
    derived_mask = vertical.nemo_fe3mask_from_tmask(
        jnp.asarray(card.recipe.z_coord.is_active, dtype=jnp.float64),
        grid=card.recipe.grid)
    mask_score = r27.array_score(derived_mask, raw.fe3mask)

    calls: list[dict[str, object]] = []
    original_flux = pe.pv_flux_al81_partial_cell
    original_builder = vertical.nemo_qco_live_vorticity_e3f_cgrid

    def selected_builder(*args, **kwargs):
        require("bridge_operands" not in kwargs,
                "caller unexpectedly supplies bridge_operands")
        return original_builder(
            *args, **kwargs, bridge_operands=VARIANTS[variant])

    def observed(
        zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask,
        f_vtx=None, eps_h=1.0e-10, q_boundary="neumann_fill",
        metric_widths=None,
    ):
        result = original_flux(
            zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask,
            f_vtx=f_vtx, eps_h=eps_h, q_boundary=q_boundary,
            metric_widths=metric_widths)

        def save(denominator, out_u, out_v):
            calls.append({
                "denominator": np.asarray(denominator, np.float64).copy(),
                "output_u": np.asarray(out_u, np.float64).copy(),
                "output_v": np.asarray(out_v, np.float64).copy(),
            })

        jax.debug.callback(save, h_vtx, result[0], result[1], ordered=True)
        return result

    vertical.nemo_qco_live_vorticity_e3f_cgrid = selected_builder
    pe.pv_flux_al81_partial_cell = observed
    try:
        exposed_u, exposed_v = r27._stage2_vorticity(deck_root, record_root)
    finally:
        pe.pv_flux_al81_partial_cell = original_flux
        vertical.nemo_qco_live_vorticity_e3f_cgrid = original_builder

    require(calls, "production EEN observer captured no calls")
    arrays: dict[str, np.ndarray] = {
        "exposed_u": exposed_u,
        "exposed_v": exposed_v,
    }
    rows = []
    for index, call in enumerate(calls):
        for name, value in call.items():
            arrays[f"call{index}_{name}"] = value
        rows.append({
            "index": index,
            "denominator_digest": _digest(call["denominator"]),
            "output_digest": (
                _digest(call["output_u"]) + ":" + _digest(call["output_v"])),
        })
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_out, **arrays)
    return {
        "status": "CAPTURED",
        "claim_label": "independent with Decision-52 SSH",
        "variant": variant,
        "selected_operands": list(VARIANTS[variant]),
        "backend": jax.default_backend(),
        "dtype": "float64",
        "record_root": str(record_root),
        "worktree": worktree_stamp(),
        "call_count": len(calls),
        "calls": rows,
        "fe3mask_carried_vs_reconstructed": mask_score,
        "npz": str(npz_out),
        "npz_sha256": hashlib.sha256(npz_out.read_bytes()).hexdigest(),
        "citations": {
            "e3f0vor": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937",
            "r3f": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:233-246",
            "fe3mask": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90:258",
            "consumer": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:733-738",
        },
    }


def evaluate(captures: dict[str, dict], arrays: dict[str, dict],
             prior_raw: dict, *, plant: bool = False) -> dict:
    for name in VARIANTS:
        row = captures[name]
        require(row["variant"] == name, f"capture label mismatch for {name}")
        require(row["selected_operands"] == list(VARIANTS[name]),
                f"selected operands mismatch for {name}")
        require(row["backend"] == "cpu" and row["dtype"] == "float64",
                f"wrong execution policy for {name}")
        require(row["worktree"]["clean"], f"dirty capture for {name}")
        require(row["record_root"] == captures["parent"]["record_root"],
                "record roots differ")
        require(row["call_count"] == captures["parent"]["call_count"],
                "EEN call counts differ")

    parent_a = arrays["parent"]
    first = 0
    component_scores = {}
    for name in ("e3f0vor", "r3f", "fe3mask", "all"):
        component_scores[name] = {
            "denominator": r27.array_score(
                arrays[name][f"call{first}_denominator"],
                parent_a[f"call{first}_denominator"]),
            "output_u": r27.array_score(
                arrays[name][f"call{first}_output_u"],
                parent_a[f"call{first}_output_u"]),
            "output_v": r27.array_score(
                arrays[name][f"call{first}_output_v"],
                parent_a[f"call{first}_output_v"]),
            "exposed_u": r27.array_score(
                arrays[name]["exposed_u"], parent_a["exposed_u"]),
            "exposed_v": r27.array_score(
                arrays[name]["exposed_v"], parent_a["exposed_v"]),
        }

    mask_score = captures["parent"]["fe3mask_carried_vs_reconstructed"]
    if plant:
        mask_score = dict(mask_score)
        mask_score["unequal"] = 1
        mask_score["bit_identical"] = False
    require(mask_score["bit_identical"],
            "carried fe3mask differs from reconstructed fe3mask")
    require(component_scores["fe3mask"]["denominator"]["bit_identical"]
            and component_scores["fe3mask"]["output_u"]["bit_identical"]
            and component_scores["fe3mask"]["output_v"]["bit_identical"],
            "fe3mask-only arm is not inert")
    require(not component_scores["e3f0vor"]["output_u"]["bit_identical"],
            "e3f0vor-only arm changes no EEN output")
    r3f_first_output_changed = not (
        component_scores["r3f"]["output_u"]["bit_identical"]
        and component_scores["r3f"]["output_v"]["bit_identical"])

    rank = sorted(
        ("e3f0vor", "r3f", "fe3mask"),
        key=lambda name: (
            component_scores[name]["output_u"]["unequal"]
            + component_scores[name]["output_v"]["unequal"],
            max(component_scores[name]["output_u"]["max_abs"],
                component_scores[name]["output_v"]["max_abs"])),
        reverse=True)
    require(rank[0] == "e3f0vor", "e3f0vor is not the dominant component")

    all_prior_u = r27.array_score(arrays["all"]["exposed_u"],
                                  prior_raw["exposed_u"])
    all_prior_v = r27.array_score(arrays["all"]["exposed_v"],
                                  prior_raw["exposed_v"])
    require(all_prior_u["bit_identical"] and all_prior_v["bit_identical"],
            "three-operand arm does not reproduce round 29")
    require(component_scores["all"]["exposed_u"]["unequal"] == 413554
            and component_scores["all"]["exposed_v"]["unequal"] == 412558,
            "three-operand unequal counts do not reproduce round 29")

    exact_owner = [name for name in ("e3f0vor", "r3f", "fe3mask")
                   if (np.array_equal(arrays[name]["exposed_u"],
                                      arrays["all"]["exposed_u"])
                       and np.array_equal(arrays[name]["exposed_v"],
                                          arrays["all"]["exposed_v"]))]
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "first_call_component_scores": component_scores,
        "component_rank": rank,
        "fe3mask_carried_vs_reconstructed": mask_score,
        "all_vs_round29_raw": {"u": all_prior_u, "v": all_prior_v},
        "single_component_exact_whole_owner": exact_owner,
        "predictions": {
            "R30-P1": "CONFIRMED",
            "R30-P2": ("CONFIRMED" if r3f_first_output_changed
                       else "REFUTED"),
            "R30-P3": "REFUTED" if plant else "CONFIRMED",
            "R30-P4": "CONFIRMED",
            "R30-P5": "REFUTED" if plant else "CONFIRMED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--deck-root", type=Path, required=True)
    cap.add_argument("--record-root", type=Path, required=True)
    cap.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    cap.add_argument("--npz-out", type=Path, required=True)
    cap.add_argument("--json-out", type=Path, required=True)
    outcome = sub.add_parser("outcome")
    for name in VARIANTS:
        outcome.add_argument(f"--{name}-json", type=Path, required=True)
        outcome.add_argument(f"--{name}-npz", type=Path, required=True)
    outcome.add_argument("--prior-raw-npz", type=Path, required=True)
    outcome.add_argument("--json-out", type=Path)
    outcome.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args.deck_root, args.record_root, args.variant,
                             args.npz_out)
        else:
            captures = {name: _read_json(getattr(args, f"{name}_json"))
                        for name in VARIANTS}
            arrays = {name: _read_npz(getattr(args, f"{name}_npz"))
                      for name in VARIANTS}
            result = evaluate(captures, arrays,
                              _read_npz(args.prior_raw_npz), plant=args.plant)
    except (GateError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if getattr(args, "json_out", None):
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 2 if result["status"] == "HELD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
