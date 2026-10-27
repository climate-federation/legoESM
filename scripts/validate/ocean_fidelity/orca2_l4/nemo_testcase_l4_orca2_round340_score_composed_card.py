#!/usr/bin/env python3
"""Score the round-340 real ORCA2 K0 composed card against admitted S4 twins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as round339,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round340_standalone_gate as gate,
)

DEFAULT_ARM_A = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round340/s4_acquisition/orca2_s4_exact_a_np2"
)
DEFAULT_ARM_B = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round340/s4_acquisition/orca2_s4_exact_b_np2"
)
DEFAULT_DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0"
)
DEFAULT_CORE_MANIFEST = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round5/acquisition/orca1ice_surface_entry_every_step_a_np2/input_files.sha256"
)
DEFAULT_ICE_COVER = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round339/k0_smoke/orca2_k0_10step_np2/ice_cover_clim.nc"
)


def explicit_k0_config() -> bulk.NemoStandaloneK0Config:
    """Return the explicit Decision-175 K0 selector tuple."""
    config = bulk.NemoStandaloneK0Config(
        nn_ice=1,
        ln_drgice_imp=False,
        ln_mxl0=True,
        nn_mxlice=0,
        ln_icebergs=False,
        initial_state="nemo_k0_istate",
    )
    bulk.validate_nemo_standalone_k0(config)
    return config


def score_reproducibility_control(arm_a: Path, arm_b: Path) -> dict[str, object]:
    """Score admitted arm B against arm A through score_composed_card (must be 0 unequal)."""
    arm_a = Path(arm_a)
    arm_b = Path(arm_b)
    try:
        manifest_a = round339.verify_manifest(arm_a / "producer_manifest.json")
        manifest_b = round339.verify_manifest(arm_b / "producer_manifest.json")
    except round339.GateError as exc:
        raise gate.GateError(str(exc)) from exc
    inventory_a = gate.inspect_acquisition_records(arm_a)
    inventory_b = gate.inspect_acquisition_records(arm_b)
    gate.require(
        inventory_a == inventory_b,
        f"twin record inventories differ: {inventory_a} vs {inventory_b}",
    )
    surface_b = gate.load_oracle_boundaries(arm_b)
    rung10_b = gate.load_oracle_state_ladder(arm_b)
    scored = gate.score_composed_card(surface_b, arm_a, rung10_b)
    surface_unequal = int(scored["surface"]["unequal"])
    surface_count = int(scored["surface"]["count"])
    rung10_unequal = sum(
        int(row["unequal"]) for row in scored["rung10"].values()
    )
    rung10_count = sum(
        int(row["count"]) for row in scored["rung10"].values()
    )
    gate.require(
        surface_unequal == 0 and rung10_unequal == 0,
        f"reproducibility control has unequal words: "
        f"surface={surface_unequal}/{surface_count}, "
        f"rung10={rung10_unequal}/{rung10_count}",
    )
    return {
        "status": "PASS_REPRODUCIBILITY_EXACT",
        "arm_a": str(arm_a),
        "arm_b": str(arm_b),
        "producer_commit_a": manifest_a["producer_commit"],
        "producer_commit_b": manifest_b["producer_commit"],
        "binary_sha256_a": manifest_a["binary_sha256"],
        "binary_sha256_b": manifest_b["binary_sha256"],
        "inventory": inventory_a,
        "surface_unequal": surface_unequal,
        "surface_count": surface_count,
        "rung10_unequal": rung10_unequal,
        "rung10_count": rung10_count,
        "rung10_rows": len(scored["rung10"]),
    }


def _summarize_surface_by_family(
    by_boundary: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate surface boundaries across steps 1..10 by boundary family."""
    families: dict[str, dict[str, object]] = {}
    for boundary, stats in by_boundary.items():
        parts = boundary.split("/")
        family = parts[1]
        entry = families.setdefault(family, {
            "unequal": 0,
            "count": 0,
            "max_abs": 0.0,
            "first_non_bit_row": None,
        })
        entry["unequal"] = int(entry["unequal"]) + int(stats["unequal"])
        entry["count"] = int(entry["count"]) + int(stats["count"])
        entry["max_abs"] = max(float(entry["max_abs"]), float(stats["max_abs"]))
        if stats["first_non_bit_row"] is not None and entry["first_non_bit_row"] is None:
            entry["first_non_bit_row"] = stats["first_non_bit_row"]
    return families


def summarize_composed_score(scored: Mapping[str, object]) -> dict[str, object]:
    """Build a compact receipt-ready summary from a score_composed_card report."""
    surface = scored["surface"]
    surface_rows: Mapping[str, Mapping[str, object]] = surface["rows"]
    first_nonzero_surface_row = next(
        (key for key, row in surface_rows.items() if float(row["max_abs"]) > 0.0),
        None,
    )
    rung10_rows: Mapping[str, Mapping[str, object]] = scored["rung10"]
    first_non_bit_rung10 = next(
        (key for key, row in rung10_rows.items() if int(row["unequal"]) > 0),
        None,
    )
    first_over_bar_rung10 = next(
        (
            key
            for key, row in rung10_rows.items()
            if row["classification"] == "OVER_BAR"
        ),
        None,
    )
    classifications = {"BIT_EXACT": 0, "AT_BAR_NOT_EXACT": 0, "OVER_BAR": 0}
    rung10_by_step: dict[str, dict[str, object]] = {}
    for key, row in rung10_rows.items():
        cls = str(row["classification"])
        classifications[cls] = classifications.get(cls, 0) + 1
        step = key.split("/", 1)[0]
        step_entry = rung10_by_step.setdefault(step, {
            "rows": 0,
            "unequal": 0,
            "count": 0,
            "max_abs": 0.0,
            "max_normalized_max_abs": 0.0,
            "BIT_EXACT": 0,
            "AT_BAR_NOT_EXACT": 0,
            "OVER_BAR": 0,
            "first_non_bit_row": None,
            "first_over_bar_row": None,
        })
        step_entry["rows"] = int(step_entry["rows"]) + 1
        step_entry["unequal"] = int(step_entry["unequal"]) + int(row["unequal"])
        step_entry["count"] = int(step_entry["count"]) + int(row["count"])
        step_entry["max_abs"] = max(
            float(step_entry["max_abs"]), float(row["max_abs"])
        )
        step_entry["max_normalized_max_abs"] = max(
            float(step_entry["max_normalized_max_abs"]),
            float(row["normalized_max_abs"]),
        )
        step_entry[cls] = int(step_entry[cls]) + 1
        if int(row["unequal"]) > 0 and step_entry["first_non_bit_row"] is None:
            step_entry["first_non_bit_row"] = key
        if cls == "OVER_BAR" and step_entry["first_over_bar_row"] is None:
            step_entry["first_over_bar_row"] = key
    return {
        "status": scored["status"],
        "surface": {
            "status": surface["status"],
            "unequal": int(surface["unequal"]),
            "count": int(surface["count"]),
            "max_abs": float(surface["max_abs"]),
            "first_non_bit_row": surface["first_non_bit_row"],
            "first_nonzero_max_abs_row": first_nonzero_surface_row,
            "by_step": surface["by_step"],
            "by_family": _summarize_surface_by_family(surface["by_boundary"]),
            "by_boundary": surface["by_boundary"],
            "rows": dict(surface_rows),
        },
        "rung10": {
            "total_rows": len(rung10_rows),
            "unequal": sum(int(r["unequal"]) for r in rung10_rows.values()),
            "count": sum(int(r["count"]) for r in rung10_rows.values()),
            "max_abs": max(
                (float(r["max_abs"]) for r in rung10_rows.values()), default=0.0
            ),
            "max_normalized_max_abs": max(
                (float(r["normalized_max_abs"]) for r in rung10_rows.values()),
                default=0.0,
            ),
            "classifications": classifications,
            "first_non_bit_row": first_non_bit_rung10,
            "first_over_bar_row": first_over_bar_rung10,
            "by_step": rung10_by_step,
            "rows": dict(rung10_rows),
        },
        "helper_provenance": scored["helper_provenance"],
    }


def run_and_score_real_composed_card(
    *,
    arm_a: Path = DEFAULT_ARM_A,
    arm_b: Path = DEFAULT_ARM_B,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER,
    model=None,
    strict: bool = True,
) -> dict[str, object]:
    """Run reproducibility control + 10-step real K0 composed card and score against arm A."""
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    reproducibility = score_reproducibility_control(arm_a, arm_b)
    k0_config = explicit_k0_config()
    card = bulk.build_nemo_orca2_k0_ocean_card(Path(deck_root))._replace(
        k0=k0_config
    )
    bulk._validate_k0_ocean_card(card)
    controls = gate.build_real_composed_inputs(
        card, Path(deck_root), Path(core_manifest), Path(ice_cover_path)
    )
    replay = gate.run_composed_card(
        card, card.ocean.recipe.initial_state, controls, model=model
    )
    failure: gate.GateError | None = None
    try:
        scored = gate.score_composed_card(
            replay["surface_actual"], Path(arm_a), replay["rung10"]
        )
    except gate.GateError as exc:
        if exc.report is None:
            raise
        scored = exc.report
        failure = exc
    summary = summarize_composed_score(scored)
    report = {
        "status": scored["status"],
        "failure_message": str(failure) if failure is not None else None,
        "k0_config": k0_config._asdict(),
        "reproducibility_control": reproducibility,
        "composed_card": summary,
    }
    if strict and failure is not None:
        raise gate.GateError(str(failure), report=report) from failure
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, default=DEFAULT_ARM_A)
    parser.add_argument("--arm-b", type=Path, default=DEFAULT_ARM_B)
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument(
        "--core-manifest", type=Path, default=DEFAULT_CORE_MANIFEST
    )
    parser.add_argument("--ice-cover", type=Path, default=DEFAULT_ICE_COVER)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the full score report and exit 0 even if score_composed_card fails.",
    )
    args = parser.parse_args(argv)

    failure: gate.GateError | None = None
    try:
        report = run_and_score_real_composed_card(
            arm_a=args.arm_a,
            arm_b=args.arm_b,
            deck_root=args.deck_root,
            core_manifest=args.core_manifest,
            ice_cover_path=args.ice_cover,
            strict=True,
        )
    except gate.GateError as exc:
        if exc.report is None:
            raise
        report = exc.report
        failure = exc

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    repro = report["reproducibility_control"]
    comp = report["composed_card"]
    surf = comp["surface"]
    r10 = comp["rung10"]
    print(
        f"REPRODUCIBILITY_CONTROL: {repro['status']} "
        f"surface={repro['surface_unequal']}/{repro['surface_count']} "
        f"rung10={repro['rung10_unequal']}/{repro['rung10_count']}"
    )
    print(
        f"COMPOSED_CARD_STATUS: {comp['status']} "
        f"surface={surf['unequal']}/{surf['count']} (max_abs={surf['max_abs']:.17g}) "
        f"first_non_bit={surf['first_non_bit_row']} "
        f"first_nonzero_max_abs={surf['first_nonzero_max_abs_row']}"
    )
    print(
        f"RUNG10_SUMMARY: classifications={r10['classifications']} "
        f"unequal={r10['unequal']}/{r10['count']} "
        f"max_abs={r10['max_abs']:.17g} "
        f"first_non_bit={r10['first_non_bit_row']} "
        f"first_over_bar={r10['first_over_bar_row']}"
    )
    if failure is not None and not args.report_only:
        print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
