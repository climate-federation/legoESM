#!/usr/bin/env python3
"""Locate the certified round-79b ORCA2 kt=10 T/S extrema without substitution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round80_extremes_gate as extremes,
)

PLANTS = ("none", "baseline", "claim-label", "location")


def measure(deck_root: Path, ten_step_root: Path, vmix_root: Path,
            landed_reference: Path) -> dict[str, object]:
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    admission = extremes.vmix.run_gate(vmix_root)
    reference = extremes._reference_row(landed_reference)
    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(ten_step_root, 1, stage=None)
    state = card.recipe.initial_state._replace(
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    last_fields = None
    for kt in range(1, 11):
        surface_fields = ladder.assemble_surface_fields(ten_step_root, kt)
        freshwater, surface = ladder._surface_forcings(
            card, deck_root, surface_fields, kt)
        trace = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
        state = trace.state_after
        last_fields = extremes._global_record(vmix_root, kt)
        print(f"BASELINE_LOCATION_PROGRESS step={kt}/10", flush=True)
    extremes.require(last_fields is not None, "no mixing record was consumed")
    oracle = ladder.read_state_frame(
        ten_step_root / "oracle_stage_kt00000010_s3.bin", kt=10, stage=3)
    candidate = ladder._rank0_fields(ladder._candidate_fields(state))
    comparison = ladder.compare_fields(candidate, oracle)
    expected = {key: reference[key] for key in (
        "rows", "ranked_non_bit_by_max_abs", "first_non_bit_field")}
    extremes.require(comparison == expected,
                     "ordinary trajectory differs from landed round-79b reference")
    locations = {}
    for field in ("T", "S"):
        index = extremes._argmax(candidate[field], oracle[field])
        residual = candidate[field] - oracle[field]
        locations[field] = {
            **extremes._location(index, card, last_fields),
            "signed_residual": float(residual[index]),
            "max_abs": float(abs(residual[index])),
        }
    return {
        "format": "nemo-testcase-l4-orca2-round80-baseline-location-v1",
        "claim_label": "given NEMO's entry",
        "initial_mode": "decision52_ssh_bridge",
        "steps_completed": 10,
        "record_admission": {"status": admission["status"],
                             "producer_commit": admission["producer_commit"]},
        "comparison": comparison,
        "locations": locations,
        "compiled_citations": {
            "process_order": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381",
            "tracer_consumer": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:178-215",
            "momentum_consumer": "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:191-205",
        },
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    extremes.require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "baseline":
        report["comparison"]["rows"]["T"]["max_abs"] += 1.0
    elif plant == "claim-label":
        report["claim_label"] = "independent"
    elif plant == "location":
        report["locations"]["T"]["index_jik"] = [-1, -1, -1]
    extremes.require(report["claim_label"] == "given NEMO's entry",
                     "claim label changed")
    extremes.require(
        report["comparison"]["rows"]["T"]["max_abs"] == 1.2367457128331782,
        "T baseline maximum changed")
    extremes.require(
        report["comparison"]["rows"]["S"]["max_abs"] == 0.2871061346986039,
        "S baseline maximum changed")
    for field in ("T", "S"):
        index = report["locations"][field]["index_jik"]
        extremes.require(len(index) == 3 and 0 <= index[0] < 148
                         and 0 <= index[1] < 90 and 0 <= index[2] < 30,
                         f"{field} argmax index is outside rank 0")
        extremes.require(report["locations"][field]["max_abs"]
                         == report["comparison"]["rows"][field]["max_abs"],
                         f"{field} argmax does not carry the field maximum")
    report["status"] = "PASS_ROUND80_BASELINE_LOCATION"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--ten-step-root", type=Path)
    parser.add_argument("--vmix-root", type=Path)
    parser.add_argument("--landed-reference", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            extremes.require(not any((args.deck_root, args.ten_step_root,
                                      args.vmix_root, args.landed_reference)),
                             "--classify-json cannot be combined with run inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            extremes.require(args.plant == "none", "plants classify existing JSON")
            extremes.require(all((args.deck_root, args.ten_step_root,
                                  args.vmix_root, args.landed_reference)),
                             "run mode requires every record input")
            raw = measure(args.deck_root, args.ten_step_root, args.vmix_root,
                          args.landed_reference)
        result = classify(raw, args.plant)
    except (extremes.GateError, ladder.GateError, extremes.vmix.GateError,
            OSError, ValueError, KeyError, TypeError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND80_BASELINE_LOCATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
