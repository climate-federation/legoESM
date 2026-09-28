#!/usr/bin/env python3
"""Gate NEMO-source-ordered OVERFLOW UP3 T-face fluxes bit-for-bit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nemo_testcase_l1_overflow_round57_up3_statement_gate as r57_gate
import numpy as np
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _nemo_up3_same_direction_flux,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from nemo_testcase_l1_overflow_round50_pair_gate import GateError, require
from nemo_testcase_phase3_trajectory_gate import expected_masks


def _production_source_flux(
    transport_pos,
    transport_neg,
    far_pos,
    adv_pos,
    adv_neg,
    far_neg,
    recorded_selector,
    mask_pos,
    mask_neg,
):
    computed_selector = adv_pos + adv_neg
    require(
        np.array_equal(
            np.asarray(recorded_selector).view(np.uint64),
            np.asarray(computed_selector).view(np.uint64),
        ),
        "production velocity-pair selector drift",
    )
    return _nemo_up3_same_direction_flux(
        transport_pos,
        transport_neg,
        far_pos,
        adv_pos,
        adv_neg,
        far_neg,
        mask_pos,
        mask_neg,
    )


def run(root: Path, expect_commit: str, plant: str | None) -> dict:
    inherited = r57_gate.run(root, expect_commit, None)
    require(inherited["status"] == "FIRST_NON_BIT_NAMED",
            "round-57 base statement no longer reproduces")
    parsed = r57_gate.r56_gate.read_up3_record(root / r57_gate.RECORD)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    active_u = np.asarray(expected_masks(card)["u"], dtype=bool)
    analysis = r57_gate._analyze_u(
        parsed["fields"],
        active_u,
        tuple(parsed["header"]["origin"]),
        plant=plant,
        production_flux_fn=_production_source_flux,
    )
    rows = {row["name"]: row for row in analysis["rows"]}
    require(
        all(rows[name]["baseline_n_unequal"] == 0 for name in r57_gate.SOURCE_ORDER),
        "source-ordered UP3 statement is not bit-exact",
    )
    flux = rows["u.t_face_flux"]
    if plant == "face_flux":
        require(flux["n_unequal"] == 1,
                "face-flux plant did not add exactly one refusal")
        status = "PLANTED_REFUSAL"
    else:
        require(flux["n_unequal"] == 0, "UP3 T-face flux is not bit-exact")
        status = "AT_BAR"
    return {
        **analysis,
        "status": status,
        "format": "nemo-testcase-l1-overflow-round58-up3-landing-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": inherited["worktree"],
        "precision": inherited["precision"],
        "record_root": inherited["record_root"],
        "record_sha256": inherited["record_sha256"],
        "record_admission": inherited["record_admission"],
        "compiled_source": inherited["compiled_source"],
        "round57_base_n_unequal": 282,
        "R58-P1": "CONFIRMED",
        "R58-P2": "CONFIRMED",
        "R58-P3": "CONFIRMED" if plant == "face_flux" else "PARTIAL",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=r57_gate.RECORD_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("face_flux",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.record_dir, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant else 0
    except (GateError, r57_gate.GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
