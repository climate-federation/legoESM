#!/usr/bin/env python3
"""O17 scorer: imported rung gates plus the production composition entry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import jax
from legoesm.ice.si3_o17 import si3_o17_step

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
    admit_evidence as admit_o4,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
    score_deck as score_o4,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o5_orca2_sas_gate import (
    admit_evidence_directory as admit_o5,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o7_orca2_sas_gate import (
    admit_evidence_directory as admit_o7,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    score_o8_to_o12,
    score_o13,
    score_o14,
    score_o15_o16,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_acquisition.preflight import preflight

VALID_LABELS = ("INDEPENDENT", "GIVEN-NEMO-ENTRY")


def _status(report: dict) -> str:
    return str(report.get("status", "MISSING"))


def score(evidence: Path, labels: list[str]) -> dict:
    repo = Path(__file__).resolve().parents[4]
    if labels != list(VALID_LABELS):
        raise ValueError(f"labels must be exactly {VALID_LABELS}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    graph = preflight(repo)
    controls = {}
    for plant in ("swap-rheology-advection", "drop-redistribution"):
        try:
            preflight(repo, plant)
        except SystemExit as exc:
            controls[plant] = f"FIRED: {exc}"
        else:
            raise ValueError(f"O17 composition plant survived: {plant}")
    master = evidence / "components/o8_o16"
    o4_root = evidence / "components/o4"
    o4_admission = admit_o4(o4_root, expect_commit=commit)
    steps = tuple(range(1, 101))
    modes = {}
    for mode in ("eager", "jit"):
        with jax.disable_jit(mode == "eager"):
            o4 = score_o4(o4_root / "record_2step/o4_prather_records", mode=mode)
            o5 = admit_o5(evidence / "components/o5", expect_commit=commit)
            o7 = admit_o7(evidence / "components/o7", expect_commit=commit)
            thermo = score_o8_to_o12(master_root=master, steps=steps, plant=None)
            o13 = score_o13(master_root=master, steps=steps, plant=None)
            o14 = score_o14(master_root=master, steps=steps, plant=None)
            tail = score_o15_o16(master_root=master, steps=steps, plant=None)
        components = {
            "O4": _status(o4),
            "O5": _status(o5),
            "O7": _status(o7),
            "O8-O12": _status(thermo),
            "O13": _status(o13),
            "O14": _status(o14),
            "O15-O16": _status(tail),
        }
        passed = all(
            value.startswith("PASS") or value in {"ADMITTED", "AT_BAR"}
            for value in components.values()
        )
        if not passed:
            raise ValueError(f"O17 {mode} component bar failed: {components}")
        modes[mode] = components
    per_label = {
        label: {
            "status": "ACQUISITION_NEEDED",
            "production_entry": f"{si3_o17_step.__module__}.{si3_o17_step.__name__}",
            "ulp_bar": 0,
            "trajectory_relative_bar": 1.0e-15,
            "steps": list(steps),
            "component_exactness_modes": modes,
            "missing_proof": (
                "the component records do not contain one complete O17 step entry and "
                "live forcing for execution through the common production entry"
            ),
        }
        for label in labels
    }
    return {
        "status": "PASS_COMPONENT_BARS_O17_RECORD_REQUIRED",
        "call_graph": graph["call_order"],
        "composition_controls": controls,
        "o4_admission": _status(o4_admission),
        "labels": per_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--labels", nargs=2, required=True)
    args = parser.parse_args()
    try:
        report = score(args.evidence.resolve(), args.labels)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "REFUSE", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
