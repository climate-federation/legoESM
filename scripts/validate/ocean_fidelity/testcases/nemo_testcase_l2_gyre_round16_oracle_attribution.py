#!/usr/bin/env python3
"""Attribute every GYRE scalar-math V1->V2 record to a live transcendental.

This is an execution-order census, not a claim that the first differing field
contains the transcendental call itself.  The input byte census identifies the
first differing payload.  This probe then names the earliest live
transcendental ancestor that can reach that payload in the NEMO step:

* ``usrdef_sbc.F90:109-145,161-184``: seasonal COS/SIN forms qsr, qns, emp,
  utau, and vtau before the RK3 stage program;
* ``dynspg_ts.F90:552-572``: surface stress enters the slow forcing consumed
  by the external mode;
* ``stprk3_stg.F90:452-565``: the surface/tracer sources advance stage-1 T/S;
* ``traqsr.F90:621,629-630,642``: two-band EXP advances the stage-3 tracer.

The classifications are deliberately fail-closed.  An unrecognised changed
record, a changed pre-forcing control, or ``--plant`` exits nonzero.  The
``ancestor`` values are causal provenance classes, not fidelity owner labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from legoesm.ocean.fidelity.provenance import worktree_stamp

PREFORCING_CONTROLS = {
    "oracle_bt_ene_coeff_kt00000001.bin",
    "oracle_rhs_kt00000001.bin",
    "oracle_rkstage2_hpg_literal_kt00000001.bin",
    "oracle_rkstage2_hpg_operands_kt00000001.bin",
    "oracle_step_entry_kt00000001.bin",
}


def _ancestor(record: str, first_field: str) -> tuple[str, str]:
    """Return the first live transcendental ancestor and its causal path."""
    if record.startswith("oracle_qsr_stage3_"):
        return (
            "usrdef_sbc COS; traqsr qsr_2BD EXP",
            "qsr forcing -> qsr_2BD attenuation -> tracer Krhs",
        )
    if record.startswith("oracle_zdf_entry_"):
        return (
            "usrdef_sbc SIN/COS",
            "utau/vtau -> surface stress magnitude -> prognostic TKE -> avm/avt",
        )
    if (
        record.startswith("oracle_rktracer_stage3_")
        or record.startswith("oracle_stage_")
        or record.startswith("oracle_step_entry_")
    ) and first_field.startswith(("T", "S", "after_")):
        return (
            "usrdef_sbc SIN/COS; traqsr qsr_2BD EXP",
            "qns/emp/qsr and stage transport -> tracer update -> propagated T/S",
        )
    if record.startswith("oracle_rkstage2_eos_operands_") and first_field in {
        "T",
        "S",
    }:
        return (
            "usrdef_sbc SIN/COS",
            "stage-1 surface/tracer forcing -> stage-2-entry T/S -> EOS",
        )

    momentum_prefixes = (
        "oracle_bt_advmean_operands_",
        "oracle_bt_drag_operands_",
        "oracle_bt_frames_",
        "oracle_bt_substeps_",
        "oracle_rkstage1_transport_operands_",
        "oracle_rkstage2_ene_operands_",
        "oracle_rkstage2_operands_",
        "oracle_rkstage2_preupdate_",
        "oracle_rkstage2_terms_",
        "oracle_rkstage3_wzv_",
        "oracle_rktracer_operands_",
        "oracle_tracer_transport_",
        "oracle_transport_",
    )
    if record.startswith(momentum_prefixes):
        return (
            "usrdef_sbc SIN/COS",
            "utau/vtau -> slow forcing -> external mode -> stage velocity/transport",
        )
    raise AssertionError(
        f"changed record has no registered transcendental ancestor: "
        f"{record} first_field={first_field}"
    )


def run(census: dict[str, object], *, plant: bool) -> dict[str, object]:
    records = list(census["records"])
    if plant:
        records.append(
            {
                "record": "oracle_unregistered_planted_kt00000001.bin",
                "status": "DIFFERENT",
                "first_field": "PLANTED",
            }
        )

    rows = []
    for record in records:
        name = str(record["record"])
        status = str(record["status"])
        if name in PREFORCING_CONTROLS:
            if status != "IDENTICAL":
                raise AssertionError(f"pre-forcing control changed: {name}")
            rows.append(
                {
                    "record": name,
                    "status": status,
                    "first_field": None,
                    "ancestor": "NONE (pre-forcing/source-invariant control)",
                    "path": "byte-identical V1/V2 control",
                }
            )
            continue
        if status != "DIFFERENT":
            raise AssertionError(f"unexpected identical non-control record: {name}")
        first_field = str(record["first_field"])
        ancestor, path = _ancestor(name, first_field)
        rows.append(
            {
                "record": name,
                "status": status,
                "first_field": first_field,
                "ancestor": ancestor,
                "path": path,
            }
        )

    counts = {
        "records": len(rows),
        "different_with_transcendental_ancestor": sum(
            row["status"] == "DIFFERENT" for row in rows
        ),
        "identical_pre_forcing_controls": sum(
            row["status"] == "IDENTICAL" for row in rows
        ),
        "unattributed": 0,
    }
    if counts != {
        "records": 46,
        "different_with_transcendental_ancestor": 41,
        "identical_pre_forcing_controls": 5,
        "unattributed": 0,
    }:
        raise AssertionError(f"unexpected attribution census counts: {counts}")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round16-v1-v2-attribution-v1",
        "plant": plant,
        "counts": counts,
        "source_citations": [
            "usrdef_sbc.F90:109-145,161-184",
            "dynspg_ts.F90:552-572",
            "stprk3_stg.F90:452-565",
            "traqsr.F90:621,629-630,642",
        ],
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.census.read_text(encoding="utf-8")), plant=args.plant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "ROUND16_V1_V2_ATTRIBUTION PASS "
        f"different={result['counts']['different_with_transcendental_ancestor']} "
        f"controls={result['counts']['identical_pre_forcing_controls']} "
        f"unattributed={result['counts']['unattributed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
