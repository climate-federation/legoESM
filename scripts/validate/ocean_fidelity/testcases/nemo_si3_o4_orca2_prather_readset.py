#!/usr/bin/env python3
"""O4 registration for the shared source-derived SI3 read-set scanner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import KINDS
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_readset import (
    DEFAULT_PPSRC, HEADER_REQUIRED_SCALARS, INTRINSICS, KEYWORDS, META_SCALARS,
    NON_REPLAY_ARRAYS, SOURCE_OR_CONTROL_SCALARS, _index_routines, _normalise,
    _routine_reads, _source_coverage,
)

ROOT = "ice_dyn_adv_pra"
ENTRY_KINDS = (1, 2, 3, 4)


def derive_and_gate(ppsrc: Path = DEFAULT_PPSRC, plant: str | None = None) -> dict[str, object]:
    routines, interfaces = _index_routines(ppsrc)
    callable_names = set(routines) | set(interfaces)
    pending = [ROOT]; visited: set[str] = set(); reads: set[str] = set(); scalars: set[str] = set(); sources: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited: continue
        if name not in routines: raise ValueError(f"missing registered NEMO routine {name} for O4")
        visited.add(name); routine = routines[name]; sources.add(f"{routine.source}:{name}")
        direct, direct_scalars, calls = _routine_reads(routine, callable_names)
        reads.update(_normalise(value) for value in direct)
        scalars.update(_normalise(value) for value in direct_scalars)
        for call in calls:
            if call in routines: pending.append(call)
            pending.extend(interfaces.get(call, ()))
    reads = {name for name in reads if name not in NON_REPLAY_ARRAYS and not name.startswith(("diag_","rdiag_"))}
    covered = {_normalise(name.replace(".", "_")) for kind in ENTRY_KINDS for name in KINDS[kind]}
    family_suffix = {"ice":"ice","sn":"sn","a":"a","age":"age","sal":"sal","ap":"ap","vp":"vp","vl":"vl",
                     "c0":"c0","e":"e","si":"si"}
    covered.update(f"{component}{family_suffix[family]}" for family in family_suffix
                   for component in ("sx","sy","sxx","syy","sxy"))
    covered.add("hi_max")
    # Dummies are deliberately absent from the shared scanner's external-state
    # set; bind their actual entry records explicitly at this call boundary.
    covered.update(("pu_ice","pv_ice","ph_i","ph_s","ph_ip","pato_i","pv_i","pv_s","psv_i","poa_i",
                    "pa_i","pa_ip","pv_ip","pv_il","pe_s","pe_i","pszv_i"))
    if plant == "drop-sxice": covered.discard("sxice")
    missing = sorted(reads - covered)
    if missing: raise ValueError(f"STATUS PLANT-FIRED: O4 source read-set is not recorded: {missing}")
    namelist, parameters = _source_coverage(ppsrc)
    scalars = {name for name in scalars - reads - covered if not name.startswith(("diag_","rdiag_"))}
    scalar_coverage = {name:("record_header" if name == "rdt_ice" else "record_metadata" if name in META_SCALARS else
        "resolved_namelist_in_manifest" if name in namelist else "pinned_binary_or_source_in_manifest")
        for name in scalars if name in META_SCALARS or name in namelist or name in parameters or
        name in SOURCE_OR_CONTROL_SCALARS or name in HEADER_REQUIRED_SCALARS or name in {"rdt_ice","ldfull","psgn"}}
    missing_scalars = sorted(scalars-set(scalar_coverage))
    if missing_scalars: raise ValueError(f"STATUS PLANT-FIRED: O4 scalar read-set is not covered: {missing_scalars}")
    return {"status":"PASS_SOURCE_DERIVED_READ_SETS","rungs":{"O4":{"roots":[ROOT],
        "callees":sorted(visited-{ROOT}),"sources":sorted(sources),"array_reads":sorted(reads),
        "scalar_reads":sorted(scalars),"scalar_coverage":scalar_coverage,"entry_kinds":list(ENTRY_KINDS),
        "boundary_kinds":[1,2,3,4,5,6],"coverage":sorted(covered)}}}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--ppsrc",type=Path,default=DEFAULT_PPSRC); parser.add_argument("--plant",choices=("drop-sxice",))
    args=parser.parse_args()
    try: print(json.dumps(derive_and_gate(args.ppsrc,args.plant),indent=2,sort_keys=True)); return 0
    except (OSError,ValueError) as exc: print(str(exc)); return 1


if __name__=="__main__": raise SystemExit(main())
