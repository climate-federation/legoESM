#!/usr/bin/env python3
"""Prove the stage-2 writer's bytes are what the admission gate reads.

The writer is compiled against small stubs and driven with indexable values,
then the record is parsed with the SAME reader the admission gate uses.  This
is a FAIL-CLOSED preflight step, not a diagnostic: it is what catches a writer
that opens its file and then silently writes no group at all -- which is what
a sign test on a ``NEWUNIT`` unit number did, because ``NEWUNIT`` hands back a
NEGATIVE unit.  An 80-byte header would otherwise have reached the operator's
NEMO run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "nemo_testcase_l2_gyre_round156_developed_stage2_gate.py"
JPI, JPJ, JPK = 6, 5, 4


def load_gate():
    spec = importlib.util.spec_from_file_location("_r156_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("REFUSE: usage: layout_roundtrip.py <record>", file=sys.stderr)
        return 64
    gate = load_gate()
    record = gate.read_record(Path(argv[0]))
    meta = record["meta"]
    fields = record["fields"]
    problems = []
    if meta["kt"] != 1081 or meta["kstg"] != 2 or meta["storage_size"] != 64:
        problems.append(f"header is {meta}")
    if meta["group_count"] != len(gate.GROUPS):
        problems.append(f"group count {meta['group_count']}")
    if set(fields) != {"rhs_entry", "r3u_r3v_Kbb", "rDt_r1_Dt", "zub_zvb"}:
        problems.append(f"driver groups {sorted(fields)}")
    if not problems:
        left, right = fields["rhs_entry"]
        if left.shape != (JPI, JPJ, JPK):
            problems.append(f"rhs_entry shape {left.shape}")
        else:
            expected = np.fromfunction(
                lambda i, j, k: (i + 1) + 10 * (j + 1) + 100 * (k + 1),
                (JPI, JPJ, JPK), dtype=np.float64)
            if not np.array_equal(left, expected):
                problems.append("rhs_entry column-major mapping is wrong")
            if not np.array_equal(right, -expected):
                problems.append("rhs_entry pair order is wrong")
        interior = fields["zub_zvb"][0]
        if interior.shape != (meta["ntei"] - meta["ntsi"] + 1,
                              meta["ntej"] - meta["ntsj"] + 1):
            problems.append(f"zub keeps {interior.shape}, not its interior")
        scalars = fields["rDt_r1_Dt"]
        if float(scalars[0]) != 7200.0 or float(scalars[1]) != 1.0 / 7200.0:
            problems.append(f"scalars {scalars}")
    if problems:
        for problem in problems:
            print(f"REFUSE: stage-2 record layout: {problem}", file=sys.stderr)
        return 1
    print(f"LAYOUT_ROUNDTRIP_PASS {len(fields)} groups, "
          f"{record['bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
