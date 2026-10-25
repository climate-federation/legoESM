#!/usr/bin/env python3
"""Per-function floating-point instruction census for the O7 observers."""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

FUNCTIONS = {
    "blk_ice_1": ("sbcblk.o", "__sbcblk_MOD_blk_ice_1"),
    "blk_ice_2": ("sbcblk.o", "__sbcblk_MOD_blk_ice_2"),
    "ice_alb": ("icealb.o", "__icealb_MOD_ice_alb"),
    "ice_flx_other": ("icesbc.o", "__icesbc_MOD_ice_flx_other"),
}
FIXTURE_FUNCTIONS = {
    name: ("fixture.o", f"__o7_passivity_fixture_MOD_{name}_probe") for name in FUNCTIONS
}
FP = re.compile(
    r"^(?:v)?(?:add|sub|mul|div|sqrt|max|min|fmadd|fmsub|fnmadd|fnmsub|cvt).*(?:ss|sd|ps|pd)$"
    r"|^f(?:add|sub|mul|div|sqrt|scale|yl2x|2xm1)",
    re.I,
)


def _functions(path: Path) -> dict[str, list[str]]:
    text = subprocess.run(
        ["objdump", "-d", str(path)], check=True, text=True, capture_output=True
    ).stdout
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^[0-9a-f]+\s+<([^>]+)>:$", line.strip())
        if match:
            current = match.group(1)
            found.setdefault(current, [])
            continue
        if current is None or "\t" not in line:
            continue
        columns = line.split("\t")
        if len(columns) >= 3:
            mnemonic = columns[2].strip().split()[0] if columns[2].strip() else ""
            if FP.match(mnemonic):
                found[current].append(mnemonic.lower())
    return found


def _flags(path: Path) -> str:
    text = path.read_text(errors="replace")
    match = re.search(r"^%FCFLAGS\s+(.+)$", text, re.M)
    if not match:
        raise ValueError("STATUS PLANT-FIRED: missing %FCFLAGS in record arch_nemo.fcm")
    flags = match.group(1).strip()
    required = ("-fdefault-real-8", "-O3", "-funroll-all-loops", "-fno-tree-vectorize")
    if any(flag not in flags.split() for flag in required) or "-O0" in flags.split():
        raise ValueError(
            f"STATUS PLANT-FIRED: record build flags are not production flags: {flags}"
        )
    return flags


def census(reference: Path, record: Path, arch: Path, *, fixture: bool = False) -> dict:
    mapping = FIXTURE_FUNCTIONS if fixture else FUNCTIONS
    flags = _flags(arch)
    cache: dict[Path, dict[str, list[str]]] = {}
    report = {}
    for label, (filename, symbol) in mapping.items():
        left, right = reference / filename, record / filename
        for path in (left, right):
            if not path.is_file():
                raise ValueError(f"STATUS PLANT-FIRED: missing census object {path}")
            cache.setdefault(path, _functions(path))
        before = cache[left].get(symbol)
        after = cache[right].get(symbol)
        if before is None or after is None:
            raise ValueError(f"STATUS PLANT-FIRED: missing function symbol {symbol}")
        if not before or not after:
            raise ValueError(f"STATUS PLANT-FIRED: zero FP instructions in {label}")
        before_counts = dict(sorted(collections.Counter(before).items()))
        after_counts = dict(sorted(collections.Counter(after).items()))
        if before_counts != after_counts:
            raise ValueError(f"STATUS PLANT-FIRED: FP instruction census changed in {label}")
        report[label] = {"count": len(before), "mnemonics": before_counts}
    return {"status": "PASS_FP_CENSUS_REAL_FLAGS", "fcflags": flags, "functions": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        result = census(args.reference, args.record, args.arch, fixture=args.fixture)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
