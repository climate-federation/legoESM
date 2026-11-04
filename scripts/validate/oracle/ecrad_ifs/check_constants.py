"""Assert the YOMCST stand-in equals 49r1 yomcst.F90 for every constant the build reads.

usage: python -I check_constants.py <build src dir> <49r1 arpifs/module/yomcst.F90>
Names come from every "USE YOMCST ..., ONLY : ..." in the compiled sources;
49r1 values are the literal assignments in yomcst.F90 (LAPE=F branch: the
LAPE branch multiplies by RPLRG and is not a plain literal). Exit 1 on any
missing or unequal value.
"""
import glob
import re
import sys


def used_names(src):
    names = set()
    for p in glob.glob(f"{src}/*.F90"):
        for m in re.finditer(r"^\s*USE\s+YOMCST\s*,\s*ONLY\s*:\s*(.*)$", open(p).read(), re.M | re.I):
            names |= {n.strip().upper() for n in m.group(1).split(",") if n.strip()}
    return names


def literals(text, pattern):
    out = {}
    for m in re.finditer(pattern, text, re.M | re.I):
        out.setdefault(m.group(1).upper(), set()).add(float(m.group(2).replace("_JPRB", "")))
    return out


def main(src, yomcst):
    names = used_names(src)
    standin = literals(open(f"{src}/yomcst_ecrad.F90").read(),
                       r"^\s*REAL\(KIND=JPRB\),\s*PARAMETER\s*::\s*(\w+)\s*=\s*([0-9.]+(?:[EeDd][+-]?\d+)?)_JPRB")
    ifs = literals(open(yomcst).read(), r"^\s*(\w+)\s*=\s*([0-9.]+(?:[EeDd][+-]?\d+)?)_JPRB\s*$")
    bad = 0
    for n in sorted(names):
        a, b = standin.get(n), ifs.get(n)
        ok = a is not None and b is not None and len(a) == 1 and len(b) == 1 and a == b
        print(f"{'OK ' if ok else 'BAD'} {n:10s} stand-in={a} 49r1={b}")
        bad += not ok
    if not names or bad:
        raise SystemExit(f"constants check failed ({bad} bad, {len(names)} names)")
    print(f"CONSTANTS_OK {len(names)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
