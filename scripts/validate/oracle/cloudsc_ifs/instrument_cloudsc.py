"""Write-only stage dumps for the 49r1 CLOUDSC oracle (V13 section-by-section port).

Copies cloudsc.F90 (OpenIFS 8f6f722) and inserts ``CALL SDUMP(...)`` lines after
the last line of each section of the level loop.  No source line is removed or
changed: the script asserts the original lines are an in-order subsequence of
the output and that every anchor line holds the expected text.
SDUMP (sdump.F90) appends ``JK`` then the values to ``$CLOUDSC_DUMP_DIR/<tag>.bin``.

usage: instrument_cloudsc.py <cloudsc.F90> <out.F90>
"""
from __future__ import annotations

import sys

# stage -> (anchor line, expected stripped text, [(field, Fortran expression, count)])
_N = "KLON"
_NV = "KLON*NCLV"
_NM = "KLON*NCLV*NCLV"
STAGES = {
    "s1": (1344, "ENDDO", [("zsolqa", "ZSOLQA", _NM)]),
    "s32": (1398, "ENDIF ! JK<KLEV", [("zsolqa", "ZSOLQA", _NM), ("zsolac", "ZSOLAC", _N),
                                       ("plude", "PLUDE(:,JK)", _N)]),
    "s33": (1514, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("zsolqb", "ZSOLQB", _NM),
                            ("zsolac", "ZSOLAC", _N), ("zsolab", "ZSOLAB", _N)]),
    "s34": (1756, "ENDIF ! on ITURBEROSION", [("zsolqa", "ZSOLQA", _NM), ("zsolac", "ZSOLAC", _N)]),
    "s41": (1877, "ENDDO ! End loop over hydrometeor type",
            [("zsolqa", "ZSOLQA", _NM), ("zqxfg", "ZQXFG", _NV), ("zqpretot", "ZQPRETOT", _N),
             ("zfallsink", "ZFALLSINK", _NV)]),
    "s42": (1919, "ENDDO", [("zcovptot", "ZCOVPTOT", _N), ("zcovpclr", "ZCOVPCLR", _N),
                            ("zraincld", "ZRAINCLD", _N), ("zraincldm1", "ZRAINCLDM1", _N),
                            ("zsnowcld", "ZSNOWCLD", _N), ("zcovpmax", "ZCOVPMAX", _N)]),
    "s43": (2259, "ENDDO", [("zsolqb", "ZSOLQB", _NM)]),
    "s44": (2378, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("zqxfg", "ZQXFG", _NV)]),
    "s45": (2604, "ENDIF ! on IDEPICE", [("zsolqa", "ZSOLQA", _NM), ("zqxfg", "ZQXFG", _NV),
                                         ("zcldtopdist", "ZCLDTOPDIST", _N)]),
    "s47": (2745, "ENDIF ! on IDEPSNOW", [("zsolqa", "ZSOLQA", _NM)]),
    "s48": (2800, "ENDDO", [("zsolqb", "ZSOLQB", _NM)]),
    "s412": (2862, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("zqxfg", "ZQXFG", _NV)]),
    "s413": (2933, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("prainfrac", "PRAINFRAC_TOPRFZ", _N),
                             ("llrainliq", "MERGE(1.0_JPRB,0.0_JPRB,LLRAINLIQ)", _N)]),
    "s414": (2960, "ENDDO", [("zsolqa", "ZSOLQA", _NM)]),
    "s49": (3239, "ENDIF ! on IWARMRAIN > 1", [("zsolqa", "ZSOLQA", _NM), ("zsolqb", "ZSOLQB", _NM)]),
    "s415": (3412, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("zqxfg", "ZQXFG", _NV),
                             ("zcovptot", "ZCOVPTOT", _N)]),
    "s521": (3536, "ENDDO", [("zsolqa", "ZSOLQA", _NM), ("zanew", "ZANEWM1", _N)]),
    "s522": (3640, "ENDDO", [("zqxn", "ZQXN", _NV)]),
    "s6": (3769, "ENDDO", [("tlt", "PTENDENCY_LOC_T(:,JK)", _N), ("tlq", "PTENDENCY_LOC_Q(:,JK)", _N),
                           ("tla", "PTENDENCY_LOC_A(:,JK)", _N),
                           ("tlc", "PTENDENCY_LOC_CLD(:,JK,1:NCLV-1)", "KLON*(NCLV-1)")]),
}


def main(src, dst):
    lines = open(src).read().splitlines()
    out = []
    by_line = {a: (tag, txt, f) for tag, (a, txt, f) in STAGES.items()}
    for i, line in enumerate(lines, 1):
        out.append(line)
        if i in by_line:
            tag, txt, fields = by_line[i]
            if line.strip() != txt:
                raise SystemExit(f"anchor {tag} line {i}: {line.strip()!r} != {txt!r}")
            for name, expr, n in fields:
                out.append(f"  CALL SDUMP('{tag}_{name}', JK, {expr}, {n})")
    it = iter(out)
    if not all(any(l == o for o in it) for l in lines):
        raise SystemExit("an original line was lost")
    if len(out) != len(lines) + sum(len(f) for _, _, f in STAGES.values()):
        raise SystemExit("line count mismatch")
    open(dst, "w").write("\n".join(out) + "\n")
    print(f"INSTRUMENTED {len(STAGES)} stages, {len(out) - len(lines)} dump lines")


if __name__ == "__main__":
    main(*sys.argv[1:3])
