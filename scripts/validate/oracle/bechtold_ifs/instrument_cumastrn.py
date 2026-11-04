"""Make an INSTRUMENTED scratch copy of OpenIFS cumastrn.F90 (8f6f722).

Only inserts CALL DUMP*/STAGE statements (write-only) at fixed anchors; no
statement of the source is modified or removed.  Every anchor must match
exactly once or this script aborts.  Output: fortran/build/cumastrn_instr.F90
and a unified diff fortran/cumastrn_instr.diff.
"""
import difflib
import sys
from pathlib import Path

SRC = Path(sys.argv[1])      # <openifs>/ifs-source/arpifs/phys_ec/cumastrn.F90
OUT = Path(sys.argv[2])      # <build dir>/cumastrn_instr.F90


def r2(name, k1="1", k2="KLEV"):
    return f"CALL DUMPR2('{name}', {name}, KLON, {k1}, {k2})"


def r1(name):
    return f"CALL DUMPR1('{name}', {name}, KLON)"


def i1(name):
    return f"CALL DUMPI1('{name}', {name}, KLON)"


def i2(name):
    return f"CALL DUMPI2('{name}', {name}, KLON, KLEV)"


def l1(name):
    return f"CALL DUMPL1('{name}', {name}, KLON)"


def block(stage, calls):
    return "\n".join([f"CALL STAGE('{stage}')"] + calls) + "\n"


INSERTS = [
    # (anchor, position 'after'|'before', text)
    ("USE SPP_GEN_MOD , ONLY : SPP_PERT\n", "after", "USE DUMPMOD\n"),
    (" & PLU     )  \n", "after", block("cuinin", [
        r2("ZTENH"), r2("ZQENH"), r2("ZQSENH"), r2("PQSEN"), i2("ILAB")])),
    (" & ICTOP0,   IDPL,     PCAPE )   \n", "after", block("cubasen", [
        r2("PTU"), r2("PQU"), r2("PLU"), r2("ZKINEU"), r1("ZWUBASE"), i2("ILAB"),
        l1("LDCUM"), l1("LDSC"), i1("KCBOT"), i1("KBOTSC"), i1("ICTOP0"),
        i1("IDPL"), r1("PCAPE")])),
    ("CALL CUASCN &\n", "before", block("firstguess", [
        i1("KTYPE"), l1("LDCUM"), r1("ZDHPBL"), r1("ZCAPPBL"), r1("ZMFUB")])),
    (" & KCBOT,    KCTOP,    ICTOP0,   IDPL,     PMFUDE_RATE,   ZKINEU,   PWMEAN ) \n",
     "after", block("cuascn", [
        r2("ZTENH"), r2("ZQENH"), r2("PTU"), r2("PQU"), r2("PLU"), r2("PLRAIN"),
        r2("PMFU"), r1("ZMFUB"), r2("ZLGLAC"), r2("ZMFUS"), r2("ZMFUQ"), r2("ZMFUL"),
        r2("PLUDE"), r2("ZDMFUP"), r2("ZDMFEN"), i1("KCBOT"), i1("KCTOP"),
        i1("ICTOP0"), r2("PMFUDE_RATE"), r2("ZKINEU"), r1("PWMEAN"), i2("ILAB"),
        l1("LDCUM"), i1("KTYPE")])),
    ("!*    5.0          CUMULUS DOWNDRAFT CALCULATIONS\n", "before",
     block("reclass", [i1("KTYPE"), l1("LDCUM")])),
    ("! rescale DD fluxes if deep and shallow convection\n", "before",
     block("closure", [
        r1("ZCAPE"), r1("ZCAPE2"), r1("ZHEAT"), r1("ZXTAU"), r1("ZTAU"), r1("ZMFUB1"),
        r1("ZCAPPBL"), r1("ZDQCV"), r1("ZSATFR"), r1("ZCAPDCYCL"), r1("ZMFUB"),
        r2("PMFD"), l1("LLDDRAF")])),
    ("!*    6.5          IN CASE THAT EITHER DEEP OR SHALLOW IS SWITCHED OFF\n", "before",
     block("scaled", [
        r1("ZMFS"), r2("PMFU"), r2("ZMFUS"), r2("ZMFUQ"), r2("ZMFUL"), r2("ZDMFUP"),
        r2("ZDMFEN"), r2("PLUDE"), r2("PMFUDE_RATE")])),
    ("!*    7.0          DETERMINE FINAL CONVECTIVE FLUXES IN 'CUFLX'\n", "before",
     block("ldreset", [i1("KTYPE"), l1("LDCUM")])),
    (" & PMFLXR,   PMFLXS,   PRAIN,    PMFUDE_RATE,  PMFDDE_RATE )  \n", "after",
     block("cuflxn", [
        r2("PMFU"), r2("ZMFUS"), r2("ZMFUQ"), r2("ZMFUL"), r2("PLUDE"), r2("ZDMFUP"),
        r2("PMFLXR", "1", "KLEV+1"), r2("PMFLXS", "1", "KLEV+1"), r2("PQSEN"),
        r2("PMFUDE_RATE"), r2("ZDPMEL"), r2("ZLGLAC"), r1("PRAIN"), r2("PMFD"),
        r2("ZMFDS"), r2("ZMFDQ"), i1("KTYPE"), l1("LDCUM")])),
    ("CALL CUDTDQN &\n", "before", block("precudtdqn", [
        r2("ZMFUS"), r2("ZMFUQ"), r2("ZMFDS"), r2("ZMFDQ"), r2("PTU"), r2("PQU"),
        r2("PLUDE"), r2("ZDMFUP"), r2("PMFU"), r2("PMFD"), r2("ZMFUL"), r2("PQSEN"),
        r2("ZTENH"), r2("ZQENH"), i1("KCTOP"), i1("KCBOT"), i1("IDTOP"),
        l1("LLDDRAF"), i1("KTYPE"), l1("LDCUM")])),
    (" & PTENT,    PTENQ,    PENTH )\n", "after", block("cudtdqn", [
        r2("PTENT"), r2("PTENQ"), r2("PENTH"), r2("PLUDE")])),
]


def main():
    src = SRC.read_text()
    out = src
    for anchor, where, text in INSERTS:
        n = out.count(anchor)
        if n != 1:
            sys.exit(f"anchor matched {n} times: {anchor!r}")
        out = out.replace(anchor, anchor + text if where == "after" else text + anchor)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out)
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                "cumastrn.F90", "cumastrn_instr.F90")
    OUT.with_suffix(".diff").write_text("".join(diff))
    # sanity: no source line was removed
    removed = [l for l in difflib.ndiff(src.splitlines(), out.splitlines()) if l.startswith("- ")]
    if removed:
        sys.exit(f"source lines removed: {removed[:3]}")
    print("instrumented OK:", OUT)


if __name__ == "__main__":
    main()
