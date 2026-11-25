"""FP-instruction passivity census for the O9 observers (preregistration section 2).

Each observer is one ``CALL`` to an external recorder inserted into the record
build's ppsrc.  For each object this tool compiles the ppsrc with the record
build's real flags: observer-free (must equal the record object's instruction
stream), instrumented, and instrumented plus one injected FP operation (the
plant).  It compares, per function, the count of every floating-point mnemonic
(arithmetic, MIN/MAX, SQRT, FMA, conversions incl. memory-operand ``cvtsi2sdl``,
compares, and the sign-bit ops ``andpd``/``xorpd``/``orpd`` behind ABS and negation).
Passive = every function's mnemonic counts equal, nonzero for the containing
routine, and the plant changes them.  The in-loop Picard form (inside NEMO's
outer ``DO WHILE``) is censused too and reported as it measures.
Prints a JSON report; ``--check`` compares it to a committed one.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")
FP = re.compile(r"^v?(?:(?:add|sub|mul|div|sqrt|max|min)(?:ss|sd|ps|pd)"
                r"|f[n]?m(?:add|sub)\w*(?:ss|sd|ps|pd)|cvt\w+|u?comis[sd]|(?:andn?|x?or)p[sd])[lq]?$")
PLANT = "            qns_ice_1d(1) = qns_ice_1d(1) + 1._wp\n"
PICARD = "      CALL si3rec_o9_picard( (iconv), (l_T_converged(1:npti)) )\n"
# Each variant: object, containing routine, (line, stripped-prefix) anchors in the
# record ppsrc, and observer statements inserted AFTER the given 1-based lines.
VARIANTS = {
    "picard_after_do_while": dict(
        obj="icethd_zdf_bl99", routine="__icethd_zdf_bl99_MOD_ice_thd_zdf_bl99",
        anchors={251: "DO WHILE", 603: "END DO", 772: "END DO  ! End of the do while iterative procedure",
                 773: "!"},
        insert={772: PICARD}),
    "picard_inside_do_while": dict(
        obj="icethd_zdf_bl99", routine="__icethd_zdf_bl99_MOD_ice_thd_zdf_bl99",
        anchors={251: "DO WHILE", 603: "END DO", 605: "ELSEIF( k_cnd == np_cnd_ON ) THEN"},
        insert={603: PICARD}),
    "frames_f1_f4": dict(
        obj="icethd", routine="__icethd_MOD_ice_thd",
        anchors={176: "DO jl = 1, jpl", 187: "IF( npti > 0 ) THEN", 195: "dh_snowice(1:npti) = 0._wp",
                 197: "CALL ice_thd_zdf", 199: "IF( ln_icedH )    CALL ice_thd_dh",
                 201: "CALL ice_thd_temp", 203: "CALL ice_thd_sal"},
        insert={195: "            CALL si3rec_o9_frame( 1, (jl), (kt) )\n",
                197: "            CALL si3rec_o9_frame( 2, (jl), (kt) )\n",
                199: "            CALL si3rec_o9_frame( 3, (jl), (kt) )\n",
                201: "            CALL si3rec_o9_frame( 4, (jl), (kt) )\n"}),
}

_mm_spec = importlib.util.spec_from_file_location("o9minmax", HERE / "nemo_si3_o9_minmax_audit.py")
_mm = importlib.util.module_from_spec(_mm_spec)
_mm_spec.loader.exec_module(_mm)


def instrument(lines: list[str], variant: dict, plant: str = "") -> list[str]:
    """Insert the variant's observers (plant after the last one); refuse if any anchor moved."""
    for ln, text in variant["anchors"].items():
        if not lines[ln - 1].strip().startswith(text):
            raise SystemExit(f"anchor {ln} is not {text!r}: {lines[ln - 1].strip()!r}")
    out = list(lines)
    last = max(variant["insert"])
    for after in sorted(variant["insert"], reverse=True):
        out.insert(after, variant["insert"][after] + (plant if after == last else ""))
    return out


def fp_counts(obj: Path) -> dict[str, collections.Counter]:
    text = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)], check=True,
                          capture_output=True, text=True).stdout
    found, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            cur = found.setdefault(m.group(1), collections.Counter())
            continue
        m = re.match(r"^\s+[0-9a-f]+:\t(\S+)", line)
        if m and cur is not None and FP.match(m.group(1)):
            cur[m.group(1)] += 1
    return found


def _flags(build: Path) -> list[str]:
    flags = _mm._flags(build)
    need = ("-fdefault-real-8", "-O3", "-funroll-all-loops", "-fno-tree-vectorize")
    if any(f not in flags for f in need) or "-O0" in flags:
        raise SystemExit(f"record build flags are not the production flags: {flags}")
    return flags


def _compile(src: Path, build: Path, toolchain: Path) -> Path:
    env = dict(os.environ, PATH=f"{toolchain / 'bin'}:{os.environ['PATH']}")
    obj = src.with_suffix(".o")
    subprocess.run(["mpif90", "-c", "-cpp", *_flags(build), f"-I{build / 'inc'}", f"-J{src.parent}",
                    f"-I{toolchain / 'include'}", str(src), "-o", str(obj)],
                   check=True, env=env, capture_output=True)
    return obj


def census(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
    report = {"build": str(build), "flags": _flags(build), "plant": PLANT.strip(), "variants": {}}
    with tempfile.TemporaryDirectory() as tmp:
        plain_cache = {}
        for name, v in VARIANTS.items():
            ppsrc = build / "ppsrc/nemo" / f"{v['obj']}.f90"
            record = build / "obj" / f"{v['obj']}.o"
            lines = ppsrc.read_text().splitlines(keepends=True)
            counts = {}
            for arm, body in (("instrumented", instrument(lines, v)),
                              ("plant", instrument(lines, v, PLANT))):
                src = Path(tmp) / f"{name}_{arm}" / f"{v['obj']}.f90"
                src.parent.mkdir()
                src.write_text("".join(body))
                counts[arm] = fp_counts(_compile(src, build, toolchain))
            if v["obj"] not in plain_cache:
                src = Path(tmp) / f"plain_{v['obj']}" / f"{v['obj']}.f90"
                src.parent.mkdir()
                src.write_text("".join(lines))
                obj = _compile(src, build, toolchain)
                if _mm._stream(obj) != _mm._stream(record):
                    raise SystemExit(f"{v['obj']}: observer-free recompile differs from the record object")
                plain_cache[v["obj"]] = fp_counts(obj)
            plain = plain_cache[v["obj"]]
            funcs = sorted(f for f in set(plain) | set(counts["instrumented"])
                           if plain.get(f) or counts["instrumented"].get(f))
            per = {}
            for f in funcs:
                a, b = plain.get(f, collections.Counter()), counts["instrumented"].get(f, collections.Counter())
                per[f] = {"plain_total": sum(a.values()), "instrumented_total": sum(b.values()),
                          "differing_mnemonics": {k: [a[k], b[k]] for k in sorted(set(a) | set(b))
                                                  if a[k] != b[k]}}
            routine = per.get(v["routine"], {})
            report["variants"][name] = {
                "object": v["obj"], "routine": v["routine"],
                "record_object_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "ppsrc_sha256": hashlib.sha256(ppsrc.read_bytes()).hexdigest(),
                "inserted_after_lines": sorted(v["insert"]),
                "observers": [v["insert"][k].strip() for k in sorted(v["insert"])],
                "functions": per,
                "passive": routine.get("plain_total", 0) > 0
                and all(not x["differing_mnemonics"] for x in per.values()),
                "plant_fired": counts["plant"].get(v["routine"]) != plain.get(v["routine"]),
            }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build", type=Path, default=BUILD)
    p.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    p.add_argument("--check", type=Path, help="committed report; exit 1 on any difference")
    args = p.parse_args(argv)
    rep = census(args.build, args.toolchain)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.check:
        same = json.loads(args.check.read_text()) == json.loads(text)
        print("O9_OBSERVER_CENSUS", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
