#!/usr/bin/env python3
"""Fail-closed FP-opcode census for every emitted function in four O4 units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o9_observer_census import _flags, fp_counts

HERE = Path(__file__).resolve().parent / "nemo_si3_o4_orca2_prather_acquisition"
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")
UNITS = ("icedyn_adv_pra", "icedyn_adv", "icedyn", "icestp")
PATCHES = {
    "icedyn_adv_pra": ("icedyn_adv_pra_o4.patch", "icedyn_adv_pra_o4_use.patch",
                        "icedyn_adv_pra_o4_capture.patch"),
    "icedyn_adv": ("icedyn_adv_o4.patch",),
    "icedyn": ("icedyn_o4.patch",),
    "icestp": ("icestp_o4.patch", "icestp_o4_capture.patch"),
}
CLONE = re.compile(r"\.(?:(?:constprop|isra|part|cold)(?:\.\d+)?).*$")


def _source_routine(symbol: str) -> str:
    """Map a compiler clone to source without merging the clone's census."""
    return CLONE.sub("", symbol)


def _unit_report(reference: Path, record: Path) -> dict[str, object]:
    a, b = fp_counts(reference), fp_counts(record)
    functions: dict[str, object] = {}
    for symbol in sorted(set(a) | set(b)):
        left, right = a.get(symbol, Counter()), b.get(symbol, Counter())
        functions[symbol] = {
            "source_routine": _source_routine(symbol),
            "reference_emitted": symbol in a,
            "record_emitted": symbol in b,
            "reference_fp_total": sum(left.values()),
            "record_fp_total": sum(right.values()),
            "differing_mnemonics": {
                op: [left[op], right[op]] for op in sorted(set(left) | set(right))
                if left[op] != right[op]
            },
        }
    same_symbols = set(a) == set(b)
    passive = same_symbols and all(not row["differing_mnemonics"] for row in functions.values())
    return {
        "reference_object": str(reference), "record_object": str(record),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "function_set_equal": same_symbols, "functions": functions,
        "reference_fp_total": sum(sum(x.values()) for x in a.values()),
        "record_fp_total": sum(sum(x.values()) for x in b.values()),
        "passive": passive,
    }


def census_real_objects(reference_dir: Path, record_dir: Path, arch: Path) -> dict[str, object]:
    flags = _flags(arch.parent)
    units = {unit: _unit_report(reference_dir / f"{unit}.o", record_dir / f"{unit}.o")
             for unit in UNITS}
    return {
        "flags": flags,
        "scope": "every emitted function and unmerged compiler clone in all patched NEMO units",
        "units": units,
        "pass": all(row["passive"] and row["reference_fp_total"] > 0 for row in units.values()),
    }


def _preprocess(source: Path, output: Path, build: Path, env: dict[str, str]) -> None:
    keys = []
    for line in (build / "cpp.fcm").read_text().splitlines():
        if "bld::tool::fppkeys" in line:
            keys.extend(f"-D{x}" for x in line.split()[1:])
    arch = (build / "arch_nemo.fcm").read_text().splitlines()
    cpp = next(line.split()[1:] for line in arch if line.startswith("%CPP "))
    fpp = next(line.split()[1:] for line in arch if line.startswith("%FPPFLAGS "))
    with output.open("w") as stream:
        subprocess.run([*cpp, *fpp, *keys, f"-I{build.parent/'WORK'}", str(source)],
                       check=True, env=env, stdout=stream, text=True)


def _compile(source: Path, obj: Path, build: Path, mod: Path,
             env: dict[str, str]) -> None:
    proc = subprocess.run(["mpif90", "-c", *_flags(build), f"-I{mod}", f"-I{build/'inc'}",
                           f"-J{mod}", str(source), "-o", str(obj)],
                          env=env, capture_output=True, text=True)
    if proc.returncode:
        raise ValueError(f"{source.name} compile failed: {proc.stderr.strip()}")


def census(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict[str, object]:
    """Compile pristine/observed whole units and compare exact clone names."""
    env = dict(os.environ, PATH=f"{toolchain/'bin'}:{os.environ['PATH']}")
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp); plain_mod = tmp / "plain_mod"; obs_mod = tmp / "obs_mod"
        plain_mod.mkdir(); obs_mod.mkdir()
        raw_dir = tmp / "raw"; pp_dir = tmp / "observed"; raw_dir.mkdir(); pp_dir.mkdir()
        for unit in UNITS:
            raw = raw_dir / f"{unit}.F90"
            shutil.copy2(NEMO / "src/ICE" / f"{unit}.F90", raw)
            for patch in PATCHES[unit]:
                subprocess.run(["patch", "--batch", "--fuzz=0", str(raw), str(HERE/patch)],
                               check=True, capture_output=True, text=True)
            _preprocess(raw, pp_dir / f"{unit}.f90", build, env)

        _compile(HERE / "o4_prather_tap.F90", tmp / "tap.o", build, obs_mod, env)
        _compile(pp_dir / "icedyn_adv_pra.f90", tmp / "obs_icedyn_adv_pra.o", build, obs_mod, env)
        _compile(HERE / "o4_prather_writer.F90", tmp / "writer.o", build, obs_mod, env)
        for unit in UNITS[1:]:
            _compile(pp_dir / f"{unit}.f90", tmp / f"obs_{unit}.o", build, obs_mod, env)

        units = {}
        for unit in UNITS:
            plain_obj = tmp / f"plain_{unit}.o"
            _compile(build / "ppsrc/nemo" / f"{unit}.f90", plain_obj, build, plain_mod, env)
            units[unit] = _unit_report(plain_obj, tmp / f"obs_{unit}.o")

        planted = tmp / "planted.f90"
        text = (pp_dir / "icedyn_adv_pra.f90").read_text()
        anchor = "      CALL o4_capture_cfl( zcfl, icycle )"
        if text.count(anchor) != 1:
            raise ValueError("unique Prather exit capture anchor absent")
        planted.write_text(text.replace(anchor, "      pato_i(1,1) = pato_i(1,1) + 1._wp\n" + anchor))
        _compile(planted, tmp / "planted.o", build, obs_mod, env)
        plant_fired = fp_counts(tmp / "planted.o") != fp_counts(tmp / "obs_icedyn_adv_pra.o")
        return {
            "flags": _flags(build),
            "scope": "every emitted function and unmerged compiler clone in four whole NEMO units",
            "clone_suffixes_preserved": True, "units": units,
            "plant": "unmarked in-function floating add", "plant_fired": plant_fired,
            "pass": plant_fired and all(row["passive"] and row["reference_fp_total"] > 0
                                         for row in units.values()),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--arch", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        actual = any((args.reference_dir, args.record_dir, args.arch))
        if actual and not all((args.reference_dir, args.record_dir, args.arch)):
            raise ValueError("actual census requires --reference-dir, --record-dir, and --arch")
        result = (census_real_objects(args.reference_dir, args.record_dir, args.arch)
                  if actual else census(args.build, args.toolchain))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"STATUS PLANT-FIRED: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered)
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
