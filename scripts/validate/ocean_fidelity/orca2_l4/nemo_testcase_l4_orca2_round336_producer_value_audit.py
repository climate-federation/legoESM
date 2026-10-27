#!/usr/bin/env python3
"""Audit admitted ORCA2 observer programs without running NEMO.

This is a read-only, offline audit of the round-328/331/332/333/336 producer
binaries and their already-admitted files.  It distinguishes an instruction-
census provenance difference from a demonstrated value difference by checking
the real build flags, FMA instructions, ordered register-free FP opcodes, and
the byte-identical stage/restart witnesses written by the observer-free twin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
EVIDENCE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
TWIN = EVIDENCE / "round326/acquisition/orca1ice_tke_hmi_10step_np2"
SOURCE_CFG = NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI"
FC = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
FP_RE = re.compile(
    r"^(?:f(?:add|sub|mul|div|sqrt)|(?:add|sub|mul|div|sqrt|min|max)(?:s|p)[sd])"
)
FMA_RE = re.compile(r"^(?:v?f(?:madd|msub|nmadd|nmsub)|vfmadd|vfmsub|vfnmadd|vfnmsub)")
FORBIDDEN_FLAGS = (
    "-ffast-math",
    "-Ofast",
    "-mfma",
    "-ffp-contract=fast",
    "-funsafe-math-optimizations",
    "-fassociative-math",
    "-fno-signed-zeros",
)
EXPECTED_FLAGS = (
    "-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer "
    "-ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize"
)
FLAG_CONFIGS = (
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI",
    "ORCA2_ORCA1ICE_OMIP_L4_R328ZDF",
    "ORCA2_ORCA1ICE_OMIP_L4_R332SO",
    "ORCA2_ORCA1ICE_OMIP_L4_R332RAW",
    "ORCA2_ORCA1ICE_OMIP_L4_R333TR",
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI",
    "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF",
)
EXPECTED_BUNDLES = {
    "restart": ("ORCA2_00000010_restart*.nc", 4,
                "9e09b85d488f60846b94a46687dfe9d757b1e09433cbcab0d60293c7d7161f7a"),
    "stage": ("oracle_stage_*.bin", 30,
              "9bebfc965ea298c5521e278ac07b41417edf2ba5f9cac8b0f704d59065f7ef15"),
    "zdf_return": ("oracle_zdf_entry_kt00000001.bin", 1,
                   "209369523aafa70ddbaa17695908b6da30929dae6bce7ee225c9ee353c993ad8"),
}


@dataclass(frozen=True)
class FunctionSpec:
    label: str
    unit: str
    symbol: str
    production_count: int | None
    producer_count: int
    expected_ordered_equal: bool | None


@dataclass(frozen=True)
class RecordSpec:
    label: str
    run: Path
    functions: tuple[FunctionSpec, ...]
    direct_census_differs: bool
    inherited_census_differs: bool
    verdict: str


SOURCES = {
    "dynzdf": NEMO / "src/OCE/DYN/dynzdf.F90",
    "stprk3_stg": NEMO / "src/OCE/stprk3_stg.F90",
    "zdftke": NEMO / "src/OCE/ZDF/zdftke.F90",
    "trazdf": NEMO / "src/OCE/TRA/trazdf.F90",
    "zdfphy": NEMO / "src/OCE/ZDF/zdfphy.F90",
    "traldf_iso": NEMO / "src/OCE/TRA/traldf_iso.F90",
}

STAGE = FunctionSpec(
    "stp_rk3_stg", "stprk3_stg", "__stprk3_stg_MOD_stp_rk3_stg",
    1175, 1367, False,
)
DYN = FunctionSpec(
    "dyn_zdf", "dynzdf", "__dynzdf_MOD_dyn_zdf", 2446, 2365, False,
)
TKE_R331 = (
    FunctionSpec("zdf_tke", "zdftke", "__zdftke_MOD_zdf_tke", 1007, 783, False),
    FunctionSpec("tke_tke", "zdftke", "__zdftke_MOD_tke_tke", None, 1054, None),
)
TKE_R332 = (
    FunctionSpec("zdf_tke", "zdftke", "__zdftke_MOD_zdf_tke", 1007, 0, False),
    FunctionSpec("tke_tke", "zdftke", "__zdftke_MOD_tke_tke", None, 1054, None),
)

RECORDS = (
    RecordSpec(
        "round 328 dyn walk",
        EVIDENCE / "round328/acquisition/orca1ice_dynzdf_kt1_np2",
        (DYN, STAGE),
        True, False, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 328 ZDF terms",
        EVIDENCE / "round328/acquisition/orca1ice_zdfphy_kt1_np2",
        (DYN, STAGE),
        False, True, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 331 TKE",
        EVIDENCE / "round331/acquisition/orca1ice_tke331_kt1_np2",
        TKE_R331,
        True, True, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 332 source order",
        EVIDENCE / "round332/acquisition/orca1ice_source_order_kt1_np2",
        TKE_R332 + (STAGE,),
        True, True, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 332 raw V",
        EVIDENCE / "round332/acquisition/orca1ice_raw_postfold_v_kt1_np2",
        (STAGE,),
        True, False, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 333 tracer stage",
        EVIDENCE / "round333/acquisition/orca1ice_tracer_stage3_kt2_np2",
        (STAGE,),
        True, False, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 333 internal LDF",
        EVIDENCE / "round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2",
        (
            STAGE,
            FunctionSpec("traldf_iso_lap", "traldf_iso", "__traldf_iso_MOD_traldf_iso_lap", 526, 526, False),
            FunctionSpec("traldf_iso_a33", "traldf_iso", "__traldf_iso_MOD_traldf_iso_a33", 538, 538, False),
        ),
        False, True, "values production-identical PLAUSIBLE",
    ),
    RecordSpec(
        "round 336 tra_zdf",
        EVIDENCE / "round336/acquisition/orca1ice_trazdf_internal_kt2_np2",
        (
            FunctionSpec("tra_zdf", "trazdf", "__trazdf_MOD_tra_zdf", 60, 60, True),
            FunctionSpec("tra_zdf_imp", "trazdf", "__trazdf_MOD_tra_zdf_imp", 1234, 1234, True),
            FunctionSpec("zdf_phy", "zdfphy", "__zdfphy_MOD_zdf_phy", 121, 121, False),
            STAGE,
        ),
        False, True, "values production-identical PLAUSIBLE",
    ),
)


class AuditError(RuntimeError):
    """A pinned audit fact moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle(root: Path, pattern: str) -> dict[str, object]:
    paths = sorted(root.glob(pattern), key=lambda path: path.name)
    lines = [f"{sha256_file(path)}  {path.name}\n" for path in paths]
    return {
        "files": len(paths),
        "sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "members": {path.name: sha256_file(path) for path in paths},
    }


def opcode_sequence(path: Path, symbol: str) -> list[str]:
    text = run([
        "objdump", "-drwC", "--no-show-raw-insn", f"--disassemble={symbol}", str(path),
    ])
    sequence: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*[0-9a-f]+:\s+(\S+)", line)
        if match and FP_RE.match(match.group(1)):
            sequence.append(match.group(1))
    return sequence


def sequence_digest(sequence: list[str]) -> str:
    return hashlib.sha256(("\n".join(sequence) + "\n").encode()).hexdigest()


def producer_symbols(path: Path) -> set[str]:
    names: set[str] = set()
    for line in run(["nm", "-a", str(path)]).splitlines():
        fields = line.split()
        if fields:
            names.add(fields[-1])
    return names


def compile_production_objects(scratch: Path) -> dict[str, Path]:
    module_dir = scratch / "mod"
    module_dir.mkdir()
    for module in (SOURCE_CFG / "BLD/inc").glob("*.mod"):
        shutil.copy2(module, module_dir / module.name)
    cpp_flags = [
        "-Dkey_nosignedzero", "-P", "-traditional", "-Dkey_si3", "-Dkey_qco",
        "-Dkey_vco_1d3d", "-Dkey_RK3", "-I", str(SOURCE_CFG / "WORK"),
        "-I", str(SOURCE_CFG / "BLD/inc"),
    ]
    compile_flags = EXPECTED_FLAGS.split()
    objects: dict[str, Path] = {}
    for unit, source in SOURCES.items():
        preprocessed = scratch / f"{unit}.f90"
        obj = scratch / f"{unit}.o"
        run(["cpp", *cpp_flags, source.name, "-o", str(preprocessed)], cwd=source.parent)
        run([
            str(FC), "-c", *compile_flags, "-I", str(module_dir), "-J", str(module_dir),
            str(preprocessed), "-o", str(obj),
        ])
        objects[unit] = obj
    return objects


def read_flags() -> dict[str, object]:
    sources: list[dict[str, str]] = []
    common_flags: str | None = None
    for config in FLAG_CONFIGS:
        makefile = NEMO / f"cfgs/{config}/BLD/Makefile"
        match = re.search(r"^export FFLAGS = (.+)$", makefile.read_text(), re.MULTILINE)
        require(match is not None, f"missing FFLAGS in {makefile}")
        flags = match.group(1)
        require(flags.startswith(EXPECTED_FLAGS), f"production flags changed in {config}")
        present = [flag for flag in FORBIDDEN_FLAGS if flag in flags]
        require(not present, f"fast-math-like flags present in {config}: {present}")
        if common_flags is None:
            common_flags = flags
        require(flags == common_flags, f"producer flags differ in {config}")
        sources.append({"config": config, "path": str(makefile)})
    return {
        "sources": sources,
        "flags": common_flags,
        "fast_math_like_flags": [],
        "retired_build_note": (
            "round-331 and round-328-ZDF build directories were retired; their "
            "manifest-pinned acquisition driver inherited the same source build flags"
        ),
    }


def audit(*, plant: str | None = None) -> dict[str, object]:
    require(TWIN.is_dir(), f"missing twin {TWIN}")
    twin_bundles: dict[str, dict[str, object]] = {}
    for label, (pattern, expected_files, expected_digest) in EXPECTED_BUNDLES.items():
        found = bundle(TWIN, pattern)
        require(found["files"] == expected_files, f"twin {label} file count moved")
        require(found["sha256"] == expected_digest, f"twin {label} digest moved")
        twin_bundles[label] = found

    flags = read_flags()
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="r336-producer-audit-") as name:
        production = compile_production_objects(Path(name))
        for record in RECORDS:
            require(record.run.is_dir(), f"missing record {record.run}")
            binary = record.run / "nemo"
            require(binary.is_file(), f"missing producer binary {binary}")
            symbols = producer_symbols(binary)
            functions: list[dict[str, object]] = []
            affected_fma = 0
            for spec in record.functions:
                base_ops = opcode_sequence(production[spec.unit], spec.symbol)
                producer_ops = opcode_sequence(binary, spec.symbol) if spec.symbol in symbols else []
                if plant == "opcode" and record.label == "round 336 tra_zdf" and spec.label == "tra_zdf":
                    producer_ops.append("addsd")
                if spec.production_count is None:
                    require(not base_ops, f"{spec.label}: production unexpectedly not inlined")
                else:
                    require(len(base_ops) == spec.production_count,
                            f"{spec.label}: production count {len(base_ops)} moved")
                require(len(producer_ops) == spec.producer_count,
                        f"{record.label} {spec.label}: producer count {len(producer_ops)} moved")
                ordered_equal = base_ops == producer_ops if spec.production_count is not None else None
                if spec.expected_ordered_equal is not None:
                    require(ordered_equal is spec.expected_ordered_equal,
                            f"{record.label} {spec.label}: ordered-opcode verdict moved")
                fma = sum(bool(FMA_RE.match(opcode)) for opcode in producer_ops)
                affected_fma += fma
                functions.append({
                    "function": spec.label,
                    "production_fp": "inlined" if spec.production_count is None else len(base_ops),
                    "producer_fp": len(producer_ops),
                    "ordered_opcode_equal": ordered_equal,
                    "production_sequence_sha256": sequence_digest(base_ops),
                    "producer_sequence_sha256": sequence_digest(producer_ops),
                    "producer_fma": fma,
                })
            require(affected_fma == 0, f"{record.label}: producer has FMA instructions")

            witnesses: dict[str, object] = {}
            for label, (pattern, _, _) in EXPECTED_BUNDLES.items():
                found = bundle(record.run, pattern)
                if plant == "restart" and record.label == "round 336 tra_zdf" and label == "restart":
                    found["sha256"] = "0" * 64
                equal = (found["files"] == twin_bundles[label]["files"] and
                         found["sha256"] == twin_bundles[label]["sha256"])
                require(equal, f"{record.label}: {label} bytes differ from twin")
                witnesses[label] = {
                    "files": found["files"], "sha256": found["sha256"], "equal_to_twin": True,
                }
            rows.append({
                "record": record.label,
                "path": str(record.run),
                "producer_binary_sha256": sha256_file(binary),
                "direct_census_differs": record.direct_census_differs,
                "inherited_census_differs": record.inherited_census_differs,
                "affected_function_fma": affected_fma,
                "restart_and_output_witnesses": witnesses,
                "functions": functions,
                "verdict": record.verdict,
            })
    return {
        "status": "PASS_R336_PRODUCER_VALUE_AUDIT",
        "observer_free_twin": str(TWIN),
        "flags": flags,
        "twin_witnesses": {
            label: {"files": item["files"], "sha256": item["sha256"]}
            for label, item in twin_bundles.items()
        },
        "records": rows,
        "summary": {
            "confirmed": sum(row["verdict"].endswith("CONFIRMED") for row in rows),
            "plausible": sum(row["verdict"].endswith("PLAUSIBLE") for row in rows),
            "values_differ": sum(row["verdict"] == "values differ" for row in rows),
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="r336-producer-selftest-") as name:
        root = Path(name)
        (root / "b").write_bytes(b"two")
        (root / "a").write_bytes(b"one")
        first = bundle(root, "*")
        (root / "a").write_bytes(b"changed")
        second = bundle(root, "*")
        require(first["sha256"] != second["sha256"], "bundle mutation plant stayed green")
        require(FP_RE.match("addsd") is not None and FP_RE.match("movsd") is None,
                "FP opcode classifier self-test failed")
        require(FMA_RE.match("vfmadd132sd") is not None and FMA_RE.match("mulsd") is None,
                "FMA classifier self-test failed")
    print("R336_PRODUCER_VALUE_AUDIT_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("opcode", "restart"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.output is not None, "--output is required")
    report = audit(plant=args.plant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"])
    print(f"records={len(report['records'])} restart_sha256={report['twin_witnesses']['restart']['sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"REFUSE: {error}") from error
