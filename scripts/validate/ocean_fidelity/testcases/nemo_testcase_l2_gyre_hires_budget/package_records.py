#!/usr/bin/env python3
"""Admit NEMO's passive Round-14 online surface-V budget records."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

TERMS = (
    "advection",
    "pressure_kinetic_energy_gradient",
    "total_vorticity_coriolis",
    "smagorinsky_lateral_diffusion",
    "vertical_mixing",
    "surface_stress",
    "external_mode_correction",
    "total_increment",
)
FIRST_DAY = 30
LAST_DAY = 180
STEPS_PER_DAY = 24
CLOSURE_BAR_ABS_M_S = 1.0e-12
MAGIC = b"NEMO_R14_GYRBUD "
HALO = 2
REGISTRATION_SOURCE_SHA256 = (
    "480a45f3d7f5e02bd5c4771316bae668e59ee33aaeab87a73c47a6212edf5174"
)
PREREGISTRATION_REPO_PATH = (
    "docs/ocean/fidelity/testcases/"
    "nemo_testcases_l2_gyre_hires_round14_preregistration.md"
)
REQUIRED_RESTART_VARIABLES = frozenset(
    {"kt", "sshn", "un", "vn", "tn", "sn", "en", "avt_k", "avm_k"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_record(path: Path, *, expected_day: int | None = None) -> dict[str, object]:
    """Decode one fixed-layout stream without applying the closure verdict."""
    raw = path.read_bytes()
    header_bytes = 16 + 9 * 4 + 2 * 8
    if len(raw) < header_bytes or raw[:16] != MAGIC:
        raise ValueError(f"{path.name}: bad or truncated magic")
    values = struct.unpack_from("=9i", raw, 16)
    version, day, step, ni, nj, nk, bits, nterms, samples = values
    dt_s, closure_bar = struct.unpack_from("=2d", raw, 16 + 9 * 4)
    if (version, nk, bits, nterms, samples) != (1, 30, 64, len(TERMS), 24):
        raise ValueError(f"{path.name}: unsupported header {values}")
    if step != day * STEPS_PER_DAY or dt_s != 3600.0:
        raise ValueError(f"{path.name}: day/step/dt cadence is inconsistent")
    if expected_day is not None and day != expected_day:
        raise ValueError(
            f"{path.name}: header day {day} does not match filename day "
            f"{expected_day}")
    if closure_bar != CLOSURE_BAR_ABS_M_S:
        raise ValueError(f"{path.name}: closure bar changed to {closure_bar}")
    words = np.frombuffer(raw, dtype="<f8", offset=header_bytes)
    expected = ni * nj * (1 + len(TERMS))
    if words.size != expected:
        raise ValueError(
            f"{path.name}: has {words.size} payload words, expected {expected}")
    mask_native = words[: ni * nj].reshape((ni, nj), order="F")
    terms_native = words[ni * nj :].reshape(
        (ni, nj, len(TERMS)), order="F")
    if ni <= 2 * HALO or nj <= 2 * HALO:
        raise ValueError(f"{path.name}: domain is smaller than its halo")
    mask = mask_native[HALO:-HALO, HALO:-HALO].T > 0.5
    terms = terms_native[HALO:-HALO, HALO:-HALO, :].transpose(2, 1, 0)
    if mask.shape != (82, 122) or int(np.count_nonzero(mask)) != 9480:
        raise ValueError(
            f"{path.name}: unexpected native mask {mask.shape}, "
            f"wet={np.count_nonzero(mask)}")
    return {
        "day": day,
        "step": step,
        "mask": mask,
        "terms": terms,
        "sha256": sha256(path),
    }


def read_record(path: Path, *, expected_day: int | None = None) -> dict[str, object]:
    """Read one fixed-layout stream and enforce finite, pointwise closure."""
    record = decode_record(path, expected_day=expected_day)
    mask = np.asarray(record["mask"])
    terms = np.asarray(record["terms"])
    if not bool(np.all(np.isfinite(terms))):
        raise ValueError(f"{path.name}: non-finite budget word")
    closure = float(np.max(np.abs(
        np.sum(terms[:-1], axis=0)[mask] - terms[-1][mask])))
    if closure > CLOSURE_BAR_ABS_M_S:
        raise ValueError(
            f"{path.name}: closure {closure} exceeds "
            f"{CLOSURE_BAR_ABS_M_S}")
    record["closure_max_abs_m_s"] = closure
    return record


def compare_restarts(control: Path, instrumented: Path) -> dict[str, object]:
    """Prove every common numeric restart word is bitwise unchanged."""
    from netCDF4 import Dataset

    unequal_words = 0
    compared_words = 0
    moved_variables = []
    with Dataset(control) as left, Dataset(instrumented) as right:
        left.set_auto_mask(False)
        right.set_auto_mask(False)
        left_names = {
            name for name, var in left.variables.items()
            if np.dtype(var.dtype).kind in "biufc"}
        right_names = {
            name for name, var in right.variables.items()
            if np.dtype(var.dtype).kind in "biufc"}
        if left_names != right_names:
            raise ValueError(
                "restart numeric-variable registry changed: "
                f"only_control={sorted(left_names - right_names)}, "
                f"only_instrumented={sorted(right_names - left_names)}")
        missing = sorted(REQUIRED_RESTART_VARIABLES - left_names)
        if missing:
            raise ValueError(
                "restart lacks required prognostic providers: "
                + ", ".join(missing))
        for name in sorted(left_names):
            a = np.asarray(left.variables[name][:])
            b = np.asarray(right.variables[name][:])
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"restart variable {name} changed shape or dtype")
            a_words = np.ascontiguousarray(a).view(np.uint8).reshape(
                a.size, a.dtype.itemsize)
            b_words = np.ascontiguousarray(b).view(np.uint8).reshape(
                b.size, b.dtype.itemsize)
            unequal = int(np.count_nonzero(np.any(a_words != b_words, axis=1)))
            unequal_words += unequal
            compared_words += int(a.size)
            if unequal:
                maximum = None
                if a.dtype.kind in "iufc":
                    differences = np.abs(a.astype(np.complex128)
                                         - b.astype(np.complex128))
                    finite = np.isfinite(differences)
                    if bool(np.any(finite)):
                        maximum = float(np.max(differences[finite]))
                moved_variables.append({
                    "variable": name,
                    "unequal_words": unequal,
                    "max_abs": maximum,
                })
    if compared_words <= 0:
        raise ValueError("restart passivity compared zero numeric words")
    return {
        "provider": "nemo",
        "status": "PASS" if unequal_words == 0 else "REFUSE",
        "unequal_words": unequal_words,
        "compared_words": compared_words,
        "compared_variables": sorted(left_names),
        "moved_variables": moved_variables,
        "control_restart": control.name,
        "instrumented_restart": instrumented.name,
        "control_sha256": sha256(control),
        "instrumented_sha256": sha256(instrumented),
    }


def verify_short(
    run_root: Path, control_restart: Path, instrumented_restart: Path,
) -> dict[str, object]:
    """Verify the minimal day-30 rerun before the 180-day acquisition."""
    path = run_root / "oracle_gyre_r14_budget_day0030.bin"
    record = read_record(path, expected_day=30)
    extras = sorted(run_root.glob("oracle_gyre_r14_budget_day*.bin"))
    if extras != [path]:
        raise ValueError(
            f"short verification found {len(extras)} records, expected day 30 only")
    passive = compare_restarts(control_restart, instrumented_restart)
    if passive["status"] != "PASS":
        raise ValueError(
            f"write-only recorder moved {passive['unequal_words']} restart words")
    report = {
        "status": "PASS",
        "purpose": "round15_day30_closure_verification_only",
        "day": 30,
        "step": 720,
        "record": path.name,
        "record_sha256": record["sha256"],
        "closure_bar_abs_m_s": CLOSURE_BAR_ABS_M_S,
        "closure_max_abs_m_s": record["closure_max_abs_m_s"],
        "wet_words": int(np.count_nonzero(record["mask"])),
        "passive_identity": passive,
    }
    (run_root / "momentum_budget_short_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def verify_registration(
    preregistration: Path, registration_commit: str, producer_commit: str,
) -> None:
    """Require exact registered content and a producer descended from it."""
    if sha256(preregistration) != REGISTRATION_SOURCE_SHA256:
        raise ValueError("preregistration content differs from its registered hash")
    repo = Path(__file__).resolve().parents[5]
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor",
         registration_commit, producer_commit],
        check=False, capture_output=True)
    if ancestor.returncode != 0:
        raise ValueError(
            f"producer {producer_commit} does not descend from registration "
            f"{registration_commit}")
    registered_result = subprocess.run(
        ["git", "-C", str(repo), "show",
         f"{registration_commit}:{PREREGISTRATION_REPO_PATH}"],
        check=False, capture_output=True)
    if (registered_result.returncode != 0
            or hashlib.sha256(registered_result.stdout).hexdigest()
            != REGISTRATION_SOURCE_SHA256):
        raise ValueError("registration commit does not contain the pinned document")


def package(
    run_root: Path, control_restart: Path, instrumented_restart: Path,
    preregistration: Path, registration_commit: str, producer_commit: str,
    resolved_namelist: Path, writer_source_manifest: Path,
    binary: Path, nemo_engine_commit: str,
    precision_flavour: str = "vector-math-conda",
) -> dict[str, object]:
    if len(registration_commit) != 40 or any(
            char not in "0123456789abcdef" for char in registration_commit):
        raise ValueError("registration commit must be a lowercase forty-character SHA")
    if len(producer_commit) != 40 or any(
            char not in "0123456789abcdef" for char in producer_commit):
        raise ValueError("producer commit must be a lowercase forty-character SHA")
    if len(nemo_engine_commit) != 40 or any(
            char not in "0123456789abcdef" for char in nemo_engine_commit):
        raise ValueError(
            "NEMO engine commit must be a lowercase forty-character SHA")
    if precision_flavour not in ("vector-math-conda", "scalar-math-conda"):
        raise ValueError(
            f"unsupported NEMO precision flavour {precision_flavour!r}")
    verify_registration(preregistration, registration_commit, producer_commit)
    records = []
    for day in range(FIRST_DAY, LAST_DAY + 1):
        path = run_root / f"oracle_gyre_r14_budget_day{day:04d}.bin"
        if not path.is_file():
            raise ValueError(f"missing {path.name}")
        records.append(read_record(path, expected_day=day))
    extras = sorted(run_root.glob("oracle_gyre_r14_budget_day*.bin"))
    if len(extras) != len(records):
        raise ValueError(
            f"found {len(extras)} budget records, expected {len(records)}")
    mask = np.asarray(records[0]["mask"])
    if any(not np.array_equal(record["mask"], mask) for record in records):
        raise ValueError("surface-V mask changes across budget records")

    passive = compare_restarts(control_restart, instrumented_restart)
    if passive["status"] != "PASS":
        raise ValueError(
            f"write-only recorder moved {passive['unequal_words']} restart words")
    passive_name = "momentum_budget_passive_identity.json"
    (run_root / passive_name).write_text(
        json.dumps(passive, indent=2, sort_keys=True) + "\n")

    control_binary = control_restart.parent / "nemo"
    instrumented_binary = instrumented_restart.parent / "nemo"
    control_binary_sha256 = sha256(control_binary)
    instrumented_binary_sha256 = sha256(instrumented_binary)
    binary_sha256 = sha256(binary)
    if not (control_binary_sha256 == instrumented_binary_sha256
            == binary_sha256):
        raise ValueError("control and instrumented runs did not use one binary")

    prereg = {
        "terms": list(TERMS),
        "closure_bar_abs_m_s": CLOSURE_BAR_ABS_M_S,
        "registered_before_data": True,
        "registration_commit": registration_commit,
        "source": str(preregistration),
        "source_sha256": sha256(preregistration),
    }
    prereg_name = "momentum_budget_preregistration.json"
    (run_root / prereg_name).write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n")

    namelist_name = "resolved_output.namelist.dyn"
    source_name = "writer_source.sha256"
    namelist_bytes = resolved_namelist.read_bytes()
    (run_root / namelist_name).write_bytes(namelist_bytes)
    (run_root / source_name).write_bytes(writer_source_manifest.read_bytes())

    data_name = "momentum_budget_terms.npz"
    arrays = {
        f"surface_v_{name}_daily_mean": np.stack([
            np.asarray(record["terms"])[index]
            for record in records])
        for index, name in enumerate(TERMS)
    }
    with (run_root / data_name).open("wb") as stream:
        np.savez_compressed(
            stream,
            day=np.asarray([record["day"] for record in records], dtype=np.int64),
            surface_v_wet_mask=mask,
            **arrays)

    contract = {
        "provider": "nemo",
        "status": "complete",
        "producer_commit": producer_commit,
        "nemo_engine_commit": nemo_engine_commit,
        "deck": "r4_smag",
        "precision_flavour": precision_flavour,
        "cadence": {
            "dt_s": 3600.0,
            "steps_per_day": STEPS_PER_DAY,
            "first_day": FIRST_DAY,
            "last_day": LAST_DAY,
            "first_step": 697,
            "last_step": 4320,
            "samples_per_record": STEPS_PER_DAY,
            "records": LAST_DAY - FIRST_DAY + 1,
        },
        "resolved_namelist_artifact": namelist_name,
        "resolved_namelist_sha256": sha256(run_root / namelist_name),
        "writer_source_manifest_artifact": source_name,
        "writer_source_manifest_sha256": sha256(run_root / source_name),
        "binary_sha256": binary_sha256,
        "control_binary_sha256": control_binary_sha256,
        "instrumented_binary_sha256": instrumented_binary_sha256,
        "same_binary_control": True,
        "activation_environment": "GYRE_R14_BUDGET=1",
        "control_environment": "GYRE_R14_BUDGET=0",
        "window_days": [FIRST_DAY, LAST_DAY],
        "terms": list(TERMS),
        "production_online": True,
        "external_mode_included": True,
        "implicit_zdf_stress_separated": True,
        "closure_total_recorded": True,
        "passive_trajectory_identity": True,
        "closure_bar_abs_m_s": CLOSURE_BAR_ABS_M_S,
        "data_artifact": data_name,
        "data_sha256": sha256(run_root / data_name),
        "passive_identity_artifact": passive_name,
        "passive_identity_sha256": sha256(run_root / passive_name),
        "preregistration_artifact": prereg_name,
        "preregistration_sha256": sha256(run_root / prereg_name),
        "record_sha256": {
            f"day{record['day']:04d}": record["sha256"]
            for record in records
        },
        "closure_max_abs_m_s": max(
            float(record["closure_max_abs_m_s"]) for record in records),
    }
    (run_root / "momentum_budget_manifest.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-restart", type=Path, required=True)
    parser.add_argument("--instrumented-restart", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--resolved-namelist", type=Path, required=True)
    parser.add_argument("--writer-source-manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--nemo-engine-commit", required=True)
    parser.add_argument(
        "--precision-flavour",
        choices=("vector-math-conda", "scalar-math-conda"),
        default="vector-math-conda",
    )
    parser.add_argument("--verify-short", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_short:
            contract = verify_short(
                args.run_root, args.control_restart,
                args.instrumented_restart)
        else:
            contract = package(
                args.run_root, args.control_restart,
                args.instrumented_restart, args.preregistration,
                args.registration_commit, args.producer_commit,
                args.resolved_namelist, args.writer_source_manifest,
                args.binary, args.nemo_engine_commit,
                args.precision_flavour)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
