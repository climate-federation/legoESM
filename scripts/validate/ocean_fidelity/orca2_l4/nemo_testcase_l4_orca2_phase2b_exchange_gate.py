#!/usr/bin/env python3
"""Validate the Phase-2b final ORCA2 ocean surface-input record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)

NX, NY, NTR, NCLASSES, HALO = 94, 152, 2, 10, 2
NFULL, NREDUCED_2D, NHALO1, NREDUCED_3D = 20, 12, 1, 2
MAGIC = "NEMO_L4_SBCIN_1"
RECORD = "oracle_ocean_surface_input_kt00000001.bin"
HEADER = (
    1, 1, 1, NX, NY, NTR, NCLASSES, HALO,
    NFULL, NREDUCED_2D, NHALO1, NREDUCED_3D, 64,
)
HEADER_FMT = "=13i"
FIELDS = (
    ("utau", "full"), ("vtau", "full"), ("utauU", "full"),
    ("vtauV", "full"), ("utau_b", "full"), ("vtau_b", "full"),
    ("utau_icb", "full"), ("vtau_icb", "full"),
    ("taum", "reduced"), ("wndm", "reduced"),
    ("qsr", "reduced"), ("qns", "reduced"), ("qns_b", "reduced"),
    ("qsr_tot", "reduced"), ("qns_tot", "reduced"),
    ("emp", "full"), ("emp_b", "full"),
    ("sfx", "reduced"), ("sfx_b", "reduced"),
    ("emp_tot", "reduced"), ("fwfice", "reduced"),
    ("rnf", "full"), ("rnf_b", "full"), ("fwficb", "reduced"),
    ("fr_i", "full"), ("snwice_mass", "full"),
    ("snwice_mass_b", "full"), ("snwice_fmass", "full"),
    ("rCdU_ice", "halo1"), ("icb_calving", "full"),
    ("icb_calving_hflx", "full"), ("icb_floating_melt", "full"),
    ("icb_stored_heat", "full"),
    ("rnf_tsc", "reduced3d"), ("rnf_tsc_b", "reduced3d"),
)
INACTIVE_ICEBERG_FIELDS = {
    "utau_icb", "vtau_icb", "icb_calving", "icb_calving_hflx",
    "icb_floating_melt", "icb_stored_heat",
}
class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_count(header: tuple[int, ...]) -> int:
    (_, _, _, nx, ny, ntr, _, halo,
     nfull, nreduced2d, nhalo1, nreduced3d, bits) = header
    require(bits == 64, f"word size {bits} != 64")
    observed = {
        "full": sum(kind == "full" for _, kind in FIELDS),
        "reduced": sum(kind == "reduced" for _, kind in FIELDS),
        "halo1": sum(kind == "halo1" for _, kind in FIELDS),
        "reduced3d": sum(kind == "reduced3d" for _, kind in FIELDS),
    }
    require((nfull, nreduced2d, nhalo1, nreduced3d) ==
            (observed["full"], observed["reduced"], observed["halo1"], observed["reduced3d"]),
            "allocation-class counts do not match write list")
    reduced = (nx - 2 * halo) * (ny - 2 * halo)
    halo1 = (nx - 2 * (halo - 1)) * (ny - 2 * (halo - 1))
    return nfull * nx * ny + (nreduced2d + nreduced3d * ntr) * reduced + nhalo1 * halo1


def validate_surface(
    path: Path,
    expected_sha256: str | None = None,
    *,
    icebergs_off: bool = False,
) -> dict:
    with path.open("rb", buffering=0) as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated magic")
        require(raw_magic.decode("ascii").rstrip() == MAGIC, "bad magic")
        raw_header = handle.read(struct.calcsize(HEADER_FMT))
        require(len(raw_header) == struct.calcsize(HEADER_FMT), "truncated header")
    header = struct.unpack(HEADER_FMT, raw_header)
    require(header == HEADER, f"header {header} != {HEADER}")
    count = payload_count(header)
    offset = 16 + struct.calcsize(HEADER_FMT)
    expected_bytes = offset + count * 8
    require(path.stat().st_size == expected_bytes,
            f"size {path.stat().st_size} != derived schema {expected_bytes}")
    payload = np.memmap(path, dtype=np.float64, mode="r", offset=offset, shape=(count,))
    require(bool(np.isfinite(payload).all()), "non-finite payload")
    digest = sha256(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "record SHA-256 mismatch")

    cursor = 0
    fields = []
    class_sizes = {
        "full": NX * NY,
        "reduced": (NX - 2 * HALO) * (NY - 2 * HALO),
        "halo1": (NX - 2 * (HALO - 1)) * (NY - 2 * (HALO - 1)),
        "reduced3d": NTR * (NX - 2 * HALO) * (NY - 2 * HALO),
    }
    for name, allocation in FIELDS:
        size = class_sizes[allocation]
        values = payload[cursor:cursor + size]
        row = {"name": name, "allocation": allocation, "values": int(values.size),
               "finite": int(np.isfinite(values).sum())}
        if allocation == "reduced3d":
            row["levels"] = NTR
        if icebergs_off and name in INACTIVE_ICEBERG_FIELDS:
            nonzero = int(np.count_nonzero(values))
            row["nonzero"] = nonzero
            require(nonzero == 0, f"inactive iceberg field is nonzero: {name}")
        fields.append(row)
        cursor += size
    require(cursor == count, f"schema walk consumed {cursor}/{count}")
    return {"magic": MAGIC, "header": list(header), "payload_f64": count,
            "bytes": expected_bytes, "sha256": digest, "fields": fields}


def validate_legacy_records(
    run_dir: Path, record_manifest: Path | None = None
) -> dict:
    """Run the frozen 90-stream Phase-1 parser despite this run's one extra file."""
    with tempfile.TemporaryDirectory(prefix="orca2-l4-phase2b-legacy-") as td:
        view = Path(td)
        for name in phase1.expected_inventory():
            os.symlink(run_dir / name, view / name)
        return phase1.validate_records(view, record_manifest)


def planted_controls(path: Path, *, icebergs_off: bool = False) -> dict[str, str]:
    expected_digest = sha256(path)
    results: dict[str, str] = {}

    def expect(name: str, mutate, *, bind_digest: bool = True) -> None:
        with tempfile.TemporaryDirectory(prefix=f"orca2-l4-phase2b-{name}-") as td:
            altered = Path(td) / RECORD
            shutil.copyfile(path, altered)
            mutate(altered)
            try:
                validate_surface(
                    altered,
                    expected_digest if bind_digest else None,
                    icebergs_off=icebergs_off,
                )
            except (GateError, OSError, UnicodeError, struct.error, ValueError):
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"plant {name} did not fail")

    def patch(target: Path, offset: int, data: bytes) -> None:
        with target.open("r+b") as handle:
            handle.seek(offset)
            handle.write(data)

    expect("bad_magic", lambda target: patch(target, 0, b"X"))
    expect("bad_derived_count", lambda target: patch(target, 16 + 8 * 4, struct.pack("=i", 19)))
    expect("truncated_payload", lambda target: target.write_bytes(target.read_bytes()[:-8]))
    payload_offset = 16 + struct.calcsize(HEADER_FMT)
    expect(
        "nan_payload",
        lambda target: patch(target, payload_offset, struct.pack("=d", float("nan"))),
    )

    def one_ulp(target: Path) -> None:
        # First owned payload value; integrity pin binds an otherwise schema-valid mutation.
        with target.open("r+b") as handle:
            handle.seek(payload_offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(payload_offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))

    expect("one_ulp_active_field", one_ulp)
    if icebergs_off:
        class_sizes = {
            "full": NX * NY,
            "reduced": (NX - 2 * HALO) * (NY - 2 * HALO),
            "halo1": (NX - 2 * (HALO - 1)) * (NY - 2 * (HALO - 1)),
            "reduced3d": NTR * (NX - 2 * HALO) * (NY - 2 * HALO),
        }
        cursor = 0
        for field, allocation in FIELDS:
            if field == "utau_icb":
                break
            cursor += class_sizes[allocation]
        inactive_offset = payload_offset + cursor * 8
        expect(
            "inactive_iceberg_nonzero",
            lambda target: patch(
                target, inactive_offset, struct.pack("=d", 1.0)
            ),
            bind_digest=False,
        )
    return results


def validate_variant_namelist(path: Path) -> None:
    values = []
    for line in path.read_text().splitlines():
        code = line.split("!", 1)[0]
        match = re.match(
            r"\s*ln_icebergs\s*=\s*(\.true\.|\.false\.)", code,
            flags=re.IGNORECASE,
        )
        if match:
            values.append(match.group(1).lower())
    require(values == [".false."], f"variant ln_icebergs values {values}")


def _kt1_prefix(path: Path, family: str) -> tuple[bytes, int]:
    schemas = {
        "bulk": ("NEMO_L3BULK_001", "=5i"),
        "exchange": ("NEMO_L3XCHG_001", "=6i"),
        "thermodynamics": ("NEMO_L3THD_001", "=11i"),
        "zdf_inputs": ("NEMO_L3ZIN_002", "=6i"),
    }
    magic, fmt = schemas[family]
    last = 0
    frames = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            raw_magic = handle.read(16)
            if not raw_magic:
                break
            require(len(raw_magic) == 16, f"{path.name}: truncated magic")
            require(
                raw_magic.decode("ascii").rstrip() == magic,
                f"{path.name}: bad magic",
            )
            raw_header = handle.read(struct.calcsize(fmt))
            require(
                len(raw_header) == struct.calcsize(fmt),
                f"{path.name}: truncated header",
            )
            header = struct.unpack(fmt, raw_header)
            kt = header[1]
            if family == "bulk":
                count = header[3]
            elif family == "exchange":
                _, _, nx, ny, _, _ = header
                interior = (nx - 4) * (ny - 4)
                count = 78 * interior + 9 * nx * ny + (nx - 2) * (ny - 2)
            elif family == "thermodynamics":
                _, _, _, kind, nx, ny, jpl, nli, nls, npti, _ = header
                count = (
                    (4 + 2 * nli + nls) * npti
                    if kind else (9 + 2 * nli + nls) * jpl * nx * ny
                )
            else:
                count = header[5]
            require(count >= 0, f"{path.name}: negative payload count")
            payload = handle.read(count * 8)
            require(len(payload) == count * 8, f"{path.name}: truncated payload")
            if kt == 1:
                last = handle.tell()
                frames += 1
            else:
                break
    require(frames > 0, f"{path.name}: no kt=1 frames")
    return path.read_bytes()[:last], frames


def _characterize_stage1_tracer(variant: Path, shipped: Path) -> dict[str, object]:
    def payload(path: Path) -> np.ndarray:
        with path.open("rb") as handle:
            require(
                handle.read(16).decode("ascii").rstrip() == "NEMO_L2_RKTRA_1",
                f"{path.name}: bad magic",
            )
            header = struct.unpack("=11i", handle.read(44))
            require(
                header == (1, 1, 1, 1, 1, 3, 3, NX, NY, 31, 64),
                f"{path.name}: bad header {header}",
            )
            values = np.fromfile(handle, dtype=np.float64)
        require(
            values.size == 15 * phase1.N3 + 3 * phase1.N2,
            f"{path.name}: bad payload size",
        )
        return values

    variant_values = payload(
        variant / "oracle_rktracer_operands_kt00000001_s1.bin"
    )
    shipped_values = payload(
        shipped / "oracle_rktracer_operands_kt00000001_s1.bin"
    )
    n3 = phase1.N3
    n2 = phase1.N2
    blocks = (
        ("Krhs_entry_T", n3), ("Krhs_entry_S", n3),
        ("zFu", n3), ("zFv", n3), ("zFw", n3),
        ("Krhs_after_tra_adv_T", n3), ("Krhs_after_tra_adv_S", n3),
        ("Krhs_after_tra_sbc_T", n3), ("Krhs_after_tra_sbc_S", n3),
        ("Kbb_T", n3), ("Kbb_S", n3),
        ("Kmm_T", n3), ("Kmm_S", n3),
        ("Kaa_T", n3), ("Kaa_S", n3),
        ("r3t_Kbb", n2), ("r3t_Kmm", n2), ("r3t_Kaa", n2),
    )
    rows = {}
    cursor = 0
    for name, count in blocks:
        candidate = variant_values[cursor:cursor + count]
        reference = shipped_values[cursor:cursor + count]
        different = int(np.count_nonzero(candidate != reference))
        delta = candidate - reference
        rows[name] = {
            "different": different,
            "count": int(count),
            "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        }
        cursor += count
    require(cursor == variant_values.size, "stage-1 tracer block walk incomplete")
    return {
        "rows": rows,
        "disposition": "POST_ICB_EFFECT_DIAGNOSTIC_NOT_AN_IDENTITY_GATE",
        "source": "stprk3_stg.F90:669-679",
    }


def validate_variant_vs_shipped(variant: Path, shipped: Path) -> dict[str, object]:
    exact_files = (
        "oracle_step_entry_kt00000001.bin",
        "oracle_si3_prather_kt00000001_s0.bin",
        "oracle_si3_prather_kt00000001_s1.bin",
    )
    rows = {}
    for name in exact_files:
        variant_bytes = (variant / name).read_bytes()
        shipped_bytes = (shipped / name).read_bytes()
        require(variant_bytes == shipped_bytes, f"pre-icb record differs: {name}")
        rows[name] = {
            "status": "EXACT_BYTES",
            "bytes": len(variant_bytes),
            "sha256": sha256(variant / name),
        }
    for family, name in (
        ("bulk", "oracle_si3_bulk_operands.bin"),
        ("exchange", "oracle_si3_exchange_frames.bin"),
        ("thermodynamics", "oracle_si3_thd_frames.bin"),
        ("zdf_inputs", "oracle_si3_zdf_inputs.bin"),
    ):
        variant_prefix, variant_frames = _kt1_prefix(variant / name, family)
        shipped_prefix, shipped_frames = _kt1_prefix(shipped / name, family)
        require(variant_frames == shipped_frames, f"{name}: kt=1 frame count differs")
        require(variant_prefix == shipped_prefix, f"pre-icb prefix differs: {name}")
        rows[name] = {
            "status": "KT1_PREFIX_EXACT_BYTES",
            "frames": variant_frames,
            "bytes": len(variant_prefix),
            "sha256": hashlib.sha256(variant_prefix).hexdigest(),
        }
    common = sorted(
        path.name for path in variant.glob("oracle_*.bin")
        if (shipped / path.name).is_file()
    )
    post_effect = {
        name: {
            "status": (
                "EXACT_BYTES"
                if sha256(variant / name) == sha256(shipped / name)
                else "DIFFERS_AFTER_ICB_BOUNDARY"
            ),
            "variant_sha256": sha256(variant / name),
            "shipped_sha256": sha256(shipped / name),
        }
        for name in common
    }
    return {
        "pre_icb": rows,
        "post_effect_common_record_inventory": post_effect,
        "stage1_tracer_diagnostic": _characterize_stage1_tracer(variant, shipped),
        "shipped_surface_record": "ABSENT_NOT_IN_PHASE1_INVENTORY",
        "interpretation": (
            "Only records written before sbcmod's first possible icb_stp call are "
            "identity-gated. Later records are diagnostics because the permitted "
            "emp/qns perturbation propagates through the ocean step."
        ),
    }


def variant_shipped_plant(variant: Path, shipped: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="orca2-l4-variant-plant-") as td:
        view = Path(td)
        for source in variant.glob("oracle_*.bin"):
            os.symlink(source, view / source.name)
        target = view / "oracle_step_entry_kt00000001.bin"
        target.unlink()
        shutil.copyfile(variant / target.name, target)
        with target.open("r+b") as handle:
            handle.seek(64)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(64)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
        try:
            validate_variant_vs_shipped(view, shipped)
        except (GateError, OSError, UnicodeError, struct.error, ValueError):
            return "PASS_NONZERO"
    raise GateError("variant-vs-shipped plant did not fail")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--expected-record-sha256")
    parser.add_argument("--legacy-record-manifest", type=Path)
    parser.add_argument("--shipped-record-root", type=Path)
    parser.add_argument("--plant-controls", action="store_true")
    parser.add_argument("--variant-icebergs-off", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = args.run_dir / RECORD
        if args.variant_icebergs_off:
            validate_variant_namelist(args.run_dir / "namelist_cfg")
        result = {
            "oracle_label": (
                "VARIANT" if args.variant_icebergs_off else "SHIPPED_DECK_RECORD"
            ),
            "surface_input": validate_surface(
                record,
                args.expected_record_sha256,
                icebergs_off=args.variant_icebergs_off,
            ),
            "legacy_records": validate_legacy_records(
                args.run_dir, args.legacy_record_manifest
            ),
        }
        if args.control is not None:
            result["identity"] = phase1.validate_identity(args.control, args.run_dir)
        if args.shipped_record_root is not None:
            result["variant_vs_shipped"] = validate_variant_vs_shipped(
                args.run_dir, args.shipped_record_root
            )
        if args.plant_controls:
            phase1.require(
                args.control is not None and args.legacy_record_manifest is not None,
                "plants require --control and --legacy-record-manifest",
            )
            result["legacy_planted_controls"] = phase1.planted_controls(
                args.run_dir,
                args.legacy_record_manifest,
                args.control,
                args.run_dir,
            )
            result["surface_planted_controls"] = planted_controls(
                record, icebergs_off=args.variant_icebergs_off
            )
            if args.shipped_record_root is not None:
                result["variant_shipped_planted_control"] = (
                    variant_shipped_plant(args.run_dir, args.shipped_record_root)
                )
        result["status"] = "PASS"
    except (GateError, phase1.GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
