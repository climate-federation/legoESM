#!/usr/bin/env python3
"""Structural admission for the passive ORCA2 round-328 dyn_zdf record."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round326_hmi_gate import (  # noqa: E402
    objects_equal,
)

MAGIC = "NEMO_L4_R328ZDF"
HEADER_NAMES = (
    "version", "kt", "stage", "kbb", "kmm", "krhs", "kaa", "rank",
    "nx", "ny", "nz", "ntsi", "ntsj", "real_bits", "n_fields",
)
STATE_PHASES = ("explicit", "baro_sub", "bottom_rhs", "top_rhs")
MATRIX_PHASES = ("base_zdf", "after_zad", "after_ldf", "bottom_diag", "top_diag")
MATRIX_TERMS = ("low", "diag", "up")
EXPECTED_ORDER = (
    "before_u", "before_v", "rhs_u", "rhs_v", "avm_input",
    "r3u_kmm", "r3v_kmm",
    "r3u_kaa", "r3v_kaa",
    "barotropic_u", "barotropic_v", "rCdU_bot", "rCdU_top", "utauU", "vtauV",
    *(f"{phase}_{component}" for phase in STATE_PHASES for component in "uv"),
    *(f"{component}_{phase}_{term}" for component in "uv"
      for phase in MATRIX_PHASES for term in MATRIX_TERMS),
    "solve_u", "solve_v", "corrected_u", "corrected_v",
)
PLANTS = ("header", "field-order", "truncation", "stamp", "avm", "bottom", "top", "solve", "endpoint")
OBSERVER_UNITS = ("dynzdf", "stprk3_stg")
WRITER_UNIT = "dynzdf_round328_writer"
BUILD_CONFIG_FILES = ("arch_nemo.fcm", "cpp.fcm", "bld.cfg")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(stream: io.BytesIO, count: int, what: str) -> bytes:
    value = stream.read(count)
    require(len(value) == count, f"truncated {what}")
    return value


def record_name(rank: int) -> str:
    return f"oracle_dynzdf_kt00000001_s3_r{rank:04d}.bin"


def _expected_extent(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    nx, ny, nz = header["nx"], header["ny"], header["nz"]
    if name == "avm_input":
        return nx + 1, ny + 1, nz + 1
    if name in {"rCdU_bot", "rCdU_top"}:
        return nx + 1, ny + 1, 1
    if name in {"r3u_kmm", "r3v_kmm", "r3u_kaa", "r3v_kaa",
                "barotropic_u", "barotropic_v", "utauU", "vtauV"}:
        return nx, ny, 1
    return nx, ny, nz


def _checksum(values: np.ndarray) -> int:
    words = np.ascontiguousarray(values).view(np.uint64).reshape(-1)
    return int(np.bitwise_xor.reduce(words, initial=np.uint64(0)))


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(f"={len(HEADER_NAMES)}i", _take(stream, 4 * len(HEADER_NAMES), "header"))
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["stage"] = 2
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["stage"]) == (1, 1, 3), "wrong version/kt/stage")
    require((header["kbb"], header["kmm"], header["krhs"], header["kaa"]) == (1, 2, 3, 3),
            "wrong RK indices")
    require(header["real_bits"] == 64, "record is not fp64")
    require(header["n_fields"] == len(EXPECTED_ORDER), "wrong field count")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = _take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack("=7i", _take(stream, 28, f"{name} metadata"))
        require(rank in (2, 3), f"{name}: bad array rank {rank}")
        expected = _expected_extent(name, header)
        require((n1, n2, n3) == expected, f"{name}: extent {(n1, n2, n3)} != {expected}")
        require((i0, j0, k0) == (header["ntsi"], header["ntsj"], 1), f"{name}: wrong origin")
        checksum, = struct.unpack("=Q", _take(stream, 8, f"{name} checksum"))
        array = np.frombuffer(_take(stream, n1 * n2 * n3 * 8, f"{name} payload"), dtype="=f8").copy()
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name] = array.reshape((n1, n2, n3), order="F")
        checksums[name] = checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == EXPECTED_ORDER, "field order differs from NEMO source order")
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _changed(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.any(a.view(np.uint64) != b.view(np.uint64)))


def _verify_checksums(record: dict) -> None:
    for name, values in record["fields"].items():
        require(_checksum(values) == record["checksums"][name],
                f"{name}: payload checksum differs from writer")


def _one_ulp(values: np.ndarray) -> np.ndarray:
    planted = values.copy()
    planted.flat[0] = np.nextafter(planted.flat[0], np.float64(np.inf))
    return planted


def read_admitted_state(path: Path, *, kt: int, stage: int | None) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Read an admitted rank-local stage/entry frame without importing a scorer."""
    raw = path.read_bytes()
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "state magic").decode("ascii").rstrip()
    if stage is None:
        values = struct.unpack("=8i", _take(stream, 32, "entry header"))
        version, got_kt, level, nx, ny, nz, ntr, real_bits = values
        require(magic == "NEMO_L1_ENTRY_1", f"entry magic {magic!r}")
        require((version, got_kt, level, ntr, real_bits) == (1, kt, 3, 2, 64),
                f"entry header {values}")
    else:
        values = struct.unpack("=9i", _take(stream, 36, "stage header"))
        version, got_kt, got_stage, level, nx, ny, nz, ntr, real_bits = values
        require(magic == "NEMO_L1_STAGE_1", f"stage magic {magic!r}")
        require((version, got_kt, got_stage, level, ntr, real_bits)
                == (1, kt, stage, 3, 2, 64), f"stage header {values}")
    payload = np.frombuffer(_take(stream, (4 * nx * ny * nz + nx * ny) * 8,
                                  "stage-3 payload"), dtype="=f8")
    require(stream.read(1) == b"", "stage-3 frame has trailing bytes")
    require(bool(np.all(np.isfinite(payload))), "stage-3 frame has non-finite payload")
    n3 = nx * ny * nz
    return ({"nx": nx, "ny": ny, "nz": nz},
            {"u": payload[2 * n3:3 * n3].reshape((nx, ny, nz), order="F"),
             "v": payload[3 * n3:4 * n3].reshape((nx, ny, nz), order="F")})


def _endpoint_check(record: dict, state_path: Path, *, kt: int, stage: int | None,
                    plant: bool, label: str) -> None:
    dims, state = read_admitted_state(state_path, kt=kt, stage=stage)
    h = record["header"]
    i0, j0 = h["ntsi"] - 1, h["ntsj"] - 1
    require(i0 + h["nx"] <= dims["nx"] and j0 + h["ny"] <= dims["ny"]
            and h["nz"] < dims["nz"], f"dyn_zdf endpoint lies outside {label} frame")
    expected_u = state["u"][i0:i0 + h["nx"], j0:j0 + h["ny"], :h["nz"]]
    expected_v = state["v"][i0:i0 + h["nx"], j0:j0 + h["ny"], :h["nz"]]
    # Canonical frames write every dry cell as +0.0, while the raw dyn_zdf
    # endpoint can retain -0.0 there.  Use each frame's nonzero support as its
    # own wet mask so A4 does not reject an otherwise correct record.
    wet_u = expected_u != 0.0
    wet_v = expected_v != 0.0
    if plant:
        wet_index = np.flatnonzero(wet_u)
        require(bool(wet_index.size), f"admitted {label} U endpoint has no wet cells")
        expected_u = expected_u.copy()
        expected_u.flat[wet_index[0]] = np.nextafter(
            expected_u.flat[wet_index[0]], np.float64(np.inf))
    require(not _changed(record["fields"]["corrected_u"][wet_u], expected_u[wet_u]),
            f"corrected_u differs from the admitted {label} endpoint")
    require(not _changed(record["fields"]["corrected_v"][wet_v], expected_v[wet_v]),
            f"corrected_v differs from the admitted {label} endpoint")


def run_gate(root: Path, expect_commit: str, plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    records = {}
    for rank in (0, 1):
        path = root / record_name(rank)
        require(path.is_file(), f"missing {path.name}")
        record = read_record(path, plant=plant if rank == 0 else None)
        require(record["header"]["rank"] == rank, f"{path.name}: wrong rank")
        records[rank] = record
    fields = records[0]["fields"]
    if plant == "avm":
        fields["avm_input"] = _one_ulp(fields["avm_input"])
    if plant == "bottom":
        fields["rCdU_bot"] = _one_ulp(fields["rCdU_bot"])
    if plant == "top":
        fields["rCdU_top"] = _one_ulp(fields["rCdU_top"])
    if plant == "solve":
        fields["solve_u"] = _one_ulp(fields["solve_u"])
    for rank, record in records.items():
        _verify_checksums(record)
        f = record["fields"]
        require(bool(np.any(f["avm_input"] != 0.0)), f"rank {rank}: inactive avm channel")
        require(_changed(f["u_after_ldf_diag"], f["u_bottom_diag_diag"]) or
                _changed(f["v_after_ldf_diag"], f["v_bottom_diag_diag"]),
                f"rank {rank}: bottom drag changes no matrix diagonal")
        require(_changed(f["u_bottom_diag_diag"], f["u_top_diag_diag"]) or
                _changed(f["v_bottom_diag_diag"], f["v_top_diag_diag"]),
                f"rank {rank}: top drag changes no matrix diagonal")
        require(_changed(f["top_rhs_u"], f["solve_u"]) or _changed(f["top_rhs_v"], f["solve_v"]),
                f"rank {rank}: implicit solve is inactive")
        require(_changed(f["solve_u"], f["corrected_u"]) or _changed(f["solve_v"], f["corrected_v"]),
                f"rank {rank}: post-dyn_zdf barotropic correction is inactive")
    stage_path = root / "oracle_stage_kt00000001_s3.bin"
    require(stage_path.is_file(), f"missing admitted stage-3 frame {stage_path.name}")
    entry_paths = (root / "oracle_step_entry_kt00000002.bin",
                   root / "oracle_step_entry_rank0001_kt00000002.bin")
    require(all(path.is_file() for path in entry_paths), "missing admitted kt-2 entry endpoint")
    _endpoint_check(records[0], stage_path, kt=1, stage=3,
                    plant=plant == "endpoint", label="stage-3")
    for rank, path in enumerate(entry_paths):
        _endpoint_check(records[rank], path, kt=2, stage=None, plant=False,
                        label=f"rank-{rank} kt-2 entry")
    return {
        "format": "nemo-testcase-l4-orca2-round328-dynzdf-v1",
        "status": "PASS_R328_DYNZDF_STRUCTURE",
        "producer_commit": produced,
        "records": {record_name(rank): record["sha256"] for rank, record in records.items()},
    }


def _regular_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _observer_only(ref: list[str], new: list[str], inserted: list[str], unit: str) -> None:
    added: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, new, autojunk=False).get_opcodes():
        require(tag in ("equal", "insert"),
                f"{unit}.f90 {tag}s record-build lines {i1 + 1}-{i2}")
        added += new[j1:j2] if tag == "insert" else []
    # FCM runs CPP with ``-P -traditional`` before writing BLD/ppsrc.  CPP
    # removes blank lines added by a unified patch, so raw patch blanks are not
    # part of the compiled source and cannot be required in its insertion list.
    compiled_insertions = [line for line in inserted if line.strip()]
    require(added == compiled_insertions,
            f"{unit}.f90 insertions differ from its committed patch")


def _strip_preprocessed_observers(
        lines: list[str], begin: str, end: str, unit: str,
        expected_nonblank: list[str] | None = None) -> tuple[list[str], int]:
    """Remove explicitly marked observer blocks from a real preprocessed unit."""
    stripped: list[str] = []
    inside = False
    blocks = 0
    observed_nonblank: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        marker = line.strip()
        if marker == begin:
            require(not inside, f"{unit}.f90 nested observer block at line {line_number}")
            inside = True
            blocks += 1
            continue
        if marker == end:
            require(inside, f"{unit}.f90 unmatched observer end at line {line_number}")
            inside = False
            continue
        if not inside:
            stripped.append(line)
        elif line.strip():
            observed_nonblank.append(line.strip())
    require(not inside, f"{unit}.f90 unterminated observer block")
    require(blocks > 0, f"{unit}.f90 has no marked observer block")
    if expected_nonblank is not None:
        require(observed_nonblank == expected_nonblank,
                f"{unit}.f90 marked observer contents differ from its committed patch")
    return stripped, blocks


def _marked_patch_contents(lines: list[str], begin: str, end: str, unit: str) -> list[str]:
    """Return nonblank patch insertions inside markers; reject any outside them."""
    contents: list[str] = []
    inside = False
    blocks = 0
    for line in lines:
        marker = line.strip()
        if marker == begin:
            require(not inside, f"{unit} patch nests observer markers")
            inside = True
            blocks += 1
        elif marker == end:
            require(inside, f"{unit} patch has unmatched observer end")
            inside = False
        elif line.strip():
            require(inside, f"{unit} patch inserts a line outside observer markers")
            contents.append(line.strip())
    require(not inside and blocks > 0, f"{unit} patch has incomplete observer markers")
    return contents


def same_program_observers(
        new_bld: Path, ref_bld: Path, observer_patches: dict[str, Path],
        writer_units: tuple[str, ...], *, new_cfg: str, ref_cfg: str,
        preprocessed_markers: dict[str, tuple[str, str]] | None = None,
        preprocessed_patch_units: dict[str, Path] | None = None) -> dict:
    """Prove a build differs only by named observer insertions and writers."""
    for name in BUILD_CONFIG_FILES:
        want = (ref_bld / name).read_text().replace(ref_cfg, new_cfg)
        require((new_bld / name).read_text() == want, f"build config differs: {name}")
    marker_units = set(preprocessed_markers or ())
    require(not marker_units or marker_units == set(observer_patches),
            "preprocessed marker units must equal observer patch units")
    inserted = {
        unit: [line[1:] for line in patch.read_text().splitlines()
               if line.startswith("+") and not line.startswith("+++")]
        for unit, patch in observer_patches.items()
    }
    if preprocessed_markers:
        require(preprocessed_patch_units is not None
                and set(preprocessed_patch_units) == set(observer_patches),
                "marker mode requires one independently preprocessed patched unit per observer")
        marked_contents = {}
        for unit, path in preprocessed_patch_units.items():
            expected_lines = path.read_text().splitlines()
            _stripped, _ = _strip_preprocessed_observers(
                expected_lines, *preprocessed_markers[unit], unit)
            contents = []
            inside = False
            for line in expected_lines:
                if line.strip() == preprocessed_markers[unit][0]:
                    inside = True
                elif line.strip() == preprocessed_markers[unit][1]:
                    inside = False
                elif inside and line.strip():
                    contents.append(line.strip())
            marked_contents[unit] = contents
    else:
        marked_contents = {}
    ref_root, new_root = f"{ref_bld.parent}/".encode(), f"{new_bld.parent}/".encode()
    counts: dict[str, int] = {}
    for sub, writers in (
            ("ppsrc", {f"nemo/{unit}.f90" for unit in writer_units}),
            ("obj", {f"{unit}.o" for unit in writer_units})):
        ref_files, new_files = _regular_files(ref_bld / sub), _regular_files(new_bld / sub)
        require(new_files - ref_files == writers,
                f"{sub}: new build adds {sorted(new_files - ref_files)}, "
                f"expected only {sorted(writers)}")
        require(not ref_files - new_files, f"{sub}: new build lacks {sorted(ref_files - new_files)}")
        observers = ({f"nemo/{unit}.f90" for unit in observer_patches} if sub == "ppsrc"
                     else {f"{unit}.o" for unit in observer_patches})
        bad = []
        for rel in sorted(ref_files - observers):
            ref, new = (ref_bld / sub / rel).read_bytes(), (new_bld / sub / rel).read_bytes()
            same = objects_equal(ref, new, ref_root, new_root) if sub == "obj" else ref == new
            bad += [] if same else [rel]
        require(not bad, f"{sub}: {len(bad)} differ from round 326: "
                + " ".join(f"{sub}/{rel}" for rel in bad))
        if sub == "ppsrc":
            for unit in observer_patches:
                rel = f"nemo/{unit}.f90"
                ref_lines = (ref_bld / sub / rel).read_text().splitlines()
                new_lines = (new_bld / sub / rel).read_text().splitlines()
                if preprocessed_markers:
                    stripped, _ = _strip_preprocessed_observers(
                        new_lines, *preprocessed_markers[unit], unit,
                        expected_nonblank=marked_contents[unit])
                    require(stripped == ref_lines,
                            f"{unit}.f90 differs after stripping real ppsrc observer blocks")
                else:
                    _observer_only(ref_lines, new_lines, inserted[unit], unit)
        counts[sub] = len(ref_files) - len(observers)
    return {"status": "PASS_SAME_PROGRAM_R326_PLUS_OBSERVERS",
            "identical_ppsrc": counts["ppsrc"], "identical_obj": counts["obj"],
            "observer_units": list(observer_patches),
            "writer_units": list(writer_units)}


def same_program(new_bld: Path, ref_bld: Path, patches: tuple[Path, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    """Round-326's program plus only the two dyn_zdf observers and writer."""
    return same_program_observers(
        new_bld, ref_bld, dict(zip(OBSERVER_UNITS, patches, strict=True)),
        (WRITER_UNIT,), new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--same-program", type=Path, metavar="NEW_BLD")
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--dynzdf-patch", type=Path)
    parser.add_argument("--stage-patch", type=Path)
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.same_program is not None:
            require(None not in (args.reference_bld, args.dynzdf_patch, args.stage_patch,
                                 args.new_cfg, args.reference_cfg),
                    "--same-program needs reference BLD, both patches, and both config names")
            result = same_program(args.same_program, args.reference_bld,
                                  (args.dynzdf_patch, args.stage_patch),
                                  new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(args.record_dir is not None and args.expect_commit is not None,
                    "admission needs --record-dir and --expect-commit")
            result = run_gate(args.record_dir, args.expect_commit, args.plant)
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
