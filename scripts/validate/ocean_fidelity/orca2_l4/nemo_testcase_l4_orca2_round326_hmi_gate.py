#!/usr/bin/env python3
"""Admit the ORCA2 round-326 record of SI3's mean ice thickness as TKE reads it.

One file per rank per step (kt 1..10, ranks 0 and 1), written by
``zdftke_round326_writer.F90`` from ``zdf_tke`` just before ``tke_avn``, whose
nn_mxlice = 2 branch reads ``hm_i`` (zdftke.f90:620-623).  Every name, extent
and payload length is read out of the file.  The gate refuses on structural
defects, on a selector other than ORCA2's (nn_mxlice 2, ln_mxl0 T, nn_fsbc 2),
and on every preregistered validity check (A2-A4): a negative, non-finite,
all-zero (on ice-covered wet points) or >= 20 m thickness; fr_i or taum unequal
to the round-5 surface frames; an even step's hm_i unequal to the previous odd
step's (hm_i is assigned only by ice_var_agg(2), icevar.f90:186, on ice steps,
icestp.f90:186).  Admission needs the surface frames; without them the status
is PASS_STRUCTURE_ONLY, never an admission.

``--same-program`` proves a fresh build is round 5's program except the
observer: build config identical (config name substituted), every
preprocessed source and object byte-identical except zdftke (whose only
difference is the patch's inserted lines) and the writer (new).
"""

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

MAGIC = "NEMO_L4_HMI_1"
HEADER_NAMES = (
    "version", "kt", "rank", "nimpp", "njmpp", "jpi", "jpj", "real_bits",
    "n_fields", "nn_mxlice", "ln_mxl0", "nn_fsbc",
)
EXPECTED_ORDER = ("hm_i", "fr_i", "taum", "tmask_1")
N_STEPS = 10
RANKS = (0, 1)
HALO = 2
PLANTS = ("header", "field-order", "truncation", "stamp", "content",
          "hm-zero", "hm-thick", "even-step", "surface")
HM_I_MAX_M = 20.0  # preregistration A4: max below 20 m
OBSERVER_UNIT = "zdftke"
WRITER_UNIT = "zdftke_round326_writer"
BUILD_CONFIG_FILES = ("arch_nemo.fcm", "cpp.fcm", "bld.cfg")
ROUND5_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_name(kt: int, rank: int) -> str:
    return f"oracle_tke_hmi_kt{kt:08d}_r{rank:04d}.bin"


def _take(stream: io.BytesIO, count: int, what: str) -> bytes:
    chunk = stream.read(count)
    require(len(chunk) == count, f"truncated {what}")
    return chunk


def read_record(path: Path, *, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    head = dict(zip(HEADER_NAMES, struct.unpack(
        f"={len(HEADER_NAMES)}i", _take(stream, 4 * len(HEADER_NAMES), "header")),
        strict=True))
    if plant == "header":
        head["nn_mxlice"] = 0
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r} in {path.name}")
    require(head["version"] == 1, f"record version {head['version']}")
    require(head["real_bits"] == 64, f"record is not fp64: {head['real_bits']}")
    require(head["n_fields"] == len(EXPECTED_ORDER),
            f"record declares {head['n_fields']} arrays")
    require((head["nn_mxlice"], head["ln_mxl0"], head["nn_fsbc"]) == (2, 1, 2),
            "record was not written under nn_mxlice=2, ln_mxl0=T, nn_fsbc=2: "
            f"{head['nn_mxlice']}/{head['ln_mxl0']}/{head['nn_fsbc']}")
    fields: dict[str, np.ndarray] = {}
    observed: list[str] = []
    for index in range(head["n_fields"]):
        name = _take(stream, 16, f"array {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", _take(stream, 28, f"{name} metadata"))
        require((rank, n3, i0, j0, k0) == (2, 1, 1, 1, 1),
                f"{name} metadata {(rank, n3, i0, j0, k0)}")
        require((n1, n2) == (head["jpi"], head["jpj"]),
                f"{name} extent {(n1, n2)} != header {(head['jpi'], head['jpj'])}")
        values = np.frombuffer(_take(stream, n1 * n2 * 8, f"{name} payload"),
                               dtype="=f8").reshape((n1, n2), order="F").copy()
        require(bool(np.all(np.isfinite(values))), f"{name} carries non-finite values")
        require(name not in fields, f"duplicate array {name}")
        fields[name] = values
        observed.append(name)
    require(stream.read(1) == b"", f"{path.name} has trailing bytes")
    order = tuple(observed)
    if plant == "field-order":
        order = tuple(reversed(order))
    require(order == EXPECTED_ORDER,
            f"array order in {path.name} differs from the writer's source order")
    if plant == "content":
        fields["hm_i"] = fields["hm_i"] - 1.0
    # hm_i = vt_i * z1_at_i with z1_at_i >= 0 (icevar.f90:182-186).
    require(bool(np.all(fields["hm_i"] >= 0.0)), f"{path.name}: negative hm_i")
    return {"header": head, "fields": fields, "sha256": sha256(path)}


def owned(values: np.ndarray) -> np.ndarray:
    """Writer (jpi, jpj) -> the ladder's owned (NY-4, NX-4), as surface 'full'."""
    return values[HALO:-HALO, HALO:-HALO].T


def load(root: Path, *, plant: str | None = None) -> dict[tuple[int, int], dict]:
    records = {}
    for rank in RANKS:
        for kt in range(1, N_STEPS + 1):
            path = root / record_name(kt, rank)
            require(path.is_file(), f"missing record {path.name}")
            record = read_record(path, plant=plant if (rank, kt) == (0, 1) else None)
            head = record["header"]
            require((head["rank"], head["kt"]) == (rank, kt),
                    f"{path.name} carries rank {head['rank']} kt {head['kt']}")
            records[(rank, kt)] = record
    return records


def surface_cross_check(records, surface_root: Path) -> dict[str, object]:
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    rows = []
    for (rank, kt), record in sorted(records.items()):
        name = (f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
                f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin")
        surface = ladder.read_surface_fields(surface_root / name, kt, rank)
        row = {"rank": rank, "kt": kt}
        for field in ("fr_i", "taum"):
            mine = owned(record["fields"][field])
            theirs = np.asarray(surface[field])
            require(mine.shape == theirs.shape,
                    f"{field} owned shape {mine.shape} != surface {theirs.shape}")
            row[f"{field}_unequal"] = int(np.count_nonzero(mine != theirs))
        row["wet_ice_points"] = int(np.count_nonzero(
            (owned(record["fields"]["fr_i"]) > 0) & (owned(record["fields"]["tmask_1"]) > 0)))
        row["hm_i_max"] = float(owned(record["fields"]["hm_i"]).max())
        rows.append(row)
    return {"rows": rows,
            "fr_i_taum_bit_equal": all(r["fr_i_unequal"] == 0 and r["taum_unequal"] == 0
                                       for r in rows)}


def even_step_identity(records) -> dict[str, object]:
    rows = []
    for rank in RANKS:
        for kt in range(2, N_STEPS + 1, 2):
            a = records[(rank, kt - 1)]["fields"]["hm_i"]
            b = records[(rank, kt)]["fields"]["hm_i"]
            rows.append({"rank": rank, "kt": kt,
                         "unequal_vs_previous_odd": int(np.count_nonzero(a != b))})
    return {"rows": rows, "holds": all(r["unequal_vs_previous_odd"] == 0 for r in rows)}


def thickness_check(records) -> dict[str, object]:
    hm_max = max(float(r["fields"]["hm_i"].max()) for r in records.values())
    ice_wet = sum(int(np.count_nonzero(
        (owned(r["fields"]["hm_i"]) > 0) & (owned(r["fields"]["fr_i"]) > 0)
        & (owned(r["fields"]["tmask_1"]) > 0))) for r in records.values())
    require(ice_wet > 0, "hm_i is zero on every ice-covered wet point (A4)")
    require(hm_max < HM_I_MAX_M, f"hm_i max {hm_max} m >= {HM_I_MAX_M} m (A4)")
    return {"hm_i_max_m": hm_max, "ice_covered_wet_points_with_hm_i": ice_wet}


def _plant_fields(records, plant: str | None) -> None:
    def bump(key, field):
        values = records[key]["fields"][field].copy()
        values[HALO, HALO] += 1.0e-3
        records[key]["fields"][field] = values
    if plant == "hm-zero":
        for record in records.values():
            record["fields"]["hm_i"] = np.zeros_like(record["fields"]["hm_i"])
    elif plant == "hm-thick":
        values = records[(0, 2)]["fields"]["hm_i"].copy()
        values[HALO, HALO] = HM_I_MAX_M
        records[(0, 2)]["fields"]["hm_i"] = values
    elif plant == "even-step":
        bump((1, 4), "hm_i")
    elif plant == "surface":
        bump((1, 3), "fr_i")


def run_gate(root: Path, *, expect_commit: str | None = None,
             plant: str | None = None, surface_root: Path | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), f"missing producer stamp {stamp}")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    if expect_commit is not None:
        require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    records = load(root, plant=plant)
    _plant_fields(records, plant)
    result = {
        "format": "nemo-testcase-l4-orca2-round326-hmi-v2",
        "status": "PASS_STRUCTURE_ONLY",
        "root": str(root),
        "producer_commit": produced,
        "records": {record_name(kt, rank): r["sha256"]
                    for (rank, kt), r in sorted(records.items())},
        "thickness": thickness_check(records),
        "even_step_identity": even_step_identity(records),
    }
    require(result["even_step_identity"]["holds"],
            "an even step's hm_i differs from the previous odd step's (A3)")
    if surface_root is not None:
        cross = surface_cross_check(records, surface_root)
        require(cross["fr_i_taum_bit_equal"],
                "fr_i/taum differ from the round-5 surface frames (A2)")
        result["surface_cross_check"] = cross
        result["status"] = "PASS_R326_HMI_ADMISSION"
    return result


def _regular_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _observer_only(ref: list[str], new: list[str], inserted: list[str]) -> None:
    added: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, new, autojunk=False).get_opcodes():
        require(tag in ("equal", "insert"),
                f"{OBSERVER_UNIT}.f90 {tag}s round-5 lines {i1 + 1}-{i2}")
        added += new[j1:j2] if tag == "insert" else []
    require(added == inserted,
            f"{OBSERVER_UNIT}.f90 inserts {added!r}, patch inserts {inserted!r}")


def same_program(new_bld: Path, ref_bld: Path, patch: Path, *,
                 new_cfg: str, ref_cfg: str) -> dict:
    for name in BUILD_CONFIG_FILES:
        want = (ref_bld / name).read_text().replace(ref_cfg, new_cfg)
        require((new_bld / name).read_text() == want, f"build config differs: {name}")
    inserted = [line[1:] for line in patch.read_text().splitlines()
                if line.startswith("+") and not line.startswith("+++")]
    counts = {}
    for sub, observer, writer in (("ppsrc", f"nemo/{OBSERVER_UNIT}.f90", f"nemo/{WRITER_UNIT}.f90"),
                                  ("obj", f"{OBSERVER_UNIT}.o", f"{WRITER_UNIT}.o")):
        ref_files, new_files = _regular_files(ref_bld / sub), _regular_files(new_bld / sub)
        require(new_files - ref_files == {writer},
                f"{sub}: new build adds {sorted(new_files - ref_files)}, expected only {writer}")
        require(not ref_files - new_files, f"{sub}: new build lacks {sorted(ref_files - new_files)}")
        for rel in sorted(ref_files - {observer}):
            require((ref_bld / sub / rel).read_bytes() == (new_bld / sub / rel).read_bytes(),
                    f"{sub}/{rel} differs from round 5")
        if sub == "ppsrc":
            _observer_only((ref_bld / sub / observer).read_text().splitlines(),
                           (new_bld / sub / observer).read_text().splitlines(), inserted)
        counts[sub] = len(ref_files) - 1
    return {"status": "PASS_SAME_PROGRAM", "identical_ppsrc": counts["ppsrc"],
            "identical_obj": counts["obj"], "observer_inserted_lines": inserted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--same-program", type=Path, metavar="NEW_BLD")
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.same_program is not None:
            require(None not in (args.reference_bld, args.patch, args.new_cfg, args.reference_cfg),
                    "--same-program needs --reference-bld, --patch, --new-cfg, --reference-cfg")
            result = same_program(args.same_program, args.reference_bld, args.patch,
                                  new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(args.root is not None and args.surface_root is not None,
                    "admission needs --root and --surface-root")
            result = run_gate(args.root, expect_commit=args.expect_commit,
                              plant=args.plant, surface_root=args.surface_root)
    except (GateError, OSError, ValueError, KeyError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered if not args.output else "", end="")
    print(f"STATUS {result['status']}")
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
