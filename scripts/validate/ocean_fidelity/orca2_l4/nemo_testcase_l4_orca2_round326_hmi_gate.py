#!/usr/bin/env python3
"""Admit the ORCA2 round-326 record of SI3's mean ice thickness as TKE reads it.

One file per rank per step (kt 1..10, ranks 0 and 1), written by
``zdftke_round326_writer.F90`` from ``zdf_tke`` just before ``tke_avn``, whose
nn_mxlice = 2 branch reads ``hm_i`` (zdftke.f90:620-623).  Every name, extent
and payload length is read out of the file.  The gate refuses on structural
defects, on a selector other than ORCA2's (nn_mxlice 2, ln_mxl0 T, nn_fsbc 2),
and on a negative or non-finite thickness.  It REPORTS, never refuses on, two
statements the preregistration predicts: fr_i and taum equal the round-5
surface frames bit for bit, and each even step's hm_i equals the previous odd
step's (hm_i is assigned only by ice_var_agg(2), icevar.f90:186, on ice steps,
icestp.f90:186).
"""

from __future__ import annotations

import argparse
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
PLANTS = ("header", "field-order", "truncation", "stamp", "content")
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
    result = {
        "format": "nemo-testcase-l4-orca2-round326-hmi-v1",
        "status": "PASS_R326_HMI_ADMISSION",
        "root": str(root),
        "producer_commit": produced,
        "records": {record_name(kt, rank): r["sha256"]
                    for (rank, kt), r in sorted(records.items())},
        "even_step_identity": even_step_identity(records),
    }
    if surface_root is not None:
        result["surface_cross_check"] = surface_cross_check(records, surface_root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_gate(args.root, expect_commit=args.expect_commit,
                          plant=args.plant, surface_root=args.surface_root)
    except (GateError, OSError, ValueError) as error:
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
