#!/usr/bin/env python3
"""Admit the ORCA2 round-79b per-step vertical-mixing record.

The record is self-describing.  Every name, rank, extent, origin and payload
length is read out of the file itself; nothing here predicts a byte count or a
header tuple, per the standing note-BD rule.  The gate refuses on any
structural defect and on any content statement that NEMO's compiled order makes
impossible.
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

MAGIC = "NEMO_L4_ZDFV_1"
HEADER_INTS = 15
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "rank", "nimpp", "njmpp",
    "jpi", "jpj", "jpk", "real_bits", "n_fields", "nn_mxl",
    "ln_zdfiwm", "ln_zdfddm",
)
EXPECTED_ORDER = (
    "avt_after_tke", "avm_after_tke", "avt_after_rnf", "avt_after_evd",
    "avm_after_evd", "avt_after_ddm", "avs_after_ddm", "avt_after_iwm",
    "avm_after_iwm", "avs_after_iwm", "avt_k", "avm_k", "en", "dissl",
    "mxlm", "mxld", "wmask", "tmask", "rnfmsk", "avtb_2d", "avtb", "avmb",
    "closure_scalars",
)
N_STEPS = 10
RANKS = (0, 1)


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


def _take(stream: io.BytesIO, count: int, what: str) -> bytes:
    chunk = stream.read(count)
    require(len(chunk) == count, f"truncated {what}")
    return chunk


def read_record(path: Path, *, plant: str | None = None) -> dict:
    """Parse one self-describing record, taking every extent from the file."""
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    header = struct.unpack(f"={HEADER_INTS}i",
                           _take(stream, 4 * HEADER_INTS, "header"))
    head = dict(zip(HEADER_NAMES, header, strict=True))
    if plant == "header":
        head["real_bits"] = 32
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r} in {path.name}")
    require(head["version"] == 1, f"record version {head['version']}")
    require(head["real_bits"] == 64, f"record is not fp64: {head['real_bits']}")
    require(head["n_fields"] == len(EXPECTED_ORDER),
            f"record declares {head['n_fields']} arrays, order has "
            f"{len(EXPECTED_ORDER)}")
    require(head["nn_mxl"] == 3, f"resolved nn_mxl {head['nn_mxl']} != 3")
    require(head["ln_zdfiwm"] == 1, "record was not written under ln_zdfiwm=T")
    require(head["ln_zdfddm"] == 1, "record was not written under ln_zdfddm=T")
    require(head["jpk"] >= 5,
            f"record has jpk {head['jpk']}, too shallow to carry the five "
            "closure scalars the writer emits")

    fields: dict[str, np.ndarray] = {}
    observed: list[str] = []
    for index in range(head["n_fields"]):
        name = _take(stream, 16, f"array {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", _take(stream, 28, f"{name} metadata"))
        require(rank in (1, 2, 3), f"{name} has rank {rank}")
        require(min(n1, n2, n3) >= 1, f"{name} has a non-positive extent")
        require((i0, j0, k0) == (1, 1, 1), f"{name} origin {(i0, j0, k0)}")
        payload = _take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        values = np.frombuffer(payload, dtype="=f8").reshape(
            (n1, n2, n3), order="F").copy()
        require(np.all(np.isfinite(values)), f"{name} carries non-finite values")
        require(name not in fields, f"duplicate array {name}")
        fields[name] = values
        observed.append(name)
    require(stream.read(1) == b"", f"{path.name} has trailing bytes")
    order = tuple(observed)
    if plant == "field-order":
        order = tuple(reversed(order))
    require(order == EXPECTED_ORDER,
            f"array order in {path.name} differs from the writer's source order")
    shape3 = (head["jpi"], head["jpj"], head["jpk"])
    for name in EXPECTED_ORDER[:18]:
        require(fields[name].shape == shape3,
                f"{name} shape {fields[name].shape} != {shape3}")
    return {"header": head, "fields": fields, "sha256": sha256(path)}


def check_content(record: dict, *, plant: str | None = None) -> dict:
    """Statements NEMO's compiled zdf_phy order makes true of every record."""
    f = record["fields"]
    wet = f["wmask"] > 0.0
    iwm_t = f["avt_after_iwm"] - f["avt_after_ddm"]
    iwm_m = f["avm_after_iwm"] - f["avm_after_evd"]
    if plant == "content":
        iwm_t = iwm_t - 1.0
    # zdfiwm.f90:314-316 adds a diffusivity clamped into [1.4e-7, 1e-2] and
    # masked by wmask, so the increment is never negative and is exactly zero
    # on dry w-points.
    require(np.all(iwm_t >= -0.0), "internal-wave increment to avt is negative")
    require(np.all(iwm_m >= -0.0), "internal-wave increment to avm is negative")
    require(not np.any(iwm_t[~wet] != 0.0),
            "internal-wave increment is non-zero on dry w-points")
    # zdfphy.F90 copies the closure output into the working arrays before any
    # additive arm, so the two agree on the copied band.
    band = slice(1, record["header"]["jpk"] - 1)
    require(np.array_equal(f["avt_after_tke"][:, :, band], f["avt_k"][:, :, band]),
            "avt_after_tke and avt_k disagree on the copied band")
    # The retired 1 m mixing-length floor would have produced this diffusivity.
    ediff = float(f["closure_scalars"][0, 0, 0])
    floor_equivalent = ediff * 1.0 * np.sqrt(np.maximum(f["en"], 0.0))
    return {
        "ediff": ediff,
        "rmxl_min": float(f["closure_scalars"][0, 0, 1]),
        "nkrnf": float(f["closure_scalars"][0, 0, 2]),
        "ln_rnf_mouth": float(f["closure_scalars"][0, 0, 3]),
        "ln_zdfevd": float(f["closure_scalars"][0, 0, 4]),
        "wet_points": int(np.count_nonzero(wet)),
        "iwm_avt_max": float(iwm_t.max()),
        "iwm_avt_mean_wet": float(iwm_t[wet].mean()) if wet.any() else 0.0,
        "iwm_avm_max": float(iwm_m.max()),
        "rnf_avt_max": float((f["avt_after_rnf"] - f["avt_after_tke"]).max()),
        "evd_avt_max": float((f["avt_after_evd"] - f["avt_after_rnf"]).max()),
        "ddm_avs_minus_avt_max": float(
            np.abs(f["avs_after_ddm"] - f["avt_after_ddm"]).max()),
        "floor_equivalent_mean_wet": (
            float(floor_equivalent[wet].mean()) if wet.any() else 0.0),
        "floor_equivalent_max": float(floor_equivalent.max()),
        "avt_final_mean_wet": (
            float(f["avt_after_iwm"][wet].mean()) if wet.any() else 0.0),
    }


def run_gate(root: Path, *, expect_commit: str | None = None,
             plant: str | None = None) -> dict:
    require(root.is_dir(), f"missing record directory {root}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), f"missing producer stamp {stamp}")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    if expect_commit is not None:
        require(produced == expect_commit,
                f"producer commit {produced} != {expect_commit}")

    summary: dict[str, dict] = {}
    shapes: dict[int, tuple] = {}
    for rank in RANKS:
        for step in range(1, N_STEPS + 1):
            path = root / f"oracle_zdf_vmix_kt{step:08d}_r{rank:04d}.bin"
            require(path.is_file(), f"missing record {path.name}")
            per_file_plant = plant if (rank == RANKS[0] and step == 1) else None
            record = read_record(path, plant=per_file_plant)
            head = record["header"]
            require(head["rank"] == rank,
                    f"{path.name} carries rank {head['rank']}")
            require(head["kt"] == step,
                    f"{path.name} carries kt {head['kt']}")
            shape = (head["jpi"], head["jpj"], head["jpk"])
            if rank in shapes:
                require(shapes[rank] == shape,
                        f"rank {rank} changed shape between steps")
            else:
                shapes[rank] = shape
            summary[path.name] = {
                "header": head,
                "sha256": record["sha256"],
                **check_content(record, plant=per_file_plant),
            }
    return {
        "status": "PASS",
        "producer_commit": produced,
        "records": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--plant", default=None,
        choices=["header", "field-order", "truncation", "stamp", "content"])
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.root, expect_commit=args.expect_commit,
                          plant=args.plant)
    except GateError as error:
        print(f"REFUSE: {error}")
        print("STATUS PLANT-FIRED" if args.plant else "STATUS REFUSE")
        return 1
    if args.plant:
        print("STATUS PLANT-SURVIVED")
        return 0
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"STATUS {result['status']} records={len(result['records'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
