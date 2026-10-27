#!/usr/bin/env python3
"""Build real-layout synthetic records for the round-336 stub admission flow."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round326_hmi_gate as hmi,
    nemo_testcase_l4_orca2_round328_dynzdf_gate as dynzdf,
    nemo_testcase_l4_orca2_round336_tke_gate as gate,
    nemo_testcase_l4_orca2_round336_trazdf_gate as trazdf,
)


def checksum(values: np.ndarray) -> int:
    words = np.asarray(values, dtype="=f8", order="F").ravel(order="F").view(np.uint64)
    return int(np.bitwise_xor.reduce(words, initial=np.uint64(0)))


def field_rank(name: str) -> int:
    if name in gate.ONE_D:
        return 1
    if name in gate.INTERIOR_2D | gate.FILLED_2D | gate.FULL_2D:
        return 2
    return 3


def make_fields(header: dict[str, int], phase: dict, ice: dict) -> dict[str, np.ndarray]:
    i0, i1 = header["ntsi"] - 1, header["ntei"]
    j0, j1 = header["ntsj"] - 1, header["ntej"]
    fields: dict[str, np.ndarray] = {}
    for number, name in enumerate(gate.FIELDS, start=1):
        n1, n2, n3 = gate._extent(name, header)
        rank = field_rank(name)
        shape = (n1,) if rank == 1 else (n1, n2) if rank == 2 else (n1, n2, n3)
        fields[name] = np.full(shape, 0.01 * number, dtype="=f8", order="F")

    tmask = np.asarray(phase["fields"]["tmask"], dtype="=f8", order="F")
    ssmask = np.any(tmask != 0.0, axis=2).astype("=f8")
    fields["entry_tmask"] = tmask.copy(order="F")
    fields["entry_umask"] = tmask.copy(order="F")
    fields["entry_vmask"] = tmask.copy(order="F")
    fields["tke_wmask"] = tmask.copy(order="F")
    fields["entry_ssmask"] = ssmask
    drag_i0, drag_i1 = header["ntsi"] - 2, header["ntei"] + 1
    drag_j0, drag_j1 = header["ntsj"] - 2, header["ntej"] + 1
    fields["entry_drag_bottom"] = (
        np.float64(-0.0004) * ssmask[drag_i0:drag_i1, drag_j0:drag_j1])
    fields["entry_fr_i"] = ice["fields"]["fr_i"].copy(order="F")
    fields["entry_hm_i"] = ice["fields"]["hm_i"][i0:i1, j0:j1].copy(order="F")
    fields["entry_taum"] = ice["fields"]["taum"][i0:i1, j0:j1].copy(order="F")
    fields["entry_htau"].fill(50.0)
    fields["entry_uu_kbb"].fill(0.1)
    fields["entry_vv_kbb"].fill(0.2)
    fields["entry_nmln"].fill(2.0)
    fields["entry_hmlp"].fill(1.0)
    fields["entry_mbkt"] = np.maximum(np.count_nonzero(tmask, axis=2), 1).astype("=f8")
    fields["entry_mikt"].fill(1.0)
    depth = np.arange(1, header["jpk"] + 1, dtype="=f8")[None, None, :]
    fields["entry_gdepw_kmm"] = np.broadcast_to(
        depth, fields["entry_gdepw_kmm"].shape).copy(order="F")
    fields["entry_e3t_kmm"] = phase["fields"]["e3t_Kmm"][i0:i1, j0:j1].copy(order="F")
    fields["entry_e3w_kmm"] = phase["fields"]["e3w_Kmm"][i0:i1, j0:j1].copy(order="F")
    fields["entry_avtb_2d"].fill(1.0e-5)
    fields["entry_avtb"].fill(1.0)
    fields["entry_avmb"].fill(1.0e-5)
    fields["tke_pdlr"].fill(0.5)
    fields["pre_sweep_mxl"].fill(1.0)
    fields["final_mxl"].fill(2.0)
    fields["final_mxld"].fill(3.0)
    fields["final_avt"] = phase["fields"]["avt_after_tke"].copy(order="F")
    for name in gate.NEGATIVE_ZERO_EDGE:
        fields[name][..., -1] = -0.0
    return fields


def write_record(path: Path, rank: int, phase: dict, ice: dict, drag: dict) -> None:
    header = dict(zip(
        gate.HEADER_NAMES,
        (2, 2, 3, 3, rank, *gate.REAL_LAYOUT, 64, len(gate.FIELDS)),
        strict=True,
    ))
    fields = make_fields(header, phase, ice)
    # Seed the exact overlap from a REAL admitted NEMO drag record.  That
    # record covers ntsi:ntei+1,ntsj:ntej+1; the new record deliberately also
    # carries NEMO's filled left/bottom halo.  Bottom values for those two
    # edges are derived from the admitted NEMO mask and source-literal linear
    # coefficient above, while top edges extend the nearest real halo value.
    source_bottom = np.squeeze(drag["fields"]["rCdU_bot"])
    source_top = np.squeeze(drag["fields"]["rCdU_top"])
    expected_overlap = fields["entry_drag_bottom"][1:, 1:]
    if not np.array_equal(source_bottom.view(np.uint64), expected_overlap.view(np.uint64)):
        raise ValueError("REAL admitted bottom-drag overlap differs from the source-literal value")
    fields["entry_drag_bottom"][1:, 1:] = source_bottom
    fields["entry_drag_top"][1:, 1:] = source_top
    fields["entry_drag_top"][0, 1:] = source_top[-2, :]
    fields["entry_drag_top"][:, 0] = fields["entry_drag_top"][:, 1]
    with path.open("wb") as stream:
        stream.write(gate.MAGIC.ljust(16).encode("ascii"))
        stream.write(struct.pack(
            f"={len(gate.HEADER_NAMES)}i",
            *(header[name] for name in gate.HEADER_NAMES),
        ))
        for name in gate.FIELDS:
            values = np.asarray(fields[name], dtype="=f8", order="F")
            rank = field_rank(name)
            n1, n2, n3 = gate._extent(name, header)
            stream.write(name.ljust(24).encode("ascii"))
            stream.write(struct.pack(
                "=7i", rank, n1, n2, n3, *gate._origin(name, header)))
            stream.write(struct.pack("=Q", checksum(values)))
            stream.write(values.tobytes(order="F"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-reference-dir", type=Path, required=True)
    parser.add_argument("--ice-reference-dir", type=Path, required=True)
    parser.add_argument("--drag-reference-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for rank in (0, 1):
        phase = trazdf.read_record(args.phase_reference_dir / trazdf.record_name(rank))
        ice = hmi.read_record(args.ice_reference_dir / hmi.record_name(2, rank))
        drag = dynzdf.read_record(args.drag_reference_dir / dynzdf.record_name(rank))
        write_record(args.output / gate.record_name(rank), rank, phase, ice, drag)
        gate.read_record(args.output / gate.record_name(rank))
    print(f"R336_TKE_STUB_RECORDS_PASS path={args.output} fields={len(gate.FIELDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
