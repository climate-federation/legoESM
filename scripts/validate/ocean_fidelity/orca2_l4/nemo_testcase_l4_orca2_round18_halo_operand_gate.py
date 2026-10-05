#!/usr/bin/env python3
"""Admit the ranked ORCA2 halo record and walk its U-flux operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round17_continuity_du_gate as round17,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo"
CITATIONS = {
    "ranked_writer_open": f"{_PP}/dynspg_ts.f90:445-455",
    "face_depth_and_transport": f"{_PP}/dynspg_ts.f90:539-544",
    "continuity_subtraction": f"{_PP}/dynspg_ts.f90:562-569",
    "ranked_writer_payload": f"{_PP}/dynspg_ts.f90:571-574",
    "ranked_writer_close": f"{_PP}/dynspg_ts.f90:845-848",
    "master_output_policy": f"{_PP}/in_out_manager.f90:180-180",
}

MAGIC = b"NEMO_L4_BTHALO1 "
FIELD_NAMES = ("zhU", "e2u", "ua_e", "zhup2_e", "ssumask")
JPI = 94
JPJ = 152
NROWS = 2
HEADER_INTS = 7
EXPECTED_BYTES = 16 + HEADER_INTS * 4 + NROWS * (
    4 + len(FIELD_NAMES) * JPI * JPJ * 8)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_halo(path: Path, expected_rank: int) -> dict[str, object]:
    """Read one direct Fortran-stream record, preserving its local layout."""
    require(path.is_file(), f"ranked halo stream is missing: {path}")
    require(path.stat().st_size == EXPECTED_BYTES,
            f"ranked halo stream has wrong byte count: {path}")
    with path.open("rb") as stream:
        magic = stream.read(16)
        header = np.fromfile(stream, dtype=np.int32, count=HEADER_INTS)
        require(magic == MAGIC, f"ranked halo magic changed: {path}")
        require(header.size == HEADER_INTS, f"short halo header: {path}")
        version, kt, nrows, rank, jpi, jpj, storage_bits = (
            int(value) for value in header)
        require(
            (version, kt, nrows, rank, jpi, jpj, storage_bits)
            == (1, 1, NROWS, expected_rank, JPI, JPJ, 64),
            f"ranked halo header mismatch: {path}",
        )
        fields = {name: [] for name in FIELD_NAMES}
        substeps = []
        for expected_substep in range(1, NROWS + 1):
            row = np.fromfile(stream, dtype=np.int32, count=1)
            require(row.size == 1 and int(row[0]) == expected_substep,
                    f"halo substep order changed: {path}")
            substeps.append(int(row[0]))
            for name in FIELD_NAMES:
                flat = np.fromfile(stream, dtype=np.float64, count=JPI * JPJ)
                require(flat.size == JPI * JPJ,
                        f"short halo payload {name}: {path}")
                fields[name].append(
                    flat.reshape((JPI, JPJ), order="F").T)
        require(stream.read(1) == b"", f"trailing halo bytes: {path}")
    return {
        "rank": rank,
        "substeps": substeps,
        "fields": {
            name: np.stack(rows, axis=0) for name, rows in fields.items()
        },
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _contains_exact_line(text: str, line: str) -> bool:
    return line in {candidate.strip() for candidate in text.splitlines()}


def admit(root: Path, *, plant: bool = False) -> dict[str, object]:
    """Fail closed on run completion, rank routing, layout, and uniqueness."""
    time_text = (root / "run.user.time.log").read_text()
    stdout_text = (root / "run.user.stdout.log").read_text()
    ocean_text = (root / "ocean.output").read_text()
    require(_contains_exact_line(time_text, "MPIRUN_RC=0"),
            "run does not carry MPIRUN_RC=0")
    require(_contains_exact_line(time_text, "RUN DONE"),
            "run does not carry RUN DONE")
    require(_contains_exact_line(stdout_text, "STOP 0"),
            "run does not carry STOP 0")

    rank0_marker = (
        "LANE4_BT_HALO_DUMP            1           2           0 "
        "oracle_bt_halo_operands_kt00000001_r0000.bin")
    rank1_marker = (
        "LANE4_BT_HALO_DUMP            1           2           1 "
        "oracle_bt_halo_operands_kt00000001_r0001.bin")
    require(sum(line.strip() == rank0_marker for line in ocean_text.splitlines())
            == 1, "rank-0 marker is not unique in ocean.output")
    require(sum(line.strip() == rank1_marker for line in stdout_text.splitlines())
            == 1, "rank-1 marker is not unique in captured MPI stdout")

    records = [read_halo(
        root / f"oracle_bt_halo_operands_kt00000001_r{rank:04d}.bin",
        expected_rank=rank,
    ) for rank in range(2)]
    require(records[0]["sha256"] != records[1]["sha256"],
            "the two ranked streams are byte-identical")

    rank_header_plant = {"requested": plant, "fires": None}
    if plant:
        try:
            read_halo(
                root / "oracle_bt_halo_operands_kt00000001_r0000.bin",
                expected_rank=1,
            )
        except GateError:
            rank_header_plant["fires"] = True
        else:
            rank_header_plant["fires"] = False
        require(bool(rank_header_plant["fires"]),
                "swapped-rank admission plant did not fire")
    return {
        "status": "ADMITTED_EXISTING_RANKED_HALO_RECORD",
        "record_bytes": EXPECTED_BYTES,
        "ranks": [{
            "rank": record["rank"],
            "substeps": record["substeps"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        } for record in records],
        "rank_header_plant": rank_header_plant,
        "_records": records,
    }


def _rank0_u_window(values: np.ndarray) -> np.ndarray:
    """Map NEMO rank 0's two-cell local halo to 148x91 global U faces."""
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (NROWS, JPJ, JPI),
            "rank-0 halo array shape changed")
    return values[:, 2:150, 1:92]


def _candidate_u_window(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)[:NROWS]
    require(values.shape[1] == 148 and values.shape[2] >= 91,
            "candidate global U array shape changed")
    return values[:, :, :91]


def _face_support(t_support: np.ndarray) -> np.ndarray:
    support = np.zeros((*t_support.shape[:-1], t_support.shape[-1] + 1),
                       dtype=bool)
    support[..., :-1] |= t_support
    support[..., 1:] |= t_support
    return support


def _operand_walk(*, card, trace, candidate, oracle, masks, inherited_first,
                  rank0_record: dict[str, object], plant: bool) -> dict[str, object]:
    require(inherited_first["differing_cells"] == round17.INHERITED_COUNT,
            "round-17 inherited boundary changed before the direct walk")
    recorded = {
        name: _rank0_u_window(values)
        for name, values in rank0_record["fields"].items()
    }
    candidate_fields = {
        "zhU": _candidate_u_window(trace["transport_metric_u"]),
        "e2u": np.broadcast_to(
            _candidate_u_window(np.asarray(card.recipe.grid.dy_u)[None, ...]),
            recorded["e2u"].shape),
        "ua_e": _candidate_u_window(trace["u_mid"]),
        "zhup2_e": _candidate_u_window(trace["transport_face_depth_u"]),
        "ssumask": np.broadcast_to(
            _candidate_u_window(np.asarray(
                card.recipe.initial_state.u_mask.data, dtype=np.float64
            )[None, ...]), recorded["ssumask"].shape),
    }

    direct_du = np.subtract(recorded["zhU"][..., 1:],
                            recorded["zhU"][..., :-1])
    candidate_du = np.subtract(candidate_fields["zhU"][..., 1:],
                               candidate_fields["zhU"][..., :-1])
    direct_replay = round14.compare(
        direct_du[1], oracle["continuity_du"][1], masks["t"])
    direct_right = round14.compare(
        recorded["zhU"][1, :, 1:], oracle["metric_transport_u"][1],
        masks["t"])
    candidate_replay = round14.compare(
        candidate_du[1], candidate["continuity_du"][1], masks["t"])
    require(direct_replay["bit_exact"],
            "direct rank-0 zhU does not replay recorded continuity_du")
    require(direct_right["bit_exact"],
            "rank-local zhU mapping does not reproduce the canonical right face")
    require(candidate_replay["bit_exact"],
            "candidate zhU does not replay traced continuity_du")

    mismatch_t = masks["t"] & (
        candidate["continuity_du"][1] != oracle["continuity_du"][1])
    require(int(np.count_nonzero(mismatch_t)) == round17.INHERITED_COUNT,
            "direct record changed the inherited mismatch support")
    support = _face_support(mismatch_t)
    rows_by_substep = [{
        name: round14.compare(
            candidate_fields[name][substep], recorded[name][substep], support)
        for name in FIELD_NAMES
    } for substep in range(NROWS)]
    rows = rows_by_substep[1]

    candidate_partial = np.multiply(candidate_fields["e2u"][1],
                                    candidate_fields["ua_e"][1])
    recorded_partial = np.multiply(recorded["e2u"][1],
                                   recorded["ua_e"][1])
    candidate_product = np.multiply(
        candidate_partial, candidate_fields["zhup2_e"][1])
    recorded_product = np.multiply(recorded_partial, recorded["zhup2_e"][1])
    arithmetic = {
        "candidate_product_replays_zhU": round14.compare(
            candidate_product, candidate_fields["zhU"][1], support),
        "recorded_product_replays_zhU": round14.compare(
            recorded_product, recorded["zhU"][1], support),
        "e2u_times_ua_e": round14.compare(
            candidate_partial, recorded_partial, support),
        "full_product": round14.compare(
            candidate_product, recorded_product, support),
    }

    order = ("e2u", "ua_e", "zhup2_e", "ssumask")
    first_non_bit_operand = next(
        (name for name in order if not rows[name]["bit_exact"]), None)

    ua_substituted = np.multiply(
        np.multiply(candidate_fields["e2u"][1], recorded["ua_e"][1]),
        candidate_fields["zhup2_e"][1])
    depth_substituted = np.multiply(
        candidate_partial, recorded["zhup2_e"][1])
    causal = {
        "recorded_ua_e_product_vs_recorded_zhU": round14.compare(
            ua_substituted, recorded["zhU"][1], support),
        "recorded_zhup2_e_product_vs_recorded_zhU": round14.compare(
            depth_substituted, recorded["zhU"][1], support),
        "recorded_ua_e_continuity_vs_oracle": round14.compare(
            np.subtract(ua_substituted[:, 1:], ua_substituted[:, :-1]),
            oracle["continuity_du"][1], mismatch_t),
        "recorded_zhup2_e_continuity_vs_oracle": round14.compare(
            np.subtract(depth_substituted[:, 1:], depth_substituted[:, :-1]),
            oracle["continuity_du"][1], mismatch_t),
        "candidate_substep1_u_exit_vs_recorded_substep2_ua_e":
            round14.compare(
                _candidate_u_window(trace["u_exit"])[0],
                recorded["ua_e"][1], support),
    }

    payload_plant = {"requested": plant, "fires": None}
    if plant:
        planted = np.array(recorded["zhU"][1], copy=True)
        index = tuple(int(value) for value in np.argwhere(support)[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        movement = round14.compare(planted, recorded["zhU"][1], support)
        payload_plant.update({
            "fires": bool(movement["differing_cells"] == 1),
            "index": list(index),
            "movement": movement,
        })
        require(bool(payload_plant["fires"]),
                "one-ULP direct-zhU plant did not fire")

    return {
        "label": "given NEMO's entry",
        "direct_continuity_replay": direct_replay,
        "direct_right_face_mapping": direct_right,
        "candidate_continuity_replay": candidate_replay,
        "mismatch_t_cells": int(np.count_nonzero(mismatch_t)),
        "mismatch_face_cells": int(np.count_nonzero(support)),
        "operand_rows_on_mismatch_faces": rows,
        "operand_rows_by_substep_on_mismatch_faces": rows_by_substep,
        "arithmetic_replays": arithmetic,
        "first_non_bit_operand": first_non_bit_operand,
        "causal_substitutions": causal,
        "payload_plant": payload_plant,
    }


def run(deck_root: Path, record_root: Path, json_out: Path | None,
        *, plant: bool = False) -> dict[str, object]:
    admission = admit(record_root, plant=plant)
    rank0_record = admission.pop("_records")[0]

    def extension(**context):
        return _operand_walk(
            **context, rank0_record=rank0_record, plant=plant)

    inherited = round17.run(
        deck_root, record_root, None, plant=False, _extension=extension)
    walk = inherited["extension"]
    require(
        inherited["round16_reproduction"]
        ["first_non_bit_after_slow_forcing_substitution"]
        ["differing_cells"] == round17.INHERITED_COUNT,
        "round-17 inherited boundary count changed",
    )
    status = (
        "WALKED_TO_FIRST_NON_BIT_HALO_OPERAND"
        if walk["first_non_bit_operand"] is not None
        else "REFUTED_NO_NON_BIT_HALO_OPERAND"
    )
    result = {
        "gate": "nemo_testcase_l4_orca2_round18_halo_operand_gate",
        "status": status,
        "label": "given NEMO's entry",
        "record_root": str(record_root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "admission": admission,
        "round17_reproduction": inherited["round16_reproduction"],
        "walk": walk,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.record_root, args.json_out, plant=args.plant)
    except (GateError, round17.GateError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "status": result["status"],
        "admission": result["admission"],
        "round17_reproduction": result["round17_reproduction"],
        "walk": result["walk"],
    }, indent=2, sort_keys=True))
    if args.plant:
        plants = (
            result["admission"]["rank_header_plant"]["fires"],
            result["walk"]["payload_plant"]["fires"],
        )
        print("PLANTS FIRED" if all(plants) else "PLANTS DID NOT FIRE")
        return 1 if all(plants) else 2
    return 0 if result["status"] == "WALKED_TO_FIRST_NON_BIT_HALO_OPERAND" else 3


if __name__ == "__main__":
    raise SystemExit(main())
