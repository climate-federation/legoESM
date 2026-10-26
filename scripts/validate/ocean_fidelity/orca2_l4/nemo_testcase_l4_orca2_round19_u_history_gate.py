#!/usr/bin/env python3
"""Admit ORCA2 ranked U histories and walk the first halo difference."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round18_halo_operand_gate as round18,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo"
CITATIONS = {
    "mixed_array_extents": f"{_PP}/dynspg_ts.f90:180-182",
    "ranked_writer_open": f"{_PP}/dynspg_ts.f90:458-467",
    "midpoint_statement_and_payload": f"{_PP}/dynspg_ts.f90:501-513",
    "vector_update": f"{_PP}/dynspg_ts.f90:688-711",
    "post_update_exchange": f"{_PP}/dynspg_ts.f90:754-770",
    "update_payload_and_rotation": f"{_PP}/dynspg_ts.f90:814-822",
    "ranked_writer_close": f"{_PP}/dynspg_ts.f90:866-870",
}

MAGIC = b"NEMO_L4_UHIST_1 "
JPI = 94
JPJ = 152
INTERIOR_NX = 90
INTERIOR_NY = 148
NROWS = 2
HEADER_INTS = 7
FULL_FIELDS_BEFORE = ("un_e", "ub_e", "ubb_e", "ua_mid")
FULL_FIELDS_AFTER = ("zu_spg", "zu_trd")
FULL_FIELDS_TAIL = ("ssumask", "ua_exit")
EXPECTED_BYTES = (
    16 + HEADER_INTS * 4
    + NROWS * (
        4 + 4 * 8
        + 8 * JPI * JPJ * 8
        + INTERIOR_NX * INTERIOR_NY * 8
    )
)
SOURCE_ORDER = (
    "coefficient_1", "coefficient_2", "coefficient_3",
    "un_e", "ub_e", "ubb_e", "ua_mid", "rDt_e",
    "zu_spg", "zu_trd", "zu_frc", "ssumask", "ua_exit",
)


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


def _read_full(stream, path: Path, name: str) -> np.ndarray:
    flat = np.fromfile(stream, dtype=np.float64, count=JPI * JPJ)
    require(flat.size == JPI * JPJ, f"short U-history payload {name}: {path}")
    return flat.reshape((JPI, JPJ), order="F").T


def _read_interior(stream, path: Path, name: str) -> np.ndarray:
    count = INTERIOR_NX * INTERIOR_NY
    flat = np.fromfile(stream, dtype=np.float64, count=count)
    require(flat.size == count, f"short U-history payload {name}: {path}")
    return flat.reshape((INTERIOR_NX, INTERIOR_NY), order="F").T


def read_u_history(path: Path, expected_rank: int) -> dict[str, object]:
    """Read the mixed-extent Fortran stream exactly as its writer emits it."""
    require(path.is_file(), f"ranked U-history stream is missing: {path}")
    require(path.stat().st_size == EXPECTED_BYTES,
            f"ranked U-history stream has wrong byte count: {path}")
    fields = {name: [] for name in (
        *FULL_FIELDS_BEFORE, *FULL_FIELDS_AFTER, "zu_frc", *FULL_FIELDS_TAIL)}
    coefficients = []
    rdt = []
    substeps = []
    with path.open("rb") as stream:
        magic = stream.read(16)
        header = np.fromfile(stream, dtype=np.int32, count=HEADER_INTS)
        require(magic == MAGIC, f"ranked U-history magic changed: {path}")
        require(header.size == HEADER_INTS, f"short U-history header: {path}")
        version, kt, nrows, rank, jpi, jpj, storage_bits = (
            int(value) for value in header)
        require(
            (version, kt, nrows, rank, jpi, jpj, storage_bits)
            == (1, 1, NROWS, expected_rank, JPI, JPJ, 64),
            f"ranked U-history header mismatch: {path}",
        )
        for expected_substep in range(1, NROWS + 1):
            row = np.fromfile(stream, dtype=np.int32, count=1)
            require(row.size == 1 and int(row[0]) == expected_substep,
                    f"U-history substep order changed: {path}")
            substeps.append(int(row[0]))
            weights = np.fromfile(stream, dtype=np.float64, count=3)
            require(weights.size == 3, f"short midpoint coefficients: {path}")
            coefficients.append(weights)
            for name in FULL_FIELDS_BEFORE:
                fields[name].append(_read_full(stream, path, name))
            step_rdt = np.fromfile(stream, dtype=np.float64, count=1)
            require(step_rdt.size == 1, f"short rDt_e payload: {path}")
            rdt.append(float(step_rdt[0]))
            for name in FULL_FIELDS_AFTER:
                fields[name].append(_read_full(stream, path, name))
            fields["zu_frc"].append(_read_interior(stream, path, "zu_frc"))
            for name in FULL_FIELDS_TAIL:
                fields[name].append(_read_full(stream, path, name))
        require(stream.read(1) == b"", f"trailing U-history bytes: {path}")
    return {
        "rank": rank,
        "substeps": substeps,
        "coefficients": np.stack(coefficients),
        "rDt_e": np.asarray(rdt, dtype=np.float64),
        "fields": {name: np.stack(rows) for name, rows in fields.items()},
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _contains_exact_line(text: str, line: str) -> bool:
    return line in {candidate.strip() for candidate in text.splitlines()}


def admit(root: Path, *, plant: bool = False) -> dict[str, object]:
    """Fail closed on completion, routing, rank identity, and mixed extents."""
    time_text = (root / "run.user.time.log").read_text()
    stdout_text = (root / "run.user.stdout.log").read_text()
    ocean_text = (root / "ocean.output").read_text()
    require(_contains_exact_line(time_text, "MPIRUN_RC=0"),
            "run does not carry MPIRUN_RC=0")
    require(_contains_exact_line(time_text, "RUN DONE"),
            "run does not carry RUN DONE")
    require(_contains_exact_line(stdout_text, "STOP 0"),
            "run does not carry STOP 0")
    markers = [
        (ocean_text, 0),
        (stdout_text, 1),
    ]
    for text, rank in markers:
        marker = (
            f"LANE4_BT_UHIST_DUMP            1           2           {rank} "
            f"oracle_bt_u_history_kt00000001_r{rank:04d}.bin")
        require(sum(line.strip() == marker for line in text.splitlines()) == 1,
                f"rank-{rank} U-history marker is not unique")
    records = [read_u_history(
        root / f"oracle_bt_u_history_kt00000001_r{rank:04d}.bin", rank)
        for rank in range(2)]
    require(records[0]["sha256"] != records[1]["sha256"],
            "the two ranked U-history streams are byte-identical")
    rank_plant = {"requested": plant, "fires": None}
    if plant:
        try:
            read_u_history(
                root / "oracle_bt_u_history_kt00000001_r0000.bin", 1)
        except GateError:
            rank_plant["fires"] = True
        else:
            rank_plant["fires"] = False
        require(bool(rank_plant["fires"]), "swapped-rank plant did not fire")
    return {
        "status": "ADMITTED_EXISTING_RANKED_U_HISTORY_RECORD",
        "record_bytes": EXPECTED_BYTES,
        "full_array_shape": [JPJ, JPI],
        "zu_frc_shape": [INTERIOR_NY, INTERIOR_NX],
        "ranks": [{
            "rank": record["rank"],
            "substeps": record["substeps"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        } for record in records],
        "rank_header_plant": rank_plant,
        "_records": records,
    }


def replay_midpoint(coefficients, un_e, ub_e, ubb_e) -> np.ndarray:
    first = np.multiply(coefficients[0], un_e)
    second = np.multiply(coefficients[1], ub_e)
    third = np.multiply(coefficients[2], ubb_e)
    return np.add(np.add(first, second), third)


def replay_vector(un_e, rdt, zu_spg, zu_trd, zu_frc, ssumask) -> np.ndarray:
    trend = np.add(np.add(zu_spg, zu_trd), zu_frc)
    increment = np.multiply(rdt, trend)
    return np.multiply(np.add(un_e, increment), ssumask)


def first_non_bit(rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in rows:
        for boundary in SOURCE_ORDER:
            result = row[boundary]
            if not result["bit_exact"]:
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _scalar_row(candidate: float, oracle: float) -> dict[str, object]:
    return round14.compare(
        np.asarray([candidate], dtype=np.float64),
        np.asarray([oracle], dtype=np.float64),
        np.asarray([True]),
    )


def _history_walk(*, card, trace, candidate, oracle, masks, inherited_first,
                  halo_record: dict[str, object],
                  history_record: dict[str, object], plant: bool) -> dict[str, object]:
    halo_walk = round18._operand_walk(
        card=card, trace=trace, candidate=candidate, oracle=oracle, masks=masks,
        inherited_first=inherited_first, rank0_record=halo_record, plant=plant)
    require(halo_walk["first_non_bit_operand"] == "ua_e",
            "round-18 direct ua_e boundary did not reproduce")

    recorded = {
        name: round18._rank0_u_window(values)
        for name, values in history_record["fields"].items()
        if name != "zu_frc"
    }
    recorded["zu_frc"] = np.asarray(
        history_record["fields"]["zu_frc"], dtype=np.float64)
    candidate_full = {
        "un_e": round18._candidate_u_window(trace["u_entry"]),
        "ub_e": round18._candidate_u_window(trace["u_history_b"]),
        "ubb_e": round18._candidate_u_window(trace["u_history_bb"]),
        "ua_mid": round18._candidate_u_window(trace["u_mid"]),
        "zu_spg": round18._candidate_u_window(trace["pgf_u"]),
        "zu_trd": round18._candidate_u_window(trace["trd_u"]),
        "ssumask": np.broadcast_to(
            round18._candidate_u_window(np.asarray(
                card.recipe.initial_state.u_mask.data, dtype=np.float64
            )[None, ...]), recorded["ssumask"].shape),
        "ua_exit": round18._candidate_u_window(trace["u_exit"]),
    }
    candidate_slow = np.asarray(trace["slow_u"], dtype=np.float64)[:NROWS, :, 1:91]
    require(candidate_slow.shape == recorded["zu_frc"].shape,
            "candidate slow-forcing interior extent changed")
    candidate_coefficients = np.stack([
        np.asarray(trace[f"mid_weight_{index}"], dtype=np.float64)[:NROWS]
        for index in (1, 2, 3)
    ], axis=-1)
    candidate_rdt = np.full(
        NROWS,
        float(card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps),
        dtype=np.float64,
    )

    mismatch_t = masks["t"] & (
        candidate["continuity_du"][1] != oracle["continuity_du"][1])
    support = round18._face_support(mismatch_t)
    require(int(np.count_nonzero(support)) == 128,
            "round-18 mismatch-adjacent face support changed")
    interior_support = support[:, 1:]
    interior_active = recorded["ssumask"][0, :, 1:] != 0.0
    rows = []
    raw_full_update_rows = []
    midpoint_replays = []
    vector_replays = []
    for substep in range(NROWS):
        raw_full_update_rows.append({
            "substep": substep + 1,
            **{
                name: round14.compare(
                    candidate_full[name][substep], recorded[name][substep], support)
                for name in ("zu_spg", "zu_trd", "ssumask")
            },
        })
        row = {
            "substep": substep + 1,
            **{
                f"coefficient_{index + 1}": _scalar_row(
                    candidate_coefficients[substep, index],
                    history_record["coefficients"][substep, index])
                for index in range(3)
            },
            **{
                name: round14.compare(
                    candidate_full[name][substep], recorded[name][substep], support)
                for name in (
                    "un_e", "ub_e", "ubb_e", "ua_mid", "ua_exit")
            },
            **{
                name: round14.compare(
                    candidate_full[name][substep, :, 1:],
                    recorded[name][substep, :, 1:], interior_support)
                for name in ("zu_spg", "zu_trd", "ssumask")
            },
            "rDt_e": _scalar_row(
                candidate_rdt[substep], history_record["rDt_e"][substep]),
            "zu_frc": round14.compare(
                candidate_slow[substep], recorded["zu_frc"][substep],
                interior_active),
        }
        rows.append(row)

        recorded_mid = replay_midpoint(
            history_record["coefficients"][substep],
            recorded["un_e"][substep], recorded["ub_e"][substep],
            recorded["ubb_e"][substep])
        candidate_mid = replay_midpoint(
            candidate_coefficients[substep],
            candidate_full["un_e"][substep], candidate_full["ub_e"][substep],
            candidate_full["ubb_e"][substep])
        midpoint_replays.append({
            "recorded": round14.compare(
                recorded_mid, recorded["ua_mid"][substep], support),
            "candidate": round14.compare(
                candidate_mid, candidate_full["ua_mid"][substep], support),
        })

        recorded_vector = replay_vector(
            recorded["un_e"][substep, :, 1:], history_record["rDt_e"][substep],
            recorded["zu_spg"][substep, :, 1:],
            recorded["zu_trd"][substep, :, 1:], recorded["zu_frc"][substep],
            recorded["ssumask"][substep, :, 1:])
        candidate_vector = replay_vector(
            candidate_full["un_e"][substep, :, 1:], candidate_rdt[substep],
            candidate_full["zu_spg"][substep, :, 1:],
            candidate_full["zu_trd"][substep, :, 1:], candidate_slow[substep],
            candidate_full["ssumask"][substep, :, 1:])
        vector_replays.append({
            "recorded": round14.compare(
                recorded_vector, recorded["ua_exit"][substep, :, 1:],
                interior_support),
            "candidate": round14.compare(
                candidate_vector, candidate_full["ua_exit"][substep, :, 1:],
                interior_support),
        })

    first = first_non_bit(rows)
    recorded_rotation = round14.compare(
        recorded["ua_exit"][0], recorded["un_e"][1], support)
    candidate_rotation = round14.compare(
        candidate_full["ua_exit"][0], candidate_full["un_e"][1], support)
    recorded_forward_mid = round14.compare(
        recorded["un_e"][1], recorded["ua_mid"][1], support)
    exit_mismatch = support & (
        candidate_full["ua_exit"][0] != recorded["ua_exit"][0])
    exit_coords = np.argwhere(exit_mismatch)

    midpoint_plant = {"requested": plant, "fires": None}
    exit_plant = {"requested": plant, "fires": None}
    if plant:
        planted_mid = np.array(recorded["ua_mid"][0], copy=True)
        mid_index = tuple(int(value) for value in np.argwhere(support)[0])
        planted_mid[mid_index] = np.nextafter(planted_mid[mid_index], np.inf)
        midpoint_plant.update({
            "index": list(mid_index),
            "fires": not round14.compare(
                replay_midpoint(
                    history_record["coefficients"][0], recorded["un_e"][0],
                    recorded["ub_e"][0], recorded["ubb_e"][0]),
                planted_mid, support)["bit_exact"],
        })
        planted_exit = np.array(recorded["ua_exit"][0], copy=True)
        exit_index = tuple(int(value) for value in np.argwhere(support)[0])
        planted_exit[exit_index] = np.nextafter(planted_exit[exit_index], np.inf)
        exit_plant.update({
            "index": list(exit_index),
            "fires": not round14.compare(
                planted_exit, recorded["ua_exit"][0], support)["bit_exact"],
        })
        require(bool(midpoint_plant["fires"]),
                "one-ULP midpoint target plant did not fire")
        require(bool(exit_plant["fires"]),
                "one-ULP exchanged-exit plant did not fire")

    stop_at_exchange = bool(
        first is not None
        and first["substep"] == 1
        and first["boundary"] == "ua_exit"
        and all(item["recorded"]["bit_exact"] and item["candidate"]["bit_exact"]
                for item in midpoint_replays)
        and all(item["recorded"]["bit_exact"] and item["candidate"]["bit_exact"]
                for item in vector_replays)
        and recorded_rotation["bit_exact"]
        and candidate_rotation["bit_exact"]
        and recorded_forward_mid["bit_exact"]
        and exit_coords.size > 0
        and np.all(exit_coords[:, 1] == 0)
    )
    return {
        "label": "given NEMO's entry",
        "halo_reproduction": halo_walk,
        "live_rows_in_compiled_source_order": rows,
        "raw_full_update_arrays_on_mismatch_adjacent_faces":
            raw_full_update_rows,
        "first_non_bit_statement": first,
        "midpoint_replays": midpoint_replays,
        "vector_replays_on_recorded_interior": vector_replays,
        "history_rotation": {
            "recorded_substep1_exit_to_substep2_entry": recorded_rotation,
            "candidate_substep1_exit_to_substep2_entry": candidate_rotation,
            "recorded_substep2_forward_midpoint": recorded_forward_mid,
        },
        "substep1_exit_mismatch": {
            "count": int(np.count_nonzero(exit_mismatch)),
            "coordinates": [[int(j), int(i)] for j, i in exit_coords],
            "all_on_unrecorded_zu_frc_halo": bool(
                exit_coords.size > 0 and np.all(exit_coords[:, 1] == 0)),
        },
        "stop_at_exchange_boundary": stop_at_exchange,
        "midpoint_plant": midpoint_plant,
        "exit_plant": exit_plant,
    }


def run(deck_root: Path, record_root: Path, json_out: Path | None,
        *, plant: bool = False) -> dict[str, object]:
    history_admission = admit(record_root, plant=plant)
    history_record = history_admission.pop("_records")[0]
    halo_admission = round18.admit(record_root, plant=plant)
    halo_record = halo_admission.pop("_records")[0]

    def extension(**context):
        return _history_walk(
            **context, halo_record=halo_record,
            history_record=history_record, plant=plant)

    inherited = round17.run(
        deck_root, record_root, None, plant=False, _extension=extension)
    walk = inherited["extension"]
    status = (
        "STOP_AT_POST_EXCHANGE_U_HALO"
        if walk["stop_at_exchange_boundary"]
        else "REFUTED_EXPECTED_U_HISTORY_BOUNDARY"
    )
    result = {
        "gate": "nemo_testcase_l4_orca2_round19_u_history_gate",
        "status": status,
        "label": "given NEMO's entry",
        "record_root": str(record_root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "history_admission": history_admission,
        "halo_admission": halo_admission,
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
    except (GateError, round17.GateError, round18.GateError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "status": result["status"],
        "history_admission": result["history_admission"],
        "round17_reproduction": result["round17_reproduction"],
        "walk": result["walk"],
    }, indent=2, sort_keys=True))
    if args.plant:
        plants = (
            result["history_admission"]["rank_header_plant"]["fires"],
            result["halo_admission"]["rank_header_plant"]["fires"],
            result["walk"]["halo_reproduction"]["payload_plant"]["fires"],
            result["walk"]["midpoint_plant"]["fires"],
            result["walk"]["exit_plant"]["fires"],
        )
        print("PLANTS FIRED" if all(plants) else "PLANTS DID NOT FIRE")
        return 1 if all(plants) else 2
    return 0 if result["status"] == "STOP_AT_POST_EXCHANGE_U_HALO" else 3


if __name__ == "__main__":
    raise SystemExit(main())
