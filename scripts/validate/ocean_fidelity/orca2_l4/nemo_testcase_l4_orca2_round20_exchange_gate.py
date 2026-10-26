#!/usr/bin/env python3
"""Admit ORCA2 pre-exchange U and walk the disputed periodic seam."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round19_u_history_gate as round19,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo"
CITATIONS = {
    "preexchange_payload_and_call": f"{_PP}/dynspg_ts.f90:766-786",
    "point_to_point_dispatch": f"{_PP}/lbclnk.f90:623-627",
    "halo_offsets": f"{_PP}/lbclnk.f90:1889-1909",
    "east_west_pack": f"{_PP}/lbclnk.f90:1960-1979",
    "east_west_receive": f"{_PP}/lbclnk.f90:2055-2073",
}

MAGIC = b"NEMO_L4_PREX_U1 "
JPI = 94
JPJ = 152
NROWS = 2
HEADER_INTS = 7
EXPECTED_BYTES = 16 + HEADER_INTS * 4 + NROWS * (4 + JPI * JPJ * 8)
INNER_ROWS = slice(2, 150)
NON_FOLD_ROWS = slice(2, 149)


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


def read_preexchange(path: Path, expected_rank: int) -> dict[str, object]:
    require(path.is_file(), f"ranked pre-exchange stream is missing: {path}")
    require(path.stat().st_size == EXPECTED_BYTES,
            f"ranked pre-exchange stream has wrong byte count: {path}")
    substeps = []
    rows = []
    with path.open("rb") as stream:
        magic = stream.read(16)
        header = np.fromfile(stream, dtype=np.int32, count=HEADER_INTS)
        require(magic == MAGIC, f"pre-exchange magic changed: {path}")
        require(header.size == HEADER_INTS, f"short pre-exchange header: {path}")
        version, kt, nrows, rank, jpi, jpj, storage_bits = (
            int(value) for value in header)
        require(
            (version, kt, nrows, rank, jpi, jpj, storage_bits)
            == (1, 1, NROWS, expected_rank, JPI, JPJ, 64),
            f"pre-exchange header mismatch: {path}",
        )
        for expected_substep in range(1, NROWS + 1):
            step = np.fromfile(stream, dtype=np.int32, count=1)
            require(step.size == 1 and int(step[0]) == expected_substep,
                    f"pre-exchange substep order changed: {path}")
            payload = np.fromfile(stream, dtype=np.float64, count=JPI * JPJ)
            require(payload.size == JPI * JPJ,
                    f"short pre-exchange payload: {path}")
            substeps.append(int(step[0]))
            rows.append(payload.reshape((JPI, JPJ), order="F").T)
        require(stream.read(1) == b"", f"trailing pre-exchange bytes: {path}")
    return {
        "rank": rank,
        "substeps": substeps,
        "u": np.stack(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _contains_exact_line(text: str, line: str) -> bool:
    return line in {candidate.strip() for candidate in text.splitlines()}


def admit(root: Path, *, plant: bool = False) -> dict[str, object]:
    time_text = (root / "run.user.time.log").read_text()
    stdout_text = (root / "run.user.stdout.log").read_text()
    ocean_text = (root / "ocean.output").read_text()
    require(_contains_exact_line(time_text, "MPIRUN_RC=0"),
            "run does not carry MPIRUN_RC=0")
    require(_contains_exact_line(time_text, "RUN DONE"),
            "run does not carry RUN DONE")
    require(_contains_exact_line(stdout_text, "STOP 0"),
            "run does not carry STOP 0")
    for text, rank in ((ocean_text, 0), (stdout_text, 1)):
        marker = (
            f"LANE4_BT_PREX_DUMP            1           2           {rank} "
            f"oracle_bt_u_preexchange_kt00000001_r{rank:04d}.bin")
        require(sum(line.strip() == marker for line in text.splitlines()) == 1,
                f"rank-{rank} pre-exchange marker is not unique")
    records = [read_preexchange(
        root / f"oracle_bt_u_preexchange_kt00000001_r{rank:04d}.bin", rank)
        for rank in range(2)]
    require(records[0]["sha256"] != records[1]["sha256"],
            "the ranked pre-exchange streams are byte-identical")
    rank_plant = {"requested": plant, "fires": None}
    if plant:
        try:
            read_preexchange(
                root / "oracle_bt_u_preexchange_kt00000001_r0000.bin", 1)
        except GateError:
            rank_plant["fires"] = True
        else:
            rank_plant["fires"] = False
        require(bool(rank_plant["fires"]), "swapped-rank plant did not fire")
    return {
        "status": "ADMITTED_EXISTING_RANKED_PREEXCHANGE_RECORD",
        "record_bytes": EXPECTED_BYTES,
        "ranks": [{
            "rank": record["rank"],
            "substeps": record["substeps"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        } for record in records],
        "rank_header_plant": rank_plant,
        "_records": records,
    }


def _compare(candidate, oracle, mask=None) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape, "exchange comparison shape changed")
    if mask is None:
        mask = np.ones(candidate.shape, dtype=bool)
    return round14.compare(candidate, oracle, np.asarray(mask, dtype=bool))


def replay_vector(un_e, rdt, zu_spg, zu_trd, zu_frc, ssumask):
    return round19.replay_vector(un_e, rdt, zu_spg, zu_trd, zu_frc, ssumask)


def _exchange_walk(*, card, trace, candidate, oracle, masks, inherited_first,
                   halo_record, history_records, pre_records, plant):
    inherited = round19._history_walk(
        card=card, trace=trace, candidate=candidate, oracle=oracle, masks=masks,
        inherited_first=inherited_first, halo_record=halo_record,
        history_record=history_records[0], plant=plant)
    require(inherited["stop_at_exchange_boundary"],
            "round-19 exchange boundary did not reproduce")

    rank0_pre = pre_records[0]["u"]
    rank1_pre = pre_records[1]["u"]
    rank0_post = history_records[0]["fields"]["ua_exit"]
    rank1_post = history_records[1]["fields"]["ua_exit"]

    # ihls=2 mappings from the compiled offsets.  Exclude the final owned row
    # here because the same lbc_lnk call subsequently applies the north fold.
    mappings = {
        "rank0_west_from_rank1_east": _compare(
            rank0_post[:, NON_FOLD_ROWS, 0:2],
            rank1_pre[:, NON_FOLD_ROWS, 90:92]),
        "rank0_east_from_rank1_west": _compare(
            rank0_post[:, NON_FOLD_ROWS, 92:94],
            rank1_pre[:, NON_FOLD_ROWS, 2:4]),
        "rank1_west_from_rank0_east": _compare(
            rank1_post[:, NON_FOLD_ROWS, 0:2],
            rank0_pre[:, NON_FOLD_ROWS, 90:92]),
        "rank1_east_from_rank0_west": _compare(
            rank1_post[:, NON_FOLD_ROWS, 92:94],
            rank0_pre[:, NON_FOLD_ROWS, 2:4]),
    }

    exit_coords = np.asarray(inherited["substep1_exit_mismatch"]["coordinates"])
    require(exit_coords.shape == (64, 2) and np.all(exit_coords[:, 1] == 0),
            "round-19 west-seam support changed")
    support_rows = exit_coords[:, 0].astype(int)
    local_rows = support_rows + 2

    source = rank1_pre[0, local_rows, 91]
    destination = rank0_post[0, local_rows, 1]
    overwritten = rank0_pre[0, local_rows, 1]
    candidate_west = np.asarray(trace["u_exit"], dtype=np.float64)[0, support_rows, 0]
    candidate_east = np.asarray(trace["u_exit"], dtype=np.float64)[0, support_rows, -1]
    disputed = {
        "rank0_post_vs_rank1_pre_source": _compare(destination, source),
        "rank0_pre_vs_rank1_pre_source": _compare(overwritten, source),
        "candidate_east_vs_rank1_pre_source": _compare(candidate_east, source),
        "candidate_west_vs_rank1_pre_source": _compare(candidate_west, source),
        "candidate_west_vs_candidate_east": _compare(
            candidate_west, candidate_east),
    }

    recorded = history_records[1]["fields"]
    candidate_fields = {
        "un_e": np.asarray(trace["u_entry"], dtype=np.float64),
        "ub_e": np.asarray(trace["u_history_b"], dtype=np.float64),
        "ubb_e": np.asarray(trace["u_history_bb"], dtype=np.float64),
        "ua_mid": np.asarray(trace["u_mid"], dtype=np.float64),
        "zu_spg": np.asarray(trace["pgf_u"], dtype=np.float64),
        "zu_trd": np.asarray(trace["trd_u"], dtype=np.float64),
        "ssumask": np.broadcast_to(
            np.asarray(card.recipe.initial_state.u_mask.data, dtype=np.float64),
            np.asarray(trace["u_exit"]).shape),
        "ua_exit": np.asarray(trace["u_exit"], dtype=np.float64),
    }
    candidate_coefficients = np.stack([
        np.asarray(trace[f"mid_weight_{index}"], dtype=np.float64)[:NROWS]
        for index in (1, 2, 3)
    ], axis=-1)
    candidate_rdt = np.full(
        NROWS,
        float(card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps),
        dtype=np.float64)
    source_rows = []
    replay_rows = []
    for substep in range(NROWS):
        row = {
            "substep": substep + 1,
            **{
                f"coefficient_{index + 1}": _compare(
                    np.asarray([candidate_coefficients[substep, index]]),
                    np.asarray([history_records[1]["coefficients"][substep, index]]))
                for index in range(3)
            },
            **{
                name: _compare(
                    candidate_fields[name][substep, support_rows, -1],
                    recorded[name][substep, local_rows, 91])
                for name in (
                    "un_e", "ub_e", "ubb_e", "ua_mid", "zu_spg", "zu_trd",
                    "ssumask", "ua_exit")
            },
            "rDt_e": _compare(
                np.asarray([candidate_rdt[substep]]),
                np.asarray([history_records[1]["rDt_e"][substep]])),
            "zu_frc": _compare(
                np.asarray(trace["slow_u"], dtype=np.float64)[
                    substep, support_rows, -1],
                recorded["zu_frc"][substep, support_rows, 89]),
        }
        source_rows.append(row)
        recorded_mid = round19.replay_midpoint(
            history_records[1]["coefficients"][substep],
            recorded["un_e"][substep, local_rows, 91],
            recorded["ub_e"][substep, local_rows, 91],
            recorded["ubb_e"][substep, local_rows, 91])
        candidate_mid = round19.replay_midpoint(
            candidate_coefficients[substep],
            candidate_fields["un_e"][substep, support_rows, -1],
            candidate_fields["ub_e"][substep, support_rows, -1],
            candidate_fields["ubb_e"][substep, support_rows, -1])
        recorded_exit = replay_vector(
            recorded["un_e"][substep, local_rows, 91],
            history_records[1]["rDt_e"][substep],
            recorded["zu_spg"][substep, local_rows, 91],
            recorded["zu_trd"][substep, local_rows, 91],
            recorded["zu_frc"][substep, support_rows, 89],
            recorded["ssumask"][substep, local_rows, 91])
        candidate_exit = replay_vector(
            candidate_fields["un_e"][substep, support_rows, -1],
            candidate_rdt[substep],
            candidate_fields["zu_spg"][substep, support_rows, -1],
            candidate_fields["zu_trd"][substep, support_rows, -1],
            np.asarray(trace["slow_u"], dtype=np.float64)[
                substep, support_rows, -1],
            candidate_fields["ssumask"][substep, support_rows, -1])
        candidate_with_recorded_slow = replay_vector(
            candidate_fields["un_e"][substep, support_rows, -1],
            candidate_rdt[substep],
            candidate_fields["zu_spg"][substep, support_rows, -1],
            candidate_fields["zu_trd"][substep, support_rows, -1],
            recorded["zu_frc"][substep, support_rows, 89],
            candidate_fields["ssumask"][substep, support_rows, -1])
        replay_rows.append({
            "substep": substep + 1,
            "recorded_midpoint": _compare(
                recorded_mid, recorded["ua_mid"][substep, local_rows, 91]),
            "candidate_midpoint": _compare(
                candidate_mid,
                candidate_fields["ua_mid"][substep, support_rows, -1]),
            "recorded_vector": _compare(
                recorded_exit, recorded["ua_exit"][substep, local_rows, 91]),
            "candidate_vector": _compare(
                candidate_exit,
                candidate_fields["ua_exit"][substep, support_rows, -1]),
            "candidate_with_recorded_zu_frc_vs_recorded_exit": _compare(
                candidate_with_recorded_slow,
                recorded["ua_exit"][substep, local_rows, 91]),
        })
    first_source = round19.first_non_bit(source_rows)

    canonical_u = round18._candidate_u_window(trace["u_mid"]).copy()
    canonical_u[:, :, 0] = np.asarray(trace["u_mid"], dtype=np.float64)[
        :NROWS, :, -1]
    e2u = np.broadcast_to(
        round18._candidate_u_window(
            np.asarray(card.recipe.grid.dy_u, dtype=np.float64)[None, ...]),
        canonical_u.shape)
    depth = round18._candidate_u_window(trace["transport_face_depth_u"])
    canonical_transport = np.multiply(np.multiply(e2u, canonical_u), depth)
    canonical_du = np.subtract(
        canonical_transport[1, :, 1:], canonical_transport[1, :, :-1])
    mismatch_t = masks["t"] & (
        candidate["continuity_du"][1] != oracle["continuity_du"][1])
    canonical = {
        "substep1_exit_vs_rank0_post": _compare(candidate_east, destination),
        "substep2_midpoint_vs_rank0_record": _compare(
            canonical_u[1, support_rows, 0],
            recorded["ua_mid"][1, local_rows, 1]),
        "substep2_continuity_vs_oracle": _compare(
            canonical_du, oracle["continuity_du"][1], mismatch_t),
    }

    plants = {
        "requested": plant,
        "wrong_neighbor_fires": None,
        "source_ulp_fires": None,
    }
    if plant:
        wrong_neighbor = rank1_pre[0, local_rows, 90]
        plants["wrong_neighbor_fires"] = not _compare(
            destination, wrong_neighbor)["bit_exact"]
        planted = np.array(source, copy=True)
        planted[0] = np.nextafter(planted[0], np.inf)
        plants["source_ulp_fires"] = not _compare(
            destination, planted)["bit_exact"]
        require(bool(plants["wrong_neighbor_fires"]),
                "wrong-neighbor face plant did not fire")
        require(bool(plants["source_ulp_fires"]),
                "one-ULP source plant did not fire")

    exchange_refuted_slow_forcing_owned = bool(
        all(row["bit_exact"] for row in mappings.values())
        and disputed["rank0_post_vs_rank1_pre_source"]["bit_exact"]
        and not disputed["rank0_pre_vs_rank1_pre_source"]["bit_exact"]
        and not disputed["candidate_east_vs_rank1_pre_source"]["bit_exact"]
        and disputed["candidate_west_vs_candidate_east"]["bit_exact"]
        and first_source is not None
        and first_source["substep"] == 1
        and first_source["boundary"] == "zu_frc"
        and all(
            value["bit_exact"]
            for replay in replay_rows for key, value in replay.items()
            if key != "substep" and key != "candidate_with_recorded_zu_frc_vs_recorded_exit")
        and replay_rows[0]["candidate_with_recorded_zu_frc_vs_recorded_exit"]["bit_exact"]
    )
    return {
        "label": "given NEMO's entry",
        "round19_reproduction": inherited,
        "compiled_two_halo_mappings_excluding_north_fold_row": mappings,
        "disputed_live_face": disputed,
        "rank1_source_rows_in_compiled_order": source_rows,
        "first_non_bit_rank1_source_statement": first_source,
        "source_expression_replays": replay_rows,
        "canonical_east_to_west_arm": canonical,
        "plants": plants,
        "exchange_hypothesis_refuted_and_slow_forcing_owned":
            exchange_refuted_slow_forcing_owned,
        "single_statement_eligible": False,
    }


def run(deck_root: Path, record_root: Path, json_out: Path | None,
        *, plant: bool = False) -> dict[str, object]:
    pre_admission = admit(record_root, plant=plant)
    pre_records = pre_admission.pop("_records")
    history_admission = round19.admit(record_root, plant=plant)
    history_records = history_admission.pop("_records")
    halo_admission = round18.admit(record_root, plant=plant)
    halo_record = halo_admission.pop("_records")[0]

    def extension(**context):
        return _exchange_walk(
            **context, halo_record=halo_record,
            history_records=history_records,
            pre_records=pre_records, plant=plant)

    inherited = round17.run(
        deck_root, record_root, None, plant=False, _extension=extension)
    walk = inherited["extension"]
    status = (
        "STOP_AT_RANK1_SLOW_FORCING_RECORD"
        if walk["exchange_hypothesis_refuted_and_slow_forcing_owned"]
        else "STOP_AT_RANK1_SOURCE_OR_MAPPING"
    )
    result = {
        "gate": "nemo_testcase_l4_orca2_round20_exchange_gate",
        "status": status,
        "label": "given NEMO's entry",
        "record_root": str(record_root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "preexchange_admission": pre_admission,
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
    except (GateError, round17.GateError, round18.GateError, round19.GateError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "status": result["status"],
        "preexchange_admission": result["preexchange_admission"],
        "round17_reproduction": result["round17_reproduction"],
        "walk": result["walk"],
    }, indent=2, sort_keys=True))
    if args.plant:
        plants = (
            result["preexchange_admission"]["rank_header_plant"]["fires"],
            result["history_admission"]["rank_header_plant"]["fires"],
            result["halo_admission"]["rank_header_plant"]["fires"],
            result["walk"]["round19_reproduction"]["midpoint_plant"]["fires"],
            result["walk"]["round19_reproduction"]["exit_plant"]["fires"],
            result["walk"]["plants"]["wrong_neighbor_fires"],
            result["walk"]["plants"]["source_ulp_fires"],
        )
        print("PLANTS FIRED" if all(plants) else "PLANTS DID NOT FIRE")
        return 1 if all(plants) else 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
