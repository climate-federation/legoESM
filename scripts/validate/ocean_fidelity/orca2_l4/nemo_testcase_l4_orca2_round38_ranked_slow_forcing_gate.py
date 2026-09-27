#!/usr/bin/env python3
"""Round 38: admit and walk rank 1's recorded slow-forcing producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round20_exchange_gate as round20,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
CITATIONS = {
    "vertical_average": f"{_PP}/stp2d.f90:189-204",
    "baroclinic_drag": f"{_PP}/stp2d.f90:218-221",
    "wind_forcing": f"{_PP}/stp2d.f90:225-235",
}

MAGIC = "NEMO_L4_SLOW_R1"
DIMS = (94, 152, 31)
RECORD_BYTES = 22_814_424
RANK_COLUMNS = 90
ROUND20_COUNT = 64
ROUND20_MAX = 7.356481146903598e-18


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


def _full_2d(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F").T


def _full_3d(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)


def read_ranked(path: Path, expected_rank: int) -> dict[str, object]:
    """Read the committed ranked schema with exact header, size, and EOF checks."""
    require(path.is_file(), f"rank-{expected_rank} record is missing: {path}")
    require(path.stat().st_size == RECORD_BYTES,
            f"rank-{expected_rank} record byte count changed: {path}")
    with path.open("rb") as stream:
        magic = stream.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", stream.read(36))
        version, kt, kbb, krhs, rank, nx, ny, nz, bits = header
        sizes = struct.unpack("=7i", stream.read(28))
        expected_sizes = (nx * ny * nz,) * 6 + ((nx - 4) * (ny - 4),)
        require(
            (magic, *header, sizes)
            == (MAGIC, 1, 1, 1, 3, expected_rank, *DIMS, 64,
                expected_sizes),
            f"rank-{expected_rank} header changed: {(magic, *header, sizes)}",
        )

        def field3() -> np.ndarray:
            values = np.fromfile(stream, dtype=np.float64, count=nx * ny * nz)
            require(values.size == nx * ny * nz, f"short 3-D field: {path}")
            return _full_3d(values, nx, ny, nz)[2:-2, 2:-2, : nz - 1]

        def full2() -> np.ndarray:
            values = np.fromfile(stream, dtype=np.float64, count=nx * ny)
            require(values.size == nx * ny, f"short full 2-D field: {path}")
            return _full_2d(values, nx, ny)

        def interior2() -> np.ndarray:
            count = (nx - 4) * (ny - 4)
            values = np.fromfile(stream, dtype=np.float64, count=count)
            require(values.size == count, f"short interior field: {path}")
            return values.reshape((nx - 4, ny - 4), order="F").T

        result = {
            "e3u": field3(), "krhs_u": field3(), "umask": field3(),
            "e3v": field3(), "krhs_v": field3(), "vmask": field3(),
            "depth_u": interior2(), "depth_v": interior2(),
            "r1_hu0": full2()[2:-2, 2:-2],
            "r1_hv0": full2()[2:-2, 2:-2],
            "post_drag_u": interior2(), "post_drag_v": interior2(),
            "cd_u": full2()[2:-2, 2:-2],
            "cd_v": full2()[2:-2, 2:-2],
        }
        scalar = np.fromfile(stream, dtype=np.float64, count=1)
        require(scalar.size == 1, f"short r1_rho0: {path}")
        result["r1_rho0"] = float(scalar[0])
        result["utau"] = full2()[2:-2, 2:-2]
        result["vtau"] = full2()[2:-2, 2:-2]
        result["r1_hu"] = full2()[2:-2, 2:-2]
        result["r1_hv"] = full2()[2:-2, 2:-2]
        result["post_wind_u"] = interior2()
        result["post_wind_v"] = interior2()
        require(stream.read(1) == b"", f"trailing payload: {path}")
    result["rank"] = rank
    result["sha256"] = _sha256(path)
    result["bytes"] = path.stat().st_size
    return result


def admit(root: Path, *, plant: bool = False) -> dict[str, object]:
    time_text = (root / "run.user.time.log").read_text()
    stdout_text = (root / "run.user.stdout.log").read_text()
    ocean_text = (root / "ocean.output").read_text()
    require("MPIRUN_RC=0" in time_text and "RUN DONE" in time_text,
            "completion time log changed")
    require(any(line.strip() == "STOP 0" for line in stdout_text.splitlines()),
            "stdout does not carry STOP 0")
    records = [read_ranked(
        root / f"oracle_slow_forcing_ranked_kt00000001_r{rank:04d}.bin", rank)
        for rank in range(2)]
    require(records[0]["sha256"] != records[1]["sha256"],
            "ranked slow-forcing streams are byte-identical")
    for text, rank in ((ocean_text, 0), (stdout_text, 1)):
        marker = f"oracle_slow_forcing_ranked_kt00000001_r{rank:04d}.bin"
        require(sum(marker in line for line in text.splitlines()) == 1,
                f"rank-{rank} marker is not unique")
    rank_plant = {"requested": plant, "fires": None}
    if plant:
        try:
            read_ranked(
                root / "oracle_slow_forcing_ranked_kt00000001_r0000.bin", 1)
        except GateError:
            rank_plant["fires"] = True
        else:
            rank_plant["fires"] = False
        require(bool(rank_plant["fires"]), "swapped-rank plant did not fire")
    return {
        "status": "ADMITTED_EXISTING_RANKED_SLOW_FORCING",
        "records": records,
        "rank_header_plant": rank_plant,
    }


def _native_rank(values, face: str, rank: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    native = values[:, 1:, ...] if face == "u" else values[1:, :, ...]
    start = rank * RANK_COLUMNS
    return native[:, start:start + RANK_COLUMNS, ...]


def _candidate_arrays(operands, masks, face: str, rank: int) -> dict[str, np.ndarray]:
    suffix = face
    e3 = _native_rank(operands[f"h_{suffix}"], face, rank)
    rhs = _native_rank(operands[f"d{suffix}_dt"], face, rank)
    depth = _native_rank(operands[f"depth_{suffix}"], face, rank)
    post_wind = _native_rank(operands[f"post_wind_{suffix}"], face, rank)
    post_drag = _native_rank(operands[f"post_drag_{suffix}"], face, rank)
    final = _native_rank(operands[f"pre_external_{suffix}"], face, rank)
    tau = _native_rank(operands[f"wind_tau_{suffix}"], face, rank)
    total_depth = _native_rank(operands[f"H_{suffix}"], face, rank)
    return {
        "e3": e3,
        "krhs": rhs,
        "mask": _native_rank(masks[face], face, rank),
        "r1_h0": 1.0 / total_depth,
        "depth_mean": depth,
        "post_wind": post_wind,
        "post_drag": post_drag,
        "final": final,
        "tau": tau,
        "drag_increment": post_drag - post_wind,
        "wind_increment": post_wind - depth,
    }


def _oracle_arrays(record: dict[str, object], face: str) -> dict[str, np.ndarray]:
    out = {
        "e3": record[f"e3{face}"],
        "krhs": record[f"krhs_{face}"],
        "mask": record[f"{face}mask"],
        "r1_h0": record[f"r1_h{face}0"],
        "depth_mean": record[f"depth_{face}"],
        "post_drag": record[f"post_drag_{face}"],
        "post_wind": record[f"post_wind_{face}"],
        "tau": record[f"{face}tau"],
    }
    out["drag_increment"] = out["post_drag"] - out["depth_mean"]
    out["wind_increment"] = out["post_wind"] - out["post_drag"]
    out["final"] = out["post_wind"]
    return out


def _support_row(left, right, support_rows, *, is3d=False):
    left = np.asarray(left)
    right = np.asarray(right)
    if is3d:
        left = left[support_rows, -1, :]
        right = right[support_rows, -1, :]
    else:
        left = left[support_rows, -1]
        right = right[support_rows, -1]
    return round14.compare(left, right, np.ones(left.shape, dtype=bool))


def run(deck_root: Path, boundary_root: Path, ranked_root: Path,
        json_out: Path | None, *, plant: bool = False) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    admission = admit(ranked_root, plant=plant)
    records = admission.pop("records")
    inherited = round20.run(deck_root, boundary_root, None, plant=False)
    first = inherited["walk"]["first_non_bit_rank1_source_statement"]
    replay = inherited["walk"]["source_expression_replays"][0][
        "candidate_with_recorded_zu_frc_vs_recorded_exit"]
    require(first["substep"] == 1 and first["boundary"] == "zu_frc"
            and first["differing_cells"] == ROUND20_COUNT
            and first["absolute_max"] == ROUND20_MAX and replay["bit_exact"],
            "round-20 rank-1 slow-forcing boundary did not reproduce")
    coordinates = np.asarray(inherited["walk"]["round19_reproduction"]
                             ["substep1_exit_mismatch"]["coordinates"])
    support_rows = coordinates[:, 0].astype(int)

    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(boundary_root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(boundary_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    trace = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    operands = trace.slow_forcing_operands
    mask_u, mask_v = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    masks = {"u": np.asarray(mask_u), "v": np.asarray(mask_v)}

    walks = {}
    controls = {}
    first_non_bit = None
    order = ("e3", "krhs", "mask", "r1_h0", "depth_mean",
             "drag_increment", "wind_increment", "tau", "final")
    for rank in range(2):
        walks[str(rank)] = {}
        controls[str(rank)] = {}
        for face in ("u", "v"):
            candidate = _candidate_arrays(operands, masks, face, rank)
            oracle = _oracle_arrays(records[rank], face)
            active3 = np.asarray(oracle["mask"]) > 0.0
            active2 = active3[..., 0]
            own_depth = round14._source_sum(
                oracle["e3"], oracle["krhs"], oracle["mask"], oracle["r1_h0"])
            own_wind = (oracle["post_drag"]
                        + (records[rank]["r1_rho0"] * oracle["tau"])
                        * records[rank][f"r1_h{face}"])
            controls[str(rank)][face] = {
                "record_replays_depth_mean": round14.compare(
                    own_depth, oracle["depth_mean"], active2),
                "record_replays_post_wind": round14.compare(
                    own_wind, oracle["post_wind"], active2),
            }
            require(all(row["bit_exact"] for row in controls[str(rank)][face].values()),
                    f"rank-{rank} {face} record does not replay itself")
            rows = {}
            for name in order:
                active = active3 if name in ("e3", "krhs", "mask") else active2
                rows[name] = round14.compare(
                    candidate[name], oracle[name], active)
                if first_non_bit is None and not rows[name]["bit_exact"]:
                    first_non_bit = {"rank": rank, "face": face,
                                     "boundary": name, **rows[name]}
            walks[str(rank)][face] = rows

    candidate_u = _candidate_arrays(operands, masks, "u", 1)
    oracle_u = _oracle_arrays(records[1], "u")
    support = {
        name: _support_row(candidate_u[name], oracle_u[name], support_rows,
                           is3d=name in ("e3", "krhs", "mask"))
        for name in order
    }
    first_support = next(
        ({"boundary": name, **support[name]} for name in order
         if not support[name]["bit_exact"]), None)
    literal_parent = round14._source_sum(
        candidate_u["e3"], candidate_u["krhs"], candidate_u["mask"],
        candidate_u["r1_h0"])
    literal_rhs_arm = round14._source_sum(
        candidate_u["e3"], oracle_u["krhs"], candidate_u["mask"],
        candidate_u["r1_h0"])
    literal_e3_arm = round14._source_sum(
        oracle_u["e3"], candidate_u["krhs"], candidate_u["mask"],
        candidate_u["r1_h0"])
    arms = {
        "parent_depth_vs_record": _support_row(
            literal_parent, oracle_u["depth_mean"], support_rows),
        "recorded_e3_only_vs_record": _support_row(
            literal_e3_arm, oracle_u["depth_mean"], support_rows),
        "recorded_rhs_only_vs_record": _support_row(
            literal_rhs_arm, oracle_u["depth_mean"], support_rows),
    }
    candidate_product = ((candidate_u["e3"] * candidate_u["krhs"])
                         * candidate_u["mask"])
    oracle_product = ((oracle_u["e3"] * oracle_u["krhs"])
                      * oracle_u["mask"])
    candidate_support_product = candidate_product[support_rows, -1, :]
    oracle_support_product = oracle_product[support_rows, -1, :]
    product_row = round14.compare(
        candidate_support_product, oracle_support_product,
        np.ones(candidate_support_product.shape, dtype=bool))
    candidate_partial = np.cumsum(candidate_support_product, axis=-1)
    oracle_partial = np.cumsum(oracle_support_product, axis=-1)
    partial_rows = [round14.compare(
        candidate_partial[:, level], oracle_partial[:, level],
        np.ones(candidate_partial.shape[0], dtype=bool))
        for level in range(candidate_partial.shape[-1])]
    first_partial = next((
        {"level_zero_based": level, **row}
        for level, row in enumerate(partial_rows) if not row["bit_exact"]
    ), None)
    candidate_reciprocal = candidate_u["r1_h0"][support_rows, -1]
    oracle_reciprocal = oracle_u["r1_h0"][support_rows, -1]
    oracle_depth = oracle_u["depth_mean"][support_rows, -1]
    product_walk = {
        "per_level_product": product_row,
        "first_non_bit_partial_sum": first_partial,
        "completed_sum": round14.compare(
            candidate_partial[:, -1], oracle_partial[:, -1],
            np.ones(candidate_partial.shape[0], dtype=bool)),
        "candidate_sum_times_candidate_reciprocal": round14.compare(
            candidate_partial[:, -1] * candidate_reciprocal, oracle_depth,
            np.ones(candidate_partial.shape[0], dtype=bool)),
        "oracle_sum_times_candidate_reciprocal": round14.compare(
            oracle_partial[:, -1] * candidate_reciprocal, oracle_depth,
            np.ones(candidate_partial.shape[0], dtype=bool)),
        "candidate_sum_times_oracle_reciprocal": round14.compare(
            candidate_partial[:, -1] * oracle_reciprocal, oracle_depth,
            np.ones(candidate_partial.shape[0], dtype=bool)),
        "product_plant": {"requested": plant, "fires": None},
    }
    if plant:
        planted_product = np.array(oracle_support_product, copy=True)
        planted_product[0, 0] = np.nextafter(planted_product[0, 0], np.inf)
        product_walk["product_plant"]["fires"] = not round14.compare(
            oracle_support_product, planted_product,
            np.ones(planted_product.shape, dtype=bool))["bit_exact"]
        require(bool(product_walk["product_plant"]["fires"]),
                "one-ULP product plant did not fire")
    scientific_plant = {"requested": plant, "fires": None}
    if plant:
        planted_target = np.array(oracle_u["depth_mean"], copy=True)
        index = (support_rows[0], planted_target.shape[1] - 1)
        planted_target[index] = np.nextafter(planted_target[index], np.inf)
        record_replay = round14._source_sum(
            oracle_u["e3"], oracle_u["krhs"], oracle_u["mask"],
            oracle_u["r1_h0"])
        scientific_plant["fires"] = not _support_row(
            record_replay, planted_target, support_rows)["bit_exact"]
        require(bool(scientific_plant["fires"]),
                "one-ULP scored-boundary plant did not fire")

    result = {
        "gate": "nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate",
        "status": "HELD_AT_FIRST_PRODUCER_STATEMENT",
        "label": "given NEMO's entry",
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "admission": admission,
        "round20_reproduction": {
            "first_non_bit": first,
            "recorded_zu_frc_closes_exit": replay,
        },
        "record_self_replays": controls,
        "full_rank_walks": walks,
        "first_non_bit_full_walk": first_non_bit,
        "rank1_disputed_source_walk": support,
        "first_non_bit_rank1_disputed_source": first_support,
        "one_variable_depth_arms": arms,
        "compiled_product_walk": product_walk,
        "scientific_plant": scientific_plant,
        "single_statement_eligible": False,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--ranked-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.boundary_root, args.ranked_root,
                     args.json_out, plant=args.plant)
    except (GateError, round20.GateError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "admission": result["admission"],
        "round20_reproduction": result["round20_reproduction"],
        "first_non_bit_full_walk": result["first_non_bit_full_walk"],
        "first_non_bit_rank1_disputed_source":
            result["first_non_bit_rank1_disputed_source"],
        "one_variable_depth_arms": result["one_variable_depth_arms"],
        "scientific_plant": result["scientific_plant"],
    }, indent=2, sort_keys=True))
    return 1 if args.plant else 3


if __name__ == "__main__":
    raise SystemExit(main())
