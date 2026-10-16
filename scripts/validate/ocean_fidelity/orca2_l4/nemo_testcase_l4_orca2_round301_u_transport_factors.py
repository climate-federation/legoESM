#!/usr/bin/env python3
"""Split OMT-4's kt=7 stage-1 U transport ``zFu`` into its factors.

NEMO ``stp_RK3_stg`` (ORCA2_OMIP_L4_R299W7 ``stprk3_stg.f90:284,290``)::

    zub = un_adv*(r1_hu_0/(1+r3u(Kmm))) - uu_b(Kmm)
    zFu = e2u*(e3u_3d*(1+r3u(Kmm)*umask)) * (uu(Kmm) + zub*umask)

NEMO's factors are rebuilt from the admitted round-299 record and must
recompose the recorded ``trp_u`` bit for bit before any factor is scored.
The bt frames are rank-0 only, so every row is scored on columns 0-89.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round298_stage7_writer as r298,
    nemo_testcase_l4_orca2_round300_w_leaf_split as r300,
)

LABELS = r300.LABELS
WEST = 90
NZ = 30
# Compiled source order: the zub statement (:284) then the product (:290).
SOURCE_ORDER = (
    "un_adv", "ssh_kmm", "uu_b_kmm", "zub",
    "e2u", "umask", "e3u_kmm", "uu_kmm", "corrected_u", "zfu",
)
PLANTS = ("none", "closure", "one-bit", "label")
EXIT_REFUSE, EXIT_PLANT_GREEN = 2, 3


class GateError(RuntimeError):
    """The record, the NEMO closure, or the frozen classifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_mesh(record_root: Path) -> dict[str, np.ndarray]:
    """Owned 148x180 mesh fields from the two self-placing rank files."""
    import netCDF4

    out: dict[str, list] = {}
    for rank, first in ((0, 1), (1, 91)):
        with netCDF4.Dataset(record_root / f"mesh_mask_{rank:04d}.nc") as ds:
            require(list(ds.getncattr("DOMAIN_position_first")) == [first, 1]
                    and list(ds.getncattr("DOMAIN_size_global")) == [180, 148]
                    and list(ds.getncattr("DOMAIN_halo_size_start")) == [0, 0],
                    f"mesh rank {rank}: placement moved")
            for name in ("e3u_0", "umask", "e2u", "e1u", "e1t", "e2t"):
                value = np.asarray(ds[name][0], np.float64)
                if value.ndim == 3:
                    value = np.moveaxis(value, 0, -1)[..., :NZ]
                out.setdefault(name, []).append(value)
    return {name: np.concatenate(parts, axis=1) for name, parts in out.items()}


def read_bt(path: Path, kt: int) -> dict[str, np.ndarray]:
    """Rank-0 ``NEMO_L1_BTFRM_1`` frame (``stprk3.f90:308-328``), owned west half."""
    raw = path.read_bytes()
    magic = raw[:16].decode("ascii").rstrip()
    version, kstp, _level, nx, ny, bits = struct.unpack_from("=6i", raw, 16)
    require(magic == "NEMO_L1_BTFRM_1" and (version, kstp, bits) == (1, kt, 64)
            and (nx, ny) == (94, 152), f"{path.name}: header moved")
    count = nx * ny
    values = np.frombuffer(raw, "=f8", offset=40)
    require(values.size == 4 * count, f"{path.name}: payload length moved")
    xy = [values[n * count:(n + 1) * count].reshape((nx, ny), order="F")[2:-2, 2:-2].T
          for n in range(4)]
    return dict(zip(("uu_b", "vv_b", "un_adv", "vn_adv"), xy))


def nemo_factors(record_root: Path, plant: str = "none") -> dict[str, np.ndarray]:
    """NEMO's stage-1 factors in its own association; west half for bt-derived."""
    mesh = read_mesh(record_root)
    frame = r300.omt4.rung0.assemble_frame(record_root, 7, 0)
    bt6 = read_bt(record_root / "oracle_bt_frames_kt00000006.bin", 6)
    bt7 = read_bt(record_root / "oracle_bt_frames_kt00000007.bin", 7)
    e2u = mesh["e2u"].copy()
    if plant == "closure":
        e2u[147, 47] = np.nextafter(e2u[147, 47], np.inf)
    umask, e3u0 = mesh["umask"], mesh["e3u_0"]
    ssh = frame["ssh"]
    # domain.f90:194-213 (sequential k sum) and dommsk ssumask = MAXVAL(umask).
    hu0 = np.zeros(umask.shape[:2])
    for level in range(NZ):
        hu0 = hu0 + e3u0[..., level] * umask[..., level]
    ssumask = np.max(umask, axis=-1)
    r1_hu0 = ssumask / (hu0 + 1.0 - ssumask)
    # domqco.f90:266-267 (east T neighbour, cyclic in i).
    e1e2t_ssh = (mesh["e1t"] * mesh["e2t"]) * ssh
    r1_e1e2u = 1.0 / (mesh["e1u"] * e2u)
    r3u = 0.5 * (e1e2t_ssh + np.roll(e1e2t_ssh, -1, axis=1)) * r1_hu0 * r1_e1e2u
    e3u_kmm = e3u0 * (1.0 + r3u[..., None] * umask)
    w = np.s_[:, :WEST]
    zub = bt7["un_adv"] * (r1_hu0[w] / (1.0 + r3u[w])) - bt6["uu_b"]
    corrected = frame["u"][w] + zub[..., None] * umask[w]
    zfu = (e2u[w][..., None] * e3u_kmm[w]) * corrected
    return {
        "un_adv": bt7["un_adv"], "ssh_kmm": ssh[w], "uu_b_kmm": bt6["uu_b"],
        "zub": corrected[..., 0] - frame["u"][w][..., 0],
        "e2u": e2u[w], "umask": umask[w], "e3u_kmm": e3u_kmm[w],
        "uu_kmm": frame["u"][w], "corrected_u": corrected, "zfu": zfu,
    }


def closure(nemo: dict[str, np.ndarray], record_root: Path) -> dict[str, object]:
    oracle, _ = r300.assemble_record(
        record_root, record_root.parents[2] / "round297/acquisition/"
        "orca2_omt4_stage7_source_10step_np2")
    recorded = oracle["trp_u"][:, :WEST, :NZ]
    return r298._regions(nemo["zfu"], recorded, nemo["umask"] > 0.0)


def _u(value) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return value[:, 1:] if value.shape[1] == 181 else value


def card_factors(deck_root: Path, frame_root: Path, label: str) -> tuple[dict, dict]:
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d

    card, state, trace, passivity = r300.traced_kt7_stage1(
        deck_root, frame_root, label)
    geometry = trace.stage_geometry[0]
    u_mask, _ = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    w = np.s_[:, :WEST]
    uu = _u(trace.stage_states[0][0])[w]
    corrected = _u(geometry[9])[w]
    return {
        "un_adv": _u(trace.barotropic_targets[2])[w],
        "ssh_kmm": np.asarray(state.eta.data, np.float64)[w],
        "uu_b_kmm": _u(state.uu_b.data)[w],
        "zub": corrected[..., 0] - uu[..., 0],
        "e2u": _u(card.recipe.grid.dy_u)[w],
        "umask": _u(u_mask)[w],
        "e3u_kmm": _u(geometry[4])[w],
        "uu_kmm": uu, "corrected_u": corrected, "zfu": _u(geometry[7])[w],
    }, passivity


def _row(candidate, oracle, support) -> dict[str, object]:
    row = r298._regions(candidate, oracle, support)
    fold = np.abs(np.asarray(oracle)[-3:][support[-3:]])
    row["fold_oracle_max"] = float(np.max(fold))
    row["fold_relative"] = (row["fold_band"]["max_abs"] / row["fold_oracle_max"]
                            if row["fold_oracle_max"] > 0.0 else
                            float("inf") if row["fold_band"]["max_abs"] > 0.0 else 0.0)
    return row


def measure_label(deck_root, frame_root, record_root, label, plant="none"):
    nemo = nemo_factors(record_root, plant)
    closed = closure(nemo, record_root)
    card, passivity = card_factors(deck_root, frame_root, label)
    wet3 = nemo["umask"] > 0.0
    wet2 = wet3[..., 0]
    rows = {}
    for name in SOURCE_ORDER:
        support = wet3 if np.ndim(nemo[name]) == 3 else wet2
        if name == "umask":
            support = np.ones_like(wet3)
        rows[name] = _row(card[name], nemo[name], support)
    return {"label": label, "passivity": passivity, "closure": closed,
            "rows": rows,
            "one_bit_control": r298._score(
                np.asarray([1.0]), np.asarray([np.nextafter(1.0, 2.0)]),
                np.asarray([True]))}


def classify(reports: list[dict], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    require(sorted(r["label"] for r in reports) == sorted(LABELS),
            "both labels are required")
    if plant == "label":
        reports = json.loads(json.dumps(reports))
        reports[1]["rows"]["zfu"]["fold_band"]["max_abs"] *= 2.0
    for report in reports:
        require(all(report["passivity"].values()), "trace is not passive")
        require(report["closure"]["complete"]["unequal"] == 0,
                f"NEMO factor closure misses trp_u: {report['closure']['complete']}")
        control = report["one_bit_control"]
        require(control["unequal"] == 1 and plant != "one-bit",
                "one-bit control did not register")
    first = reports[0]["rows"]
    for report in reports[1:]:
        require(report["rows"] == first, "labels disagree")
    nonbit = [n for n in SOURCE_ORDER if first[n]["fold_band"]["unequal"] > 0]
    require(nonbit, "every factor is exact in the fold band")
    dominant = max(nonbit, key=lambda n: first[n]["fold_relative"]
                   if n not in {"corrected_u", "zfu"} else -1.0)
    return {
        "status": "HELD_R301_FOLD_FACTOR",
        "first_nonbit_factor": nonbit[0], "dominant_input_factor": dominant,
        "fold_relative": {n: first[n]["fold_relative"] for n in SOURCE_ORDER},
        "fold_band": {n: first[n]["fold_band"] for n in SOURCE_ORDER},
        "interior": {n: first[n]["interior"] for n in SOURCE_ORDER},
        "closure": reports[0]["closure"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--report-in", type=Path, action="append")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            import jax
            from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

            stamp = worktree_stamp()
            require(stamp["clean"] and stamp["commit"].lower()
                    == str(args.expect_commit).lower(),
                    "measurement requires its clean committed tree")
            policy = PrecisionPolicy.fp64(transcendentals="libm")
            set_policy(policy)
            require(get_policy() == policy and bool(jax.config.jax_enable_x64)
                    and jax.default_backend() == "cpu", "fp64/libm CPU not active")
            report = measure_label(args.deck_root, args.deck_root,
                                   args.record_root, args.label,
                                   "closure" if args.plant == "closure" else "none")
            report["worktree"] = stamp
            if args.raw_output:
                args.raw_output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n")
            reports = None
        else:
            require(bool(args.report_in), "--report-in is required")
            reports = [json.loads(p.read_text()) for p in args.report_in]
        if reports is None:
            require(report["closure"]["complete"]["unequal"] == 0,
                    f"NEMO factor closure misses trp_u: {report['closure']['complete']}")
            print(f"STATUS MEASURED {args.label}")
            return 0
        result = classify(reports, args.plant)
    except (GateError, OSError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return EXIT_REFUSE
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if args.plant != "none":
        print(f"STATUS PLANT-GREEN {args.plant}: the plant did not refuse")
        return EXIT_PLANT_GREEN
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
