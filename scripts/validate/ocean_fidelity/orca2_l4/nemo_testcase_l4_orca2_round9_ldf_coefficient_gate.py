#!/usr/bin/env python3
"""Bit-exact gate for ORCA2's READ lateral momentum viscosity coefficient.

The resolved ORCA2 run prints ``nn_ahm_ijk_t = -30`` (run
``ocean.output:1184``), so NEMO computes no coefficient: ``ldf_dyn_init`` reads
``ahmt_3d``/``ahmf_3d`` whole from ``eddy_viscosity_3D.nc``
(``ldfdyn.f90:348-353``), the read path completes each field with the lateral
boundary exchange for its own grid-point nature (``iom.f90:958-975``), and the
laplacian arm then multiplies levels one to ``jpkm1`` by ``tmask``/``fmask``
(``ldfdyn.f90:388-393``).

This gate compares the card's carried coefficient against NEMO's own recorded
``ahmt``/``ahmf`` streams in ``output.init`` on EVERY owned cell of BOTH ranks,
and refuses unless three controls each leave it unequal.  It also refuses
unless the input file is already consistent with both compiled T-pivot fold
rules, which is what licenses the transcription to omit the exchange.
"""

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
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

GLOBAL_NX, GLOBAL_NY, ACTIVE_NZ, FILE_NZ = 180, 148, 30, 31
VISCOSITY_SHA256 = (
    "fc142ce094a0f68d255ff79b04f8fe5d6634b33b0bc1c07b61870cede39499cb")
CITATIONS = {
    "read": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353",
    "exchange": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iom.f90:958-975",
    "mask": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:388-393",
    "fold_t": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:584-638",
    "fold_f": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lbcnfd.f90:722-746",
    "record": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/diawri.f90:639-641",
}


class GateError(RuntimeError):
    """A mechanically binding round-9 condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """NEMO's own ahmt/ahmf, both ranks, stitched into the global domain."""

    import netCDF4  # noqa: N813

    blocks_t, blocks_f = [], []
    for rank in (0, 1):
        path = root / f"output.init_{rank:04d}.nc"
        require(path.is_file(), f"missing recorded init stream: {path}")
        with netCDF4.Dataset(path, "r") as ds:
            ds.set_auto_maskandscale(False)
            for name, sink in (("ahmt", blocks_t), ("ahmf", blocks_f)):
                require(name in ds.variables,
                        f"{path.name}: no {name} stream")
                sink.append(np.asarray(ds.variables[name][0], dtype=np.float64))
    ahmt = np.concatenate(blocks_t, axis=2)
    ahmf = np.concatenate(blocks_f, axis=2)
    require(ahmt.shape == (FILE_NZ, GLOBAL_NY, GLOBAL_NX),
            f"recorded ahmt has shape {ahmt.shape}")
    require(ahmf.shape == (FILE_NZ, GLOBAL_NY, GLOBAL_NX),
            f"recorded ahmf has shape {ahmf.shape}")
    # (z, y, x) -> the card's (y, x, z)
    return np.moveaxis(ahmt, 0, -1), np.moveaxis(ahmf, 0, -1)


def read_mesh_masks(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """NEMO's own tmask and fmask, both ranks, on the card's axis order."""

    import netCDF4  # noqa: N813

    blocks_t, blocks_f = [], []
    for rank in (0, 1):
        path = root / f"mesh_mask_{rank:04d}.nc"
        require(path.is_file(), f"missing recorded mesh mask: {path}")
        with netCDF4.Dataset(path, "r") as ds:
            ds.set_auto_maskandscale(False)
            blocks_t.append(np.asarray(ds.variables["tmask"][0], dtype=np.float64))
            blocks_f.append(np.asarray(ds.variables["fmask"][0], dtype=np.float64))
    return (np.moveaxis(np.concatenate(blocks_t, axis=2), 0, -1),
            np.moveaxis(np.concatenate(blocks_f, axis=2), 0, -1))


def read_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import netCDF4  # noqa: N813

    with netCDF4.Dataset(path, "r") as ds:
        ds.set_auto_maskandscale(False)
        raw_t = np.asarray(ds.variables["ahmt_3d"][0], dtype=np.float64)
        raw_f = np.asarray(ds.variables["ahmf_3d"][0], dtype=np.float64)
    return np.moveaxis(raw_t, 0, -1), np.moveaxis(raw_f, 0, -1)


def score(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, object]:
    unequal = int(np.sum(candidate != oracle))
    row = {
        "cells": int(oracle.size),
        "unequal": unequal,
        "bit_identical": unequal == 0,
    }
    if unequal:
        delta = np.abs(candidate - oracle)
        flat = int(np.argmax(delta))
        row["max_abs_difference"] = repr(float(delta.flat[flat]))
        row["worst_cell"] = [int(v) for v in np.unravel_index(flat, oracle.shape)]
    return row


def fold_consistency(ahmt_file: np.ndarray, ahmf_file: np.ndarray) -> dict:
    """Is the input file already what the compiled T-pivot exchange would write?

    ``lbcnfd.f90:584-638`` (a T-point field under a T pivot) rewrites the RIGHT
    half of the last owned row from its mirrored left half; ``:722-746`` (an
    F-point field) rewrites the WHOLE last owned row from the row below at the
    reversed longitude.  If either already holds on the shipped file, omitting
    the exchange is exact -- and this gate refuses if it stops holding.
    """

    right = np.arange(GLOBAL_NX // 2 + 1, GLOBAL_NX)
    left = GLOBAL_NX - right
    t_row_unequal = int(np.sum(
        ahmt_file[-1, right] != ahmt_file[-1, left]))
    reversed_below = ahmf_file[-2, ::-1]
    f_row_unequal = int(np.sum(ahmf_file[-1] != reversed_below))
    result = {
        "t_point_rule": CITATIONS["fold_t"],
        "t_point_row_unequal": t_row_unequal,
        "t_point_row_cells": int(right.size * ahmt_file.shape[-1]),
        "f_point_rule": CITATIONS["fold_f"],
        "f_point_row_unequal": f_row_unequal,
        "f_point_row_cells": int(ahmf_file.shape[-1] * GLOBAL_NX),
        "exchange_is_identity_on_owned_cells": (
            t_row_unequal == 0 and f_row_unequal == 0),
    }
    require(
        result["exchange_is_identity_on_owned_cells"],
        "the eddy viscosity input file is NOT fold-consistent, so NEMO's "
        "lateral boundary exchange is not the identity on the owned domain "
        "and must be transcribed rather than omitted "
        f"(T row {t_row_unequal}, F row {f_row_unequal} unequal)",
    )
    return result


def run_gate(deck_root: Path, record_root: Path, *, plant: str | None = None):
    import jax.numpy as jnp  # noqa: F401  (card build needs jax configured)

    from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_orca2_ldf_dyn_coefficients,
        build_orca2_zps_card,
    )
    from legoesm.ocean.vertical import nemo_fe3mask_from_tmask

    viscosity_path = deck_root / "eddy_viscosity_3D.nc"
    require(viscosity_path.is_file(),
            f"missing ORCA2 eddy viscosity file: {viscosity_path}")
    digest = sha256(viscosity_path)
    require(digest == VISCOSITY_SHA256,
            f"eddy_viscosity_3D.nc digest moved: {digest}")

    card = build_orca2_zps_card(deck_root)
    z_coord = card.recipe.z_coord
    ahmt = np.asarray(z_coord.nemo_ldf_ahmt, dtype=np.float64)
    ahmf_vertex = np.asarray(z_coord.nemo_ldf_ahmf, dtype=np.float64)
    require(ahmt.shape == (GLOBAL_NY, GLOBAL_NX, ACTIVE_NZ),
            f"card ahmt has shape {ahmt.shape}")
    require(ahmf_vertex.shape == (GLOBAL_NY + 1, GLOBAL_NX + 1, ACTIVE_NZ),
            f"card ahmf has shape {ahmf_vertex.shape}")

    if plant == "ahmt":
        ahmt = ahmt.copy()
        ahmt[70, 40, 5] = np.nextafter(ahmt[70, 40, 5], np.inf)
    if plant == "ahmf":
        ahmf_vertex = ahmf_vertex.copy()
        ahmf_vertex[70, 40, 5] = np.nextafter(ahmf_vertex[70, 40, 5], np.inf)

    record_t, record_f = read_record(record_root)
    file_t, file_f = read_file(viscosity_path)
    fold = fold_consistency(file_t, file_f)

    # The card's F coefficient lives on legoESM's VERTEX layout; map it back to
    # NEMO's native F layout to compare with the record: vertex[j, i] is NEMO's
    # F point (j-1, (i-1) mod n_lon), so NEMO F (j, i) is vertex[j+1, i+1].
    ahmf_native = ahmf_vertex[1:, 1:GLOBAL_NX + 1]

    tmask = np.asarray(z_coord.is_active, dtype=np.float64)
    rows = {
        "ahmt_all_cells_both_ranks": score(ahmt, record_t[..., :ACTIVE_NZ]),
        "ahmf_all_cells_both_ranks": score(
            ahmf_native, record_f[..., :ACTIVE_NZ]),
        "ahmt_fold_row": score(ahmt[-1], record_t[-1, :, :ACTIVE_NZ]),
        "ahmf_fold_row": score(ahmf_native[-1], record_f[-1, :, :ACTIVE_NZ]),
    }

    # Controls.  Each must leave the comparison unequal, or the gate cannot
    # tell the transcription apart from a wrong one.
    controls = {
        "no_mask": score(file_t[..., :ACTIVE_NZ], record_t[..., :ACTIVE_NZ]),
        # the VERTEX-layout array read without the F index shift
        "no_f_index_shift": score(
            ahmf_vertex[1:, :GLOBAL_NX], record_f[..., :ACTIVE_NZ]),
        "f_read_as_t": score(ahmt, record_f[..., :ACTIVE_NZ]),
    }
    for name, row in controls.items():
        require(not row["bit_identical"],
                f"control {name!r} is vacuous: it did not change any cell")

    # The masks the transcription multiplies in are the card's OWN, so they
    # are a claim in their own right: compare them with NEMO's mesh mask over
    # every cell, not only where the coefficient happens to be non-zero.
    mesh_t, mesh_f = read_mesh_masks(record_root)
    mask_rows = {
        "tmask_vs_nemo_mesh_mask": score(tmask, mesh_t[..., :ACTIVE_NZ]),
        "fmask_vs_nemo_mesh_mask": score(
            np.asarray(card.recipe.z_coord.nemo_een_barotropic.fmask)[
                ..., :ACTIVE_NZ],
            mesh_f[..., :ACTIVE_NZ]),
    }
    for name, row in mask_rows.items():
        require(row["bit_identical"],
                f"{name}: the card's mask is not NEMO's "
                f"({row['unequal']} cells)")

    # Coverage NEMO carries and the card does not: level 31 is never masked
    # (ldfdyn.f90:388-393 stops at jpkm1), and the card has no such level.
    coverage = {
        "nemo_level_31_is_the_raw_file_value": {
            "ahmt": score(file_t[..., ACTIVE_NZ], record_t[..., ACTIVE_NZ]),
            "ahmf": score(file_f[..., ACTIVE_NZ], record_f[..., ACTIVE_NZ]),
        },
        "card_has_no_level_31": True,
    }

    # The F index map, proven against two pieces of code that both predate this
    # round: legoESM's own vertex mask, and NEMO's four-T-cell F-point product
    # (``nemo_fe3mask_from_tmask``).  Under the map this round uses the two
    # must agree cell for cell; a shifted map fails.  The FOLD ROW is scored
    # separately because the two disagree there for a reason that is about the
    # mask, not about this coefficient.
    surface = tmask[..., 0]
    vertex_mask = np.asarray(compute_vertex_mask(
        jnp.asarray(surface), grid=card.recipe.grid))
    nemo_f_product = np.asarray(nemo_fe3mask_from_tmask(
        jnp.asarray(surface), grid=card.recipe.grid))
    mapped = vertex_mask[1:, 1:GLOBAL_NX + 1]
    index_map = score(mapped[:-1], nemo_f_product[:-1])
    index_map_controls = {
        "column_shifted_east": score(
            np.roll(mapped, -1, axis=1)[:-1], nemo_f_product[:-1]),
        "column_unshifted": score(
            np.roll(mapped, 1, axis=1)[:-1], nemo_f_product[:-1]),
        "row_unshifted": score(
            vertex_mask[:-1, 1:GLOBAL_NX + 1][:-1], nemo_f_product[:-1]),
    }
    for name, row in index_map_controls.items():
        require(not row["bit_identical"],
                f"index-map control {name!r} is vacuous")
    fold_row_mask = score(mapped[-1], nemo_f_product[-1])

    # REPORTED, not asserted: what the OPERATOR then does with the coefficient.
    # NEMO stores ahmf already multiplied by fmask and dyn_ldf_lev reads it as
    # stored; legoESM's div-curl multiplies it again by its own per-level
    # vertex mask.  Where that mask is zero and NEMO's coefficient is not, the
    # two disagree -- a statement about the MASK, downstream of this round.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,  # noqa: F401  (import parity with the operator)
    )
    import jax
    cell3 = tmask * surface[..., np.newaxis]
    visc_vmask = np.asarray(jax.vmap(
        lambda m2: compute_vertex_mask(m2, grid=card.recipe.grid),
        in_axes=-1, out_axes=-1)(jnp.asarray(cell3)))
    visc_vmask = visc_vmask * vertex_mask[..., np.newaxis]
    downstream_mask = {
        "nemo_coefficient_zeroed_by_the_legoesm_vertex_mask": int(np.sum(
            (visc_vmask[1:, 1:GLOBAL_NX + 1] == 0.0)
            & (ahmf_native != 0.0))),
        "cells": int(ahmf_native.size),
        "note": (
            "NEMO's dyn_ldf_lev reads ahmf as ldf_dyn_init stored it; the "
            "legoESM operator masks it again with its own vertex mask"),
    }

    ahmf_south_row = {
        "card_value_is_zero": bool(np.all(ahmf_vertex[0] == 0.0)),
        "legoesm_vertex_mask_is_zero": bool(np.all(vertex_mask[0] == 0.0)),
    }
    require(ahmf_south_row["card_value_is_zero"]
            and ahmf_south_row["legoesm_vertex_mask_is_zero"],
            "the vertex row with no NEMO source is not inert")

    # The builder is the card's, not a second copy: call it directly and
    # require the card's carried arrays back.
    rebuilt_t, rebuilt_f = build_orca2_ldf_dyn_coefficients(
        viscosity_path,
        tmask,
        np.asarray(card.recipe.z_coord.nemo_een_barotropic.fmask)[
            ..., :ACTIVE_NZ],
    )
    single_implementation = (
        np.array_equal(rebuilt_t, np.asarray(z_coord.nemo_ldf_ahmt))
        and np.array_equal(rebuilt_f, np.asarray(z_coord.nemo_ldf_ahmf)))
    require(single_implementation or plant is not None,
            "the card's coefficient is not what its own builder returns")

    result = {
        "provenance": worktree_stamp(),
        "claim_label": "INDEPENDENT",
        "resolved_setting": "nn_ahm_ijk_t = -30 (run ocean.output:1184)",
        "citations": CITATIONS,
        "eddy_viscosity_file": {
            "resolved_path": str(viscosity_path.resolve()),
            "sha256": digest,
        },
        "record_root": str(record_root),
        "fold_consistency": fold,
        "rows": rows,
        "card_masks_vs_nemo_mesh_mask": mask_rows,
        "controls": controls,
        "coverage": coverage,
        "f_index_map_vs_nemo_f_product": index_map,
        "f_index_map_controls": index_map_controls,
        "fold_row_vertex_mask_vs_nemo_f_product": fold_row_mask,
        "ahmf_south_vertex_row": ahmf_south_row,
        "downstream_vertex_masking": downstream_mask,
        "one_implementation": single_implementation,
    }
    failed = [name for name, row in rows.items() if not row["bit_identical"]]
    require(index_map["bit_identical"],
            "legoESM's vertex mask is not NEMO's four-T-cell product under "
            "the F index map this round uses")
    result["status"] = "AT_BAR" if not failed else "DEBT"
    result["failed_rows"] = failed
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=("ahmt", "ahmf"))
    args = parser.parse_args()
    try:
        result = run_gate(args.deck_root, args.record_root, plant=args.plant)
    except (GateError, OSError, ValueError, KeyError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    if result["status"] != "AT_BAR":
        print(f"REFUSE: lateral viscosity coefficient is DEBT: "
              f"{result['failed_rows']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
