#!/usr/bin/env python
"""Round-35 GYRE ``tra_zdf`` matrix record: reader, calibration, and the
given-inputs gate for the implicit vertical TRACER solve.

WHY THIS EXISTS.  Round 34 localised GYRE's kt=2 T/S failure to one routine:
at kt=1 stage 3 the accumulator handed to ``tra_zdf`` is AT-BAR on both
tracers (T ``6.05e-16``, S ``5.79e-16``) while ``tra_zdf``'s OUTPUT is
``1.36e-12`` on T.  Every operator upstream of it in that stage is exonerated
by that one measurement, and ``tra_ldf`` twice over -- its own increment to
``ts(Krhs)`` is exactly ``0``.  What no record carried was the routine's own
tridiagonal: round 29 instrumented ``dyn_zdf``'s MOMENTUM matrix, not this one.

WHAT THE RECORD IS.  ``oracle_trazdf_matrix_kt00000001.bin``, magic
``NEMO_L2_TRAZD_1``, written by ``trazdf_round35.patch`` from ``tra_zdf``
AFTER the DRAKKAR clamp so the solved column is in it on both sides of that
clamp.  Self-describing: a 16-byte magic, sixteen header integers, then one
``(name[16], rank, n1, n2, n3, payload)`` group per array until EOF.

THE ARMS, and what each can decide.

* ``calibration`` -- rebuild ``zwt_mix``, ``zwi``/``zwd``/``zws``, the LU
  diagonal, the per-level right-hand side, the forward sweep and the solved
  column from the record's OWN operands, by ``trazdf.F90:172-174`` (the
  ``ah_wslp2`` arm), ``:204``, ``:217-223`` (the non-Aimp arm), ``:256-261``,
  ``:271-279`` and ``:281-287``, with the ``key_qco``/``key_vco_1d3d``
  substitutions of ``domzgr_substitute.h90:126`` and ``:131``.  Zero cells
  unequal, or NO number from this record may be quoted (Rule 1e).
* ``sweep`` -- legoESM's OWN ordered solve, the function object the
  trajectory runs, given NEMO's dumped matrix and NEMO's dumped RHS.
* ``assembly`` -- legoESM's OWN tracer assembly, the function the literal
  solve calls, given NEMO's dumped operands, scored PER DIAGONAL.
* ``rhs_content`` -- legoESM's own content-RHS builder given NEMO's operands.
  This arm exists because an independent claim review found that the first two
  arms, on their own, CANNOT see the right-hand side at all: the production
  solve takes it as an argument.
* ``named_deviations`` -- the three places legoESM's transcription is not
  NEMO's statement, each with the MEASURED condition that makes it inert on
  this card.  A gate that cannot see a deviation must say so rather than
  imply it checked.

THE BLIND SPOT, NAMED.  ``tracer_combine`` resolves to ``"concentration"`` on
this card, so ``ocean_model_latlon_cgrid.py:11943-11988`` leaves
``_nemo_tracer_content_rhs`` at ``None`` and the literal solve is fed
``T_solve_in * dz_cell`` at ``:10220-10221`` instead of NEMO's two-term
content RHS.  NO arm here drives that path -- it needs the model's own
intermediates -- so the gate PRINTS the resolved arm and reports the row
UNMEASURED with that reason.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp

MAGIC = "NEMO_L2_TRAZD_1"
BAR = 1.0e-15
CASE = "GYRE-zco"
HEADER_FIELDS = (
    "version", "kt", "kstg", "Kbb", "Kmm", "Krhs", "Kaa",
    "jpi", "jpj", "jpk", "jpkm1", "ntsi", "ntei", "ntsj", "ntej", "bits",
)
# Every array the writer emits, in stream order, so a SHORT record is a hard
# failure rather than a quietly missing arm (Rule 1: coverage is driven by
# what the oracle provides, not by what the reader happens to ask for).
EXPECTED_ARRAYS = (
    "ln_zdfddm", "ln_zad_Aimp", "ln_zdfmfc", "ln_traldf_msc", "l_ldfslp",
    "ln_SEOS", "a33_allocated", "rDt", "rn_b0", "jp_tem", "jp_sal",
    "T_Kbb_in", "S_Kbb_in", "T_Kmm_in", "S_Kmm_in", "T_Krhs_in", "S_Krhs_in",
    "zwt_mix", "zwi", "zwd", "zws", "zwt_lu",
    "rhs_T", "rhs_S", "fwd_T", "fwd_S",
    "sol_T_pre_clamp", "sol_S_pre_clamp",
    "sol_T_post_clamp", "sol_S_post_clamp",
    "avt", "avs", "ah_wslp2", "akz", "tmask",
    "e3t_Kbb", "e3t_Kmm", "e3t_Kaa", "e3w_Kmm", "e3t_0", "e3w_0",
    "r3t_Kbb", "r3t_Kmm", "r3t_Kaa",
)
# The branch this reader transcribes.  A record written by any other arm is
# REFUSED rather than scored against statements that did not run.
REQUIRED_ARM = {
    "ln_zdfddm": 0.0,        # trazdf.F90:159-160 -- one matrix, reused for S
    "ln_zad_Aimp": 0.0,      # :217-223 rather than :207-216
    "ln_zdfmfc": 0.0,        # neither :229-235 nor :265-269
    "ln_traldf_msc": 0.0,    # :172-174 (ah_wslp2) rather than :167-170 (akz)
    "l_ldfslp": 1.0,         # :166 -- the a33 fold IS in the matrix
}
PLANT_ARMS = ("operand", "matrix", "sweep", "assembly", "rhs", "a33", "clamp")


class RecordError(Exception):
    """A record this reader refuses; always reported as a VERDICT."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordError(message)


def read_trazdf_matrix(path: Path) -> dict:
    """Parse the self-describing record to EOF, refusing anything malformed.

    Nothing here knows the array list in advance; the list is only used
    AFTERWARDS, to refuse a short record.  Every refusal is a ``RecordError``
    and therefore a printed verdict -- a decoder traceback would leave a
    malformed record with no status at all, which is how a defeat hides.
    """
    raw = path.read_bytes()
    require(len(raw) >= 16 + 64, f"{path}: shorter than one header")
    try:
        magic = raw[:16].decode("ascii").rstrip()
    except UnicodeDecodeError:
        raise RecordError(f"{path}: magic is not ASCII")
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    header = dict(zip(HEADER_FIELDS, struct.unpack("=16i", raw[16:80])))
    require(header["version"] == 1, f"{path}: bad version {header['version']}")
    require(header["bits"] == 64, f"{path}: payload is not 64-bit")
    require(header["kstg"] == 3, f"{path}: not a stage-3 record")
    require(header["kt"] >= 1, f"{path}: bad kt {header['kt']}")
    for name in ("jpi", "jpj", "jpk", "jpkm1"):
        require(header[name] >= 1, f"{path}: bad {name} {header[name]}")
    require(header["jpkm1"] == header["jpk"] - 1,
            f"{path}: jpkm1 {header['jpkm1']} is not jpk-1")
    require(1 <= header["ntsi"] <= header["ntei"] <= header["jpi"],
            f"{path}: ntsi/ntei {header['ntsi']}/{header['ntei']} outside jpi")
    require(1 <= header["ntsj"] <= header["ntej"] <= header["jpj"],
            f"{path}: ntsj/ntej {header['ntsj']}/{header['ntej']} outside jpj")

    arrays: dict[str, np.ndarray | float] = {}
    order: list[str] = []
    off = 80
    while off < len(raw):
        require(off + 32 <= len(raw), f"{path}: truncated array header at {off}")
        try:
            name = raw[off:off + 16].decode("ascii").rstrip()
        except UnicodeDecodeError:
            raise RecordError(f"{path}: array name at {off} is not ASCII")
        require(name.isprintable() and name != "",
                f"{path}: array name at {off} is not a printable name")
        rank, n1, n2, n3 = struct.unpack("=4i", raw[off + 16:off + 32])
        off += 32
        require(rank in (0, 2, 3), f"{path}: array {name!r} has rank {rank}")
        require(min(n1, n2, n3) >= 1, f"{path}: array {name!r} has extent <= 0")
        count = n1 * n2 * n3
        end = off + 8 * count
        require(end <= len(raw), f"{path}: array {name!r} runs past EOF")
        values = np.frombuffer(raw, dtype="<f8", count=count, offset=off)
        off = end
        require(name not in arrays, f"{path}: array {name!r} written twice")
        if rank == 0:
            arrays[name] = float(values[0])
        elif rank == 2:
            require((n1, n2) == (header["jpi"], header["jpj"]),
                    f"{path}: rank-2 {name!r} is {n1}x{n2}, not the header's "
                    f"{header['jpi']}x{header['jpj']}")
            arrays[name] = values.reshape((n1, n2), order="F")
        else:
            require((n1, n2) == (header["jpi"], header["jpj"])
                    and n3 in (header["jpk"], header["jpkm1"]),
                    f"{path}: rank-3 {name!r} is {n1}x{n2}x{n3}, not the "
                    f"header's {header['jpi']}x{header['jpj']}x"
                    f"{header['jpk']} or ...x{header['jpkm1']}")
            arrays[name] = values.reshape((n1, n2, n3), order="F")
        order.append(name)
    require(off == len(raw), f"{path}: {len(raw) - off} trailing bytes")
    missing = [n for n in EXPECTED_ARRAYS if n not in arrays]
    require(not missing, f"{path}: record is short of {missing}")
    for name, want in REQUIRED_ARM.items():
        require(arrays[name] == want,
                f"{path}: written by an arm this reader does not transcribe: "
                f"{name} is {arrays[name]}, expected {want}")
    require(arrays["rDt"] > 0.0, f"{path}: rDt is {arrays['rDt']}")
    require(int(arrays["jp_tem"]) != int(arrays["jp_sal"]),
            f"{path}: jp_tem and jp_sal are the same index")
    for name in EXPECTED_ARRAYS:
        block = np.asarray(arrays[name], dtype=np.float64)
        require(bool(np.isfinite(block).all()),
                f"{path}: {name!r} carries a non-finite value")
    return {"header": header, "arrays": arrays, "order": order,
            "path": str(path)}


def _interior(header: dict) -> tuple[slice, slice]:
    """The i-k slice loop's own bounds, as the writer recorded them.

    ``DO_1Dj(0,0)`` runs ``jj = ntsj..ntej`` and ``DO_1Di(0,0)`` runs
    ``ji = ntsi..ntei`` (``do_loop_substitute.h90``), so the matrix buffers
    are only filled there; everything else is the writer's canonical zero.
    """
    return (slice(header["ntsi"] - 1, header["ntei"]),
            slice(header["ntsj"] - 1, header["ntej"]))


def _box(rec: dict, name: str, kmax: int | None = None) -> np.ndarray:
    isl, jsl = _interior(rec["header"])
    block = rec["arrays"][name]
    if block.ndim == 2:
        return block[isl, jsl]
    return block[isl, jsl, :] if kmax is None else block[isl, jsl, :kmax]


def bit_row(name: str, oracle, candidate, *, note: str | None = None) -> dict:
    """A row at the campaign's exact bar: value AND bits, both reported."""
    o = np.asarray(oracle, dtype=np.float64).ravel()
    c = np.asarray(candidate, dtype=np.float64).ravel()
    require(o.shape == c.shape, f"{name}: shape {o.shape} vs {c.shape}")
    # A comparison over zero cells would report absolute_max 0.0 and AT-BAR,
    # which is a vacuous pass rather than a measurement.
    require(o.size > 0, f"{name}: nothing to compare")
    require(bool(np.isfinite(o).all()), f"{name}: oracle side is not finite")
    require(bool(np.isfinite(c).all()), f"{name}: candidate side is not finite")
    diff = np.abs(o - c)
    absolute_max = float(diff.max()) if diff.size else 0.0
    scale = max(float(np.abs(o).max()) if o.size else 0.0, 1.0)
    unequal = o.view(np.uint64) != c.view(np.uint64)
    bit_unequal = int(np.count_nonzero(unequal))
    # SIGNED ZERO IS ITS OWN CLASS, reported, never folded into AT-BAR.
    # NEMO's zwi at the surface and its zws at the bottom row are
    # ``-p2dt * 0.0 / e3w`` = NEGATIVE zero (trazdf.F90:219-220 with
    # zwt(:,1) = 0 at :204), where legoESM writes a POSITIVE zero into the
    # same slots (implicit_solver.py, ``zero`` in nemo_tracer_tridiagonal).
    # The bits differ; every arithmetic result downstream does not, because
    # -0.0 + x == x exactly.  Counting these as AT-BAR would hide a real
    # difference; counting them as DEBT would claim a numerical one that is
    # not there.  So they get their own status and their own count.
    signed_zero = int(np.count_nonzero(
        unequal & (o == 0.0) & (c == 0.0))) if bit_unequal else 0
    if bit_unequal == 0:
        status = "AT-BAR"
    elif signed_zero == bit_unequal:
        status = "AT-BAR-SIGNED-ZERO"
    elif absolute_max / scale <= BAR:
        status = "VALUE-AT-BAR"
    else:
        status = "DEBT"
    normalized = absolute_max / scale
    row = {"name": name, "n": int(o.size), "bit_unequal": bit_unequal,
           "signed_zero_only": signed_zero,
           "absolute_max": absolute_max, "normalized_max_abs": normalized,
           "reference_max_abs": scale, "bar": BAR, "status": status}
    if note:
        row["note"] = note
    return row


PASSING = ("AT-BAR", "AT-BAR-SIGNED-ZERO")


# --------------------------------------------------------------------------
# ARM 1 -- NEMO rebuilt from NEMO's own operands (the reader's calibration)
# --------------------------------------------------------------------------
def nemo_rebuild(rec: dict) -> dict[str, np.ndarray]:
    """Every dumped array, recomputed from the record's own operands.

    Literal transcription of the arm ``REQUIRED_ARM`` pins.  ``zwt_mix`` is
    ``trazdf.F90:172-174`` plus ``:204``; the diagonals are ``:217-223``; the
    LU sweep is ``:256-261``; the right-hand side and forward sweep are
    ``:271-279``; the back substitution is ``:281-287``.
    """
    h = rec["header"]
    a = rec["arrays"]
    jpk, jpkm1 = h["jpk"], h["jpkm1"]
    p2dt = a["rDt"]

    avt, a33 = _box(rec, "avt"), _box(rec, "ah_wslp2")
    e3w = _box(rec, "e3w_Kmm")
    e3t_kaa, e3t_kbb, e3t_kmm = (_box(rec, "e3t_Kaa"), _box(rec, "e3t_Kbb"),
                                 _box(rec, "e3t_Kmm"))
    tmask = _box(rec, "tmask")

    zwt_mix = np.zeros_like(avt)
    zwt_mix[:, :, 1:jpk] = avt[:, :, 1:jpk] + a33[:, :, 1:jpk]
    zwt_mix[:, :, 0] = 0.0

    zwi = np.zeros(avt.shape[:2] + (jpkm1,))
    zws = np.zeros_like(zwi)
    zwi[:, :, :] = -p2dt * zwt_mix[:, :, 0:jpkm1] / e3w[:, :, 0:jpkm1]
    zws[:, :, :] = -p2dt * zwt_mix[:, :, 1:jpkm1 + 1] / e3w[:, :, 1:jpkm1 + 1]
    zwd = e3t_kaa[:, :, 0:jpkm1] - (zwi + zws)

    zwt_lu = zwt_mix.copy()
    zwt_lu[:, :, 0] = zwd[:, :, 0]
    for k in range(1, jpkm1):
        zwt_lu[:, :, k] = (zwd[:, :, k]
                           - zwi[:, :, k] * zws[:, :, k - 1] / zwt_lu[:, :, k - 1])

    out = {"zwt_mix": zwt_mix, "zwi": zwi, "zwd": zwd, "zws": zws,
           "zwt_lu": zwt_lu}
    for tag, jn in (("T", "jp_tem"), ("S", "jp_sal")):
        t_bb, t_rhs = _box(rec, f"{tag}_Kbb_in"), _box(rec, f"{tag}_Krhs_in")
        rhs = np.zeros_like(zwt_mix)
        rhs[:, :, 0] = e3t_kbb[:, :, 0] * t_bb[:, :, 0] \
            + p2dt * e3t_kmm[:, :, 0] * t_rhs[:, :, 0]
        for k in range(1, jpkm1):
            rhs[:, :, k] = e3t_kbb[:, :, k] * t_bb[:, :, k] \
                + p2dt * e3t_kmm[:, :, k] * t_rhs[:, :, k]
        fwd = rhs.copy()
        for k in range(1, jpkm1):
            fwd[:, :, k] = (rhs[:, :, k]
                            - zwi[:, :, k] / zwt_lu[:, :, k - 1] * fwd[:, :, k - 1])
        sol = fwd.copy()
        sol[:, :, jpkm1 - 1] = (fwd[:, :, jpkm1 - 1] / zwt_lu[:, :, jpkm1 - 1]
                                * tmask[:, :, jpkm1 - 1])
        for k in range(jpkm1 - 2, -1, -1):
            sol[:, :, k] = ((fwd[:, :, k] - zws[:, :, k] * sol[:, :, k + 1])
                            / zwt_lu[:, :, k] * tmask[:, :, k])
        out[f"rhs_{tag}"] = rhs
        out[f"fwd_{tag}"] = fwd
        out[f"sol_{tag}"] = sol
    # the two thickness substitutions themselves, not only their result
    out["e3t_Kaa_sub"] = (_box(rec, "e3t_0")
                          * (1.0 + _box(rec, "r3t_Kaa")[:, :, None] * tmask))
    out["e3w_Kmm_sub"] = (_box(rec, "e3w_0")
                          * (1.0 + _box(rec, "r3t_Kmm")[:, :, None]))
    return out


# --------------------------------------------------------------------------
# ARMS 2-4 -- legoESM's own production functions, given NEMO's own inputs
# --------------------------------------------------------------------------
def _lego_operands(rec: dict):
    """NEMO's own operands, mapped onto legoESM's argument order.

    legoESM cell ``k`` is NEMO's ``jk = k+1``; legoESM interior face ``f`` is
    the interface between cells ``f`` and ``f+1``, which is NEMO's
    ``jk = f+2`` -- so the face arrays are the record's own ``jk >= 2`` slice.
    """
    h = rec["header"]
    jpkm1 = h["jpkm1"]
    zwt_mix = np.zeros_like(_box(rec, "avt"))
    zwt_mix[:, :, 1:] = _box(rec, "avt")[:, :, 1:] + _box(rec, "ah_wslp2")[:, :, 1:]
    return {
        "K": zwt_mix[:, :, 1:jpkm1],
        "e3t_after": _box(rec, "e3t_Kaa", jpkm1),
        "e3w_now": _box(rec, "e3w_Kmm")[:, :, 1:jpkm1],
        "dt": rec["arrays"]["rDt"],
        "wet": _box(rec, "tmask", jpkm1),
        "nlev": jpkm1,
    }


def lego_assembly(rec: dict) -> dict[str, np.ndarray]:
    """legoESM's own tracer assembly -- the function the literal solve calls."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import nemo_tracer_tridiagonal
    op = _lego_operands(rec)
    lower, diagonal, upper = nemo_tracer_tridiagonal(
        jnp.asarray(op["K"]), jnp.asarray(op["e3t_after"]),
        jnp.asarray(op["e3w_now"]), float(op["dt"]),
        jnp.asarray(op["wet"]) > 0.0)
    return {"zwi": np.asarray(lower), "zwd": np.asarray(diagonal),
            "zws": np.asarray(upper)}


def lego_sweep(rec: dict) -> dict[str, np.ndarray]:
    """legoESM's own ordered solve, on NEMO's dumped matrix and dumped RHS.

    Multiplied by ``tmask`` because that is what
    ``implicit_vertical_diffusion_nemo_tracer_pair`` does with the result;
    the comparison is against NEMO's PRE-clamp column.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import (
        nemo_ordered_tridiagonal_solve)
    h = rec["header"]
    jpkm1 = h["jpkm1"]
    lower = jnp.asarray(_box(rec, "zwi"))
    diagonal = jnp.asarray(_box(rec, "zwd"))
    upper = jnp.asarray(_box(rec, "zws"))
    wet = np.asarray(_box(rec, "tmask", jpkm1))
    out = {}
    for tag in ("T", "S"):
        rhs = jnp.asarray(_box(rec, f"rhs_{tag}", jpkm1))
        out[tag] = np.asarray(
            nemo_ordered_tridiagonal_solve(lower, diagonal, upper, rhs)) * wet
    return out


def lego_rhs_content(rec: dict) -> dict[str, np.ndarray]:
    """legoESM's own content-RHS builder, given NEMO's operands.

    ``thickness_weighted_tracer_content`` groups the update as
    ``h_bef*T_bb + (h_aft*T_expl - h_now*T_now) + h_now*d_diss`` where NEMO
    writes ``e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*T(Krhs)`` (``trazdf.F90:272``,
    ``:276-277``).  Feeding NEMO's own operands into the equivalent slots --
    ``t_now`` and ``d_diss`` zero, ``t_expl`` the pre-multiplied
    ``p2dt*T(Krhs)`` -- leaves exactly ONE difference between the two, the
    ASSOCIATION of the triple product.  So this arm answers a question round
    36 needs answered before it can flip ``tracer_combine``: would the content
    form reproduce NEMO's right-hand side bit for bit given NEMO's inputs.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        thickness_weighted_tracer_content)
    h = rec["header"]
    jpkm1 = h["jpkm1"]
    p2dt = rec["arrays"]["rDt"]
    e3t_kbb = _box(rec, "e3t_Kbb", jpkm1)
    e3t_kmm = _box(rec, "e3t_Kmm", jpkm1)
    zero = np.zeros_like(e3t_kbb)
    out = {}
    for tag in ("T", "S"):
        out[tag] = np.asarray(thickness_weighted_tracer_content(
            t_before=_box(rec, f"{tag}_Kbb_in", jpkm1),
            t_now=zero, t_expl=p2dt * _box(rec, f"{tag}_Krhs_in", jpkm1),
            d_diss=zero, h_before=e3t_kbb, h_now=e3t_kmm, h_after=e3t_kmm))
    return out


# --------------------------------------------------------------------------
# The named deviations, each with the measured condition that makes it inert
# --------------------------------------------------------------------------
def named_deviations(rec: dict) -> list[dict]:
    """Where legoESM's transcription is not NEMO's statement, and why it holds.

    A gate that silently passes over a real difference has not checked it; a
    gate that names the difference AND measures the condition under which it
    cannot bite has.  All three below were raised by an independent claim
    review of this round's preregistration.
    """
    h = rec["header"]
    jpkm1, jpk = h["jpkm1"], h["jpk"]
    tmask = _box(rec, "tmask")
    zwt_mix_bottom = (_box(rec, "avt")[:, :, jpkm1]
                      + _box(rec, "ah_wslp2")[:, :, jpkm1])
    dry_in_box = int(np.count_nonzero(tmask[:, :, :jpkm1] == 0.0))
    return [
        {"deviation": "legoESM multiplies each face coefficient by "
                      "interface_wet and replaces each DRY diagonal by 1.0; "
                      "NEMO does neither and leaves its dry diagonal at e3t "
                      "(trazdf.F90:219-221), relying on avt being masked",
         "inert_when": "the scored box has no dry cell",
         "measured": {"dry_cells_in_scored_box": dry_in_box,
                      "cells": int(tmask[:, :, :jpkm1].size)},
         "inert_here": dry_in_box == 0},
        {"deviation": "NEMO's back substitution multiplies by tmask at EVERY "
                      "level before the next row reads it (trazdf.F90:282, "
                      ":286); legoESM masks ONCE after the sweep",
         "inert_when": "the scored box has no dry cell above a wet one",
         "measured": {"dry_cells_in_scored_box": dry_in_box},
         "inert_here": dry_in_box == 0},
        {"deviation": "NEMO's bottom row takes zws(jpkm1) = "
                      "-p2dt*zwt(jpk)/e3w(jpk) (trazdf.F90:220); legoESM's "
                      "upper is an EXACT zero there",
         "inert_when": "the matrix diffusivity at jk = jpk is exactly zero",
         "measured": {"max_abs_zwt_mix_at_jpk":
                      float(np.abs(zwt_mix_bottom).max()),
                      "level_jk": jpk},
         "inert_here": float(np.abs(zwt_mix_bottom).max()) == 0.0},
        {"deviation": "NEMO's zwi at the surface row and its zws at the "
                      "bottom row are NEGATIVE zero -- -p2dt*0.0/e3w with "
                      "zwt(:,1)=0 (trazdf.F90:204, :219-220) -- where "
                      "legoESM writes a POSITIVE zero into both slots",
         "inert_when": "always, arithmetically: -0.0 + x == x exactly, and "
                       "neither slot is ever divided by",
         "measured": {"reported_as": "AT-BAR-SIGNED-ZERO rows, with their "
                                     "own signed_zero_only count"},
         "inert_here": True},
    ]


def resolved_rhs_arm() -> dict:
    """Rule 10: PRINT which right-hand side the card actually selects.

    Instantiating the card is the only way to know; the docstring at
    ``ocean_model_latlon_cgrid.py:2179-2182`` says the content builder exists
    so the literal trazdf matrix can consume NEMO's content RHS, and whether
    it does is a per-card resolution, not a property of the code.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    rows = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        cfg = build_nemo_testcase_card(case).recipe.model_config
        rows[case] = {
            "tracer_combine": getattr(cfg, "tracer_combine", "<<absent>>"),
            "zdf_implicit_solver_evaluation": getattr(
                cfg, "zdf_implicit_solver_evaluation", "<<absent>>"),
        }
    return rows


def run(record: Path, *, plant: str | None = None,
        with_card: bool = True) -> dict:
    require(record.exists(), f"missing {record}")
    require(plant is None or plant in PLANT_ARMS,
            f"unknown plant {plant!r}; expected one of {PLANT_ARMS}")
    rec = read_trazdf_matrix(record)
    h = rec["header"]
    a = rec["arrays"]
    jpkm1 = h["jpkm1"]
    isl, jsl = _interior(h)

    def _perturb(name: str, index) -> None:
        block = np.array(a[name], copy=True)
        block[index] = np.nextafter(block[index], np.inf)
        a[name] = block

    if plant == "operand":       # every arm reads the matrix through avt
        _perturb("avt", (isl, jsl, 1))
    elif plant == "matrix":      # the dumped diagonal itself
        _perturb("zwd", (isl, jsl, 1))
    elif plant == "sweep":       # the dumped RHS the sweep consumes
        _perturb("rhs_T", (isl, jsl, 1))
    elif plant == "assembly":    # the thickness the assembly divides by
        _perturb("e3w_Kmm", (isl, jsl, 2))
    elif plant == "rhs":         # the before-tracer the content form reads
        _perturb("T_Kbb_in", (isl, jsl, 0))
    elif plant == "a33":         # the fold PR5 predicts is exactly zero
        _perturb("ah_wslp2", (isl, jsl, 3))
    elif plant == "clamp":       # a cell the clamp would have had to move
        block = np.array(a["sol_S_post_clamp"], copy=True)
        block[isl, jsl, 0] = np.nextafter(block[isl, jsl, 0], np.inf)
        a["sol_S_post_clamp"] = block

    # The row prefix comes from the RECORD's own kt, never a hardcoded "kt1":
    # a label that names a step the record is not from is a lie the reader
    # would carry into the receipt.
    step = f"{CASE}.kt{h['kt']}.stage{h['kstg']}.trazdf"

    rows: list[dict] = []
    rebuilt = nemo_rebuild(rec)
    for name in ("zwt_mix", "zwi", "zwd", "zws", "zwt_lu"):
        kmax = jpkm1 if name in ("zwi", "zwd", "zws") else None
        rows.append(bit_row(f"{step}.calibration.{name}",
                            _box(rec, name, kmax), rebuilt[name]))
    for tag in ("T", "S"):
        for stem, dumped in (("rhs", f"rhs_{tag}"), ("fwd", f"fwd_{tag}"),
                             ("sol", f"sol_{tag}_pre_clamp")):
            rows.append(bit_row(
                f"{step}.calibration.{stem}_{tag}",
                _box(rec, dumped, jpkm1), rebuilt[f"{stem}_{tag}"][:, :, :jpkm1]))
    rows.append(bit_row(f"{step}.calibration.e3t_Kaa_sub",
                        _box(rec, "e3t_Kaa"), rebuilt["e3t_Kaa_sub"],
                        note="domzgr_substitute.h90:126 -- WITH tmask"))
    rows.append(bit_row(f"{step}.calibration.e3w_Kmm_sub",
                        _box(rec, "e3w_Kmm"), rebuilt["e3w_Kmm_sub"],
                        note="domzgr_substitute.h90:131 -- NO tmask, r3t at "
                             "the T point, e3w_1d reference ladder"))
    calibration = [r for r in rows]
    calibrated = all(r["status"] in PASSING for r in calibration)

    lego_rows: list[dict] = []
    assembled = lego_assembly(rec)
    for name in ("zwi", "zwd", "zws"):
        lego_rows.append(bit_row(
            f"{step}.assembly.{name}",
            _box(rec, name), assembled[name]))
    swept = lego_sweep(rec)
    for tag in ("T", "S"):
        lego_rows.append(bit_row(
            f"{step}.sweep.{tag}",
            _box(rec, f"sol_{tag}_pre_clamp", jpkm1), swept[tag]))
    content = lego_rhs_content(rec)
    for tag in ("T", "S"):
        lego_rows.append(bit_row(
            f"{step}.rhs_content.{tag}",
            _box(rec, f"rhs_{tag}", jpkm1), content[tag],
            note="the content form given NEMO's operands; the ONLY residual "
                 "it can carry is the association of the triple product"))

    a33_max = float(np.abs(_box(rec, "ah_wslp2")).max())
    clamp_rows = []
    for tag in ("T", "S"):
        clamp_rows.append(bit_row(
            f"{step}.clamp.{tag}",
            _box(rec, f"sol_{tag}_pre_clamp"),
            _box(rec, f"sol_{tag}_post_clamp"),
            note="trazdf.F90:89-91 runs over ALL jk including jpk, which "
                 "tra_zdf_imp never assigns, so the whole column is scored"))

    condition_rows = [
        {"name": f"{step}.a33_fold_inert",
         "claim": "ah_wslp2 is exactly zero on every cell of the scored box",
         "max_abs": a33_max, "status": "AT-BAR" if a33_max == 0.0 else "DEBT"},
    ]
    deviations = named_deviations(rec)

    lego_at_bar = all(r["status"] in PASSING for r in lego_rows)
    clamp_at_bar = all(r["status"] in PASSING for r in clamp_rows)
    condition_at_bar = all(r["status"] in PASSING for r in condition_rows)
    status = ("AT-BAR" if (calibrated and lego_at_bar and clamp_at_bar
                           and condition_at_bar) else "DEBT")

    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round35-trazdf-matrix-v1",
        "case": CASE,
        "bar": BAR,
        "record": str(record),
        "header": h,
        "arrays_read": rec["order"],
        "planted": plant,
        "calibration_rows": calibration,
        "given_inputs_rows": lego_rows,
        "clamp_rows": clamp_rows,
        "condition_rows": condition_rows,
        "named_deviations": deviations,
        "status": status,
        "calibrated": calibrated,
        "rhs_blind_spot": {
            "status": "UNMEASURED",
            "reason": (
                "no arm here drives legoESM's OWN right-hand side as the card "
                "builds it: the production solve takes the RHS as an argument "
                "(implicit_solver.py:417-424) and this card resolves "
                "tracer_combine='concentration', so "
                "ocean_model_latlon_cgrid.py:11943-11988 leaves the content "
                "RHS at None and :10220-10221 feeds the literal matrix "
                "T_solve_in*dz_cell instead of NEMO's two-term content form.  "
                "Closing it needs the model's own intermediates, i.e. the "
                "pre_implicit_tracer_content_override hook, on a run."),
            "rows": [f"{step}.model_rhs.{f}"
                     for f in ("T", "S")],
        },
    }
    if with_card:
        report["resolved_rhs_arm"] = resolved_rhs_arm()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANT_ARMS, default=None,
                        help="perturb one operand by one ulp; the gate must "
                             "go red and this process must exit non-zero")
    parser.add_argument("--no-card", action="store_true",
                        help="skip the card instantiation (Rule-10 print)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.record, plant=args.plant,
                     with_card=not args.no_card)
    except RecordError as error:
        print(f"FAIL: {error}")
        return 2
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    for group in ("calibration_rows", "given_inputs_rows", "clamp_rows"):
        for row in report[group]:
            print(f"{row['status']:<12} {row['name']:<58} "
                  f"bit_unequal {row['bit_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.17g}")
    for row in report["condition_rows"]:
        print(f"{row['status']:<12} {row['name']:<58} "
              f"max {row['max_abs']:.17g}")
    for row in report["named_deviations"]:
        print(f"{'INERT' if row['inert_here'] else 'LIVE':<12} "
              f"{row['deviation'][:100]}")
    print(f"STATUS {report['status']}")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())
