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
this card, so ``ocean_model_latlon_cgrid.py:11954-11999`` leaves
``_nemo_tracer_content_rhs`` at ``None`` and the literal solve is fed
``T_solve_in * dz_cell`` at ``:10220-10221`` instead of NEMO's two-term
content RHS.  NO arm here drives that path -- it needs the model's own
intermediates -- so the gate PRINTS the resolved arm and reports the row
UNMEASURED with that reason.
"""

from __future__ import annotations

import argparse
import hashlib
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
# The expected RANK and DEPTH of every array, so a record whose extents are
# structurally plausible but WRONG gets a verdict instead of a broadcast
# traceback three functions later.  "k" is the full jpk ladder, "km1" the
# jpkm1 matrix rows, "2d" a horizontal field, "s" a scalar.
ARRAY_SHAPE = {
    **{n: "s" for n in ("ln_zdfddm", "ln_zad_Aimp", "ln_zdfmfc",
                        "ln_traldf_msc", "l_ldfslp", "ln_SEOS",
                        "a33_allocated", "rDt", "rn_b0", "jp_tem", "jp_sal")},
    **{n: "k" for n in (
        "T_Kbb_in", "S_Kbb_in", "T_Kmm_in", "S_Kmm_in", "T_Krhs_in",
        "S_Krhs_in", "zwt_mix", "zwt_lu", "rhs_T", "rhs_S", "fwd_T", "fwd_S",
        "sol_T_pre_clamp", "sol_S_pre_clamp", "sol_T_post_clamp",
        "sol_S_post_clamp", "avt", "avs", "ah_wslp2", "akz", "tmask",
        "e3t_Kbb", "e3t_Kmm", "e3t_Kaa", "e3w_Kmm", "e3t_0", "e3w_0")},
    **{n: "km1" for n in ("zwi", "zwd", "zws")},
    **{n: "2d" for n in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")},
}
# Arrays the record carries that NO arm consumes.  Declared rather than
# silent: Rule 1 says coverage is driven by what the oracle provides, so an
# unused array is a disposition, not an omission.
UNCONSUMED = {
    "avs": "ln_zdfddm is F, so the matrix is built from avt alone; avs is "
           "carried to PROVE it never enters",
    "akz": "ln_traldf_msc is F, so the akz arm at trazdf.F90:167-170 does not "
           "run; carried for the same reason",
    "T_Kmm_in": "the RHS reads Kbb and Krhs; Kmm enters only through e3t_Kmm",
    "S_Kmm_in": "as T_Kmm_in",
    "e3t_0": "consumed by the substitution row, not by the matrix",
    "e3w_0": "as e3t_0",
    "r3t_Kbb": "the Kbb substitution is not rebuilt; e3t_Kbb is dumped whole",
    "rn_b0": "read by the clamp-ran consistency check, not by an arm",
}

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
# THE DOMAIN THIS GATE SCORES, PINNED.  An independent attack on round 36
# cropped the record to a 5x5x31 domain, kept the halo symmetric at 2, and
# got exit 0 / STATUS AT-BAR out of a run that scored THIRTY cells instead of
# 21120 -- with all seven plants still reporting landed=True, because a plant
# lands inside whatever box the header declares.  The symmetric-halo check
# below cannot see that: it constrains the halo, never the domain, and jpi /
# jpj / jpk all come from the record's own header.  So the domain is pinned
# to the one this card runs, from the run's own resolved namelist output
# (round35_oracle_trazdf_matrix/ocean.output:36 jpkglo=31, :63 jpiglo=36,
# :64 jpjglo=26), and a record from any other geometry is REFUSED rather
# than scored on a fraction of its cells.
EXPECTED_DOMAIN = (36, 26, 31)

# THE TWO ARRAYS WHOSE DECLARED EXTENT IS NOT THEIR WRITTEN EXTENT.
#
# ``zdf_oce.f90:85-86`` (this round's own compiled ppsrc under
# ``cfgs/GYRE_OMIP_L2_P3_SM_R35TRAZDF/BLD/ppsrc/nemo``) allocates
#
#     avs(Nis0-(0):Nie0+(0), Njs0-(0):Nje0+(0), jpk)
#     avt(Nis0-(0):Nie0+(0), Njs0-(0):Nje0+(0), jpk)
#
# -- the INTERIOR box, not ``(jpi,jpj,jpk)``.  ``avm`` on the same line IS
# full-domain, which is what makes the mistake easy: the round-35 instrument
# writes ``WRITE(il2_unit) avt`` after a header declaring ``jpi, jpj, jpk``
# (``trazdf.f90:246-249``), so each of those two payloads is
# ``(ntei-ntsi+1)*(ntej-ntsj+1)*jpk`` values while its header claims
# ``jpi*jpj*jpk``.  Reading on the header alone lands the next array header
# 57536 bytes early and the record decodes as garbage from ``avs`` onward.
#
# The payload itself is COMPLETE over the interior, and the interior is
# exactly the box this gate scores, so the record is mislabelled rather than
# short.  The salvage below is therefore permitted -- but it is PROVEN, never
# assumed: the declared extent is used unless it fails to leave a known array
# name at the next header, and the tile extent is accepted only when it DOES.
# Its arithmetic is checked a second time by the calibration arm, which
# rebuilds NEMO's own ``zwt_mix`` from the embedded ``avt``; a misaligned
# embedding cannot reproduce it.
TILE_SHAPED = {
    "avt": "zdf_oce.f90:86 allocates avt over Nis0:Nie0 x Njs0:Nje0 x jpk",
    "avs": "zdf_oce.f90:85 allocates avs over the same interior box",
}


class RecordError(Exception):
    """A record this reader refuses; always reported as a VERDICT."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordError(message)


_KNOWN_NAMES = frozenset(n.ljust(16).encode("ascii") for n in EXPECTED_ARRAYS)


def _starts_a_known_array(raw: bytes, off: int, *, eof_counts: bool) -> bool:
    """Do the 16 bytes at ``off`` begin an array this record may carry?

    This is what turns the tile-extent salvage into a PROOF rather than a
    guess -- the stream itself says where the payload ended.

    ``eof_counts`` is the asymmetry that keeps it a proof.  For the array's
    OWN declared extent, running exactly to EOF is a valid ending: the last
    array is followed by nothing.  For the SHORTER tile extent it is not,
    because a record whose final array was truncated at the tile boundary
    would then "prove" a salvage that is really data loss.  So a tile extent
    is accepted only when a real array header follows it.
    """
    if off == len(raw):
        return eof_counts
    if off + 32 > len(raw):
        return False
    return raw[off:off + 16] in _KNOWN_NAMES


def read_trazdf_matrix(path: Path, *, expect_kt: int = 1,
                       truncate: bool = False) -> dict:
    """Parse the self-describing record to EOF, refusing anything malformed.

    Nothing here knows the array list in advance; the list is only used
    AFTERWARDS, to refuse a short record.  Every refusal is a ``RecordError``
    and therefore a printed verdict -- a decoder traceback would leave a
    malformed record with no status at all, which is how a defeat hides.

    ``expect_kt`` is the step this caller intends to score, and the record's
    own ``kt`` must equal it.  It defaults to 1 so every existing caller keeps
    the round-35 refusal unchanged; the round-38 acquisition writes a SECOND
    record at ``kt = nit000 + 1``, and reading it needs the step named at the
    call site rather than the guard removed.  A caller that passes the wrong
    ``expect_kt`` is refused exactly as a wrong record is: the guard moves, it
    does not weaken.
    """
    if expect_kt < 1:
        raise RecordError(f"expect_kt must be a positive step, got {expect_kt}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if truncate:
        raw = raw[:-8]
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
    for name in ("jpi", "jpj", "jpk", "jpkm1"):
        require(header[name] >= 1, f"{path}: bad {name} {header[name]}")
    require(header["jpkm1"] == header["jpk"] - 1,
            f"{path}: jpkm1 {header['jpkm1']} is not jpk-1")
    require(1 <= header["ntsi"] <= header["ntei"] <= header["jpi"],
            f"{path}: ntsi/ntei {header['ntsi']}/{header['ntei']} outside jpi")
    require(1 <= header["ntsj"] <= header["ntej"] <= header["jpj"],
            f"{path}: ntsj/ntej {header['ntsj']}/{header['ntej']} outside jpj")
    # THE SCORED BOX IS A CHECKED QUANTITY, NOT A HEADER CLAIM.  A shrunken
    # box scores a handful of cells and reports the same green verdict, with
    # every plant still red -- so the controls cannot catch it and every n in
    # the report becomes un-auditable.  NEMO's own loop bounds are the whole
    # computed domain minus a SYMMETRIC halo, so require exactly that, and
    # report the width and the cell count next to the verdict.
    halo_i = header["ntsi"] - 1
    halo_j = header["ntsj"] - 1
    require(halo_i == header["jpi"] - header["ntei"]
            and halo_j == header["jpj"] - header["ntej"]
            and halo_i == halo_j and halo_i >= 1,
            f"{path}: the scored box "
            f"[{header['ntsi']},{header['ntei']}]x"
            f"[{header['ntsj']},{header['ntej']}] on a "
            f"{header['jpi']}x{header['jpj']} domain is not the whole domain "
            f"minus one symmetric halo; a sub-box would score a fraction of "
            f"the cells and report the same verdict")
    require((header["jpi"], header["jpj"], header["jpk"]) == EXPECTED_DOMAIN,
            f"{path}: domain "
            f"{(header['jpi'], header['jpj'], header['jpk'])} is not "
            f"{CASE}'s {EXPECTED_DOMAIN}; a smaller domain passes the "
            "symmetric-halo check above and then scores a handful of cells "
            "while reporting the same verdict")
    header["halo"] = halo_i
    # A record from a step other than the one the CALLER named cannot be
    # scored against that caller's claim, whatever its rows are named.
    require(header["kt"] == expect_kt,
            f"{path}: kt is {header['kt']}, not {expect_kt}; this reader was "
            f"asked for step {expect_kt}")

    arrays: dict[str, np.ndarray | float] = {}
    order: list[str] = []
    salvaged: dict[str, dict] = {}
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
        tile = None
        if name in TILE_SHAPED and rank == 3:
            # The salvage is inside the SAME refusals every other array gets:
            # the duplicate-name check and the ARRAY_SHAPE rank/depth check
            # below both run first, so a salvaged array cannot skip them.
            require(name not in arrays, f"{path}: array {name!r} written twice")
            require((n1, n2, n3) == (header["jpi"], header["jpj"],
                                     header["jpk"]),
                    f"{path}: {name!r} declares {(n1, n2, n3)}, not the "
                    f"header's full-domain shape; the interior-extent "
                    "recovery is defined against that declaration alone")
            declared_ok = _starts_a_known_array(raw, off + 8 * count,
                                                eof_counts=True)
            ni = header["ntei"] - header["ntsi"] + 1
            nj = header["ntej"] - header["ntsj"] + 1
            tile_count = ni * nj * n3
            tile_ok = (tile_count < count
                       and _starts_a_known_array(raw, off + 8 * tile_count,
                                                 eof_counts=False))
            require(declared_ok or tile_ok,
                    f"{path}: array {name!r} is followed by neither a known "
                    f"array header at its declared extent {(n1, n2, n3)} nor "
                    f"one at NEMO's interior extent {(ni, nj, n3)}")
            if not declared_ok:
                tile, count = (ni, nj, n3), tile_count
        end = off + 8 * count
        require(end <= len(raw), f"{path}: array {name!r} runs past EOF")
        values = np.frombuffer(raw, dtype="<f8", count=count, offset=off)
        off = end
        if tile is not None:
            # embed the interior payload where NEMO's indices put it, so every
            # arm downstream sees ONE shape.  The halo ring is the READER's
            # zero -- no arm reads it, and _box() never reaches it.
            block = np.zeros((header["jpi"], header["jpj"], n3))
            block[header["ntsi"] - 1:header["ntei"],
                  header["ntsj"] - 1:header["ntej"], :] = \
                values.reshape(tile, order="F")
            arrays[name] = block
            order.append(name)
            salvaged[name] = {"declared": [n1, n2, n3], "written": list(tile),
                              "reason": TILE_SHAPED[name]}
            continue
        require(name not in arrays, f"{path}: array {name!r} written twice")
        expected = ARRAY_SHAPE.get(name)
        require(expected is None
                or rank == {"s": 0, "2d": 2, "k": 3, "km1": 3}[expected],
                f"{path}: {name!r} has rank {rank}, expected the "
                f"{expected!r} shape")
        if rank == 0:
            arrays[name] = float(values[0])
        elif rank == 2:
            require((n1, n2) == (header["jpi"], header["jpj"]),
                    f"{path}: rank-2 {name!r} is {n1}x{n2}, not the header's "
                    f"{header['jpi']}x{header['jpj']}")
            arrays[name] = values.reshape((n1, n2), order="F")
        else:
            want = {"k": header["jpk"], "km1": header["jpkm1"]}.get(
                ARRAY_SHAPE.get(name), None)
            require((n1, n2) == (header["jpi"], header["jpj"])
                    and (want is None or n3 == want),
                    f"{path}: rank-3 {name!r} is {n1}x{n2}x{n3}, not the "
                    f"header's {header['jpi']}x{header['jpj']}x{want}")
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
    # THE FLAGS MUST AGREE WITH EACH OTHER, or a zero can be manufactured.
    # The writer emits zeros for ah_wslp2 when the array is UNALLOCATED, so a
    # record claiming l_ldfslp with a33_allocated false would let the
    # "the fold is exactly zero" row pass by construction rather than by
    # physics.  ldfslp.F90:571 allocates it whenever the slopes are required.
    require(not (arrays["l_ldfslp"] == 1.0 and arrays["a33_allocated"] != 1.0),
            f"{path}: claims l_ldfslp but says ah_wslp2 is unallocated, so "
            "its a33 field is the writer's zero rather than NEMO's")
    # Likewise the clamp rows are only a measurement if the clamp RAN.  Its
    # guard is .NOT.(ln_SEOS .AND. rn_b0 == 0) at trazdf.F90:89.
    require(not (arrays["ln_SEOS"] == 1.0 and arrays["rn_b0"] == 0.0),
            f"{path}: the DRAKKAR clamp did not run on this record "
            "(ln_SEOS with rn_b0 = 0), so its pre/post rows compare a column "
            "with itself")
    require(int(arrays["jp_tem"]) != int(arrays["jp_sal"]),
            f"{path}: jp_tem and jp_sal are the same index")
    for name in EXPECTED_ARRAYS:
        block = np.asarray(arrays[name], dtype=np.float64)
        require(bool(np.isfinite(block).all()),
                f"{path}: {name!r} carries a non-finite value")
    return {"header": header, "arrays": arrays, "order": order,
            "sha256": digest,
            "path": str(path), "tile_shaped_salvage": salvaged}


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


# ROUND 37: AT-BAR-SIGNED-ZERO IS NO LONGER A PASSING STATUS.  It existed
# because legoESM wrote +0.0 into the two boundary slots where NEMO writes
# -0.0; that transcription landed, no row needs the exemption any more, and an
# independent diff review DEFEATED the sweep discharge through it -- a solve
# corrupted in the fully-dry columns was absorbed as a 5-cell signed-zero row
# and the gate exited 0 with STATUS AT-BAR.  A status that can swallow a wrong
# answer is not a pass.  The CLASSIFICATION stays, so a +0.0 regression is
# still named rather than folded into AT-BAR or over-claimed as DEBT; it just
# no longer passes.
PASSING = ("AT-BAR",)


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
    content = jnp.asarray(op["e3t_after"])   # the production caller's RHS dtype
    lower, diagonal, upper = nemo_tracer_tridiagonal(
        jnp.asarray(op["K"]), jnp.asarray(op["e3t_after"]),
        jnp.asarray(op["e3w_now"]), float(op["dt"]),
        jnp.asarray(op["wet"]) > 0.0, dtype=content.dtype)
    return {"zwi": np.asarray(lower), "zwd": np.asarray(diagonal),
            "zws": np.asarray(upper)}


def lego_sweep(rec: dict) -> dict[str, np.ndarray]:
    """legoESM's own ordered solve, on NEMO's dumped matrix and dumped RHS.

    Multiplied by ``tmask`` because that is what
    ``implicit_vertical_diffusion_nemo_tracer_pair`` does with the result;
    the comparison is against NEMO's PRE-clamp column.

    BLIND SPOT, named because a diff review walked through it: NEMO's own back
    substitution masks at every level (``trazdf.f90:543``, ``:546-547``), so
    BOTH sides of this comparison are zero at every dry cell and a solve that
    is arbitrarily wrong there scores as equal.  3120 of the 21120 scored
    cells are dry.  What covers those cells is the round-29 momentum arm,
    whose oracle is NEMO's UNMASKED ``uu(Kaa)``: the reviewer's corruption,
    invisible here, moved 3720 of its 21120 cells by 12345.
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
    writes, LEFT TO RIGHT,

        zrhs = (e3t(Kbb)) * pt(Kbb) + p2dt * (e3t(Kmm)) * pt(Krhs)

    (``trazdf.f90:528-529`` of this round's compiled ppsrc), i.e.
    ``(p2dt*e3t_Kmm) * T_Krhs``.  TWO mappings of NEMO's operands onto the
    builder's slots are scored, because round 35 quoted only the second and
    read its residual as a property of the builder:

    ``faithful``  -- ``h_after = p2dt*e3t_Kmm``, ``t_expl = T_Krhs``, which
        reproduces NEMO's own association exactly.
    ``premultiplied`` -- ``h_after = e3t_Kmm``, ``t_expl = p2dt*T_Krhs``,
        round 35's mapping, which forms ``e3t_Kmm*(p2dt*T_Krhs)`` -- an
        association NEMO never writes.  Kept, labelled, as the ASSOCIATION
        SENSITIVITY of this product; it is not a claim about the builder.

    In both, ``t_now`` and ``d_diss`` are zero.  Neither mapping reaches the
    production path, which forms ``h_naa*T_expl - h_now*T_now`` instead: that
    is the blind spot this gate declares, and it is unchanged by either row.
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
        t_rhs = _box(rec, f"{tag}_Krhs_in", jpkm1)
        common = dict(t_before=_box(rec, f"{tag}_Kbb_in", jpkm1),
                      t_now=zero, d_diss=zero,
                      h_before=e3t_kbb, h_now=e3t_kmm)
        out[f"faithful_{tag}"] = np.asarray(thickness_weighted_tracer_content(
            t_expl=t_rhs, h_after=p2dt * e3t_kmm, **common))
        out[f"premultiplied_{tag}"] = np.asarray(
            thickness_weighted_tracer_content(
                t_expl=p2dt * t_rhs, h_after=e3t_kmm, **common))
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
                                     "own signed_zero_only count",
                      "note": "THIS, not the dry diagonal, is the FIRST "
                              "statement of tra_zdf whose bits legoESM does "
                              "not reproduce: trazdf.f90:443-444 runs before "
                              ":445.  It is scored as its own status rather "
                              "than as AT-BAR precisely so the ordering "
                              "claim cannot be made without it."},
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


def _row_signature(report: dict) -> dict[str, tuple]:
    """Every scored row's status and value, for comparing two runs."""
    signature = {row["name"]: (row["status"], row["absolute_max"],
                               row["bit_unequal"])
                 for group in ("calibration_rows", "given_inputs_rows",
                               "clamp_rows")
                 for row in report[group]}
    # The condition rows are scored too, and the a33 plant lands ONLY there --
    # nextafter(0) is a denormal that vanishes when added to avt, so it moves
    # the fold-inert row and no matrix row.  Leaving them out of the signature
    # made that plant read as landing nowhere.
    signature.update({row["name"]: (row["status"], row["max_abs"], 0)
                      for row in report["condition_rows"]})
    # The ASSOCIATION rows are scored on the same 21120 cells as every other
    # row -- they are REPORTED rather than gated, which is not the same thing
    # as unscored.  Leaving them out made a plant that moves only an
    # association row read as landing nowhere, exactly as the condition rows
    # did before the line above.
    signature.update({row["name"]: (row["status"], row["absolute_max"],
                                    row["bit_unequal"])
                      for row in report["rhs_association_sensitivity"]})
    return signature


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
            _box(rec, f"rhs_{tag}", jpkm1), content[f"faithful_{tag}"],
            note="the content builder given NEMO's operands under NEMO's own "
                 "association (p2dt*e3t_Kmm)*T_Krhs, trazdf.f90:528-529"))
    association_rows = []
    for tag in ("T", "S"):
        association_rows.append(bit_row(
            f"{step}.rhs_content_premultiplied.{tag}",
            _box(rec, f"rhs_{tag}", jpkm1), content[f"premultiplied_{tag}"],
            note="ASSOCIATION SENSITIVITY, not a claim about the builder: "
                 "round 35's mapping forms e3t_Kmm*(p2dt*T_Krhs), which NEMO "
                 "never writes.  A residual here is this product's "
                 "sensitivity to grouping, and it is REPORTED, not scored -- "
                 "see rhs_association_sensitivity"))

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
        "tile_shaped_salvage": rec["tile_shaped_salvage"],
        "planted": plant,
        "calibration_rows": calibration,
        "given_inputs_rows": lego_rows,
        "clamp_rows": clamp_rows,
        # REPORTED, NOT SCORED.  These rows measure how much this product's
        # value depends on how it is GROUPED; they do not measure legoESM
        # against NEMO, because the grouping they use is not NEMO's.  Letting
        # them gate would make the verdict turn on a harness choice.
        "rhs_association_sensitivity": association_rows,
        "condition_rows": condition_rows,
        "named_deviations": deviations,
        "status": status,
        "calibrated": calibrated,
        "scored_box": {
            "i": [h["ntsi"], h["ntei"]], "j": [h["ntsj"], h["ntej"]],
            "halo": h["halo"], "levels": jpkm1,
            "cells": (h["ntei"] - h["ntsi"] + 1)
                     * (h["ntej"] - h["ntsj"] + 1) * jpkm1,
            "domain": [h["jpi"], h["jpj"], h["jpk"]],
        },
        "arrays_present_but_consumed_by_no_arm": UNCONSUMED,
        # AND THE SLICES NO ARM READS, inside arrays that ARE consumed.
        # MEASURED by an independent attack on round 36, not reasoned: setting
        # the WHOLE surface plane of avt to 999.0 leaves all thirteen
        # calibration rows at 0 cells unequal, and so does setting the whole
        # jk=jpk level of rhs_T/S, fwd_T/S, e3t_Kbb/Kmm, T/S_Kbb_in and
        # T/S_Krhs_in -- 11232 values -- to 12345.0.  Both follow from NEMO's
        # own structure (zwt(:,1)=0 at trazdf.f90:419 discards avt at jk=1,
        # and the matrix has jpkm1 rows), so "the calibration is 0 cells
        # unequal" certifies the slices the SOLVE reads and nothing else.
        "slices_no_arm_reads": {
            "avt/avs at jk=1": "trazdf.f90:419 sets zwt(:,1) = 0 after the "
                               "assembly, so the surface value is discarded",
            "every k-array at jk=jpk": "the tridiagonal has jpkm1 rows; the "
                                       "clamp rows are the only ones that "
                                       "score the full jpk ladder",
        },
        "rhs_blind_spot": {
            "status": "UNMEASURED",
            "reason": (
                "no arm here drives legoESM's OWN right-hand side as the card "
                "builds it: the production solve takes the RHS as an argument "
                "(implicit_solver.py:417-424) and this card resolves "
                "tracer_combine='concentration', so "
                "ocean_model_latlon_cgrid.py:11954-11999 leaves the content "
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
    if plant is not None:
        # A PLANT THAT LANDS NOWHERE IS NOT A CONTROL.  This gate's baseline
        # already carries a DEBT row -- the RHS association -- so a planted
        # run exiting non-zero proves nothing on its own: it would exit
        # non-zero with the plant deleted.  Compare the planted rows against
        # the unplanted ones and name what actually moved; a plant that moved
        # nothing is itself a violation.
        baseline = run(record, plant=None, with_card=False)
        before, after = _row_signature(baseline), _row_signature(report)
        moved = sorted(n for n in after if before.get(n) != after[n])
        report["plant_moved_rows"] = moved
        report["plant_landed"] = bool(moved)
        report["baseline_status"] = baseline["status"]
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
    for group in ("calibration_rows", "given_inputs_rows", "clamp_rows",
                  "rhs_association_sensitivity"):
        for row in report[group]:
            label = ("REPORTED" if group == "rhs_association_sensitivity"
                     else row["status"])
            print(f"{label:<12} {row['name']:<58} "
                  f"bit_unequal {row['bit_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.17g}"
                  + (f"  [status {row['status']}]"
                     if group == "rhs_association_sensitivity" else ""))
    for row in report["condition_rows"]:
        print(f"{row['status']:<12} {row['name']:<58} "
              f"max {row['max_abs']:.17g}")
    for row in report["named_deviations"]:
        print(f"{'INERT' if row['inert_here'] else 'LIVE':<12} "
              f"{row['deviation'][:100]}")
    for name, row in report["tile_shaped_salvage"].items():
        print(f"{'SALVAGED':<12} {name:<10} header said {row['declared']}, "
              f"NEMO wrote {row['written']} -- {row['reason']}")
    box = report["scored_box"]
    print(f"SCORED {box['cells']} cells, i {box['i']} j {box['j']} "
          f"halo {box['halo']}, on a {box['domain']} domain")
    # The blind spot goes in the PRINTED verdict, not only in the JSON: a
    # human reading "STATUS AT-BAR" would otherwise never learn that the
    # model's own right-hand side was never compared.
    print(f"BLIND-SPOT {report['rhs_blind_spot']['status']} "
          f"{', '.join(report['rhs_blind_spot']['rows'])}")
    if args.plant:
        print(f"PLANT {args.plant} landed={report['plant_landed']} "
              f"moved={report['plant_moved_rows']} "
              f"baseline_status={report['baseline_status']}")
    print(f"STATUS {report['status']}")
    if args.plant:
        # exit 1 = the plant landed AND the gate is red, which is the control
        # passing; exit 3 = the plant moved nothing, which is a violation of
        # the control itself and must never read as success.
        if not report["plant_landed"]:
            print("FAIL: the plant moved no row; this control proves nothing")
            return 3
        return 1 if report["status"] not in PASSING else 3
    return 0 if report["status"] in PASSING else 1


if __name__ == "__main__":
    sys.exit(main())
