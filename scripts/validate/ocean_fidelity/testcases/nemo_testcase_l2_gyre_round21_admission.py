#!/usr/bin/env python3
"""Admit an oracle acquisition by CONSUMED-FIELD identity, on any card.

WHY THIS EXISTS.  A WRITE-only instrument added to NEMO must change no model
state.  The proof is that every record the previous run already produced comes
back byte for byte.  Sometimes it does not, and the difference is not a state
change: NEMO's stream dumps write WHOLE work arrays including the ``nn_hls``
halo, which NEMO neither owns nor initialises, and one record (the pre
``tra_adv_trp`` momentum-side transport) carries a ``zFw`` slot the executed
branch has not defined yet.  Those bytes are uninitialised memory and they
differ between two runs of the same executable.

So the admission rule is: a raw byte difference is admitted only if every
differing ELEMENT is either

  * in the halo -- ``nn_hls = 2`` on every card in this campaign, read from
    each run's own ``ocean.output`` ("halo width (applies to both rows and
    columns) nn_hls = 2"): GYRE ``round19_oracle_v2_external/ocean.output:41``,
    ``lock_kt1_10/ocean.output:42``, ``overflow_kt1_10/ocean.output:44`` -- or
  * inside a slot this record's WRITER has not defined at the write point,
    declared per (magic, field) in ``SCHEMAS`` with its reason recorded there.

Every admitted difference is PRINTED with its index and its two values, so the
strength of the claim is visible rather than asserted.  A single differing bit
in an OWNED cell of a defined field is a hard failure, and so is a changed
restart, a changed mesh, a missing inherited record or an unexpected new one.

CARD-INDEPENDENT BY CONSTRUCTION.  Nothing here is GYRE-specific:

  * the RECORD INVENTORY is discovered by globbing the SOURCE run directory;
  * each record's KIND is its own 16-byte magic, not its file name;
  * each record's DIMENSIONS come from its own header, so the same gate reads
    GYRE's 36x26x31, LOCK_EXCHANGE's 134x7x21 and OVERFLOW's 206x7x101 with no
    argument;
  * every path, the byte-identical file list and the allowed additions are CLI
    arguments.

The declared layout is CHECKED against the file: if the fields a schema
declares do not account for the payload exactly, the gate raises rather than
comparing a misaligned buffer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np


HALO = 2  # nn_hls; cited per card in the module docstring above.
DEFAULT_BASE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external")
DEFAULT_CAND = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round21_oracle_v2_stage_ww")
DEFAULT_TWIN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round19_oracle_v2_external_coeff")
DEFAULT_IDENTICAL = ("GYRE_OMIP_L2_P3_00000010_restart.nc",)


class AdmissionError(RuntimeError):
    """A record the gate cannot parse.  Never downgraded to a comparison."""


def _worktree_stamp() -> dict:
    """Which tree produced this report -- BEST EFFORT, and it says which.

    This gate runs at ACQUISITION time, from ``run.sh``, in a shell whose
    ``PYTHONPATH`` does not carry legoESM: it is deliberately dependency-free
    apart from numpy so it can admit a record on the machine that just wrote
    one.  So the stamp is attempted and its absence is REPORTED rather than
    faked -- an artifact that cannot say which tree produced it must say that,
    not omit the question.
    """
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp
    except Exception as error:                    # pragma: no cover - env only
        return {"unavailable": f"legoesm is not importable here: {error}"}
    try:
        return worktree_stamp()
    except Exception as error:
        return {"unavailable": str(error)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _f(*specs):
    return specs


# --- record schemas, keyed by the record's own 16-byte magic ---------------
#
# Each entry is (n_header_ints, (ix, iy, iz), fields).  ``iz`` is None for a
# purely 2-D record.  ``fields`` is a callable of (n2, n3, nc) returning
# (name, count, projection, defined) tuples in stream order, where projection
# is "3" (nx,ny,nz), "2" (nx,ny), "c" (already owned-only) or "s" (scalar),
# and ``defined`` is False only where the WRITER has not initialised the slot
# at the write point.
#
# Every writer in this campaign emits nx, ny, nz, bits as the LAST four
# integers of its PRIMARY header block; the indices are still written out per
# schema rather than derived, because NEMO_L2_SLOW_2 appends a second integer
# block and a derived rule would silently take the wrong four there.
SCHEMAS: dict[str, tuple] = {
    # stprk3.F90:300-306 -- uu_b, vv_b, un_adv, vn_adv at one level.
    "NEMO_L1_BTFRM_1": (6, (3, 4, None), lambda n2, n3, nc: _f(
        *[(n, n2, "2", True) for n in ("uu_b", "vv_b", "un_adv", "vn_adv")])),
    # stprk3.F90:338-343 -- uu/vv(Krhs).
    "NEMO_L1_RHS___1": (7, (3, 4, 5), lambda n2, n3, nc: _f(
        ("uu_Krhs", n3, "3", True), ("vv_Krhs", n3, "3", True))),
    # stprk3.F90:322-327 -- ts(:,:,:,:,klevel), uu, vv, ssh.  jpts = 2.
    "NEMO_L1_ENTRY_1": (8, (3, 4, 5), lambda n2, n3, nc: _f(
        ("T", n3, "3", True), ("S", n3, "3", True), ("u", n3, "3", True),
        ("v", n3, "3", True), ("ssh", n2, "2", True))),
    "NEMO_L1_STAGE_1": (9, (4, 5, 6), lambda n2, n3, nc: _f(
        ("T", n3, "3", True), ("S", n3, "3", True), ("u", n3, "3", True),
        ("v", n3, "3", True), ("ssh", n2, "2", True))),
    # stprk3_stg.F90:343-345 -- the momentum-side transport, written BEFORE
    # tra_adv_trp.  zFw's disposition depends on the ARM the run resolves and
    # is decided per run by ``_undefined_slots`` below, never hardcoded.
    "NEMO_L1_TRANSP_1": (8, (4, 5, 6), lambda n2, n3, nc: _f(
        ("zFu", n3, "3", True), ("zFv", n3, "3", True),
        ("zFw", n3, "3", True))),
    # stprk3_stg.F90:622-626 -- the tracer-side transport, written AFTER
    # tra_adv_trp, so zFw IS defined here.
    "NEMO_L2_TRTRP_1": (11, (7, 8, 9), lambda n2, n3, nc: _f(
        ("zFu", n3, "3", True), ("zFv", n3, "3", True),
        ("zFw", n3, "3", True))),
    "NEMO_L2_WZVOP_1": (8, (4, 5, 6), lambda n2, n3, nc: _f(
        *[(n, n3, "3", True)
          for n in ("ww_pre_aimp", "ww_post_aimp", "pFw")])),
    "NEMO_L2_TRPOP_2": (8, (4, 5, 6), lambda n2, n3, nc: _f(
        ("e2u", n2, "2", True), ("e3u", n3, "3", True), ("uu", n3, "3", True),
        ("zub", n2, "2", True), ("umask", n3, "3", True),
        ("zFu", n3, "3", True),
        ("e1v", n2, "2", True), ("e3v", n3, "3", True), ("vv", n3, "3", True),
        ("zvb", n2, "2", True), ("vmask", n3, "3", True),
        ("zFv", n3, "3", True),
        ("un_adv", n2, "2", True), ("r1_hu", n2, "2", True),
        ("uu_b", n2, "2", True), ("vn_adv", n2, "2", True),
        ("r1_hv", n2, "2", True), ("vv_b", n2, "2", True))),
    "NEMO_L2_RKTRA_1": (11, (7, 8, 9), lambda n2, n3, nc: _f(
        *[(n, n3, "3", True) for n in (
            "zero_T", "zero_S", "zFu", "zFv", "zFw", "after_advection_T",
            "after_advection_S", "after_sbc_T", "after_sbc_S", "Kbb_T",
            "Kbb_S", "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S")],
        *[(n, n2, "2", True) for n in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")])),
    "NEMO_L2_SLOW_2": (15, (4, 5, 6), lambda n2, n3, nc: _f(
        *[(n, n3, "3", True)
          for n in ("e3u", "krhs_u", "umask", "e3v", "krhs_v", "vmask")],
        *[(n, nc, "c", True) for n in ("depth_u", "depth_v")],
        *[(n, n2, "2", True) for n in ("r1_hu0", "r1_hv0")],
        *[(n, nc, "c", True) for n in ("post_drag_u", "post_drag_v")],
        *[(n, n2, "2", True) for n in ("cd_u", "cd_v")],
        ("r1_rho0", 1, "s", True),
        *[(n, n2, "2", True) for n in ("utau", "vtau", "r1_hu", "r1_hv")],
        *[(n, nc, "c", True) for n in ("post_wind_u", "post_wind_v")])),
}

# --- the ONLY admission ground that is not the halo -----------------------
#
# A slot may be waived from the bit test only where the WRITER has not defined
# it at the write point, and whether it has is a property of the ARM THE RUN
# RESOLVES, not of the record kind.  So each entry names the namelist switch
# that decides it and the value under which the slot is undefined, and the
# gate reads that switch out of the run's OWN ocean.output.
#
# THE CASE THIS EXISTS FOR.  ``zFw`` in the momentum-side transport record:
# under the VECTOR-INVARIANT arm the write at stprk3_stg.F90:343-345 precedes
# tra_adv_trp (:592/:623), which is what creates zFw -- the source says so at
# :318, "zFw used in tracers only and computed in tra_adv_trp" -- so the slot
# is uninitialised memory there.  Under the FLUX-FORM arm the ELSE branch at
# :326-334 fills zFw = e1e2t*ww BEFORE that same write, so the slot is a real
# state-carrying field and waiving it would switch off the bit test on it.
# GYRE resolves ln_dynadv_vec = T; both tanks resolve F.
#
# FAIL CLOSED: if the run's ocean.output is missing or does not print the
# switch, the slot is treated as DEFINED -- the strict reading -- and the
# report says why.
UNDEFINED_SLOTS = {
    ("NEMO_L1_TRANSP_1", "zFw"): {
        "switch": "ln_dynadv_vec",
        "undefined_when": "T",
        "reason": (
            "under ln_dynadv_vec = T the write at stprk3_stg.F90:343-345 "
            "precedes tra_adv_trp (:592, :623), which is what computes zFw "
            "(:318); the slot is uninitialised memory there.  Under "
            "ln_dynadv_vec = F the flux-form branch at :326-334 fills "
            "zFw = e1e2t*ww before the same write, so the slot IS defined "
            "and is bit-tested"),
    },
}


def _resolved_switch(run: Path, switch: str) -> tuple[str | None, str]:
    """Read one resolved namelist switch out of a run's own ocean.output."""
    log = run / "ocean.output"
    if not log.is_file():
        return None, f"{log} does not exist"
    pattern = re.compile(rf"\b{re.escape(switch)}\s*=\s*([TF])\b")
    try:
        text = log.read_text(errors="replace")
    except OSError as error:                      # pragma: no cover - IO only
        return None, f"{log} unreadable: {error}"
    for number, line in enumerate(text.splitlines(), 1):
        found = pattern.search(line)
        if found:
            return found.group(1), f"{log.name}:{number}"
    return None, f"{log} does not print {switch}"


def _undefined_slots(run: Path) -> tuple[dict, list]:
    """Resolve every waiver against THIS run, and record how it resolved."""
    waived, rows = {}, []
    for (magic, field), spec in UNDEFINED_SLOTS.items():
        value, where = _resolved_switch(run, spec["switch"])
        undefined = value == spec["undefined_when"]
        if undefined:
            waived[(magic, field)] = spec["reason"]
        rows.append({
            "magic": magic, "field": field, "switch": spec["switch"],
            "resolved_value": value, "resolved_at": where,
            "undefined_when": spec["undefined_when"],
            "waived": undefined,
            "reason": spec["reason"] if undefined else (
                "NOT waived on this run: the switch does not resolve to the "
                "value that leaves the slot undefined, so the slot is "
                "bit-tested like any other"),
        })
    return waived, rows

# WHERE EACH RECORD IS WRITTEN.  Verified line by line against the MY_SRC
# files that are actually compiled, not carried over from an older round:
#   grep -n "<magic>" <card>/MY_SRC/<file>.F90
# LOCK_EXCHANGE_OMIP_L1_P3/MY_SRC/stprk3.F90 and
# GYRE_OMIP_L2_P3_SM_R29ZDF/MY_SRC/{stprk3_stg,stp2d,traadv}.F90 carry the
# same writers on every card in this campaign.
SOURCES = {
    "NEMO_L2_TRPOP_2": "MY_SRC/stprk3_stg.F90:297",
    "NEMO_L2_WZVOP_1": "MY_SRC/traadv.F90:225",
    "NEMO_L2_RKTRA_1": "MY_SRC/stprk3_stg.F90:673",
    "NEMO_L2_SLOW_2": "MY_SRC/stp2d.F90:187",
    "NEMO_L2_TRTRP_1": "MY_SRC/stprk3_stg.F90:638",
    "NEMO_L1_TRANSP_1": "MY_SRC/stprk3_stg.F90:343",
    "NEMO_L1_STAGE_1": "MY_SRC/stprk3.F90:369",
    "NEMO_L1_ENTRY_1": "MY_SRC/stprk3.F90:96",
    "NEMO_L1_RHS___1": "MY_SRC/stprk3.F90:385",
    "NEMO_L1_BTFRM_1": "MY_SRC/stprk3.F90:348",
}
PARSERS = {
    "NEMO_L2_TRPOP_2": "nemo_testcase_l2_gyre_round13_tracer.py:93-140",
    "NEMO_L2_WZVOP_1": "nemo_testcase_l2_gyre_stage3_completion_gate.py:110-147",
    "NEMO_L2_RKTRA_1": "nemo_testcase_l2_gyre_round13_tracer.py:46-91",
    "NEMO_L2_SLOW_2": "nemo_testcase_l2_gyre_round16_slow_forcing.py:49-116",
    "NEMO_L2_TRTRP_1": "nemo_testcase_l2_gyre_phase3_gate.py:190-234",
    "NEMO_L1_TRANSP_1": "nemo_testcase_l2_gyre_phase3_gate.py:190-234",
    "NEMO_L1_STAGE_1": "nemo_testcase_phase3_first_divergence_gate.py:67-89",
    "NEMO_L1_ENTRY_1": "nemo_testcase_phase3_first_divergence_gate.py:135-158",
    "NEMO_L1_RHS___1": "nemo_testcase_phase3_first_divergence_gate.py:115-133",
    "NEMO_L1_BTFRM_1": "nemo_testcase_phase3_first_divergence_gate.py:160-185",
}


def read_header(raw: bytes, path: Path):
    """Magic, dims and declared field layout, all from the record itself."""
    if len(raw) < 16:
        raise AdmissionError(f"{path}: shorter than a magic")
    magic = raw[:16].decode("ascii", "replace").rstrip()
    if magic not in SCHEMAS:
        raise AdmissionError(f"{path}: unregistered record magic {magic!r}")
    n_ints, (ix, iy, iz), fields = SCHEMAS[magic]
    header_bytes = 16 + 4 * n_ints
    if len(raw) < header_bytes:
        raise AdmissionError(f"{path}: truncated {magic} header")
    ints = struct.unpack(f"={n_ints}i", raw[16:header_bytes])
    nx, ny = ints[ix], ints[iy]
    nz = ints[iz] if iz is not None else 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise AdmissionError(f"{path}: bad dims {(nx, ny, nz)} in {magic}")
    if nx <= 2 * HALO or ny <= 2 * HALO:
        raise AdmissionError(
            f"{path}: {magic} declares {nx}x{ny}, which has no owned cell "
            f"outside a {HALO}-cell halo")
    layout = list(fields(nx * ny, nx * ny * nz, (nx - 4) * (ny - 4)))
    declared = header_bytes + 8 * sum(count for _, count, _, _ in layout)
    if declared != len(raw):
        raise AdmissionError(
            f"{path}: {magic} declares {declared} bytes for {nx}x{ny}x{nz} "
            f"but the file is {len(raw)}; the schema does not fit this record")
    return magic, header_bytes, nx, ny, nz, layout


def _owned_selector(projection: str, ids: np.ndarray, nx: int, ny: int):
    """Split differing flat indices into owned and halo, per projection."""
    if projection == "3":
        ii, jj = ids % nx, (ids // nx) % ny
        index = np.stack([ii, jj, ids // (nx * ny)], axis=1)
    elif projection == "2":
        ii, jj = ids % nx, ids // nx
        index = np.stack([ii, jj], axis=1)
    else:
        # "c" arrays are already owned-only; "s" is a scalar.  Nothing in
        # either is halo, so every difference in them is a violation.
        return np.ones(ids.size, dtype=bool), ids[:, None]
    owned = ((ii >= HALO) & (ii < nx - HALO)
             & (jj >= HALO) & (jj < ny - HALO))
    return owned, index


def _project(values: np.ndarray, projection: str, nx: int, ny: int, nz: int):
    if projection == "3":
        return values.reshape((nx, ny, nz), order="F")[
            HALO:-HALO, HALO:-HALO, :]
    if projection == "2":
        return values.reshape((nx, ny), order="F")[HALO:-HALO, HALO:-HALO]
    return values


def compare_record(a_path: Path, b_path: Path, plant, *, waived=None,
                   max_listed: int = 16):
    """Compare one record; admit halo and undefined-slot differences only.

    ``waived`` is the RESOLVED waiver map for this run -- ``{(magic, field):
    reason}`` -- built by ``_undefined_slots`` from the run's own
    ``ocean.output``.  An empty map means every field is bit-tested, which is
    the strict reading and the default.
    """
    waived = {} if waived is None else waived
    a_bytes, b_bytes = a_path.read_bytes(), b_path.read_bytes()
    if len(a_bytes) != len(b_bytes):
        return {"consumed_equal": False, "error": "size mismatch"}
    magic, header_bytes, nx, ny, nz, layout = read_header(a_bytes, a_path)
    b_magic, _, bx, by, bz, _ = read_header(b_bytes, b_path)
    if (b_magic, bx, by, bz) != (magic, nx, ny, nz):
        return {"consumed_equal": False,
                "error": f"schema/dims changed {(magic, nx, ny, nz)} -> "
                         f"{(b_magic, bx, by, bz)}"}
    if a_bytes[:header_bytes] != b_bytes[:header_bytes]:
        return {"consumed_equal": False, "error": "header mismatch"}

    offset, changed, admitted, consumed_equal = header_bytes, [], [], True
    for name, count, projection, _ in layout:
        defined = (magic, name) not in waived
        aa = np.frombuffer(a_bytes, np.float64, count, offset)
        bb = np.frombuffer(b_bytes, np.float64, count, offset)
        element_diff = aa.view(np.uint64) != bb.view(np.uint64)
        if np.any(element_diff):
            ids = np.flatnonzero(element_diff)
            owned, index = _owned_selector(projection, ids, nx, ny)
            changed.append({
                "field": name,
                "changed_elements": int(ids.size),
                "changed_bytes": int(np.count_nonzero(
                    np.frombuffer(a_bytes, np.uint8, count * 8, offset)
                    != np.frombuffer(b_bytes, np.uint8, count * 8, offset))),
                "changed_in_owned_cells": int(np.count_nonzero(owned)),
                "first_index_0based": [int(v) for v in index[0]],
                "slot_defined_at_write_point": defined,
            })
            # Every admitted difference is listed with its VALUES, capped so a
            # whole undefined array cannot bury the report.  An owned cell of
            # a DEFINED field is never listed here: it is a violation, and the
            # bit test below is what fails on it.
            listed = 0
            for position in range(ids.size):
                if owned[position] and defined:
                    continue
                if listed >= max_listed:
                    admitted.append(
                        {"record": a_path.name, "field": name,
                         "note": f"{ids.size - listed} further differences of "
                                 "the same class not listed"})
                    break
                admitted.append({
                    "record": a_path.name, "field": name,
                    "index_0based": [int(v) for v in index[position]],
                    "reason": ("halo" if not owned[position]
                               else "slot undefined at the write point"),
                    "baseline_value": float(aa[ids[position]]),
                    "candidate_value": float(bb[ids[position]]),
                })
                listed += 1
        pa = _project(aa, projection, nx, ny, nz)
        pb = _project(bb, projection, nx, ny, nz)
        if defined and plant and not plant[0]:
            pb = np.ascontiguousarray(pb).copy()
            pb.reshape(-1)[0:1].view(np.uint64)[:] ^= np.uint64(1)
            plant[0] = True
        if defined and not _bits_equal(np.ascontiguousarray(pa),
                                       np.ascontiguousarray(pb)):
            consumed_equal = False
        offset += count * 8
    assert offset == len(a_bytes), (a_path, offset, len(a_bytes))
    raw = np.frombuffer(a_bytes, np.uint8) != np.frombuffer(b_bytes, np.uint8)
    undefined = sorted({n for n, _, _, _ in layout
                        if (magic, n) in waived})
    return {
        "consumed_equal": consumed_equal,
        "magic": magic,
        "dims_with_halo": [nx, ny, nz],
        "raw_differing_bytes": int(np.count_nonzero(raw)),
        "first_differing_byte_1based": (
            int(np.flatnonzero(raw)[0] + 1) if np.any(raw) else None),
        "changed_fields": changed,
        "admitted_differences": admitted,
        "undefined_slots": undefined,
        "undefined_slot_reasons": [waived[(magic, n)] for n in undefined],
        "writer": SOURCES.get(magic, "unregistered"),
        "parser": PARSERS.get(magic, "unregistered"),
    }


# --- the ordered barotropic operand stream, whose schema GREW in round 21 ---
BT_NAMES = ("u_entry", "v_entry", "u_history_b", "v_history_b", "u_history_bb",
            "v_history_bb", "eta_entry", "eta_history_b", "eta_history_bb", "u_mid",
            "v_mid", "eta_mid", "face_depth_u_mid", "face_depth_v_mid",
            "metric_transport_u", "metric_transport_v", "metric_e2u", "metric_e1v",
            "r1_area", "continuity_du", "continuity_dv", "continuity_divergence",
            "continuity_forcing", "eta_exit", "face_ssh_u_exit", "face_ssh_v_exit",
            "eta_pgf", "r1_dx_u", "r1_dy_v", "pgf_u", "pgf_v", "cor_u", "cor_v",
            "trd_u", "trd_v", "slow_u", "slow_v", "u_exit", "v_exit",
            "face_depth_u_exit", "face_depth_v_exit", "r1_face_depth_u_exit",
            "r1_face_depth_v_exit")
BT_EXTRA = ("ffu_nw", "ffu_ne", "ffu_sw", "ffu_se", "ffv_sw", "ffv_se",
            "ffv_nw", "ffv_ne")


def _read_bt(path: Path):
    """Dims come from this record's own header too (jpi, jpj at 3, 4)."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        nx, ny = header[3], header[4]
        n2, nc = nx * ny, (nx - 4) * (ny - 4)
        dt = float(np.fromfile(handle, np.float64, 1)[0])
        names = BT_NAMES + (BT_EXTRA if magic.endswith("_2") else ())
        rows = []
        for _ in range(2):
            jn = struct.unpack("=i", handle.read(4))[0]
            weights = np.fromfile(handle, np.float64, 7)
            values = {}
            for name in names:
                values[name] = np.fromfile(
                    handle, np.float64,
                    nc if name in {"slow_u", "slow_v", *BT_EXTRA} else n2)
            rows.append((jn, weights, values))
        if handle.read(1) != b"":
            raise AdmissionError(f"{path}: trailing bytes after two BTORD rows")
    return magic, header, dt, rows


def _compare_bt(a: Path, b: Path, plant=None) -> dict:
    """Compare the ordered barotropic stream on EVERY field both sides carry.

    This record's schema GREW in round 21 (the ``_2`` magic appends eight
    ``ff*`` fields), which is the one legitimate reason the two files differ
    in length.  The comparison is therefore over the INTERSECTION of the two
    field sets, not over a hardcoded list: with the old hardcoded ``BT_NAMES``
    loop the eight appended fields were read and never compared, so a state
    change in them was invisible once both sides carried them.
    """
    aa, bb = _read_bt(a), _read_bt(b)
    equal = aa[2] == bb[2]
    common = [name for name in BT_NAMES + BT_EXTRA
              if name in aa[3][0][2] and name in bb[3][0][2]]
    common_diffs = []
    for ar, br in zip(aa[3], bb[3]):
        equal &= ar[0] == br[0] and _bits_equal(ar[1], br[1])
        for name in common:
            left, right = ar[2][name], br[2][name]
            if plant is not None and plant and not plant[0]:
                right = right.copy()
                right[0:1].view(np.uint64)[:] ^= np.uint64(1)
                plant[0] = True
            if not _bits_equal(left, right):
                common_diffs.append([ar[0], name])
                equal = False
    raw_a, raw_b = a.read_bytes(), b.read_bytes()
    limit = min(len(raw_a), len(raw_b))
    raw = (np.frombuffer(raw_a[:limit], np.uint8)
           != np.frombuffer(raw_b[:limit], np.uint8))
    return {
        "consumed_equal": bool(equal), "common_field_differences": common_diffs,
        "schema": f"{aa[0]}->{bb[0]}", "compared_fields": common,
        "fields_only_on_one_side": sorted(
            set(aa[3][0][2]) ^ set(bb[3][0][2])),
        "raw_differing_bytes": int(np.count_nonzero(raw)),
        "appended_bytes": max(0, len(raw_b) - len(raw_a)),
        "admitted_differences": [],
        "first_differing_byte_1based": (
            int(np.flatnonzero(raw)[0] + 1) if np.any(raw) else limit + 1),
        "writer": "MY_SRC/dynspg_ts.F90:598-599,934-942",
        "parser": "nemo_testcase_l2_gyre_round14_advmean.py:123-179",
    }


def run(baseline: Path, candidate: Path, *, twin=None,
        identical=DEFAULT_IDENTICAL, allowed_new=None, writer=None,
        plant_consumed: bool = False) -> dict:
    plant = [not plant_consumed]
    baseline_names = sorted(p.name for p in baseline.glob("oracle_*.bin"))
    candidate_names = sorted(p.name for p in candidate.glob("oracle_*.bin"))
    if not baseline_names:
        raise AdmissionError(f"{baseline}: no oracle_*.bin records to inherit")
    allowed_new = set() if allowed_new is None else set(allowed_new)
    violations, rows, exact = [], [], 0
    unexpected = set(candidate_names) - set(baseline_names) - allowed_new
    if unexpected:
        violations.append(f"unexpected candidate records: {sorted(unexpected)}")
    missing = set(baseline_names) - set(candidate_names)
    if missing:
        violations.append(f"missing inherited records: {sorted(missing)}")
    declared_absent = allowed_new - set(candidate_names)
    if declared_absent:
        violations.append(
            "--allowed-new names records the candidate never wrote: "
            f"{sorted(declared_absent)}")
    waived, waiver_rows = _undefined_slots(baseline)
    for name in baseline_names:
        a, b = baseline / name, candidate / name
        if not b.is_file():
            continue  # already reported as missing
        raw_equal = a.read_bytes() == b.read_bytes()
        # THE PLANT MUST ALWAYS LAND.  It used to be reachable only through
        # compare_record, which only runs on records that already differ raw
        # -- so a PERFECT twin, the strongest possible outcome, produced a
        # plant that silently did nothing and a run.sh that then refused the
        # round.  A pending plant now forces the comparison open on the first
        # record whether or not its bytes differ.
        if raw_equal and plant[0]:
            exact += 1
            continue
        if name.startswith("oracle_bt_ordered_operands"):
            result = _compare_bt(a, b, plant)
        else:
            result = compare_record(a, b, plant, waived=waived)
        if raw_equal:
            result["raw_identical_before_plant"] = True
        if twin is not None and twin.is_dir() and (twin / name).is_file():
            t = twin / name
            other = (_compare_bt(t, b, [True])
                     if name.startswith("oracle_bt_ordered_operands")
                     else compare_record(t, b, [True], waived=waived))
            result["twin_raw_equal"] = t.read_bytes() == b.read_bytes()
            result["twin_consumed_equal"] = other.get("consumed_equal")
        rows.append({"record": name, **result})
        if not result.get("consumed_equal", False):
            violations.append(name)
    # --writer stamps ONE provenance string, so it is only truthful when every
    # changed record comes from the same instrument.  Fail closed rather than
    # publish one file:line over records written by different ones.
    if plant_consumed and not plant[0]:
        violations.append(
            "the requested plant was never applied: no record reached the "
            "field comparison, so this run proves nothing")
    if writer:
        magics = {row.get("magic") for row in rows if row.get("magic")}
        if len(magics) > 1:
            violations.append(
                f"--writer names one instrument but {sorted(magics)} record "
                "kinds changed; pass no --writer or split the run")
    identity_rows, artifacts = [], {}
    for name in identical:
        a, b = baseline / name, candidate / name
        if not a.is_file() or not b.is_file():
            violations.append(f"{name}: missing on one side")
            identity_rows.append(
                {"file": name, "byte_identical": False, "error": "missing"})
            continue
        sa, sb = _sha(a), _sha(b)
        identity_rows.append({"file": name, "byte_identical": sa == sb,
                              "baseline_sha256": sa, "candidate_sha256": sb})
        artifacts[f"{name}.baseline_sha256"] = sa
        artifacts[f"{name}.candidate_sha256"] = sb
        if sa != sb:
            violations.append(name)
    admitted = [entry for row in rows
                for entry in row.get("admitted_differences", [])]
    # An admission ground with no registered reason would be a silent waiver.
    for row in rows:
        for reason in row.get("undefined_slot_reasons", []):
            if not reason or reason == "NO REASON REGISTERED":
                violations.append(
                    f"{row['record']}: a slot is waived from the bit test "
                    "with no registered reason")
    return {
        "worktree": _worktree_stamp(),
        "format": "nemo-testcase-consumed-field-admission-v2",
        "baseline": str(baseline), "candidate": str(candidate),
        "halo_width_nn_hls": HALO,
        "undefined_slot_resolution": waiver_rows,
        "baseline_oracle_records": len(baseline_names),
        "candidate_oracle_records": len(candidate_names),
        "byte_identical_records": exact,
        "classified_changed_records": rows,
        "admitted_differences": admitted,
        "admitted_difference_count": len(admitted),
        "byte_identical_files": identity_rows,
        "allowed_new_records": sorted(allowed_new),
        "plant_consumed": plant_consumed,
        "plant_applied": bool(plant_consumed and plant[0]),
        "writer_override": writer,
        "verdict": "PASS" if not violations else "FAIL",
        "violations": violations,
        "artifacts": artifacts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASE,
                        help="the SOURCE run whose records must be inherited")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CAND,
                        help="the instrumented run under admission")
    parser.add_argument("--twin", type=Path, default=DEFAULT_TWIN,
                        help="optional third run; reported, never gating")
    parser.add_argument("--identical", nargs="+",
                        default=list(DEFAULT_IDENTICAL),
                        help="files that must be byte-identical on both sides")
    parser.add_argument("--allowed-new", nargs="*", default=None,
                        help="record names the candidate may add")
    parser.add_argument("--writer",
                        help="instrument file:line for this card's records")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-consumed", action="store_true")
    args = parser.parse_args(argv)
    allowed_new = (
        {f"oracle_rkstage_ww_kt00000001_s{s}.bin" for s in (1, 2, 3)}
        if args.allowed_new is None else set(args.allowed_new))
    try:
        report = run(args.baseline, args.candidate, twin=args.twin,
                     identical=tuple(args.identical), allowed_new=allowed_new,
                     writer=args.writer, plant_consumed=args.plant_consumed)
    except AdmissionError as error:
        print(json.dumps({"verdict": "GATE-ERROR", "detail": str(error)},
                         indent=2))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for entry in report["admitted_differences"]:
        if "note" in entry:
            print(f"  ADMITTED {entry['record']} {entry['field']:<12} "
                  f"{entry['note']}")
            continue
        print(f"  ADMITTED {entry['record']} {entry['field']:<12} "
              f"{str(entry['index_0based']):<16} {entry['reason']:<34} "
              f"{entry['baseline_value']:.6e} vs {entry['candidate_value']:.6e}")
    print(f"CONSUMED_FIELD_ADMISSION {report['verdict']}: "
          f"exact={report['byte_identical_records']}/"
          f"{report['baseline_oracle_records']} "
          f"changed={len(report['classified_changed_records'])} "
          f"admitted={report['admitted_difference_count']} "
          f"plant={report['plant_consumed']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
