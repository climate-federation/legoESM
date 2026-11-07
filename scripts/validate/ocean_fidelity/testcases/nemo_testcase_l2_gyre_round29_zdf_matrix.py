#!/usr/bin/env python
"""Round-29 GYRE ``dyn_zdf`` matrix record: reader, calibration, and gate.

WHY THIS EXISTS.  GYRE's kt=1 momentum divergence is localised per stage by
the campaign's own gate --- stage 1 ``2.710505431213761e-19`` (AT-BAR),
stage 2 ``4.740083109008864e-13``, stage 3 ``9.481924730527598e-07`` --- so
essentially all of it enters at stage 3.  Two momentum operators run at
stage 3 and nowhere else: ``dyn_ldf`` (``stprk3_stg.F90:493``, compiled
``BLD/ppsrc/nemo/stprk3_stg.f90:485``) and ``dyn_zdf`` (``:523`` / ppsrc
``:498``).  No dumped GYRE record holds a stage-3 momentum frame of any kind:
every momentum-operand writer in ``cfgs/GYRE_OMIP_L2_P3_SM/MY_SRC`` is gated
on ``kstg == 2``.  The round-29 instrument fills that gap; this module reads
what it writes.

WHAT THE GATE PROVES WITHOUT A MODEL.  Two arms run on the record alone and
are the reader's calibration, in the Rule-1e sense: a number from this record
may not be quoted until both are at the bar.

* ``matrix`` --- rebuild ``zwi``/``zwd``/``zws`` for both faces from the
  record's OWN ``avm``, ``e3u_Kaa``, ``e3uw_Kmm``, ``wumask``, ``rCdU_bot``
  and ``mbku``, by the statement ``dynzdf.F90:176-192`` (inner), ``:193-199``
  (surface) and ``:293-296`` (implicit bottom drag), and require bit
  equality with the dumped matrix.
* ``solve`` --- run NEMO's own three recurrences (``dynzdf.F90:307-346``)
  on the dumped matrix and the dumped pre-solve column vector, with the
  ``key_RK3`` wind term (``:328-330``), and require bit equality with the
  dumped solved ``uu``/``vv(Kaa)``.

Both are only correct because four contributions are DEAD on GYRE, read off
the oracle's own printed output rather than assumed: ``ln_zad_Aimp = F``
(``ocean.output:553``) kills the Courant-dependent implicit vertical
advection (``dynzdf.F90:229-260``, ``:406-442``); the resolved ``iso-level
laplacian operator`` (``ocean.output:706``) means ``nldf_dyn /= np_lap_i`` and
kills the rotated lateral-mixing contribution (``:277-291``, ``:444-458``);
``ln_isfcav = F`` (``:338``) and ``ln_drgice_imp = F`` (``:630``) kill every
top-friction block.  If either arm is over the bar, one of those four readings
is wrong or the instrument is --- and that is the finding, not a number from
the record.

WHAT IT DOES NOT PROVE.  Scoring legoESM's stage-3 momentum RHS against
``uu_Krhs_in`` is what decides whether the owner is inside ``dyn_zdf`` or at
``dyn_ldf``, and legoESM has no hook that exposes its pre-``dyn_zdf``
momentum RHS.  That row is reported UNMEASURED with that reason; it is not
silently omitted.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

MAGIC = "NEMO_L2_ZDFMX_1"
BAR = 1.0e-15
CASE = "GYRE-zco"
HEADER_FIELDS = (
    "version", "kt", "kstg", "Kbb", "Kmm", "Krhs", "Kaa",
    "jpi", "jpj", "jpk", "jpkm1", "ntsi", "ntei", "ntsj", "ntej", "bits",
)
# Every array the writer emits, so a SHORT record is a hard failure rather
# than a quietly missing arm (Rule 1: coverage is driven by what the oracle
# provides, not by what the reader happens to ask for).
EXPECTED_ARRAYS = (
    "uu_Krhs_in", "vv_Krhs_in", "uu_Kbb_in", "vv_Kbb_in",
    "uu_b_Kaa", "vv_b_Kaa", "uu_Kaa_pre", "vv_Kaa_pre",
    "zwi_u", "zwd_u", "zws_u", "zwi_v", "zwd_v", "zws_v",
    "avm", "e3u_Kaa", "e3uw_Kmm", "e3v_Kaa", "e3vw_Kmm",
    "wumask", "wvmask", "umask", "vmask",
    "rCdU_bot", "mbku", "mbkv", "utauU", "vtauV",
    "uu_Kaa_out", "vv_Kaa_out", "rDt", "rho0",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(2)


def read_zdf_matrix(path: Path) -> dict:
    """Parse the self-describing record to EOF.

    The layout is a 16-byte magic, sixteen default integers, then one
    ``(name[16], rank, n1, n2, n3, payload)`` group per array repeated until
    the stream ends.  Nothing here knows the array list in advance; the list
    is only used AFTERWARDS, to refuse a short record.
    """
    raw = path.read_bytes()
    require(len(raw) > 16 + 64, f"{path}: shorter than one header")
    try:
        magic = raw[:16].decode("ascii").rstrip()
    except UnicodeDecodeError:
        require(False, f"{path}: magic is not ASCII")
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    header = dict(zip(HEADER_FIELDS, struct.unpack("=16i", raw[16:80])))
    require(header["version"] == 1, f"{path}: bad version")
    require(header["bits"] == 64, f"{path}: payload is not 64-bit")
    require(header["kstg"] == 3, f"{path}: not a stage-3 record")

    arrays: dict[str, np.ndarray] = {}
    order: list[str] = []
    off = 80
    while off < len(raw):
        require(off + 32 <= len(raw), f"{path}: truncated array header at {off}")
        # A corrupt name must get a VERDICT, not a decoder traceback: a
        # malformed record that raises an unhandled exception has no status at
        # all, which is how a defeat hides.
        try:
            name = raw[off:off + 16].decode("ascii").rstrip()
        except UnicodeDecodeError:
            require(False, f"{path}: array name at {off} is not ASCII")
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
            arrays[name] = values.reshape((n1, n2), order="F")
        else:
            arrays[name] = values.reshape((n1, n2, n3), order="F")
        order.append(name)
    require(off == len(raw), f"{path}: {len(raw) - off} trailing bytes")
    missing = [n for n in EXPECTED_ARRAYS if n not in arrays]
    require(not missing, f"{path}: record is short of {missing}")
    return {"header": header, "arrays": arrays, "order": order, "path": str(path)}


def _interior(header: dict) -> tuple[slice, slice]:
    """The i-k slice loop's own bounds, as the writer recorded them.

    ``DO_1Dj(0,0)`` runs ``jj = ntsj..ntej`` and ``DO_1Di(0,0)`` runs
    ``ji = ntsi..ntei`` (``do_loop_substitute.h90:117-118``), so the matrix
    buffers are only filled there; everything else is the writer's canonical
    zero.  These bounds are read off the record rather than reconstructed
    from ``nn_hls``, which the record does not carry.
    """
    return (slice(header["ntsi"] - 1, header["ntei"]),
            slice(header["ntsj"] - 1, header["ntej"]))


def nemo_matrix(rec: dict, face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``zwi``/``zwd``/``zws`` rebuilt from the record's own operands.

    Literal transcription of ``dynzdf.F90``: the U face is ``:176-192``
    (inner), ``:193-199`` (surface) and ``:293-296`` (implicit bottom drag);
    the V face is ``:355-368``, ``:369-375`` and ``:465-470``.  The operand
    stencil is the only difference between the two -- ``avm(ji+1,jj)`` and
    ``rCdU_bot(ji+1,jj)`` on U against ``avm(ji,jj+1)`` and
    ``rCdU_bot(ji,jj+1)`` on V.

    Everything is computed on the writer's own ``[ntsi,ntei] x [ntsj,ntej]``
    box and returned on it, because that is the only place NEMO fills the
    matrix.  Computing on the full halo instead would divide by the writer's
    canonical zeros and manufacture NaNs that no NEMO statement produces.
    """
    require(face in ("u", "v"), f"unknown face {face!r}")
    a = rec["arrays"]
    h = rec["header"]
    jpkm1 = h["jpkm1"]
    zdt2 = a["rDt"] * 0.5
    isl, jsl = _interior(h)
    # the +1 neighbour of the same box; it lives in the halo, which is why
    # NEMO can take it without a second exchange
    inb = slice(isl.start + 1, isl.stop + 1) if face == "u" else isl
    jnb = jsl if face == "u" else slice(jsl.start + 1, jsl.stop + 1)

    avm = a["avm"][isl, jsl, :]
    avm_far = a["avm"][inb, jnb, :]
    drg = a["rCdU_bot"][isl, jsl]
    drg_far = a["rCdU_bot"][inb, jnb]
    if face == "u":
        e3, e3w, wmask, mbk = (a["e3u_Kaa"][isl, jsl, :], a["e3uw_Kmm"][isl, jsl, :],
                               a["wumask"][isl, jsl, :], a["mbku"][isl, jsl])
    else:
        e3, e3w, wmask, mbk = (a["e3v_Kaa"][isl, jsl, :], a["e3vw_Kmm"][isl, jsl, :],
                               a["wvmask"][isl, jsl, :], a["mbkv"][isl, jsl])

    ni, nj = avm.shape[0], avm.shape[1]
    zwi = np.zeros((ni, nj, jpkm1))
    zwd = np.zeros((ni, nj, jpkm1))
    zws = np.zeros((ni, nj, jpkm1))

    # inner values, jk = 2..jpkm1 (1-based) -> k = 1..jpkm1-1 (0-based)
    k = slice(1, jpkm1)
    kp = slice(2, jpkm1 + 1)
    zzwi = -zdt2 * (avm_far[:, :, k] + avm[:, :, k]) / (e3[:, :, k] * e3w[:, :, k]) * wmask[:, :, k]
    zzws = (-zdt2 * (avm_far[:, :, kp] + avm[:, :, kp])
            / (e3[:, :, k] * e3w[:, :, kp]) * wmask[:, :, kp])
    zwi[:, :, k] = zzwi
    zws[:, :, k] = zzws
    zwd[:, :, k] = 1.0 - zzwi - zzws

    # surface boundary condition, jk = 1
    zzws = -zdt2 * (avm_far[:, :, 1] + avm[:, :, 1]) / (e3[:, :, 0] * e3w[:, :, 1]) * wmask[:, :, 1]
    zwi[:, :, 0] = 0.0
    zws[:, :, 0] = zzws
    zwd[:, :, 0] = 1.0 - zzws

    # implicit bottom friction: ONE level per column, chosen by mbk
    ix, iy = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    kb = np.clip(mbk.astype(int) - 1, 0, jpkm1 - 1)
    zwd[ix, iy, kb] -= zdt2 * (drg_far + drg) / e3[ix, iy, kb]
    return zwi, zwd, zws


def dumped_matrix_and_rhs(rec: dict, face: str):
    """NEMO's own dumped matrix and the pre-solve column ``dyn_zdf`` sweeps.

    Split out of :func:`nemo_solve` so that legoESM's own sweep can be driven
    on exactly the same operands, with no second copy of the surface-stress
    statement (``dynzdf.F90:328-330``) to drift.  Interior box only, for the
    reason given in ``nemo_matrix``.  Returns the three diagonals sliced to
    ``jpkm1`` rows, the right-hand side, and ``jpkm1``.
    """
    require(face in ("u", "v"), f"unknown face {face!r}")
    a = rec["arrays"]
    h = rec["header"]
    jpkm1 = h["jpkm1"]
    isl, jsl = _interior(h)
    if face == "u":
        zwi, zwd, zws = (a["zwi_u"][isl, jsl, :].copy(), a["zwd_u"][isl, jsl, :].copy(),
                         a["zws_u"][isl, jsl, :].copy())
        pre, e3 = a["uu_Kaa_pre"][isl, jsl, :], a["e3u_Kaa"][isl, jsl, :]
        tau, mask = a["utauU"][isl, jsl], a["umask"][isl, jsl, :]
    else:
        zwi, zwd, zws = (a["zwi_v"][isl, jsl, :].copy(), a["zwd_v"][isl, jsl, :].copy(),
                         a["zws_v"][isl, jsl, :].copy())
        pre, e3 = a["vv_Kaa_pre"][isl, jsl, :], a["e3v_Kaa"][isl, jsl, :]
        tau, mask = a["vtauV"][isl, jsl], a["vmask"][isl, jsl, :]
    x = pre[:, :, :jpkm1].copy()
    # key_RK3 arm: utau only, never the utau/utau_b average (dynzdf.F90:328-330)
    x[:, :, 0] = x[:, :, 0] + a["rDt"] * tau / (e3[:, :, 0] * a["rho0"]) * mask[:, :, 0]
    return zwi[:, :, :jpkm1], zwd[:, :, :jpkm1], zws[:, :, :jpkm1], x, jpkm1


def lego_solve(rec: dict, face: str) -> np.ndarray:
    """legoESM's OWN ordered sweep, on NEMO's dumped matrix and dumped RHS.

    :func:`nemo_solve` is a NumPy transcription and therefore calibrates the
    RECORD; this one drives ``nemo_ordered_tridiagonal_solve`` -- the function
    object BOTH the momentum and the tracer literal solves call -- so a
    difference here is inside legoESM and nowhere else.  It is the momentum
    half of the Rule-12 discharge for any change to that shared sweep, and it
    is the only such measurement the two tank cards have.

    NEMO's momentum back substitution carries no ``tmask`` factor
    (``dynzdf.F90:344-345``), so neither does this; the comparison is against
    the dumped ``uu``/``vv(Kaa)``.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import (
        nemo_ordered_tridiagonal_solve)
    zwi, zwd, zws, x, _ = dumped_matrix_and_rhs(rec, face)
    return np.asarray(nemo_ordered_tridiagonal_solve(
        jnp.asarray(zwi), jnp.asarray(zwd), jnp.asarray(zws), jnp.asarray(x)))


def nemo_solve(rec: dict, face: str) -> np.ndarray:
    """The three recurrences of ``dynzdf.F90:307-346``, on NEMO's own inputs.

    Uses the DUMPED matrix, not the rebuilt one, so a matrix defect and a
    solve defect stay separable.
    """
    zwi, zwd, zws, x, jpkm1 = dumped_matrix_and_rhs(rec, face)
    zwd = zwd.copy()
    for k in range(1, jpkm1):                       # first recurrence
        zwd[:, :, k] = zwd[:, :, k] - zwi[:, :, k] * zws[:, :, k - 1] / zwd[:, :, k - 1]

    x = x.copy()
    for k in range(1, jpkm1):                       # second recurrence
        x[:, :, k] = x[:, :, k] - zwi[:, :, k] / zwd[:, :, k - 1] * x[:, :, k - 1]

    x[:, :, jpkm1 - 1] = x[:, :, jpkm1 - 1] / zwd[:, :, jpkm1 - 1]
    for k in range(jpkm1 - 2, -1, -1):              # third recurrence
        x[:, :, k] = (x[:, :, k] - zws[:, :, k] * x[:, :, k + 1]) / zwd[:, :, k]
    return x


def bit_row(name: str, oracle: np.ndarray, candidate: np.ndarray) -> dict:
    """A row at the campaign's exact bar: value AND bits, both reported."""
    o = np.asarray(oracle, dtype=np.float64).ravel()
    c = np.asarray(candidate, dtype=np.float64).ravel()
    require(o.shape == c.shape, f"{name}: shape {o.shape} vs {c.shape}")
    require(np.isfinite(o).all(), f"{name}: oracle side is not finite")
    require(np.isfinite(c).all(), f"{name}: candidate side is not finite")
    diff = np.abs(o - c)
    absolute_max = float(diff.max()) if diff.size else 0.0
    scale = max(float(np.abs(o).max()) if o.size else 0.0, 1.0)
    bit_unequal = int(np.count_nonzero(o.view(np.uint64) != c.view(np.uint64)))
    normalized = absolute_max / scale
    status = "AT-BAR" if bit_unequal == 0 else (
        "VALUE-AT-BAR" if normalized <= BAR else "DEBT")
    return {"name": name, "n": int(o.size), "bit_unequal": bit_unequal,
            "absolute_max": absolute_max, "normalized_max_abs": normalized,
            "reference_max_abs": scale, "bar": BAR, "status": status}


def run(record: Path, *, plant: bool = False) -> dict:
    require(record.exists(), f"missing {record}")
    rec = read_zdf_matrix(record)
    h = rec["header"]
    a = rec["arrays"]
    isl, jsl = _interior(h)

    if plant:
        # Non-vacuity, TWO operands, because the two families of arm read
        # DIFFERENT ones.  ``avm`` is what the rebuilt matrix is built from,
        # so it moves every ``zdf_matrix`` row -- but the two solve arms read
        # the DUMPED diagonal and never touch ``avm``, so a plant that moved
        # only ``avm`` left them at 0/21120 and proved nothing about them.
        # That hole was found by an independent diff review of round 37 and is
        # closed here by planting the dumped diagonal as well.
        a["avm"] = a["avm"].copy()
        a["avm"][isl, jsl, 1] = np.nextafter(a["avm"][isl, jsl, 1], np.inf)
        for name in ("zwd_u", "zwd_v"):
            a[name] = a[name].copy()
            a[name][isl, jsl, 1] = np.nextafter(a[name][isl, jsl, 1], np.inf)

    rows = []
    for face in ("u", "v"):
        zwi, zwd, zws = nemo_matrix(rec, face)
        for tag, rebuilt in (("zwi", zwi), ("zwd", zwd), ("zws", zws)):
            rows.append(bit_row(
                f"{CASE}.kt1.stage3.zdf_matrix.{tag}_{face}",
                a[f"{tag}_{face}"][isl, jsl, :], rebuilt))
        out = a[f"{'uu' if face == 'u' else 'vv'}_Kaa_out"][isl, jsl, :h["jpkm1"]]
        rows.append(bit_row(
            f"{CASE}.kt1.stage3.zdf_solve.{face}", out, nemo_solve(rec, face)))
        # legoESM's OWN sweep on the same operands -- the momentum half of the
        # Rule-12 discharge for any change to the shared ordered solve.
        rows.append(bit_row(
            f"{CASE}.kt1.stage3.zdf_solve_lego.{face}", out,
            lego_solve(rec, face)))

    at_bar = all(row["status"] == "AT-BAR" for row in rows)
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round29-zdf-matrix-v1",
        "case": CASE,
        "bar": BAR,
        "record": str(record),
        "header": h,
        "arrays_read": rec["order"],
        "planted": plant,
        "calibration_rows": rows,
        "status": "AT-BAR" if at_bar else "DEBT",
        "legoesm_comparison": {
            "status": "UNMEASURED",
            "reason": ("scoring legoESM's stage-3 momentum RHS against "
                       "uu_Krhs_in is what separates dyn_ldf from dyn_zdf, and "
                       "the model exposes no pre-dyn_zdf momentum frame; "
                       "_NEMOWSRK3TestHooks carries expose_momentum_stage and "
                       "expose_stage2_raw_momentum but no pre-implicit "
                       "momentum hook"),
            "rows": [f"{CASE}.kt1.stage3.pre_ldf.{f}" for f in ("u", "v")]
            + [f"{CASE}.kt1.stage3.pre_zdf.{f}" for f in ("u", "v")],
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--plant", action="store_true",
                        help="perturb one operand by one ulp; the gate must go red")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run(args.record, plant=args.plant)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    for row in report["calibration_rows"]:
        print(f"{row['status']:<12} {row['name']:<44} "
              f"bit_unequal {row['bit_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g}")
    print(f"STATUS {report['status']}")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())
