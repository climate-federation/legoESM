#!/usr/bin/env python
"""Read the rank-tagged DINO kt=1 barotropic record, and prove it is a twin.

The record is produced by ``run.sh`` in this directory.  Two jobs:

``--run-dir DIR --substep N``
    stitch the 16 per-rank tiles of substep ``N`` into one global field per
    variable and return/print them.  Each tile is self-describing: the file
    begins with its own ``jpi, jpj, icycle, jn, narea, nimpp, njmpp, nn_hls,
    Nis0, Nie0, Njs0, Nje0, jpiglo, jpjglo``, so the reader NEVER has to guess
    a stride or a halo width -- which is exactly what made the raced record
    unreadable (its header said ``jpj=28`` while its length was the ``jpj=29``
    record length).

``--twin-check DIR --reference REF``
    every variable of every restart tile in ``DIR`` against ``REF``'s, byte for
    byte.  The instrumentation is additive, so the physics must be identical;
    this is the measurement that says so rather than the claim.

    A differing variable is FATAL when the barotropic ladder CONSUMES it.
    Anything else falls through to ADMITTED and is printed with its name and
    max|delta|, so a difference is never silent -- but a clock or a filename
    written into the file cannot stop the acquisition either.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

#: Header fields, in the order the patched Fortran writes them.  Keep in step
#: with rankdump_patch.py::_HEADER -- the count is asserted below.
HEADER_FIELDS = ("jpi", "jpj", "icycle", "jn", "narea", "nimpp", "njmpp",
                 "nn_hls", "Nis0", "Nie0", "Njs0", "Nje0", "jpiglo", "jpjglo")

#: Arrays, in the order the patched Fortran writes them.
ARRAYS = ("sshn_e", "ssha_e", "zsshp2_e", "un_e", "vn_e", "ua_e", "va_e")

#: Restart variables the barotropic ladder gate reads.  A difference in any of
#: these means the instrumented binary is NOT a twin and the record is void.
CONSUMED = ("sshn", "sshb", "ubtacc_aa", "vbtacc_aa", "ub", "vb",
            "tn", "sn", "un", "vn", "tb", "sb")


def read_tile(path: str) -> dict:
    """One per-(rank, substep) file: its header plus its seven arrays."""
    raw = np.fromfile(path, dtype=np.uint8)
    nh = len(HEADER_FIELDS)
    hdr = raw[:4 * nh].view(np.int32)
    h = dict(zip(HEADER_FIELDS, (int(v) for v in hdr), strict=True))
    jpi, jpj = h["jpi"], h["jpj"]
    expect = 4 * nh + 8 * jpi * jpj * len(ARRAYS)
    if raw.size != expect:
        raise SystemExit(
            f"{path}: {raw.size} bytes but its own header says jpi={jpi} "
            f"jpj={jpj}, i.e. {expect}. The header and the length disagree, "
            "which is the signature of a raced writer -- do not use this file.")
    body = raw[4 * nh:].view(np.float64).reshape(len(ARRAYS), jpj, jpi)
    out = {"header": h}
    for k, a in zip(ARRAYS, body, strict=True):
        out[k] = a                       # (jpj, jpi) == NEMO (j, i)
    return out



def haloless_shape(h: dict) -> tuple[int, int]:
    """The global domain WITHOUT the global halo, from a tile's own header.

    ``(jpjglo - 2*nn_hls, jpiglo - 2*nn_hls)`` -- 199 x 52 on DINO R1 -- so a
    stitched field indexes exactly like ``mesh_mask.nc``.
    """
    hls = h["nn_hls"]
    shape = (h["jpjglo"] - 2 * hls, h["jpiglo"] - 2 * hls)
    if min(shape) < 1:
        raise SystemExit(f"header gives a haloless shape {shape}")
    return shape


def tile_slices(h: dict, shape: tuple[int, int], path: str = "<tile>"):
    """Where one rank's INNER region lands in the haloless global domain.

    NEMO: the global (with-halo) index of local ``ji`` is ``ji + nimpp - 1``,
    so the HALOLESS index is ``ji + nimpp - nn_hls - 2`` (0-based).  Only the
    INNER region is taken, so a halo copy never overwrites its owner's cell.

    Returns ``(gj0, gj1, gi0, gi1, j0, j1, i0, i1)``.  Shared by every
    per-rank record on this branch (the barotropic substeps and the kt=1
    slope dump), so the two cannot drift apart -- one decomposition, one
    placement rule.
    """
    i0, i1 = h["Nis0"] - 1, h["Nie0"]
    j0, j1 = h["Njs0"] - 1, h["Nje0"]
    gi0, gj0 = h["nimpp"] - 1, h["njmpp"] - 1
    ni, nj = i1 - i0, j1 - j0
    if (gi0 < 0 or gj0 < 0 or gi0 + ni > shape[1] or gj0 + nj > shape[0]):
        raise SystemExit(
            f"{path}: rank {h['narea']} inner block [{gj0}:{gj0 + nj}, "
            f"{gi0}:{gi0 + ni}] does not fit the haloless global {shape} -- "
            "the header's nimpp/njmpp/Nis0 do not describe this decomposition")
    return gj0, gj0 + nj, gi0, gi0 + ni, j0, j1, i0, i1

def stitch(run_dir: str, substep: int) -> dict:
    """The 16 tiles of one substep, assembled on the HALOLESS global domain.

    Haloless, i.e. ``(jpjglo - 2*nn_hls, jpiglo - 2*nn_hls)`` -- 199 x 52 on
    DINO R1 -- so the result indexes exactly like ``mesh_mask.nc`` and can be
    compared to it without a second convention.  NEMO's ``jpiglo``/``jpjglo``
    INCLUDE the global halo (``layout.dat``: 56 x 203 for a 52 x 199 domain at
    ``nn_hls = 2``), which no rank's inner region ever covers.
    """
    paths = sorted(glob.glob(os.path.join(
        run_dir, f"substep_r*_s{substep:03d}.bin")))
    if not paths:
        raise SystemExit(
            f"no substep_r*_s{substep:03d}.bin under {run_dir}. Produce the "
            "record with run.sh in this directory first.")
    out, seen, cover = {}, [], None
    for p in paths:
        t = read_tile(p)
        h = t["header"]
        if h["jn"] != substep:
            raise SystemExit(f"{p}: header says substep {h['jn']}, "
                             f"filename says {substep}")
        seen.append(h["narea"])
        # (nn_hls now read inside haloless_shape)
        if out == {}:
            shape = haloless_shape(h)
            out = {k: np.full(shape, np.nan) for k in ARRAYS}
            cover = np.zeros(shape, dtype=np.int32)
        gj0, gj1, gi0, gi1, j0, j1, i0, i1 = tile_slices(h, cover.shape, p)
        for k in ARRAYS:
            out[k][gj0:gj1, gi0:gi1] = t[k][j0:j1, i0:i1]
        cover[gj0:gj1, gi0:gi1] += 1
    if sorted(seen) != list(range(1, len(seen) + 1)):
        raise SystemExit(f"ranks present are {sorted(seen)} -- not 1..N")
    # A tile map that double-covers or leaves a hole silently produces a
    # plausible field, which is worse than a crash (a plausible value is more
    # dangerous than a NaN).
    if int((cover != 1).sum()):
        raise SystemExit(
            f"{int((cover == 0).sum())} global cells uncovered and "
            f"{int((cover > 1).sum())} covered twice -- the tile map is wrong")
    out["ranks"] = len(seen)
    return out


def restart_tiles(d: str, kt: int):
    """The restart tiles for time step ``kt`` under ``d``."""
    return sorted(glob.glob(os.path.join(
        d, "DINO_%08d_restart_*.nc" % kt)))


def restart_steps(d: str):
    """Every ``kt`` for which ``d`` holds restart tiles."""
    out = set()
    for f in glob.glob(os.path.join(d, "DINO_*_restart_*.nc")):
        tok = os.path.basename(f).split("_")
        if len(tok) > 1 and tok[1].isdigit():
            out.add(int(tok[1]))
    return sorted(out)


def twin_check(run_dir: str, reference: str, kt: int = 1) -> int:
    import netCDF4 as nc
    fatal = admitted = 0
    pats = restart_tiles(run_dir, kt)
    if not pats:
        have = restart_steps(run_dir)
        raise SystemExit(
            f"no kt={kt} restart tiles under {run_dir}; it holds restarts at "
            f"{have or 'no step at all'}.\n"
            "NEMO writes a restart only when MOD(kt-1, nn_stock) == 0 sets "
            "nitrst and kt reaches it (restart.f90:112, :121), so a record "
            "acquired with nn_stock = N carries ONLY the kt = N restart.  A "
            "record whose kt=1 restart was never written cannot be "
            "twin-checked at kt=1 -- use --record-check, which scores what "
            "the record DOES contain and says what it does not prove.")
    print(f"TWIN CHECK  {len(pats)} tiles at kt={kt}, every variable, byte "
          "for byte")
    for p in pats:
        q = os.path.join(reference, os.path.basename(p))
        if not os.path.exists(q):
            raise SystemExit(
                f"reference tile missing: {q} (the reference has restarts at "
                f"{restart_steps(reference)})")
        a, b = nc.Dataset(p), nc.Dataset(q)
        if set(a.variables) != set(b.variables):
            print(f"  FATAL {os.path.basename(p)}: variable sets differ")
            fatal += 1
        for v in sorted(set(a.variables) & set(b.variables)):
            x = np.asarray(a[v][:])
            y = np.asarray(b[v][:])
            if x.shape == y.shape and np.array_equal(x, y):
                continue
            try:
                mx = float(np.max(np.abs(x.astype(np.float64)
                                         - y.astype(np.float64))))
            except (TypeError, ValueError):
                mx = float("nan")
            if v in CONSUMED:
                print(f"  FATAL {os.path.basename(p)}:{v} max|d|={mx:.3e} "
                      "-- the ladder reads this; the record is void")
                fatal += 1
            else:
                print(f"  ADMITTED {os.path.basename(p)}:{v} max|d|={mx:.3e} "
                      "-- not consumed by the ladder")
                admitted += 1
        a.close()
        b.close()
    print(f"  {fatal} fatal, {admitted} admitted")
    if fatal:
        print("TWIN CHECK FAIL: the instrumented binary is not a twin")
        return 1
    print("TWIN CHECK PASS: the instrumentation changed no consumed field")
    return 0


def _namelist_values(path: str) -> dict:
    """Every ``key = value`` in a NEMO namelist, comments stripped."""
    out = {}
    for line in open(path):
        line = line.split("!", 1)[0]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip().rstrip(",").strip()
    return out


def record_check(run_dir: str, reference: str, kt: int, allow) -> int:
    """The post-check for a record whose kt is NOT the reference's.

    ``twin_check`` proves a record is a twin by comparing the SAME restart in
    both directories.  That is impossible for the kt=2 record: with
    ``nn_stock = 2`` NEMO never writes a kt=1 restart (``restart.f90:112``
    sets ``nitrst = kt + nn_stock - 1 = 2`` at kt=1 and ``:121`` writes only
    at ``nitrst``), and the ``.bin`` debug streams cannot stand in for it --
    every rank opens them under one filename with no rank in it
    (``stpmlf.F90:737``), so their bytes are whichever rank closed last.
    MEASURED: 117 of 170 of those streams differ between two records that are
    identical by construction.

    So this checks what the record CAN be held to, and prints what it cannot:

      1. the kt restart is present and covers every rank in ``layout.dat``;
      2. the namelist differs from the reference's ONLY in ``allow``.

    What underwrites feeding the reference's kt=1 restart to the gate is then
    a separate, MEASURED fact: two independent runs of this configuration
    wrote bit-identical kt=1 restarts (``--twin-check`` on the kt=1 slopes
    record against ``RUN_FROMREST_KT1``: 16 tiles, every variable, 0 fatal,
    0 admitted).  Determinism, not assumption.
    """
    tiles = restart_tiles(run_dir, kt)
    if not tiles:
        raise SystemExit(
            f"no kt={kt} restart tiles under {run_dir}; it holds restarts at "
            f"{restart_steps(run_dir) or 'no step at all'}")
    # NEMO's own rank count, read from the file rather than counted: the
    # header line names jpnij and the next line carries it.
    lay = os.path.join(run_dir, "layout.dat")
    want = None
    if os.path.exists(lay):
        L = open(lay).read().splitlines()
        for i, ln in enumerate(L):
            if ln.split()[:1] == ["jpnij"] and i + 1 < len(L):
                want = int(L[i + 1].split()[0])
                break
    print(f"RECORD CHECK  kt={kt}: {len(tiles)} restart tiles"
          + (f", layout.dat lists {want} ranks" if want else ""))
    bad = 0
    if want is None:
        print("  FATAL: no readable layout.dat, so the tile count cannot be "
              "checked against NEMO's own jpnij -- a 1-of-16 record would "
              "pass this check silently")
        bad += 1
    if want and len(tiles) != want:
        print(f"  FATAL: {len(tiles)} tiles for {want} ranks -- the record "
              "is incomplete and any stitch of it would have holes")
        bad += 1
    a = _namelist_values(os.path.join(run_dir, "namelist_cfg"))
    b = _namelist_values(os.path.join(reference, "namelist_cfg"))
    keys = sorted(set(a) | set(b))
    delta = [k for k in keys if a.get(k) != b.get(k)]
    print(f"  namelist delta vs {reference}: "
          + (", ".join(f"{k} {b.get(k)!r} -> {a.get(k)!r}" for k in delta)
             or "none"))
    for k in delta:
        if k not in allow:
            print(f"  FATAL: {k} differs and is not in the allowed set "
                  f"{sorted(allow)} -- this record is not the reference's "
                  "trajectory with a different end step")
            bad += 1
    print("  NOT PROVEN BY THIS CHECK: that the kt=1 step of this run matched "
          "the certified one, because this run never wrote a kt=1 restart "
          "(restart.f90:112, :121 with nn_stock=%d)." % int(
              a.get("nn_stock", -1) if str(a.get("nn_stock", "")).lstrip("-")
              .isdigit() else -1))
    print("  What carries it instead: the namelist delta above, and the "
          "MEASURED bit-reproducibility of this configuration's kt=1 restart "
          "across independent runs (--twin-check on the kt=1 slopes record).")
    print("RECORD CHECK " + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


def main() -> int:
    assert len(HEADER_FIELDS) == 14, "header field list drifted from the patch"
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--substep", type=int, default=1)
    ap.add_argument("--twin-check")
    ap.add_argument("--record-check")
    ap.add_argument("--reference")
    ap.add_argument("--kt", type=int, default=1,
                    help="the restart time step to check (default 1)")
    ap.add_argument("--allow", default="",
                    help="comma-separated namelist keys allowed to differ "
                         "from the reference, for --record-check")
    a = ap.parse_args()
    if a.twin_check:
        if not a.reference:
            raise SystemExit("--twin-check needs --reference")
        return twin_check(a.twin_check, a.reference, a.kt)
    if a.record_check:
        if not a.reference:
            raise SystemExit("--record-check needs --reference")
        return record_check(a.record_check, a.reference, a.kt,
                            {k for k in a.allow.split(",") if k})
    if not a.run_dir:
        raise SystemExit(__doc__)
    s = stitch(a.run_dir, a.substep)
    print(f"substep {a.substep}: {s['ranks']} tiles stitched")
    for k in ARRAYS:
        f = s[k]
        print(f"  {k:9s} shape {f.shape}  min {f.min():+.6e}  "
              f"max {f.max():+.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
