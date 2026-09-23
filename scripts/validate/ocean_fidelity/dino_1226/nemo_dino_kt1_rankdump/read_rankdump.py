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
        hls = h["nn_hls"]
        if out == {}:
            shape = (h["jpjglo"] - 2 * hls, h["jpiglo"] - 2 * hls)
            if min(shape) < 1:
                raise SystemExit(f"{p}: header gives a haloless shape {shape}")
            out = {k: np.full(shape, np.nan) for k in ARRAYS}
            cover = np.zeros(shape, dtype=np.int32)
        # NEMO: the global (with-halo) index of local ji is ji + nimpp - 1, so
        # the HALOLESS index is ji + nimpp - nn_hls - 2 (0-based).  Take the
        # INNER region only, so a halo copy never overwrites its owner's cell.
        i0, i1 = h["Nis0"] - 1, h["Nie0"]
        j0, j1 = h["Njs0"] - 1, h["Nje0"]
        gi0, gj0 = h["nimpp"] - 1, h["njmpp"] - 1
        ni, nj = i1 - i0, j1 - j0
        if (gi0 < 0 or gj0 < 0 or gi0 + ni > cover.shape[1]
                or gj0 + nj > cover.shape[0]):
            raise SystemExit(
                f"{p}: rank {h['narea']} inner block [{gj0}:{gj0 + nj}, "
                f"{gi0}:{gi0 + ni}] does not fit the haloless global "
                f"{cover.shape} -- the header's nimpp/njmpp/Nis0 do not "
                "describe this decomposition")
        for k in ARRAYS:
            out[k][gj0:gj0 + nj, gi0:gi0 + ni] = t[k][j0:j1, i0:i1]
        cover[gj0:gj0 + nj, gi0:gi0 + ni] += 1
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


def twin_check(run_dir: str, reference: str) -> int:
    import netCDF4 as nc
    fatal = admitted = 0
    pats = sorted(glob.glob(os.path.join(
        run_dir, "DINO_00000001_restart_*.nc")))
    if not pats:
        raise SystemExit(f"no restart tiles under {run_dir}")
    print(f"TWIN CHECK  {len(pats)} tiles, every variable, byte for byte")
    for p in pats:
        q = os.path.join(reference, os.path.basename(p))
        if not os.path.exists(q):
            raise SystemExit(f"reference tile missing: {q}")
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


def main() -> int:
    assert len(HEADER_FIELDS) == 14, "header field list drifted from the patch"
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--substep", type=int, default=1)
    ap.add_argument("--twin-check")
    ap.add_argument("--reference")
    a = ap.parse_args()
    if a.twin_check:
        if not a.reference:
            raise SystemExit("--twin-check needs --reference")
        return twin_check(a.twin_check, a.reference)
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
