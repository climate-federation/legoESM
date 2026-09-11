#!/usr/bin/env python
"""Stitch NEMO's per-rank kt=1 slope record onto the haloless global domain.

The record is written by ``slopes_patch.py`` at ``tra_ldf``'s own call site,
one file per rank, each beginning with its own
``jpi, jpj, narea, nimpp, njmpp, nn_hls, Nis0, Nie0, Njs0, Nje0, jpiglo,
jpjglo, jpk`` header -- so this reader NEVER guesses a stride or a halo width.

The tile-placement rule is NOT re-implemented here: it is
``nemo_dino_kt1_rankdump.read_rankdump.tile_slices``, the same function the
barotropic substep reader uses, so the two records cannot drift apart.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "nemo_dino_kt1_rankdump"))
from read_rankdump import haloless_shape, tile_slices  # noqa: E402

HEADER_FIELDS = ("jpi", "jpj", "narea", "nimpp", "njmpp", "nn_hls",
                 "Nis0", "Nie0", "Njs0", "Nje0", "jpiglo", "jpjglo", "jpk")

#: Arrays, in the order the patched Fortran writes them.
ARRAYS = ("uslp", "vslp", "wslpi", "wslpj", "ah_wslp2", "akz")


def read_tile(path: str) -> dict:
    raw = np.fromfile(path, dtype=np.uint8)
    nh = len(HEADER_FIELDS)
    h = dict(zip(HEADER_FIELDS,
                 (int(v) for v in raw[:4 * nh].view(np.int32)), strict=True))
    jpi, jpj, jpk = h["jpi"], h["jpj"], h["jpk"]
    expect = 4 * nh + 8 * jpi * jpj * jpk * len(ARRAYS)
    if raw.size != expect:
        raise SystemExit(
            f"{path}: {raw.size} bytes but its own header says "
            f"jpi={jpi} jpj={jpj} jpk={jpk}, i.e. {expect}. The header and "
            "the length disagree -- do not use this file.")
    body = raw[4 * nh:].view(np.float64).reshape(len(ARRAYS), jpk, jpj, jpi)
    out = {"header": h}
    for k, a in zip(ARRAYS, body, strict=True):
        out[k] = np.transpose(a, (1, 2, 0))       # -> (jpj, jpi, jpk)
    return out


def stitch(run_dir: str) -> dict:
    """The N tiles assembled on the haloless global domain, (jpj, jpi, jpk)."""
    paths = sorted(glob.glob(os.path.join(run_dir, "ldfslp_at_traldf_r*.bin")))
    if not paths:
        raise SystemExit(
            f"no ldfslp_at_traldf_r*.bin under {run_dir}. Produce the record "
            "with run.sh in this directory first.")
    out, seen, cover = {}, [], None
    for p in paths:
        t = read_tile(p)
        h = t["header"]
        seen.append(h["narea"])
        if out == {}:
            shape = haloless_shape(h) + (h["jpk"],)
            out = {k: np.full(shape, np.nan) for k in ARRAYS}
            cover = np.zeros(shape[:2], dtype=np.int32)
        elif h["jpk"] != out[ARRAYS[0]].shape[-1]:
            raise SystemExit(f"{p}: jpk={h['jpk']} disagrees with the record")
        gj0, gj1, gi0, gi1, j0, j1, i0, i1 = tile_slices(h, cover.shape, p)
        for k in ARRAYS:
            out[k][gj0:gj1, gi0:gi1, :] = t[k][j0:j1, i0:i1, :]
        cover[gj0:gj1, gi0:gi1] += 1
    if sorted(seen) != list(range(1, len(seen) + 1)):
        raise SystemExit(f"ranks present are {sorted(seen)} -- not 1..N")
    if int((cover != 1).sum()):
        raise SystemExit(
            f"{int((cover == 0).sum())} global cells uncovered and "
            f"{int((cover > 1).sum())} covered twice -- the tile map is wrong")
    # A record whose slopes are identically zero is EXACTLY the defect this
    # whole acquisition exists to repair.  Refuse it loudly rather than let a
    # gate score against nothing and report agreement.
    for k in ("uslp", "vslp", "wslpi", "wslpj"):
        if not np.any(out[k]):
            raise SystemExit(
                f"REFUSING: {k} is identically zero over the whole stitched "
                "record -- that is the same empty dump RUN_FROMREST_KT1 "
                "carries, and scoring against it would manufacture agreement.")
    out["ranks"] = len(seen)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    s = stitch(a.run_dir)
    print(f"{s['ranks']} tiles stitched")
    for k in ARRAYS:
        f = s[k]
        print(f"  {k:10s} shape {f.shape}  max|.| = {np.abs(f).max():.6e}  "
              f"nonzero {int((f != 0).sum())}")
