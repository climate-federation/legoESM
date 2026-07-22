#!/usr/bin/env python
"""Diff the FMS mpp_get_boundary sentinel oracle against the legoESM
edge-partner ports.

Consumes fms_sentinel_table.txt (fms_sentinel_boundary.F90 output: the
REAL FMS BGRID_NE + CGRID_NE boundary buffers on the genuine 6-tile
cube with index-coded fields) and reproduces every buffer slot through
`_bgrid_edge_partner` / `_cgrid_edge_partner`.  Any pairing, position,
or vector-sign discrepancy prints as a MISMATCH line with both codings
decoded — this is the certification the two hand-derived probes could
not provide (2026-07-21 memory: never hand-derive seam conventions).

Coding: v = s*(2e6*t + 1e6*c + 1e3*i + j); c=0 fieldx, c=1 fieldy.

Usage: compare_fms_sentinel.py --table fms_sentinel_table.txt [--n 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def decode(v: float) -> tuple:
    s = 1 if v >= 0 else -1
    a = abs(v)
    t = int(a // 2e6)
    c = int((a - 2e6 * t) // 1e6)
    i = int((a - 2e6 * t - 1e6 * c) // 1e3)
    j = int(round(a - 2e6 * t - 1e6 * c - 1e3 * i))
    return s, t, c, i, j


def code(t: int, c: int, i: int, j: int) -> float:
    return 2e6 * t + 1e6 * c + 1e3 * i + j


def parse_table(path: str) -> dict:
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) != 5 or p[0] not in ("BG", "CG"):
            continue
        tile = int(p[1].lstrip("t"))
        out[(p[0], tile, p[2], int(p[3]))] = float(p[4])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.grids.fv3_native_gridstruct import (
        _bgrid_edge_partner,
        _cgrid_edge_partner,
        neighbor_tiles,
    )

    n = args.n
    npx = n + 1
    ng = 3

    # coded per-face arrays in OUR layouts
    xb6 = [np.zeros((npx, npx)) for _ in range(6)]
    yb6 = [np.zeros((npx, npx)) for _ in range(6)]
    fcx6 = [np.zeros((npx, n)) for _ in range(6)]
    fcy6 = [np.zeros((n, npx)) for _ in range(6)]
    for t in range(1, 7):
        for fi in range(1, npx + 1):
            for fj in range(1, npx + 1):
                xb6[t - 1][fi - 1, fj - 1] = code(t, 0, fi, fj)
                yb6[t - 1][fi - 1, fj - 1] = code(t, 1, fi, fj)
        for fi in range(1, npx + 1):
            for fj in range(1, n + 1):
                fcx6[t - 1][fi - 1, fj - 1] = code(t, 0, fi, fj)
        for fi in range(1, n + 1):
            for fj in range(1, npx + 1):
                fcy6[t - 1][fi - 1, fj - 1] = code(t, 1, fi, fj)

    oracle = parse_table(args.table)
    n_hit = n_miss = n_skip = 0
    for (grid, tile, edge, slot), want in sorted(oracle.items()):
        if want < -9e9:   # buffer slot FMS never filled
            n_skip += 1
            continue
        nw, ne, ns, nn = neighbor_tiles(tile)
        try:
            if grid == "BG":
                if edge in ("W", "E"):
                    fi = 1 if edge == "W" else npx
                    fj = slot
                    n_src = nw if edge == "W" else ne
                    got = _bgrid_edge_partner(
                        xb6, yb6, tile, 2 * fi - 1, 2 * fj - 1, "i", n, ng,
                        n_src)
                else:
                    fi = slot
                    fj = 1 if edge == "S" else npx
                    n_src = ns if edge == "S" else nn
                    got = _bgrid_edge_partner(
                        xb6, yb6, tile, 2 * fi - 1, 2 * fj - 1, "j", n, ng,
                        n_src)
            else:
                if edge in ("W", "E"):
                    fi = 1 if edge == "W" else npx
                    fj = slot
                    n_src = nw if edge == "W" else ne
                    got = _cgrid_edge_partner(
                        fcx6, fcy6, tile, 2 * fi - 1, 2 * fj, "i", n, ng,
                        n_src)
                else:
                    fi = slot
                    fj = 1 if edge == "S" else npx
                    n_src = ns if edge == "S" else nn
                    got = _cgrid_edge_partner(
                        fcx6, fcy6, tile, 2 * fi, 2 * fj - 1, "j", n, ng,
                        n_src)
        except Exception as e:   # port raises = slot unmapped
            print(f"PORT-ERROR {grid} t{tile} {edge} {slot}: {e}")
            n_miss += 1
            continue
        if abs(got - want) < 1e-6:
            n_hit += 1
        else:
            n_miss += 1
            print(f"MISMATCH {grid} t{tile} {edge} slot {slot}: "
                  f"fms={want:.0f} {decode(want)} "
                  f"ours={got:.0f} {decode(got)}")
    print(f"hits {n_hit}  mismatches {n_miss}  unfilled-skipped {n_skip}")
    sys.exit(0 if n_miss == 0 else 1)


if __name__ == "__main__":
    main()
