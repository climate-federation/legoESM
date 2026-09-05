"""Infer FESOM2's level-assignment rule from a shipped mesh, then apply it to
another vertical ladder.

FESOM's Fortran mesh setup derives ``nlvls.out`` (levels per node) and
``elvls.out`` (levels per element) from ``aux3d.out`` (nl, zbar[nl], node
depth[nod2D]).  No FESOM build is available here, so the rule is INFERRED and
must reproduce the shipped 47-level files EXACTLY (every node, every element)
before it is trusted on the 75-level NEMO ladder.

Candidate rules tested (FESOM2 oce_mesh_setup conventions):
  element: depth_e = min over its 3 nodes of |depth| (shallowest vertex);
           elvls_e = number of zbar interfaces with |zbar| <= depth_e,
           floored at a minimum level count (3 or 4 tested);
  node:    nlvls_n = max over incident elements of elvls_e.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_raw(d: Path):
    with open(d / "aux3d.out") as f:
        nl = int(f.readline())
        zbar = np.array([float(f.readline()) for _ in range(nl)])
        depth = np.loadtxt(f)
    elem = np.loadtxt(d / "elem2d.out", dtype=np.int64, skiprows=1) - 1
    nlvls = np.loadtxt(d / "nlvls.out", dtype=np.int64)
    elvls = np.loadtxt(d / "elvls.out", dtype=np.int64)
    return nl, zbar, depth, elem, nlvls, elvls


def apply_rule(zbar, depth, elem, *, agg, floor, cmp):
    zb = np.abs(zbar); dep = np.abs(depth)
    de = agg(dep[elem], axis=1)
    if cmp == "le":
        elv = (zb[None, :] <= de[:, None]).sum(axis=1)
    else:
        elv = (zb[None, :] < de[:, None]).sum(axis=1)
    elv = np.maximum(elv, floor)
    nlv = np.zeros(dep.size, dtype=np.int64)
    for j in range(3):
        np.maximum.at(nlv, elem[:, j], elv)
    return elv, nlv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--nemo-domain-cfg", default=None,
                   help="write a 76-interface NEMO ladder version of aux3d/nlvls/elvls to --out")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    nl, zbar, depth, elem, nlvls, elvls = read_raw(Path(a.raw))
    print(f"mesh: nod2D={depth.size} elem2D={elem.shape[0]} nl={nl} zbar[0..3]={zbar[:4]} "
          f"nlvls range {nlvls.min()}..{nlvls.max()} elvls range {elvls.min()}..{elvls.max()}")
    best = None
    for name, agg in (("min", np.min), ("max", np.max), ("mean", np.mean)):
        for floor in (1, 2, 3, 4):
            for cmp in ("le", "lt"):
                elv, nlv = apply_rule(zbar, depth, elem, agg=agg, floor=floor, cmp=cmp)
                ne = int((elv != elvls).sum()); nn = int((nlv != nlvls).sum())
                print(f"  rule elem={name:4s} floor={floor} cmp={cmp}: elvls mismatches {ne:7d}  nlvls mismatches {nn:7d}")
                if best is None or ne + nn < best[0]:
                    best = (ne + nn, name, floor, cmp)
    print(f"best rule: elem-depth={best[1]} floor={best[2]} cmp={best[3]} total mismatches {best[0]}")
    if a.nemo_domain_cfg and a.out:
        import netCDF4
        with netCDF4.Dataset(a.nemo_domain_cfg) as ds:
            e3t = np.squeeze(np.asarray(ds["e3t_1d"][:], dtype=np.float64))
        zb_nemo = -np.concatenate([[0.0], np.cumsum(e3t)])
        agg = {"min": np.min, "max": np.max, "mean": np.mean}[best[1]]
        elv, nlv = apply_rule(zb_nemo, depth, elem, agg=agg, floor=best[2], cmp=best[3])
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "aux3d.out", "w") as f:
            f.write(f"{zb_nemo.size}\n")
            for z in zb_nemo:
                f.write(f"{z:.6f}\n")
            for dd in depth:
                f.write(f"{dd:.6f}\n")
        np.savetxt(out / "nlvls.out", nlv, fmt="%d"); np.savetxt(out / "elvls.out", elv, fmt="%d")
        print(f"NEMO ladder: nl={zb_nemo.size} bottom {zb_nemo[-1]:.1f} m; nlvls range {nlv.min()}..{nlv.max()}, "
              f"elvls range {elv.min()}..{elv.max()}; written to {out}")
        print("NOTE: only the level files are regenerated; copy nod2d/elem2d/edges/edge_tri from the raw mesh "
              "(the raw-derived edge files do not depend on the vertical ladder).")


if __name__ == "__main__":
    main()
