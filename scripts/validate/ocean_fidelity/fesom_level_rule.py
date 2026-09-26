"""FESOM2's level assignment (fvom_init.F90 ``find_levels``), ported to numpy,
verified against the shipped CORE2 level files, then applied to another
vertical ladder (NEMO's 75 levels).

The Fortran (FESOM2 ``src/fvom_init.F90``, subroutine ``find_levels``):
  1. node depth x (negative down), clamped: ``if (x > zbar(thers)) x = zbar(thers)``
     with ``thers_zbar_lev = 5`` (no column shallower than interface 5);
  2. element depth = MEAN of its 3 node depths (``which_depth_n2e='mean'``);
  3. ``nlevels(e)`` = first 1-based layer index nz (1..nl-1) whose mid-depth
     ``Z(nz) = 0.5*(zbar(nz)+zbar(nz+1)) < dmean``; none -> nl; floored at 5;
  4. isolated-cell elimination, per level nz = 6..nl, iterate to convergence: a
     cell with ``nlevels >= nz`` and fewer than 2 edge-neighbours with
     ``nlevels >= nz`` is shallowed to nz-1 (or, if nz-1 < 5, its neighbours are
     deepened to nz);
  5. ``nlevels_nod2D(n)`` = max over incident elements.

Self-check: the port must reproduce the shipped ``elvls.out``/``nlvls.out``
(the shipped files were regenerated 2026-07-03 by FESOM itself, so a
mismatch of more than a few cells means the port is wrong, not the mesh).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

THERS_ZBAR_LEV = 5


def read_raw(d: Path):
    with open(d / "aux3d.out") as f:
        nl = int(f.readline())
        zbar = np.array([float(f.readline()) for _ in range(nl)])
        depth = np.loadtxt(f)
    elem = np.loadtxt(d / "elem2d.out", dtype=np.int64, skiprows=1) - 1
    nlvls = np.loadtxt(d / "nlvls.out", dtype=np.int64)
    elvls = np.loadtxt(d / "elvls.out", dtype=np.int64)
    edge_tri = np.loadtxt(d / "edge_tri.out", dtype=np.int64) - 1   # (nEdges, 2), -1 = boundary
    return nl, zbar, depth, elem, nlvls, elvls, edge_tri


def elem_neighbors(n_elem, edge_tri):
    nb = np.full((n_elem, 3), -1, dtype=np.int64)
    cnt = np.zeros(n_elem, dtype=np.int64)
    for a, b in edge_tri:
        if a >= 0 and b >= 0:
            nb[a, cnt[a]] = b; cnt[a] += 1
            nb[b, cnt[b]] = a; cnt[b] += 1
    return nb


def find_levels(zbar, depth, elem, nb, thers=THERS_ZBAR_LEV):
    """``thers`` = FESOM's ``thers_zbar_lev`` (1-based interface index): no
    column shallower than ``zbar[thers-1]`` and at least ``thers`` levels."""
    zbar = -np.abs(zbar); nl = zbar.size
    x = -np.abs(depth)
    x = np.minimum(x, zbar[thers - 1])                    # step 1 (zbar negative: x > zbar(thers) -> clamp)
    Z = 0.5 * (zbar[:-1] + zbar[1:])                       # (nl-1,)
    dmean = x[elem].sum(axis=1) / 3.0                      # step 2
    below = Z[None, :] < dmean[:, None]                    # step 3: first nz with Z(nz) < dmean
    nlev = np.where(below.any(axis=1), below.argmax(axis=1) + 1, nl)
    nlev = np.where(dmean >= 0, thers, nlev)
    nlev = np.maximum(nlev, thers)
    # step 4: isolated-cell elimination, per level, iterate (vectorised sweep
    # == one Fortran pass; repeated until no change, like the do-while)
    for nz in range(thers + 1, nl + 1):
        for _ in range(1000):
            active = nlev >= nz
            nb_open = np.zeros(nlev.size, dtype=np.int64)
            for j in range(3):
                nbj = nb[:, j]
                nb_open += (nbj >= 0) & (nlev[np.maximum(nbj, 0)] >= nz)
            iso = active & (nb_open < 2)
            if not iso.any():
                break
            if nz - 1 < thers:
                for j in range(3):
                    nbj = nb[iso, j]; nbj = nbj[nbj >= 0]
                    nlev[nbj] = np.maximum(nlev[nbj], nz)
            else:
                nlev[iso] = nz - 1
    nlev_nod = np.zeros(depth.size, dtype=np.int64)
    for j in range(3):
        np.maximum.at(nlev_nod, elem[:, j], nlev)
    return nlev, nlev_nod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--nemo-domain-cfg", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--min-depth-m", type=float, default=None,
                   help="NEMO ladder only: minimum column depth [m] (the "
                        "interface nearest this depth becomes FESOM's "
                        "thers_zbar_lev); default = FESOM's own 5th interface "
                        "(4.5 m on the NEMO ladder, 30 m on FESOM's).")
    a = p.parse_args()
    raw = Path(a.raw)
    nl, zbar, depth, elem, nlvls, elvls, edge_tri = read_raw(raw)
    nb = elem_neighbors(elem.shape[0], edge_tri)
    print(f"mesh: nod2D={depth.size} elem2D={elem.shape[0]} nl={nl}; shipped elvls {elvls.min()}..{elvls.max()} "
          f"nlvls {nlvls.min()}..{nlvls.max()}")
    elv, nlv = find_levels(zbar, depth, elem, nb)
    ne = int((elv != elvls).sum()); nn = int((nlv != nlvls).sum())
    print(f"PORT vs shipped (47-level ladder): elvls mismatches {ne}/{elv.size}, nlvls mismatches {nn}/{nlv.size}")
    if ne:
        idx = np.flatnonzero(elv != elvls)[:10]
        print("  first element mismatches (port, shipped):", [(int(i), int(elv[i]), int(elvls[i])) for i in idx])
    if a.nemo_domain_cfg and a.out:
        if ne + nn > 8:
            raise SystemExit("port does not reproduce the shipped ladder; refusing to generate a new one")
        import netCDF4
        with netCDF4.Dataset(a.nemo_domain_cfg) as ds:
            e3t = np.squeeze(np.asarray(ds["e3t_1d"][:], dtype=np.float64))
        zb_nemo = -np.concatenate([[0.0], np.cumsum(e3t)])
        thers = THERS_ZBAR_LEV
        if a.min_depth_m is not None:
            thers = int(np.argmin(np.abs(np.abs(zb_nemo) - a.min_depth_m))) + 1
            print(f"min column depth {a.min_depth_m} m -> thers_zbar_lev {thers} "
                  f"(interface at {abs(zb_nemo[thers - 1]):.2f} m)")
        elv2, nlv2 = find_levels(zb_nemo, depth, elem, nb, thers=thers)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "aux3d.out", "w") as f:
            f.write(f"{zb_nemo.size}\n")
            for z in zb_nemo:
                f.write(f"{z:.6f}\n")
            for dd in depth:
                f.write(f"{dd:.6f}\n")
        np.savetxt(out / "nlvls.out", nlv2, fmt="%d"); np.savetxt(out / "elvls.out", elv2, fmt="%d")
        for fn in ("nod2d.out", "elem2d.out", "edges.out", "edge_tri.out", "edgenum.out"):
            if (raw / fn).exists():
                (out / fn).write_bytes((raw / fn).read_bytes())
        print(f"NEMO ladder: nl={zb_nemo.size} (bottom {zb_nemo[-1]:.1f} m); elvls {elv2.min()}..{elv2.max()}, "
              f"nlvls {nlv2.min()}..{nlv2.max()}; min column depth clamp = zbar[{thers - 1}] = {zb_nemo[thers - 1]:.2f} m; written to {out}")


if __name__ == "__main__":
    main()
