#!/usr/bin/env python
"""Face maps of the FV3 duo-grid one-step state, and the cube-edge check.

Input is the .npz written by ``full_step_oracle_parity.py --save-fields``:
the MAPPED port and oracle planes, i.e. the very arrays the parity gate
scored, after the face permutation and dihedral.  Plotting those and not a
freshly assembled state is deliberate -- a second rendering path is a
second chance to be wrong about the orientation, and the point here is to
look at what was scored.

Two products:

1. Per field, a 2x6 figure -- the port's own field on all six faces, and
   the port-minus-oracle difference on the same panels and the same
   orientation.

2. A cube-edge artifact check, printed and gated.  A "visual artifact on a
   cube edge" is grid-scale structure that appears at a panel boundary and
   not in the panel interior.  The metric is the mean magnitude of the
   second difference of the field (a high-pass; a smooth feature, however
   sharp, contributes little), evaluated on the ring of cells within
   ``--edge-width`` of the panel boundary and on the interior, as a ratio.
   That ratio is NOT compared against an invented threshold -- it is
   compared against the ORACLE's ratio on the same field and the same
   cells.  The reference has whatever genuine edge structure the geometry
   imposes; only an EXCESS over it is the port's own artifact.  A metric
   scored against an absolute number could not tell the two apart.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def high_pass(plane: np.ndarray) -> np.ndarray:
    """|second difference| summed over both horizontal directions.

    Grid-scale (2*dx) structure survives this; a smooth gradient does not.
    Returned on the interior-shrunk index space (i-1, j-1 offsets), which
    the caller accounts for when it splits edge from interior.
    """
    d2i = plane[2:, 1:-1] - 2.0 * plane[1:-1, 1:-1] + plane[:-2, 1:-1]
    d2j = plane[1:-1, 2:] - 2.0 * plane[1:-1, 1:-1] + plane[1:-1, :-2]
    return np.abs(d2i) + np.abs(d2j)


def edge_interior_ratio(plane: np.ndarray, width: int) -> tuple:
    """(edge mean, interior mean, ratio) of the high-pass on one panel."""
    hp = high_pass(plane)
    ni, nj = hp.shape
    if min(ni, nj) <= 2 * width + 2:
        raise SystemExit(
            f"panel {plane.shape} too small for --edge-width {width}: the "
            f"edge ring and the interior would overlap, and the ratio "
            f"would compare a set against itself")
    mask = np.zeros((ni, nj), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    e = float(hp[mask].mean())
    i = float(hp[~mask].mean())
    return e, i, (e / i if i > 0 else float("inf"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--level", type=int, default=-1,
                    help="vertical index of the plotted horizontal slice "
                         "(default -1, the bottom model level). The edge "
                         "check runs over EVERY level regardless -- an "
                         "artifact confined to one level would otherwise "
                         "be invisible to a single-slice plot.")
    ap.add_argument("--edge-width", type=int, default=3,
                    help="width in cells of the panel-boundary ring")
    ap.add_argument("--max-edge-excess", type=float, default=None,
                    help="gate: exit 1 if the port's edge/interior "
                         "high-pass ratio exceeds the oracle's by more "
                         "than this factor on any field, face and level")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.npz)
    fields = [str(f) for f in z["fields"]]
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- product 1: the maps -------------------------------------------
    written = []
    for f in fields:
        port = [z[f"port_f{pf+1}_{f}"] for pf in range(6)]
        orc = [z[f"oracle_f{pf+1}_{f}"] for pf in range(6)]
        k = args.level
        fig, ax = plt.subplots(2, 6, figsize=(21, 7))
        vmin = min(float(p[..., k].min()) for p in port)
        vmax = max(float(p[..., k].max()) for p in port)
        dmax = max(float(np.abs(p[..., k] - o[..., k]).max())
                   for p, o in zip(port, orc))
        for pf in range(6):
            a = ax[0, pf].pcolormesh(port[pf][..., k].T, vmin=vmin,
                                     vmax=vmax, cmap="viridis")
            ax[0, pf].set_title(f"face {pf+1}", fontsize=9)
            ax[0, pf].set_aspect("equal")
            d = np.abs(port[pf][..., k] - orc[pf][..., k])
            b = ax[1, pf].pcolormesh(d.T, vmin=0.0,
                                     vmax=(dmax if dmax > 0 else 1.0),
                                     cmap="magma")
            ax[1, pf].set_aspect("equal")
            for a_ in (ax[0, pf], ax[1, pf]):
                a_.set_xticks([])
                a_.set_yticks([])
        fig.colorbar(a, ax=ax[0, :].tolist(), fraction=0.02)
        fig.colorbar(b, ax=ax[1, :].tolist(), fraction=0.02)
        ax[0, 0].set_ylabel("port", fontsize=11)
        ax[1, 0].set_ylabel("|port - oracle|", fontsize=11)
        fig.suptitle(f"FV3 duo one step: {f}  (level {k}; "
                     f"|d|max = {dmax:.3e})")
        out = os.path.join(args.out_dir, f"fv3_duo_{f}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"wrote {out}")

    # ---- product 2: the cube-edge check --------------------------------
    print(f"\ncube-edge artifact check (high-pass mean in the "
          f"{args.edge_width}-cell panel-boundary ring / interior), port "
          f"against the ORACLE's ratio on the same cells:")
    worst = 0.0
    worst_where = None
    for f in fields:
        for pf in range(6):
            port = z[f"port_f{pf+1}_{f}"]
            orc = z[f"oracle_f{pf+1}_{f}"]
            nk = port.shape[2]
            for k in range(nk):
                pe, pi, pr = edge_interior_ratio(port[..., k],
                                                 args.edge_width)
                oe, oi, orr = edge_interior_ratio(orc[..., k],
                                                  args.edge_width)
                if not np.isfinite(pr) or not np.isfinite(orr):
                    raise SystemExit(
                        f"non-finite edge ratio at {f} face {pf+1} level "
                        f"{k}: interior high-pass is exactly zero, so the "
                        f"ratio carries no information and must not be "
                        f"reported as a pass")
                excess = pr / orr
                if excess > worst:
                    worst = excess
                    worst_where = (f, pf + 1, k, pr, orr)
        f_rows = [(pf, edge_interior_ratio(z[f"port_f{pf+1}_{f}"][..., -1],
                                           args.edge_width),
                   edge_interior_ratio(z[f"oracle_f{pf+1}_{f}"][..., -1],
                                       args.edge_width))
                  for pf in range(6)]
        print(f"  {f:5s} " + "  ".join(
            f"f{pf+1}:{p[2]:.3f}/{o[2]:.3f}" for pf, p, o in f_rows))

    f, pf, k, pr, orr = worst_where
    print(f"\nWORST port/oracle edge-ratio excess over every field, face "
          f"and level: {worst:.4f}x at {f} face {pf} level {k} "
          f"(port {pr:.4f}, oracle {orr:.4f})")
    print("  (1.0 = the port has exactly the reference's edge structure; "
          "the bottom row of each figure is the residual that any excess "
          "would have to come from)")

    if args.max_edge_excess is not None and worst > args.max_edge_excess:
        raise SystemExit(
            f"EDGE GATE FAILED: {worst:.4f}x > {args.max_edge_excess}")
    if args.max_edge_excess is not None:
        print(f"EDGE GATE PASSED ({worst:.4f}x <= "
              f"{args.max_edge_excess})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
