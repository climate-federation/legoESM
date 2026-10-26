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

2. Two numbers about the panel boundary, both report-only, and neither
   of them a cross-seam continuity test.  The saved planes are the
   COMPUTE WINDOW: no halo and no neighbouring panel's values are in
   them, so nothing here can form a difference ACROSS a cube edge.  What
   is measured is the boundary STRIP of each panel, one side only.

   (a) In-panel boundary-strip roughness -- the mean magnitude of the
       second difference of the field on the strip within ``--edge-width``
       of the panel border, over the same on the interior, for the port
       and for the oracle.  Reported as the port's ratio over the
       oracle's.  This is DESCRIPTIVE: it is blind to any error that is
       smooth (a constant offset, a scaling of the whole field), and a
       handful of bad corner cells is diluted by the ~500-cell strip it
       is averaged over.  It is not gated for exactly that reason -- both
       reviewers of this instrument, independently, found fields it would
       pass while visibly wrong.

   (b) Where the ERROR lives -- the same edge/interior ratio taken on
       |port - oracle| itself.  This one needs no reference ratio and no
       invented threshold: 1.0 means the residual is spread evenly over
       the panel, and a large value means it is concentrated on the
       boundary strip, which is the signature of a halo or edge defect.
       The residual's SIZE is the parity gate's business; this says only
       where it sits.

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


def ring_mask(shape: tuple, width: int) -> np.ndarray:
    """Boolean frame of ``width`` cells around the border of ``shape``.

    One definition, used by both metrics, so the two cannot drift apart
    about what "the boundary strip" means.
    """
    ni, nj = shape
    m = np.zeros((ni, nj), dtype=bool)
    m[:width, :] = m[-width:, :] = True
    m[:, :width] = m[:, -width:] = True
    return m


def edge_interior_ratio(plane: np.ndarray, width: int) -> tuple:
    """(edge mean, interior mean, ratio) of the high-pass on one panel.

    THE RING IS IN THE ORIGINAL PLANE'S COORDINATES.  ``high_pass``
    returns a plane shrunk by one on every side -- its ``[r, c]`` is a
    second difference CENTRED on the original ``[r + 1, c + 1]`` -- so a
    ring of ``width`` original cells is a ring of ``width - 1`` cells of
    the high-pass.  Taking ``width`` there instead, as this did when it
    was first written, walks one cell further in than advertised.

    The original boundary cells covered are therefore ``1 .. width - 1``
    and ``n - width .. n - 2`` on each axis.  Cell ``0`` and cell
    ``n - 1`` are NOT covered and cannot be: a centred second difference
    has no value there.  That is a real limit of this metric, not an
    oversight -- the outermost owned cell is exactly where an edge defect
    would be largest.
    """
    if width < 2:
        raise SystemExit(
            f"--edge-width must be at least 2: a centred second "
            f"difference has no value on the outermost cell, so a "
            f"width-1 ring would be empty (got {width})")
    hp = high_pass(plane)
    w = width - 1
    ni, nj = hp.shape
    if min(ni, nj) <= 2 * w + 2:
        raise SystemExit(
            f"panel {plane.shape} too small for --edge-width {width}: the "
            f"edge ring and the interior would overlap, and the ratio "
            f"would compare a set against itself")
    mask = ring_mask((ni, nj), w)
    e = float(hp[mask].mean())
    i = float(hp[~mask].mean())
    return e, i, (e / i if i > 0 else float("inf"))


def edge_concentration(port: np.ndarray, oracle: np.ndarray,
                       width: int) -> float:
    """How boundary-heavy the port-minus-oracle residual is on one level.

    Mean |d| on the boundary strip over mean |d| on the interior, in the
    ORIGINAL plane's coordinates (no second difference here -- the error
    field is already the thing of interest, and differencing it would
    throw away a smooth edge bias, which is a defect this campaign has
    actually shipped).

    1.0 = the residual is spread evenly.  Large = it sits on the
    boundary.  There is no threshold and none is invented: this is a
    LOCATION statement, and the parity gate owns the magnitude.
    """
    d = np.abs(np.asarray(port) - np.asarray(oracle))
    ni, nj = d.shape
    if min(ni, nj) <= 2 * width + 2:
        raise SystemExit(
            f"panel {d.shape} too small for --edge-width {width}")
    mask = ring_mask((ni, nj), width)
    e = float(d[mask].mean())
    i = float(d[~mask].mean())
    if e == 0.0 and i == 0.0:
        return 1.0        # bitwise equal: evenly spread, trivially
    return e / i if i > 0 else float("inf")


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
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.npz)
    fields = [str(f) for f in z["fields"]]
    # Refuse a level the arrays do not have BEFORE drawing anything: the
    # parity run that produced this npz is expensive, and an IndexError
    # thrown after it is a lost job (codex review, 2026-08-23).
    nk = int(z[f"port_f1_{fields[0]}"].shape[2])
    if not (-nk <= args.level < nk):
        raise SystemExit(
            f"--level {args.level} is outside the {nk} levels these "
            f"planes carry")
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

    # ---- product 2: the two boundary numbers ---------------------------
    print(f"\nIN-PANEL BOUNDARY-STRIP ROUGHNESS, port/oracle (mean "
          f"|second difference| on the {args.edge_width}-cell strip over "
          f"the interior, port's ratio / oracle's ratio). DESCRIPTIVE, "
          f"NOT GATED: blind to a smooth or proportional error, and a "
          f"few bad corner cells are diluted by the whole strip.")
    worst = 0.0
    worst_where = None
    for f in fields:
        for pf in range(6):
            port = z[f"port_f{pf+1}_{f}"]
            orc = z[f"oracle_f{pf+1}_{f}"]
            for k in range(port.shape[2]):
                _, _, pr = edge_interior_ratio(port[..., k],
                                               args.edge_width)
                _, _, orr = edge_interior_ratio(orc[..., k],
                                                args.edge_width)
                if not np.isfinite(pr) or not np.isfinite(orr):
                    print(f"    {f} face {pf+1} level {k}: interior "
                          f"high-pass is exactly zero, ratio carries no "
                          f"information -- skipped")
                    continue
                if orr == 0.0:
                    print(f"    {f} face {pf+1} level {k}: oracle edge "
                          f"high-pass is exactly zero -- skipped")
                    continue
                excess = pr / orr
                if excess > worst:
                    worst = excess
                    worst_where = (f, pf + 1, k, pr, orr)
        rows = [(pf, edge_interior_ratio(z[f"port_f{pf+1}_{f}"][..., -1],
                                         args.edge_width)[2],
                 edge_interior_ratio(z[f"oracle_f{pf+1}_{f}"][..., -1],
                                     args.edge_width)[2])
                for pf in range(6)]
        print(f"  {f:5s} " + "  ".join(
            f"f{pf+1}:{p:.3f}/{o:.3f}" for pf, p, o in rows))

    if worst_where is None:
        print("\n  no field/face/level carried a usable ratio")
    else:
        f, pf, k, pr, orr = worst_where
        print(f"\n  worst port/oracle excess: {worst:.4f}x at {f} face "
              f"{pf} level {k} (port {pr:.4f}, oracle {orr:.4f})")

    print(f"\nWHERE THE RESIDUAL SITS, mean |port - oracle| on the "
          f"{args.edge_width}-cell boundary strip over the interior. "
          f"1.0 = spread evenly over the panel; large = concentrated on "
          f"the boundary, which is what a halo or edge defect looks "
          f"like. The parity gate owns the SIZE of the residual; this "
          f"says only where it is.")
    worst_c, worst_c_where = 0.0, None
    for f in fields:
        rows = []
        for pf in range(6):
            port = z[f"port_f{pf+1}_{f}"]
            orc = z[f"oracle_f{pf+1}_{f}"]
            per_level = [edge_concentration(port[..., k], orc[..., k],
                                            args.edge_width)
                         for k in range(port.shape[2])]
            finite = [c for c in per_level if np.isfinite(c)]
            top = max(finite) if finite else float("inf")
            rows.append(top)
            if np.isfinite(top) and top > worst_c:
                worst_c = top
                worst_c_where = (f, pf + 1,
                                 int(np.argmax([c if np.isfinite(c)
                                                else -1.0
                                                for c in per_level])))
        print(f"  {f:5s} " + "  ".join(f"f{pf+1}:{r:8.3f}"
                                       for pf, r in enumerate(rows)))
    if worst_c_where is not None:
        f, pf, k = worst_c_where
        print(f"\n  most boundary-concentrated residual: {worst_c:.3f}x "
              f"at {f} face {pf} level {k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
