#!/usr/bin/env python3
"""Is a field's small-scale structure RESOLVED or is it grid-scale speckle?

The horizontal moisture smoother exists to suppress cell-to-cell noise in
humidity, so weakening it has to be checked against the thing it was repairing.
A plain variance or neighbour-difference measure cannot answer that: a sharper
but perfectly resolved front raises cell-to-cell differences exactly as noise
does.  What separates them is how the difference GROWS with separation.

For pairs of cells a distance ``d`` apart, take the second-order structure
function ``S(d) = <(q_a - q_b)^2>`` at one-cell and two-cell separation and
report the slope

    p = log(S2 / S1) / log(d2 / d1).

A smooth, differentiable field has ``S ~ d^2`` and ``p ~ 2``; a field whose
variance sits at the grid scale decorrelates within one cell, so ``S2 ~ S1``
and ``p ~ 0``; a 2-dx checkerboard, whose two-cell-apart values are back IN
phase, gives ``p < 0``.  The slope is dimensionless and independent of the
field's overall variance, so a run with genuinely sharper gradients and a run
with speckle are distinguishable.

Both synthetic limits are asserted in ``tests/unit/test_gridscale_speckle.py``
before any run is reported.

``--budget`` answers the follow-up question -- WHERE the two-cell variance comes
from -- by evaluating, on a saved state, the rate at which each operator changes
the one-cell mean squared difference,

    dS1/dt = 2 <(q_a - q_b)(dq_a/dt - dq_b/dt)>,

for the horizontal moisture advection and for the smoother separately.  A
positive rate MAKES two-cell structure, a negative rate removes it, and the
ratio says whether the filter is simply undoing the transport scheme.

    gridscale_speckle.py --runs g30_ctl g30_qvsm1e4 --day 110 --field trc_q_v
    gridscale_speckle.py --runs g30_ctl --day 110 --budget --nu 2e5 --dt 112.5
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh

ROOT = pathlib.Path("/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")


def ring_pairs(mesh):
    """(1-ring, 2-ring) cell-index pairs as two (2, npair) int arrays.

    The 2-ring is the neighbours-of-neighbours with the cell itself and its own
    1-ring removed, so the two sets are disjoint and the separations do not
    overlap.  Padding in ``cellsOnCell`` is -1 and is dropped.
    """
    con = np.asarray(mesh.cellsOnCell)            # (maxEdges, nCells)
    if con.shape[1] < con.shape[0]:
        raise SystemExit(f"cellsOnCell is {con.shape}; expected (maxEdges, nCells)")
    n_cells = con.shape[1]
    coe = np.asarray(mesh.cellsOnEdge)            # (2, nEdges)
    one = coe[:, (coe >= 0).all(axis=0)]

    ring1 = [set(c for c in con[:, i] if c >= 0) for i in range(n_cells)]
    a, b = [], []
    for i in range(n_cells):
        second = set()
        for j in ring1[i]:
            second |= ring1[j]
        second -= ring1[i] | {i}
        for j in second:
            if j > i:                              # each pair once
                a.append(i); b.append(j)
    return one, np.array([a, b], dtype=np.int64)


def separation(mesh, pairs):
    """Great-circle distance [m] between the cell centres of each pair."""
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    i, j = pairs
    cos_d = (np.sin(lat[i]) * np.sin(lat[j])
             + np.cos(lat[i]) * np.cos(lat[j]) * np.cos(lon[i] - lon[j]))
    return constants.R_earth * np.arccos(np.clip(cos_d, -1.0, 1.0))


def structure_slope(field, mesh, pairs1, pairs2, d1=None, d2=None):
    """(p, S1, S2, d1, d2) for a per-cell field; p is the growth exponent."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 1 or field.shape[0] != np.asarray(mesh.latCell).shape[0]:
        raise SystemExit(f"field {field.shape} is not one value per cell")
    if d1 is None:
        d1 = float(separation(mesh, pairs1).mean())
    if d2 is None:
        d2 = float(separation(mesh, pairs2).mean())
    s1 = float(((field[pairs1[0]] - field[pairs1[1]]) ** 2).mean())
    s2 = float(((field[pairs2[0]] - field[pairs2[1]]) ** 2).mean())
    if not (s1 > 0.0 and s2 > 0.0):
        raise SystemExit("a structure function is zero; the field is constant")
    return float(np.log(s2 / s1) / np.log(d2 / d1)), s1, s2, d1, d2


def extremum_fraction(field, mesh):
    """Fraction of cells that are a strict local extremum among their own
    neighbours.

    The growth exponent alone cannot separate cell-scale OSCILLATION from a
    genuinely patchy field (cloud condensate is intermittent by nature, and
    isolated positive blobs on a zero background score low for a physical
    reason).  A smooth field, however patchy, has few cells that beat all six
    neighbours; a field oscillating at the grid scale has close to all of them.
    Random cell-scale noise on a hexagonal mesh sits near 2/7 = 0.29.
    """
    field = np.asarray(field, dtype=np.float64)
    con = np.asarray(mesh.cellsOnCell)
    n_cells = con.shape[1]
    if field.shape[0] != n_cells:
        raise SystemExit(f"field {field.shape} is not one value per cell")
    hi = np.ones(n_cells, dtype=bool)
    lo = np.ones(n_cells, dtype=bool)
    any_nb = np.zeros(n_cells, dtype=bool)
    for k in range(con.shape[0]):
        nb = con[k]
        ok = nb >= 0
        v = np.where(ok, field[np.maximum(nb, 0)], np.nan)
        hi &= ~ok | (field > v)
        lo &= ~ok | (field < v)
        any_nb |= ok
    return float((any_nb & (hi | lo)).mean())


def variance_rate(field, tend, pairs):
    """d/dt of the mean squared pair difference, 2<(dq)(d tendency)>."""
    i, j = pairs
    dq = field[i] - field[j]
    dt_ = tend[i] - tend[j]
    return float(2.0 * (dq * dt_).mean())


def stage_budget(run, day, nu, dt, mesh, p1, p2, levels):
    """Rows of (level, S1, transport rate, smoother rate) for water vapour."""
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        tracer_horizontal_advection,
    )
    from legoesm.core.operators_voronoi import scalar_del2_cell_3d

    z = np.load(ROOT / run / f"checkpoint_day_{day:04d}.npz", allow_pickle=True)
    q = np.asarray(z["trc_q_v"], dtype=np.float64)
    u = np.asarray(z["u"], dtype=np.float64)
    n_edges = np.asarray(mesh.cellsOnEdge).shape[1]
    if u.shape[0] != n_edges:
        raise SystemExit(f"{run}: u is {u.shape}, expected ({n_edges}, nlev) "
                         "edge-normal velocity")
    adv = np.asarray(tracer_horizontal_advection(
        jnp.asarray(q)[..., None], jnp.asarray(u), mesh)[..., 0], dtype=np.float64)
    lap = np.asarray(scalar_del2_cell_3d(jnp.asarray(q), mesh), dtype=np.float64)
    rows = []
    for lev in levels:
        s1 = float(((q[p1[0], lev] - q[p1[1], lev]) ** 2).mean())
        rows.append((lev, s1,
                     variance_rate(q[:, lev], adv[:, lev], p1),
                     variance_rate(q[:, lev], nu * lap[:, lev], p1)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--field", default="trc_q_v")
    ap.add_argument("--levels", nargs="*", type=int, default=[2, 8, 14, 20, 26])
    ap.add_argument("--subdivision", type=int, default=6)
    ap.add_argument("--budget", action="store_true",
                    help="report which operator makes the two-cell variance")
    ap.add_argument("--nu", type=float, default=2.0e5,
                    help="smoother diffusivity for --budget [m^2/s]")
    ap.add_argument("--lat-band", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None,
                    help="restrict the pair statistics to cells whose BOTH "
                         "members lie in this latitude band [degrees]; the "
                         "exponent is a ratio of means, so a regional answer "
                         "needs regional pairs, not a global mean")
    ap.add_argument("--dt", type=float, default=112.5,
                    help="dynamics timestep for --budget [s]")
    args = ap.parse_args(argv)

    mesh = create_voronoi_mesh(args.subdivision)
    p1, p2 = ring_pairs(mesh)
    if args.lat_band is not None:
        lo, hi = args.lat_band
        lat_deg = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
        inband = (lat_deg >= lo) & (lat_deg < hi)
        keep1 = inband[p1[0]] & inband[p1[1]]
        keep2 = inband[p2[0]] & inband[p2[1]]
        if keep1.sum() < 100 or keep2.sum() < 100:
            raise SystemExit(f"latitude band {lo}..{hi} keeps only "
                             f"{keep1.sum()} / {keep2.sum()} pairs")
        p1, p2 = p1[:, keep1], p2[:, keep2]
        print(f"latitude band {lo:g}..{hi:g}: {int(inband.sum())} cells, "
              f"{p1.shape[1]} one-ring and {p2.shape[1]} two-ring pairs")
    d1 = float(separation(mesh, p1).mean())
    d2 = float(separation(mesh, p2).mean())
    print(f"mesh subdivision {args.subdivision}: {np.asarray(mesh.latCell).shape[0]} cells, "
          f"1-ring {d1 / 1e3:.1f} km, 2-ring {d2 / 1e3:.1f} km "
          f"({p1.shape[1]} and {p2.shape[1]} pairs)")
    if args.budget:
        for run in args.runs:
            print(f"\n{run}, day {args.day}: rate of change of the one-cell mean "
                  f"squared difference in water vapour [(kg/kg)^2/s]\n"
                  f"  transport = unlimited centred advection, "
                  f"smoother = del2 at nu={args.nu:g} m^2/s")
            print(f"{'level':>6s}{'S1':>13s}{'transport':>13s}{'smoother':>13s}"
                  f"{'net':>13s}{'e-fold [h]':>12s}")
            for lev, s1, r_adv, r_sm in stage_budget(
                    run, args.day, args.nu, args.dt, mesh, p1, p2, args.levels):
                net = r_adv + r_sm
                tau = s1 / abs(net) / 3600.0 if net != 0.0 else float("inf")
                print(f"{lev:6d}{s1:13.3e}{r_adv:13.3e}{r_sm:13.3e}{net:13.3e}"
                      f"{tau:12.2f}")
        return

    print(f"\n{args.field}, day {args.day}: growth exponent p "
          "(2 = smooth, 0 = grid-scale noise, <0 = 2dx pile-up)")
    print(f"{'level':>6s}" + "".join(f"{r:>16s}" for r in args.runs))
    for lev in args.levels:
        row = []
        for run in args.runs:
            z = np.load(ROOT / run / f"checkpoint_day_{args.day:04d}.npz",
                        allow_pickle=True)
            if args.field not in z:
                raise SystemExit(f"{run}: no field {args.field!r} in the day-"
                                 f"{args.day} checkpoint")
            arr = z[args.field]
            if arr.ndim != 2:
                raise SystemExit(f"{run}/{args.field}: shape {arr.shape}, expected (nCells, nlev)")
            p, *_ = structure_slope(arr[:, lev], mesh, p1, p2, d1, d2)
            row.append(p)
        print(f"{lev:6d}" + "".join(f"{v:16.3f}" for v in row))


if __name__ == "__main__":
    main()
