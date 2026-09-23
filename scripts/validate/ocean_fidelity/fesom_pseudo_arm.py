"""Geometry-only control for the FESOM surface-salinity gap.

THE QUESTION. At day 30 the FESOM arm scores surface-salinity rmse 0.465
against the ORCA1 oracle where the tripole scores 0.258, and 86% of that gap
sits in the single ring of target cells that touches something not scored.
Two explanations survive the measurements already made (records a154bc322 and
b23f6a1a1): the remap drawing values ACROSS a coastline or barrier, which the
scorer's own docstring already names as needing a topology-aware remapper, and
a genuine coastal model difference.  Smoothing cannot separate them -- a
cross-barrier draw is an OFFSET, not noise, so averaging does not remove it.

THE CONTROL.  Push the ORACLE'S OWN FIELD through FESOM's geometry and then
through the scorer's own path, and score the result against the oracle.  The
model error is zero by construction, so whatever rmse comes out is
manufactured entirely by the scoring path.

WHAT THIS CONTROL CANNOT DO -- read before quoting any number from it.  An
earlier version of this docstring called the result an UPPER BOUND on the
geometry term, on the grounds that putting the oracle on FESOM's nodes needs
one operator a real run never applies (~111 km oracle cells onto a ~38 km node
cloud) which can itself draw across a barrier.  That is RETRACTED.  Adversarial
review pointed out the same leg also DEFLATES the control, because the pseudo
field is band-limited to the oracle's own resolution and its kernel is built
from the same wet oracle cells the reference uses, so shared scoring-path bias
cancels between them.  Both effects live at coasts, where the ring
stratification cannot separate them.  The deflation half is now MEASURED and
absent -- the pseudo error correlates with the tripole's at about +0.04 in
ring 1 -- but the label stays off: this is an estimate, not a bound.

MORE IMPORTANT, AND NOT FIXABLE HERE: this FESOM snapshot carries NO LAND, so
every node is wet and the control stamps the ORACLE'S coastline through the
node values while the real arm carries FESOM's own 38 km coastline.  A
disagreement about where the ocean ENDS is therefore INVISIBLE to this control
by construction -- and that is one of the two hypotheses it was written to
separate.  It separates neither; what it measures is how much ring-1 error the
scoring path alone manufactures for a node cloud whose field equals the
oracle's, which bounds how much of ring 1 could ever be attributed to physics
and nothing more.

THREE GATES RUN BEFORE ANY NEW NUMBER IS PRODUCED, because a probe's first
output is untrusted and this one reconstructs a path it does not own:
  1. the oracle regridded here must reproduce the dumped oracle field;
  2. FESOM regridded here must reproduce the dumped FESOM field;
  3. ``idw_to_points`` must agree with ``regrid_curv_to_latlon`` to 1e-12 when
     handed that function's own target points, which is what licenses using it
     for the point-cloud leg at all.
Gates 1 and 2 are the "run the proxy against a case whose answer you already
know" rule: if this script cannot reproduce the scorecard it is standing on,
nothing downstream is readable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # scripts/validate
from compare_omip_nemo import (  # noqa: E402
    _load_legoesm,
    _load_nemo,
    regrid_curv_to_latlon,
)

_EXACT_TOL = 1e-12          # gate 3: same arithmetic, so agreement is exact
_REPRO_TOL = 1e-10          # gates 1-2: same arithmetic on the same inputs


def _xyz_deg(lat_deg, lon_deg):
    """Unit-sphere Cartesian from DEGREES, matching the regridder's helper.

    DO NOT promote the coordinates to float64 here.  The regridder converts
    whatever dtype the source carries -- and NEMO's nav_lat/nav_lon are
    single precision -- so promoting builds a DIFFERENT tree, which breaks
    near-ties differently and can select a different member of a coincident
    pair.  On a tripole fold two coincident cells can hold different values,
    so that is not a rounding difference, it is a different answer.  Found by
    gate 3 on real data after the synthetic unit test (float64 throughout)
    passed.
    """
    la = np.deg2rad(np.asarray(lat_deg))
    lo = np.deg2rad(np.asarray(lon_deg))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def idw_to_points(field, src_lat_deg, src_lon_deg, ocean_mask,
                  tgt_lat_deg, tgt_lon_deg, k=4, max_deg=2.5):
    """Inverse-distance interpolation onto ARBITRARY POINTS.

    The point-target sibling of ``compare_omip_nemo.regrid_curv_to_latlon``,
    which can only target a regular lat-lon grid because it meshgrids its two
    axis vectors.  A node cloud is not expressible that way, so this is a new
    capability rather than a second copy of one -- and the weighting is kept
    identical (the same ``k`` nearest source cells, weights 1/chord-distance
    normalised, the same ``max_deg`` chord cutoff for the coverage flag) so
    that the two agree wherever both are defined.  Gate 3 in ``main`` checks
    that agreement rather than asserting it.

    ``tgt_lat_deg``/``tgt_lon_deg`` are PAIRED coordinate arrays of the same
    length, not axes.  Returns ``(values, coverage)``, both 1-D.
    """
    from scipy.spatial import cKDTree

    m = np.asarray(ocean_mask).ravel() > 0.5
    if not m.any():
        raise ValueError("no ocean source cells")
    tlat = np.asarray(tgt_lat_deg, dtype=np.float64).ravel()
    tlon = np.asarray(tgt_lon_deg, dtype=np.float64).ravel()
    if tlat.size != tlon.size:
        raise ValueError(f"target lat/lon are paired arrays and must have the "
                         f"same length, got {tlat.size} and {tlon.size}")
    src_xyz = _xyz_deg(np.asarray(src_lat_deg).ravel()[m],
                       np.asarray(src_lon_deg).ravel()[m])
    vals = np.asarray(field, dtype=np.float64).ravel()[m]
    tree = cKDTree(src_xyz)
    d, idx = tree.query(_xyz_deg(tlat, tlon), k=k)
    # cKDTree drops the neighbour axis entirely at k=1, returning (n,) rather
    # than (n,1), so the normalisation below would sum over the TARGET points.
    # regrid_curv_to_latlon carries the same latent defect and never trips it
    # because every caller uses its k=4 default; this routine is exercised at
    # k=1 by its own tests, which is how it was found.
    if np.ndim(d) == 1:
        d, idx = d[:, None], idx[:, None]
    d = np.maximum(d, 1e-12)
    w = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
    out = (vals[idx] * w).sum(axis=1)
    chord = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    return out, (d[:, 0] < chord).astype(np.float64)


def wrms(err, area, mask):
    w = area[mask]
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    return float(np.sqrt((w * err[mask] ** 2).sum() / tot))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True,
                   help="cells npz from compare_three_way_nemo --dump-cells; "
                        "supplies the scored mask, area weights, edge rings "
                        "and the dumped fields the gates reproduce.")
    p.add_argument("--fesom", type=Path, required=True)
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-time-idx", type=int, required=True)
    p.add_argument("--res-deg", type=float, default=1.0)
    a = p.parse_args()

    d = np.load(a.dump, allow_pickle=True)
    for key in ("scored", "area", "SSS_b", "SSS_nemo", "edge_ring"):
        if key not in d.files:
            raise SystemExit(
                f"FATAL: dump lacks {key!r}; regenerate it with a scorer new "
                f"enough to write edge rings. Keys: {sorted(d.files)}")
    scored = np.asarray(d["scored"], dtype=bool)
    area = np.asarray(d["area"], dtype=np.float64)
    ring = np.asarray(d["edge_ring"], dtype=int)
    sss_b = np.asarray(d["SSS_b"], dtype=np.float64)
    sss_a = np.asarray(d["SSS_a"], dtype=np.float64)
    sss_n = np.asarray(d["SSS_nemo"], dtype=np.float64)

    tgt_lat = (-90.0 + a.res_deg / 2.0
               + a.res_deg * np.arange(int(180.0 / a.res_deg)))
    tgt_lon = a.res_deg / 2.0 + a.res_deg * np.arange(int(360.0 / a.res_deg))
    if scored.shape != (tgt_lat.size, tgt_lon.size):
        raise SystemExit(
            f"FATAL: dump is on a {scored.shape} grid but --res-deg "
            f"{a.res_deg} builds {(tgt_lat.size, tgt_lon.size)}; the dump and "
            f"this run disagree about the target.")

    F = _load_legoesm(a.fesom)
    N = _load_nemo(a.nemo_gridt, a.nemo_time_idx)
    print(f"[load] FESOM {F['sss'].shape} nodes, oracle {N['sss'].shape}")

    def rg(src, field):
        return regrid_curv_to_latlon(field, src["lat"], src["lon"],
                                     src["mask"], tgt_lat, tgt_lon)

    # --- gate 1 + 2: reproduce the dumped fields -----------------------------
    for label, src, want in (("oracle", N, sss_n), ("FESOM", F, sss_b)):
        got, _ = rg(src, src["sss"])
        err = float(np.nanmax(np.abs(got - want)))
        if not (err <= _REPRO_TOL):
            raise SystemExit(
                f"FATAL gate: this script's {label} regrid differs from the "
                f"dumped one by {err:.3e}; it is not reconstructing the "
                f"scorecard's path and no number below is readable.")
        print(f"[gate] {label} regrid reproduces the dump (max |diff| "
              f"{err:.2e})")

    # --- gate 3: the point-target sibling agrees on grid points --------------
    lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)
    print(f"[dtype] oracle lat {np.asarray(N['lat']).dtype} lon "
          f"{np.asarray(N['lon']).dtype}; FESOM lat "
          f"{np.asarray(F['lat']).dtype} -- the tree is built in whatever "
          f"these carry, so they decide how near-ties break")
    ref, ref_cov = rg(N, N["sss"])
    pts, pts_cov = idw_to_points(N["sss"], N["lat"], N["lon"], N["mask"],
                                 lat2d.ravel(), lon2d.ravel())
    diff = np.abs(pts.reshape(ref.shape) - ref)
    err = float(np.max(diff))
    if not (err <= _EXACT_TOL):
        # A bare max tells you the gate fired, not what to fix. Two very
        # different faults produce a large max: a systematic difference in the
        # weighting (then most cells disagree), or tie-breaking between
        # coincident source cells that carry DIFFERENT values, which on a
        # tripole fold is a handful of cells and a real hazard for both
        # routines. Print enough to tell them apart before touching anything.
        bad = diff > _EXACT_TOL
        n_bad = int(bad.sum())
        frac = n_bad / diff.size
        qs = np.percentile(diff[bad], [50, 90, 99]) if n_bad else [0, 0, 0]
        j, i = np.unravel_index(int(np.argmax(diff)), diff.shape)
        worst = [(float(lat2d[y, x]), float(lon2d[y, x]), float(diff[y, x]))
                 for y, x in zip(*np.unravel_index(
                     np.argsort(diff.ravel())[-5:][::-1], diff.shape))]
        loc = "; ".join(f"({la:.1f}N,{lo:.1f}E) {v:.3f}" for la, lo, v in worst)
        raise SystemExit(
            f"FATAL gate: idw_to_points disagrees with the grid regridder by "
            f"{err:.3e} on its own target points; the node leg cannot be "
            f"trusted to use the same weighting.\n"
            f"  disagreeing cells: {n_bad} of {diff.size} ({frac:.3%}); "
            f"median {qs[0]:.3e}, p90 {qs[1]:.3e}, p99 {qs[2]:.3e}\n"
            f"  worst five: {loc}\n"
            f"  worst cell is target ({j},{i}) at "
            f"{float(lat2d[j, i]):.2f}N {float(lon2d[j, i]):.2f}E")
    if not np.array_equal(pts_cov.reshape(ref_cov.shape) > 0.5, ref_cov > 0.5):
        raise SystemExit("FATAL gate: idw_to_points coverage flag disagrees "
                         "with the grid regridder's.")
    print(f"[gate] idw_to_points matches the grid regridder to {err:.2e}")

    # --- the control ---------------------------------------------------------
    # Leg 1: oracle -> FESOM's node positions.  Leg 2: those nodes -> the 1 deg
    # target through the SAME call the real FESOM arm takes.  Model error zero.
    pseudo_nodes, node_cov = idw_to_points(N["sss"], N["lat"], N["lon"],
                                           N["mask"], F["lat"], F["lon"])
    far = int((node_cov < 0.5).sum())
    print(f"[leg1] oracle sampled onto {pseudo_nodes.size} FESOM nodes; "
          f"{far} are farther than the 2.5 deg cutoff from any oracle wet cell")
    pseudo, _ = regrid_curv_to_latlon(
        pseudo_nodes.reshape(np.asarray(F["lat"]).shape),
        F["lat"], F["lon"], F["mask"], tgt_lat, tgt_lon)

    err_a, err_b = sss_a - sss_n, sss_b - sss_n
    err_p = pseudo - sss_n
    use = scored & np.isfinite(err_a) & np.isfinite(err_b) & np.isfinite(err_p)
    if int(use.sum()) < 1000:
        raise SystemExit(f"VACUOUS: only {int(use.sum())} usable cells")

    print(f"\n{'edge ring':>12} {'cells':>7} {'tripole':>9} {'FESOM':>9} "
          f"{'PSEUDO':>9} {'FES-PSE':>9} {'excess':>9} {'pseudo/ex':>10}")
    rows = [(k, str(k)) for k in range(1, int(ring.max()) + 1)]
    rows.append((0, "interior"))
    rows.append((None, "ALL"))
    for val, lab in rows:
        m = use if val is None else (use & (ring == val))
        n = int(m.sum())
        if n == 0:
            continue
        ra, rb, rp = (wrms(err_a, area, m), wrms(err_b, area, m),
                      wrms(err_p, area, m))
        # FESOM's error with the geometry-only error REMOVED, on the model
        # that the scoring path contributes additively. That model is an
        # assumption, not a result; it is stated in the caveat below. The
        # subtraction is the number that answers "how much of FESOM's coastal
        # error is the scoring path", which neither the ratio nor the
        # correlation answers on its own -- a pattern can correlate strongly
        # and still carry little of the amplitude.
        rr = wrms(err_b - err_p, area, m)
        # The excess is what the control has to explain: the part of FESOM's
        # scatter the tripole's does not account for.
        dd = rb ** 2 - ra ** 2
        ex = np.sqrt(dd) if dd > 0 else -np.sqrt(-dd)
        frac = rp / ex if ex > 0 else float("nan")
        print(f"{lab:>12} {n:7d} {ra:9.4f} {rb:9.4f} {rp:9.4f} {rr:9.4f} "
              f"{ex:9.4f} {frac:10.2f}")

    # GLM's tightening, and the reason the word "bound" is not used above.
    # The pseudo field is band-limited to the ORACLE's resolution and its
    # interpolation kernel is built from the same wet oracle cells the
    # reference uses, so part of the scoring path's bias cancels between them
    # and the control is DEFLATED by an unknown amount. If that cancellation
    # were absent the pseudo error would be uncorrelated with the tripole's.
    # Correlation well away from zero means it is real and the control
    # understates the geometry term.
    def wcorr(x, y, m):
        w = area[m]
        tot = w.sum()
        xm = (w * x[m]).sum() / tot
        ym = (w * y[m]).sum() / tot
        cov = (w * (x[m] - xm) * (y[m] - ym)).sum() / tot
        sx = np.sqrt((w * (x[m] - xm) ** 2).sum() / tot)
        sy = np.sqrt((w * (y[m] - ym) ** 2).sum() / tot)
        return float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")

    print(f"\narea-weighted correlation of the PSEUDO error field with")
    print(f"  the tripole's error: ring 1 "
          f"{wcorr(err_p, err_a, use & (ring == 1)):+.3f}, all "
          f"{wcorr(err_p, err_a, use):+.3f}")
    print(f"  FESOM's error:       ring 1 "
          f"{wcorr(err_p, err_b, use & (ring == 1)):+.3f}, all "
          f"{wcorr(err_p, err_b, use):+.3f}")

    print("\nPSEUDO is the oracle's own field carried through FESOM's mesh: "
          "zero model error by construction, so it measures what the SCORING "
          "PATH alone manufactures. It is NOT a bound in either direction. "
          "Leg 1 inflates it (the oracle-to-node step can draw across "
          "barriers) and deflates it (shared kernel with the reference), and "
          "both effects sit at coasts where the rings cannot separate them. "
          "It also CANNOT see a coastline disagreement at all: this FESOM "
          "snapshot carries no land, so the control stamps the ORACLE's "
          "coastline while the real arm carries its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
