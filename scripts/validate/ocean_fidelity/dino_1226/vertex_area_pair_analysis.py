#!/usr/bin/env python
"""#1455 -- the vertex (F-cell) AREA gap: is it half of a CANCELLING PAIR?

WHY THIS EXISTS.  The v-face width fix left the vertex-area gap standing at
2.2e-05 median against NEMO's ``e1f*e2f``, and the campaign registered it as
the fifth Rule-8 cancelling-pair instance.  Rule 8's standing answer is "a
joint arm, never a revert", so before this half is fixed ALONE somebody has to
show it is not paired.  This probe decides that offline, with no model run.

WHAT IT MEASURES, in three parts.

1.  THE NAMED DIFF.  NEMO forms the F-cell area as the product of two F-point
    scale factors evaluated at the F-point's OWN Mercator latitude::

        zfj   = REAL( mjg(jj,0) - nn_jeq_s, wp ) + 0.5            ! :97
        pphif = 1./rad * ASIN( TANH( rn_e1_deg *rad* zfj ) )      ! :109
        pe1f  = ra * rad * COS( rad * pphif ) * rn_e1_deg         ! :114
        pe2f  = ra * rad * COS( rad * pphif ) * rn_e1_deg         ! :118

    legoESM builds the exact spherical cap between the two adjacent TRACER
    latitudes.  On the Mercator coordinate ``sin(phi) = tanh(dlon*j)`` gives
    ``d(sin phi)/dj = dlon*cos^2(phi)``, so the cap is the EXACT INTERVAL
    INTEGRAL of ``cos^2`` where NEMO's product is its MIDPOINT value.  Exact
    quadrature versus the midpoint rule, predicting a relative gap of
    ``(dlon^2/12)(3 sin^2(phi_f) - 1)``.  Part 1 checks the measured gap
    against that closed form -- including its SIGN CHANGE at 35.26 deg, which
    no wrong radius or wrong dlon could produce.

2.  THE PAIR.  The vertex Coriolis carries the SAME class of defect at the
    SAME point (legoESM averages two tracer-row values, NEMO evaluates ``f`` at
    the F-point latitude), with closed form ``-(dlon^2/4) cos^2(phi_f)``.  The
    two meet in ONE quantity: the F-point absolute vorticity ``zeta + f`` that
    EEN's ``q`` is built from.  Part 2 measures both channels on NEMO's own
    restart.

    PRE-REGISTERED BAR, fixed before any number below was read: the two are a
    cancelling pair requiring a JOINT fix iff the area half is within a factor
    of 10 of the Coriolis half at the median in that channel.  Below 1/10 the
    area half cannot materially change the total whichever way it points, and
    the fix is unpaired-safe.

3.  THE COST, named rather than buried.  The area and the Coriolis conventions
    are also the two halves of a legoESM-internal exactness: the discrete curl
    of solid-body rotation equals the discrete ``f`` EXACTLY on the
    (cap, cell-average) pair.  Part 3 prints all four corners so the trade is
    visible, including NEMO's own -- which is larger than either half alone.

This probe prints numbers and a pre-registered comparison against a stated
bar.  It does NOT print a verdict; the interpretation belongs in the analysis
after the controls pass, not baked into the tool.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.vertex_area_pair_analysis
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]

RUN_TRAJ = os.environ.get(
    "DINO_NEMO_RUN_TRAJ",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ")
RESTART = os.environ.get("DINO_NEMO_RESTART", "DINO_00005760_restart.nc")

# NEMO's own constants for this mesh: phycst.F90 :26 (ra) and :37 (rad);
# rn_e1_deg from usrdef_nam.F90:30 and DINO's namelist_cfg.
_RA, _RAD, _RN_E1_DEG = constants.R_earth, np.pi / 180.0, 1.0

# The pre-registered bar for part 2.  Stated here, above the measurement.
PAIR_BAR = 0.1


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  run_traj={RUN_TRAJ}")
    print(f"PROVENANCE  restart={RESTART}")
    print(f"PROVENANCE  pre-registered pair bar = {PAIR_BAR}")


def _self_test() -> None:
    """The instrument before its output.  On a mesh whose answer is known by
    construction -- a Mercator column built straight from ``usrdef_hgr.F90``
    -- the measured gap must reproduce the midpoint-rule closed form.  If it
    does not, nothing below this line means anything.
    """
    j = np.arange(-90.0, 90.0)
    a = _RN_E1_DEG * _RAD
    phi_t = np.arcsin(np.tanh(a * j))            # T rows, integer index
    phi_f = np.arcsin(np.tanh(a * (j + 0.5)))    # F rows, half-integer index
    cap = _RA ** 2 * a * np.abs(np.sin(phi_t[1:]) - np.sin(phi_t[:-1]))
    prod = (_RA * a * np.cos(phi_f[:-1])) ** 2
    measured = (cap - prod) / prod
    predicted = (a ** 2 / 12.0) * (3.0 * np.sin(phi_f[:-1]) ** 2 - 1.0)
    ratio = np.median(measured / predicted)
    assert abs(ratio - 1.0) < 1e-3, (
        f"SELF-TEST FAILED: the midpoint-rule decomposition does not "
        f"reproduce a mesh built from NEMO's own transform (ratio {ratio})")
    assert measured.min() < 0.0 < measured.max(), (
        "SELF-TEST FAILED: no sign change on a mesh spanning +-64 deg")
    print(f"SELF-TEST  midpoint-rule decomposition reproduces a "
          f"NEMO-transform mesh to {abs(ratio - 1.0):.2e}; sign change present")


def main() -> int:
    import netCDF4 as nc

    stamp()
    _self_test()

    mm = nc.Dataset(os.path.join(RUN_TRAJ, "mesh_mask.nc"))
    rst = nc.Dataset(os.path.join(RUN_TRAJ, RESTART))

    def _m(k):
        return np.asarray(mm[k][0], dtype=np.float64)

    e1f, e2f, e1u, e2v = _m("e1f"), _m("e2f"), _m("e1u"), _m("e2v")
    gphit, gphif = _m("gphit"), _m("gphif")
    ff = _m("ff_f")
    fmask = np.asarray(mm["fmask"][0], dtype=np.float64)
    a = _RN_E1_DEG * _RAD
    i = 5                      # any interior column; these metrics are lat-only

    # --- co-location check, before anything is compared ----------------------
    # ``pphif`` and ``pphiv`` carry the same +0.5 row offset (:97 vs :96), so
    # NEMO's F and V latitudes must be IDENTICAL on this mesh.  If they are
    # not, this probe is comparing across a staggering.
    dfv = np.abs(gphif - _m("gphiv")).max()
    assert dfv == 0.0, f"gphif != gphiv by {dfv} -- the F/V co-location fails"
    assert np.array_equal(e1f, e2f), "e1f != e2f -- NEMO's mesh is not isotropic"

    print("\n=== PART 1 -- the named diff ===")
    phit, phif = gphit[:, i] * _RAD, gphif[:, i] * _RAD
    n = len(phit) - 1
    A_nemo = (e1f * e2f)[:n, i]
    A_cap = _RA ** 2 * a * np.abs(np.sin(phit[1:]) - np.sin(phit[:-1]))
    signed = (A_cap - A_nemo) / A_nemo
    predicted = (a ** 2 / 12.0) * (3.0 * np.sin(phif[:n]) ** 2 - 1.0)
    # The ratio is only meaningful away from the predicted curve's OWN zero
    # crossing (35.26 deg), where the denominator vanishes BY CONSTRUCTION and
    # min/max would report roundoff as a spread (review MINOR 13).
    _off_zero = np.abs(predicted) > 0.05 * np.abs(predicted).max()
    ratio = (signed / predicted)[_off_zero]
    print(f"  measured  (cap - e1f*e2f)/e1f*e2f : signed median "
          f"{np.median(signed):+.6e}   |.| median {np.median(np.abs(signed)):.6e}"
          f"   |.| max {np.abs(signed).max():.6e}")
    print(f"  predicted (dlon^2/12)(3sin^2-1)   : signed median "
          f"{np.median(predicted):+.6e}   |.| max {np.abs(predicted).max():.6e}")
    print(f"  measured / predicted              : median {np.median(ratio):.6f}"
          f"   min {ratio.min():.6f}   max {ratio.max():.6f}"
          f"   (over the {int(_off_zero.sum())} rows away from the "
          f"denominator's own zero)")
    zero_lat = gphif[int(np.argmin(np.abs(signed))), i]
    print(f"  sign change at                    : {zero_lat:+.3f} deg "
          f"(predicted +-{np.degrees(np.arcsin(np.sqrt(1 / 3.0))):.3f})")
    print("  NOTE these rows include NEMO's 2-row halo; the model's own grid "
          "carries 194 interior rows and a |.| median of 2.1872e-05.")

    print("\n=== PART 2 -- the pair, in the channel where the two halves meet ===")
    un = np.asarray(rst["un"][0], dtype=np.float64)
    vn = np.asarray(rst["vn"][0], dtype=np.float64)
    dz = np.zeros_like(un)
    e2v_v, e1u_u = e2v[None] * vn, e1u[None] * un
    dz[:, :-1, :-1] = ((e2v_v[:, :-1, 1:] - e2v_v[:, :-1, :-1])
                       - (e1u_u[:, 1:, :-1] - e1u_u[:, :-1, :-1]))
    zeta = dz / (e1f * e2f)[None]
    s2 = np.sin(gphif * _RAD) ** 2
    d_area = ((a ** 2 / 12.0) * (3.0 * s2 - 1.0))[None]
    d_cori = (-(a ** 2 / 4.0) * (1.0 - s2))[None]
    f2 = np.broadcast_to(ff[None], zeta.shape)
    # The circulation stencil fills only [:-1, :-1]; the last j-row and
    # i-column are left at exactly 0 and are NOT ocean, so they are excluded
    # rather than allowed to drag the |zeta| median down (review MINOR 15).
    _filled = np.zeros(zeta.shape, dtype=bool)
    _filled[:, :-1, :-1] = True
    wet = (fmask > 0.5) & np.isfinite(zeta) & _filled
    print(f"  wet F-points (3-D)          : {int(wet.sum())}")
    print(f"  |zeta|                      : median "
          f"{np.median(np.abs(zeta)[wet]):.4e}  p99 "
          f"{np.percentile(np.abs(zeta)[wet], 99):.4e}")
    print(f"  |f|                         : median "
          f"{np.median(np.abs(f2)[wet]):.4e}  p99 "
          f"{np.percentile(np.abs(f2)[wet], 99):.4e}")
    # Contributions to the error in the F-point ABSOLUTE vorticity.  zeta =
    # Gamma/A, so an area too large by d_area makes zeta too small by d_area.
    err_area = (-d_area * zeta)[wet]
    err_cori = (d_cori * f2)[wet]
    chan = np.abs(err_area) / np.maximum(np.abs(err_cori), 1e-300)
    print(f"  |area half| [1/s]           : median "
          f"{np.median(np.abs(err_area)):.4e}  p99 "
          f"{np.percentile(np.abs(err_area), 99):.4e}")
    print(f"  |coriolis half| [1/s]       : median "
          f"{np.median(np.abs(err_cori)):.4e}  p99 "
          f"{np.percentile(np.abs(err_cori), 99):.4e}")
    med = float(np.median(chan))
    print(f"  RATIO area/coriolis         : median {med:.5e}  p99 "
          f"{np.percentile(chan, 99):.5e}  max {chan.max():.5e}")
    print(f"  against the pre-registered bar of {PAIR_BAR}: "
          f"{'BELOW' if med < PAIR_BAR else 'AT OR ABOVE'}")
    same = np.sign(err_area) == np.sign(err_cori)
    print(f"  the two halves REINFORCE at {100 * same.mean():.1f}% of wet "
          f"F-points, OPPOSE at {100 * (~same).mean():.1f}%")
    big = chan > PAIR_BAR
    print(f"  points where the area half exceeds the bar: {int(big.sum())} "
          f"({100 * big.mean():.2f}%), of which "
          f"{100 * same[big].mean():.1f}% REINFORCE")
    # PAIRED, per point.  A ratio of two MEDIANS of two different
    # distributions is not the change in the error and can carry the wrong
    # SIGN -- the first version of this probe reported +0.055% (a worsening)
    # where the paired statistic is NEGATIVE (an improvement), and it
    # contradicted the reinforce/oppose split printed just above it
    # (adversarial review MAJOR 1).  Scored the only way that answers the
    # question: at each point, how much does removing the area half change
    # that point's own error?
    tot_both = np.abs(err_area + err_cori)
    tot_fixed = np.abs(err_cori)
    rel_change = (tot_fixed - tot_both) / np.maximum(tot_both, 1e-300)
    print(f"  |total| both halves present : median {np.median(tot_both):.4e}")
    print(f"  |total| area half removed   : median {np.median(tot_fixed):.4e}")
    print(f"  PAIRED per-point change on removing the area half:")
    print(f"     median {100 * np.median(rel_change):+.6f}%   "
          f"mean {100 * float(rel_change.mean()):+.6f}%")
    print(f"     improves at {100 * (rel_change < 0).mean():.1f}% of wet "
          f"F-points, worsens at {100 * (rel_change > 0).mean():.1f}%")
    print(f"     energy-norm ratio (L2 after / L2 before) "
          f"{np.sqrt((tot_fixed ** 2).sum() / (tot_both ** 2).sum()):.6f}")
    print(f"     max |error| {tot_both.max():.4e} -> {tot_fixed.max():.4e}")

    # REGIONAL, because a global median hides a structural enrichment: the
    # Coriolis half vanishes at the equator while the area half tracks
    # relative vorticity (review MINOR).
    lat_f = np.broadcast_to(gphif[None], zeta.shape)[wet]
    print("  regional split of the area/coriolis ratio")
    for name, sel in (("|lat| < 5 deg", np.abs(lat_f) < 5.0),
                      ("|lat| 5-40 deg", (np.abs(lat_f) >= 5.0) & (np.abs(lat_f) < 40.0)),
                      ("|lat| > 40 deg", np.abs(lat_f) >= 40.0)):
        if sel.sum():
            print(f"    {name:16s} median {np.median(chan[sel]):.4e}   "
                  f"above bar {100 * (chan[sel] > PAIR_BAR).mean():.2f}%")

    print("\n=== PART 3 -- the cost: discrete solid-body vorticity consistency ===")
    print("  |curl(solid body) - f| / f, analytic, interior rows")
    s = np.sin(phif[1:-1])
    # The first row is ANALYTIC and exactly zero by construction: dividing the
    # solid-body circulation by the exact cap RETURNS the cell-average f, so
    # the two agree identically.  Derived here rather than asserted, so the row
    # is a computation like the other three (review MINOR 14).
    _cap = _RA ** 2 * a * (np.sin(phit[1:-1]) - np.sin(phit[:-2]))
    _circ = (_RA ** 2 * a) * (np.sin(phit[1:-1]) ** 2 - np.sin(phit[:-2]) ** 2)
    _f_cell = np.sin(phit[1:-1]) + np.sin(phit[:-2])
    corners = {
        "legoESM now   (cap, cell_average)": (_circ / _cap) / _f_cell - 1.0,
        "AREA fix only (e1f*e2f, cell_avg)": (a ** 2 / 12.0) * (3 * s ** 2 - 1),
        "CORIOLIS only (cap, face_lat)": -(a ** 2 / 4.0) * (1 - s ** 2),
        "NEMO itself   (e1f*e2f, face_lat)": (a ** 2 / 6.0) * (3 * s ** 2 - 2),
    }
    for k, v in corners.items():
        print(f"    {k:36s} median {np.median(v):+.4e}   "
              f"|.| max {np.abs(v).max():.4e}")

    # THE BAR, APPLIED IN THIS CHANNEL TOO (review MAJOR 2).  The first
    # version scored the pre-registered bar only in the absolute-vorticity
    # channel, which it passes by ~660x, and never applied it here -- where it
    # FAILS.  Reported, because a bar applied only where it passes is not a
    # bar.
    d_area_sb = (a ** 2 / 12.0) * (3 * s ** 2 - 1)
    d_cori_sb = -(a ** 2 / 4.0) * (1 - s ** 2)
    r_sb = np.abs(d_area_sb) / np.abs(d_cori_sb)
    print(f"    ratio area/coriolis IN THIS CHANNEL: median {np.median(r_sb):.4f}"
          f"  min {r_sb.min():.4f}  max {r_sb.max():.4f}"
          f"  -> {'BELOW' if np.median(r_sb) < PAIR_BAR else 'AT OR ABOVE'}"
          f" the {PAIR_BAR} bar")
    # And the column an oracle lane actually cares about: distance to NEMO.
    nemo_sb = (a ** 2 / 6.0) * (3 * s ** 2 - 2)
    before = np.abs(0.0 - nemo_sb)
    after = np.abs(d_area_sb - nemo_sb)
    print(f"    distance to NEMO in this channel: before {np.median(before):.4e}"
          f" -> after {np.median(after):.4e} (median);"
          f" max {before.max():.4e} -> {after.max():.4e};"
          f" improves at {100 * (after < before).mean():.0f}% of rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
