#!/usr/bin/env python
"""#1455 -- read the pre-registered numbers off the lateral-viscosity ablation arms.

Pre-registration: ``PREREG_wall_viscosity_ablation.md`` (written before either
arm was launched).  This script produces exactly the two quantities that
document registered, for each arm, against NEMO's own day-90 continuation of
the same restart:

  G4     the day-90 southern-basin zonal-transport gap (legoESM - NEMO) summed
         over the four wall rows 1..4 (69.50S..68.43S), in Sv.
  A_wall the day-90 sea-surface excess against the wall [mm] and L_e, its
         meridional e-folding scale in rows and km.

Neither the row-transport formula nor the e-folding fit is re-derived here:
``row_transport`` is imported from ``basin_seasonal_decomp`` (the function that
produced the parent document's latitude table) and the e-folding fit is the same
interpolation ``southern_wall_balance.b1`` used to report 2.6 rows / 112 km.
The candidate loader and the NEMO-side stitching are the acceptance gate's.

The NEMO baseline is ``RUN_90D_TWIN/DINO_00008640_restart_*`` -- NEMO's own
90-day continuation of the SAME day-180 restart both arms start from, which is
the only day-90 NEMO state that is a controlled partner for a 90-day twin.

THE DRY-ROW TRAP (0be305459).  Row 0 is entirely dry and both models store
exact zeros there beside wet values near -0.97 m.  Every reduction below is
taken over ``tmask`` wet cells only, and ``--self-test`` plants a large value on
the dry row and requires no scored number to move.

This script prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.wall_visc_ablation_gap \
      ARM1.npz ARM2.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR.parent))

import acc_thermal_wind as A            # noqa: E402
import acceptance_gate_90d as G         # noqa: E402
from basin_seasonal_decomp import row_transport  # noqa: E402
from rebuild_nemo_restart import rebuild          # noqa: E402

from legoesm import constants           # noqa: E402

WALL_ROWS = [1, 2, 3, 4]                # southern_wall_balance.py:51
DAY = 90
KT_DAY90 = 8640
RUN_90D_TWIN = f"{A.DINO}/RUN_90D_TWIN"


def stamp(arms) -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  NEMO baseline={RUN_90D_TWIN}/DINO_{KT_DAY90:08d}_restart_*")
    print(f"PROVENANCE  day={DAY}  wall rows={WALL_ROWS}  "
          f"southern basin=rows 0..{A.J0 - 1}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    for a in arms:
        print(f"PROVENANCE  arm={a}")


def nemo_day90():
    pattern = f"{RUN_90D_TWIN}/DINO_{KT_DAY90:08d}_restart_*.nc"
    if not glob.glob(pattern):
        raise SystemExit(f"NEMO day-90 baseline not found: {pattern}")
    raw = rebuild(pattern, ["un", "sshn"])
    return {"u": np.moveaxis(raw["un"], 0, -1), "eta": raw["sshn"]}


def efold_rows(prof: np.ndarray) -> float:
    """e-folding scale [rows] of a wall-anchored profile.

    Same interpolation ``southern_wall_balance.b1`` used (that file's
    ``efold`` block); ``prof[0]`` is the wall row.
    """
    pos = np.abs(np.asarray(prof, dtype=np.float64))
    try:
        return float(np.interp(pos[0] / np.e, pos[::-1],
                               np.arange(len(pos))[::-1]))
    except Exception:
        return float("nan")


def measure(arm_path, nem, wet_u, wet_T, rows_basin, eta_plant=None):
    d = np.load(arm_path)
    cand = G.load_candidate(arm_path, day=DAY)
    lego_rows = row_transport(cand["u"], wet_u, rows_basin)
    nemo_rows = row_transport(nem["u"], wet_u, rows_basin)
    gap = lego_rows - nemo_rows                       # Sv per row

    eta_l = np.asarray(d[f"eta3d_day{DAY}"], dtype=np.float64)
    if eta_plant is not None:
        eta_l = eta_l.copy()
        eta_l[0, :] = eta_plant                       # dry row 0
    de = eta_l - nem["eta"]
    prof_rows = list(range(0, A.J0))
    prof = np.array([float(np.mean(de[j][wet_T[j]])) if wet_T[j].any()
                     else np.nan for j in prof_rows])
    wall_prof = prof[WALL_ROWS[0]:WALL_ROWS[-1] + 1]
    # e-folding measured from the wall row outward over the whole southern band
    tail = prof[WALL_ROWS[0]:]
    return dict(rows=gap, G4=float(gap[WALL_ROWS].sum()),
                Gbasin=float(gap[:A.J0].sum()),
                A_wall_mm=1e3 * float(wall_prof[0]),
                eta_prof_mm=1e3 * prof,
                Le_rows=efold_rows(tail))


def munk_width_km():
    """(A_h/beta)^(1/3) at the wall rows, from the oracle's own coefficient."""
    e1t = np.asarray(A.mm["e1t"][0]).squeeze()
    e2t = np.asarray(A.mm["e2t"][0]).squeeze()
    ahmt = 0.5 * 0.27 * np.maximum(e1t, e2t)   # coeff-ok: rn_Uv, namelist_cfg
    beta = (2.0 * constants.Omega
            * np.cos(np.deg2rad(np.asarray(A.gphit, float)))
            / constants.R_earth)
    Lm = (ahmt / beta) ** (1.0 / 3.0)
    return float(np.mean(Lm[WALL_ROWS, 25])) / 1e3, float(
        np.mean(ahmt[WALL_ROWS, 25]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="arm .npz files, baseline first")
    ap.add_argument("--labels", default=None,
                    help="comma-separated labels, one per arm")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    stamp(args.arms)

    wet_u = A.umask
    wet_T = A.tmask[:, :, 0]
    rows_basin = slice(0, A.J0)
    nem = nemo_day90()

    if args.self_test:
        if not args.arms:
            raise SystemExit("--self-test needs one arm npz to plant into")
        base = measure(args.arms[0], nem, wet_u, wet_T, rows_basin)
        planted = measure(args.arms[0], nem, wet_u, wet_T, rows_basin,
                          eta_plant=1e3)
        for k in ("G4", "Gbasin", "A_wall_mm", "Le_rows"):
            d = abs(base[k] - planted[k])
            print(f"  dry-row plant, {k:10s}: |change| = {d:.6e}")
            assert d == 0.0 or np.isnan(d), (
                f"a dry-row value reached {k} -- the reduction is contaminated")
        d4 = np.abs(base["eta_prof_mm"][WALL_ROWS] -
                    planted["eta_prof_mm"][WALL_ROWS]).max()
        assert d4 == 0.0, "dry-row plant moved a scored wall row"
        # non-vacuity: planting into a WET row MUST move the score
        d2 = np.load(args.arms[0])
        eta = np.asarray(d2[f"eta3d_day{DAY}"], np.float64).copy()
        de_ref = base["eta_prof_mm"][WALL_ROWS[0]]
        eta[WALL_ROWS[0]][wet_T[WALL_ROWS[0]]] += 1.0
        moved = 1e3 * float(np.mean((eta - nem["eta"])[WALL_ROWS[0]][
            wet_T[WALL_ROWS[0]]]))
        assert abs(moved - de_ref) > 1.0, (
            "planting a WET wall-row value moved nothing -- the check above "
            "is vacuous")
        print(f"  wet-row plant of 1 m moved the wall row "
              f"{de_ref:.3f} -> {moved:.3f} mm")
        print("SELF-TEST PASS: dry cells excluded, and the check can fail")
        return 0

    lm_km, ah_wall = munk_width_km()
    labels = (args.labels.split(",") if args.labels
              else [Path(a).stem for a in args.arms])
    res = [measure(a, nem, wet_u, wet_T, rows_basin) for a in args.arms]
    dy_km = float(np.mean(np.asarray(A.mm["e2v"][0]).squeeze()[WALL_ROWS, 25])
                  ) / 1e3

    print(f"\nMunk width at the wall rows from the oracle's own coefficient "
          f"({ah_wall:.0f} m2/s): {lm_km:.0f} km = {lm_km / dy_km:.1f} rows "
          f"(1 row = {dy_km:.0f} km)")

    print("\nPER-ROW SOUTHERN-BASIN TRANSPORT GAP, legoESM - NEMO [Sv], day 90")
    hdr = f"{'row':>4}{'lat':>9}" + "".join(f"{lab:>14}" for lab in labels)
    print(hdr)
    for j in range(A.J0):
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"{j:>4}{A.gphit[j, 25]:>9.2f}"
              + "".join(f"{r['rows'][j]:>14.4f}" for r in res) + tag)
    print(f"{'sum1-4':>13}" + "".join(f"{r['G4']:>14.4f}" for r in res))
    print(f"{'sum0-13':>13}" + "".join(f"{r['Gbasin']:>14.4f}" for r in res))

    print("\nSEA-SURFACE DIFFERENCE, legoESM - NEMO [mm, row mean over wet]")
    print(f"{'row':>4}{'lat':>9}" + "".join(f"{lab:>14}" for lab in labels))
    for j in range(A.J0):
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"{j:>4}{A.gphit[j, 25]:>9.2f}"
              + "".join(f"{r['eta_prof_mm'][j]:>14.4f}" for r in res) + tag)

    print("\nTHE REGISTERED NUMBERS")
    print(f"{'quantity':>22}" + "".join(f"{lab:>14}" for lab in labels))
    for key, name, fmt in (("G4", "G4 wall rows [Sv]", "{:>14.4f}"),
                           ("Gbasin", "basin rows 0-13 [Sv]", "{:>14.4f}"),
                           ("A_wall_mm", "A_wall [mm]", "{:>14.4f}"),
                           ("Le_rows", "L_e [rows]", "{:>14.2f}")):
        print(f"{name:>22}" + "".join(fmt.format(r[key]) for r in res))
    print(f"{'L_e [km]':>22}"
          + "".join(f"{r['Le_rows'] * dy_km:>14.0f}" for r in res))

    if len(res) == 2:
        a, b = res
        for key, name in (("G4", "|G4|"), ("Le_rows", "L_e")):
            va, vb = abs(a[key]), abs(b[key])
            ch = (vb - va) / va if va else float("nan")
            print(f"  {name:>8}: {va:.4f} -> {vb:.4f}   "
                  f"change {100 * ch:+.1f}%")
        print("  (registered bands: >=15% on either quantity = CONFIRM "
              "friction owns the wall scale; <5% on BOTH = REFUTE; between "
              "= no verdict)")
        print(f"  |dG4| = {abs(b['G4'] - a['G4']):.4f} Sv against the "
              f"day-90 circumpolar ensemble floor 0.091 Sv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
