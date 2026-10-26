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
    """e-folding scale [rows] of a wall-anchored profile, or NaN if ill-posed.

    Same interpolation ``southern_wall_balance.b1`` used (that file's ``efold``
    block), with a guard that block did not need and this one does.

    ``np.interp`` requires an increasing ``xp`` and returns a plausible-looking
    number rather than an error when it is not.  A profile that RISES away from
    the wall, or crosses zero and grows again with the opposite sign, is not
    decaying at all, and feeding it here returns "the row where the anomaly
    first falls to 1/e" from the wrong end of the band.  Such a profile has no
    e-folding scale, so this returns NaN and the caller reports it as
    unmeasurable instead of quoting a length.

    The guard: the profile must be monotonically non-increasing in magnitude
    from the wall down to the first row below ``|prof[0]|/e``, and it must get
    there.
    """
    pos = np.abs(np.asarray(prof, dtype=np.float64))
    if pos.size < 2 or not np.isfinite(pos).all() or pos[0] <= 0.0:
        return float("nan")
    target = pos[0] / np.e
    below = np.nonzero(pos <= target)[0]
    if below.size == 0:
        return float("nan")          # never decays to 1/e inside the band
    k = int(below[0])
    seg = pos[:k + 1]
    if np.any(np.diff(seg) > 0.0):
        return float("nan")          # not a decay: rises somewhere on the way
    return float(np.interp(target, seg[::-1], np.arange(k + 1)[::-1]))


def _efold_self_check() -> None:
    """The guard must accept a real decay and reject the shapes that fooled it."""
    decay = np.exp(-np.arange(10) / 2.5)
    got = efold_rows(decay)
    assert abs(got - 2.5) < 0.05, f"clean exponential misread as {got}"
    # the 2x arm's measured shape: rises, crosses zero, grows the other way
    bad = np.array([1.61, 2.46, 2.52, 1.98, 1.03, -0.04, -0.86, -1.30,
                    -1.45, -1.39, -1.14, -0.81, -0.45])
    assert np.isnan(efold_rows(bad)), "a sign-crossing profile got a length"
    assert np.isnan(efold_rows(np.array([1.0, 0.9, 0.8]))), "no 1/e in band"
    print("OK: efold_rows accepts a decay (2.50 rows) and rejects the "
          "sign-crossing and never-decaying shapes")


def measure(arm_path, nem, wet_u, wet_T, rows_basin, eta_plant=None,
            wet_plant=None, u_dry_plant=None, u_wet_plant=None):
    """Score one arm.  The four ``*_plant`` arguments exist only for the
    self-test: they inject a value into the dry or the wet part of the field a
    given score reads, so that score can be shown both to ignore land and to
    respond to water."""
    d = np.load(arm_path)
    cand = G.load_candidate(arm_path, day=DAY)
    u = cand["u"]
    if u_dry_plant is not None:
        u = np.where(np.asarray(wet_u, dtype=bool), u, u_dry_plant)
    if u_wet_plant is not None:
        u = np.where(np.asarray(wet_u, dtype=bool), u + u_wet_plant, u)
    lego_rows = row_transport(u, wet_u, rows_basin)
    nemo_rows = row_transport(nem["u"], wet_u, rows_basin)
    gap = lego_rows - nemo_rows                       # Sv per row

    eta_l = np.asarray(d[f"eta3d_day{DAY}"], dtype=np.float64)
    if eta_plant is not None:
        eta_l = eta_l.copy()
        eta_l[0, :] = eta_plant                       # dry row 0
        eta_l = np.where(wet_T, eta_l, eta_plant)     # every dry cell
    if wet_plant is not None:
        eta_l = np.where(wet_T, eta_l + wet_plant, eta_l)
    de = eta_l - nem["eta"]
    prof_rows = list(range(0, A.J0))
    prof = np.array([float(np.mean(de[j][wet_T[j]])) if wet_T[j].any()
                     else np.nan for j in prof_rows])
    wall_prof = prof[WALL_ROWS[0]:WALL_ROWS[-1] + 1]
    # e-folding measured from the wall row outward over the whole southern band
    tail = prof[WALL_ROWS[0]:]
    return dict(rows=gap, G4=float(gap[WALL_ROWS].sum()),
                G4abs=float(np.abs(gap[WALL_ROWS]).sum()),
                Gbasin=float(gap[:A.J0].sum()),
                Gbasinabs=float(np.abs(gap[:A.J0]).sum()),
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
    # The parent probe (basin_seasonal_decomp.py:429-437) refuses to score an
    # arm whose land mask disagrees with NEMO's surface tmask, because then the
    # two sides are not integrating the same ocean. Carried over rather than
    # dropped: it costs nothing and a future arm on a different mask would
    # otherwise be scored silently.
    for a in args.arms:
        lm = np.load(a)["land_mask"].astype(np.float64)
        nbad = int(np.abs(lm - np.asarray(wet_T, dtype=np.float64)).sum())
        if nbad:
            raise SystemExit(
                f"{a}: its land mask disagrees with NEMO's surface tmask at "
                f"{nbad} cells -- the two sides are not the same ocean and the "
                "row sums are not comparable.")
    print(f"land-mask gate: all {len(args.arms)} arm(s) agree with NEMO's "
          f"surface tmask at every one of {wet_T.size} cells")

    if args.self_test:
        _efold_self_check()
        if not args.arms:
            raise SystemExit("--self-test needs one arm npz to plant into")
        base = measure(args.arms[0], nem, wet_u, wet_T, rows_basin)

        # (i) the dry-row trap, on the quantities that actually read eta.
        planted = measure(args.arms[0], nem, wet_u, wet_T, rows_basin,
                          eta_plant=1e3)
        for k in ("A_wall_mm", "Le_rows"):
            b, q = base[k], planted[k]
            assert np.isfinite(b), (
                f"{k} is not finite in the unplanted arm -- the dry-row check "
                "on it would pass vacuously")
            assert b == q, (
                f"a dry-row value reached {k}: {b} -> {q}")
            print(f"  dry-row plant, {k:10s}: {b:.6f} unchanged")
        d4 = np.abs(base["eta_prof_mm"][WALL_ROWS]
                    - planted["eta_prof_mm"][WALL_ROWS]).max()
        assert d4 == 0.0, "dry-row plant moved a scored wall row"

        # (ii) non-vacuity, THROUGH `measure` rather than around it: planting a
        # wet wall row must move the same scored number the check above froze.
        moved = measure(args.arms[0], nem, wet_u, wet_T, rows_basin,
                        eta_plant=None, wet_plant=1.0)
        dm = abs(moved["A_wall_mm"] - base["A_wall_mm"])
        assert dm > 100.0, (
            f"planting 1 m on the WET wall row moved A_wall by only {dm} mm -- "
            "the dry-row check above is vacuous")
        print(f"  wet-row plant of 1 m moved A_wall {base['A_wall_mm']:.3f} "
              f"-> {moved['A_wall_mm']:.3f} mm")

        # (iii) the transport scores read u, not eta, so they need their own
        # plant or the checks above say nothing about them.
        u_moved = measure(args.arms[0], nem, wet_u, wet_T, rows_basin,
                          u_dry_plant=1.0)
        for k in ("G4", "Gbasin"):
            assert base[k] == u_moved[k], (
                f"a dry u-face value reached {k}: {base[k]} -> {u_moved[k]}")
            print(f"  dry-u-face plant, {k:8s}: {base[k]:.6f} unchanged")
        u_wet = measure(args.arms[0], nem, wet_u, wet_T, rows_basin,
                        u_wet_plant=0.01)
        assert abs(u_wet["G4"] - base["G4"]) > 0.01, (
            "planting a WET u-face moved no transport -- the check above is "
            "vacuous")
        print(f"  wet-u-face plant of 0.01 m/s moved G4 {base['G4']:.4f} -> "
              f"{u_wet['G4']:.4f} Sv")
        print("SELF-TEST PASS: dry cells excluded on BOTH the sea-surface and "
              "the transport scores, and each check can fail")
        return 0

    # The 2-D-vs-3-D masking of the viscosity operator (wall_ldf_alignment.py
    # part C) is answer-neutral only while the velocity below the sea floor
    # stays zero. That is a property of the STATE, so it has to be re-checked on
    # the states being scored, not assumed from the initial condition.
    for a in args.arms:
        u90 = G.load_candidate(a, day=DAY)["u"]
        sub = np.abs(np.where(np.asarray(wet_u, dtype=bool), 0.0, u90)).max()
        print(f"below-seafloor |u| at day {DAY} in {Path(a).stem}: {sub:.3e} "
              f"m/s (must be 0 for the operator's 2-D masking to be "
              f"answer-neutral)")
        if sub != 0.0:
            print("  NOT zero -- the lateral-viscosity operator's 2-D cell and "
                  "face masking is no longer equivalent to NEMO's 3-D masking "
                  "on this arm; see wall_ldf_alignment.py part C.")

    lm_km, ah_wall = munk_width_km()
    # Label each arm by the viscosity it ACTUALLY ran at, read from the
    # artifact's own stamp, not from its filename. A swapped file otherwise
    # produces a confidently wrong sign with nothing to catch it.
    stamped = []
    for a in args.arms:
        d = np.load(a)
        if "rn_Uv" not in d.files:
            raise SystemExit(
                f"{a} carries no rn_Uv stamp -- it predates the ablation stamp "
                "and does not record which lateral viscosity it ran at. Re-run "
                "the arm; a viscosity ablation cannot be scored off a filename.")
        stamped.append(float(d["rn_Uv"]))
    print("arm viscosities read from the artifacts: "
          + ", ".join(f"{Path(a).stem} rn_Uv={u}" for a, u in
                      zip(args.arms, stamped)))
    if len(set(stamped)) != len(stamped):
        print("  WARNING: two arms carry the SAME rn_Uv -- this is not a "
              "one-variable viscosity A/B.")
    labels = (args.labels.split(",") if args.labels
              else [f"rn_Uv={u:g}" for u in stamped])
    res = [measure(a, nem, wet_u, wet_T, rows_basin) for a in args.arms]
    dy_km = float(np.mean(np.asarray(A.mm["e2v"][0]).squeeze()[WALL_ROWS, 25])
                  ) / 1e3

    print(f"\nMunk width at the wall rows from the oracle's own coefficient "
          f"({ah_wall:.0f} m2/s): {lm_km:.0f} km = {lm_km / dy_km:.1f} rows "
          f"(1 row = {dy_km:.0f} km)")
    print(f"  a DOUBLED coefficient predicts 2^(1/3) = 1.26x that width: "
          f"{lm_km * 2 ** (1 / 3):.0f} km = {lm_km * 2 ** (1 / 3) / dy_km:.1f} "
          f"rows -- the pre-registered prediction the arms test")

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
              + "".join(f"{r['eta_prof_mm'][j]:>14.3f}" for r in res) + tag)

    print("\nTHE REGISTERED NUMBERS")
    print(f"{'quantity':>22}" + "".join(f"{lab:>14}" for lab in labels))
    for key, name, fmt in (
            ("G4", "G4 wall rows [Sv]", "{:>14.4f}"),
            ("G4abs", "  same, sum of |.| ", "{:>14.4f}"),
            ("Gbasin", "basin rows 0-13 [Sv]", "{:>14.4f}"),
            ("Gbasinabs", "  same, sum of |.| ", "{:>14.4f}"),
            ("A_wall_mm", "A_wall [mm]", "{:>14.3f}"),
            ("Le_rows", "L_e [rows]", "{:>14.2f}")):
        print(f"{name:>22}" + "".join(fmt.format(r[key]) for r in res))
    print(f"{'L_e [km]':>22}"
          + "".join(f"{r['Le_rows'] * dy_km:>14.0f}" for r in res))
    print("  The SIGNED sum and the sum of magnitudes are both printed on "
          "purpose: four rows whose errors cancel give a small signed sum "
          "while every individual row is worse, and the signed number alone "
          "reads as an improvement that did not happen.")

    if len(res) == 2:
        a, b = res
        for key, name in (("G4", "|signed G4|"), ("G4abs", "sum |row|"),
                          ("Le_rows", "L_e")):
            va, vb = abs(a[key]), abs(b[key])
            ch = (vb - va) / va if va else float("nan")
            print(f"  {name:>11}: {va:.4f} -> {vb:.4f}   "
                  f"change {100 * ch:+.1f}%")
        if not np.isfinite(b["Le_rows"]) or not np.isfinite(a["Le_rows"]):
            print("  L_e is NaN in at least one arm: that arm's sea-surface "
                  "profile does not decay away from the wall, so it has no "
                  "e-folding scale and the registered L_e leg cannot be "
                  "scored on it. The |G4| leg still can.")
        print("  (registered bands: >=15% on either quantity = CONFIRM "
              "friction owns the wall scale; <5% on BOTH = REFUTE; between "
              "= no verdict)")
        print(f"  |dG4| = {abs(b['G4'] - a['G4']):.4f} Sv against the "
              f"day-90 circumpolar ensemble floor 0.091 Sv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
