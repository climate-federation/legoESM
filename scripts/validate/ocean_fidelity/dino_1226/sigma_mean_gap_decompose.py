#!/usr/bin/env python
"""#1455: CHARACTERIZE the 90-day gate's largest violation -- southern surface
sigma MEAN (240x its noise floor), offline, from saved states only.

The five-metric acceptance gate fails on all five at day 90, but not equally.
Against each metric's own #1492 2.1 noise floor the exceedances are roughly:
surface sigma MEAN 240x, surface sigma MAX 31x, upper contrast 17x, ACC 8x,
deep contrast 6x.  This probe takes the largest one apart.

THE METRIC, read from the gate rather than restated
---------------------------------------------------
``acceptance_gate_90d.surface_sigma_south`` is::

    jmid = (J0 + J1) // 2
    s0   = (rho_of(st, wet) - RHO0)[J0:jmid + 1, :, 0]
    s0   = s0[isfinite(s0)];  return s0.max(), s0.mean()

so the MEAN is: in-situ density anomaly from the production S-EOS
(``legoesm.ocean.eos.nemo_seos_eos``, DINO coefficients, pressure taken as
``gdept_0 * rho0 * g``), evaluated at the TOP MODEL LEVEL ONLY (k=0), over the
SOUTHERN HALF of the re-entrant channel band (T-rows ``J0..jmid``), over ALL
longitudes, as a PLAIN UNWEIGHTED ARITHMETIC MEAN of the finite (wet) cells.
No area weights, no thickness weights, one level.  Both models go through the
identical call, and the wet mask is the candidate's own land mask ANDed with
mesh_mask ``tmask`` -- the same mask on both sides.

WHAT THIS PROBE MEASURES
------------------------
(1) INHERITANCE vs DRIFT.  The twin starts from NEMO's own day-180 restart, so
    the day-0 metric must be ~0 on both sides; anything else is the bridge or
    the metric, not the physics.  The metric is then evaluated at every day
    where BOTH sides have a saved state (0/30/60/90: legoESM from the twin
    ``--save-3d`` snapshots, NEMO from its 10-day restarts), so the gap's time
    shape is measured, not assumed.
(2) LOCALIZATION.  The zonal-mean depth-latitude map of the day-90 density
    difference, plus the surface latitude profile inside the gate window.
(3) THE T-vs-S SPLIT, using the model's own EOS helpers and nothing else:
    ``nemo_seos_alpha_beta`` (NEMO ``rab``) gives alpha = -(1/rho0) drho/dT and
    beta = (1/rho0) drho/dS at the MIDPOINT state of the two models, so

        d_rho ~= rho0 * (-alpha * dT + beta * dS)

    and the closure residual against the exact difference of two full EOS calls
    is printed next to every split number.

    HONESTY ABOUT THAT RESIDUAL: it is NOT a control.  The DINO S-EOS density
    anomaly is quadratic in (T - T0) and (S - S0) with a bilinear cross term,
    so derivatives taken at the MIDPOINT state make thermal + haline identically
    equal to the exact difference by the mean-value theorem -- the residual is
    roundoff (~1e-15) by construction and cannot report a bad split.  It is
    kept only as a roundoff monitor.  The split's real justification is that
    the EOS is exactly quadratic here, not that a residual came out small.

NOT A VERDICT MACHINE.  Measurements and controls only; the ranked pointers to
open ledger items are written up separately, not printed here.

Provenance and fp64 are inherited from ``acc_metric_reconciliation``
(imported): every input file is stamped with path/mtime/SHA-256, and this
probe's own source and the repo HEAD are stamped too.

NaN POLICY, stated accurately.  MEASURED: the NEMO restarts and the twin npz
files contain NO NaN -- land is filled with **0.0**.  So the fatal finite
checks are tripwires that cannot fire on today's inputs and are NOT what
protects these numbers; the control with teeth is the asserted WET-CELL COUNT
of the gate window, identical on both sides.  The only NaNs in play are the
ones ``rho_of`` manufactures itself on dry cells.  No reduction that produces a
reported number is ``nan``-prefixed; ``np.nanmax`` appears once, in the FIGURE
colour-limit code only, where all-dry rows are NaN by design.

Usage
-----
  JAX_ENABLE_X64=1 .venv/bin/python sigma_mean_gap_decompose.py \
      --arm armA=/tmp/dino_gate90/armA_transport.npz \
      --fig /path/to/sigma_gap.png
  JAX_ENABLE_X64=1 .venv/bin/python sigma_mean_gap_decompose.py --self-test

SECOND-REVIEW CORRECTIONS (aeaf42, 2026-08-20): a compensating cold layer DOES exist at 900-1200 m, 13% of the warm anomaly (column net +3.61e4 K.m, heat gained -- headline survives); 'saturating' is an extrapolation from 4 points -- 'sub-linear' is what is earned; the day-0 3.4e-8 row is near-tautological (bit-for-bit fp32 cast of NEMO's field) and has no power over vertical-grid/ssh/velocity bridge defects. Physical-range check PASSED: implied d(sigma)/dT -0.12675 vs S-EOS -0.12678 kg/m3/K at the window's own mean T.
"""
import argparse
import glob
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import acc_metric_reconciliation as R  # noqa: E402  (stamp/head_sha/require_finite)
import acc_thermal_wind as A  # noqa: E402  (EOS + geometry + band)
import acceptance_gate_90d as G  # noqa: E402  (the gate's own metric)
from legoesm.ocean.eos import nemo_seos_alpha_beta  # noqa: E402  (production rab)

DAYS = (0, 30, 60, 90)
JMID = (A.J0 + A.J1) // 2               # the gate's southern-half split row
GATE_WINDOW_WET_CELLS = 936             # PINNED: (JMID - J0 + 1) x NX, all wet
SELFTEST_T_SIGMA = 0.024139             # PINNED: -0.2 K over the window [kg/m3]
SELFTEST_S_SIGMA = 0.038277             # PINNED: +0.05 PSU over the window


# ------------------------------------------------------------------ loaders ---
def nemo_state(day):
    """NEMO's own NOW-level (tn, sn) at ``day`` of the 90-day twin, or None."""
    from rebuild_nemo_restart import rebuild
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    tiled = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart_*.nc"
    single = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart.nc"
    if glob.glob(tiled):
        src = tiled
        raw = rebuild(tiled, ["tn", "sn"])
        T, S = raw["tn"], raw["sn"]
    elif os.path.exists(single):
        import netCDF4 as nc
        src = single
        with nc.Dataset(single) as ds:
            # Fill (not drop) the netCDF mask: np.asarray on a MaskedArray
            # silently returns .data, turning a masked value into a plausible
            # number instead of a visible NaN.
            T = np.ma.filled(np.ma.asarray(ds["tn"][0]), np.nan).astype(np.float64)
            S = np.ma.filled(np.ma.asarray(ds["sn"][0]), np.nan).astype(np.float64)
    else:
        return None, None
    yxz = lambda a: np.moveaxis(a, 0, -1)                          # noqa: E731
    return {"T": yxz(T), "S": yxz(S)}, src


def lego_state(path, day):
    d = np.load(path)
    key = f"T3d_day{day}"
    if key not in d:
        return None
    return {"T": d[key].astype(np.float64),
            "S": d[f"S3d_day{day}"].astype(np.float64),
            "land_mask": d["land_mask"].astype(np.float64)}


# ------------------------------------------------------------------ metrics ---
def sigma_field(st, wet):
    """Full 3-D in-situ density ANOMALY sigma = rho - rho0, NaN on dry."""
    return A.rho_of(st, wet) - A.RHO0


def ts_contributions(lego, nemo, wet):
    """(d_rho_exact, thermal, haline, residual) 3-D fields, all NaN on dry.

    thermal = -rho0 * alpha * (T_lego - T_nemo)
    haline  = +rho0 * beta  * (S_lego - S_nemo)
    with (alpha, beta) from the production ``nemo_seos_alpha_beta`` evaluated at
    the MIDPOINT state of the two models and each cell's own gdept_0 depth.
    residual = d_rho_exact - (thermal + haline): the linearisation error, which
    bounds how much of the split can be trusted.
    """
    dT = lego["T"] - nemo["T"]
    dS = lego["S"] - nemo["S"]
    al, be = nemo_seos_alpha_beta(0.5 * (lego["T"] + nemo["T"]),
                                 0.5 * (lego["S"] + nemo["S"]),
                                 A.gdept0, A.CFG)
    al, be = np.asarray(al, dtype=np.float64), np.asarray(be, dtype=np.float64)
    exact = sigma_field(lego, wet) - sigma_field(nemo, wet)
    thermal = np.where(wet, -A.RHO0 * al * dT, np.nan)
    haline = np.where(wet, A.RHO0 * be * dS, np.nan)
    return exact, thermal, haline, exact - (thermal + haline)


def wmean(fld, wet):
    """Unweighted mean over wet cells -- the gate's own reduction.  FATAL if
    any wet cell is non-finite (never a nan-prefixed reduction)."""
    v = fld[wet]
    if v.size == 0:
        raise SystemExit("empty wet selection -- FATAL")
    if not np.all(np.isfinite(v)):
        raise SystemExit(f"NON-FINITE in {int((~np.isfinite(v)).sum())} wet "
                         f"cells -- FATAL")
    return float(v.mean())


def zonal_mean(fld, wet):
    """Zonal mean over wet cells, (y, z); NaN where a (row, level) is all dry."""
    n = wet.sum(axis=1)
    # No nan_to_num: the np.where already drops dry cells, and swallowing a WET
    # NaN would bias the panel toward zero while still counting the cell.
    s = np.sum(np.where(wet, fld, 0.0), axis=1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


# -------------------------------------------------------------------- report ---
def time_series(path, label):
    print(f"\n=== (1) INHERITANCE vs DRIFT -- the gate metric at every matched "
          f"day  [{label}] ===")
    print(f"{'day':>5}{'lego MEAN':>12}{'NEMO MEAN':>12}{'gap':>12}"
          f"{'lego MAX':>12}{'NEMO MAX':>12}{'gap':>12}")
    out = {}
    for d in DAYS:
        lg = lego_state(path, d)
        nm, _ = nemo_state(d)
        if lg is None or nm is None:
            print(f"{d:>5}{'--':>12}{'--':>12}{'--':>12}"
                  f"{'--':>12}{'--':>12}{'--':>12}")
            continue
        wet = A.tmask & (lg["land_mask"] > 0.5)[:, :, None]
        R.require_finite(f"{label} T day {d}", lg["T"], wet)
        R.require_finite(f"NEMO T day {d}", nm["T"], wet)
        lmax, lmean = G.surface_sigma_south(lg, wet)
        nmax, nmean = G.surface_sigma_south(nm, wet)
        out[d] = (lmean, nmean, lmax, nmax)
        print(f"{d:>5}{lmean:12.6f}{nmean:12.6f}{lmean - nmean:+12.6f}"
              f"{lmax:12.6f}{nmax:12.6f}{lmax - nmax:+12.6f}")
    print(f"\ngate floor for both sigma metrics: {G.FLOORS['smean']:g} kg/m3")
    if 0 in out:
        lm, nmn, lx, nx = out[0]
        print(f"day-0 gap at FULL PRECISION (the %.6f column above cannot "
              f"resolve it):\n  MEAN {lm - nmn:+.3e}   MAX {lx - nx:+.3e} "
              f"kg/m3  -- vs the {G.FLOORS['smean']:g} floor, "
              f"{abs(lm - nmn) / G.FLOORS['smean']:.1e} x the floor")
    return out


def decompose(path, label, day=90):
    lg, (nm, src) = lego_state(path, day), nemo_state(day)
    if lg is None or nm is None:
        raise SystemExit(f"no matched day-{day} pair for {label}")
    wet = A.tmask & (lg["land_mask"] > 0.5)[:, :, None]
    exact, thermal, haline, resid = ts_contributions(lg, nm, wet)
    win = _mk(wet, slice(A.J0, JMID + 1), 0)          # the gate's exact window
    if int(win.sum()) != GATE_WINDOW_WET_CELLS:
        raise SystemExit(f"gate window has {int(win.sum())} wet cells, "
                         f"expected {GATE_WINDOW_WET_CELLS} -- the mask or the "
                         f"band definition changed, FATAL")

    print(f"\n=== (3) T-vs-S SPLIT of the day-{day} density difference "
          f"(lego - NEMO) [kg/m3] ===")
    print(f"{'region':<40}{'d_sigma':>12}{'thermal':>12}{'haline':>12}"
          f"{'residual':>12}{'n cells':>10}")
    regions = {
        "GATE WINDOW (k=0, rows J0..jmid)": win,
        "band surface k=0, rows J0..J1": _mk(wet, slice(A.J0, A.J1 + 1), 0),
        "band rows J0..J1, k=0..4 (~top 100 m)": _mk(wet, slice(A.J0, A.J1 + 1),
                                                     slice(0, 5)),
        "band rows J0..J1, all depths": _mk(wet, slice(A.J0, A.J1 + 1),
                                            slice(None)),
        "whole domain, k=0": _mk(wet, slice(None), 0),
        "whole domain, all depths": wet,
    }
    for name, m in regions.items():
        print(f"{name:<40}{wmean(exact, m):12.6f}{wmean(thermal, m):12.6f}"
              f"{wmean(haline, m):12.6f}{wmean(resid, m):12.2e}"
              f"{int(m.sum()):10d}")
    print("\n(thermal = -rho0*alpha*dT, haline = +rho0*beta*dS, both from "
          "nemo_seos_alpha_beta at the\n midpoint state.  residual = exact - "
          "(thermal + haline) is ROUNDOFF BY CONSTRUCTION for this\n quadratic "
          "S-EOS -- a roundoff monitor, NOT evidence that the split is right.)")

    print("\n=== (2) SURFACE LATITUDE PROFILE inside and around the gate "
          "window (k=0) ===")
    print(f"{'row':>5}{'lat':>9}{'in gate':>9}{'d_sigma':>11}{'thermal':>11}"
          f"{'haline':>11}{'dT [K]':>10}{'dS [PSU]':>10}")
    dT = lg["T"] - nm["T"]
    dS = lg["S"] - nm["S"]
    for j in range(A.J0, A.J1 + 1, 2):
        m = np.zeros_like(wet)
        m[j, :, 0] = wet[j, :, 0]
        tag = "yes" if j <= JMID else "no"
        print(f"{j:>5}{A.gphit[j, 25]:9.2f}{tag:>9}"
              f"{wmean(exact, m):11.5f}{wmean(thermal, m):11.5f}"
              f"{wmean(haline, m):11.5f}"
              f"{wmean(dT, m):10.4f}{wmean(dS, m):10.4f}")

    print("\n=== (2b) DEPTH STRUCTURE, gate rows J0..jmid, zonal+meridional "
          "mean per level ===")
    print(f"{'k':>4}{'gdept_1d':>10}{'d_sigma':>11}{'thermal':>11}"
          f"{'haline':>11}{'dT [K]':>10}{'dS [PSU]':>10}")
    for k in range(A.NZ):
        m = np.zeros_like(wet)
        m[A.J0:JMID + 1, :, k] = wet[A.J0:JMID + 1, :, k]
        if not m.any():
            continue
        print(f"{k:>4}{A.gdept1d[k]:10.1f}{wmean(exact, m):11.5f}"
              f"{wmean(thermal, m):11.5f}{wmean(haline, m):11.5f}"
              f"{wmean(dT, m):10.4f}{wmean(dS, m):10.4f}")
    return lg, nm, wet, exact, thermal, haline, src


def _mk(wet, jsel, ksel):
    m = np.zeros_like(wet)
    m[jsel, :, ksel] = wet[jsel, :, ksel]
    return m


def figure(exact, thermal, haline, wet, out_png, label, sha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = A.gphit[:, 25]
    dep = A.gdept1d
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    panels = [("(a) d_sigma = sigma(lego) - sigma(NEMO)", exact),
              ("(b) thermal part  -rho0*alpha*dT", thermal),
              ("(c) haline part  +rho0*beta*dS", haline),
              ("(d) residual = (a) - (b) - (c)", exact - thermal - haline)]
    zm = [zonal_mean(f, wet) for _, f in panels]
    lim = float(np.nanmax(np.abs(np.array(zm[:3]))))
    for ax, (title, _), z in zip(axes.ravel(), panels, zm):
        v = lim if "residual" not in title else float(np.nanmax(np.abs(z))) or 1e-12
        im = ax.pcolormesh(lat, dep, z.T, cmap="RdBu_r", vmin=-v, vmax=v,
                           shading="nearest")
        ax.axvline(A.gphit[A.J0, 25], color="k", lw=0.8, ls=":")
        ax.axvline(A.gphit[JMID, 25], color="k", lw=1.4)
        ax.axvline(A.gphit[A.J1, 25], color="k", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        fig.colorbar(im, ax=ax, label="kg/m3")
    for ax in axes[1]:
        ax.set_xlabel("latitude [deg N]")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth [m]")
    fig.suptitle(f"DINO 90-day twin, zonal-mean density difference "
                 f"(lego - NEMO), day 90 -- {label}\n"
                 f"solid line = gate window northern edge (row {JMID}); dotted "
                 f"= channel band edges (rows {A.J0}/{A.J1});  repo {sha}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nFIGURE {out_png}")


def self_test():
    """Non-vacuity, on four axes the first draft of this probe failed.

    1. THE GATE METRIC ITSELF is exercised (``G.surface_sigma_south``), not
       only the T/S split -- otherwise the whole time series comes from a
       function no control touches, and stubbing it out still PASSES.
    2. THE WINDOW IS CHECKED FROM OUTSIDE ITSELF: a perturbation applied OFF
       the gate window must leave the metric UNCHANGED.  Perturbing the same
       rows the metric reads is self-referential and passes even with the
       window set to the wrong (northern) half.
    3. T AND S ARE PERTURBED TOGETHER, so the "no leak into the other channel"
       assertion is not multiplying an exact zero.
    4. Magnitudes are checked against PINNED literals, not against a
       recomputation from the same globals.
    """
    nm, _ = nemo_state(90)
    if nm is None:
        raise SystemExit("self-test needs the NEMO day-90 restart")
    wet = A.tmask
    win = _mk(wet, slice(A.J0, JMID + 1), 0)
    if int(win.sum()) != GATE_WINDOW_WET_CELLS:
        raise SystemExit(f"gate window wet cells {int(win.sum())} != pinned "
                         f"{GATE_WINDOW_WET_CELLS} -- FATAL")

    e, t, h, r = ts_contributions(nm, nm, wet)
    assert wmean(e, win) == 0.0 and wmean(t, win) == 0.0, \
        "identical states must give an exactly zero split"
    print("[self-test] identical states: d_sigma, thermal, haline all exactly 0")

    base_max, base_mean = G.surface_sigma_south(nm, wet)

    # (1)+(2): the GATE METRIC responds to a perturbation INSIDE its window by
    # the analytic amount, and NOT AT ALL to the same perturbation applied
    # OUTSIDE it.  The second half is the check that has teeth on the window.
    for rows, inside in ((slice(A.J0, JMID + 1), True),
                         (slice(JMID + 1, A.J1 + 1), False)):
        p_ = {k: v.copy() for k, v in nm.items()}
        p_["T"][rows, :, 0] -= 0.2
        _, mean_ = G.surface_sigma_south(p_, wet)
        moved = mean_ - base_mean
        if inside:
            assert abs(moved - SELFTEST_T_SIGMA) < 1e-5, (
                f"gate metric moved {moved} inside its window, pinned "
                f"{SELFTEST_T_SIGMA}")
        else:
            assert abs(moved) < 1e-12, (
                f"gate metric moved {moved} for a perturbation OUTSIDE its "
                f"window -- the window is wrong")
        print(f"[self-test] gate MEAN, -0.2 K applied "
              f"{'INSIDE' if inside else 'OUTSIDE'} the window: "
              f"{moved:+.6f} kg/m3")

    # (3)+(4): both channels perturbed at once, each recovered against its pin.
    p_ = {k: v.copy() for k, v in nm.items()}
    p_["T"][A.J0:JMID + 1, :, 0] -= 0.2
    p_["S"][A.J0:JMID + 1, :, 0] += 0.05
    e, t, h, r = ts_contributions(p_, nm, wet)
    gt, gh, ge, gr = (wmean(t, win), wmean(h, win), wmean(e, win),
                      wmean(r, win))
    assert abs(gt - SELFTEST_T_SIGMA) < 1e-4, f"thermal {gt} vs pin"
    assert abs(gh - SELFTEST_S_SIGMA) < 1e-4, f"haline  {gh} vs pin"
    assert abs(ge - (gt + gh)) < 1e-10, (
        f"the split does not close to roundoff: {ge - (gt + gh)}")
    print(f"[self-test] T -0.2 K AND S +0.05 PSU together: d_sigma {ge:+.5f} = "
          f"thermal {gt:+.5f} (pin {SELFTEST_T_SIGMA}) + haline {gh:+.5f} "
          f"(pin {SELFTEST_S_SIGMA}), residual {gr:+.2e}")
    print("SELF-TEST PASS")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="armA=/tmp/dino_gate90/armA_transport.npz",
                   metavar="LABEL=PATH", help="twin npz to characterize")
    p.add_argument("--fig", default=None, help="output PNG path")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    sha = R.head_sha()
    print(f"repo HEAD = {sha}   fp64 = {np.zeros(1).dtype}")
    print(f"NEMO twin dir = {G.RUN_90D_TWIN}")
    print(f"gate window: T-rows {A.J0}..{JMID} "
          f"({A.gphit[A.J0, 25]:.2f}..{A.gphit[JMID, 25]:.2f} degN), "
          f"level k=0 only, all {A.NX} longitudes, unweighted mean")
    if args.self_test:
        return self_test()

    label, path = args.arm.split("=", 1)
    print(f"\n=== PROVENANCE ===\n  {label:<10}{R.stamp(path)}")
    _, nsrc = nemo_state(90)
    print(f"  {'NEMO d90':<10}{nsrc}")
    print()
    G.instrument_self_checks(A.tmask)

    time_series(path, label)
    _, _, wet, exact, thermal, haline, _ = decompose(path, label, 90)
    if args.fig:
        figure(exact, thermal, haline, wet, args.fig, label, sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
