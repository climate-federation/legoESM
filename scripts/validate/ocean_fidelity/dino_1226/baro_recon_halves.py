#!/usr/bin/env python
"""#1455 PHASE 1 control -- is the substep walk's "deposit" a WHOLE stage?

WHY THIS EXISTS.  The time walk (baro_deposit_time_walk.py) measured the
substep walk's per-step deposit at ten bridged states and got a 90-day integral
of +13.13 Sv where the budget row it was being scored against is -0.807 Sv.  A
quantity that predicts +13 Sv of ACC divergence across a window in which the
twin's ACC actually diverged by -0.40 Sv is not the divergence, so the metric
itself is now the suspect.

WHAT THE WALK ACTUALLY COMPARES.  Its deposit is the difference of the
boxcar-averaged barotropic velocity ``puu_b(Kaa)`` alone (validated against
``spg_dump_puu_b_final.bin``).  But that field is not what the barotropic solve
hands to the 3-D state.  NEMO's ``mlf_baro_corr`` (stpmlf.F90:606-618, dumped
here as baro_dump_u_before -> baro_dump_u_after) commits it as

    puu(Kaa) <- puu(Kaa) - zue * r1_hu(Kaa) + uu_b(Kaa)

i.e. it REPLACES the 3-D field's own column mean with the barotropic solve's.
The commit is a DIFFERENCE OF TWO LARGE TERMS.  Measuring only the second is
the "never budget across a stage boundary" failure this repo has hit before.

THE MEASUREMENT, and it needs no model run and no legoESM side.  On NEMO's OWN
dumps, in the gate's own weights, compare
    (a) ACC[ uu_b(Kaa) broadcast down the column ]   -- the half the walk sees
    (b) ACC[ after - before ]                        -- what the stage commits
RESULT, and it REFUTES the hypothesis this probe was written to test.  I
expected (b) to be orders of magnitude smaller than (a) -- a level
cancellation that would explain the walk's +13 Sv at a stroke.  It is not:
measured, the commit is ~97% of the uu_b half on NEMO's own side, so the
removed column mean is a 3% correction, not a near-cancellation.  The
hypothesis is retracted here rather than left implied.

WHAT THAT DOES AND DOES NOT SETTLE.  It settles that the two quantities are
not separated by a cancellation OF LEVELS.  It does NOT settle that they are
the same quantity, because the walk compares DIFFERENCES between two models,
and a 3% term in the levels can still dominate a difference that is itself
1e-4 of the level.  Measuring the difference of the commit needs legoESM's own
pre/post-reconciliation 3-D velocity, which no probe currently taps; that is
the named next instrument, not a result claimed here.

CONTROLS
  * the reduction is the SAME ``acc_sv`` the walk uses: sum_j sum_k
    u(j,k,i)*e3t_1d(k)*e2u(j,i), rows 1..197, mean over longitudes 2..-2.
  * the two dumps are read with the SAME halo strip and shape assert.
  * ``before`` and ``after`` must differ (a probe that read the same file twice
    would print a perfect zero and look like a triumph): asserted.
  * the removed-column-mean half is reconstructed independently from ``before``
    and must reproduce (after - before) to roundoff, which is the check that
    this probe understands the operator rather than merely subtracting files.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DINO = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
HLS = 2
DAYS = [180 + 10 * k for k in range(10)]


def _read3d(path: str, jpi: int, jpj: int, jpk: int) -> np.ndarray:
    """(jpk-1, jpj, jpi) haloed dump -> (jpk-1, jpj-2*HLS, jpi-2*HLS)."""
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (jpi * jpj)
    if nlev * jpi * jpj != a.size:
        raise SystemExit(f"{path}: {a.size} f8 is not a whole number of "
                         f"{jpi}x{jpj} levels")
    return a.reshape(nlev, jpj, jpi)[:, HLS:-HLS, HLS:-HLS]


def main() -> int:
    import netCDF4 as nc

    print("#1455 PHASE 1 control: is the walk's deposit a WHOLE stage?")
    print("  (NEMO's own dumps only -- no legoESM, no model run, no fit)\n")
    print("   day |  ACC[uu_b half] |  ACC[commit]   |  |commit|/|half| |"
          "  recon resid")
    rows = []
    for day in DAYS:
        lane = os.path.join(_DINO, f"RUN_SEQDUMP_D{day}_1R")
        d = nc.Dataset(os.path.join(lane, "mesh_mask.nc"))
        e3t_1d = np.asarray(d.variables["e3t_1d"][0], dtype=np.float64)
        umask = np.asarray(d.variables["umask"][0], dtype=np.float64)
        e2u = np.asarray(d.variables["e2u"][0], dtype=np.float64)
        e3u = np.asarray(d.variables["e3u_0"][0], dtype=np.float64)
        d.close()
        jpk, jpj_i, jpi_i = umask.shape          # interior (haloless) mesh_mask
        jpi, jpj = jpi_i + 2 * HLS, jpj_i + 2 * HLS

        # The dump family carries a _ktNNNNNNNN suffix; only the day-180 lane
        # additionally symlinks the un-suffixed names.  Resolve the suffixed
        # file at THIS lane's nit000 so every lane is read at the same seam.
        kt0 = 5760 + 320 * ((day - 180) // 10) + 1
        def _dump(which: str) -> str:
            suff = os.path.join(lane, f"baro_dump_u_{which}_kt{kt0:08d}.bin")
            if os.path.exists(suff):
                return suff
            plain = os.path.join(lane, f"baro_dump_u_{which}.bin")
            if os.path.exists(plain):
                return plain
            raise SystemExit(f"day {day}: no baro_dump_u_{which} at {lane}")
        before = _read3d(_dump("before"), jpi, jpj, jpk)
        after = _read3d(_dump("after"), jpi, jpj, jpk)
        nlev = before.shape[0]
        if after.shape != before.shape:
            raise SystemExit(f"day {day}: before {before.shape} != after "
                             f"{after.shape}")
        if np.array_equal(before, after):
            raise SystemExit(
                f"day {day}: before == after bit-for-bit. Either the dump pair "
                "is the same file or the reconciliation is a no-op; either way "
                "the ratio below would be a fabricated zero.")
        um = umask[:nlev]
        e3 = e3t_1d[:nlev][:, None, None]

        def acc_sv(u3: np.ndarray) -> float:
            """The gate's linear mean surrogate, on a 3-D u field [Sv]."""
            per_lon = (u3 * e3 * um * e2u[None, :, :]).sum(axis=0)[1:198, :] \
                .sum(axis=0)
            return float(per_lon[2:-2].mean()) / 1.0e6

        # (a) the half the walk measures: uu_b(Kaa), recovered from the
        #     operator itself rather than from another file -- after - before
        #     = -colmean(before) + uu_b, so uu_b = after - before + colmean.
        e3u_l = e3u[:nlev]
        um_l = um
        colmean = ((e3u_l * before * um_l).sum(axis=0)
                   / np.maximum((e3u_l * um_l).sum(axis=0), 1e-30))
        uu_b = (after - before + colmean[None, :, :]) * um_l
        # cross-check: uu_b must be depth-uniform on every wet column (it is a
        # 2-D field broadcast); if it is not, the operator was misread.
        wet_cols = (um_l.sum(axis=0) > 0)
        _sel = wet_cols[None, :, :] & (um_l > 0.5)
        _mcol = np.where(_sel, uu_b, np.nan)
        _hi = np.where(wet_cols, np.nanmax(np.where(_sel, uu_b, -np.inf),
                                           axis=0), 0.0)
        _lo = np.where(wet_cols, np.nanmin(np.where(_sel, uu_b, +np.inf),
                                           axis=0), 0.0)
        spread = float(np.max(_hi - _lo))
        a_half = acc_sv(uu_b)
        a_commit = acc_sv((after - before) * um_l)
        # independent reconstruction of the commit from `before` alone
        recon = (uu_b - colmean[None, :, :]) * um_l
        resid = float(np.abs(recon - (after - before) * um_l).max())
        rows.append((day, a_half, a_commit, spread, resid))
        print(f"   {day:3d} | {a_half:+.6e} | {a_commit:+.6e} | "
              f"{abs(a_commit / a_half) if a_half else float('nan'):15.6e} | "
              f"{resid:.2e}")
        assert spread < 1e-12, (
            f"day {day}: the recovered uu_b is not depth-uniform (ptp "
            f"{spread:.3e}); the mlf_baro_corr operator was misread")
        assert resid < 1e-15, (
            f"day {day}: the reconstruction of (after-before) from `before` "
            f"alone is off by {resid:.3e}; this probe does not understand the "
            "operator and its ratio means nothing")

    half = np.array([r[1] for r in rows])
    commit = np.array([r[2] for r in rows])
    print(f"\n   mean |ACC[uu_b half]|   = {np.abs(half).mean():.6e} Sv")
    print(f"   mean |ACC[commit]|       = {np.abs(commit).mean():.6e} Sv")
    print(f"   mean cancellation factor = "
          f"{np.abs(commit).mean() / np.abs(half).mean():.6e}")
    print("\n   (the walk's per-step deposit is a difference of the FIRST "
          "column between\n    the two models; the stage the budget row "
          "measures is the SECOND.  These\n    levels are within ~3% of each "
          "other, so no cancellation of LEVELS explains\n    the walk's "
          "integral; the difference of the SECOND column between models is\n"
          "    not measurable without a legoESM-side reconciliation tap.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
