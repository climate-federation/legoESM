#!/usr/bin/env python
"""#1455: RECONCILE the two contradictory day-90 ACC headlines, offline.

Two numbers for the SAME quantity have been quoted by this campaign:

  * "day-90 ACC deficit ~1.74 Sv, legoESM BELOW NEMO"  -- the gate2 / acc_
    trajectory_series era (``eaaa920a9`` / ``b2ec54382`` lineage), and the
    number ``true_tdepth_ladder_twin.py`` still carries as its analytic-ladder
    reference arm.
  * "legoESM 66.10 Sv vs NEMO 65.37 = 0.73 Sv EXCESS" -- the 90-day gate A/B
    at ``c38e8a3ea`` (pre-registered in ``9920ffe15``).

Neither may be cited until the disagreement is explained.  This probe replays
ONE metric definition -- ``acc_thermal_wind.acc_full``, imported not copied --
over EVERY saved day-90 twin state still on disk, plus NEMO's own restarts, so
the whole history is a single column of comparable numbers.

WHAT IT SEPARATES (the three candidate owners, in order):

 1. METRIC DEFINITION.  ``acc_trajectory_series`` and ``acceptance_gate_90d``
    both call ``acc_thermal_wind.acc_full`` on the u returned by
    ``acceptance_gate_90d.load_candidate`` -- the same function object on the
    same array, against the same NEMO restart family (RUN_90D_TWIN, NOW level
    ``un``).  Neither reads ``LEGOESM_NEMO_E3T``: that variable selects the
    MODEL's vertical ladder at run time, while the metric's weights come from
    ``mesh_mask`` (``e3t_1d`` for ACC) unconditionally.  So a metric-definition
    difference is not merely unlikely, it is structurally impossible between
    those two callers -- but "structurally impossible" is a claim, so the
    VARIANTS block below measures how far the number can be moved by changing
    the reduction (median->mean), the vertical weight (e3t_1d->e3t_0) and the
    longitude window, on an old arm and a new arm alike.
 2. CODE / RUN STATE.  Every arm's day-0 ACC is printed next to its day-90 ACC.
    A common day-0 across arms means the bridge and the metric agree and the
    day-90 spread is the model's, not the instrument's.
 3. PROBE DEFECT.  The two instrument self-checks that ``acceptance_gate_90d``
    runs (NEMO y10 ACC 121.07 Sv; band volume 2.694775e16 m3) are run here too,
    fatally, before any arm is read.

The BAROCLINIC/BAROTROPIC split of the old-vs-new day-90 difference
(``acc_thermal_wind.bc_bt_band``, verbatim) is printed for the two arms named
by ``--split-old`` / ``--split-new``: a change that moves transport while
leaving the density metrics put is barotropic, and that constrains which class
of run-state difference can own the step.

NOT A VERDICT MACHINE.  This probe prints measurements and controls.  It does
not classify, rank or conclude; the reconciliation is written up separately.

Provenance: every input file is stamped with its absolute path, mtime and a
SHA-256 prefix, and both this probe's own source and the repository HEAD are
stamped once.

NaN POLICY, stated accurately (an earlier draft of this docstring was wrong,
and the parent gate's "NaN on land" comment is wrong too).  MEASURED: neither
the stitched NEMO restarts nor the twin npz files contain any NaN at all --
land is filled with **0.0**, which is a valid float of the right dtype and so
passes every finite check downstream.  ``require_finite`` is therefore a
tripwire that cannot fire on today's inputs, and it is NOT the control that
protects these numbers.  The control that has teeth is the WET-CELL COUNT,
asserted identical across arms, plus the two gate self-checks.  No reduction in
this file is ``nan``-prefixed.

All arithmetic is float64: twin snapshots are stored fp32 and cast up on load
by ``acceptance_gate_90d.load_candidate``; the NEMO restarts are read fp64.

Usage
-----
  JAX_ENABLE_X64=1 .venv/bin/python acc_metric_reconciliation.py
  ...                              --arm label=/path/to/twin.npz     (repeatable;
                                     replaces the built-in registry when given)
  ...                              --split-old arm3_bn2 --split-new armA_new
  ...                              --self-test

SECOND-REVIEW CORRECTIONS (aeaf42, 2026-08-20): the 'no reduction/weight/window moves either era's sign' claim is limited to the THREE variants tested (largest new-era variant +0.8220); '85.6% barotropic' does NOT license 'not a buoyancy change' -- the two eras' full-3D density differs rms 1.3e-3, max 5.9e-2 kg/m3 (dT up to 0.41 K locally); and the irreproducibility verdict is STRONGER than 'unresolved': run logs eliminate every harness env knob (no ABLATION banner, 90 integer day labels pin DINO_DT), leaving an uncommitted working-tree edit as the sole surviving candidate.
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)                       # acc_thermal_wind, acceptance_gate_90d
sys.path.insert(0, os.path.dirname(_DIR))      # rebuild_nemo_restart

import acc_thermal_wind as A  # noqa: E402  (recorded harness, imported)
import acceptance_gate_90d as G  # noqa: E402  (the gate itself, imported)

DAYS = (0, 30, 60, 90)

# PINNED self-test literals (fp64, mesh_mask RUN_TRAJ geometry).  These are
# regression pins, deliberately NOT recomputed from the A.* globals they are
# meant to check -- a self-test that rebuilds its own expectation from the same
# arrays cancels any corruption of them and proves nothing.
NEMO_D90_ACC_SV = 65.3692043752875        # acc_full on RUN_90D_TWIN kt=8640
SELFTEST_BAROTROPIC_SV = 59.20409119673511  # +1e-3 m/s everywhere, mean reduction
SELFTEST_BT_SV = 8.072821946649123          # its barotropic band part, mean reduction

# The saved twin arms of this campaign, oldest first.  Paths are session
# scratchpads: any that no longer exists is reported MISSING and skipped, never
# silently dropped.  Override/extend with --arm label=path.
_S1 = ("/tmp/claude-10257/-home-dbalwada-legoESM/"
       "853ee94c-2651-44cc-ba12-f55bf3ed1979/scratchpad")
_S2 = ("/tmp/claude-10257/-home-dbalwada-legoESM/"
       "bb38bce6-ece1-479a-b2ff-9730aa0697cd/scratchpad")
REGISTRY = [
    ("arm1_pre",      f"{_S1}/gate2/arm1_pre.npz"),
    ("arm2_fixes",    f"{_S1}/gate2/arm2_fixes.npz"),
    ("rec_transport", f"{_S1}/reconcile_ab/arm_transport.npz"),
    ("rec_velocity",  f"{_S1}/reconcile_ab/arm_velocity.npz"),
    ("arm3_bn2",      f"{_S1}/gate2/arm3_bn2.npz"),
    ("arm3_repro",    f"{_S2}/arm3_repro.npz"),
    ("arm4_gdept",    f"{_S2}/ladder/arm4_gdept_true.npz"),
    ("armA_new",      "/tmp/dino_gate90/armA_transport.npz"),
    ("armB_new",      "/tmp/dino_gate90/armB_velocity.npz"),
]


# --------------------------------------------------------------- provenance ---
def stamp(path):
    """'<path>  mtime=<iso>  bytes=<n>  sha256=<12 hex>' for one input file."""
    import datetime
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    st = os.stat(path)
    mt = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"{path}  mtime={mt}  bytes={st.st_size}  sha256={h.hexdigest()[:12]}"


def head_sha():
    try:
        return subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True
                              ).stdout.strip()[:9]
    except Exception:                                    # noqa: BLE001
        return "UNKNOWN"


def require_finite(name, arr, wet):
    """FATAL on any non-finite value inside the wet mask (land NaN is legal)."""
    bad = int(np.count_nonzero(~np.isfinite(np.where(wet, arr, 0.0))))
    if bad:
        raise SystemExit(f"NON-FINITE in wet cells of {name}: {bad} cells -- FATAL")


# ------------------------------------------------------------------ loaders ---
def nemo_u(day):
    """NEMO's own NOW-level ``un`` at ``day`` of the 90-day twin, or None.

    Day 0 is the single stitched restart the bridge itself reads; days 10..90
    are 16 per-rank tiles.  Both patterns are tried (the tiled-only version of
    this lookup is the defect ``acc_trajectory_series`` records in its day-0
    control).
    """
    from rebuild_nemo_restart import rebuild
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    tiled = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart_*.nc"
    single = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart.nc"
    if glob.glob(tiled):
        src, un = tiled, rebuild(tiled, ["un"])["un"]
    elif os.path.exists(single):
        import netCDF4 as nc
        with nc.Dataset(single) as ds:
            # np.asarray on a MaskedArray silently returns .data and DROPS the
            # mask; fill with NaN so a masked value becomes visible instead of
            # arriving as a plausible number.
            src = single
            un = np.ma.filled(np.ma.asarray(ds["un"][0]),
                              np.nan).astype(np.float64)
    else:
        return None, None
    u = np.moveaxis(un, 0, -1)
    require_finite(f"NEMO un day {day}", u, A.umask)
    return u, src


def lego_u(path, day):
    """Candidate u at ``day`` via the gate's own loader, or None if absent."""
    try:
        u = G.load_candidate(path, day)["u"]
    except SystemExit:
        return None
    require_finite(f"{path} u3d_day{day}", u, A.umask)
    return u


# ------------------------------------------------------- metric definitions ---
# The ONE metric under reconciliation is A.acc_full (imported).  The variants
# below exist only to bound how far the number can be moved by re-choosing the
# reduction and the vertical weight; they are NOT alternative truths.
def acc_variant(u, wet_u, e3, reduce_, lon_slice):
    """acc_full's integrand with the weight/reduction/window made explicit."""
    sec = np.einsum("jik,jik,j->i", np.where(wet_u, u, 0.0),
                    np.broadcast_to(e3, u.shape), A.e2u_col) / 1e6
    return float(reduce_(sec[lon_slice]))


VARIANTS = {
    "acc_full (median, e3t_1d, lons 2..-2)":
        lambda u: acc_variant(u, A.umask, A.e3t1d[None, None, :], np.median,
                              slice(2, -2)),
    "mean instead of median":
        lambda u: acc_variant(u, A.umask, A.e3t1d[None, None, :], np.mean,
                              slice(2, -2)),
    "e3t_0 (partial cell) instead of e3t_1d":
        lambda u: acc_variant(u, A.umask, A.e3t0, np.median, slice(2, -2)),
    "all longitudes instead of 2..-2":
        lambda u: acc_variant(u, A.umask, A.e3t1d[None, None, :], np.median,
                              slice(None)),
}


# --------------------------------------------------------------------- main ---
def history(arms, nemo):
    print(f"{'arm':<15}{'d0':>10}{'d30':>10}{'d60':>10}{'d90':>10}"
          f"{'gap d90':>12}   metric")
    print("-" * 79)
    n90 = nemo.get(90)
    for label, path in arms:
        if not os.path.exists(path):
            print(f"{label:<15}{'MISSING -- not on disk: ' + path}")
            continue
        vals = []
        for d in DAYS:
            u = lego_u(path, d)
            vals.append(None if u is None else A.acc_full(u, A.umask))
        cell = [("--" if v is None else f"{v:10.4f}") for v in vals]
        gap = ("--" if (vals[-1] is None or n90 is None)
               else f"{vals[-1] - n90:+12.4f}")
        print(f"{label:<15}" + "".join(f"{c:>10}" if c == "--" else c
                                       for c in cell)
              + f"{gap:>12}   acc_full")
    if n90 is not None:
        row = "".join(f"{nemo[d]:10.4f}" if nemo.get(d) is not None
                      else f"{'--':>10}" for d in DAYS)
        print(f"{'NEMO':<15}{row}{'0.0000':>12}   acc_full  (reference)")
    print("\n(gap = ACC_lego(day 90) - ACC_NEMO(day 90) [Sv]; sign kept: "
          "'+' = legoESM ABOVE NEMO)")


def variants_block(arms, nemo_u90, old, new):
    print("\n=== METRIC-DEFINITION SENSITIVITY (day 90) ===")
    print("Every row re-reduces the SAME saved states.  A metric-definition "
          "owner would need\na row where the old and new arm gaps swap sign or "
          "magnitude class.")
    by = dict(arms)
    cols = [(lbl, lego_u(by[lbl], 90)) for lbl in (old, new)
            if lbl in by and os.path.exists(by[lbl])]
    if len(cols) < 2 or nemo_u90 is None:
        print("  (need both --split-old and --split-new arms plus NEMO day 90)")
        return
    print(f"\n{'variant':<42}{'NEMO':>10}" +
          "".join(f"{lb:>12}" for lb, _ in cols) +
          "".join(f"{'gap ' + lb:>14}" for lb, _ in cols))
    for name, fn in VARIANTS.items():
        n = fn(nemo_u90)
        print(f"{name:<42}{n:10.4f}" +
              "".join(f"{fn(u):12.4f}" for _, u in cols) +
              "".join(f"{fn(u) - n:+14.4f}" for _, u in cols))


def split_block(arms, old, new):
    """Baroclinic / barotropic split of the day-90 band transport, per arm and
    for their difference (A.bc_bt_band, verbatim: bottom-referenced shear vs
    reference-level transport; bc + bt == acc_band identically per longitude).

    REDUCED WITH THE MEAN, NOT THE MEDIAN.  acc_full's median is not additive,
    so median(bc) + median(bt) is neither the median nor the mean band
    transport -- an earlier draft printed exactly that sum as a "total" and it
    was off the real median band transport by ~1 Sv.  acc_thermal_wind's own
    decomposition switches to the mean for this reason.  The median band
    transport is printed alongside, as median(bc + bt), so both are available
    and neither is a sum of two independent medians.

    NOTE ON SCOPE: this is the CHANNEL BAND (rows J0..J1) weighted with the
    real partial-cell e3t_0.  The headline acc_full is the FULL section with
    e3t_1d.  The two integrals are different quantities and the step in one is
    not the step in the other -- both are printed so the difference is visible.
    """
    by = dict(arms)
    print("\n=== BAROCLINIC / BAROTROPIC SPLIT of the day-90 channel-band "
          "transport [Sv per longitude,\n    MEAN over lons 2..-2 -- the "
          "additive reduction; the median band total is shown separately] ===")
    got = {}
    for lbl in (old, new):
        if lbl not in by or not os.path.exists(by[lbl]):
            print(f"  {lbl}: MISSING")
            continue
        u = lego_u(by[lbl], 90)
        if u is None:
            print(f"  {lbl}: no day-90 snapshot")
            continue
        bc, bt = A.bc_bt_band(u, A.umask)
        if not np.allclose(bc + bt, A.acc_band(u, A.umask), rtol=0, atol=1e-10):
            raise SystemExit(f"{lbl}: bc + bt does not reconstruct acc_band "
                             f"-- the split is invalid, FATAL")
        got[lbl] = (float(np.mean(bc[2:-2])), float(np.mean(bt[2:-2])),
                    float(np.median((bc + bt)[2:-2])),
                    A.acc_full(u, A.umask))
        print(f"  {lbl:<15} baroclinic {got[lbl][0]:9.4f}   "
              f"barotropic {got[lbl][1]:9.4f}   band total (mean) "
              f"{got[lbl][0] + got[lbl][1]:9.4f}   band total (median) "
              f"{got[lbl][2]:9.4f}   acc_full {got[lbl][3]:9.4f}")
    # NEMO's OWN band transport, same reduction, so each arm's band number can
    # be read as a GAP against the oracle instead of only against another arm.
    # Without this row the block compares legoESM to legoESM (the exact class of
    # error the campaign has had to retract before).
    un, _src = nemo_u(90)
    if un is not None:
        nbc, nbt = A.bc_bt_band(un, A.umask)
        if not np.allclose(nbc + nbt, A.acc_band(un, A.umask), rtol=0, atol=1e-10):
            raise SystemExit("NEMO: bc + bt does not reconstruct acc_band "
                             "-- the split is invalid, FATAL")
        nrow = (float(np.mean(nbc[2:-2])), float(np.mean(nbt[2:-2])),
                float(np.median((nbc + nbt)[2:-2])), A.acc_full(un, A.umask))
        print(f"  {'NEMO d90':<15} baroclinic {nrow[0]:9.4f}   "
              f"barotropic {nrow[1]:9.4f}   band total (mean) "
              f"{nrow[0] + nrow[1]:9.4f}   band total (median) "
              f"{nrow[2]:9.4f}   acc_full {nrow[3]:9.4f}")
        for lbl in (old, new):
            if lbl in got:
                g = [got[lbl][i] - nrow[i] for i in range(4)]
                print(f"  {'GAP ' + lbl + '-NEMO':<15} baroclinic {g[0]:+9.4f}   "
                      f"barotropic {g[1]:+9.4f}   band total (mean) "
                      f"{g[0] + g[1]:+9.4f}   band total (median) {g[2]:+9.4f}"
                      f"   acc_full {g[3]:+9.4f}")

    if len(got) == 2:
        d = [got[new][i] - got[old][i] for i in range(4)]
        tot = d[0] + d[1]
        print(f"  {'DIFF ' + new + '-' + old:<15} baroclinic {d[0]:+9.4f}   "
              f"barotropic {d[1]:+9.4f}   band total (mean) {tot:+9.4f}   "
              f"band total (median) {d[2]:+9.4f}   acc_full {d[3]:+9.4f}")
        print(f"  barotropic share of the ADDITIVE (mean) band step: "
              f"{100.0 * d[1] / tot:.1f}%   "
              f"[the acc_full step is a DIFFERENT integral: {d[3]:+.4f} Sv]")


def gate_block(arms, nemo_state):
    """The full five-metric gate table for every arm, one row per arm, so the
    ACC step can be read against the density metrics measured on the same
    states (acceptance_gate_90d.metrics, imported)."""
    print("\n=== FIVE-METRIC GATE TABLE at day 90 (acceptance_gate_90d.metrics) ===")
    nm, mask_ref, n_wet = None, None, None
    hdr = f"{'arm':<15}" + "".join(f"{k:>13}" for k in G.KEYS)
    print(hdr)
    print("-" * len(hdr))
    for label, path in arms:
        if not os.path.exists(path):
            continue
        try:
            cand = G.load_candidate(path, 90)
        except SystemExit:
            continue
        wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]
        for f in ("T", "S", "u"):
            require_finite(f"{label}.{f}", cand[f], wet if f != "u" else A.umask)
        # The gate recomputes the NEMO metrics with EACH candidate's own mask.
        # Printing ONE NEMO row for all arms is only legitimate if every arm's
        # land_mask is the same array -- so assert it rather than assume it.
        if mask_ref is None:
            mask_ref, n_wet = cand["land_mask"], int(wet.sum())
            nm = G.metrics(nemo_state, wet)
        elif not np.array_equal(mask_ref, cand["land_mask"]):
            raise SystemExit(f"{label}: land_mask differs from the first arm's "
                             f"-- one shared NEMO row is invalid, FATAL")
        m = G.metrics(cand, wet)
        print(f"{label:<15}" + "".join(f"{m[k]:13.6f}" for k in G.KEYS))
    if nm is not None:
        print(f"{'NEMO d90':<15}" + "".join(f"{nm[k]:13.6f}" for k in G.KEYS))
        print(f"(every arm's land_mask is ASSERTED identical, so one NEMO row "
              f"is valid for all of them;\n wet cells in the 3-D mask: {n_wet})")


def self_test():
    """Non-vacuity: the reconciliation table must SEPARATE two states that
    differ, and must report zero for a state against itself.

    Uses NEMO's own day-90 field, perturbed by a known barotropic increment, so
    the expected answer is analytic: +1e-3 m/s over the full section is
    sum_k e3t_1d * e2u * 1e-3, and the split must place ~all of it in the
    barotropic part.
    """
    u, _ = nemo_u(90)
    if u is None:
        raise SystemExit("self-test needs the NEMO day-90 restart")
    base = A.acc_full(u, A.umask)
    assert A.acc_full(u, A.umask) - base == 0.0, "the metric is not deterministic"
    # PIN the metric itself.  Without this the whole self-test is f(x)-f(x)==0
    # and passes with acc_full replaced by a constant -- the exact vacuity the
    # adversarial review demonstrated on the first draft.
    assert abs(base - NEMO_D90_ACC_SV) < 1e-9, (
        f"acc_full on the NEMO day-90 restart gives {base}, recorded "
        f"{NEMO_D90_ACC_SV} -- this instrument is not the gate's")
    # The VARIANTS row advertised as "acc_full" must BE acc_full, bit for bit,
    # or the sensitivity block is measuring a different metric than the table.
    v = VARIANTS["acc_full (median, e3t_1d, lons 2..-2)"](u)
    assert v == base, f"VARIANTS acc_full row diverges from A.acc_full: {v} vs {base}"
    layout = A.acc_full(np.ascontiguousarray(u), A.umask) - base
    # The analytic check uses the MEAN reduction, which is additive; acc_full's
    # MEDIAN is not (the median can select a different longitude once the
    # sections shift), so a median-based analytic assertion would be wrong.
    mean_acc = VARIANTS["mean instead of median"]
    du = 1e-3
    sec = np.einsum("jik,k,j->i", np.where(A.umask, du, 0.0), A.e3t1d,
                    A.e2u_col) / 1e6
    # NOTE (2nd review): this `expect` IS a recomputation from the same A.*
    # globals and passes even with e3t_1d scaled by 7 (measured 1.14e-13).
    # The check that actually carries the self-test is the PINNED LITERAL
    # asserted below; this line only feeds the printout.
    expect = float(np.mean(sec[2:-2]))
    got = mean_acc(u + du) - mean_acc(u)
    assert abs(got - expect) < 1e-6, (
        f"barotropic perturbation not recovered: {got} vs {expect}")
    assert abs(got - SELFTEST_BAROTROPIC_SV) < 1e-5, (
        f"the geometry this metric integrates over has changed: {got} vs the "
        f"pinned {SELFTEST_BAROTROPIC_SV}")
    bc0, bt0 = A.bc_bt_band(u, A.umask)
    bc1, bt1 = A.bc_bt_band(u + du, A.umask)
    # The split is exact per longitude; assert that identity before reducing.
    assert np.allclose(bc0 + bt0, A.acc_band(u, A.umask), rtol=0, atol=1e-10), \
        "bc + bt does not reconstruct the band transport"
    dbc = float(np.mean(bc1[2:-2]) - np.mean(bc0[2:-2]))
    dbt = float(np.mean(bt1[2:-2]) - np.mean(bt0[2:-2]))
    # abs(dbc) < tol is an algebraic identity and cannot fail; the number that
    # can fail is dbt against its pinned value.
    assert abs(dbc) < 1e-9, f"uniform increment leaked into baroclinic: {dbc}"
    assert abs(dbt - SELFTEST_BT_SV) < 1e-5, (
        f"barotropic band response {dbt} vs the pinned {SELFTEST_BT_SV}")
    print("[self-test] repeated call on the same array: delta ACC exactly 0.0")
    print(f"[self-test] memory-layout rounding quantum: {layout:.2e} Sv "
          f"(the ACC noise floor is {G.FLOORS['acc']} Sv)")
    print(f"[self-test] +{du} m/s barotropic, mean reduction: delta ACC "
          f"{got:.6f} Sv (analytic {expect:.6f})")
    print(f"[self-test] split of that increment (mean reduction): baroclinic "
          f"{dbc:.3e}, barotropic {dbt:.4f} (pinned {SELFTEST_BT_SV})")
    print("SELF-TEST PASS")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                   help="replace the built-in arm registry (repeatable)")
    p.add_argument("--split-old", default="arm3_bn2",
                   help="arm label for the pre-step side of the split blocks")
    p.add_argument("--split-new", default="armA_new",
                   help="arm label for the post-step side of the split blocks")
    p.add_argument("--self-test", action="store_true",
                   help="non-vacuity check on an analytic perturbation")
    args = p.parse_args(argv)

    print(f"repo HEAD = {head_sha()}   fp64 = {np.zeros(1).dtype}")
    print(f"NEMO twin dir = {G.RUN_90D_TWIN}")
    print(f"mesh_mask     = {A.DINO}/RUN_TRAJ/mesh_mask.nc\n")

    if args.self_test:
        return self_test()

    arms = ([(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args.arm]
            if args.arm else REGISTRY)

    print(f"=== PROVENANCE ===\n  {'THIS PROBE':<15}{stamp(__file__)}")
    seen = {}
    for label, path in arms:
        if not os.path.exists(path):
            print(f"  {label:<15}MISSING {path}")
            continue
        st = stamp(path)
        sha = st.split("sha256=")[1]
        print(f"  {label:<15}{st}")
        seen.setdefault(sha, []).append(label)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    for sha, labels in dups.items():
        print(f"  !! DUPLICATE INPUT: {' == '.join(labels)} are the SAME bytes "
              f"(sha256={sha}); they are ONE state, not {len(labels)}")
    print(f"  distinct states: {len(seen)} of {len([1 for _, p in arms if os.path.exists(p)])} "
          f"files")

    # Fatal instrument self-checks, the gate's own (NEMO y10 ACC; band volume).
    print()
    G.instrument_self_checks(A.tmask)

    nemo, nemo_src = {}, {}
    for d in DAYS:
        u, src = nemo_u(d)
        if u is not None:
            nemo[d] = A.acc_full(u, A.umask)
            nemo_src[d] = src
    print("=== NEMO reference (RUN_90D_TWIN restarts, NOW level 'un') ===")
    for d in DAYS:
        print(f"  day {d:3d}  ACC {nemo.get(d, float('nan')):8.4f} Sv   "
              f"{nemo_src.get(d, 'MISSING')}")
    print()

    print("=== ONE METRIC, EVERY SAVED DAY-90 STATE ===")
    history(arms, nemo)
    u90, _ = nemo_u(90)
    variants_block(arms, u90, args.split_old, args.split_new)
    split_block(arms, args.split_old, args.split_new)
    gate_block(arms, G.load_nemo_day90())
    return 0


if __name__ == "__main__":
    sys.exit(main())
