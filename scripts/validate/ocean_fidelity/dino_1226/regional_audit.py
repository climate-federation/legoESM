#!/usr/bin/env python
"""#1455 THE REGIONAL AUDIT: the equatorial and northern bands of the DINO
verdict year, offline from recorded states.

Pre-registration: ``PREREG_regional_audit.md`` (f83dc681e, before any regional
statistic existed).  Read it first; every band, bar, control and prediction
below is registered there and none was chosen after seeing a number.

WHY THIS EXISTS
---------------
The verdict run (``verdict360.py``, a1387f1f7) split the DINO section into three
latitude groups and found the shipped card right in the circumpolar channel and
wrong in the southern basin.  Its third group, "north of band", is **150 of 199
rows** -- everything from 44.6 S to 69.9 N, the entire tropics and the whole
northern hemisphere -- reported as one number.  Nobody has opened it.

This probe opens it into a registered six-band partition, measures each band's
own floor from the two 4-member ensembles rather than borrowing a global one,
and re-reduces the T/S divergence atlas into the same bands.

WHAT IS IMPORTED AND NEVER RE-TYPED
-----------------------------------
Every reduction comes from a recorded harness:

* ``acc_driver_decomp.group_transport`` / ``section_bt_bc`` / ``_avg``
  -- the verdict run's own transport integrand and its exact barotropic /
  baroclinic split;
* ``acc_driver_decomp.LAT_GROUPS`` -- bands P1 and P2 are these slices, taken by
  reference, so a change upstream moves this probe with it;
* ``ts_divergence_atlas`` -- the loaders, the volume weights, ``wrms``,
  ``wcorr``, ``hotspot_set``, ``retention`` and the depth classes;
* ``floor90_ensemble.spread`` -- the ensemble spread, with its NaN propagation;
* ``verdict360`` -- ``K_PREREG``, ``K_WELCH``, ``QUANTUM_MARGIN``,
  ``SATURATION_RATIO_MAX``, ``SATURATION_QUARTERS``, ``ONE_SIDED_DECADES``.

The only arithmetic this file owns is the band partition, the per-band floor
assembly and the verdict vocabulary -- and ``self_test`` checks all three
against cases with known answers, each shown to FAIL on a planted violation.

THE EQUATORIAL BLIND SPOT, DESIGNED AROUND RATHER THAN DISCLAIMED
-----------------------------------------------------------------
``f`` is **exactly 0.0** at T-row 99.  The campaign has already withdrawn a
published 3.51x enrichment that existed only because that row sat in a
denominator where the vorticity flux structurally vanishes.  This probe computes
**no vorticity-projection, no Coriolis-normalised and no f-weighted statistic
anywhere**; control K11 asserts ``f[99] == 0.0``, prints it, and prints the
row-99 denominator of every normalised map the probe emits so a reader can see
it is not structurally zero.

Usage
-----
  JAX_ENABLE_X64=1 regional_audit.py --self-test      # controls only, no data
  JAX_ENABLE_X64=1 regional_audit.py --out-dir DIR    # the audit + figures
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_driver_decomp as D          # noqa: E402  group_transport, section_bt_bc, _avg
import acc_thermal_wind as A           # noqa: E402  mesh, masks, J0/J1
import acceptance_gate_90d as G        # noqa: E402  load_candidate (stamp refusals)
import floor90_ensemble as F           # noqa: E402  spread()
import ts_divergence_atlas as X        # noqa: E402  loaders, weights, wrms/wcorr
import verdict360 as V                 # noqa: E402  the registered verdict constants

# ---------------------------------------------------------------- registered ---
# PREREG sec.2.  P1 and P2 are TAKEN FROM the recorded three-group split rather
# than re-typed as literals; P3..P6 refine LAT_GROUPS[2] and must sum back to it.
_P1 = D.LAT_GROUPS[0][1]                       # slice(0, A.J0)      south of band
_P2 = D.LAT_GROUPS[1][1]                       # slice(A.J0, A.J1+1) channel band
_NORTH = D.LAT_GROUPS[2][1]                    # slice(A.J1+1, A.NY) everything else

# The registered latitude cuts, resolved to rows ONCE, from the mesh itself.
_LAT = np.asarray(A.gphit, dtype=np.float64)[:, 25]
_EQ20_LO, _EQ20_HI = 79, 119                   # |lat| <= 20 deg
_EQ10_LO, _EQ10_HI = 89, 109                   # |lat| <= 10 deg
_SUBTROP_N_HI = 150                            # +45.353 deg, the channel's mirror
# The structural zero, needed here because the registered cuts are defined
# RELATIVE to it.  Registered in PREREG sec.7 and re-asserted by K11.
OMEGA = 7.292115e-5
STRUCTURAL_ZERO_ROW = 99
# ASSERTED, not merely printed.  Code review of 17881d91d planted a one-row
# shift of the equatorial band (79-119 -> 80-120, i.e. -18.7..+20.6 deg,
# asymmetric about the f=0 row) and the ENTIRE self-test passed: K7 only checks
# that the six bands partition 199 rows and that E10 is inside P4, and both
# survive any shift.  The registered latitudes were printed and never checked.
# These three lines catch every off-by-one in the registered cuts.
assert _EQ20_LO + _EQ20_HI == 2 * STRUCTURAL_ZERO_ROW, \
    "P4 is not symmetric about the f=0 row"
assert _EQ10_LO + _EQ10_HI == 2 * STRUCTURAL_ZERO_ROW, \
    "E10 is not symmetric about the f=0 row"
assert _SUBTROP_N_HI == 2 * STRUCTURAL_ZERO_ROW - A.J1, \
    "P5's north cut is not the channel's southern mirror"

BANDS = (
    ("P1 south of band",        _P1),
    ("P2 channel band",         _P2),
    ("P3 southern subtropics",  slice(_NORTH.start, _EQ20_LO)),
    ("P4 equatorial +/-20",     slice(_EQ20_LO, _EQ20_HI + 1)),
    ("P5 northern subtropics",  slice(_EQ20_HI + 1, _SUBTROP_N_HI + 1)),
    ("P6 northern subpolar",    slice(_SUBTROP_N_HI + 1, A.NY)),
)
# Reported ALONGSIDE P4, never summed with it (PREREG sec.2).
NESTED = (("E10 equatorial +/-10", slice(_EQ10_LO, _EQ10_HI + 1)),)
ALL_BANDS = BANDS + NESTED
FULL = slice(0, A.NY)

HORIZONS = (90, 180, 270, 360)                 # registered, scored
CONTROL_DAYS = (0, 10)                         # controls only, never scored
KEEP_DAYS = (10, 90, 180, 270, 360)            # difference fields retained
N_MEM = X.N_MEM

# The verdict run's own recorded day-360 "south of band" gap, reproduced by K9.
D360_P1_RECORDED_SV = X.D360_SOUTH_GAP_RECORDED_SV      # -0.95191
D360_P1_TOL_SV = X.D360_SOUTH_GAP_TOL_SV                # 1e-4

# PREREG sec.10: a band over this many of its OWN floors is a NEW REGIONAL
# FINDING and needs dual sign-off.  Registered before any number existed.
ESCALATION_FLOORS = 5.0

_QUIET = io.StringIO()


def _f64(a):
    return np.asarray(a, dtype=np.float64)


# ------------------------------------------------------------------ geometry ---
def band_row_mask(rows):
    """(NY,) boolean of the T-rows in `rows`."""
    m = np.zeros(A.NY, dtype=bool)
    m[rows] = True
    return m


def band_cell_mask(rows):
    """(NY,NX,NZ) boolean: the band's rows, everywhere in x and z."""
    return band_row_mask(rows)[:, None, None] & np.ones(
        (1, A.NX, A.NZ), dtype=bool)


DEPTH_SEL = {name: sel for name, sel in X.DEPTH_CLASSES}


def depth_cell_mask(name):
    return np.ones((A.NY, A.NX, 1), dtype=bool) & DEPTH_SEL[name][None, None, :]


# ---------------------------------------------------------------- reductions ---
def band_transport(u, wet_u, rows):
    """Mean-reduced band transport [Sv], the verdict table's own reduction.

    ``group_transport`` gives Sv per longitude with the recorded e3t_1d weights
    and ``A.e2u_col`` widths; ``D._avg`` is the verdict table's mean over
    longitudes with two columns trimmed at each end.  The MEDIAN reduction is
    deliberately not offered: the verdict document states in as many words that
    its median "ACC" row is not additive over bands, and every additive claim in
    this audit would be void under it.
    """
    return D._avg(D.group_transport(u, wet_u, rows))


def band_bt_bc(u, wet_u, rows):
    """(bt, bc) mean-reduced [Sv] for `rows`, with bt+bc == the band total.

    ``bt = u_bottom * H`` is the REFERENCE-LEVEL transport -- what a
    depth-uniform field equal to the deepest wet value would carry.  It is NOT a
    true depth mean and must not be read as one; the name is
    ``acc_driver_decomp``'s and its docstring is the authority.
    """
    bt, bc = D.section_bt_bc(u, wet_u, rows=rows)
    return D._avg(bt), D._avg(bc)


def transport_row(st, wet_u):
    """Every band statistic for one state.  Keys are the band labels."""
    out = {}
    for name, rows in ALL_BANDS:
        tot = band_transport(st["u"], wet_u, rows)
        bt, bc = band_bt_bc(st["u"], wet_u, rows)
        # K8, asserted per band per state rather than once: the split is exact
        # by construction, and an assertion that only runs in the self-test
        # cannot catch a shape bug that appears on real data.
        if abs(bt + bc - tot) > 1e-9:
            raise SystemExit(
                f"FATAL K8: {name} bt+bc = {bt + bc:.12f} but the band total is "
                f"{tot:.12f} Sv -- the barotropic/baroclinic split is not exact")
        out[name] = tot
        out[name + " |bt"] = bt
        out[name + " |bc"] = bc
    out["FULL section"] = band_transport(st["u"], wet_u, FULL)
    return out


def band_depth_sel(rows, depth_name, wet):
    """The 3-D selection for one (band, depth class), wet cells only.

    An empty selection is FATAL inside ``X.wrms``; this returns the mask and
    lets that abort fire rather than silently emptying a table cell.
    """
    return band_cell_mask(rows) & depth_cell_mask(depth_name) & wet


# --------------------------------------------------------------------- floors ---
def spread_of(vals):
    """(max-pairwise, sample std) -- the recorded statistic, std primary."""
    return F.spread(vals)


def two_sided_floor(lego_vals, nemo_vals):
    """RSS of the two sides' measured sample stds, and each side's own."""
    ls = spread_of(lego_vals)[1]
    ns = spread_of(nemo_vals)[1]
    return float(np.sqrt(ls ** 2 + ns ** 2)), ls, ns


def saturated(spread_by_side_day):
    """Saturated iff BOTH sides' spreads are under SATURATION_RATIO_MAX over
    BOTH registered quarters.

    Constructed exactly as ``verdict360.saturation_table``: PER SIDE, not on the
    combined floor.  A combined-floor test is looser -- one side can still be
    growing while the RSS looks flat because the other side dominates it -- and
    matching the recorded harness is what makes this audit's flags comparable
    to the verdict run's.  Two quarters, never one: the recorded false positive
    is NEMO's ACC spread falling from day 30 to 60 (ratio 0.52, "saturated")
    and then growing ~300x as the Asselin filter stopped dominating.
    """
    worst = []
    for a, b in V.SATURATION_QUARTERS:
        for side in ("lego", "nemo"):
            if (side, a) not in spread_by_side_day or (side, b) not in spread_by_side_day:
                return False, "horizon missing"
            lo = spread_by_side_day[(side, a)]
            r = (spread_by_side_day[(side, b)] / lo) if lo > 0 else float("inf")
            if not (r < V.SATURATION_RATIO_MAX):
                worst.append(f"{side} {a}->{b} {r:.2f}")
    if worst:
        return False, "; ".join(worst)
    return True, ""


def u_is_material(ratio, spread_by_side_day, day):
    """Could the `u` flag ACTUALLY overturn this row's `no`?

    Verbatim the recorded ``verdict360.unsaturated_materiality`` arithmetic.
    "Not saturated" otherwise voids EVERY `no` in the table for free, which is
    not honest: a gap sitting 15x its floor needs the floor to grow ~8x before
    it becomes a YES, and a floor whose last-quarter growth is ~1.0x will not
    do that.  The flag is attached to a verdict only where the arithmetic lets
    it matter; the saturation FACT is printed either way.
    """
    x2yes = ratio / V.K_PREREG
    g = []
    for side in ("lego", "nemo"):
        a = spread_by_side_day.get((side, 270), 0.0)
        b = spread_by_side_day.get((side, day), 0.0)
        g.append(b / a if a > 0 else float("inf"))
    gl = max(g)
    return (gl > 1.0 and x2yes <= gl ** V.UNSAT_CREDIT_QUARTERS), x2yes, gl


def classify(gap, floor, lego_spread, nemo_spread, quantum, is_saturated):
    """The registered verdict vocabulary (PREREG sec.6).

    Returns (verdict string, ratio or None, flag string).  UNMEASURABLE is NOT
    a pass -- it withholds a verdict in both directions.
    """
    flags = []
    if lego_spread > 0.0 and nemo_spread > 0.0:
        dec = abs(np.log10(lego_spread / nemo_spread))
        if dec > V.ONE_SIDED_DECADES:
            flags.append("1s")
    if quantum is not None and lego_spread <= V.QUANTUM_MARGIN * quantum:
        return "UNMEASURABLE", None, ",".join(flags + ["q"])
    if not floor > 0.0:
        return "UNMEASURABLE", None, ",".join(flags + ["nofloor"])
    ratio = abs(gap) / floor
    if not is_saturated:
        flags.append("u")
    if ratio <= V.K_PREREG:
        # An UNSATURATED yes still stands: the runs did not separate even
        # though the denominator had room to grow.  Registered asymmetry.
        return "INDISTINGUISHABLE", ratio, ",".join(flags)
    # An UNSATURATED `no` does NOT stand -- the denominator is still growing, so
    # the multiple is an upper bound.  The flag travels with the row.
    return f"gap-at-{ratio:.3g}x", ratio, ",".join(flags)


# -------------------------------------------------------------------- loaders ---
def load_lego(i, day):
    """The gate's own loader, with its stamp refusals.  Its per-call banner is
    suppressed AFTER control K1 has printed the stamps once per member -- 160
    repeats of the same five lines buries the numbers, and the refusals still
    fire because they raise rather than print.
    """
    with contextlib.redirect_stdout(_QUIET):
        return G.load_candidate(X.lego_npz(i), day)


def load_nemo(i, day):
    return X.load_nemo(i, day, fields=("tn", "sn", "un"))


# ------------------------------------------------------------------- controls ---
def control_geometry():
    """K7 (partition identity) and K11 (the structural zero), on the mesh alone.

    Runs WITHOUT data so a partition defect is caught before an hour of loads.
    """
    print("[K7 PARTITION]")
    covered = np.zeros(A.NY, dtype=int)
    for name, rows in BANDS:
        covered[rows] += 1
        j = np.arange(A.NY)[rows]
        print(f"  {name:<26} rows {j.min():3d}-{j.max():3d} "
              f"({len(j):3d} rows)  lat {_LAT[j.min()]:+7.2f} .. "
              f"{_LAT[j.max()]:+7.2f}")
    if not np.array_equal(covered, np.ones(A.NY, dtype=int)):
        bad = np.where(covered != 1)[0]
        raise SystemExit(f"FATAL K7: the six bands are not a partition -- rows "
                         f"{bad.tolist()} covered {covered[bad].tolist()} times")
    print(f"  partition: all {A.NY} rows covered exactly once")
    for name, rows in NESTED:
        j = np.arange(A.NY)[rows]
        p4 = np.arange(A.NY)[BANDS[3][1]]
        if not set(j.tolist()) <= set(p4.tolist()):
            raise SystemExit(f"FATAL K7: {name} is not nested inside P4")
        print(f"  {name:<26} rows {j.min():3d}-{j.max():3d} "
              f"({len(j):3d} rows)  lat {_LAT[j.min()]:+7.2f} .. "
              f"{_LAT[j.max()]:+7.2f}  [nested in P4, never summed with it]")
    # PREREG sec.7 item 3, the heterogeneous-band masking lesson: a dry row of
    # exact zeros sitting beside wet values inverted a verdict in this campaign
    # (94% -> 37%).  The census below finds the dry rows and refuses any that
    # are not the two DOMAIN WALLS.
    #
    # The two walls are a NAMED exception with a measured reason, not an
    # argued one.  Row 0 (south wall) is already inside the recorded P1 slice
    # this probe imports, and row 198 (north wall) is its exact mirror.  They
    # are admitted because every reduction in this probe is a width- or
    # volume-weighted SUM over rows, in which a dry row contributes exactly
    # 0.0 -- never a MEAN over rows, which is the reduction the recorded
    # failure used and the one a dry row dilutes.  That is not taken on trust:
    # control K3c (in ``control_plant``, on a real state) poisons BOTH entire
    # wall rows with 1e6 and requires every reported band statistic to move by
    # exactly 0.0.
    print("[K3b DRY-ROW CENSUS]")
    wet_cols = A.tmask[:, :, 0].sum(axis=1)
    dry_all = np.where(wet_cols == 0)[0]
    walls = {0, A.NY - 1}
    for name, rows in ALL_BANDS:
        j = np.arange(A.NY)[rows]
        dry = j[wet_cols[j] == 0]
        tag = "" if dry.size == 0 else f"  DRY ROWS {dry.tolist()} (domain wall)"
        print(f"  {name:<26} min wet cols/row {int(wet_cols[j].min()):3d}{tag}")
        bad = [int(x) for x in dry if int(x) not in walls]
        if bad:
            raise SystemExit(
                f"FATAL K3b: {name} contains fully dry INTERIOR row(s) {bad} -- "
                f"a dry row of exact zeros beside wet values has inverted a "
                f"verdict in this campaign and the band is refused, not averaged")
    if set(int(x) for x in dry_all) != walls:
        raise SystemExit(
            f"FATAL K3b: the dry rows are {dry_all.tolist()}, but the only ones "
            f"this probe admits are the two domain walls {sorted(walls)} -- the "
            f"exception is NAMED, so a new dry row is a refusal, not a widening")
    print(f"  dry rows are EXACTLY the two domain walls {sorted(walls)}; "
          f"admitted, and K3c measures that they move nothing")
    print("[K11 STRUCTURAL ZERO]")
    f = 2.0 * OMEGA * np.sin(np.deg2rad(_LAT))
    if f[STRUCTURAL_ZERO_ROW] != 0.0:
        raise SystemExit(f"FATAL K11: f at row {STRUCTURAL_ZERO_ROW} is "
                         f"{f[STRUCTURAL_ZERO_ROW]!r}, not exactly 0.0 -- the "
                         f"registered structural-zero row is not where the "
                         f"registration says it is")
    print(f"  f[row {STRUCTURAL_ZERO_ROW}] = {f[STRUCTURAL_ZERO_ROW]:.1e} s-1 "
          f"EXACTLY (lat {_LAT[STRUCTURAL_ZERO_ROW]:+.6f}); neighbours "
          f"{f[STRUCTURAL_ZERO_ROW - 1]:+.3e} / {f[STRUCTURAL_ZERO_ROW + 1]:+.3e}")
    print("  this probe computes NO vorticity-projection, NO Coriolis-normalised")
    print("  and NO f-weighted statistic; the row is therefore excluded from")
    print("  nothing by exclusion -- it is excluded by never entering a")
    print("  denominator.  The denominators it DOES enter, printed so a reader")
    print("  can see they are not structurally zero:")
    e2 = _f64(A.e2u_col)[STRUCTURAL_ZERO_ROW]
    h = float(np.sum(np.where(A.umask[STRUCTURAL_ZERO_ROW], 1.0, 0.0)
                     * _f64(A.e3t1d)[None, :]))
    vol = float(np.sum(np.where(A.tmask[STRUCTURAL_ZERO_ROW],
                                (_f64(A.mm["e1t"][0]).squeeze()
                                 * _f64(A.mm["e2t"][0]).squeeze()
                                 )[STRUCTURAL_ZERO_ROW][:, None]
                                * _f64(A.e3t0)[STRUCTURAL_ZERO_ROW], 0.0)))
    print(f"    transport width e2u  = {e2:.6e} m      (nonzero)")
    print(f"    summed wet thickness = {h:.6e} m      (nonzero)")
    print(f"    row wet volume       = {vol:.6e} m3   (nonzero)")
    print(f"    wet u-cells in the row = "
          f"{int(A.umask[STRUCTURAL_ZERO_ROW].sum())} of {A.NX * A.NZ}")
    return {"f_row99": float(f[STRUCTURAL_ZERO_ROW]), "e2u_row99": float(e2),
            "row99_volume_m3": vol}


def control_stamps():
    """K1 + K2, delegated to the atlas's own provenance and clock controls.

    Delegated rather than re-implemented: these two refusals are the campaign's
    gate against a stampless artifact, and a second copy is a second thing to
    drift.  ``control_provenance`` refuses the ladder on VALUE and refuses a
    member outside the certified NEMO tree unless it is the one named exception;
    ``control_clock`` supplies the missing seasonal stamp from the oracle
    restart, which is what makes ``G.load_candidate`` loadable at all.
    """
    prov = X.control_provenance()
    clock = X.control_clock()
    return prov, clock


def control_day0(lego0, nemo0, wet, wet_u, rows_by_side, quantum):
    """K3: at day 0 the two models hold the SAME state.

    THE PREVIOUS VERSION OF THIS CONTROL COULD NOT FAIL, and the code review of
    17881d91d proved it.  It compared the day-0 band transport gap against that
    band's fp32 storage quantum -- but the legoESM day-0 snapshot IS
    ``fp32(NEMO day-0)``, so the gap and the quantum are the SAME NUMBER with a
    sign, and the assertion was |x| <= 10|x|.  It read 1.000000 in every one of
    the seven bands and the matching mantissas were printed without comment.
    (It also silently applied ``QUANTUM_MARGIN`` where PREREG sec.8 registers
    the bare quantum -- an undisclosed 10x loosening.)

    Replaced by the statement that is actually true and actually checkable:
    the two sides are BIT-IDENTICAL at day 0 on the wet mask, once NEMO is cast
    to the precision legoESM stored.  That can fail -- a re-ingested restart, a
    different member, a changed mask all break it -- and it makes the identity
    the old control was hiding explicit instead of dressing it as agreement.
    """
    print("[K3 DAY-0 IDENTITY]")
    ident = (("T", lego0["T"], nemo0["T"], wet),
             ("S", lego0["S"], nemo0["S"], wet),
             ("u", lego0["u"], nemo0["u"], wet_u))
    for name, a, b, m in ident:
        b32 = np.asarray(b, dtype=np.float32).astype(np.float64)
        if not np.array_equal(a[m], b32[m]):
            bad = int(np.sum(a[m] != b32[m]))
            raise SystemExit(
                f"FATAL K3: legoESM day-0 {name} differs from fp32(NEMO day-0) "
                f"in {bad} wet cells (max {np.abs(a[m] - b32[m]).max():.3e}) -- "
                f"the two sides are NOT starting from the same state")
        print(f"  {name}: legoESM day-0 is BIT-IDENTICAL to fp32(NEMO day-0) on "
              f"all {int(m.sum())} wet cells")
    print("  CONSEQUENCE, stated so nobody reads the next line as agreement:")
    print("  the day-0 band transport gap below is EXACTLY this band's fp32")
    print("  storage quantum, by construction.  It is an identity, not a test.")
    worst = 0.0
    for name, _ in ALL_BANDS:
        gap = (rows_by_side["lego"][0][0][name]
               - rows_by_side["nemo"][0][0][name])
        worst = max(worst, abs(gap))
        q = quantum[name]
        r = abs(gap) / q if q > 0 else float("nan")
        print(f"  {name:<26} day-0 gap {gap:+.6e} Sv   quantum {q:.6e}   "
              f"ratio {r:.6f}")
    print(f"  worst day-0 band gap {worst:.3e} Sv")
    return worst


def control_finite(tag, values):
    """K4: NaN is FATAL, never skipped and never nan-reduced away."""
    arr = np.asarray(list(values), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        n = int((~np.isfinite(arr)).sum())
        raise SystemExit(f"FATAL K4: {n} non-finite value(s) in {tag} -- a "
                         f"blown-up member or a masked division, not a cell to "
                         f"skip")


def control_plant(st, wet_u, wet_t, w):
    """K5 + K6: the non-vacuity pair.

    K5  a DRY cell given u = 1e6 must move EVERY band transport by exactly 0.0.
    K6  the same plant on a WET cell inside P4 must move P4 by orders more than
        any floor -- otherwise K5 passes for a reason unrelated to its claim.

    The campaign's own recorded failure here is a DOUBLE MASK: a land control
    whose poison was already zeroed by the volume weights before the band mask
    was consulted, so it passed while proving nothing.  Both legs are therefore
    planted on the SAME array the reduction reads, and K6 is required to move.
    """
    print("[K5/K6 PLANTED VIOLATIONS]")
    # LAYOUT MATCHING, and it is not cosmetic.  The NEMO loader hands back a
    # ``np.moveaxis`` VIEW, so ``.copy()`` changes the memory layout from
    # strided to C-contiguous.  ``np.einsum`` then sums in a different order and
    # the SAME VALUES give a band transport differing by up to 2.1e-14 Sv --
    # measured below, not asserted.  A control that compares a view against a
    # copy therefore reports a "leak" of 2e-14 Sv that is pure summation order.
    # Both arms are taken from ONE contiguous baseline so the planted value is
    # the only difference between them.
    clean = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v)
             for k, v in st.items()}
    base = transport_row(clean, wet_u)
    layout_shift = max(abs(transport_row(st, wet_u)[k] - base[k]) for k in base)
    print(f"  [layout] the identical state through a strided view vs a "
          f"C-contiguous copy moves a band transport by {layout_shift:.3e} Sv "
          f"-- summation order in einsum, NOT a mask leak; recorded because it "
          f"is the size a sloppy A/B would manufacture")
    dry = np.argwhere(~wet_u)
    if dry.size == 0:
        raise SystemExit("FATAL K5: no dry u-cell exists to plant in")
    jd, id_, kd = dry[len(dry) // 2]
    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in clean.items()}
    poisoned["u"][jd, id_, kd] = 1e6
    after = transport_row(poisoned, wet_u)
    moved = {k: abs(after[k] - base[k]) for k in base}
    worst = max(moved.values())
    print(f"  dry plant at (row {jd}, col {id_}, lev {kd}) u=1e6: "
          f"worst band move {worst:.3e} Sv")
    if worst != 0.0:
        raise SystemExit(f"FATAL K5: a dry cell moved a band transport by "
                         f"{worst:.3e} Sv -- the wet mask leaks")
    # K6 -- the same plant, on a wet cell inside P4, must move P4.
    p4_rows = np.arange(A.NY)[BANDS[3][1]]
    wet_in_p4 = np.argwhere(wet_u[p4_rows.min():p4_rows.max() + 1])
    if wet_in_p4.size == 0:
        raise SystemExit("FATAL K6: no wet u-cell inside P4")
    jw, iw, kw = wet_in_p4[len(wet_in_p4) // 2]
    jw += p4_rows.min()
    poisoned2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in clean.items()}
    poisoned2["u"][jw, iw, kw] = 1e6
    after2 = transport_row(poisoned2, wet_u)
    d_p4 = abs(after2[BANDS[3][0]] - base[BANDS[3][0]])
    d_p1 = abs(after2[BANDS[0][0]] - base[BANDS[0][0]])
    print(f"  wet plant at (row {jw}, col {iw}, lev {kw}) u=1e6: "
          f"P4 moves {d_p4:.3e} Sv, P1 moves {d_p1:.3e} Sv")
    if not d_p4 > 1.0:
        raise SystemExit(f"FATAL K6: the wet plant moved P4 by only "
                         f"{d_p4:.3e} Sv -- K5 is vacuous")
    if d_p1 != 0.0:
        raise SystemExit(f"FATAL K6: a plant inside P4 moved P1 by "
                         f"{d_p1:.3e} Sv -- the bands are not disjoint")
    # THE T/S LEG, REBUILT.  The previous K5b/K6b were VACUOUS and the code
    # review of 17881d91d proved it twice over.  (1) It drew a "dry" T-cell from
    # the whole domain and scored it through a P4-UPPER selection; the cell it
    # picked sat at 4253 m, so the DEPTH mask killed the poison before the wet
    # mask was ever consulted -- the campaign's own double-mask defect, in the
    # control that claimed to design against it.  (2) Even with an honest dry
    # cell inside P4-upper it still proved nothing, because ``build_weights``
    # already zeroes the volume weight on land, so the ``& wet`` term in the
    # selection is redundant with the weight and the leg only re-tested the
    # imported atlas's own C5.
    #
    # What THIS file owns in the T/S path is the band-AND-depth INTERSECTION,
    # so that is what is planted against: a poison in one band's cells must not
    # reach another band's rms, and must reach its own.
    print("[K5t/K6t BAND-DEPTH SELECTION]")
    sel_p4u = band_depth_sel(BANDS[3][1], X.DEPTH_CLASSES[0][0], wet_t)
    sel_p5u = band_depth_sel(BANDS[4][1], X.DEPTH_CLASSES[0][0], wet_t)
    sel_p4a = band_depth_sel(BANDS[3][1], X.DEPTH_CLASSES[2][0], wet_t)
    for nm, sl in (("P4-upper", sel_p4u), ("P5-upper", sel_p5u),
                   ("P4-abyss", sel_p4a)):
        if not sl.any():
            raise SystemExit(f"FATAL K5t: the {nm} selection is empty")
    base_d = np.zeros((A.NY, A.NX, A.NZ), dtype=np.float64) + 1e-9
    base_rms = X.wrms(base_d, w, sel_p4u)
    for nm, other in (("another BAND (P5-upper)", sel_p5u),
                      ("another DEPTH CLASS (P4-abyss)", sel_p4a)):
        d_out = base_d.copy()
        cells = np.argwhere(other & ~sel_p4u)
        if cells.size == 0:
            raise SystemExit(f"FATAL K5t: {nm} does not disjointly exist")
        d_out[tuple(cells[len(cells) // 2])] = 1e6
        got = X.wrms(d_out, w, sel_p4u)
        print(f"  poison in {nm}: P4-upper rms "
              f"{base_rms:.6e} -> {got:.6e}")
        if got != base_rms:
            raise SystemExit(
                f"FATAL K5t: a poison in {nm} reached the P4-upper rms -- the "
                f"band-and-depth selection is not disjoint")
    d_in = base_d.copy()
    own = np.argwhere(sel_p4u)
    d_in[tuple(own[len(own) // 2])] = 1e6
    wet_rms = X.wrms(d_in, w, sel_p4u)
    print(f"  poison INSIDE P4-upper: rms {base_rms:.6e} -> {wet_rms:.6e}")
    if not wet_rms > 1.0:
        raise SystemExit("FATAL K6t: a poison inside P4-upper did not move its "
                         "own rms -- K5t is vacuous")

    # K3c -- the named exception, MEASURED.  Poison BOTH entire domain-wall
    # rows and require every reported band statistic to move by exactly 0.0.
    # This is what admits rows 0 and 198 into P1/P6; without it the admission
    # would be an argument about reductions rather than a measurement of them.
    print("[K3c DRY WALL ROWS]")
    poisoned3 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in clean.items()}
    poisoned3["u"][0, :, :] = 1e6
    poisoned3["u"][A.NY - 1, :, :] = 1e6
    after3 = transport_row(poisoned3, wet_u)
    worst3 = max(abs(after3[k] - base[k]) for k in base)
    print(f"  both dry wall rows (0 and {A.NY - 1}) set to u=1e6 in full: "
          f"worst band move {worst3:.3e} Sv")
    if worst3 != 0.0:
        raise SystemExit(
            f"FATAL K3c: poisoning the dry domain-wall rows moved a band "
            f"statistic by {worst3:.3e} Sv -- the walls are NOT inert in these "
            f"reductions and their admission into P1/P6 is not supportable")
    # and the same on the volume weights the T/S reductions read
    d3 = np.zeros((A.NY, A.NX, A.NZ), dtype=np.float64) + 1e-9
    d3[0, :, :] = 1e6
    d3[A.NY - 1, :, :] = 1e6
    sel_p6 = band_depth_sel(BANDS[5][1], X.DEPTH_CLASSES[0][0], wet_t)
    rms3 = X.wrms(d3, w, sel_p6)
    if abs(rms3 - 1e-9) > 1e-15:
        raise SystemExit(f"FATAL K3c: the poisoned wall rows reached the P6 "
                         f"volume-weighted rms ({rms3:.3e}, want 1e-09)")
    print(f"  the same poison on the volume weights: P6-upper rms "
          f"{rms3:.3e} (want 1.000e-09) -- the walls carry no volume")
    return {"dry_worst_sv": float(worst), "wet_p4_move_sv": float(d_p4),
            "p4upper_base_rms": float(base_rms),
            "p4upper_wet_rms": float(wet_rms),
            "dry_wall_worst_sv": float(worst3),
            "layout_shift_sv": float(layout_shift)}


def control_known_answer(rows_by_side, recorded=None):
    """K9: the day-360 P1 gap must reproduce the verdict run's own recorded
    -0.95191 Sv, compared IN CODE.  A control that only checks this probe's
    bands against this probe's own sum is an algebraic identity of two linear
    reductions and cannot fail -- which is exactly what the atlas's first C7
    was, and it was retracted."""
    rec = D360_P1_RECORDED_SV if recorded is None else recorded
    got = (rows_by_side["lego"][0][360][BANDS[0][0]]
           - rows_by_side["nemo"][0][360][BANDS[0][0]])
    print("[K9 KNOWN ANSWER]")
    print(f"  day-360 P1 gap  measured {got:+.5f} Sv   recorded {rec:+.5f} Sv"
          f"   |diff| {abs(got - rec):.2e}")
    if abs(got - rec) > D360_P1_TOL_SV:
        raise SystemExit(
            f"FATAL K9: this probe's day-360 south-of-band gap is {got:.5f} Sv "
            f"but the verdict run recorded {rec:.5f} Sv -- the transport "
            f"reduction is not the recorded one")
    return float(got)


def control_full_section(rows_by_side, day):
    """K7b: the six bands must sum to the FULL section, on real data.

    The mean reduction is linear in the rows, so this is an identity -- and it
    is kept for exactly the reason the atlas kept its C7a: to catch a future
    non-linear refactor or a mis-sliced band, not because it can fail on
    physics.  Labelled as such rather than sold as a control that can fail.
    """
    worst = 0.0
    for side in ("lego", "nemo"):
        for i in range(N_MEM):
            r = rows_by_side[side][i][day]
            s = sum(r[n] for n, _ in BANDS)
            worst = max(worst, abs(s - r["FULL section"]))
    if worst > 1e-9:
        raise SystemExit(f"FATAL K7b: the six bands sum to within {worst:.3e} Sv "
                         f"of the full section at day {day} -- not a partition")
    return worst


def control_dtype(*arrays):
    """K10."""
    for a in arrays:
        if np.asarray(a).dtype != np.float64:
            raise SystemExit(f"FATAL K10: a comparison array is "
                             f"{np.asarray(a).dtype}, not float64")


def ts_quantum(nemo_st, wet, w):
    """Per (band, depth class, field): the fp32 STORAGE quantum of the rms.

    Measured the same way ``band_quantum`` measures the transport's -- on the
    ORACLE side, which is the one stored at full precision -- as the
    volume-weighted rms of ``NEMO - fp32(NEMO)``.  Without it the T/S table
    prints multiples up to 7e8 that are a floor sitting at the storage quantum,
    and a paragraph telling the reader to discount a column is not a bar.
    """
    out = {}
    for fld in ("T", "S"):
        a = _f64(nemo_st[fld])
        d = np.where(wet, a - np.asarray(a, dtype=np.float32).astype(np.float64),
                     0.0)
        for bname, rows in ALL_BANDS:
            for dname, _ in X.DEPTH_CLASSES:
                sel = band_depth_sel(rows, dname, wet)
                out[(bname, dname, fld)] = X.wrms(d, w, sel)
    return out


def jet_quantum(nemo_st, wet_u):
    """Per (band, level): the fp32 storage quantum of the zonal-mean profile."""
    lo = {"u": np.asarray(nemo_st["u"], dtype=np.float32).astype(np.float64)}
    a = jet_profiles([nemo_st], [nemo_st], wet_u)
    b = jet_profiles([lo], [lo], wet_u)
    return {t: np.abs(a[(t, "lego")] - b[(t, "lego")])
            for t in ("equator row", "E10 +/-10", "P4 +/-20")}


def band_quantum(nemo_st, wet_u):
    """The per-band fp32 storage quantum: what round-tripping a NEMO (fp64)
    state through float32 costs each band's transport.

    Measured on the ORACLE side because that is the side stored at full
    precision -- round-tripping the legoESM snapshots, which are ALREADY fp32,
    is idempotent and would report a quantum of exactly zero, making every band
    look measurable.  Same construction as ``verdict360.fp32_quantum``.
    """
    lo = {k: (np.asarray(v, dtype=np.float32).astype(np.float64)
              if k in ("T", "S", "u") else v) for k, v in nemo_st.items()}
    a, b = transport_row(nemo_st, wet_u), transport_row(lo, wet_u)
    return {k: abs(a[k] - b[k]) for k in a}


# ------------------------------------------------------------------ self-test ---
def self_test():
    """Every piece of arithmetic this file owns, against a known answer, and
    each shown to FAIL on a planted violation."""
    print("=" * 78)
    print("SELF-TEST -- the arithmetic this file owns, against known answers")
    print("=" * 78)

    # 1. the partition, on the mesh alone
    control_geometry()

    # 2. the floor: RSS of two sample stds, and the sqrt(2) equal-wobble limit
    fl, ls, ns = two_sided_floor([1.0, 1.1, 0.9, 1.0], [2.0, 2.2, 1.8, 2.0])
    ref = float(np.sqrt(np.std([1.0, 1.1, 0.9, 1.0], ddof=1) ** 2
                        + np.std([2.0, 2.2, 1.8, 2.0], ddof=1) ** 2))
    assert abs(fl - ref) < 1e-15, (fl, ref)
    fe, le, _ = two_sided_floor([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert abs(fe - np.sqrt(2.0) * le) < 1e-15
    print(f"  two-sided floor           {fl:.6f} (want {ref:.6f}); equal "
          f"wobble reduces to sqrt(2)x one side exactly")

    # 3. the verdict vocabulary -- and it MUST be able to say `no`
    v_in, r_in, _ = classify(1.0, 1.0, 1.0, 1.0, 0.0, True)
    v_out, r_out, _ = classify(5.0, 1.0, 1.0, 1.0, 0.0, True)
    assert v_in == "INDISTINGUISHABLE" and abs(r_in - 1.0) < 1e-15, v_in
    assert v_out.startswith("gap-at-5") and abs(r_out - 5.0) < 1e-15, v_out
    v_edge, _, _ = classify(2.0, 1.0, 1.0, 1.0, 0.0, True)
    assert v_edge == "INDISTINGUISHABLE", "the 2x bar must be inclusive"
    v_eps, _, _ = classify(2.0 + 1e-9, 1.0, 1.0, 1.0, 0.0, True)
    assert v_eps.startswith("gap-at-"), "the 2x bar must be a bar"
    print(f"  verdict rule              1x -> {v_in}; 5x -> {v_out}; the 2.0 "
          f"bar is inclusive and 2.0+1e-9 is not")

    # 4. UNMEASURABLE withholds a verdict in BOTH directions, and is not a pass
    v_q, r_q, f_q = classify(0.0, 1.0, 1.0, 1.0, 1.0, True)
    assert v_q == "UNMEASURABLE" and r_q is None and "q" in f_q, (v_q, f_q)
    v_ok, _, _ = classify(0.0, 1.0, 1.0, 1.0, 0.01, True)
    assert v_ok == "INDISTINGUISHABLE"
    print(f"  quantum gate              spread at 1x quantum -> {v_q} "
          f"(a zero gap does NOT buy a pass); at 100x -> {v_ok}")

    # 5. the u flag: asymmetric exactly as registered
    _, _, f_yes = classify(1.0, 1.0, 1.0, 1.0, 0.0, False)
    v_no, _, f_no = classify(9.0, 1.0, 1.0, 1.0, 0.0, False)
    assert "u" in f_yes and "u" in f_no and v_no.startswith("gap-at-9")
    flat = {(sd, d): 1.0 for sd in ("lego", "nemo") for d in (180, 270, 360)}
    sat, why = saturated(flat)
    assert sat and not why
    grow = dict(flat); grow[("lego", 360)] = 2.0
    unsat, why2 = saturated(grow)
    assert not unsat and "270->360" in why2, why2
    one_q = dict(flat); one_q[("lego", 270)] = 2.0; one_q[("lego", 360)] = 2.0
    ok3, _ = saturated(one_q)
    assert not ok3, "one quarter under the ratio is not saturation"
    # PER SIDE, not on the combined floor: one side growing must refuse even
    # when the other side dominates the RSS and makes it look flat.
    one_side = dict(flat)
    one_side[("nemo", 180)] = one_side[("nemo", 270)] = 1e-6
    one_side[("nemo", 360)] = 1e-2
    ok4, why4 = saturated(one_side)
    assert not ok4 and "nemo" in why4, why4
    print(f"  saturation                two quarters required and BOTH sides; "
          f"'{why2}' refused, one-quarter-only refused, and a growing NEMO "
          f"side refused even when it is 1e-4 of the RSS")

    # the u-flag materiality: a 15x gap whose floor grows 1.0x per quarter can
    # NOT be rescued; a 2.1x gap whose floor is doubling every quarter can.
    dead = {(sd, d): 1.0 for sd in ("lego", "nemo") for d in (270, 360)}
    m_dead, x_dead, g_dead = u_is_material(15.0, dead, 360)
    live = dict(dead); live[("lego", 360)] = 2.0
    m_live, x_live, g_live = u_is_material(2.1, live, 360)
    assert not m_dead and m_live, (m_dead, m_live)
    print(f"  u materiality             15x gap on a flat floor (g={g_dead:.1f}) "
          f"-> NOT material (needs {x_dead:.1f}x); 2.1x on a doubling floor "
          f"(g={g_live:.1f}) -> material (needs {x_live:.2f}x)")

    # 6. the one-sided flag
    _, _, f1 = classify(1.0, 1.0, 1.0, 1.0e-3, 0.0, True)
    _, _, f2 = classify(1.0, 1.0, 1.0, 1.0, 0.0, True)
    assert "1s" in f1 and "1s" not in f2
    print(f"  one-sided flag            1000x spread ratio -> {f1!r}; "
          f"equal -> {f2!r}")

    # 7. the band reductions on a KNOWN field: u = 1 m/s everywhere wet gives
    #    each band sum(e2u * e3t1d * wet) / 1e6, computed independently here.
    ones = {"u": np.ones((A.NY, A.NX, A.NZ), dtype=np.float64)}
    got = transport_row(ones, A.umask)
    e2 = _f64(A.e2u_col)
    e3 = _f64(A.e3t1d)
    per_lon = np.einsum("jik,k,j->ji", np.where(A.umask, 1.0, 0.0), e3, e2) / 1e6
    for name, rows in ALL_BANDS:
        want = D._avg(per_lon[rows].sum(axis=0))
        assert abs(got[name] - want) < 1e-9, (name, got[name], want)
    tot = sum(got[n] for n, _ in BANDS)
    assert abs(tot - got["FULL section"]) < 1e-9, (tot, got["FULL section"])
    print(f"  band transport on u=1     six bands sum to the full section "
          f"{got['FULL section']:.6f} Sv to {abs(tot - got['FULL section']):.1e}")
    # and the split is exact on the same field
    for name, rows in ALL_BANDS:
        bt, bc = band_bt_bc(ones["u"], A.umask, rows)
        assert abs(bt + bc - got[name]) < 1e-9
        # u is depth-uniform => the shear part is EXACTLY zero
        assert abs(bc) < 1e-12, (name, bc)
    print(f"  bt/bc split               a depth-uniform u gives bc == 0 "
          f"exactly in every band, and bt+bc == the band total")
    # a SHEARED field must put the signal in bc, not bt -- otherwise the check
    # above passes for a field that cannot distinguish the two legs
    shear = {"u": np.zeros((A.NY, A.NX, A.NZ), dtype=np.float64)}
    shear["u"][:, :, 0] = 1.0
    bt_s, bc_s = band_bt_bc(shear["u"], A.umask, BANDS[3][1])
    assert abs(bt_s) < 1e-12 and abs(bc_s) > 1e-3, (bt_s, bc_s)
    print(f"  bt/bc non-vacuity         a surface-only jet gives bt "
          f"{bt_s:.1e} and bc {bc_s:.4f} Sv -- the legs are distinguishable")

    # 8. the K9 known-answer control must ABORT on a planted wrong value
    fake = {"lego": {0: {360: {BANDS[0][0]: 0.0}}},
            "nemo": {0: {360: {BANDS[0][0]: -D360_P1_RECORDED_SV}}}}
    with contextlib.redirect_stdout(_QUIET):
        control_known_answer(fake)                       # passes at the truth
        try:
            control_known_answer(fake, recorded=-0.90000)
        except SystemExit:
            pass
        else:
            raise AssertionError("K9 accepted a planted wrong recorded value")
    print(f"  K9 non-vacuity            planting -0.90000 as the recorded "
          f"value ABORTS; the true {D360_P1_RECORDED_SV} passes")

    # 9. NaN is fatal, in both slots of the spread
    try:
        control_finite("self-test", [1.0, np.nan])
    except SystemExit:
        pass
    else:
        raise AssertionError("K4 accepted a NaN")
    mx, sd = spread_of([1.0, np.nan, 2.0, 3.0])
    assert np.isnan(mx) and np.isnan(sd), (mx, sd)
    print(f"  NaN                       fatal in K4, and PROPAGATES through "
          f"both slots of the spread rather than being discarded")

    print("SELF-TEST PASSED")


# ------------------------------------------------------------------------ run ---
def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sha_start, dirty_start = git_sha_now()
    print(f"producer revision at START of run: {sha_start}"
          f"{' +dirty' if dirty_start else ''}  (re-checked before the artifact "
          f"is written; a tree that moves mid-run is REFUSED)")
    print("=" * 78)
    print("#1455 REGIONAL AUDIT -- the equatorial and northern bands")
    print("=" * 78)
    prov, clock = control_stamps()
    geom = control_geometry()

    days = X.scored_days(0)
    for d in HORIZONS + CONTROL_DAYS:
        if d not in days:
            raise SystemExit(f"FATAL: registered horizon day {d} is not in the "
                             f"lego/NEMO intersection {days}")
    print(f"\nscored horizons (lego n NEMO): {days}")
    print(f"registered PRIMARY horizons  : {list(HORIZONS)}  "
          f"(controls only: {list(CONTROL_DAYS)}; the rest are POST-HOC "
          f"trajectory context)")

    lego0 = load_lego(0, 0)
    wet, w = X.build_weights(lego0["land_mask"])
    wet_u = A.umask
    control_dtype(w, lego0["T"], lego0["u"])
    print(f"\nmask: tmask & land_mask, {int(wet.sum())} wet T-cells; "
          f"u-mask A.umask, {int(wet_u.sum())} wet u-cells")
    print(f"weights: e1t*e2t*e3t_0 (partial cell), total "
          f"{float(w.sum()):.6e} m3 -- the REFERENCE geometry, identical on "
          f"both sides, so every share and ratio is valid and the total is not "
          f"the model's live volume")

    nemo0 = load_nemo(0, 0)
    quantum = band_quantum(nemo0, wet_u)
    tsq = ts_quantum(nemo0, wet, w)
    jq = jet_quantum(nemo0, wet_u)
    plant = control_plant(nemo0, wet_u, wet, w)

    # ---- the sweep -------------------------------------------------------
    rows = {"lego": {i: {} for i in range(N_MEM)},
            "nemo": {i: {} for i in range(N_MEM)}}
    ts = {}                     # day -> band/depth T,S rms + floors
    keep = {}                   # day -> difference fields (member 0)
    jets = {}                   # day -> zonal-mean u profiles
    print("\nloading 8 states per horizon (4 members x 2 sides) ...", flush=True)
    for day in days:
        st_l = [load_lego(i, day) for i in range(N_MEM)]
        st_n = [load_nemo(i, day) for i in range(N_MEM)]
        for i in range(N_MEM):
            rows["lego"][i][day] = transport_row(st_l[i], wet_u)
            rows["nemo"][i][day] = transport_row(st_n[i], wet_u)
            control_finite(f"lego m{i} day {day}", rows["lego"][i][day].values())
            control_finite(f"nemo m{i} day {day}", rows["nemo"][i][day].values())
        ts[day] = ts_rows(st_l, st_n, wet, w)
        control_finite(f"T/S table day {day}",
                       [v for tup in ts[day].values() for v in tup])
        jets[day] = jet_profiles(st_l, st_n, wet_u)
        if day in KEEP_DAYS:
            keep[day] = {"dT": np.where(wet, st_l[0]["T"] - st_n[0]["T"], 0.0),
                         "dS": np.where(wet, st_l[0]["S"] - st_n[0]["S"], 0.0)}
        print(f"  day {day:3d} done", flush=True)

    control_day0(lego0, nemo0, wet, wet_u, rows, quantum)
    worst_partition = control_full_section(rows, 360)
    print(f"[K7b PARTITION ON DATA] six bands sum to the full section at day "
          f"360 to {worst_partition:.1e} Sv over all 8 runs "
          f"(an identity in the row index -- kept to catch a mis-sliced band, "
          f"not because physics can break it)")
    p1_360 = control_known_answer(rows)

    art = report(rows, ts, keep, jets, days, quantum, tsq, jq, w, wet, out_dir)
    art.update({"provenance": prov, "clock": clock, "geometry": geom,
                "plant": plant, "day360_P1_gap_sv": p1_360})
    figures(rows, ts, keep, jets, days, w, wet, out_dir)
    stamp(art, out_dir, sha_start, dirty_start)
    return art


def ts_rows(st_l, st_n, wet, w):
    """Per band x depth class: the cross-model volume-weighted rms of dT and dS,
    and each side's own member-vs-control rms, RSS'd into the band's floor.

    The floor convention is the atlas's and the verdict run's: each side's own
    ensemble dispersion measured in the SAME band with the SAME reduction.  A
    global floor is never transferred to a band.
    """
    out = {}
    dT = np.where(wet, st_l[0]["T"] - st_n[0]["T"], 0.0)
    dS = np.where(wet, st_l[0]["S"] - st_n[0]["S"], 0.0)
    mem = {}
    for fld, key in (("T", "T"), ("S", "S")):
        mem[("lego", key)] = [np.where(wet, st_l[i][fld] - st_l[0][fld], 0.0)
                              for i in range(1, N_MEM)]
        mem[("nemo", key)] = [np.where(wet, st_n[i][fld] - st_n[0][fld], 0.0)
                              for i in range(1, N_MEM)]
    for bname, rows in ALL_BANDS:
        for dname, _ in X.DEPTH_CLASSES:
            sel = band_depth_sel(rows, dname, wet)
            if not sel.any():
                raise SystemExit(f"FATAL: ({bname}, {dname}) selects no wet "
                                 f"cell -- a defect in the partition")
            for key, d in (("T", dT), ("S", dS)):
                gap = X.wrms(d, w, sel)
                ls = float(np.sqrt(np.mean(
                    [X.wrms(m, w, sel) ** 2 for m in mem[("lego", key)]])))
                ns = float(np.sqrt(np.mean(
                    [X.wrms(m, w, sel) ** 2 for m in mem[("nemo", key)]])))
                out[(bname, dname, key)] = (gap, float(np.sqrt(ls ** 2 + ns ** 2)),
                                            ls, ns)
    return out


def jet_profiles(st_l, st_n, wet_u):
    """Zonal-mean u(z) on the equator row and on the +/-10 band, both models.

    The reduction is the WET-CELL zonal mean at each level -- sum(u)/count(wet)
    -- not a transport, because the question is the current STRUCTURE and a
    transport hides a sign-changing profile inside one number.  Land contributes
    nothing and an empty level is FATAL, not a NaN.
    """
    out = {}
    for tag, rows in (("equator row", slice(STRUCTURAL_ZERO_ROW,
                                            STRUCTURAL_ZERO_ROW + 1)),
                      ("E10 +/-10", NESTED[0][1]),
                      ("P4 +/-20", BANDS[3][1])):
        m = wet_u[rows]
        n = m.sum(axis=(0, 1)).astype(np.float64)
        # DINO's tropical bathymetry does not reach the deepest model level, so
        # the bottom level of every equatorial band is dry EVERYWHERE.  That is
        # real geometry, not a mis-specified band -- so the profile is
        # TRUNCATED at the deepest wet level and the truncation is reported,
        # rather than being averaged over nothing or filled with a NaN.  The
        # refusal is kept for the two cases that ARE defects: no wet level at
        # all, and wet levels that are not a contiguous prefix from the surface
        # (a hole in the middle of a column is a mask defect, and it is exactly
        # what an "all levels wet" assumption would hide).
        wetlev = np.where(n > 0)[0]
        if wetlev.size == 0:
            raise SystemExit(f"FATAL: {tag} has NO wet u-cell at any level -- "
                             f"a zonal mean over nothing is not a NaN to print")
        kmax = int(wetlev.max())
        if not np.array_equal(wetlev, np.arange(kmax + 1)):
            raise SystemExit(
                f"FATAL: {tag}'s wet levels {wetlev.tolist()} are not a "
                f"contiguous prefix from the surface -- a dry level above a wet "
                f"one is a mask defect, not bathymetry")
        for side, sts in (("lego", st_l), ("nemo", st_n)):
            profs = []
            for st in sts:
                u = _f64(st["u"])[rows]
                prof = (np.where(m, u, 0.0).sum(axis=(0, 1))[:kmax + 1]
                        / n[:kmax + 1])
                if not np.all(np.isfinite(prof)):
                    raise SystemExit(f"FATAL: non-finite in the {tag}/{side} "
                                     f"zonal-mean profile")
                profs.append(prof)
            out[(tag, side)] = profs[0]                # the control member
            # the band's OWN floor for THIS statistic, measured the same way
            # every other floor here is: the sample std over that side's four
            # members, level by level.  Without it the jet comparison is a
            # difference with no bar, which is the one thing this campaign
            # refuses to publish.
            # ddof=1 needs n>1; jet_quantum calls this with a single state and
            # wants only the profile, never the spread.
            out[(tag, side, "spread")] = (
                np.std(np.asarray(profs), axis=0, ddof=1) if len(profs) > 1
                else np.zeros_like(profs[0]))
        out[(tag, "floor")] = np.sqrt(out[(tag, "lego", "spread")] ** 2
                                      + out[(tag, "nemo", "spread")] ** 2)
        out[(tag, "n_wet")] = n[:kmax + 1]
        out[(tag, "kmax")] = kmax
    return out


# --------------------------------------------------------------------- report ---
def _tbl(title, header, lines):
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for ln in lines:
        print(ln)


def report(rows, ts, keep, jets, days, quantum, tsq, jq, w, wet, out_dir):
    art = {}
    print("\n" + "=" * 78)
    print("1. THE REGIONAL TRANSPORT LEDGER")
    print("=" * 78)
    print("gap = legoESM - NEMO, control member, mean reduction over longitudes")
    print("floor = sqrt(spread_lego^2 + spread_NEMO^2), each a sample std over")
    print("        that side's OWN 4 members IN THIS BAND at THIS day (n=4 per")
    print("        side -- ~41% relative standard error, a factor-of-two")
    print("        estimate).  No global floor is transferred to any band.")

    ledger = {}
    for day in HORIZONS:
        lines = []
        for bname, _ in ALL_BANDS:
            gap = rows["lego"][0][day][bname] - rows["nemo"][0][day][bname]
            fl, ls, ns = two_sided_floor(
                [rows["lego"][i][day][bname] for i in range(N_MEM)],
                [rows["nemo"][i][day][bname] for i in range(N_MEM)])
            sp = {(sd, d): spread_of([rows[sd][i][d][bname]
                                      for i in range(N_MEM)])[1]
                  for sd in ("lego", "nemo") for d in (180, 270, 360)}
            sat, why = saturated(sp)
            verd, ratio, flag = classify(gap, fl, ls, ns, quantum[bname], sat)
            # the `u` flag is only ATTACHED where it could overturn the row
            mat, x2yes, glast = (True, float("nan"), float("nan"))
            # Only a `no` can be overturned by a growing floor.  Applying the
            # materiality test to an INDISTINGUISHABLE row prints a YES that
            # means nothing, which is how the headline row (P4 at day 360) came
            # to carry a decorative flag.  Caught in code review of 17881d91d.
            if not sat and ratio is not None and verd.startswith("gap-at-"):
                mat, x2yes, glast = u_is_material(ratio, sp, day)
                if not mat and "u" in flag.split(","):
                    flag = ",".join([f for f in flag.split(",") if f != "u"]
                                    + ["u!"])
            lego_share = (ls ** 2 / fl ** 2) if fl > 0 else float("nan")
            gbt = (rows["lego"][0][day][bname + " |bt"]
                   - rows["nemo"][0][day][bname + " |bt"])
            gbc = (rows["lego"][0][day][bname + " |bc"]
                   - rows["nemo"][0][day][bname + " |bc"])
            # The two legs get their OWN floors.  Code review of 17881d91d found
            # R3 ("the equatorial gap is predominantly baroclinic") scored
            # CONFIRMED on an 0.0059 Sv margin -- HALF the floor of either leg --
            # with no floor computed anywhere, while two of the four ensemble
            # members flip the comparison.  A decomposition whose two legs are
            # ~10x the band total and nearly cancel cannot be published without
            # a bar on the difference.
            fbt = two_sided_floor(
                [rows["lego"][i][day][bname + " |bt"] for i in range(N_MEM)],
                [rows["nemo"][i][day][bname + " |bt"] for i in range(N_MEM)])[0]
            fbc = two_sided_floor(
                [rows["lego"][i][day][bname + " |bc"] for i in range(N_MEM)],
                [rows["nemo"][i][day][bname + " |bc"] for i in range(N_MEM)])[0]
            f_split = float(np.sqrt(fbt ** 2 + fbc ** 2))
            split_margin = abs(gbc) - abs(gbt)
            ledger[(bname, day)] = {
                "gap": gap, "floor": fl, "lego_spread": ls, "nemo_spread": ns,
                "ratio": ratio, "verdict": verd, "flag": flag,
                "gap_bt": gbt, "gap_bc": gbc, "saturated": sat,
                "floor_bt": fbt, "floor_bc": fbc, "floor_split": f_split,
                "split_margin": split_margin,
                "sat_why": why, "quantum": quantum[bname],
                "u_material": bool(mat), "x2yes": x2yes, "g_lastQ": glast,
                "lego_share_of_floor": lego_share,
                "lego": rows["lego"][0][day][bname],
                "nemo": rows["nemo"][0][day][bname]}
            # K_WELCH, registered in PREREG sec.6 as its own column and
            # missing from the first version of this table.  The floor is a
            # variance ESTIMATE with 3 dof per side, so the two-sided ~95%
            # constant is a Welch t at nu~6 of about 2.45, not 2.0.  Ratios in
            # [2.0, 2.45) are printed `no` by the REGISTERED rule while NOT
            # being statistically resolved; this column marks that band and
            # never overturns the verdict.
            welch = ("" if ratio is None else
                     "unres" if V.K_PREREG < ratio < V.K_WELCH else "")
            lines.append(
                f"{bname:<26}{gap:>+10.4f}{fl:>11.4f}"
                f"{('  --  ' if ratio is None else f'{ratio:>8.2f}')}"
                f"{welch:>7}"
                f"  {verd:<22}{gbt:>+9.4f}{fbt:>8.4f}{gbc:>+9.4f}{fbc:>8.4f}"
                f"  {flag}")
        _tbl(f"--- day {day} ---",
             f"{'band':<26}{'gap[Sv]':>10}{'floor[Sv]':>11}{'x':>8}"
             f"{'welch':>7}"
             f"  {'verdict':<22}{'gap bt':>9}{'fl bt':>8}{'gap bc':>9}"
             f"{'fl bc':>8}  flags",
             lines)
    art["ledger"] = {f"{b}|{d}": v for (b, d), v in ledger.items()}

    # the materiality table -- the recorded discipline, per band
    lines = []
    for bname, _ in ALL_BANDS:
        cells = []
        for day in HORIZONS:
            e = ledger[(bname, day)]
            r = e["ratio"]
            is_no = e["verdict"].startswith("gap-at-")
            cells.append(f"{(r if r is not None else float('nan')):>8.2f}"
                         f"{e['x2yes']:>8.2f}{e['g_lastQ']:>9.2f}"
                         f"{('YES' if (is_no and not e['saturated']
                                       and e['u_material']) else
                            ('n/a' if not is_no else 'no')):>5}")
        lines.append(f"{bname:<26}" + "".join(cells)
                     + f"{ledger[(bname, 360)]['lego_share_of_floor']:>9.1%}")
    _tbl("--- DOES `u` MATTER?  the growth an unsaturated floor would need "
         "against the growth it shows ---",
         f"{'band':<26}" + "".join(f"{'d' + str(d) + ' x':>8}{'x2YES':>8}"
                                   f"{'g_lastQ':>9}{'mat':>5}" for d in HORIZONS)
         + f"{'lego share':>9}",
         lines)
    print(f"  x2YES = ratio / {V.K_PREREG} -- the growth this band's floor would")
    print(f"          need for its `no` to become a YES.")
    print(f"  g_lastQ = spread(day)/spread(270), the growth it actually shows,")
    print(f"          taken as the LARGER of the two sides.")
    print(f"  `u` is ATTACHED to a verdict only when another "
          f"{V.UNSAT_CREDIT_QUARTERS} quarters at g_lastQ")
    print(f"          could close x2YES; otherwise the row prints `u!` -- the")
    print(f"          ensemble is unsaturated as a FACT, but it cannot rescue")
    print(f"          the verdict, and voiding every `no` for free is not honest.")
    print(f"  lego share = lego_std^2 / floor^2: how much of the two-sided floor")
    print(f"          is legoESM's OWN dispersion rather than a two-model bar.")

    print("\nflags:  u = the band's ensemble is UNSATURATED (its spread is still")
    print("            growing over the last two quarters).  Asymmetric, as")
    print("            registered: an INDISTINGUISHABLE still stands, a")
    print("            gap-at-Nx does NOT -- N is an UPPER BOUND on the true")
    print("            multiple because the denominator has not finished.")
    print("        1s= the two sides' spreads differ by over one decade, so the")
    print("            RSS floor is numerically ONE model's own dispersion.")
    print("        q = the legoESM spread is at or under 10x this band's fp32")
    print("            storage quantum -> UNMEASURABLE, which is NOT a pass.")
    print(f"     welch= the ratio is in [{V.K_PREREG}, {V.K_WELCH}), where the")
    print(f"            REGISTERED rule says `no` but an ESTIMATED floor with 3")
    print(f"            degrees of freedom per side does not resolve it. Marked,")
    print(f"            never used to overturn -- the registered rule decides.")
    print("        u!= unsaturated as a FACT, but the growth it shows could not")
    print("            close the gap in another year -- the flag does NOT void")
    print("            this row's verdict.")

    # Is each band's barotropic-vs-baroclinic ORDERING resolved at all?
    # The two legs are routinely ~10x the band total and nearly cancel, so
    # "predominantly barotropic" is a claim about a MARGIN and needs the
    # margin's own bar.  Without this table the ordering is a coin flip
    # published as a finding -- which is exactly what happened to R3.
    lines = []
    for bname, _ in ALL_BANDS:
        e = ledger[(bname, 360)]
        m = abs(e["gap_bc"]) - abs(e["gap_bt"])
        bar = V.K_PREREG * e["floor_split"]
        if abs(m) <= bar:
            verd = "UNRESOLVED"
        else:
            verd = "baroclinic-led" if m > 0 else "barotropic-led"
        lines.append(f"{bname:<26}{abs(e['gap_bt']):>10.4f}"
                     f"{abs(e['gap_bc']):>10.4f}{m:>+10.4f}{bar:>10.4f}"
                     f"  {verd}")
    _tbl("--- day 360: is the barotropic/baroclinic ORDERING resolved? ---",
         f"{'band':<26}{'|gap bt|':>10}{'|gap bc|':>10}{'margin':>10}"
         f"{'2x floor':>10}  verdict",
         lines)
    print("  margin = |gap bc| - |gap bt|; the bar is the registered "
          f"{V.K_PREREG:.0f}x")
    print("  applied to the margin's OWN two-sided floor (the RSS of the two")
    print("  legs' floors).  UNRESOLVED means the ordering is inside the bar --")
    print("  neither leg is shown to lead, and no `predominantly' claim is")
    print("  supportable for that band.")
    art["split_ordering"] = {
        b: {"gap_bt": ledger[(b, 360)]["gap_bt"],
            "gap_bc": ledger[(b, 360)]["gap_bc"],
            "floor_bt": ledger[(b, 360)]["floor_bt"],
            "floor_bc": ledger[(b, 360)]["floor_bc"],
            "margin": abs(ledger[(b, 360)]["gap_bc"]) - abs(ledger[(b, 360)]["gap_bt"]),
            "bar": V.K_PREREG * ledger[(b, 360)]["floor_split"]}
        for b, _ in ALL_BANDS}

    # the trajectory, POST-HOC
    lines = []
    for bname, _ in ALL_BANDS:
        tr = [rows["lego"][0][d][bname] - rows["nemo"][0][d][bname]
              for d in days]
        lines.append(f"{bname:<26}" + "".join(f"{v:>+8.3f}" for v in tr[:12]))
    _tbl("--- POST-HOC trajectory of the gap [Sv] (not a registered horizon; "
         "cannot promote or demote a verdict) ---",
         f"{'band':<26}" + "".join(f"{d:>8d}" for d in days[:12]), lines)
    art["trajectory_days"] = days
    art["trajectory"] = {b: [rows["lego"][0][d][b] - rows["nemo"][0][d][b]
                             for d in days] for b, _ in ALL_BANDS}

    # --- 2. the T/S divergence, re-reduced -------------------------------
    print("\n" + "=" * 78)
    print("2. REGIONAL T/S DIVERGENCE -- the atlas re-reduced into these bands")
    print("=" * 78)
    print("volume-weighted rms of (legoESM - NEMO), thickness-weighted by e3t_0")
    print("-- NEVER layer-averaged (a layer mean over-weights the 10 m surface")
    print("layer ~54x against the 545 m bottom and has already inverted the sign")
    print("of three published rows in this campaign).")
    for day in HORIZONS:
        for key, unit in (("T", "K"), ("S", "g/kg")):
            lines = []
            for bname, _ in ALL_BANDS:
                cells = []
                for dname, _ in X.DEPTH_CLASSES:
                    gap, fl, ls, ns = ts[day][(bname, dname, key)]
                    q = tsq[(bname, dname, key)]
                    if ls <= V.QUANTUM_MARGIN * q:
                        cells.append(f"{gap:>10.3e}/{'UNMEAS':>7}")
                    else:
                        r = gap / fl if fl > 0 else float("inf")
                        cells.append(f"{gap:>10.3e}/{r:>7.1f}")
                lines.append(f"{bname:<26}" + "".join(cells))
            _tbl(f"--- day {day}: d{key} [{unit}]  (value / multiples of that "
                 f"band-and-class's OWN measured floor; UNMEAS = that band's "
                 f"legoESM spread is at or under {V.QUANTUM_MARGIN:.0f}x its "
                 f"own fp32 storage quantum, so the multiple would be measuring "
                 f"the npz dtype -- withheld, NOT a pass) ---",
                 f"{'band':<26}" + "".join(f"{n:>18}" for n, _ in X.DEPTH_CLASSES),
                 lines)
    art["ts_all"] = {f"{day}|{b}|{d}|{k}": list(v)
                     for day in HORIZONS for (b, d, k), v in ts[day].items()}

    # pattern correlation across horizons, per band
    print("\n--- does each band HOLD its own divergence pattern? ---")
    print("volume-weighted correlation of the band's difference field between")
    print("horizons, plus the atlas's forward-retention test of the day-10")
    print("hotspot set (5% of the band's OWN volume, ranked by intensity;")
    print(f"geometric null {X.Q2_NULL}, registered bar {X.Q2_RET_BAR}).")
    print("A band whose pattern is INHERITED shows retention collapsing to the")
    print("null while the amplitude stays up; a band that HOLDS its own shows")
    print("retention staying above the bar.")
    ret = {}
    for key in ("T", "S"):
        lines = []
        for bname, brows in ALL_BANDS:
            wb = np.where(band_cell_mask(brows) & wet, w, 0.0)
            d10 = np.where(band_cell_mask(brows), keep[10]["d" + key], 0.0)
            hot = X.hotspot_set(d10, wb, X.Q2_NULL)
            cells = []
            for day in (90, 180, 270, 360):
                dd = np.where(band_cell_mask(brows), keep[day]["d" + key], 0.0)
                r = X.retention(dd, wb, hot)
                c = X.wcorr(d10, dd, wb)
                ret[(bname, key, day)] = (r, c)
                cells.append(f"{r:>8.3f}{c:>+8.3f}")
            lines.append(f"{bname:<26}" + "".join(cells))
        _tbl(f"--- d{key}: forward retention of the day-10 hotspot set, and the "
             f"pattern correlation against day 10 ---",
             f"{'band':<26}" + "".join(f"{'ret d' + str(d):>8}{'corr':>8}"
                                       for d in (90, 180, 270, 360)),
             lines)
    art["retention"] = {f"{b}|{k}|{d}": list(v) for (b, k, d), v in ret.items()}

    # --- 3. the equatorial jets ------------------------------------------
    print("\n" + "=" * 78)
    print("3. EQUATORIAL SPECIFICS -- the tropical current structure")
    print("=" * 78)
    print("Zonal-mean u [m/s] over the WET u-cells at each level.  This is a")
    print("STRUCTURE statistic, not a transport: a transport hides a")
    print("sign-changing profile inside one number, and the equatorial current")
    print("system is exactly a sign-changing profile.")
    print(f"NO f-weighted or vorticity-projected statistic appears here or")
    print(f"anywhere in this probe -- f is EXACTLY 0.0 at row "
          f"{STRUCTURAL_ZERO_ROW} and a normalised map with it in the")
    print("denominator has already produced one withdrawn result in this campaign.")
    for tag in ("equator row", "E10 +/-10"):
        kmax = min(jets[d][(tag, "kmax")] for d in HORIZONS)
        print(f"\n[{tag}] profile TRUNCATED at level {kmax} "
              f"({float(A.gdept1d[kmax]):.0f} m): levels below it hold no wet "
              f"u-cell anywhere in the band -- DINO's tropical bathymetry, not "
              f"a mis-specified band.  The truncation is printed rather than "
              f"averaged over nothing.")
        # EVERY level down to 10, then every third.  A uniform stride of 3 hid
        # the structure on the first pass: the equatorial difference is a
        # DIPOLE peaking at level 2, which a k=0,3,6 sample steps straight over.
        lines = []
        klist = list(range(0, min(11, kmax + 1))) + list(range(12, kmax + 1, 3))
        for k in klist:
            cells = []
            for day in HORIZONS:
                a = jets[day][(tag, "lego")][k]
                b = jets[day][(tag, "nemo")][k]
                fl = jets[day][(tag, "floor")][k]
                ls = jets[day][(tag, "lego", "spread")][k]
                if ls <= V.QUANTUM_MARGIN * jq[tag][k]:
                    cells.append(f"{a:>+8.4f}{b:>+8.4f}{a - b:>+8.4f}"
                                 f"{'UNMEAS':>8}")
                else:
                    r = abs(a - b) / fl if fl > 0 else float("inf")
                    cells.append(f"{a:>+8.4f}{b:>+8.4f}{a - b:>+8.4f}{r:>8.1f}")
            lines.append(f"{k:>3d}{float(A.gdept1d[k]):>8.0f}"
                         f"{int(jets[HORIZONS[0]][(tag, 'n_wet')][k]):>6d}"
                         + "".join(cells))
        print("  READ THE `xfloor` COLUMNS RIGHT: at day 90 the 1e-14 kick has")
        print("  barely grown, so the ensemble denominator is 3-5 orders under")
        print("  its day-360 value and every multiple in that column is enormous")
        print("  for that reason alone.  The day-360 column is the one whose")
        print("  floor was measured on a grown ensemble.  Same rule as the")
        print("  transport table: the bar belongs to the statistic AND to the")
        print("  horizon.")
        _tbl(f"--- {tag}: zonal-mean u(z) ---",
             f"{'k':>3}{'depth':>8}{'nwet':>6}"
             + "".join(f"{'lego d' + str(d):>8}{'nemo':>8}{'diff':>8}{'xfloor':>8}"
                       for d in HORIZONS),
             lines)
    art["jets"] = {f"{t}|{sd}|{d}": jets[d][(t, sd)].tolist()
                   for d in HORIZONS for t in ("equator row", "E10 +/-10",
                                               "P4 +/-20")
                   for sd in ("lego", "nemo")}

    # surface jet summary -- the one number the campaign has quoted before
    lines = []
    for tag in ("equator row", "E10 +/-10", "P4 +/-20"):
        cells = []
        for day in HORIZONS:
            a = float(jets[day][(tag, "lego")][0])
            b = float(jets[day][(tag, "nemo")][0])
            fl = float(jets[day][(tag, "floor")][0])
            ls = float(jets[day][(tag, "lego", "spread")][0])
            rtxt = ("UNMEAS" if ls <= V.QUANTUM_MARGIN * jq[tag][0]
                    else f"{abs(a - b) / fl:.0f}" if fl > 0 else "inf")
            cells.append(f"{a:>+9.4f}{b:>+9.4f}{a - b:>+9.4f}"
                         f"{100.0 * (a - b) / b:>+8.1f}%{rtxt:>10}")
        lines.append(f"{tag:<16}" + "".join(cells))
    _tbl("--- surface (k=0) zonal-mean u [m/s], with each band's OWN measured "
         "floor for THIS statistic ---",
         f"{'band':<16}" + "".join(f"{'lego d' + str(d):>9}{'nemo':>9}"
                                   f"{'diff':>9}{'rel':>9}{'xfloor':>10}"
                                   for d in HORIZONS),
         lines)
    # The dipole, and the cancellation that reconciles section 3 with section 1.
    print("\n--- POST-HOC: the SHAPE of the equatorial difference, and why the")
    print("    transport statistic cannot see it ---")
    print("The difference is not a uniform weakening: it changes sign in the")
    print("upper ocean.  Reported here are its two extrema, the depth where it")
    print("crosses zero, and its THICKNESS-WEIGHTED vertical integral over the")
    print("top 200 m -- which is the quantity a depth-integrated transport")
    print("actually sees.  POST-HOC: registered nowhere.")
    e3 = _f64(A.e3t1d)
    for tag in ("equator row", "E10 +/-10"):
        for day in HORIZONS:
            a = jets[day][(tag, "lego")]
            b = jets[day][(tag, "nemo")]
            d = a - b
            top = _f64(A.gdept1d)[:len(d)] < 200.0
            kp, kn = int(np.argmax(d)), int(np.argmin(d))
            # the shallowest sign change
            sgn = np.sign(d)
            cross = np.where(sgn[:-1] * sgn[1:] < 0)[0]
            zc = (f"{float(A.gdept1d[int(cross[0])]):.0f}-"
                  f"{float(A.gdept1d[int(cross[0]) + 1]):.0f} m"
                  if cross.size else "none")
            pos = float(np.sum(np.where(d > 0, d, 0.0)[top] * e3[:len(d)][top]))
            neg = float(np.sum(np.where(d < 0, d, 0.0)[top] * e3[:len(d)][top]))
            net = pos + neg
            canc = abs(net) / max(abs(pos), abs(neg)) if max(abs(pos), abs(neg)) > 0 else float("nan")
            print(f"  {tag:<14} day {day:3d}: peak {d[kp]:+.4f} m/s at "
                  f"{float(A.gdept1d[kp]):.0f} m, trough {d[kn]:+.4f} m/s at "
                  f"{float(A.gdept1d[kn]):.0f} m, sign change at {zc}")
            print(f"  {'':<14}          top-200 m integral: "
                  f"+{pos:.5f} / {neg:.5f} = NET {net:+.5f} m2/s "
                  f"-- the two lobes cancel to {canc:.1%} of the larger")
    print("  READ THE CANCELLATION PER BAND, not as one sentence: on the")
    print("  +/-10 band the residual is 1-17% of the larger lobe, i.e. the")
    print("  vertical integral annihilates most of the signal.  On the SINGLE")
    print("  equator row it is 45-57%, which is not 'largely cancels' and is")
    print("  not claimed to be.  The first version of this line covered both")
    print("  with one phrase and is withdrawn.")
    print("  AND THE LINK IS PLAUSIBLE, NOT MEASURED.  This integral is over a")
    print("  WET-CELL-COUNT zonal mean times the reference thicknesses; the")
    print("  band transport is a width- and thickness-weighted SUM.  The two")
    print("  functionals differ by the per-level wet count and by the e2u row")
    print("  weighting, so the cancellation is CONSISTENT WITH the transport")
    print("  reading INDISTINGUISHABLE and is not a demonstration of it.")

    # where does the profile disagreement LIVE?  Registered nowhere, so this is
    # POST-HOC and labelled: the deepest level at which the two models still
    # differ by more than 1% of NEMO's own value there.
    print("\n--- POST-HOC: how deep does the equatorial disagreement reach? ---")
    print("The equatorial profile CROSSES ZERO, so |lego-NEMO| divided by NEMO's")
    print("own value at that level is a ratio on a sign-changing field: it blows")
    print("up wherever NEMO passes through zero and reports depths that are")
    print("artifacts of the crossing, not of the disagreement.  (A first version")
    print("of this line did exactly that and read 234% at the equator; it is")
    print("withdrawn.)  The scale below is therefore the profile's OWN SURFACE")
    print("MAGNITUDE, one fixed nonzero number per profile, and the same one on")
    print("both sides.  POST-HOC: not registered, and it cannot promote or")
    print("demote any verdict above.")
    for tag in ("equator row", "E10 +/-10", "P4 +/-20"):
        for day in HORIZONS:
            a = jets[day][(tag, "lego")]
            b = jets[day][(tag, "nemo")]
            scale = abs(float(b[0]))
            if not scale > 0.0:
                raise SystemExit(f"FATAL: {tag} day {day} has a zero surface "
                                 f"magnitude -- the normalising scale is not "
                                 f"a number to divide by")
            rel = np.abs(a - b) / scale
            bad = np.where(rel >= 0.01)[0]
            k0 = int(bad.max()) + 1 if bad.size else 0
            zz = float(A.gdept1d[min(k0, A.NZ - 1)])
            # THICKNESS-WEIGHTED.  The first version summed |diff| over 36
            # levels ranging 10 m to 545 m thick with no e3, in the one probe
            # that lectures about layer-vs-thickness weighting -- and PREREG
            # sec.4 states in as many words that no unweighted mean over cells
            # of unequal volume appears here.  Caught in code review of
            # 17881d91d; it overstated the surface share by up to 18 points.
            ee = e3[:len(d)]
            share = float((np.abs(a - b)[:2] * ee[:2]).sum()
                          / (np.abs(a - b) * ee).sum())
            print(f"  {tag:<14} day {day:3d}: |diff| is under 1% of the "
                  f"surface magnitude ({scale:.4f} m/s) from level {k0:2d} "
                  f"({zz:.0f} m) down; the top TWO levels hold "
                  f"{share:.1%} of the summed |diff| over the column")

    # --- 4. the verdict table ---------------------------------------------
    print("\n" + "=" * 78)
    print("4. THE VERDICT TABLE -- band x horizon")
    print("=" * 78)
    hdr = f"{'band':<26}" + "".join(f"{'day ' + str(d):>26}" for d in HORIZONS)
    lines = []
    for bname, _ in ALL_BANDS:
        cells = []
        for day in HORIZONS:
            e = ledger[(bname, day)]
            r = "  --  " if e["ratio"] is None else f"{e['ratio']:.2f}x"
            f = f" [{e['flag']}]" if e["flag"] else ""
            cells.append(f"{e['gap']:>+8.3f} {r:>7}{f:<10}")
        lines.append(f"{bname:<26}" + "".join(cells))
    _tbl("gap [Sv] and multiples of that band's OWN two-sided floor", hdr, lines)

    print("\nverdicts, spelled out:")
    for day in HORIZONS:
        print(f"  day {day}:")
        for bname, _ in ALL_BANDS:
            e = ledger[(bname, day)]
            print(f"    {bname:<26} {e['verdict']:<24}"
                  f"gap {e['gap']:+.4f} Sv, floor {e['floor']:.4f} Sv"
                  + (f"  [{e['flag']}]" if e["flag"] else ""))

    # escalation
    esc = sorted({b for (b, d), e in ledger.items()
                  if e["ratio"] is not None and e["ratio"] > ESCALATION_FLOORS})
    print(f"\nESCALATION (PREREG sec.10, registered before any number): a band "
          f"over {ESCALATION_FLOORS:.0f}x its own floor is a NEW REGIONAL "
          f"FINDING and needs dual sign-off.")
    if esc:
        for b in esc:
            worst = max(ledger[(b, d)]["ratio"] for d in HORIZONS
                        if ledger[(b, d)]["ratio"] is not None)
            print(f"  ESCALATED: {b}  worst {worst:.1f}x its own floor")
    else:
        print("  none -- no band clears the escalation bar")
    art["escalated"] = esc
    # APPLIED AS REGISTERED the rule escalates EVERY band, because the day-90
    # floors are three to five orders under the day-360 ones (the kick has not
    # grown) and any real gap divided by them is enormous.  That is the rule
    # doing what it was written to do on a horizon whose denominator is not yet
    # a physical floor -- it is not a finding about seven bands.  The rule is
    # reported as registered above and NOT quietly narrowed; the day-360 subset
    # below is the informative reading, labelled as a refinement rather than
    # substituted for the registered answer.  The practical consequence is the
    # same either way: this audit goes to dual review.
    esc360 = [b for b, _ in ALL_BANDS
              if ledger[(b, 360)]["ratio"] is not None
              and ledger[(b, 360)]["ratio"] > ESCALATION_FLOORS]
    print(f"\n  REFINEMENT (labelled, not a substitution): at the day-360")
    print(f"  endpoint alone -- the horizon whose floors were measured on a")
    print(f"  grown ensemble -- the bands over {ESCALATION_FLOORS:.0f}x are:")
    if esc360:
        for b in esc360:
            e = ledger[(b, 360)]
            print(f"    {b}  {e['ratio']:.2f}x  (gap {e['gap']:+.4f} Sv, "
                  f"floor {e['floor']:.4f} Sv)")
    else:
        print("    none")
    art["escalated_day360"] = esc360

    # predictions
    print("\n--- REGISTERED PREDICTIONS (PREREG sec.9), scored ---")
    d360 = {b: ledger[(b, 360)] for b, _ in ALL_BANDS}
    p4, p1, p5, p6 = (BANDS[3][0], BANDS[0][0], BANDS[4][0], BANDS[5][0])
    preds = []
    r4 = d360[p4]["ratio"]
    preds.append(("R1 equatorial gap over its own floor at day 360",
                  r4 is not None and r4 > V.K_PREREG,
                  f"P4 {d360[p4]['gap']:+.4f} Sv at "
                  + ("UNMEASURABLE" if r4 is None else f"{r4:.2f}x")))
    biggest = max(((abs(d360[b]["gap"]), b) for b, _ in BANDS))[1]
    preds.append(("R2 P1 (south) is still the largest absolute band gap",
                  biggest == p1, f"largest is {biggest} at "
                  f"{abs(d360[biggest]['gap']):.4f} Sv"))
    # R3 is scored against a BAR, not on the sign of a margin.  Registered as
    # "|bc| > |bt|"; the honest scoring of that against this campaign's own rule
    # is |margin| > K_PREREG x the margin's own floor, and anything inside the
    # bar is UNRESOLVED -- neither confirmed nor refuted.
    m3, f3 = d360[p4]["split_margin"], d360[p4]["floor_split"]
    r3_resolved = abs(m3) > V.K_PREREG * f3
    preds.append(("R3 the equatorial gap is predominantly BAROCLINIC",
                  r3_resolved and m3 > 0.0,
                  f"P4 bt {d360[p4]['gap_bt']:+.4f} (floor "
                  f"{d360[p4]['floor_bt']:.4f}) vs bc "
                  f"{d360[p4]['gap_bc']:+.4f} (floor "
                  f"{d360[p4]['floor_bc']:.4f}) Sv; margin |bc|-|bt| = "
                  f"{m3:+.4f} Sv against {V.K_PREREG:.0f}x its own floor "
                  f"{V.K_PREREG * f3:.4f} -- "
                  + ("RESOLVED" if r3_resolved else
                     "UNRESOLVED: the margin is inside the bar, so this "
                     "prediction is neither confirmed nor refuted")))
    rt = ret[(p4, "T", 360)][0]
    preds.append(("R4 the equatorial T divergence holds its own hotspots",
                  rt > X.Q2_RET_BAR,
                  f"P4 forward retention {rt:.3f} against bar "
                  f"{X.Q2_RET_BAR} and null {X.Q2_NULL}"))
    preds.append(("R5 P6 (subpolar) gap is smaller than P5 (subtropics)",
                  abs(d360[p6]["gap"]) < abs(d360[p5]["gap"]),
                  f"P6 {abs(d360[p6]['gap']):.4f} vs P5 "
                  f"{abs(d360[p5]['gap']):.4f} Sv"))
    for name, ok, why in preds:
        # THREE states, not two.  A prediction whose margin sits inside its own
        # bar is UNRESOLVED -- printing it as FAILED claims a refutation the
        # measurement does not support, which is the mirror of the CONFIRMED it
        # replaced.  R3 is the case that forced this.
        tag = ("UNRESOLVED" if "UNRESOLVED" in why
               else "CONFIRMED" if ok else "REFUTED")
        print(f"  [{tag:^10}] {name}\n                 {why}")
    art["predictions"] = [
        {"prediction": n,
         "status": ("UNRESOLVED" if "UNRESOLVED" in e
                    else "CONFIRMED" if o else "REFUTED"),
         "evidence": e} for n, o, e in preds]
    return art


# -------------------------------------------------------------------- figures ---
def figures(rows, ts, keep, jets, days, w, wet, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F1 -- the gap trajectory per band, with each band's own floor envelope
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for bname, _ in ALL_BANDS:
        g = [rows["lego"][0][d][bname] - rows["nemo"][0][d][bname] for d in days]
        ax.plot(days, g, marker="o", ms=3, label=bname)
    ax.axhline(0.0, color="k", lw=0.6)
    for d in HORIZONS:
        ax.axvline(d, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("day from the shared NEMO day-180 restart")
    ax.set_ylabel("legoESM - NEMO zonal transport [Sv]")
    ax.set_title("Regional transport gap by band\n"
                 "(vertical lines = the four registered horizons; all other "
                 "days are POST-HOC context)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/F1_band_gap_trajectory.png", dpi=140)
    plt.close(fig)

    # F2 -- band x horizon gap in units of that band's OWN floor
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [b for b, _ in ALL_BANDS]
    M = np.full((len(names), len(HORIZONS)), np.nan)
    for a, bname in enumerate(names):
        for b, day in enumerate(HORIZONS):
            fl = two_sided_floor(
                [rows["lego"][i][day][bname] for i in range(N_MEM)],
                [rows["nemo"][i][day][bname] for i in range(N_MEM)])[0]
            g = rows["lego"][0][day][bname] - rows["nemo"][0][day][bname]
            M[a, b] = abs(g) / fl if fl > 0 else np.nan
    im = ax.imshow(np.log10(M), aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"day {d}" for d in HORIZONS])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for a in range(len(names)):
        for b in range(len(HORIZONS)):
            ax.text(b, a, f"{M[a, b]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="log10( |gap| / that band's OWN floor )")
    ax.set_title("Gap in multiples of each band's own measured floor\n"
                 "(2.0 is the registered INDISTINGUISHABLE bar)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/F2_band_floor_multiples.png", dpi=140)
    plt.close(fig)

    # F3 -- the equatorial jet profiles.
    # THREE rows, because one is misleading: the full column is 4506 m and the
    # disagreement lives in the top ~50 m, so a single full-depth panel squashes
    # the entire finding into a sliver a reader cannot see.  Row 1 is the full
    # column (so nobody can accuse the zoom of hiding deep structure), row 2 is
    # the top 200 m where the difference actually is, and row 3 is the
    # difference itself against the band's OWN measured ensemble floor.
    z = _f64(A.gdept1d)
    fig, axes = plt.subplots(3, len(HORIZONS), figsize=(14, 11))
    for col, day in enumerate(HORIZONS):
        for row, zmax in enumerate((None, 200.0)):
            axx = axes[row, col]
            for tag, ls in (("equator row", "-"), ("E10 +/-10", "--")):
                nk = len(jets[day][(tag, "lego")])
                axx.plot(jets[day][(tag, "lego")], z[:nk], ls, color="C0", lw=1.4,
                         label=f"legoESM, {tag}")
                axx.plot(jets[day][(tag, "nemo")], z[:nk], ls, color="C3", lw=1.4,
                         label=f"NEMO, {tag}")
            axx.axvline(0.0, color="k", lw=0.6)
            axx.invert_yaxis()
            if zmax is not None:
                axx.set_ylim(zmax, 0.0)
            axx.set_title(f"day {day}" + ("" if zmax is None
                                          else "  (top 200 m)"), fontsize=9)
            axx.set_xlabel("zonal-mean u [m/s]", fontsize=8)
            axx.tick_params(labelsize=7)
        axd = axes[2, col]
        for tag, ls in (("equator row", "-"), ("E10 +/-10", "--")):
            nk = len(jets[day][(tag, "lego")])
            d = jets[day][(tag, "lego")] - jets[day][(tag, "nemo")]
            fl = jets[day][(tag, "floor")]
            axd.plot(d, z[:nk], ls, color="C2", lw=1.4,
                     label=f"legoESM - NEMO, {tag}")
            axd.fill_betweenx(z[:nk], -2.0 * fl, 2.0 * fl, color="0.8",
                              lw=0, label="2x this band's own floor"
                              if tag == "equator row" else None)
        axd.axvline(0.0, color="k", lw=0.6)
        axd.set_ylim(200.0, 0.0)
        axd.set_xlabel("difference [m/s]", fontsize=8)
        axd.set_title(f"day {day}  difference (top 200 m)", fontsize=9)
        axd.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("depth [m]")
    axes[1, 0].set_ylabel("depth [m]")
    axes[2, 0].set_ylabel("depth [m]")
    axes[0, 0].legend(fontsize=6)
    axes[2, 0].legend(fontsize=6)
    fig.suptitle("Equatorial current structure, both models -- wet-cell zonal "
                 "mean, no f-weighted statistic anywhere.\n"
                 "The grey band in the bottom row is 2x that band's OWN measured "
                 "ensemble floor: outside it, the two models differ.")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/F3_equatorial_jets.png", dpi=140)
    plt.close(fig)

    # F4 -- zonal-mean dT by row and depth at day 360, with the six bands drawn.
    # TWO panels, because ONE is a lie either way.  A linear scale set by the
    # field's own maximum is set by the single most extreme equatorial cell and
    # blanks every other band; a symmetric-log scale shows all of them but
    # exaggerates the near-zero interior.  Both are shown, both are labelled,
    # and the linear panel's limit is a stated PERCENTILE rather than the max,
    # so the reader can see what the choice cost.
    d = keep[360]["dT"]
    n = wet.sum(axis=1).astype(np.float64)
    # the two DRY DOMAIN-WALL rows (0 and NY-1) have no wet cell at any level.
    # They are BLANKED, not divided by zero and not filled with 0.0 -- a 0.0
    # sitting beside wet values in a colour map is precisely the
    # heterogeneous-band trap this campaign has already been burnt by, and it
    # would also set the colour scale.  The blanking is stated on the figure.
    interior = np.ones(A.NY, dtype=bool)
    interior[0] = interior[A.NY - 1] = False
    if not np.all(n[interior].max(axis=1) > 0):
        raise SystemExit("FATAL: an INTERIOR T-row has no wet cell anywhere -- "
                         "that is a mask defect, not a domain wall")
    zm = np.where(wet, d, 0.0).sum(axis=1) / np.where(n > 0, n, 1.0)
    zm = np.where(n > 0, zm, np.nan)
    zm[~interior, :] = np.nan
    finite = zm[np.isfinite(zm)]
    vmax_full = float(np.max(np.abs(finite)))
    v995 = float(np.percentile(np.abs(finite), 99.5))
    lin_thresh = float(np.percentile(np.abs(finite), 50.0))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9), sharex=True)
    norms = (
        ("linear, clipped at the 99.5th percentile of |value|",
         matplotlib.colors.Normalize(vmin=-v995, vmax=v995)),
        ("symmetric log (every band visible; the near-zero interior is "
         "exaggerated by construction)",
         matplotlib.colors.SymLogNorm(linthresh=max(lin_thresh, 1e-12),
                                      vmin=-vmax_full, vmax=vmax_full)),
    )
    for axx, (label, norm) in zip(axes, norms):
        im = axx.pcolormesh(np.arange(A.NY), z, zm.T, cmap="RdBu_r",
                            norm=norm, shading="auto")
        for _, brows in BANDS[1:]:
            axx.axvline(brows.start - 0.5, color="k", lw=0.8)
        axx.axvline(STRUCTURAL_ZERO_ROW, color="0.4", lw=0.8, ls=":")
        axx.invert_yaxis()
        axx.set_ylabel("depth [m]")
        axx.set_title(label, fontsize=9)
        fig.colorbar(im, ax=axx, label="zonal-mean (lego - NEMO) T [K]")
    for a, (bname, brows) in enumerate(BANDS):
        j = np.arange(A.NY)[brows]
        axes[0].text(float(np.mean(j)), 150.0, bname.split()[0],
                     ha="center", fontsize=7, color="0.25")
    axes[1].set_xlabel("T-row (dotted = row 99, where f is exactly 0; the two "
                       "dry domain-wall rows 0 and 198 are blanked, not "
                       "zero-filled)")
    fig.suptitle(f"Where the water differs at one year, with the six bands "
                 f"drawn\nfull-field max |value| = {vmax_full:.3f} K, 99.5th "
                 f"percentile = {v995:.4f} K")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/F4_zonal_mean_dT_day360.png", dpi=140)
    plt.close(fig)
    print(f"\nfigures written to {out_dir}/F1..F4")


def git_sha_now():
    """(sha, dirty) of the tree this process is running from."""
    try:
        sha = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", _DIR, "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as exc:                                # pragma: no cover
        raise SystemExit(f"FATAL: cannot read a producer revision ({exc}) -- "
                         f"an unstamped artifact is refused")
    return sha, dirty


def stamp(art, out_dir, sha_start, dirty_start):
    """Provenance on the artifact.  A stampless artifact is refused elsewhere in
    this campaign; one written by this probe must not be refusable."""
    # The revision is captured at the START of the run and RE-CHECKED here.
    # Code review of 17881d91d caught this artifact stamped with a commit that
    # landed WHILE THE RUN WAS IN FLIGHT: the log's own provenance line read
    # fe2ec432f at the top and the stamp read 17881d91d at the bottom, and
    # `producer_tree_dirty` was written False. A stamp naming a revision that
    # did not produce the numbers is worse than no stamp, because it is
    # believed. The run is now REFUSED rather than mislabelled.
    sha, dirty = git_sha_now()
    if (sha, dirty) != (sha_start, dirty_start):
        raise SystemExit(
            f"FATAL: the working tree CHANGED during this run -- it started at "
            f"{sha_start}{' +dirty' if dirty_start else ''} and ended at "
            f"{sha}{' +dirty' if dirty else ''}. The numbers above were not all "
            f"produced by one revision and the artifact is refused rather than "
            f"stamped with whichever commit happened to be HEAD at write time.")
    art["_stamp"] = {
        "producer_sha": sha,
        "producer_tree_dirty": dirty,
        "probe": os.path.basename(__file__),
        "prereg": "PREREG_regional_audit.md",
        "lego_dir": X.LEGO_DIR,
        "nemo_dirs": [X.nemo_dir(i) for i in range(N_MEM)],
        "horizons_registered": list(HORIZONS),
        "bands": {n: [int(np.arange(A.NY)[r].min()), int(np.arange(A.NY)[r].max())]
                  for n, r in ALL_BANDS},
        "verdict_constants": {
            "K_PREREG": V.K_PREREG, "K_WELCH": V.K_WELCH,
            "QUANTUM_MARGIN": V.QUANTUM_MARGIN,
            "SATURATION_RATIO_MAX": V.SATURATION_RATIO_MAX,
            "ONE_SIDED_DECADES": V.ONE_SIDED_DECADES,
            "ESCALATION_FLOORS": ESCALATION_FLOORS},
    }
    p = f"{out_dir}/regional_audit.json"
    with open(p, "w") as fh:
        json.dump(art, fh, indent=1, default=float)
    print(f"artifact stamped at {sha}{' +dirty' if dirty else ''} -> {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", default="/tmp/dino_regional_audit")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    self_test()
    run(a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
