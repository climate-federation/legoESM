# PRE-REGISTRATION — the 3-D T/S divergence atlas (#1455 follow-on)

Written and committed BEFORE any statistic in `ts_divergence_atlas.py` was
computed. The atlas is DESCRIPTIVE: it maps where the legoESM twin and NEMO
tracer fields depart over the verdict year and how the pattern evolves. The
verdict run (`a1387f1f7`) already established WHAT is wrong (the southern
basin transport, not the channel); this instrument asks WHERE the tracer
fields carry it and WHETHER the divergence moves.

## 0. Data, and the refusal rules

Twin pair only for the gap (member 0 each side); members 1-3 each side supply
the per-horizon chaotic floor.

* legoESM `/tmp/dino_verdict360/m{0..3}_*.npz` — `T3d_day{d}` / `S3d_day{d}`,
  d = 0,10,...,360 (37 horizons), fp32 storage, `PROVENANCE: HEAD=a7b940f75`
  stamped in each member's `.log`, `.launch_sha` = a7b940f75, control dtype
  fp64, ladder `both`, seasonal clock from restart adatrj.
* NEMO `.../DINO/RUN_VERDICT360_M{0..3}/DINO_<kt>_restart_*.nc` — `tn`/`sn`,
  kt = 5760 + 32*day, 20 horizons: 0,10,30,60,90,120,150,180,210,240,270,
  280,290,300,310,320,330,340,350,360.

SCORED HORIZONS = the 20 NEMO stamps (the intersection). Any horizon present
on one side only is DROPPED, never interpolated.

REFUSALS (all fatal, no fallback):
* a member missing its provenance stamp, or a stamp that does not match
  a7b940f75 / the certified NEMO binary's run dirs, is refused;
* any non-finite value on the wet mask is fatal;
* the day-0 identity control must hold on both T and S to the fp32 snapshot
  quantum (bar: max|d| <= 1e-5 in K and in g/kg); a larger day-0 difference
  means the two sides are not the same twin and the run aborts.

## 1. Mask and weights, stated once

WET MASK = `mesh_mask.tmask` AND the legoESM `land_mask` broadcast down the
column — the SAME 3-D mask on both sides, exactly as `acc_thermal_wind` and
`acc_driver_decomp` use it. 342134 wet T-cells.

VOLUME WEIGHT = `e1t * e2t * e3t_0` (partial-cell thickness), zero outside the
wet mask. THICKNESS-WEIGHTED throughout: the campaign's layer-averaging
retraction forbids any unweighted mean over levels or over cells of unequal
volume. Every rms in this instrument is

    rms(subset) = sqrt( sum_subset V * d^2 / sum_subset V ),   d = lego - NEMO.

MASK-BITE PLANT (mandatory control, printed before any number is used): set
one DRY cell's difference to 1e6 and assert every reported statistic is
unchanged to 1e-12 relative. A statistic that moves means the mask does not
bite and the run aborts.

## 2. Partitions, fixed here so they cannot drift

REGIONS (T-rows, `acc_driver_decomp.LAT_GROUPS`, J0=14, J1=48):
* `south` rows 0..13 (row 0 is dry; the basin is rows 1..13)
* `channel` rows 14..48 (the re-entrant band)
* `north` rows 49..198

DEPTH CLASSES — FIXED DEPTH BANDS on `gdept_1d`, IDENTICAL on both sides:
* `upper`    gdept_1d < 200 m   (levels 0..12)
* `interior` 200 m <= gdept_1d < 1400 m (levels 13..27)
* `abyss`    gdept_1d >= 1400 m (levels 28..35)

These are NOT a diagnosed mixed layer. A model-diagnosed MLD would differ
between the two models and so would be a protocol difference — a confound —
so the split is geometric and shared. 1400 m is the campaign's own
upper/deep split (`acc_thermal_wind.DEEP_M`).

WALL SUB-ROWS: the campaign's committed southern split
(`southern_circulation_budget.py`) is wall rows 1-5 vs main rows 6-13. That
definition is used verbatim; no new wall definition is invented here.

## 3. QUESTION 2 — first departure: does divergence grow in place or spread?

EARLIEST SCORED HORIZON = day 10 (the first horizon both sides carry).

Pre-registered statistics, computed on |d| over the wet mask, volume-weighted:

* **S1 pattern correlation.** Volume-weighted Pearson r between |d| at day 10
  and |d| at each later horizon, over all wet cells; and the same for the
  SIGNED d. Reported for T and S separately.
* **S2 centroid displacement.** Volume-weighted centroid of `V*d^2` in
  (T-row index, gdept_1d). Reported as displacement from the day-10 centroid,
  in rows and in metres.
* **S3 day-10 hotspot retention.** Let H10 = the set of wet cells in the top
  5% of `V*d^2` at day 10 (by volume share, so H10 holds 5% of the wet
  volume). Retention(t) = fraction of the total volume-weighted `d^2` at
  horizon t that lies inside H10. Null (divergence uniformly re-drawn) = 0.05.

BARS, registered here:
* GROWS IN PLACE if r(|d| day10, day360) >= 0.5 AND retention(360) >= 0.25
  (5x the null).
* SPREADS/PROPAGATES if retention(360) < 0.15 AND the S2 centroid moves more
  than 10 T-rows or more than 500 m.
* Anything else is reported as MIXED and no mechanism is claimed.

## 4. QUESTION 3 — feedback fingerprint: tracer divergence vs the momentum deficit

The momentum deficit geography is measured here from the SAME artifacts, not
quoted: per-row zonal transport deficit
`D(j,t) = group_transport(lego u, row j) - group_transport(NEMO u, row j)`
using `acc_driver_decomp.group_transport` with `A.umask` and the `_avg`
reducer (lons 2..-2) — the campaign's committed reduction, imported, not
re-spelled. Restricted to southern-basin rows 1..13.

Per-row tracer divergence `P_X(j,t)` = the volume-weighted rms of `d_X` over
row j, all depths, wet cells, X in {T,S}.

Pre-registered statistics:
* **F1 co-location.** Spearman rank correlation across the 13 rows between
  `P_X(j,t)` and `|D(j,t)|`, at every scored horizon. BAR: |rho| >= 0.56
  (two-sided p < 0.05 at n=13) called COUPLED; below that, NOT COUPLED at
  this sample size.
* **F2 lead/lag.** Pearson correlation of the 13-row profiles across horizon
  offsets, `corr( P_X(., t), |D|(., t+k) )` for k in -2..+2 scored horizons,
  pooled over t. The argmax k is reported: k<0 = tracer LEADS momentum,
  k>0 = tracer LAGS, k=0 = simultaneous. Reported with the full k-profile so
  a flat profile is visible as flat.
* **F3 wall-vs-main share.** Share of the southern-basin volume-weighted
  `d^2` in wall rows 1-5 vs main rows 6-13, against the same rows' share of
  |D|. AVOIDS is declared only if the tracer share is below half the deficit
  share at day 360 AND at the full-year mean.

No mechanism will be claimed from F1-F3 alone. If any mechanism claim
emerges, a second independent reviewer is required per the campaign rules;
the descriptive atlas needs one.

## 5. What this instrument does NOT do

It does not edit any floor constant, does not print a verdict string, and
does not compute a transport metric that is not already committed. It reads
recorded artifacts and writes figures plus a table.
