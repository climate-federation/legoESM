# Preregistration — NEMO testcase L2 GYRE round 122 day-240 magnitude ranking

Date: 2026-09-19

Incoming lane tip: `57051cbe2cf3adad83961286cf6d18c279df8c38`

This document is frozen before any Round-122 scientific scoring.  Evidence
will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round122/`.  Decision 45 and
operator note AB supersede Round 121's bit-walk handoff: this round ranks the
existing 360-day GYRE residual by its day-240 magnitude and does not repair the
one-ULP W plant, walk another operator, run a substitution ladder, or propose a
physics change.

## P0 — reference continuity and admitted inputs

The legoESM member is the immutable clean CPU/fp64 seed-0 run under
`phase3/year_equivalence/gyre/lego_seed0_year`, produced at
`4d250301588d3ed0ad83fb20d6bf520e175d576e` with 2,160 steps,
`dt=14,400 s`, and snapshots every six steps.  Its manifest and the requested
snapshots at days 30, 90, 180, 240 and 360 must exist and remain finite.

The user-named NEMO root `phase3/year_owners/nemo_seed0` stops at day 30.  The
already registered full-year continuation is
`phase3/year_fromrest/nemo_seed0`.  Before any later day is scored, the two
roots' day-30 restart must be byte-identical with the previously recorded
SHA-256
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`.
REFUTE and stop if the shared endpoint differs, if either source is missing,
or if the full-year file's recorded step is not exactly `6 * day`.  This is the
same fail-closed continuity rule used by the committed year-equivalence
receipt; no reference is silently substituted.

## P1 — instrument calibration and temporal birth

The committed `nemo_testcase_l2_gyre_year_owners.py` instrument is used without
changing its masks, metrics, regions, depth bands or loaders.  Its `--self-check`
must pass.  A current-tip `--step-gap 60` walk must reproduce the post-Round-110
trajectory: temperature remains at the rounding scale after one completed step
and first becomes materially non-bit after two completed steps, at approximately
`8.80e-8 K` wet RMS.  The exact number is a forecast, not a bar; REFUTE temporal
birth if the first material row is not the entry to step 3.  The walk may name
the birth step but, by its documented state-to-state design, cannot name an
operator inside that step.

The fixed-day scorer must reproduce T3D wet RMS
`6.890484901489568e-5 K` at day 30 and
`1.6446741930292448e-2 K` at day 240.  The latter is the Round-120/Decision-45
headline and is the primary instrument check.  REFUTE and stop the ranking on
any disagreement beyond the exact fp64 values emitted by the common scorer.

## P2 — one mechanically ranked table

The gap will be decomposed independently at days 30, 90, 180, 240 and 360.
The receipt will contain one table whose rows are:

1. the five physical fields, ranked only by the dimensionless fraction of
   NEMO's own from-rest signal because their native units differ;
2. the three disjoint depth bands `0--100 m`, `100--1000 m`, and `1000+ m`;
3. the three disjoint longitude thirds; and
4. the overlapping forcing-defined latitude cuts, visibly labelled
   overlapping rather than added to the disjoint rows.

For a disjoint T band with squared-error share `q` and whole-domain T3D RMS
`R`, the table's “T3D RMS carried” is fixed as `R * sqrt(q)`.  Its square is
that band's contribution to the whole-domain mean-square error, so the squared
contributions of each disjoint partition sum exactly to `R**2`.  The native
within-band RMS and cell count remain beside it.  No layer averaging, thickness
weighting or new mask is introduced.

Frozen spatial prediction: at day 240, `100--1000 m` is the largest disjoint
depth contribution, the west third is the largest disjoint longitude
contribution, and the `15--29 N` wind band contains more than half of the total
squared T error.  The same three rows led the residual at day 30 after the
Round-110 landing.  Each prediction is independently REFUTED if another
disjoint row is larger or if the wind-band share is at most one half.  Peak
cell, level and latitude are reported post-measurement but are not used to
select an owner.

## P3 — causal ownership boundary and candidate rule

`--decompose` partitions an already existing state difference.  It is blind to
which process created that difference, and its forcing latitude cuts overlap
the longitude and depth partitions.  Therefore a large region or depth row is
a localization target, not a process attribution.  Likewise `--step-gap`
names a step boundary, not an operator.  Round 122 will not call either one a
source statement and will not infer vertical mixing, advection, forcing, or a
configuration choice from spatial resemblance.

The largest day-240 **measured** row becomes the next candidate target in this
strict sense:

* if an existing controlled same-protocol counterfactual already isolates a
  process through day 240, that process may be ranked by its measured
  before-minus-after effect;
* otherwise the candidate is a process-level decomposition of the largest
  spatiotemporal target, and its day-240 contribution remains `UNMEASURED`.

Frozen prediction: the current artifacts contain no controlled process arm
through day 240, so this round will rank field/depth/region magnitudes but will
not name a causal physics owner.  REFUTE if a committed, one-variable,
same-protocol day-240 arm is found.  In the expected case the round is HELD,
no source code lands, and no `DECISION_NEEDED` is raised merely from a spatial
pattern.  A decision is raised only if a measured largest process owner itself
requires a new configuration or carried-state choice.

## P4 — scope, controls and reporting

The NEMO compiled record writes the exact Nbb entry state for the first sixty
steps, swaps the completed RK3 state into Nbb before output, calls `rst_write`
with that level, and the restart routine writes `sshn/un/vn/tn/sn` from Kbb.
The receipt will cite those statements from the compiled
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo` branch that produced the record.

The existing instrument's region-partition and time-level-registry controls
must pass.  The receipt citation gate must pass, and a deliberately shifted
compiled-source citation must print `status=FAIL` and exit nonzero.  A separate
read-only Codex pass must try to refute the ranking, the use of the full-year
reference continuation, and the no-causal-owner conclusion.  Since no model
or scientific instrument is changed, focused verification targets the owners
instrument, provenance/stamp checks, and citation gate; no expensive model
trajectory is rerun.

No NEMO source is modified; `makenemo` and `mpirun` are forbidden.  No card,
default, coefficient, timestep, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, `#1484` guard, or stabilizer changes.
No configuration choice is made.  DINO, LOCK_EXCHANGE, OVERFLOW and ORCA2 do
not execute any new statement because there is no production change; ORCA2
remains `UNMEASURED-WITH-SPEC` for a future native year ranking.
