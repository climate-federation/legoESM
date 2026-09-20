# NEMO testcase L2 GYRE phase 3 — round 122 day-240 magnitude-ranking receipt

Date: 2026-09-19

Incoming tip: `57051cbe2cf3adad83961286cf6d18c279df8c38`

Status: **HELD — the day-240 residual is localized to a southern, western,
upper-100-m mode, but no same-protocol controlled arm assigns that mode to a
process; no physics or configuration changed.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round122/`

## Outcome first

The registered scorer reproduces the immutable day-240 T3D wet RMS exactly:
`1.6446741930292448e-2 K`.  This is not a continuation of the last-bit stage
walk.  It is a magnitude result on the already admitted one-year trajectory.

Three independent partitions identify the same spatial target at day 240:

- the upper `0--100 m` contains `92.82437411570927%` of squared T error;
- the west third contains `90.96679270762623%`; and
- the forcing-defined `15--29 N` wind band contains `91.01474126351784%`.

The broad southern cut at or below `37.2 N` contains
`93.65233038106547%`.  The peak is `-1.8862674902076684 K` at array index
`[j=1,i=2,k=1]`, `15.09642710832668 m` and `16.867151077876947 N`.
The same peak location remains at day 360.  At day 180 the pattern is
different: only `19.52679751417199%` is above 100 m, `7.404526371146058%`
is in the west third, and `8.365950712258485%` is in the wind band.  At the
requested observation cadence, the large surface-west-wind mode is therefore
born between days 180 and 240 and persists through day 360.

This directly **REFUTES** the frozen prediction that `100--1000 m` would be
the largest day-240 depth owner.  That band carries only
`7.173040933054871%`; `0--100 m` is larger by squared-error share by a factor
of `12.941`.  The west-third prediction and the greater-than-half wind-band
prediction are confirmed.  At day 30, the original pattern does hold:
`100--1000 m` is the largest depth band (`54.05023224826212%`), the west
third is the largest longitude band (`54.97896836971331%`), and the wind band
contains `83.93404235550986%`.

No process owner is measured.  The decomposition partitions the state error;
it does not run a counterfactual process trajectory.  The largest actionable
target for the next round is therefore the **southern/western upper-ocean mode
that appears between days 180 and 240**, not W, ZAD, vertical mixing,
advection, surface forcing, or any other inferred operator.  No
`DECISION_NEEDED` follows from a spatial pattern.

## Frozen provenance and reference continuity

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round122.md`, committed
before measurement as full commit
`f563e05b8fe4cd31b00a786bdb8436984a57a938`.  Every authoritative scorer
artifact carries that clean commit.  The scientific-measurement artifact list
frozen before verification is `round122/artifacts.sha256`, whose own SHA-256 is
`742097f1aae14d03a6d2ee8ed6b8d47180eee95c357b1ff55fb645abb24015a2`.

The user-named daily NEMO root stops at day 30.  Before using the registered
full-year continuation, the gate re-hashed both roots' day-30 restart.  Both
are byte-identical at
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`.
All requested legoESM snapshots and NEMO restarts at days 30, 90, 180, 240
and 360 exist.  `input_continuity.log` also records the clean scorer commit.
Thus no later-day NEMO root was silently substituted.

The immutable legoESM member is
`phase3/year_equivalence/gyre/lego_seed0_year`, clean commit
`4d250301588d3ed0ad83fb20d6bf520e175d576e`, 2,160 fp64 CPU steps at
`14,400 s`, with snapshots every six steps.  Round 122 reads it; it does not
rerun or modify it.  The decompose loader's fixed directory spelling is
`lego_seed0`, so the evidence root contains a read-only symbolic-link adapter
from `round122/decompose_input/lego_seed0` to that exact `lego_seed0_year`
directory.  The emitted JSON records the adapter as `lego_root`; the source
manifest and snapshot bytes remain the immutable member above.

## Instrument calibration and temporal birth

The unmodified owners instrument's self-check passes all controls.  In
particular, the longitude thirds and the two `37.2 N` latitude halves
partition the wet surface, every seasonal surface field moves under its
phase control, the literal-pi and year controls move shortwave radiation, and
the beyond-step-60 registry plant raises.  The literal terminal line is:

```text
self-check: all checks passed
```

The 60-entry state walk reproduces exact initial identity, T RMS
`2.0293251845119845e-15 K` after one completed step, and the first material T
row after two completed steps at `8.802191255844365e-8 K`.  It therefore
confirms that the global residual is first visible at the entry to step 3.
That row is a state boundary, not an operator attribution.

The fixed-day scorer reproduces the required T3D values:

- day 30: `6.890484901489568e-5 K`;
- day 90: `1.8645021144913585e-3 K`;
- day 180: `3.580551011866709e-3 K`;
- day 240: `1.6446741930292448e-2 K`; and
- day 360: `1.1223573910167267e-2 K`.

The day-240 instrument check is exact, so the ranking is admitted.  The
partition-closure artifact reports depth/longitude squared shares of
`0.9999999999999999`/`1.0` at day 240 and `1.0`/`1.0` at day 30; the latitude
halves close to fp64 summation.  The wind band is explicitly overlapping and
is never added to a disjoint partition.

## One day-240-ranked owner table

“Carried RMS” for each T row is the preregistered global-equivalent quantity
`R * sqrt(q)`, where `R` is whole-domain T3D RMS and `q` is that row's share
of squared T error.  Its square is the row's contribution to global MSE.
Native RMS and cells are shown so a broad cut cannot look artificially small.
Rows from different partitions overlap and must not be added.  The five field
rows use their own units and are ranked separately by gap divided by NEMO's
from-rest signal; they are not converted into temperature.

| category rank / measured owner | day-240 magnitude | day-30 magnitude | where and when born | evidence row |
|---|---:|---:|---|---|
| Process 1. causal operator | **UNMEASURED** | **UNMEASURED** | Global state first material after two steps; dominant long-horizon spatial mode appears between days 180 and 240 | 360-day manifest census contains no current-baseline controlled process arm |
| Stage/state 1. whole T residual | `1.6446741930292448e-2 K` | `6.890484901489568e-5 K` | first material boundary: entry to step 3, `8.802191255844365e-8 K` RMS | `day_gap_registered.json`; `step60/step_gap.json` |
| Field 1. v | `4.360950166196743e-4 m s-1` / `1.728880192701625e-2` of NEMO signal | `4.620276844589294e-6 m s-1` / `3.290394947928958e-4` | leading dimensionless field changes from u at day 30 to v at days 180 and 240 | `decompose_day240.json` field row |
| Field 2. u | `3.0729451134022727e-4 m s-1` / `1.3809479848006697e-2` | `5.579572380229224e-6 m s-1` / `4.976972232332483e-4` | already non-bit after two steps; process unassigned | field row |
| Field 3. T | `1.6446741930292448e-2 K` / `8.832501255081351e-3` | `6.890484901489568e-5 K` / `5.868608987447977e-5` | first material after two steps; large mode appears days 180--240 | field row |
| Field 4. S | `1.231719834021483e-3 psu` / `8.588514941277712e-3` | `1.1781247458573714e-5 psu` / `4.0214936448938944e-4` | coupled tracer response; process unassigned | field row |
| Field 5. SSH | `1.9347330129779557e-4 m` / `1.4664221471299658e-3` | `6.802771093693658e-6 m` / `1.3460277052470017e-4` | already non-bit after two steps; process unassigned | field row |
| Latitude 1. south at or below 37.2 N | carried `1.5916192096906763e-2 K`; `93.6523%`; native `1.8440005054143303e-2 K`; 13,410 cells | carried `6.80320519651173e-5 K`; `97.4827%`; native `7.881981911516568e-5 K` | broad southern localization; overlaps depth/longitude axes | `ranked_spatial_rows.json` |
| Depth 1. 0--100 m | carried `1.584568041818321e-2 K`; `92.8244%`; native `3.068502818447437e-2 K`; 4,800 cells | carried `4.649466678665804e-5 K`; `45.5309%`; native `9.003653507608921e-5 K` | becomes dominant between days 180 and 240 | depth row |
| Overlap 1. wind band 15--29 N | carried `1.569046257248757e-2 K`; `91.0147%`; native `2.6777976881024523e-2 K`; 6,180 cells | carried `6.312753847797236e-5 K`; `83.9340%`; native `1.0773600574932048e-4 K` | dominant at days 30, 240 and 360, not at day 180 | region row; not additive |
| Longitude 1. west third | carried `1.5686328988609177e-2 K`; `90.9668%`; native `2.7169518792511695e-2 K`; 6,000 cells | carried `5.109143238918017e-5 K`; `54.9790%`; native `8.849295672952466e-5 K` | becomes dominant again between days 180 and 240 | region row |
| Depth 2. 100--1000 m | carried `4.404854144500463e-3 K`; `7.17304%`; native `7.6294311782051675e-3 K`; 6,000 cells | carried `5.065806158517379e-5 K`; `54.0502%`; native `8.774233647846869e-5 K` | day-30 leader; displaced by the surface mode by day 240 | depth row |
| Longitude 2. east third | carried `4.209177382097215e-3 K`; `6.54990%`; native `7.290509083862157e-3 K`; 6,000 cells | carried `1.688340914006502e-5 K`; `6.00372%`; native `2.9242922435563546e-5 K` | dominates at day 180, not day 240 | region row |
| Latitude 2. north above 37.2 N | carried `4.1436878811031376e-3 K`; `6.34767%`; native `8.20572474867301e-3 K`; 4,590 cells | carried `1.0932434458149988e-5 K`; `2.51730%`; native `2.164944623498014e-5 K` | day-180 mode is northern; day-240 mode is not | region row; overlaps depth/longitude axes |
| Longitude 3. interior third | carried `2.591761707564726e-3 K`; `2.48331%`; native `4.4890629586135855e-3 K`; 6,000 cells | carried `4.304061163595572e-5 K`; `39.0173%`; native `7.454852614231084e-5 K` | secondary | region row |
| Depth 3. below 1000 m | carried `8.361920903955442e-5 K`; `0.002585%`; native `1.322135783534449e-4 K`; 7,200 cells | carried `4.45925717990591e-6 K`; `0.41882%`; native `7.050704680480471e-6 K` | negligible in the day-240 peak | depth row |

The table deliberately does not assign the broad south row, the surface row,
the wind-band row and the west row to four processes.  They are overlapping
views of one state difference.  Likewise v being the leading dimensionless
field is not proof that momentum physics created T's surface mode.

## Controlled-arm audit and causal boundary

`gyre_360_manifest_census.json` mechanically enumerates all 13 GYRE-zco
360-day manifests under the campaign root.  It finds the immutable current
baseline, old pre/post-ENE ensembles, the old Round-60 control, and a historical
card-reconciliation pair.  It finds no intervention based on the current
immutable member that changes one physical process and follows that change to
day 240.

The historical card pair is not silently promoted into that role.  Its arms
were produced at `a25271eb6` and `f94061551`, before the current baseline and
before the Round-110 magnitude landing; the selected change was the bundled
freshwater/fix-eta card reconciliation, not a one-variable explanation of the
current residual.  The pre/post-ENE and Round-60 members are earlier still.
Cross-tip subtraction of any of those from the current `4d2503015` member
would mix upstream trajectories and fail the preregistered same-protocol
counterfactual rule.

P3 is therefore **CONFIRMED**: process, RK3 stage, and source statement
contributions at day 240 remain `UNMEASURED`.  This is the round's limiting
result, not permission to infer a process from latitude or depth.

## Prediction ledger

- **P0 CONFIRMED.** The two NEMO day-30 endpoints are byte-identical at the
  frozen hash; all five requested later inputs exist.
- **P1 CONFIRMED.** Self-check passes, the first material state row is the
  entry to step 3 at `8.802191255844365e-8 K`, and the day-30/day-240
  headlines reproduce exactly.
- **P2 PARTLY CONFIRMED AND PARTLY REFUTED.** West-third leadership and the
  greater-than-half wind-band share are confirmed.  The frozen depth forecast
  is refuted: `0--100 m`, not `100--1000 m`, owns the day-240 depth partition.
  All three frozen day-30 rankings reproduce.
- **P3 CONFIRMED.** No current-baseline same-protocol process counterfactual
  exists, so no causal physics owner is named.
- **P4 CONFIRMED.** The measurement is read-only, the committed controls pass,
  and no source, card, carried state, harness, or stabilizer changed.

## Landing, blast radius and required headlines

Nothing lands.  There is no candidate trajectory, so Decision 43 and Decision
45 admission are not invoked and no before arm moves.  The required unchanged
headlines remain:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

Because no executable statement changed, GYRE, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW have zero numerical blast radius in this round.
ORCA2 remains **UNMEASURED-WITH-SPEC**: acquire a native year member and NEMO
restart series, apply this exact fp64 wet-mask/depth/region scorer, and then
require the same causal process counterfactual before naming an owner.

## Compiled source that defines the scored states

The producer's compiled step program opens the first-60-step entry record,
writes `Nbb` plus the resolved dimensions, then writes T/S, U, V and SSH from
that exact `Nbb` slot at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:89-99`.
This is why `--step-gap` rows are entry states rather than operator tendencies.

For the year restarts, the compiled program executes stage 3 and swaps the
completed `Naa` state into `Nbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222`.
It passes that `Nbb` to the restart writer at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:260`, and the compiled
restart routine writes SSH, U, V, T and S from `Kbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90:176-180`.
Thus the day rows are completed-step prognostic states from the record's
compiled branch.  These citations do not turn a state decomposition into an
operator attribution.

## Verification

The clean-tree citation gate found all four compiled-source citations, no
unmapped citations, no failures and no failing map entries.  Its literal
verdict was `status=PASS`; the artifact and log have identical SHA-256
`a0869a19690cc46ff76108079166d843668af1ccf59a9c536b2e54b565ea35e8`.
The clean worktree stamp names commit
`af588e6a8767d078a37d36f8d168ff74a23c2002`.

The deliberately shifted first-entry citation moved both endpoints by two
lines.  It found the real first symbol at line 89, emitted
`SYMBOL-NOT-AT-LINE`, printed `status=FAIL`, and exited 1 as required.  The
plant JSON/log SHA-256 is
`58f21f4741eff604629439dd27e8090e13e6d777d43857e970d856fae2ab6953`.

The focused owner, citation and provenance suites ended literally:

```text
============================= 39 passed in 31.56s ==============================
```

The focused log SHA-256 is
`0ea23e58a6b4e27f210f42aa540504e483563f0a4c4d966ffbdc41147007a1ca`.
No model implementation or scientific instrument changed, so the frozen P4
verification plan did not spend the round's CPU budget rerunning the 8,000+
model-test trees.  The changed executable file is only the receipt citation
map, covered by the cited gate suite.

The required independent command was run as `codex exec --sandbox read-only`
against the complete incoming-tip-to-receipt diff with an adversarial prompt
covering the metric, root continuity, temporal localization, counterfactual
census, causal boundary and compiled citations.  Its verbatim terminal result
was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review is unavailable in-sandbox; no `SHIP` or `DO NOT SHIP`
verdict is fabricated.  The complete review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN — exact handoff to round 123

1. Stay on magnitude.  Do not resume the W/ZAD one-ULP selector, the local
   stage-bit walk, or infer a process from the day-240 map.
2. The next candidate target is the southern/western `0--100 m` temperature
   mode that appears between days 180 and 240.  Preregister a process-level
   discriminator over that interval, with the whole-domain T3D day-240 effect
   as the ranking quantity and the localized band only as a diagnostic.
3. Extend an existing committed owner/stage instrument rather than creating a
   parallel harness.  The discriminator must compare NEMO and legoESM process
   boundaries on their own independent trajectories and must include a
   propagating production control.  A one-step tendency, a spatial resemblance,
   or a legoESM-only omit arm is not a day-240 owner.
4. If existing records cannot provide those boundaries, preregister the exact
   compiled-order fields and write a new fail-closed NEMO acquisition card for
   the day-180-to-240 interval.  Do not request that acquisition until its
   process rows, magnitude metric, expected byte layout, passive-instrument
   restart check and falsifiers are frozen.
5. Only the largest measured process effect becomes an implementation
   candidate.  If it requires a card or carried-state choice, stop with
   `DECISION_NEEDED`; otherwise it still must pass the Decision-43 ladder/month
   and Decision-45 registered year rows before landing.  DINO and every
   recipe-derived executing card remain required for any shared statement.

No acquisition script and no configuration or carried-state decision are
needed from Round 122 itself.
