# NEMO testcase L2 GYRE phase 3 — Round 89 RK3 assignment-boundary receipt

Date: 2026-09-14

## Outcome

**HELD; no production physics landed.**  Round 89 combined the authorized,
locally exact Round 88 Kaa/W/WZV/ZAD arm with source-operation materialization
of NEMO's vector RK3 assignment.  The new member was bit-exact on all six
recorded stage/face rows, and the inherited five-member Kaa/W/ZAD chain
remained bit-exact.  The complete 954-row trajectory comparison nevertheless
failed Rule 12 with 88 violations.  First-over-bar expanded at kt2 from U/V to
T/S/U/V, so the candidate was removed and preserved only as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round89_held_source_rounded_kaa_bundle.patch`.

The frozen mechanism prediction is **REFUTED and retained**.  Making the RK3
assignment observable did not compensate the authorized Kaa state.  The
candidate's kt2 and kt3 whole-step gaps are bit-for-bit the same printed values
as Round 88's rejected arm.  The first live non-bit statement boundary in the
new kt2 walk is instead stage 1's completed barotropic correction: U/V maxima
`1.2184073888699132e-10` / `1.6794210466741788e-10`, with every active cell
unequal.  This names the first statement boundary, not a causal owner: the
probe does not yet separate its depth-mean operands from its final add.

Candidate commits were `950677c2f718` and the instrument correction
`c47b6e7ea895`; `b8db71ab1a87` reapplied the identical held manifest solely
for the corrected complete ladder.  Production was restored at
`b93b9c8ea611`.  Every affected production path is byte-identical to frozen
preregistration parent `a845a6e2366b` (the Round 85 production baseline plus
the Round 89 preregistration).

## Frozen registration and retractions

The preregistration is
`docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l2_gyre_round89_rk3_assignment_boundary.md`,
committed as `a845a6e2366b` before the candidate was applied or measured.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round89/`.  All scientific
executions used CPU, JAX fp64/libm, production JIT, and a clean commit stamp.

Two procedural findings are retained rather than hidden:

1. `stage_walk.json` used the Round 46 stage-1 callback's stale pre-SELECT
   reciprocal and falsely printed a `6.0327552247607316e-2` assignment miss.
   The compiled program sets stage 1 to `rn_Dt/3`; commit `c47b6e7ea895`
   changed the instrument to derive all three clocks from the compiled stage
   schedule.  The corrected `stage_walk_corrected.json` makes all six local
   assignment rows bit-exact, so the old row is **RETRACTED** and cannot be
   printed by the current tool.
2. `gyre_rule12.json` was invoked with an unnecessary ENE waiver even though
   the oracle record exists.  It failed with 144 entries, of which 56 were
   coverage-only artifacts.  That count is **RETRACTED**.  The exact held
   manifest was reapplied at `b8db71ab1a87`; `gyre_rule12_complete.json`
   compares all 954 rows, has no present-only entry, and is the sole Rule-12
   verdict cited below.

## Local exactness and ordered walk

`stage_walk_corrected.json` (SHA-256
`a14ed78c360b25aba4687b7e007de71cb345848510f8530fc301b3c77da9d194`)
drives the one shared production helper with each NEMO stage's own Kbb, Krhs,
mask, and compiled duration.  Stage 1 U/V, stage 2 U/V, and stage 3 U/V each
have zero unequal active cells and zero maximum difference.  The stage-3 bare
row is explicitly a transcription identity because that intermediate was not
dumped; `stage_update_rule12.json` carries the helper through NEMO's calibrated
follow-on statements to a NEMO output and discharges both faces with zero
unequal cells (SHA-256
`00f3adda6fd61a8f6e89463ba4769437c2c8a9344b6699b60b9468cfc47447b8`).

The new one-ULP walk control prints `PLANT_FIRED` and exits 1
(`stage_walk_plant.json`, SHA-256
`7439994654af1e9cbe8f15b90f786bac1bbdca7852e0997361e67e883d81ad46`).
The composed Round-31 control also exits 1 with a one-cell
`4.235164736271502e-22` transcription move
(`stage_update_rule12_plant.json`, SHA-256
`01fb358a88d355d3279fef49fd7c9242fd2f4f782d7c6527b4c61c82722fdb6a`).

The inherited chain was re-proved rather than borrowed.  `kaa_scratch.json`
(SHA-256
`2bbf30f4d447f4afafa2120727da5826341850629cee9f68d7c62d5946ef5bc1`)
reports zero unequal cells for carried Kaa SSH (600), direct shared W (18,000),
captured stage-1 W (18,000), ZAD U (17,400), and ZAD V (17,100).

The live source-order rows are:

| boundary | U maximum | V maximum | interpretation |
|---|---:|---:|---|
| local stage-1 RK3 assignment | 0 | 0 | bit-exact on NEMO inputs |
| stage-1 post-barotropic | `1.2184073888699132e-10` | `1.6794210466741788e-10` | first live non-bit statement |
| stage-2 post-barotropic | `6.9206275655712274e-6` | `1.0321290368606531e-5` | compounding |
| stage-3 pre-LDF RHS | `9.3736777867059717e-10` | `1.9461433002220127e-9` | already non-bit |
| stage-3 post-LDF RHS | `9.3735933135246871e-10` | `1.9461512820065452e-9` | LDF is not the magnitude jump |
| stage-3 pre-implicit exposure | `1.3497974371473598e-5` | `2.8024578460895261e-5` | upstream inputs differ |
| stage-3 post-barotropic | `1.0649121615136092e-5` | `2.5414368832993306e-5` | completed stage |

## Rule-12 adjudication

The corrected ladder is `gyre_ladder_complete.json` (SHA-256
`f5106ec0ae1b0010e230a96650aa5821b87fe32d2d3e49f1dfa8f70f63f4d3d7`)
with residual sidecar SHA-256
`3903f03998ff114cb615b651a9d6e8762b4c2d4bfbde57e7b1cc373ed99aa112`.
`gyre_rule12_complete.json` (SHA-256
`b4431f915add31ec40ea871ed8109ab6f561b29eeb90e891f82ba6c73e4a074c`)
prints `FAIL`, compares 954 certified rows, finds 88 violations, and has a
largest worsening of `71009755143897.44` row-scale oracle ULPs.

| arm | kt2 U | kt2 V | kt2 T | kt2 S | kt3 T | kt3 S | first-over-bar | Rule 12 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Round 85 before | `2.7377110452773967e-12` | `3.284922138989399e-12` | `1.4210854715202004e-14` | `2.1316282072803006e-14` | `1.627497246303733e-4` | `6.327735185607253e-6` | kt2 U/V | baseline |
| Round 89 bundle | `6.733005735178965e-7` | `1.3183569256688065e-6` | `3.0652394795183113e-3` | `4.835710022078388e-3` | `1.566544749833554e-2` | `4.324733116938262e-3` | kt2 T/S/U/V | FAIL |

The bundle loses the kt2 T/S AT-BAR rows and enlarges rather than improves the
standing kt3-T magnitude target.  Therefore the day 1-30 lane was
**UNMEASURED because the mandatory ladder failed**.  The recorded before-arm
day-30 T gap remains `1.2397011295506804e-2 K`; no candidate day-30 value is
inferred.

LOCK_EXCHANGE and OVERFLOW candidate integrations were also withheld after the
GYRE rejection.  Both use the shared WS-RK3 implementation, while their
resolved thickness-weighted momentum arm does not execute the new vector-only
materialization; the other held Kaa/W/ZAD members still make the bundle a
shared-card risk.  DINO uses its MLF program and does not allocate the Kaa RK3
slot, but the held source-rounded W helper is shared, so DINO risk is explicit
and no DINO candidate claim is made.  ORCA2 is **UNMEASURED-with-spec**: before
any claim, a future arm must prove its selected integrator and restart schema;
if it selects this RK3 state, a missing Kaa scratch must fail loudly, then its
existing fidelity gate must pass.

## Compiled-source basis

GYRE's compiled stage clock sets stage 1 to `rn_Dt/3`, stage 2 to `rn_Dt/2`,
and stage 3 to `rn_Dt`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-148`,
`:198-202`, `:242-246`).  Those are the durations now used by the corrected
instrument.

For stages 1 and 2, the executing vector statement writes Kaa U/V as Kbb plus
`rDt*Krhs`, then applies the face mask
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674`).
The stage-3 vertical routine executes the same vector statement before its
implicit solve
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzdf.f90:163-170`).  These are
the statements materialized by the held member and proven exact above.

The compiled stage-2/3 source sequence is EOS, HPG, vorticity, then advection
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:433-482`).
Stage 3 next adds LDF
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:692-714`), calls
the vertical solve, forms depth means, and adds the barotropic correction
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:728-759`).  The
last range is the cited first non-bit statement boundary.  It supports only
the source-order location; a next-round operand split must decide whether the
depth mean, target barotropic velocity, or final add is first non-bit.

The other held members retain their Round 88 compiled-source basis: the
external solve precedes stage 1
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:188-201`); the compiled
stage-1 path constructs and consumes W before the RK stage program
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:326-339`); the QCO
stretch recurrence uses Kaa minus Kbb
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`); and the
stage program rotates the next Kaa scratch after stage 3
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226`).

## Review and verification

The required separate read-only Codex review failed before a reviewer model
started.  Its terminal verdict is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**.
The review artifact SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
The candidate was already mechanically rejected, so no `SHIP` interpretation
is possible.

Before scientific measurement, 121 focused source-assignment, Kaa/restart,
recipe, and restart tests passed.  On the final restored tree, 161 focused
tests covering the phase gate, assignment controls, citation machinery,
restart loading, and testcase recipe pass in 36.72 seconds; the log SHA-256 is
`ad8e65db0f5127c013bacd2bb9381224b0c2230f4428a9702db537bb85c57a37`.
The receipt citation gate passes all 12/12 compiled-source citations with no
failure or unmapped citation (SHA-256
`d6db66ce8cb349385c7efcfa342f8d987598a05d807bd13b8448a62e22a85851`).
Shifting the stage-1 clock citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE` (SHA-256
`a41c6ac446ed369cbf7c08f7279aa8b6c6ea15686d9d65d39edf48afad30524b`).

GitHub CLI authentication is invalid and no GitHub connector is installed, so
the receipt could not be posted to issue #1455; no external state was mutated.
The authentication audit is preserved with SHA-256
`65e4875da9730edfbfb6b4b3ee6461a0095f331b95bc078a7649d90d18e40b42`.

## OPEN — next round

1. Start from restored Round 85 production, never from the Round 89 held arm.
2. Preregister a kt2 stage-1 split that exposes the raw post-assignment U/V,
   the computed depth mean, the `uu_b/vv_b(Kaa)` target, `zub/zvb`, and the
   final add.  Compare each against the existing Round 46 record and name the
   first non-bit operand/statement inside the compiled barotropic correction.
3. Rank the compensating statement by whether it can explain the growth from
   the `1.2184e-10`/`1.6794e-10` stage-1 residual to the
   `6.9206e-6`/`1.0321e-5` stage-2 residual and ultimately kt3 T
   `1.627497246303733e-4 K` and day-30 T
   `1.2397011295506804e-2 K` on production.
4. Do not retry the Kaa/W/ZAD or source-rounded assignment members until that
   paired barotropic statement is locally bit-exact on NEMO inputs.  Then use
   the permitted bundle/bisection rule and the full 954-row comparison.
5. The day-30 and cross-card lanes remain contingent on a passing GYRE ladder;
   no result from either retracted Round 89 artifact may be promoted.
