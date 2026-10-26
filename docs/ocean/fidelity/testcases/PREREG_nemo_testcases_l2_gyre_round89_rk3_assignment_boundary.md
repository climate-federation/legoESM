# Round 89 preregistration: RK3 assignment boundary for the Kaa/W/ZAD bundle

Date: 2026-09-13

Frozen production commit: `3f64aeea16a435c9abb689f2f3580854c926dec0`

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round89/`

## Inherited magnitude and baseline

The production baseline is the landed Round 85 arm.  Its certified ladder rows
are kt2 T/S/U/V = `1.4210854715202004e-14`,
`2.1316282072803006e-14`, `2.7377110452773967e-12`, and
`3.284922138989399e-12`; kt3 T/S = `1.627497246303733e-4` and
`6.327735185607253e-6`.  Its day-30 maximum temperature gap is
`1.2397011295506804e-2 K`.

Decision 39 authorizes the shared NEMO-identity state to carry and
restart-load NEMO's pre-solve RK3 Kaa SSH scratch.  Round 88 proved the Kaa,
W, WZV, and ZAD members locally exact, but every tested arm produced the same
large trajectory failure: kt2 T/S/U/V = `3.0652394795183113e-3`,
`4.835710022078388e-3`, `6.733005735178965e-7`, and
`1.3183569256688065e-6`; kt3 T/S = `1.566544749833554e-2` and
`4.324733116938262e-3`.  The production code and recorded Round 85 before arm
are therefore unchanged.

The Round 88 stage rows constrain the first source-order divergence.  Its
stage-1 and stage-2 completed U/V rows remained AT-BAR, while the completed
stage-3 U/V rows acquired the large failure.  The first remaining non-bit
statement in that walk is the shared explicit RK3 vector assignment: the
Round 88 receipt records `5.421010862427522e-20` at stage 1 instead of zero.

## Compiled statements and branch proof

The compiled GYRE record is
`cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo`.  This round cites that
record, never generic source.

* `stprk3_stg.f90:666-675` executes the GYRE vector branch and writes Kaa U/V
  from Kbb plus `rDt * Krhs`, followed by the face mask, at stages 1 and 2.
* `dynzdf.f90:162-170` executes the corresponding vector-form explicit Kaa
  write before the stage-3 implicit vertical solve.
* `stprk3_stg.f90:433-482` establishes the stage-2/3 EOS, HPG, vorticity, and
  advection source order; `stprk3_stg.f90:692-760` establishes the stage-3
  lateral diffusion, vertical solve, and barotropic-correction order used by
  the diagnostic walk.

The candidate does not add a stabilizer or alter these formulas.  It makes the
two binary64 operations and the mask multiplication observable at the same
assignment boundary as the compiled Fortran statement.  Calibration-grade
acceptance requires zero unequal elements against a direct NumPy replay of the
compiled expression on the recorded NEMO inputs.  A merely small residual is a
failure.

## Reuse and candidate membership

The implementation search found the shared update in
`rk3_stage_velocity_update`, the Round 88 Kaa/W/ZAD arm in the committed held
manifest, and existing stage hooks and Round 46 oracle records.  This round
will extend those seams rather than add a second RK3 implementation or another
record parser.

The preregistered candidate bundle contains every member below:

1. Round 88's authorized pre-solve Kaa SSH scratch state and loud restart-load
   contract.
2. Round 88's compiled-order initial W/WZV association.
3. Round 88's source-round WZV construction and removal of the post-solve
   stage-1 ZAD reassociation.
4. A source-rounded vector RK3 assignment in the one shared stage update used
   by stages 1, 2, and the pre-implicit part of stage 3.

Members 1-3 must retain their Round 88 zero-unequal local proofs.  Member 4
must independently produce zero unequal U and V elements on every recorded
NEMO input row exercised by the new walk.  No member may be credited from its
trajectory effect alone.

## Frozen prediction and falsifiers

The prediction is that the common Round 88 failure came from carrying the
authorized scratch through a JAX expression that lacked the compiled RK3
assignment boundary.  With all four members together:

* the stage-1 and stage-2 completed U/V rows will become bit exact rather than
  merely AT-BAR;
* the first non-bit stage-3 row will move later than the explicit Kaa update,
  and the completed kt2 U/V gaps will not exceed the Round 85 values
  `2.7377110452773967e-12` and `3.284922138989399e-12`;
* kt2 T/S will remain AT-BAR;
* kt3 T/S will improve from `1.627497246303733e-4` and
  `6.327735185607253e-6`, respectively; and
* the day-30 maximum T gap will improve from `1.2397011295506804e-2 K`.

The mechanism is **REFUTED** if the new assignment is not locally bit exact,
if the Round 88 local proofs regress, or if the first stage-3 divergence is
already present before the explicit Kaa write.  The candidate is **HELD** and
reverted if any AT-BAR row leaves the bar, first-over-bar moves earlier, an
unregistered row moves, or the kt3-T and day-30 magnitude criteria do not
improve.  A failed full candidate may be bisected on the certified ladder, but
no bisection arm lands independently without a new locally exact ownership
proof.

## Frozen measurement order

1. Add a fail-closed Round 89 stage-walk probe, stamped to the candidate
   commit.  For kt2 it reports, in source order, completed stage-1 U/V,
   completed stage-2 U/V, stage-3 RHS before and after LDF, the explicit Kaa
   U/V immediately before the implicit solve, and completed post-barotropic
   U/V.  Each available row is compared with the Round 46 recorded oracle.
   Missing oracle rows are printed `UNMEASURED`, not inferred.
2. Run its planted one-ULP violation.  The plant must exit nonzero.
3. Re-run the Round 88 local Kaa/W/WZV/ZAD proof and restart-contract tests.
4. Run the certified GYRE kt=1..10 ladder.  Every moved row is registered in
   the receipt.
5. Only if the ladder is admissible, run member 0 for days 1-30 from the
   frozen Round 85 before arm and score every day.  The baseline is never a
   scratch toggle.
6. Exercise LOCK_EXCHANGE and OVERFLOW focused tanks, state whether DINO
   executes each changed statement, and leave ORCA2 `UNMEASURED` with an
   explicit execution/specification statement.
7. Run a separate read-only Codex review, citation gate plus shifted-citation
   plant, and focused CPU tests.

All JAX measurements use CPU, fp64, the repository precision policy, and the
campaign PYTHONPATH.  No NEMO build or integration is requested: all needed
oracle inputs already exist.  This round makes no new configuration or
carried-state choice; Decision 39 supplies the only required authorization.
