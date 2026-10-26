# NEMO testcase L2 GYRE round 98: kt=1 stage-1 W walk

Date: 2026-09-16  
Status: **STOPPED_FOR_RECORD; direct stage-output proof did not close**
Writable clone: `/tmp/autopilot-work-f6Sj77YL`  
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round98`

## Scope and immutable before arm

This round executed the round-97 OPEN item and Decision 41: remain at kt=1
stage 1, walk W in compiled order from NEMO's recorded stage entry, and permit
the held round-97 full-RHS patch to re-enter only if a same-stage W candidate
makes the direct stage output bit-identical.  The preregistration was committed
as `b081110fce98eb04ed2050a8ef8eaaa9d4751b69` before this measurement.

The immutable trajectory before arm remains the round-96/97 arm (the round-85
after arm).  No shared ocean implementation, selector, configuration, restart,
freshwater term, or carried state changed in this round.

## Instrument correction and controls

The former stage table compared kt=1 stage-1 `ww` to the pre-external,
velocity-form W used by ZAD.  That is not the W consumed by the stage-1 tracer
program.  The consolidated round-46 stage gate now retains that field as the
explicit `pre_external_zad_operand_w` diagnostic and scores stage-1 output W
against the already-admitted direct post-transport record
`round21_oracle_v2_stage_ww/oracle_rkstage_ww_kt00000001_s1.bin`.

The first attempt failed before producing a scientific JSON because it mixed
the pre-external `r1_Dt=1/14400` and pre-external SSH slot with the direct
tracer boundary (`rDt=4800`).  The corrected instrument uses the direct record's
clock and post-external SSH.  A literal scalar replay remains 5,289 cells from
the direct compiled output, maximum `2.6469779601696886e-23`; therefore it is
used only to discriminate statement ordering, never as the final W bit oracle.

Final controls:

| Control | Expected exit | Observed exit | Verdict |
|---|---:|---:|---|
| one-ULP transport operand | nonzero | 1 | fired |
| one-ULP recurrence carry | nonzero | 1 | fired |
| wrong commit stamp | nonzero | 1 | fired |

Artifacts are `stage_w_walk.json`, `stage_w_transport_plant_final.log`,
`stage_w_carry_plant.log`, and `commit_stamp_plant.log`.  The focused test file
passed 17 tests after adding a fail-closed reader for the next discriminating
record.

## Compiled-order W result

Given NEMO's admitted kt=1 stage-1 operands, production JIT first ceases to be
bit-identical at `hdiv`, not at the preregistered `e3div` boundary:

| Boundary | production unequal | production max abs | source-rounded unequal | source-rounded max abs |
|---|---:|---:|---:|---:|
| transport U/V through scaled numerator | 0 | 0 | 0 | 0 |
| `hdiv` | 4,766 | `6.462348535570529e-27` | 0 | 0 |
| `e3div` | 3,752 | `3.308722450212111e-24` | 0 | 0 |
| stretch | 0 | 0 | 0 | 0 |
| bracket | 11,937 | `3.9549573037691636e-24` | 0 | 0 |
| outgoing carry | 8,543 | `7.651420666115506e-24` | 0 | 0 |
| direct recorded `ww` | 10,827 | `2.9778502051908996e-23` | 5,289 | `2.6469779601696886e-23` |

Thus the preregistered prediction "exact through hdiv; first non-bit at
e3div" is **REFUTED**.  The prediction that source-statement materialization
would close the direct W output is also **REFUTED**.  It closes every replayed
scalar boundary but not the compiled direct record.  Decision 41 therefore
forbids combining it with the held full-RHS patch, and there is no production
candidate to submit to Rule 12.

## Stage-twin tables

The last complete consolidated tables are the round-97 tables at the same
production physics tip.  The only invalid row in their given-entry table is the
old kt=1 stage-1 W reference; round 98 replaces that row with the direct result
above.  The corrected full consolidated process was attempted twice, but the
environment killed it during the large CPU compilation with an empty log and
no JSON.  No row is inferred from that failed process.

Given NEMO entry, complete prior table plus the measured replacement:

| step/stage | rows | non-BIT | maximum non-BIT row |
|---|---:|---:|---|
| kt1 external | 11 | 0 | none |
| kt1 stage 1 | 17 | 2 | `zFw`: 10,232, `1.9895196601282805e-13`; corrected `ww`: 10,827, `2.9778502051908996e-23` |
| kt1 stage 2 | 17 | 3 | `ww`: 18,000, `7.90831526305093e-7` |
| kt1 stage 3 | 17 | 6 | T: 8,851, `1.2830175028447854e-2` |
| kt2 external | 11 | 11 | Hu_avg: 580, `1.2029895814613667e-4` |
| kt2 stage 1 | 17 | 7 | V: 17,100, `9.326814945542078e-6` |
| kt2 stage 2 | 17 | 6 | V: 17,100, `9.619435136954887e-6` |
| kt2 stage 3 | 17 | 7 | T: 17,988, `7.97034157194787e-3` |

The first owned non-exact stage output is now kt1 stage-1 W, AT-BAR but not
BIT.  The old `3.5937485546815465e-8` W classification is retracted by the
instrument itself and cannot print as the tracer-W stage result again.

Chained table (unchanged production, complete round-97 certification):

| step/stage | rows | non-BIT | maximum non-BIT row |
|---|---:|---:|---|
| kt1 external | 11 | 1 | SSH: 600, `4.336808689942018e-19` |
| kt1 stage 1 | 17 | 7 | TKE avm: 5,714, `8.176154756539789e-2` |
| kt1 stage 2 | 17 | 8 | TKE avm: 5,714, `8.176154756539789e-2` |
| kt1 stage 3 | 17 | 14 | TKE avm: 5,714, `8.176154756539789e-2` |
| kt2 external | 11 | 11 | Hu_avg: 580, `1.2029895778109534e-4` |
| kt2 stage 1 | 17 | 14 | zFu: 17,400, `8.916107314898909e-1` |
| kt2 stage 2 | 17 | 17 | zFw: 17,993, `4.586852146673482e1` |
| kt2 stage 3 | 17 | 17 | zFw: 17,985, `2.36801992795954e1` |

Because the corrected chained W row did not complete, its old W cell is not
used for a new ownership claim.  The given-NEMO-entry W walk alone establishes
the first owned stage.

## Rule 12, trajectory, and magnitude

No candidate passed the local stage-output prerequisite.  Consequently the
held full-RHS patch was not applied and no candidate 954-row table or 30-day
arm exists.  A baseline recertification of the unchanged production tree was
attempted, but the environment likewise killed its large CPU compilation with
an empty log.  The recorded immutable before arm remains mechanically valid:

| Headline | before | after this round |
|---|---:|---:|
| kt2 U rms | `2.7377110452773967e-12` | unchanged; no candidate |
| kt2 V rms | `3.284922138989399e-12` | unchanged; no candidate |
| kt3 T rms | `1.627497246303733e-4` | unchanged; no candidate |
| kt3 S rms | `6.327735185607253e-6` | unchanged; no candidate |
| day-30 T rms | `1.2397011295506804e-2 K` | unchanged; no candidate |

No AT-BAR row can leave the bar and first-over-bar cannot move earlier because
no shared numerical code moved.  This round does not claim an improvement.

## Tanks and shared-statement risk

GYRE executes the measured transport-form statement at kt1 stage 1.  No
shared implementation changed, so LOCK_EXCHANGE and OVERFLOW have no candidate
to exercise.  DINO does use the shared QCO W helper; any eventual rounding
change is therefore a shared-statement risk and must run DINO's applicable
gate after the direct stage proof closes.  ORCA2 remains
UNMEASURED_WITH_SPEC: exercise its NEMO-identity QCO/transport arm if enabled,
otherwise prove from the resolved card and compiled branch that this statement
does not execute.  No stabilizer absent from NEMO was added.

## Review and focused validation

The separate read-only Codex review attempt is recorded at
`codex_review.log`.  Its verbatim result was: **"independent review unavailable
in-sandbox"**.  The CLI could not initialize its in-process app-server client
because the sandbox made its setup path read-only; per the operator's note this
does not block the round.

Focused validation completed:

- stage-gate and citation-gate unit tests: the stage suite passed; citation
  tests pass after commit-stamped execution (dirty-tree failures before the
  evidence commit were expected fail-closed behavior);
- `bash -n` passed for the acquisition script;
- the additive source patch applied to the canonical source card and
  `gfortran -fsyntax-only` passed at `/tmp/r98syntax.C3aBcB`;
- acquisition layout and admission plants are embedded in `run.sh` and must
  exit nonzero when the operator runs it.

## Compiled-source citations

The resolved vector branch skips the stage-level W call at stage 1 and says
the tracer path computes `zFw` instead
(`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:331-339`).
That executed tracer path calls transport-form `wzv`, then writes the direct W
and transport record after the call
(`GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90:267-295`).

The first non-bit statement is the transport divergence's divide by live T
thickness
(`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:132-139`);
the next statement remultiplies that result by the same live thickness
(`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:152-154`).
The QCO recurrence then consumes `ze3div`, `r1_Dt`, static thickness, and the
two stretch slots in bottom-up order
(`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/sshwzv.f90:293-300`).

## OPEN for round 99

1. Run the requested acquisition and admit
   `round98/oracle_stage1_w_walk/oracle_stage1_w_walk_kt00000001.bin`.
2. Extend the existing stage-W walk (not a second harness) to compare direct
   `hdiv`, `e3div`, `r3t(Kaa)`, `r3t(Kbb)`, static thickness, and final W.
3. Re-name the first non-bit statement against direct compiled intermediates.
   The discriminating question is whether the remaining 5,289-cell final-W
   difference is inherited from stretch operands or owned by the recurrence.
4. Only if the direct W row becomes BIT may the held round-97 full-RHS patch be
   combined, followed by the 954-row Rule-12 ladder, 30-day arm, and tanks.

No configuration or carried-state decision is requested.
