# NEMO testcase L2 GYRE Round 157 receipt — developed stage-2 right-hand-side walk

Date: 2026-09-23
Status: **HELD** — no physics, no configuration, no carried state changed.

The round-156 record is ADMITTED: the operator's run was sound and only the
run script's own environment was wrong.  With it, the developed stage-2
velocity difference is OWNED by the stage's own right-hand side, not
inherited from its entry velocity, and inside that right-hand side the owner
is the momentum ADVECTION.  The pressure gradient and the vorticity sit at
the compiled-rounding floor; the advection is wrong at four parts in ten
thousand of its own size and carries the whole 1.3093e-6 m/s velocity
difference through the stage assignment.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round157.md`, committed as
`75d77a77b` before any legoESM stage-2 measurement ran.  The authoritative
measurement is `developed_stage2_walk.json` under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round157/walk/`, produced on
a clean tree at `234b27345`.  Nothing in the model changed this round; every
edit is in the walk, its reader, its plants and its tests.

## The admission, and why it refused

The operator ran the round-156 acquisition.  NEMO built, ran to `STOP 0` and
wrote its 19-group record, and then the script's own admission loop exited 72
with `REFUSE: round156 stage3-alignment plant lacks its marker`.

The record is not at fault and neither is the check.  The run script never put
this repository's packages on the interpreter path, so `legoesm` resolved to
the editable install, which points at a DIFFERENT working tree.  Four plants
passed because they never leave the gate's own module; the fifth loads the
round-154 reader by file path, that reader imports a module which exists only
on this lane, and the plant therefore died on `ModuleNotFoundError` — nonzero,
so "stayed green" never fired, and silent, so the marker never printed.  The
ordinary admission would have died at the same line.

Run here, with the path set and NOTHING else changed:

* `STATUS PASS: round-156 stage-2 record admitted (4744400 bytes, 19 groups)`;
* the stage-3-alignment plant prints
  `STATUS PLANT-FIRED: stage3-alignment: stage-2 after-velocity is not
  byte-identical to the Round-154 stage-3 Kmm velocity` and exits 1.

So the record's stage-2 after-velocity IS byte-identical to the round-154
record's stage-3 `uu(Kmm)`, its restarts are byte-identical to the
un-instrumented round-132 daily reference, and its in-run calibration closes.
No rebuild and no re-acquisition.  The script now exports the lane's package
path and creates its evidence directory before asking the filesystem how much
room is free there, which is what stopped an earlier attempt.  Neither edit
changes what any check verifies.

## A correction to the round-156 receipt

Round 156 named the stage assignment "thickness weighted" and its gate
rebuilds that arm.  This deck does not take it.  The record's own flag group
carries `ln_dynadv_vec = 1`, `lk_linssh = 0`, and the compiled selector at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:721` then takes the
VECTOR arm at `:723`:
`uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask`.
No free-surface ratio enters the stage-2 velocity at all, so the bounded
`r3u(Kaa)` term round 156 registered as unvaried is not a term of this
statement either.  The round-156 gate reports its thickness-weighted rebuild
and does not gate it, so nothing was certified on the wrong arm; the walk's
reader now REFUSES a record whose flags select the other arm.

## The compiled program, in execution order

All from the record's own build,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90`.  Stage 2 binds
`Kbb = N`, `Kmm = N+1/3`, `Krhs = Kaa = N+1/2`.

| step | compiled line | recorded as |
|---|---|---|
| right-hand-side slot at entry | `:462` | `rhs_entry` |
| `dyn_hpg`, which OVERWRITES the slot | `:497` | `after_hpg` at `:499` |
| `dyn_vor`, accumulates | `:510` | `after_vor` at `:512` |
| `dyn_adv`, vector-invariant arm selected at `:523`, accumulates | `:525` | `after_adv` at `:531` |
| vector-arm assignment, selector `:721` | `:723` | `uu_vv_Kaa_raw` at `:738` |
| barotropic correction | `:787`, applied at `:810` | `uu_vv_Kaa_final` |

`Krhs` aliases `Kaa`, so the entry slot still holds a velocity and is an
operand of nothing; that is why the pressure gradient's snapshot is the
pressure-gradient term alone.

## Controls, before any attribution

* **Grid.** NEMO's recorded before-level velocity is the step-entry level,
  which this walk loads from the same restart.  It is BIT on both components,
  0 of 21,780 and 0 of 22,080 cells, which is what proves the record's window
  and transpose land on legoESM's face grid.  The northward component takes a
  DIFFERENT window from the eastward one; the first version used one window
  for both and stopped on a shape mismatch.
* **Passivity.** Each exposure hook substitutes its own slots only after the
  ordinary step completes; every other prognostic field is byte-identical to
  an ordinary step, 0 bytes on all eight runs.
* **Cross-check.** The corrected stage-2 velocity row reproduces round 155's
  number exactly: 1.309260e-6 m/s maximum over 17,400 active eastward faces.

## The walk

One production step from NEMO's admitted day-180 entry through
`LatLonCGridOceanModel.step` under production just-in-time compilation, one
arm per exposed frame.  Scores are over NEMO's own active faces.

| row | active unequal | active max abs | active rms |
|---|---:|---:|---:|
| stage-2 entry velocity, u | 17,400 / 17,400 | 6.134203e-10 | 1.901165e-10 |
| stage-2 entry velocity, v | 17,100 / 17,100 | 5.800129e-10 | 1.674494e-10 |
| after pressure gradient, u | 17,398 / 17,400 | 7.600132e-17 | 1.292406e-17 |
| after pressure gradient, v | 17,100 / 17,100 | 8.644649e-17 | 1.578966e-17 |
| after vorticity, u | 17,400 / 17,400 | 2.867262e-14 | 1.120967e-14 |
| after vorticity, v | 17,100 / 17,100 | 3.728115e-14 | 1.307121e-14 |
| after advection, u | 17,400 / 17,400 | 1.828743e-10 | 3.844166e-12 |
| after advection, v | 17,100 / 17,100 | 3.026376e-10 | 6.428546e-12 |
| production right-hand-side total, u | 17,400 / 17,400 | 1.828743e-10 | 3.844166e-12 |
| production right-hand-side total, v | 17,100 / 17,100 | 3.026376e-10 | 6.428546e-12 |
| raw stage-2 velocity, u | 17,400 / 17,400 | 1.316695e-06 | 2.767800e-08 |
| raw stage-2 velocity, v | 17,100 / 17,100 | 2.178990e-06 | 4.628553e-08 |
| corrected stage-2 velocity, u | 17,400 / 17,400 | 1.309260e-06 | 2.741052e-08 |
| corrected stage-2 velocity, v | 17,100 / 17,100 | 2.166216e-06 | 4.581503e-08 |

The two cumulative rows carry ONE instrument-side addition each, because
legoESM publishes its three operator components separately while NEMO
accumulates into one slot.  They are not load-bearing: the production
right-hand-side total, which carries no instrument addition at all,
reproduces the after-advection row to every digit printed.

Every row is non-bit somewhere, because the compiled-rounding floor touches
all of them (notes L and AL), so the row ORDER names nothing.  The magnitude
does.

## Ranking by magnitude, which is the answer

Each family's difference next to the size of the term it sits in.  NEMO's own
increments are differences of its recorded cumulative snapshots and carry one
rounding on the oracle side; they are context for the ranking, never a scored
row.

| family | NEMO's term rms | cumulative difference rms | relative |
|---|---:|---:|---:|
| advection, v | 5.538409e-09 | 6.428546e-12 | 1.160721e-03 |
| advection, u | 8.709119e-09 | 3.844166e-12 | 4.413956e-04 |
| vorticity, v | 1.076450e-06 | 1.307121e-14 | 1.214289e-08 |
| vorticity, u | 1.247684e-06 | 1.120967e-14 | 8.984379e-09 |
| pressure gradient, v | 1.555936e-06 | 1.578966e-17 | 1.014802e-11 |
| pressure gradient, u | 1.384764e-06 | 1.292406e-17 | 9.333043e-12 |

Five orders of magnitude separate the advection from the vorticity and eight
from the pressure gradient.  The pressure gradient's `1e-11` and the
vorticity's `1e-8` are the floor this campaign has measured many times; the
advection's `4e-4` is not a floor, it is a transcription difference.

## Owned, not inherited

The stage is handed an entry velocity that already differs by 6.13e-10 m/s,
so the rows above cannot say by themselves whether stage 2 makes the
magnitude or passes it on.  One directed substitution settles it: replace the
stage-2 entry velocity with NEMO's own recorded one and leave every other
entry field at legoESM's value.

| arm | after-advection rms, u | removed | after-advection rms, v | removed |
|---|---:|---:|---:|---:|
| arm baseline | 3.844166e-12 | — | 6.428546e-12 | — |
| NEMO's entry velocity installed | 3.844036e-12 | +0.0034% | 6.429669e-12 | −0.017% |

The substitution mechanism is not byte-neutral and that is registered rather
than excused: the production step averages the step's live thicknesses to
build the stage thickness while the override rebuilds it from the stage free
surface, which agree as algebra and not as arithmetic.  Its measured offset
from the production row is 1.694066e-21 (u) and 1.270549e-21 (v), eleven
orders below the difference being split, and BOTH arms carry it, so the
comparison stays one variable.

NEMO's own entry velocity removes nothing.  The stage-2 right-hand-side
difference is OWNED by the stage's operators.

## Budget

The vector arm makes the raw stage-2 velocity difference exactly `rDt` times
the right-hand-side difference.  With `rDt = 7200` s:

| component | predicted from the right-hand side | observed | relative disagreement |
|---|---:|---:|---:|
| u | 1.3166950919207677e-06 | 1.3166950919157872e-06 | 3.78e-12 |
| v | 2.1789903999098076e-06 | 2.1789903999064330e-06 | 1.55e-12 |

So 100% of the developed stage-2 velocity difference is the right-hand-side
difference, and the assignment adds nothing.  Round 155 measured that this
velocity is 95.2% by rms and 99.3% by maximum of the stage-3 transport
difference, and round 156 that the external barotropic half carries none of
it.  The chain closes on `dyn_adv` at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525`.

## Predictions and verdict

1. The stage-2 entry velocity is non-bit with an active maximum of at least
   1e-7 m/s, so the difference is substantially inherited: **REFUTED**.  It is
   non-bit, but only at 6.134203e-10 m/s, and installing NEMO's own value
   removes none of the right-hand-side difference.  The correct statement is
   the opposite of the prediction: stage 2 OWNS its difference.
2. The first non-bit family in compiled order is the pressure gradient:
   **CONFIRMED and USELESS**.  It is non-bit, and so is everything else, at
   1e-17.  Registered as a defect in the prediction, not a finding: on a
   developed state the compiled-rounding floor makes "first non-bit" name the
   first row rather than an owner, which is why the ranking above exists.
3. The raw stage-2 velocity difference is `rDt` times the right-hand-side
   difference to better than one percent: **CONFIRMED**, to 4e-12.
4. legoESM's transcription of the vector assignment rebuilds NEMO's raw
   stage-2 velocity bit for bit from NEMO's OWN operands: **REFUTED**.  It
   leaves 4,169 of 17,400 eastward and 2,368 of 17,100 northward active faces
   unequal, at a maximum of 2.7755575615628914e-17 — exactly one unit in the
   last place at the 0.1 m/s scale, the known compiled-rounding floor of a
   fused multiply-add under just-in-time compilation.  It is eleven orders
   below the 1.3e-6 m/s being attributed, so the assignment is exonerated by
   magnitude and NOT by exactness, and this receipt says so rather than
   calling it BIT.
5. The plants fire: **CONFIRMED**, both, see below.

## First non-bit statement that carries magnitude

`CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)` at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525`, entered
through the vector-invariant arm selected at `:523`, recorded at `:531`.

What it carries at day 180: the whole 1.828743e-10 m/s^2 right-hand-side
difference, hence the whole 1.316695e-06 m/s raw stage-2 velocity difference
and the 1.309260e-06 m/s corrected one, which is 95.2% by rms of the stage-3
transport difference round 155 measured.

What it carries at day 240 is NOT measured, because there is no candidate to
score, and Decision 45 ranks at day 240.  The honest bound from what IS
measured: round 134 found velocity resets remove 6.5% of the day-240 gap, and
round 149's momentum landing moved day 240 by 2.2e-8 K on 1.6447e-2 K, which
is within 1e-6 relative and near the harness's own run-to-run floor of about
2e-10 K (round 129).  A round-158 landing here should be expected to move the
month and the stage rows, and to leave the year essentially where it is.
That is a bound read off earlier measurements, labelled **PLAUSIBLE**, not a
measurement of this term.

No candidate is named inside `dyn_adv` this round.  Under the vector-invariant
arm that routine is the kinetic-energy gradient plus the vertical advection,
and legoESM publishes them as one bucket, so separating them needs a frame the
model does not currently expose.  The record already carries the operands both
need — NEMO's `ww` and its `uu(Kmm)` at this stage — so round 158 can split
them without a new acquisition.

## Landing

Nothing landed.  No single source-exact statement was proven, so the full
Decision 43/45 gate — month, day 240, day 360, the kt2 T/S bar, first-over-bar,
every moved row, DINO, the generic NEMO-GYRE card, the LOCK_EXCHANGE and
OVERFLOW tanks — has no candidate arm to run on and was not run.  The campaign
headline rows are inherited and unmodified: kt2 T/S at the bar, kt2 U/V about
2.7377e-12 / 3.2849e-12, kt3 T about 8.60e-7 K, day-30 6.888194e-5 K, day-240
1.644671864e-2 K, day-360 1.122345086e-2 K.  ORCA2 is
**UNMEASURED-WITH-SPEC**: repeat this stage-2 right-hand-side walk on the
ocean-only ORCA2 card before transferring the verdict.

## Plants, review and tests

Two plants, each printing its own marker and exiting nonzero:

* `entry-kbb-ulp` moves NEMO's recorded before-level velocity by one unit in
  the last place; the grid control must refuse it, and it prints
  `STATUS PLANT-FIRED: entry-kbb-ulp` with one unequal active face at
  6.938894e-18 and exits 1.
* `hpg-rank-scale` scales the recorded pressure-gradient snapshot by one part
  in a million.  The eastward pressure-gradient row then rises from last place
  to third, above BOTH vorticity rows, at a measured relative difference of
  9.999966e-07 against the one part in a million that was planted; the
  ranking becomes advection v, advection u, pressure gradient u, vorticity v,
  vorticity u, pressure gradient v.  It prints
  `STATUS PLANT-FIRED: hpg-rank-scale` and exits 1.

  Registered rather than quietly fixed: the first version of this plant
  demanded the TOP of the ranking and fired because its own assertion was
  wrong, not because the check caught the planted defect.  A scaling of one
  part in a million lands two orders above the floor rows and four below the
  advection, which is where the corrected control puts it.

Two new tests, each shown to FAIL when the thing it checks is removed: the
face scorer's mask awareness (removing the mask makes the dry-face arm fail)
and the reader's refusal of a record whose flags select the thickness-weighted
assignment (removing the guard makes the refusal arm fail).

The six-file push gate and this round's harness tests were run together and
reported exactly:

> 1 failed, 170 passed in 939.18s (0:15:39)

The one failure is this walk's own working copy, not a defect: the receipt was
still uncommitted, and `test_forcing_gate_plants_all_exit_non_zero` spawns a
subprocess whose provenance stamp refuses a dirty tree.  Re-run on the
committed tree it reports `1 passed in 25.66s`.

The citation gate on this receipt reported `PASS` with 13 citations, zero
failures, zero unmapped citations and zero map entries failing audit; its
shifted-line plant on the advection citation exited 1 with
`SYMBOL-NOT-AT-LINE`.

Independent review not run (codex paused).

## OPEN — Round 158

1. Split `dyn_adv` at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525` into its
   two compiled halves under the vector-invariant arm — the kinetic-energy
   gradient and the vertical advection — and rank them the same way.  legoESM
   publishes them as one bucket today, so this needs one more WRITE-only frame
   in the model's stage program; gate it on the same predicate its consumer
   uses (operator note AR, finding 1) and do not widen any other hook.
2. Calibrate before naming: the record carries NEMO's own `uu(Kmm)` and its
   `ww` at this stage, so legoESM's transcription of each half can be driven
   from NEMO's OWN operands and scored against NEMO's own increment before
   any statement is proposed.  The stage-entry substitution measured here
   already shows the entry velocity is not the operand that matters, so the
   suspect operands are the ones the advection reads for itself.
3. Rank at day 240, not at day 180, and quote the run-to-run floor next to any
   improvement.  If the split names a statement whose day-240 carry is within
   the floor, say so and put the question to the user rather than landing it
   as an improvement.
4. The compiled-rounding floor is not fixable by transcription (notes L, AL).
   The pressure gradient at 1e-11 relative and the vorticity at 1e-8 relative
   are that floor; do not walk them.
5. The round-156 gate's thickness-weighted rebuild is reported for an arm this
   deck does not take.  It is harmless because it is not gated, but a future
   round that needs it should rebuild the vector arm instead.
