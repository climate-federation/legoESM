# Preregistration — NEMO testcase L2 GYRE round 120

Date: 2026-09-19

Incoming lane tip: `b0b8938af89a05c15f6814967c2ee311ac57d1b4`

This document is frozen before any Round-120 scientific measurement.  Evidence
will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round120/`.  The round first
implements Decision 45's year admission in the existing Decision-43 gate, then
performs the exact production discriminator handed off by Round 119.  No
production physics, configuration, NEMO source, carried state, restart schema,
or trajectory arm is changed by this preregistration.

## P1 — extend the existing landing gate to the registered year

The existing
`nemo_testcase_l2_gyre_decision43_gate.py` will be extended; no second landing
gate or second year scorer will be created.  Given a before member root, an
after member root, and the recorded NEMO year root, it will call the committed
`nemo_testcase_l2_gyre_year_owners.py` day-gap implementation on the fixed
Decision-45 days `30,60,90,120,180,240,300,360`.  Both legoESM members must be
unperturbed seed 0, tag `year`, 360 days, 2,160 steps, `dt=14,400 s`, fp64, and
snapshotted every six steps by the existing from-rest harness.  Their manifests
must be clean and match explicitly supplied before/candidate commits.  Missing
days, an unexpected manifest field, a dirty producer, a different cadence, or
an unregistered extra/missing row is a refusal.

The report will register all eight T3D wet-RMS rows before and after.  A landing
passes the new criterion only when both day 240 and day 360 do not worsen;
equality is allowed exactly as Decision 45 says.  This criterion is added to,
not substituted for, Decision 43's day-30 improvement, first-over-bar, kt1,
moved-row registry, and executing-card requirements.

Frozen baseline prediction: scoring the immutable before root
`phase3/year_equivalence/gyre` against `phase3/year_fromrest` reproduces the
already recorded T RMS values:

| day | frozen T3D RMS [K] |
|---:|---:|
| 30 | `6.890484901489568e-5` |
| 60 | `1.9329973681936875e-4` |
| 90 | `1.8645021144913585e-3` |
| 120 | `1.0501256819510476e-3` |
| 180 | `3.580551011866709e-3` |
| 240 | `1.6446741930292448e-2` |
| 300 | `1.3597404177319843e-2` |
| 360 | `1.1223573910167267e-2` |

A `year-day240-worse` plant will replace only the candidate day-240 value by
the next representable fp64 value above the before value.  It must make the
year criterion false, print `STATUS PLANT-FIRED`, and exit nonzero.  A plant
that stays green, a baseline value different from the table, or a gate that
does not emit every registered row refutes P1 and stops the production walk.

## P2 — reproduce the Round-119 production-only KEG word

Only after P1 and its tests pass, the existing Round-83 producer gate will be
extended in place.  It must first reproduce Round 119 under the full production
step JIT: source order and isolated order are BIT through HPG, LDF and VOR;
the first production-versus-isolated difference is exactly one active U cell
after KEG at `4.1359030627651384e-25 m s-2`; V is BIT; production eager is BIT
at every boundary; and ordinary-versus-source-order finals retain
6,882/17,400 U plus 6,566/17,100 V unequal cells with maximum
`8.470329472543003e-22 m s-2`.  Failure to reproduce any row invalidates the
new discriminator rather than superseding Round 119.

The compiled C2 program forms `zhke` and then updates the shared U/V `Krhs`
accumulator by subtraction at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:117-131`, with the
specific writes at `:129-130`.  The new artifact will register the sole native
index, its corresponding full-face index, the active mask value, the input
after-VOR word, the unmasked and WRITE-only face-masked KEG words, the
production and isolated output words, the oracle output word, hexadecimal fp64
bit patterns, and the signed ULP direction.  A count other than one, a dry
target, or an unregistered bit pattern refutes P2.

## P3 — one-variable materialization and association discriminator

Three private full-production-step arms will consume the same HPG, LDF, VOR,
KEG and ZAD operands.  None is selectable by a recipe.

1. `unmasked-materialized` places a production optimization boundary around
   the unmasked pre-negated KEG addend before adding it to `Krhs`.
2. `masked-materialized` changes only that operand to the face-masked WRITE
   value before the same boundary and addition.
3. `literal-subtract` evaluates the compiled statement's `Krhs - dKE` form
   rather than adding a pre-negated term.

Frozen prediction: the target is wet, its mask is exactly one, and unmasked
versus face-masked KEG operand bits are identical.  The
`unmasked-materialized` arm is predicted to close the one-cell
production-versus-isolated split, which would identify full-step XLA
materialization/fusion rather than mask arithmetic as its owner.  If only the
masked arm closes, mask materialization owns it.  If only literal subtraction
closes, the add-versus-subtract association owns it.  If none closes, ownership
remains withheld.  Every arm will be separately labelled production JIT; the
unchanged eager and isolated-JIT controls remain separately labelled.

A production KEG-ULP plant will perturb the active target addend by one ULP
inside the full step.  It must move the target after-KEG, after-ZAD and
after-ADV words while leaving HPG/LDF/VOR and the untargeted raw operands BIT,
print `STATUS PLANT-FIRED`, and exit nonzero.  An isolated-only control is not
accepted.

## P4 — magnitude and landing stop

The immutable year arm will also be scored at day 240 with the committed
day-gap instrument, establishing the magnitude against which future candidates
are ranked.  Round 119's current instantaneous ordering is retained only as a
candidate-ordering fact: W is first and largest among the ZAD operands
(18,000/18,000 cells, maximum `7.946658315637966e-7 m s-1`), and ZAD's
incremental U/V discrepancy is about `1.9e-9 m s-2`; the KEG discriminator is
one `4.1e-25` word.  Neither fact measures how much W carries at day 240.
Therefore the receipt will label W's day-240 ownership **UNMEASURED**, not rank
it ahead of a measured year effect.

No physics candidate is preregistered.  If P3 merely classifies the last-bit
word, no ladder, month, DINO, or candidate year run is spent and the round is
HELD; the OPEN work returns to W's current-tip producer with a mandatory
day-240 intervention measurement.  If an arm unexpectedly proves a shared
source-exact production statement of material magnitude, a separate committed
addendum must freeze the candidate's exact predictions before editing
production.  Only such an addendum may authorize the kt1..10 ladder, days
1--30, DINO measurement, and 360-day run.

GYRE and the generic NEMO-GYRE recipe execute the underlying KEG/ZAD path.
DINO shares the momentum statements and is AT RISK for any future landing.
LOCK_EXCHANGE and OVERFLOW do not execute this private diagnostic.  ORCA2
remains `UNMEASURED-WITH-SPEC`: record its native stage-entry KEG/ZAD operands
and compiled accumulator boundaries, run production JIT/eager/isolated arms
with propagating plants, and run its certified trajectory plus the registered
year rows before any shared landing.  No stabilizer, configuration choice,
freshwater-pair change, year-harness change, reconciliation-gate change, or
`#1484` guard change is permitted in this round.
