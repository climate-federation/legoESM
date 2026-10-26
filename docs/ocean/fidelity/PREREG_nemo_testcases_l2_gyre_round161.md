# Preregistration — round 161, the live thickness ratio and the year's owner

Committed before any round-161 measurement ran.  Entry for every
developed-state row: NEMO's admitted day-180 daily restart (step 1080, the
entry to step 1081).  Every model arm is one production step through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure (operator note L-amend).  Oracle: the admitted round-156
stage-2 record under `round157/oracle_developed_stage2`, read through the
reader rounds 158, 159 and 160 already share, on the same windows.

## What round 160 left, in one paragraph

Round 160 built NEMO's second per-stage continuity solve and held it: the
developed stage-2 momentum vertical velocity fell from
`1.2326857042024439e-08` to `2.334682468902387e-13` m/s, the certified
step-2 velocity rows went to the bar, the month improved, and day 240
(`1.6446718648e-02` -> `1.6448360701e-02` K) and day 360
(`1.1223450850e-02` -> `1.1225660019e-02` K) both worsened, so Decision 45
refused it.  It also separated the remaining residual: the stage clock and
after-level pair is MEASURABLY INERT (it removes `-5.70e-11` of the residual),
while the live face free-surface ratio differs from the oracle's own recorded
one on 580 of 580 u faces and 570 of 570 v faces, at `1.4516405645036015e-13`
and `1.3880628732222712e-13` root mean square, i.e. `5.724055868505435e-09`
and `5.536987604354082e-09` of the operand's own size.  Round 160 measured
that the OPERAND differs; it did not measure that the operand OWNS the
residual.  This round measures that, and asks the year question.

## Order A — does the live face ratio own the residual, and where is it born

### A1, the substitution

With everything else held, the stage-2 velocity-indicator continuity solve
takes the ORACLE'S OWN recorded `r3u(Kmm)`/`r3v(Kmm)` in place of legoESM's,
through NEMO's own thickness statement
`e3u = e3u_0*(1 + r3u*umask)`.  Two arms get it, because two arms take the
velocity indicator:

* the CORRECTED arm — round 160's second continuity solve, where the momentum
  vertical advection reads its own solve
  (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`, velocity
  indicator at
  `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`);
* the SHARED arm — round 159's one-variable arm, where the single field is
  built in the velocity form and both consumers read it.

The substitution is made by an observer that wraps the producer for exactly
one call, the same technique round 152's tracer and transport observers
already use in this instrument; no production line changes, so no card can
reach it.

### A2, the producer walk, offline

NEMO does NOT rebuild the stage face ratio from the stage sea surface height.
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:185` calls
`dom_qco_r3c_RK3` ONCE per step on the after sea surface height, and each
stage then COMBINES two ratios:
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:266-267` is the ratio
statement itself, `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211`
is the stage-1 combination that SETS the ratio this walk scores (the level the
stage-2 solve reads), and
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:255` is the stage-2
one.  legoESM builds ONE ratio from the already-combined sea surface height,
at `vertical.py:247` fed by `vertical.py:238-239`, and multiplies it into the
face thickness at `vertical.py:251`.  The two are the same number
algebraically, so a difference of `1.45e-13` — forty million units in the last
place of a ratio whose own size is `2.5e-05` — cannot be the rounding of that
composition and must be an OPERAND.

So the walk decomposes the operand difference with NO model step at all, from
the record alone:

* **the statement row**: legoESM's ratio form evaluated on the ORACLE'S OWN
  recorded `ssh(Kmm)` and legoESM's own mesh operands, scored against the
  oracle's recorded `r3u(Kmm)`/`r3v(Kmm)`;
* **the operand row**: legoESM's LIVE stage-2 ratio (round 160's exposure)
  scored against that same offline reconstruction.

Whichever row carries the `1.45e-13` is the first non-bit statement's home:
the statement row means the composition is the owner, the operand row means
the stage sea surface height upstream of it is.

## Order B — why exactness at stage 2 makes the year worse

Not a fix.  The corrected velocity reaches the tracers only through the
stage-3 transport, and the year got worse, so the measurement names WHICH
tracer process the corrected velocity feeds.  Round 152's developed one-step
magnitude ranking is the reference and the instrument: each row is
legoESM-minus-NEMO for one isolated one-step temperature contribution at the
same day-180 entry, ranked by root mean square.  Its published production
table is

| rank | contribution | RMS (K) |
|---:|---|---:|
| 1 | vertical diffusion | `2.1834089000437955e-5` |
| 2 | lateral diffusion | `1.0655217366755294e-5` |
| 3 | shortwave | `2.513690760498398e-7` |
| 4 | advection | `1.0974591404090626e-8` |
| 5 | geometry | `6.281725340556424e-12` |
| 6 | surface boundary | `4.6577410196032515e-15` |

This round runs that ranking TWICE at the same entry, once in production and
once with the second continuity solve on, and reports every row before and
after with the sign of the change.  Round 134's family decomposition (the
tracer pair carries 99.7% of day 240) is what makes a grown tracer row the
answer to the year question rather than a curiosity.

## Predictions, each with its falsifier

1. **The live face ratio does NOT own the stage-2 residual.**  Installing the
   oracle's own `r3u(Kmm)`/`r3v(Kmm)` removes less than 10% of the
   `2.334682468902387e-13` m/s residual in the corrected arm.  The reason is
   arithmetic, READ OFF THE CODE and not measured: the ratio enters only
   through `e3u = e3u_0*(1+r3u)`, so a `1.45e-13` absolute ratio difference is
   a `1.45e-13` RELATIVE difference in each face transport, while the residual
   is `9.74e-08` of the vertical velocity's own size — six orders of
   amplification would be needed.  REFUTED if it removes 50% or more, in which
   case the ratio IS the owner and round 162 lands its producer.
2. **The two arms agree.**  The shared velocity-form arm (round 159's) and the
   corrected arm both sit at `2.334682468902387e-13` m/s before the
   substitution, and the substitution removes the same fraction in both to
   within a factor of two.  REFUTED if the fractions differ by more than a
   factor of two, which would mean the substitution is reaching something
   other than the one producer.
3. **The ratio difference is an OPERAND, not the composition.**  legoESM's
   ratio form evaluated on the oracle's own recorded `ssh(Kmm)` reproduces the
   oracle's recorded `r3u(Kmm)`/`r3v(Kmm)` to within 4 units in the last place
   of the ratio (about `1.4e-20`), i.e. the statement row is at the
   compiled-rounding floor and the live-versus-offline row carries essentially
   all of the `1.45e-13`.  REFUTED if the statement row carries more than 10%
   of the `1.45e-13`, in which case the COMPOSITION is the first non-bit
   statement and its citation is `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211` against
   `vertical.py:247`.
4. **The observer is passive and live.**  With the observer installed and the
   substitution disabled, every state leaf is BIT-IDENTICAL to the unobserved
   arm — zero bytes, not a bound.  With it enabled, the momentum vertical
   velocity moves on at least one cell.  REFUTED either way if a byte moves
   with the substitution off, or if nothing moves with it on, and then the arm
   proves nothing and the round says so.
5. **The substitution does not touch the tracer path.**  In the corrected arm
   the tracer's own vertical velocity and the stage-2 tracer temperature and
   salinity are bit-identical with and without the substitution, 0 of 18,000.
   REFUTED if any cell moves, which would mean the observer reached two calls.
6. **The corrected arm's grown process row is ADVECTION.**  In Order B's
   ranking the advection row grows and the vertical-diffusion row stays the
   largest, because the corrected stage-2 velocity reaches the tracers only
   through the stage-3 transport.  REFUTED if advection does not grow, or if
   any row other than advection grows by more than advection does — and the
   refutation is the finding, because it names a different owner.
7. **The compiled-scheduling floor is NOT a general floor.**  Round 160's
   passivity control was loosened from a zero-byte requirement to two units in
   the last place at row scale, and its reviewer asked for that floor to be
   reproduced on an arm this round did not write.  Two unrelated exposure arms
   — the stage-1 output velocity and the stage-3 tracer exposure — are
   measured at the same entry.  Prediction: both move ZERO bytes, so round
   160's single one-unit row is specific to its own exposure and the zero-byte
   control is restored as the default.  REFUTED if either moves, in which case
   the floor is real and the bound stays.
8. **Both plants fire.**  `face-r3-inert` hands the substitution legoESM's OWN
   live ratio while claiming it is the oracle's; the liveness control must
   refuse it.  `face-r3-tracer` applies the substitution to the tracer's own
   continuity solve as well; the tracer-identity control must refuse it.  A
   control that does NOT catch its plant raises with its own marker and its
   own exit code, never the marker a caught plant prints.

## Landing rule

This is a MEASUREMENT round for Order B by the operator's own order, and
Order A lands only if it names a single source-exact statement AND the FULL
Decision 43/45 gate runs on it: day-30 temperature root-mean-square decreases,
the first-over-bar row is not earlier, no kt=1 at-bar row leaves the bar,
every moved row registered with the harness's run-to-run floor of about
`2e-10` K quoted next to it, DINO measured if the statement is shared, the
generic NEMO-GYRE card and the LOCK_EXCHANGE and OVERFLOW tanks per operator
note AR, day 240 `1.6446718648e-02` K and day 360 `1.1223450850e-02` K not
worse, the push gate green, and a Claude reviewer's verdict quoted verbatim.
Anything that fails a row is HELD behind its receipt with production restored
and proven, 70 rows and 0 moved against the base commit.  A candidate that
needs a carried-state change or a configuration choice is written up as a
DECISION_NEEDED instead of landed.

## Before arm

The tip this round starts from, `1339f01de8126169069f2a87cff9a356402c5978`,
whose certified trajectory round 160 proved bit-identical to its own base
commit on all 70 rows.  The lane's inherited year rows are day 30
`6.88819351379691829e-05` K, day 240 `1.64467186440671112e-02` K and day 360
`1.12234508615602115e-02` K.
