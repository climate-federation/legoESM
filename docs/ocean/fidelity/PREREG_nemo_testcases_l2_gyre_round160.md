# Preregistration — round 160, the second continuity solve

Committed before any round-160 measurement of the split producer was taken.
Entry for every developed-state row: NEMO's admitted day-180 daily restart
(step 1080, the entry to step 1081).  Every arm is one production step through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure (operator note L-amend).  Oracle: the admitted round-156
stage-2 record under `round157/oracle_developed_stage2`, read through the
reader rounds 158 and 159 already use, on the same windows.

## What round 159 established, and what it left open

Round 159 measured that the oracle solves continuity TWICE per stage on this
deck.  The stage program takes the vector-invariant arm at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356`, skips the
solve at stage 1 (`:358`), and for later stages solves it on the RAW stage
velocity at `:360` through the velocity indicator
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`.  The tracer
transport then re-solves it in the transport form at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`, entered because
`:195` sets the vertical-transport flag, and overwrites the same array before
forming the tracer's own vertical transport at `:280`.

legoESM builds ONE stage vertical velocity, in the transport form
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138`), and hands it
to both consumers.  Round 159 measured that selecting the velocity form on the
SHARED field removes 99.998% of the stage-2 vertical-velocity difference and
puts the stage-2 momentum right-hand side on round 158's oracle-`ww` ceiling,
but also moves the one-step tracer temperature by 1.203492e-07 K, which the
oracle's tracer path never sees.  So the faithful change is TWO vertical
velocities per stage, not one changed field.

## The candidate

A second continuity solve at stages 2 and 3 only.  The momentum vertical
advection reads a vertical velocity built through the velocity indicator from
the raw stage velocity; the tracer transport keeps today's transport-form
field, byte for byte.  Stage 1 gets NO momentum solve, because the compiled
program skips it there
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358`) and the stage-1 right-hand side is built in the
two-dimensional step.  The adaptive-implicit partition is a separate program
in the oracle as well —
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:362` partitions the
momentum pair under the velocity
indicator and `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:277`
partitions the tracer pair under the transport indicator — and that second
partition is NOT transcribed here, so selecting both must refuse loudly rather
than silently run one partition for two consumers.

Card census, read off each card's RESOLVED configuration, not off a deck
comment: the route needs the RK3-WS momentum integrator, the vector-invariant
momentum advection and the literal continuity solve together.  GYRE-zco has
all three.  LOCK_EXCHANGE-zco and OVERFLOW-zps take the flux-form momentum
advection and the generic continuity solve.  The generic NEMO-GYRE recipe takes
the generic continuity solve.  Both DINO cards take the Euler momentum
integrator, so the RK3 stage program does not run at all.  The gate's census
imports the model's OWN predicate rather than restating it (operator note AR,
finding 2).

## The remaining 9.74e-08, and the discriminating measurement

Round 159 registered that the corrected arm still sits at 9.74e-08 of the
field's own root mean square, four to ten times the floor that round declared,
so a second producer operand is still wrong.  The two candidates already in the
record:

* **the live thickness ratio.**  The velocity indicator rebuilds each face
  transport from `e3u_0*(1+r3u(Kmm))` at
  `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`.  The oracle
  carries `r3u(Kmm)` as an INTERPOLATED RATIO set one stage earlier, while
  legoESM rebuilds it from the interpolated sea surface height.  The admitted
  record carries the oracle's own `r3u(Kmm)` and `r3v(Kmm)`.
* **the stretching term's clock and level pair.**  The quasi-Eulerian term at
  `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298` is
  `r1_Dt * e3t_0 * (r3t(Kaa) - r3t(Kbb))`.  At stage 2 the oracle's clock is
  `rDt = rn_Dt/2` (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:221-222`)
  and its `r3t(Kaa)` is the HALF level
  (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:254`, the np_HYB
  arm this deck selects at `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:52`),
  while legoESM uses the full step with the full after level.  The two
  products are algebraically the same number, so they can only differ in
  rounding; they are inseparable, and an arm that moves one without the other
  is a factor-of-two error, not a control.

**The discriminating measurement, named in advance:** with the split already
in place, run two one-variable arms on the stage-2 producer and score the
resulting vertical velocity against the oracle's own recorded field.  Arm T
installs the oracle's OWN recorded `r3u(Kmm)`/`r3v(Kmm)` in the velocity-form
face transports.  Arm C uses the oracle's stage clock with the matching half
level.  Whichever removes more of the 2.334682e-13 m/s residual owns it.  A
third, free discriminator comes with the split itself: at stage 3 the oracle's
clock and after level are already legoESM's (`:265-266`, `:268`), so if the
clock and level pair owned the residual the stage-3 producer would be exact
where the stage-2 one is not.

## Predictions, each with its falsifier

1. **The tracer field does not move at the stage the split is made.**  With
   the split in place, the stage-2 tracer vertical velocity and the stage-2
   tracer temperature and salinity are BIT-IDENTICAL to the pre-split arm:
   0 of 18,000 cells.  REFUTED if any cell moves, which would mean the round
   changed two things rather than one.  The FINAL one-step tracer state is
   expected to move, through the corrected stage-2 velocity, and that move is
   expected to be SMALLER than round 159's 1.203492e-07 K shared-field move.
2. **The split reproduces round 159's corrected arm at stage 2.**  The stage-2
   vertical velocity difference against the oracle falls from 1.2326857e-08 to
   2.334682e-13 m/s, within 1% of round 159's velocity-form row.  REFUTED if
   it differs from 2.334682e-13 by more than 1%, which would mean the
   production split is not the arm round 159 measured.
3. **The stage-2 momentum right-hand side reaches round 158's ceiling.**
   1.121355e-14 (u) and 1.307575e-14 (v), within 1%.  REFUTED outside 5e-14.
4. **Stage 3 inherits the correction.**  The stage-2 OUTPUT velocity — the
   oracle's `uu(Kaa)` at stage 2, which is the entry the stage-3 transport
   reads — improves by at least 50% against the oracle's recorded row, whose
   production value is round 157's 1.30926020461275e-06 m/s.  REFUTED if it
   improves by less than 20%, which would mean the stage-2 right-hand side is
   not what carries the stage-3 transport difference round 155 attributed to
   it.
5. **Neither named candidate owns the remaining 9.74e-08.**  Arm T removes
   less than 10% of the 2.334682e-13 m/s residual and arm C removes less than
   10%.  REFUTED for a candidate if it removes 50% or more, in which case that
   candidate is NAMED the owner and round 161 lands it.  If both are refuted
   the residual is unowned and the receipt says so rather than closing it.
6. **The year verdict is a coin flip and is not predicted.**  The change is
   momentum-only at the stage it is made, so the Decision-43/45 rows are
   expected to move by less than 1e-05 relative — round 149's momentum landing
   moved day 240 by 2.2e-08 K on 1.6447e-02 K, against a harness run-to-run
   floor of about 2e-10 K (round 129).  The SIGN is NOT predicted.  The round
   LANDS only if day-30 temperature root-mean-square decreases against the
   before arm, the first-over-bar row is not earlier, no kt=1 at-bar row leaves
   the bar, every moved row is registered, and day 240 and day 360 do not
   worsen.  Otherwise it HOLDS with the build committed behind its receipt and
   the failing row named.
7. **Both plants fire.**  `wzv-split-shared` claims the split is on while the
   tracer consumer is wired to the momentum field, and the tracer-identity
   control must refuse it.  `wzv-split-inert` claims the split is on while both
   consumers keep the transport form, and the liveness control must refuse it.
   A control that does NOT catch its plant raises with its own marker and its
   own exit code, never the marker a caught plant prints.

## Before arm

The before arm is this landing's OWN base commit, `de4e0cfe35da`, measured
(operator note W, item 2): the certified ladder for kt=1..10, a 30-day member
with the day-by-day gap, and a 360-day member with the day-by-day gap, all
produced from a pristine clone of that commit before any round-160 edit
existed.  The lane's inherited year rows, for reference, are day 30
6.88819351379691829e-05 K, day 240 1.64467186440671112e-02 K and day 360
1.12234508615602115e-02 K.

## Landing rule

Nothing lands unless the single source-exact statement above is proven AND the
full Decision 43/45 gate runs on it: the month, day 240 and day 360 not worse,
the kt2 temperature and salinity rows at the bar, first-over-bar not earlier,
every moved row registered with the harness's run-to-run floor of about
2e-10 K quoted next to it, DINO measured if the statement is shared, the
generic NEMO-GYRE card and the LOCK_EXCHANGE and OVERFLOW tanks per operator
note AR, the six-file push gate green, and a Claude reviewer's verdict quoted
verbatim.  A candidate that requires a carried-state change or a configuration
choice is written up as a DECISION_NEEDED instead of landed.
