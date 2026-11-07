# Preregistration — round 159, the developed stage-2 vertical-velocity producer

Committed before any round-159 measurement of legoESM's stage-2 vertical
velocity was taken.  Entry: NEMO's admitted day-180 daily restart (step 1080,
the entry to step 1081).  Every arm is one production step through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure (operator note L-amend).  Oracle: the admitted round-156
stage-2 record under `round157/oracle_developed_stage2`, read through the
reader round 158 already uses, on the same windows.

## What round 158 left open

Round 158 measured that the developed stage-2 momentum difference is owned by
the VERTICAL VELOCITY handed to the vertical advection: installing NEMO's own
recorded stage-2 vertical velocity at that one call removes 99.708% (u) and
99.797% (v) of the stage right-hand-side difference and leaves the vorticity
floor.  The advection arithmetic is exonerated.  Round 159 walks the PRODUCER
of that vertical velocity.

## The compiled program, read before predicting

The record's own namelist sets `ln_dynadv_vec = .true.`, so the stage program
takes the vector-invariant arm and calls the continuity solve at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360` as
`wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )`,
i.e. on the RAW stage velocity.  The flux-form call on the already-built
transports, `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:367`,
sits in the `ELSE` of that selector and does not execute on this deck.  The
routine itself is
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:236-311`: its divergence
comes from `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:123-130`
under the velocity indicator, which rebuilds each face transport as
`e2u * e3u_live * pu` from the velocity it was handed, and the bottom-up
integration adds the quasi-Eulerian stretching term at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:296-297`.

The transport indicator branch is
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:132-138`; it differences
the already-built `zFu/zFv`, which carry the barotropic velocity correction
`zub` added at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:310-313`.

NEMO runs the continuity solve TWICE per stage on this deck.  The second one is
inside the tracer transport assembly,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`, entered because
`ln_dynadv_vec` sets the vertical-transport flag at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:195`, and it overwrites
the same shared array with the TRANSPORT form before the tracer's vertical
transport is formed at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:279-281`.

legoESM builds ONE stage vertical velocity and hands it to both consumers.

## Predictions, each with its falsifier

1. **The producer's output is not at the floor.**  legoESM's production stage-2
   vertical velocity differs from NEMO's recorded one on more than half of
   NEMO's wet interfaces, at more than 1e-6 of the field's own root mean
   square.  REFUTED if the relative difference is below 1e-8, which would put
   the producer at the compiled-rounding floor and send round 158's
   attribution back for re-examination.
2. **The owner is the CALL FORM, not the arithmetic and not the inputs.**
   Driving legoESM's own producer through its velocity-indicator arm — the raw
   stage velocity, no barotropic correction, every other operand and the whole
   quasi-Eulerian term left at legoESM's value — removes at least 90% of the
   stage-2 vertical-velocity difference.  REFUTED if it removes less than 50%.
3. **The difference is not inherited from stage 1.**  Substituting NEMO's own
   recorded stage-2 entry velocity into legoESM's producer, with the
   production call form, removes less than 20% of the vertical-velocity
   difference.  REFUTED if it removes 50% or more, in which case the owner is
   upstream in stage 1 and round 160 walks that instead.
4. **The round-158 ceiling is reached.**  With prediction 2's arm in place, the
   stage-2 right-hand-side difference falls to within a factor of two of the
   1.121563e-14 (u) / 1.308047e-14 (v) residue that installing NEMO's own
   recorded vertical velocity left.  REFUTED if it removes less than half of
   the 3.844166e-12 / 6.428546e-12 production rows.
5. **The tracer premise is false for NEMO.**  Because the tracer transport
   re-solves continuity in the transport form, correcting the MOMENTUM
   vertical velocity must not change the tracer terms; but legoESM shares one
   vertical velocity between the two consumers, so applying prediction 2's arm
   to the shared field WILL move the day-180 one-step tracer state.  Predicted:
   a non-zero one-step temperature move, which is the measured reason a landing
   must produce two vertical velocities rather than change the shared one.
   REFUTED if the one-step temperature move is exactly zero, which would mean
   the tracer path does not read this field at all.
6. **Both plants fire.**  A plant that moves the arm's own operand by one part
   in a million million must move the scored vertical velocity, and a plant
   that scales the scored field must be caught by the window control.

## Landing rule

Nothing lands unless a single source-exact statement is proven AND the full
Decision 43/45 gate runs on it: the month, day 240 and day 360 not worse, the
kt2 temperature and salinity rows at the bar, first-over-bar not earlier, every
moved row registered with the harness's run-to-run floor of about 2e-10 K
quoted next to it, DINO measured if the statement is shared, the generic
NEMO-GYRE card and the LOCK_EXCHANGE and OVERFLOW tanks per operator note AR,
the six-file push gate green, and a Claude reviewer's verdict quoted verbatim.
A candidate that requires a carried-state change or a configuration choice is
written up as a DECISION_NEEDED instead of landed.
