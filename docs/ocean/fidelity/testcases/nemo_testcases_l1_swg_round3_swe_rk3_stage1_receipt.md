# NEMO SWG level-1 round 3 — SWE RK3 stage-1 production replay

Date: 2026-10-11. Base: `fbece7bc7c7f546aefef462dcc9634a99465d66b`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round3`.

## Preregistration

This section was committed before reading a numerical value from the admitted
stage-entry payload or running a legoESM stage measurement.  The complete
three-stage program is too large to implement and validate as one controlled
round, so this round lands the stage-1 program first, as the operator's round-3
instruction explicitly permits.  Stages 2 and 3, including all stage-3-only
forcing, remain fail-closed and unmeasured.

The admitted compiled program zeroes the momentum RHS, calls vector advection
and total ENS vorticity, adds the reduced-gravity surface-pressure gradient,
advances SSH with `rn_Dt/3`, rebuilds all QCO ratios, advances velocity from the
before level, applies U/V masks and lateral boundary conditions, then rotates
the stage indices at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:164-210`.  Vector advection dispatches
to C2 kinetic energy followed by vertical advection at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-139`; the one executed layer makes
the vertical-advection flux divergence structurally zero.  Continuity is the
live-thickness transport divergence and masked SSH update at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:135-149`, followed by the T/U/V/F QCO
ratios at `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:207-234`.

Frozen predictions and falsifiers:

1. The SWG card will select a guarded `swe_rk3` outer program while ordinary
   `step()` remains fail-closed until all three stages exist.  The production
   stage entry point will accept stage 1 only and execute under JIT in fp64;
   an absent/unknown selector, stage 2 or 3 silently running, a non-fp64
   output, or eager-only replay **REFUTES** the production boundary.
2. A nonzero manufactured stage will match independent source-literal C2 KE,
   ENS, pressure, continuity, QCO-ratio, and masked velocity statements.  The
   direct ENS division must remain direct and the QCO T ratio must not execute
   legoESM's generic positive-thickness floor.  Replacing C2 by the centered
   square-of-mean form, changing the source order, enabling a thickness floor,
   or dropping any wet/staggered mask **REFUTES** the transcription.
3. Driving the jitted stage from the admitted `kt=1, stage=1` entry will score
   its 13-field output bundle against NEMO's `kt=1, stage=2` entry.  Every
   computed or pass-through row is predicted bit-identical and at the
   `1e-15` relative bar.  Any unequal wet value is **DEBT**, never rounded up
   to AT-BAR, and **REFUTES** this prediction.
4. The zero and one controls each perturb one wet U stage-entry value by
   exactly one `nextafter` step *before* the jitted production stage.  The
   resulting production output must differ from its corresponding unperturbed
   production output and the ladder must exit nonzero.  A post-computation
   perturbation, a control that never enters JIT, or either control remaining
   bit-identical **REFUTES** the instrument.
5. `g=0.02` will be read from the card's constants by the only stage-1 gravity
   consumer, the inline pressure-gradient statements at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:170-176`.  A literal gravity in
   the stage program, a different resolved value, or another stage-1 gravity
   consumer that does not receive the card value **REFUTES** the constants
   path.
6. No non-SWG card selects `swe_rk3`.  TSUNAMI, VORTEX, VORTEX_VEC, GYRE,
   LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 must retain their selectors and
   certified hashes/test disposition.  Any existing AT-BAR row leaving the
   bar or a first-over-bar boundary moving earlier **REFUTES** Rule 12
   eligibility.

No configuration or carried-state choice is introduced.  `key_RK3`, vector
advection, C2 kinetic energy, total ENS with `ln_dynvor_msk=.true.`, QCO/VCO,
one layer, `rn_Dt=3600 s`, and reduced gravity `0.02 m s-2` are already stated
by the shipped deck and the round-1 card.  The stage-entry value object is a
replay boundary, not a new prognostic carry.

## Measurement

Pending preregistration commit.

## First non-bit statement

Pending measurement.

## Rule 12 disposition

Pending measurement.

## Gates

Pending measurement.

## Independent review

Independent review run by the operator.

## OPEN

Pending measurement.  Stage 2, stage 3, the chained ten-step ladder, 30 days,
and shipped-run statistics remain outside this round.
