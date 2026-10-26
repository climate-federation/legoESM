# PREREGISTRATION — who owns the VORTEX flux card's second step (round 3, part B)

Frozen 2026-09-29 on lane tip `eeef1ec91397` plus this round's part-A commit,
BEFORE any substitution is run. Round 2 measured the ladder and deliberately
named no owner; this part names one.

## The debt being walked

Round 2's ladder, against the round-2 record. The card enters at the bar and
leaves it at the first step it takes:

| kt | T | S | u | v | ssh |
|---|---|---|---|---|---|
| 1 | exact | exact | 2.220e-16 | 2.220e-16 | 1.355e-20 |
| 2 | 1.643e-09 | 4.060e-16 | 1.136e-07 | 1.135e-07 | 3.709e-08 |
| 10 | 2.103e-08 | 1.218e-15 | 2.464e-06 | 2.192e-06 | 5.336e-06 |

Momentum leads temperature by two orders of magnitude.

## The instrument, and why it is this one

NEMO's own record already contains the inside of the first step: the
barotropic frame written immediately after the external solve, the three
per-stage states written at each stage's pointer boundary, and the step entry
at kt=2. legoESM has private substitution hooks at exactly those boundaries —
the external-solve handoff and a stage's entry bundle — which is the campaign's
one-variable method. So the walk is cumulative substitution in NEMO's own
execution order, and every arm's number is the SAME normalized maximum the
ladder reports, so the arms and the ladder are directly comparable.

| arm | what is substituted | what its residual is owned by |
|---|---|---|
| 0 | nothing (the card) | everything, including the initial state |
| 1 | the step-entry state comes from NEMO's kt=1 record | the whole first step |
| 2 | arm 1 + NEMO's external-solve handoff | the three stages |
| 3 | arm 2 + NEMO's stage-1 output as stage 2's entry | stages 2 and 3 |
| 4 | arm 2 + NEMO's stage-2 output as stage 3's entry | stage 3 alone |

Differences between consecutive arms attribute the residual: arm 1 minus arm 0
is the initial state's share, arm 2 minus arm 1 is the external solve's,
arm 3's drop from arm 2 is stage 1's, arm 4's drop from arm 3 is stage 2's.

## Predictions, frozen

**B1. The initial state owns essentially none of it.** Arm 1's kt=2 velocity
stays within a factor of two of arm 0's `1.136e-07`. A one-to-two-last-bit
seed would have to be amplified by about five hundred million in a single step
to produce that residual, which no stable operator does. Falsifier: arm 1
drops the velocity below `1e-8`, which would convict the initial state and
retract round 2's "not worth closing" verdict on the exponential floor.

**B2. The external solve owns the sea surface height row outright.** Under
this time-stepping program the end-of-step height is produced by the external
solve and the stages do not touch it, so arm 1's height residual should be
`3.709e-08` unchanged, and arm 2 should put it exactly at zero. Falsifier:
arm 2 leaves height over the bar, which would mean something other than the
external solve writes it.

**B3, THE PICK. The external solve owns the majority of the momentum
residual too**: arm 2 drops the kt=2 velocity by at least a factor of ten
below arm 1's. Reasoning, labelled PLAUSIBLE rather than measured: this card's
whole rotation is new this round, it enters the step twice — once in the
baroclinic trend and once inside every one of the forty-eight barotropic
substeps — and the height row, which is the external solve's own output, is
already over the bar. Falsifier: arm 2 drops the velocity by less than a
factor of two, which moves the walk into the stages.

**B4. Among the stages, stage 1 leads.** Arm 3's drop from arm 2 is the
largest of the three stage drops. Falsifier: a later stage's drop is larger,
in which case nothing upstream may be blamed and the walk continues there.

**B5. The owner is a TRANSCRIPTION difference, not a compiled-rounding
floor.** The residual is `1.1e-07` normalized, eight orders above the
campaign's measured compiled-rounding floor, so it must be a statement legoESM
evaluates differently from NEMO rather than the same statement rounding
differently. Falsifier: the walk bottoms out in an arm whose residual is
already at the bar, with no statement left to name.

## What lands, and what does not

A fix lands only if it is NEMO's own statement, cited from the compiled source
of the round-2 build, proven one-variable, and it passes the gates: the VORTEX
ladder with no at-bar row leaving the bar and the first-over-bar never earlier,
the two tanks unchanged through the same gate, the three certified cards'
configurations unchanged, and the DINO month gate at its pinned value. If the
owner is named but the fix is not proven inside this round's budget, the round
reports the owner and HOLDS — naming an owner is the deliverable, landing an
unproven fix is not.

No configuration choice is made in this part. If the walk ends at a place
where NEMO's behaviour depends on a selection this card has not made, that is
written up as a decision for the user, not taken.
