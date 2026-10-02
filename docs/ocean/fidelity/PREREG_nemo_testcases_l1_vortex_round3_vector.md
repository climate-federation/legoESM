# PREREGISTRATION — VORTEX vector-EEN card (round 3, part A)

Frozen 2026-09-29 on lane tip `eeef1ec91397`, BEFORE the NEMO run for this card
exists and before anything about it is measured. Decision 73 (operator note BJ).

## What this card is

A second VORTEX card. Same box, same beta-plane, same flat bottom, same
simplified equation of state (decision 69), same eddy, no forcing, no implicit
vertical advection (decision 70) — and ONE thing changed: the momentum scheme
set becomes the one ORCA2 and GYRE run, vector-invariant advection with the
energy-and-enstrophy vorticity.

Why it is worth a card: in the round-2 card the vorticity routine is handed the
planetary rotation plus a metric term that this Cartesian mesh makes bitwise
zero, so the operator degenerates to a triad-weighted Coriolis. Under vector
form NEMO hands the SAME routine the planetary rotation plus the LIVE RELATIVE
vorticity, and adds the kinetic-energy gradient and the vertical advection of
momentum. None of that code has ever run in a clean rotating flow on this lane:
ORCA2 runs it under topography and forcing, GYRE runs the energy-conserving
cousin. This card runs it alone.

## The deck, and the whole of the difference

Against the round-2 deck, exactly two lines move, in NEMO's momentum-advection
block:

| namelist entry | round 2 (flux-EEN) | round 3 (vector-EEN) |
|---|---|---|
| `ln_dynadv_vec` | `.false.` | `.true.` |
| `ln_dynadv_up3` | `.true.` | `.false.` |

`ln_dynvor_een = .true.` is unchanged: both cards run the energy-and-enstrophy
vorticity, and only what it is HANDED differs. `ln_dynadv_up3` must go false
because NEMO counts the advection forms and stops unless exactly one is
selected (`dynadv.F90:184-190`); that count is also what routes the vorticity
(`dynvor.f90:855-868`). Everything else in the deck — geometry, equation of
state, tracer advection, pressure gradient, free surface, vertical physics,
boundary conditions, and the ten-step run length — is the round-2 deck.

Build names are new and nothing is overwritten: `VORTEX_VEC_OMIP_L1` for the
un-instrumented reference and `VORTEX_VEC_OMIP_L1_P3` for the record writer,
with evidence in a round-3 directory beside round 2's. The acquisition refuses
any target that already exists; moving an old build aside is the operator's
call, never the script's.

## Predictions, frozen

**P1. The two records' kt=1 step-entry files are BYTE-IDENTICAL.** Nothing in
the initial state reads the momentum scheme, and round 2 already established
that the equation-of-state switch left the initial state byte-identical to
round 1's. Falsifier: any byte differs. That would mean the momentum switch
reached the initial state, so the two cards would not be a one-variable pair
and the ladder could not be read as a momentum comparison.

**P2. The ladder's kt=1 row is AT-BAR on all five fields, with the same
normalized numbers round 2 reports to every digit** — temperature and salinity
exact, both velocities `2.220e-16`, sea surface height `1.355e-20`, and the
same 788 / 788 / 104 cells unequal at one or two last bits. This follows from
P1 and from the card carrying round 2's initial state unchanged. Falsifier: a
different count or a different normalized number.

**P3. The card is constructible and EXECUTES.** legoESM already has the
vector-invariant arm with the total-vorticity energy-and-enstrophy operator —
it is what the ORCA2 card selects — so this card should need no new numerics,
only a new composition. Falsifier: the card needs an operator legoESM does not
have, in which case it ships with that gap declared and fixed closed, exactly
as round 1 did, and the ladder waits for the round that transcribes it.

**P4. First over the bar is kt=2, not kt=1**, and the momentum fields lead the
tracers by at least one order of magnitude, as they do on the flux card.
Falsifier: kt=1 leaves the bar, or a tracer leads.

**P5. The vector card's kt=2 velocity residual is NOT bit-identical to the flux
card's** `1.136e-07` / `1.135e-07`. The two cards run different momentum
operators, so a bit-identical residual would mean the operator that owns the
residual is not the momentum operator at all — which would be a finding about
the flux card, not about this one. No prediction is made about which card is
worse; that is measured.

**P6. The three certified cards are unchanged.** LOCK_EXCHANGE-zco,
OVERFLOW-zps and GYRE-zco keep their ladder rows and their resolved
configurations, with the only permitted configuration-digest movement being an
added field that is unset on all three. The DINO month gate stays at its
pinned day-30 value within the harness's own floor. Falsifiers: any moved
ladder row, any changed field on another card, a worse DINO day-30.

## Gates this part must show

The acquisition preflight and its refusals; the record admission (restart
byte-identical, every record parsed from its own header); the card test and the
constructibility tripwire; the kt=1..10 ladder with its plant firing; the two
tanks re-run through the same gate; the citation gate with its planted shift
exiting non-zero; and the DINO month gate.

## What is NOT claimed here

Nothing about WHY either card leaves the bar at kt=2. Part B of this round
walks the flux card's kt=2 debt, and its own preregistration is separate. No
owner is named for the vector card in this part; if the vector card's ladder
differs from the flux card's, that difference is evidence for part B's walk and
is reported as such, not as an attribution.
