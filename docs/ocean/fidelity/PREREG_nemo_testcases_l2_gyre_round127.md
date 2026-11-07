# Preregistration — NEMO testcase L2 GYRE round 127

Date: 2026-09-20

Incoming lane tip: `48af510469da30266f0d05187350da8e6788776c`

This document is frozen before any Round-127 trigger-state decomposition.
Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round127/`.  Round 126 found
that the `+7.158989942001176e-1 K` heat-diffusivity row is an upstream-state
response: model-own EVD trigger masks differ from NEMO in 15 cells at day 180
and 11 at day 210, while the same production closure driven from NEMO's entry
has zero trigger disagreements.  This round separates the upstream
temperature, salinity and live-depth/free-surface inputs to those crossings.

No production physics, card, configuration value, restart schema, carried
state, stabilizer, year harness, reconciliation gate, freshwater pair or
`#1484` guard changes in this round.

## P0 — compiled statement and inherited controls

The compiled Round-125 program computes both stability arms from the
whole-step entry.  It evaluates `eos_rab` and `bn2` on `Nbb`, copies
`rn2b` into `rn2`, and calls `zdf_phy` with both formal time levels bound to
`Nbb` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168`.
The resolved binary selects the TEOS-10 polynomial.  In the compiled source,
that polynomial makes alpha and beta functions of temperature, salinity and
the live stretched T-point depth at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1257-1310`.
The source then interpolates alpha/beta to W points, forms the separate
temperature and salinity contributions, divides by live `e3w`, applies
`wmask`, and writes `rn2` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.
Finally EVD replaces tracer diffusivity when
`MIN(rn2,rn2b) <= -1e-12` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:107-109`.

Before any new attribution, the existing admission gates must re-admit the
Round-125 oracle record and Round-126 production trace.  The new gate must
reproduce exactly:

* 15 model-own trigger disagreements at the day-180/step-1081 entry;
* 11 at the day-210/step-1261 entry;
* zero disagreements at both entries when the production closure receives the
  complete NEMO entry; and
* bit-identical `rn2` and `rn2b` for every scored GYRE hybrid, as required by
  the compiled `rn2 = rn2b` copy.

Any changed headline or failed admission **REFUTES** the inherited premise and
stops the round.  A fresh number is not allowed to supersede the admitted one
without reconciling the difference.

## P1 — extend the existing production-closure gate, not a new harness

Round 127 extends `nemo_testcase_l2_gyre_year_owners.py` and its existing
`conditional_vertical_closure`; it does not create another state bridge,
N2 formula or integration harness.  The current `bn2_intermediate` private
hook will expose the actual `rn2/rn2b` bundle that the same
`diagnose_vertical_K` production closure consumes.  The default return and
every ordinary model call remain byte-for-byte unchanged.

Pre-measurement inventory found that the acquired Round-125 frames, rather
than separate daily restart files, carry the needed daily oracle operands.
For each daily whole-step entry from day 180 through day 239, the gate loads:

* NEMO's independently recorded `T_Kbb_in`, `S_Kbb_in`, and `r3t_Kbb` at
  entry step `6 * day + 1` (the day-180 restart supplies the inert carrier
  fields that the returned N2 bundle does not read);
* the independently recorded legoESM daily core state at the same day; and
* the admitted NEMO `avt` and legoESM `heat_K` masks at that entry step.

The oracle `eta` representation is chosen by an exact checked inverse of
NEMO's recorded `r3t = ssh * r1_ht_0`: sending it through the production
`nemo_reciprocal` evaluation must recover every wet `r3t` bit.  This avoids
inventing a free-surface value from a rounded quotient.  Failure to find that
inverse refuses the instrument.  The returned `rn2/rn2b`, not the unrelated
diffusivity contributions that also run inside `diagnose_vertical_K`, is the
scored production output.

Starting from the bridged NEMO state, all eight subsets of
`{temperature, salinity, eta}` are replaced by legoESM's corresponding field.
Velocity, bathymetry, closure memory, forcing and every unlisted field remain
the day-180 NEMO carrier arm; none is in the causal input set of the returned
N2 bundle.  Each subset runs through the same full production closure under
`jax.jit`; no isolated N2 calculation is scored.  `eta` is the complete
live-depth/free-surface row because the static grid and bathymetry are already
shared and bit-identical.

The empty subset must reproduce the recorded NEMO trigger mask, and the full
subset must reproduce the recorded legoESM trigger mask, at every one of the
60 daily checkpoints.  One unequal bit refuses the decomposition.  This is
the controlled-comparison and instrument-calibration gate.

Frozen prediction: all 120 endpoint-mask controls are bit-exact, and the
production-returned `rn2` and `rn2b` are bit-identical in every arm.  Any
failure **REFUTES** the instrument and stops the round without an owner.

## P2 — order-independent crossing attribution

For each cell and checkpoint define

`f(S) = 1[trigger_mask(S) != trigger_mask(NEMO)]`,

where `S` is one of the eight input subsets.  Temperature, salinity and eta
receive their exact three-player Shapley contributions: the mean marginal
change in `f` over all six substitution orders.  These signed cell
contributions sum exactly to the full model-input trigger disagreement;
negative values expose cancellation instead of being reassigned.  The same
all-subset construction is applied to the production-returned continuous
margin `MIN(rn2,rn2b) - (-1e-12)` at the cells where the full masks disagree.

The primary magnitude ranking is the absolute signed Shapley trigger-cell
visits summed over the 60 equal-cadence daily checkpoints.  The receipt also
reports signed cell visits, each checkpoint's exact closure, the continuous
margin RMS, and the cancellation ratio.  No post-hoc reordering or preferred
owner receives an interaction residual: the all-subset Shapley identity
distributes the complete three-input interaction symmetrically by definition.

Frozen magnitude prediction: temperature is the largest absolute trigger-bit
owner at day 180, day 210 and over the full daily interval, and it carries at
least half of the summed absolute Shapley magnitude in each of those three
reductions.  Salinity ranks second.  Eta contributes no more than one absolute
cell-equivalent at either inherited checkpoint.  Any failed inequality is
**REFUTED**, retained in the receipt, and the measured ranking wins.

## P3 — per-step birth and non-vacuous controls

The already admitted 360-step records are scanned directly: NEMO fires where
recorded `avt == 100 m2/s`; legoESM fires where recorded `heat_K == 100 m2/s`.
The scan reports model-only and NEMO-only cells at every step, the first
differing step, and the depth, longitude-third and latitude-band census at
that first step and over the interval.  It does not infer a state birth before
the record begins.

Because Round 126 already measured 15 disagreements at step 1081, the frozen
prediction is that the first in-window trigger mismatch is inherited at step
1081 (day 180), not born later in this record.  The largest spatial bin at
that first entry is predicted to be 0--100 m, west third and south of or equal
to 37.2 degrees north, matching the dominant Round-126 coefficient-carry bin.
If another bin is larger, that spatial prediction is **REFUTED**.  The receipt
must say that the true pre-day-180 birth remains outside this interval.

Two controls are required.  A stored-mask bit flip must be rejected and exit
nonzero.  A production trigger plant must perturb one real wet temperature
operand through the jitted closure until at least one returned `rn2` threshold
bit flips; it must leave the untouched NEMO arm unchanged, print
`STATUS PLANT-FIRED`, and exit nonzero.  A zero perturbation, a dry-cell change
or a plant that only changes an unconsumed reconstruction fails the control.

## P4 — round boundary

This is a diagnostic magnitude round.  No production statement lands, so the
Decision-43 ladder/month and Decision-45 year landing gates do not run and the
immutable before arms do not move.  The largest measured state row becomes
Round 128's single upstream process candidate.  If temperature wins, the next
round decomposes the temperature-stratification difference in compiled process
order; if salinity wins, it first proves whether the existing records carry the
needed salinity process boundaries; if eta wins, it moves to the free-surface
dynamics owner.  A missing required boundary produces `STOPPED_FOR_RECORD`,
not an inferred owner.

The new observer is private and statically absent from ordinary construction.
GYRE production, generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW execute no
changed production statement.  ORCA2 is `UNMEASURED-WITH-SPEC`: repeat its
native daily state bridge, all-subset production-closure cube, exact endpoint
mask controls and per-step trigger trace before transferring the owner.

A separate read-only Codex pass must try to refute the state bridge, production
execution claim, Shapley closure, controls and owner verdict.  Every compiled
source citation in the receipt is mapped by the citation gate, and its shifted
citation plant must exit nonzero.
