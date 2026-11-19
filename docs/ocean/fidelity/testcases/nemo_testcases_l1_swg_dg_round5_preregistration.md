# SWG double-gyre round 5 preregistration

Date: 2026-10-12

The operator selected this round's protocol before the measurement: localise
the two exact-bit ladder debts without changing physics, add an in-scan
daily-mean output, and compare matched daily means only after a new GPU year is
acquired.  The existing exact IEEE-word bar remains unchanged during
localisation.

## Stage boundary

Run the production JIT on CPU under the fp64/libm certification policy.
Compare the admitted kt-2 stage-1 operands to the card's from-rest state after
step 1, compare the card after step 2 to the admitted kt-3 stage-1 entry, and
replay kt-2 stage 1 from the admitted operands.  Every difference is classified
as wet interior, wet wall-adjacent, or dry/storage halo.

- CONFIRM record/representation ownership if the from-rest dynamic state is
  exact but replaying exact admitted operands does not reproduce the next
  record.
- REFUTE record/representation ownership if the from-rest dynamic state first
  differs before the recorded transition.
- The signed-zero forcing question is reported separately and cannot change
  the bar in this round.

## Thirty-day boundary

The registered month arm integrates all 12,960 production steps from rest at
25 km under the same CPU policy and compares every daily instantaneous restart.
The localisation additionally compares day 1 by region and compares NEMO's
day-1 restart directly with its next recorded stage entry.

- CONFIRM a NEMO record discontinuity if restart-to-next-entry differs.
- CONFIRM a model-arithmetic boundary if that NEMO pair is exact but the
  from-rest legoESM day-1 state differs.
- CONFIRM a pure representation debt only where numerical values agree and
  IEEE words differ.

## Daily means and climate boundary

The non-certified GPU mode will carry three online means inside the same
compiled day as the production steps.  Its association is fixed to IOIPSL
`moycum`: `(old_mean * old_count + post_step_state) / new_count`.  Each output
file retains the end-of-day snapshot and names both sampling conventions.

The direct control uses a non-steady tiny production case: the compiled mean
must equal a separately associated mean of every post-step state, and must not
equal the final snapshot.  The climate reader must refuse NEMO fields unless
their attributes identify `ave(x)`, a 50-second sample interval, and an
86,400-second write interval.  The year-1 comparison remains unmeasured until
the operator acquires 365 daily-mean legoESM files.

## Open inherited pins

The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digest pins remain stale:
the pinned values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
`7f1ec89b3e6578f5`, versus current round-6 values
`f02d2f42a570f413` / `d5dfdf83b3a1e0ab` / `83de0010e1170032`.
This lane does not re-pin them.
