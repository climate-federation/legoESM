# Preregistration — NEMO testcase L2 GYRE round 157

Date: 2026-09-23

Incoming lane tip: `8c35e0e9c72fc50c0b6f20dfa69665b90664fcdc`.  Evidence
belongs under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round157/`.
This document is frozen before any legoESM stage-2 measurement is run.

## What was already read before freezing this document

Two things were read from the oracle side only, and both are recorded here so
that nothing below is a post-hoc prediction:

1. The round-156 acquisition was run by the operator.  NEMO reached `STOP 0`
   and the 19-group record exists at
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round157/oracle_developed_stage2`,
   but the run script's own admission step exited 72 with
   `REFUSE: round156 stage3-alignment plant lacks its marker`.  The cause is
   diagnosed and the record is admitted in this round's receipt.
2. The record's `flags_vec_linssh` group and the compiled selector at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:721` say that
   this run takes the VECTOR arm of the stage-1/2 time-stepping assignment,
   `uu(Kaa) = ( uu(Kbb) + rDt * uu(Krhs) ) * umask` at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:723`, and NOT the
   thickness-weighted arm the round-156 receipt named.  No free-surface ratio
   enters the stage-2 velocity at all.

## The compiled stage-2 program, in execution order

Cited from the record's OWN build,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90`.  Stage 2 binds
`Kbb = N`, `Kmm = N+1/3`, `Krhs = Kaa = N+1/2`.

1. `:462` — the right-hand-side slot at stage entry, recorded before
   `dyn_hpg` overwrites it.  `Krhs` aliases `Kaa`, so this slot still holds a
   velocity and is an operand of nothing.
2. `:497` `dyn_hpg`, recorded at `:499` as `after_hpg`.  It OVERWRITES the
   slot, so `after_hpg` is the pressure-gradient term alone.
3. `:510` `dyn_vor`, recorded at `:512` as `after_vor`.  Accumulates.
4. `:525` `dyn_adv`, recorded at `:531` as `after_adv`.  Accumulates.
5. `:721-724` — the vector-arm assignment, recorded at `:738` as
   `uu_vv_Kaa_raw`.
6. `:787` and `:810` — the barotropic correction, measured in round 156 to
   carry none of the difference.

## Predictions, frozen

1. **Stage-2 entry.** legoESM's stage-1 output velocity at `N+1/3`, taken
   from the production step under JIT, is NOT bit-identical to NEMO's
   recorded `uu_vv_Kmm`, and its active maximum is at least `1e-7` m/s, so
   the stage-2 velocity difference is substantially INHERITED rather than
   born inside stage 2.  REFUTED if that maximum is below `1e-9` m/s.
2. **First non-bit family.** In compiled order the first non-bit stage-2
   right-hand-side family is the pressure gradient, `after_hpg` at `:499`.
   REFUTED if `after_hpg` is bit-identical on every active U face.
3. **Budget closure.** Because the assignment is the vector arm, the raw
   stage-2 velocity difference is `rDt` times the `after_adv` difference:
   `max|Δuu_Kaa_raw|` equals `7200 * max|Δafter_adv|` to better than one
   percent.  REFUTED outside that.
4. **Assignment calibration.** legoESM's own transcription of the vector
   assignment, driven by NEMO's OWN recorded `uu_vv_Kbb`, `after_adv`, `rDt`
   and `umask`, rebuilds NEMO's `uu_vv_Kaa_raw` with zero unequal active U
   faces.  REFUTED by any unequal active face, in which case the assignment
   statement is itself the candidate.
5. **Plant.** Moving NEMO's recorded `after_hpg` by one unit in the last
   place changes the walk's reported first-non-bit row or its fingerprint,
   and the instrument exits nonzero with `STATUS PLANT-FIRED`.

## Landing rule

No candidate lands unless it is a SINGLE source-exact statement proven under
production JIT and it passes the full Decision 43/45 gate: the month, day 240
and day 360 not worse, kt2 T/S at the bar, first-over-bar not earlier, every
moved row registered, DINO if the statement is shared, the generic NEMO-GYRE
card, the LOCK_EXCHANGE and OVERFLOW tanks, and the six-file push gate.
Ranking is by what a statement carries at day 240, not at day 180.  The
expected outcome of a walk round is HELD.

## Rules

No configuration choice, no carried-state change, no stabiliser NEMO lacks,
no NEMO source modified, no `makenemo` and no `mpirun` from inside the round.
