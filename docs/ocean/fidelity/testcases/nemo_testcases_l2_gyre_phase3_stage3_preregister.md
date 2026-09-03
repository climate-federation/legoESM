# NEMO testcase lane 2 GYRE — stage-3 completion preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`

This freezes the next ordered measurement after the bottom-drag landing and
the Nbb TKE/EVD time-level correction.  It was derived afresh from NEMO 5.0.2
source; the abandoned round-7 WIP receipt and its nonexistent preregistration
SHAs are not evidence.

## Entering boundary

The CPU/fp64 gate now clears all 800 barotropic substep rows, both stage-1
momentum rows, and the stage-1 and stage-2 T/S/SSH operands.  Stage-2 tracer
normalized maxima are `4.542928987878046e-16` (T),
`5.786483703728924e-16` (S), and `3.2526065174565133e-18` (SSH).  The first
tracer debt is therefore the stage-3 completion returned at `kt=2`: T
`2.2049230768358347e-3`, S `8.792858585325299e-11`.  The first momentum debt
is likewise stage 3: u `9.481924730527598e-7`, v
`8.988523102725608e-7`.

## Source-ordered register

For tracers, `stprk3_stg.F90:565-600` zeros `Krhs`, then executes advection,
`tra_sbc_RK3`, QSR, lateral diffusion, and finally the implicit `tra_zdf`
time integration.  The resolved GYRE branches have no ice shelf, BBL tracer,
damping, MFC, OSMOSIS, or NPC term.  For momentum, `:317-336,357-430`
executes HPG, ENE vorticity, vector-invariant advection, level Laplacian
viscosity, then implicit `dyn_zdf`; `:433-446` applies the final fixed-depth
barotropic mean.

A WRITE-only MY_SRC record will capture the accumulated stage-3 T/S `Krhs`
immediately before `tra_zdf`, plus T/S `Kbb`, `Kmm`, and final `Kaa`.  The
candidate will expose its normal pre-implicit state only after completing the
ordinary step.  The gate will reconstruct NEMO's explicit QCO seed from the
dumped operands and score it before scoring final `Kaa`.

- If the pre-implicit seed clears `1e-15` but final `Kaa` does not, the
  implicit ZDF solve is the first owner-capable boundary.
- If the seed is already DEBT, the first source-ordered accumulated RHS term
  over bar remains the boundary; no ZDF owner label is allowed.
- A one-variable oracle-seed arm is `CONFIRMED_CAUSAL_OWNER` only if it clears
  the final field.  Movement is printed relative to the faithful residual
  before any label.

The existing direct QSR row is retained: its tendency differs by
`5.257960831729332e-13 K/s`, but the oracle-QSR injection moves the kt=2 T
row by only `1.416330476565004e-7` of its residual, so QSR is
`CAUSAL_NONPRIMARY_AT_KT2`, not exonerated globally.

## Controls and provenance

The record must validate magic, header, dimensions, fp64 dtype, finite owned
values, and a central time-level-registry entry.  The same instrumented run's
ordinary stage and kt=2 state hashes must equal the already pinned hashes, or
the record is rejected.  A planted pre-ZDF tracer value must become DEBT.
No external adversarial review has occurred for this round.
