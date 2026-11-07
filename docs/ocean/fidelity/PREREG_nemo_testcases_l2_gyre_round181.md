# Preregistration — round 181, developed `zdf_mxl` recurrence walk

Committed before extending or running the Round-180 scientific walk.  Round
180 established that the first causal boundary before developed `uslp` is the
inherited mixed-layer index `nmln`: 14 of 600 wet columns differ by exactly one
level under both the complete production JIT step and complete eager execution.
This round stays upstream, extends the same production-step harness through the
`zdf_mxl` producer, and does not create another bridge.

No production physics, configuration, carried state, restart schema, card
default, or immutable trajectory changes unless one source-exact candidate
closes `nmln` and passes the complete Decisions 43/45/55/59 gates.

## Compiled statement and one-variable hypothesis

The admitted build computes the before-level expansion coefficients and
`rn2b` once, copies it to `rn2`, and calls vertical physics with both formal
time levels bound to `Nbb` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168`.
The same stored `rn2b` is then passed to `ldf_slp` at `stprk3.f90:173-178`.
Its compiled producer evaluates levels 2 through `jpkm1` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619`.

Before any slope calculation, `zdf_mxl` initializes `nmln` and the accumulator,
forms `zN2_c`, then for each level adds
`MAX(rn2b,0)*e3w_1d*(1+r3t)` and updates the index only while the accumulated
value is strictly below the threshold at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-119`.
It converts the final index to live depth at `zdfmxl.f90:120-124`.

The instantiated GYRE card resolves `mld_criterion="n2_integral"` but
`slope_n2_evaluation="recompute"`.  The model therefore recomputes N2 inside
the mixed-layer helper even though the production step has already built the
step-entry N2 bundle for vertical physics.  The one-variable hypothesis is
that routing that existing bundle's `rn2b` and live `e3w` into the mixed-layer
recurrence removes the 14-column `nmln` boundary.  The ordinary production arm
remains unchanged until the full gate accepts a candidate.

## Frozen predictions and falsifiers

1. The unchanged baseline reproduces Round 180: `nmln` differs in exactly
   14/600 wet columns by one level, and `hmlp` differs in those 14 columns with
   maximum `1.33101408745854371e+02 m`, under production JIT and eager.  Any
   different baseline invalidates the walk until reconciled.
2. A literal scalar-level reconstruction from the admitted Round-179 `pn2`,
   `e3w_1d`, `r3t_Kmm`, bottom index, and compiled constants reproduces the
   stored NEMO `nmln` and `hmlp` bit-for-bit.  Any unequal cell invalidates the
   instrument; no model attribution is reported.
3. The baseline model recurrence's first non-bit direct operand is its locally
   recomputed N2, before any cumulative or index row.  If that operand is bit,
   or an earlier registered operand differs, this prediction is **REFUTED**.
4. Routing only the already-built production step-entry N2/e3w bundle into the
   mixed-layer recurrence makes `nmln` and `hmlp` bit-exact under production JIT
   and eager.  Any unequal column **REFUTES** the one-variable hypothesis; the
   walk then names the first remaining recurrence row and no physics lands.
5. The complete diagnostic step is passive: the baseline diagnostic and
   separately compiled ordinary steps differ in zero carried-state bytes under
   JIT and eager.  A moved byte invalidates that execution mode.
6. A one-level production-step plant on one wet carried-route `nmln` must move
   its immediate mixed-layer-depth and slope consumers while the registered N2
   and recurrence inputs remain unchanged.  A blind plant invalidates the
   instrument.
7. If the carried route closes `nmln`, its candidate must make the producer
   exact given NEMO inputs and then run the full ladder, 30-day, 360-day,
   recipe-derived card census, tanks, and DINO measurement.  It lands only if
   Decisions 43/45/55/59 accept every registered row.  Otherwise production is
   restored and the round remains **HELD**.

No acquisition or configuration decision is authorized in this round.  The
admitted Round-179 operands are sufficient unless prediction 2 fails.
